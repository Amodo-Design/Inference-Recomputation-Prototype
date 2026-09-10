"""Drain-and-exit verification loop for a single model's pending events.

One runner is paired with one vLLM instance serving one model. It fetches the
pending (unverified) inference events for exactly that model, verifies them
newest-first, writes the verdicts, and EXITS 0 once the queue is empty — the
orchestrator reaps the Job and its vLLM to free the GPU. Nonzero exit means a
fatal error (Job backoff handles retries).

The pass threshold is read from the ledger's model row each cycle
(verification_threshold). NULL means verification is paused for the model:
the runner exits 0 without consuming any events, so they verify retroactively
once a threshold is set in the UI.

Transient verify failures (vLLM down, timeouts) are retried with exponential
backoff up to a limit, then recorded as unverifiable so the event stops being
polled. Bookkeeping is in-memory only; a Job restart simply retries.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol

import httpx

from .config import Settings
from .ledger_client import LedgerClient
from .mapper import build_verification_event_payload, inference_event_to_verify_request
from .models import ErrorCode, VerificationStatus, VerifyRequest, VerifyResponse
from .pod_identity import PodIdentityResolver

log = logging.getLogger(__name__)

# Unverifiable outcomes that cannot improve on retry.
PERMANENT_ERROR_CODES = {
    ErrorCode.UNSUPPORTED_MODEL,
    ErrorCode.SAMPLING_CONFIG_MISSING,
    ErrorCode.TOKENIZATION_MISMATCH,
}


class VerificationServiceLike(Protocol):
    """The slice of VerificationService the runner needs (kept as a Protocol
    so tests never import the torch-heavy verifier module)."""

    async def verify(
        self,
        request: VerifyRequest,
        pass_threshold: float,
        margin_clip: float | None = None,
    ) -> VerifyResponse: ...


class VerificationRunner:
    def __init__(
        self,
        settings: Settings,
        ledger: LedgerClient,
        verification_service: VerificationServiceLike,
        hardware_id: str,
        verifier_model_id: str,
        pod_identity_resolver: "PodIdentityResolver | None" = None,
    ) -> None:
        self._settings = settings
        self._ledger = ledger
        self._service = verification_service
        self._hardware_id = hardware_id
        self._verifier_model_id = verifier_model_id
        self._pod_identity_resolver = pod_identity_resolver
        self._attempts: dict[str, int] = {}
        self._next_retry_at: dict[str, float] = {}

    async def run_until_drained(self) -> int:
        """Drain the model's pending queue; returns the process exit code.

        0 = queue empty, or verification_threshold is NULL (paused — events
        are deliberately left pending). 1 = ledger stayed unreachable.
        2 = launch sampling config disagrees with the ledger model row.
        """
        settings = self._settings
        log.info(
            "Runner started model_id=%s batch=%s max_attempts=%s",
            settings.runner_model_id,
            settings.poll_batch_size,
            settings.max_verify_attempts,
        )
        ledger_failures = 0
        checked_launch_config = False
        while True:
            try:
                model_row = await self._ledger.get_model(settings.runner_model_id)

                if not checked_launch_config:
                    problems = settings.launch_sampling.mismatches(model_row)
                    if problems:
                        log.error(
                            "Launch sampling config disagrees with ledger model row "
                            "model_id=%s: %s — refusing to verify",
                            settings.runner_model_id,
                            "; ".join(problems),
                        )
                        return 2
                    checked_launch_config = True

                threshold = model_row.get("verification_threshold")
                if threshold is None:
                    log.info(
                        "verification_threshold is NULL for model_id=%s — "
                        "verification paused, exiting without consuming events",
                        settings.runner_model_id,
                    )
                    return 0

                outcome = await self._poll_once(
                    threshold, model_row.get("delta_max")
                )
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:
                ledger_failures += 1
                if ledger_failures >= settings.max_ledger_failures:
                    log.error(
                        "Ledger unreachable after %s consecutive failures, giving up: %s",
                        ledger_failures,
                        exc,
                    )
                    return 1
                log.warning(
                    "Ledger error (%s/%s), will retry: %s",
                    ledger_failures,
                    settings.max_ledger_failures,
                    exc,
                )
                await asyncio.sleep(settings.poll_interval_seconds)
                continue

            ledger_failures = 0
            if outcome == "drained":
                log.info(
                    "Pending queue drained for model_id=%s, exiting",
                    settings.runner_model_id,
                )
                return 0
            if outcome == "cooling_down":
                await asyncio.sleep(settings.poll_interval_seconds)

    async def _poll_once(
        self, pass_threshold: float, delta_max: float | None = None
    ) -> str:
        """Verify up to `max_concurrent_verifications_per_model` events.

        Returns "processed" (at least one verdict was written), "cooling_down"
        (events remain but all eligible ones were in a retry backoff — or the
        ones we picked all turned out to need cooling), or "drained" (queue
        empty).
        """
        items = await self._ledger.fetch_unverified(
            self._settings.poll_batch_size, model_id=self._settings.runner_model_id
        )
        if not items:
            return "drained"

        concurrency = max(1, self._settings.max_concurrent_verifications_per_model)
        if concurrency == 1:
            item = self._pick_eligible(items)
            if item is None:
                return "cooling_down"
            return await self._process_item(item, pass_threshold, delta_max)

        # The whole round's picks happen here, up front, before any
        # verification starts — so no item can be selected twice within one
        # round even though the picks are then processed concurrently below.
        picked = self._pick_eligible_many(items, concurrency)
        if not picked:
            return "cooling_down"

        results = await asyncio.gather(
            *(
                self._process_item(item, pass_threshold, delta_max)
                for item in picked
            ),
            return_exceptions=True,
        )

        cancelled = next(
            (r for r in results if isinstance(r, asyncio.CancelledError)), None
        )
        if cancelled is not None:
            raise cancelled
        http_errors = [r for r in results if isinstance(r, httpx.HTTPError)]
        if http_errors:
            raise http_errors[0]
        other_errors = [r for r in results if isinstance(r, BaseException)]
        if other_errors:
            raise other_errors[0]

        if any(r == "processed" for r in results):
            return "processed"
        return "cooling_down"

    async def _process_item(
        self,
        item: dict[str, Any],
        pass_threshold: float,
        delta_max: float | None,
    ) -> str:
        """Verify one already-picked item and, if the outcome is recordable,
        write it to the ledger. Returns "processed" or "cooling_down"."""
        event_id = item["event"]["id"]
        request = inference_event_to_verify_request(item)
        response = await self._service.verify(
            request, pass_threshold, margin_clip=delta_max
        )

        if not self._should_record(event_id, response):
            return "cooling_down"

        pod_identity = (
            await self._pod_identity_resolver.get()
            if self._pod_identity_resolver is not None
            else None
        )
        payload = build_verification_event_payload(
            request,
            response,
            self._hardware_id,
            self._verifier_model_id,
            pod_identity=pod_identity,
            runner_concurrency=max(1, self._settings.max_concurrent_verifications_per_model),
        )
        await self._ledger.create_verification_event(payload)
        self._clear(event_id)
        log.info(
            "Recorded verification inference_event_id=%s status=%s error_code=%s",
            event_id,
            response.status.value,
            response.error_code.value if response.error_code else None,
        )
        return "processed"

    def _pick_eligible(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        """First (= newest) item not sitting in a retry cooldown, so one
        cooling-down event doesn't starve the rest of the batch."""
        now = time.monotonic()
        for item in items:
            if self._next_retry_at.get(item["event"]["id"], 0.0) <= now:
                return item
        return None

    def _pick_eligible_many(
        self, items: list[dict[str, Any]], n: int
    ) -> list[dict[str, Any]]:
        """Up to n newest-first items not sitting in a retry cooldown, so
        cooling-down events don't starve the rest of the batch.

        Invariant: this is called exactly once per _poll_once round, before
        any of the returned items are processed, so the same item can never
        be picked twice within one round even though the picks are then
        verified concurrently.
        """
        now = time.monotonic()
        picked: list[dict[str, Any]] = []
        for item in items:
            if len(picked) >= n:
                break
            if self._next_retry_at.get(item["event"]["id"], 0.0) <= now:
                picked.append(item)
        return picked

    def _should_record(self, event_id: str, response: VerifyResponse) -> bool:
        """Decide whether the outcome is written to the ledger now, or the
        event is left unverified for a backed-off retry."""
        if response.status is not VerificationStatus.UNVERIFIABLE:
            return True
        if response.error_code in PERMANENT_ERROR_CODES:
            return True

        attempts = self._attempts.get(event_id, 0) + 1
        if attempts >= self._settings.max_verify_attempts:
            response.reason = (
                f"{response.reason} (giving up after {attempts} attempts)"
            )
            return True

        self._attempts[event_id] = attempts
        delay = self._settings.retry_backoff_seconds * 2 ** (attempts - 1)
        self._next_retry_at[event_id] = time.monotonic() + delay
        log.info(
            "Transient failure inference_event_id=%s error_code=%s attempt=%s retry_in=%.0fs",
            event_id,
            response.error_code.value if response.error_code else None,
            attempts,
            delay,
        )
        return False

    def _clear(self, event_id: str) -> None:
        self._attempts.pop(event_id, None)
        self._next_retry_at.pop(event_id, None)
