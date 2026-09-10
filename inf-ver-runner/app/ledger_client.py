"""Thin async client for the Ledger Access Layer.

All verifier I/O goes through the ledger API — the verifier never talks to
the ledger Postgres directly.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LedgerClient:
    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.ledger_api_url,
            timeout=settings.ledger_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def declare(self, hostname: str, model_name: str) -> dict[str, Any]:
        """Declare this verifier's model deployment: upserts our hardware (by
        hostname) and model config, and opens a deployment — the ledger owns
        all ids. Returns the deployment JSON (model_id, hardware_id, ...).

        Sampling params are deliberately omitted: the runner re-executes with
        the PROVER model's declared sampling config, so its own model row
        carries none of its own.
        """
        resp = await self._client.post(
            "/model-deployments/declare",
            json={
                "hostname": hostname,
                "model_name": model_name,
                "decoding_algorithm": "gumbel_max",
                # Runners are the verification side of the ledger.
                "owner_name": "verifier",
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def close_deployment(self, hostname: str) -> bool:
        """Report this runner's shutdown: sets ended_at on our active
        deployment. Best-effort — a 404 (already closed by a re-declare)
        or an unreachable ledger is logged, never raised."""
        try:
            resp = await self._client.post(
                "/model-deployments/close", json={"hostname": hostname}
            )
        except httpx.HTTPError as exc:
            log.warning("Deployment close failed for %s: %s", hostname, exc)
            return False
        if resp.status_code == 200:
            log.info("Closed runner deployment hostname=%s", hostname)
            return True
        log.warning(
            "Deployment close returned %s for %s", resp.status_code, hostname
        )
        return False

    async def fetch_unverified(
        self, limit: int, model_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Newest-first inference events with no verification result yet,
        optionally scoped to one model (a runner drains only its own)."""
        params: dict[str, Any] = {"limit": limit}
        if model_id is not None:
            params["model_id"] = model_id
        resp = await self._client.get("/inference-events/unverified", params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_model(self, model_id: str) -> dict[str, Any]:
        """The ledger's model row — the source of truth for sampling config
        and the per-model verification_threshold."""
        resp = await self._client.get(f"/models/{model_id}")
        resp.raise_for_status()
        return resp.json()

    async def create_verification_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post("/verification-events", json=payload)
        resp.raise_for_status()
        return resp.json()
