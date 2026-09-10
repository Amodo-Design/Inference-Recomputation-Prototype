"""VerificationRunner drain/exit/retry behaviour with stubbed ledger + verifier."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

import app.runner as runner_module
from app.models import ErrorCode, VerificationStatus, VerifyRequest, VerifyResponse
from app.runner import VerificationRunner

from helpers import MODEL_ID, make_item, make_response, make_settings


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class StubLedger:
    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        threshold: float | None = 0.1,
        model_row: dict[str, Any] | None = None,
    ) -> None:
        self.items = items
        self.written: list[dict[str, Any]] = []
        self.fetch_error: Exception | None = None
        self.write_error: Exception | None = None
        self.model_error: Exception | None = None
        self.fetch_model_ids: list[str | None] = []
        self.model_row = model_row or {
            "model_id": MODEL_ID,
            "model_name": "Qwen/Qwen3-8B",
            "seed": 42,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "verification_threshold": threshold,
            "delta_max": None,
        }

    async def get_model(self, model_id: str) -> dict[str, Any]:
        if self.model_error is not None:
            raise self.model_error
        assert model_id == MODEL_ID
        return self.model_row

    async def fetch_unverified(
        self, limit: int, model_id: str | None = None
    ) -> list[dict[str, Any]]:
        if self.fetch_error is not None:
            raise self.fetch_error
        self.fetch_model_ids.append(model_id)
        return self.items[:limit]

    async def create_verification_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.write_error is not None:
            raise self.write_error
        self.written.append(payload)
        self.items = [
            item for item in self.items
            if item["event"]["id"] != payload["inference_event_id"]
        ]
        return payload


class ConcurrencyTrackingService:
    """Verifies with a real `asyncio.sleep`, tracking the peak number of
    in-flight `verify()` calls so tests can prove real concurrency."""

    def __init__(self) -> None:
        self.responses: dict[str, VerifyResponse] = {}
        self.calls: list[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0

    def queue(self, event_id: str, response: VerifyResponse) -> None:
        self.responses[event_id] = response

    async def verify(
        self,
        request: VerifyRequest,
        pass_threshold: float,
        margin_clip: float | None = None,
    ) -> VerifyResponse:
        self.calls.append(request.request_id)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.05)
            return self.responses[request.request_id]
        finally:
            self.in_flight -= 1


class FlakyLedger(StubLedger):
    """A StubLedger whose create_verification_event raises for one specific
    event id, so a concurrent batch can have one write fail while the rest
    succeed."""

    def __init__(self, items: list[dict[str, Any]], *, fail_event_id: str) -> None:
        super().__init__(items)
        self._fail_event_id = fail_event_id

    async def create_verification_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload["inference_event_id"] == self._fail_event_id:
            raise httpx.ConnectError("ledger down")
        return await super().create_verification_event(payload)


class StubService:
    """Returns queued responses per event id (falls back to the last one)."""

    def __init__(self) -> None:
        self.responses: dict[str, list[VerifyResponse]] = {}
        self.calls: list[str] = []
        self.thresholds: list[float] = []
        self.margin_clips: list[float | None] = []

    def queue(self, event_id: str, response: VerifyResponse) -> None:
        self.responses.setdefault(event_id, []).append(response)

    async def verify(
        self,
        request: VerifyRequest,
        pass_threshold: float,
        margin_clip: float | None = None,
    ) -> VerifyResponse:
        self.calls.append(request.request_id)
        self.thresholds.append(pass_threshold)
        self.margin_clips.append(margin_clip)
        queued = self.responses[request.request_id]
        return queued.pop(0) if len(queued) > 1 else queued[0]


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(runner_module.time, "monotonic", fake.monotonic)
    return fake


@pytest.fixture
def instant_sleep(monkeypatch, clock: FakeClock):
    """asyncio.sleep advances the fake clock instead of waiting."""

    async def fake_sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(runner_module.asyncio, "sleep", fake_sleep)


def make_runner(
    ledger: StubLedger, service: StubService, **settings_overrides
) -> VerificationRunner:
    return VerificationRunner(
        make_settings(**settings_overrides),
        ledger,
        service,
        hardware_id="hw-1",
        verifier_model_id="vm-1",
    )


async def test_drains_queue_then_exits_zero(clock, instant_sleep):
    items = [make_item(), make_item()]
    ledger = StubLedger(items)
    service = StubService()
    for item in items:
        event_id = item["event"]["id"]
        service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service)

    assert await runner.run_until_drained() == 0
    assert len(ledger.written) == 2
    assert all(p["verifier_model_id"] == "vm-1" for p in ledger.written)
    # The runner polls only its own model.
    assert set(ledger.fetch_model_ids) == {MODEL_ID}


async def test_null_threshold_exits_zero_without_consuming(clock, instant_sleep):
    item = make_item()
    ledger = StubLedger([item], threshold=None)
    service = StubService()
    runner = make_runner(ledger, service)

    assert await runner.run_until_drained() == 0
    assert ledger.written == []
    assert service.calls == []
    # The events were never even fetched — they stay pending.
    assert ledger.fetch_model_ids == []


async def test_ledger_threshold_passed_to_verifier(clock, instant_sleep):
    item = make_item()
    event_id = item["event"]["id"]
    ledger = StubLedger([item], threshold=0.037)
    service = StubService()
    service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service)

    assert await runner.run_until_drained() == 0
    assert service.thresholds == [0.037]


async def test_launch_sampling_mismatch_exits_two(clock, instant_sleep):
    item = make_item()
    ledger = StubLedger([item])  # ledger row has seed=42
    service = StubService()
    runner = make_runner(ledger, service, launch_seed=7)

    assert await runner.run_until_drained() == 2
    assert ledger.written == []
    assert service.calls == []


async def test_matching_launch_sampling_proceeds(clock, instant_sleep):
    item = make_item()
    event_id = item["event"]["id"]
    ledger = StubLedger([item])
    service = StubService()
    service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(
        ledger, service,
        launch_seed=42, launch_temperature=1.0, launch_top_k=50, launch_top_p=0.95,
    )

    assert await runner.run_until_drained() == 0
    assert len(ledger.written) == 1


async def test_ledger_down_exits_one_after_max_failures(clock, instant_sleep):
    ledger = StubLedger([])
    ledger.model_error = httpx.ConnectError("ledger down")
    service = StubService()
    runner = make_runner(ledger, service, max_ledger_failures=3)

    assert await runner.run_until_drained() == 1


async def test_transient_failures_retry_then_give_up_and_drain(clock, instant_sleep):
    item = make_item()
    event_id = item["event"]["id"]
    ledger = StubLedger([item])
    service = StubService()
    service.queue(
        event_id,
        make_response(
            event_id,
            VerificationStatus.UNVERIFIABLE,
            error_code=ErrorCode.VERIFICATION_TIMEOUT,
        ),
    )
    runner = make_runner(ledger, service, max_verify_attempts=3, retry_backoff_seconds=1.0)

    # The loop retries through cooldowns (instant_sleep advances the clock),
    # eventually records the give-up verdict, then drains and exits 0.
    assert await runner.run_until_drained() == 0
    assert len(service.calls) == 3
    assert ledger.written[0]["result"] == "unverifiable"
    assert "giving up after 3 attempts" in ledger.written[0]["result_detail"]


async def test_permanent_unverifiable_written_immediately(clock, instant_sleep):
    item = make_item()
    event_id = item["event"]["id"]
    ledger = StubLedger([item])
    service = StubService()
    service.queue(
        event_id,
        make_response(
            event_id,
            VerificationStatus.UNVERIFIABLE,
            error_code=ErrorCode.UNSUPPORTED_MODEL,
        ),
    )
    runner = make_runner(ledger, service)

    assert await runner.run_until_drained() == 0
    assert ledger.written[0]["result"] == "unverifiable"
    assert ledger.written[0]["error_code"] == "unsupported_model"
    assert len(service.calls) == 1


async def test_cooling_down_head_does_not_starve_next(clock):
    newest = make_item()
    older = make_item()
    newest_id = newest["event"]["id"]
    older_id = older["event"]["id"]
    ledger = StubLedger([newest, older])  # newest-first, as the ledger returns them
    service = StubService()
    service.queue(
        newest_id,
        make_response(
            newest_id,
            VerificationStatus.UNVERIFIABLE,
            error_code=ErrorCode.VERIFIER_UNHEALTHY,
        ),
    )
    service.queue(older_id, make_response(older_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service)

    # Newest fails transiently and enters cooldown.
    assert await runner._poll_once(0.1) == "cooling_down"
    # Next poll skips the cooling-down head and verifies the older event.
    assert await runner._poll_once(0.1) == "processed"
    assert ledger.written[0]["inference_event_id"] == older_id


async def test_write_errors_do_not_count_attempts(clock):
    item = make_item()
    event_id = item["event"]["id"]
    ledger = StubLedger([item])
    service = StubService()
    service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service)

    ledger.write_error = httpx.ConnectError("ledger down")
    with pytest.raises(httpx.HTTPError):
        await runner._poll_once(0.1)
    assert runner._attempts == {}

    # Ledger back up: the event is re-picked and written normally.
    ledger.write_error = None
    assert await runner._poll_once(0.1) == "processed"
    assert [p["inference_event_id"] for p in ledger.written] == [event_id]


async def test_delta_max_passed_to_verifier(clock, instant_sleep):
    item = make_item()
    event_id = item["event"]["id"]
    ledger = StubLedger([item])
    ledger.model_row["delta_max"] = 5.0
    service = StubService()
    service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service)

    assert await runner.run_until_drained() == 0
    assert service.margin_clips == [5.0]


async def test_null_delta_max_passed_as_none(clock, instant_sleep):
    item = make_item()
    event_id = item["event"]["id"]
    ledger = StubLedger([item])  # delta_max: None in the default row
    service = StubService()
    service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service)

    assert await runner.run_until_drained() == 0
    assert service.margin_clips == [None]


async def test_drain_loop_verifies_concurrently_up_to_model_concurrency():
    # Deliberately does not use the `clock` fixture: it freezes
    # time.monotonic globally, which also stalls asyncio's own scheduling
    # and would hang the real asyncio.sleep() used by the fake service below.
    items = [make_item() for _ in range(6)]
    ledger = StubLedger(items)
    service = ConcurrencyTrackingService()
    for item in items:
        event_id = item["event"]["id"]
        service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service, max_concurrent_verifications_per_model=3)

    assert await runner.run_until_drained() == 0
    assert len(ledger.written) == 6
    # Real concurrency: with 3 slots and 6 slow items, multiple verify()
    # calls must have overlapped in flight at once.
    assert service.peak_in_flight >= 2


async def test_drain_loop_stays_serial_when_concurrency_is_one():
    # Same reason as above: no `clock` fixture, so the real event loop
    # clock actually advances for asyncio.sleep().
    items = [make_item() for _ in range(3)]
    ledger = StubLedger(items)
    service = ConcurrencyTrackingService()
    for item in items:
        event_id = item["event"]["id"]
        service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service, max_concurrent_verifications_per_model=1)

    assert await runner.run_until_drained() == 0
    assert len(ledger.written) == 3
    assert service.peak_in_flight == 1


async def test_poll_once_cooling_down_when_all_picked_items_in_cooldown(clock):
    items = [make_item(), make_item()]
    ledger = StubLedger(items)
    service = StubService()
    runner = make_runner(ledger, service, max_concurrent_verifications_per_model=3)
    for item in items:
        runner._next_retry_at[item["event"]["id"]] = clock.now + 100

    outcome = await runner._poll_once(0.1)

    assert outcome == "cooling_down"
    assert ledger.written == []
    assert service.calls == []


async def test_poll_once_concurrent_ledger_error_still_records_other_verdicts(clock):
    items = [make_item() for _ in range(3)]
    fail_event_id = items[1]["event"]["id"]
    ledger = FlakyLedger(items, fail_event_id=fail_event_id)
    service = StubService()
    for item in items:
        event_id = item["event"]["id"]
        service.queue(event_id, make_response(event_id, VerificationStatus.PASS, margins=[0.1]))
    runner = make_runner(ledger, service, max_concurrent_verifications_per_model=3)

    with pytest.raises(httpx.HTTPError):
        await runner._poll_once(0.1)

    written_ids = {p["inference_event_id"] for p in ledger.written}
    assert written_ids == {items[0]["event"]["id"], items[2]["event"]["id"]}
