"""Verification-event payloads use the captured vLLM-call bracket for
ts/started_at and carry the resolved pod identity."""

from __future__ import annotations

from datetime import datetime, timezone

from app.mapper import build_verification_event_payload
from app.models import (
    ModelRequest,
    SamplingConfig,
    TokenMetricSummary,
    VerificationStatus,
    VerifyRequest,
    VerifyResponse,
)
from app.pod_identity import PodIdentity

T0 = datetime(2026, 7, 30, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 7, 30, 0, 0, 1, tzinfo=timezone.utc)


def _request() -> VerifyRequest:
    return VerifyRequest(
        request_id="11111111-1111-1111-1111-111111111111",
        prompt_token_ids=[1, 2],
        output_token_ids=[3],
        model=ModelRequest(name="m"),
        sampling_config=SamplingConfig(),
    )


def _response(**kwargs) -> VerifyResponse:
    return VerifyResponse(
        request_id="11111111-1111-1111-1111-111111111111",
        status=VerificationStatus.PASS,
        reason="ok",
        metrics=TokenMetricSummary(
            token_count=1, exact_match_count=1, exact_match_ratio=1.0
        ),
        latency_ms=1000,
        **kwargs,
    )


def test_captured_timestamps_and_identity():
    payload = build_verification_event_payload(
        _request(),
        _response(vllm_call_started_at=T0, vllm_call_completed_at=T1),
        hardware_id="hw",
        verifier_model_id="vm",
        pod_identity=PodIdentity("verify-m-pod-1", "gpu-node-1"),
    )
    assert payload["ts"] == T1.isoformat()
    assert payload["started_at"] == T0.isoformat()
    assert payload["pod_name"] == "verify-m-pod-1"
    assert payload["node_name"] == "gpu-node-1"


def test_no_call_falls_back_to_now_and_null_identity():
    payload = build_verification_event_payload(
        _request(), _response(), hardware_id="hw", verifier_model_id="vm"
    )
    # No vLLM call happened: started_at stays None (the poller skips the
    # event), ts falls back to build time as before.
    assert payload["started_at"] is None
    assert payload["ts"] is not None
    assert payload["pod_name"] is None
    assert payload["node_name"] is None
