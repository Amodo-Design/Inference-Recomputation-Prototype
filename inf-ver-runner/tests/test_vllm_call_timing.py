"""VerifyResponse carries the vLLM-call bracket timestamps (None when the
call never ran)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import (
    TokenMetricSummary,
    VerificationStatus,
    VerifyResponse,
)


def test_vllm_call_timestamps_default_none():
    resp = VerifyResponse(
        request_id="r",
        status=VerificationStatus.UNVERIFIABLE,
        reason="no call",
        metrics=TokenMetricSummary(
            token_count=0, exact_match_count=0, exact_match_ratio=0.0
        ),
        latency_ms=1,
    )
    assert resp.vllm_call_started_at is None
    assert resp.vllm_call_completed_at is None


def test_vllm_call_timestamps_roundtrip():
    t0 = datetime(2026, 7, 30, 0, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 7, 30, 0, 0, 1, tzinfo=timezone.utc)
    resp = VerifyResponse(
        request_id="r",
        status=VerificationStatus.PASS,
        reason="ok",
        metrics=TokenMetricSummary(
            token_count=1, exact_match_count=1, exact_match_ratio=1.0
        ),
        latency_ms=1000,
        vllm_call_started_at=t0,
        vllm_call_completed_at=t1,
    )
    assert (resp.vllm_call_completed_at - resp.vllm_call_started_at).total_seconds() == 1.0
