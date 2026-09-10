"""Shared builders for the runner/mapper tests: settings, ledger items in the
exact shape message-writer writes them, and canned VerifyResponses."""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

from app.config import LaunchSampling, Settings, VerifierModelConfig
from app.models import (
    ErrorCode,
    TokenMetricSummary,
    VerificationStatus,
    VerifyResponse,
)


def canonical_b64(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return base64.b64encode(raw).decode()


# The PROVER model whose events the test runner drains.
MODEL_ID = "11111111-2222-5333-8444-555555555555"


def make_settings(**overrides: Any) -> Settings:
    launch_sampling = LaunchSampling(
        seed=overrides.pop("launch_seed", None),
        temperature=overrides.pop("launch_temperature", None),
        top_k=overrides.pop("launch_top_k", None),
        top_p=overrides.pop("launch_top_p", None),
    )
    values: dict[str, Any] = dict(
        ledger_api_url="http://ledger-api:8000",
        ledger_timeout_seconds=10,
        poll_interval_seconds=0.01,
        poll_batch_size=10,
        max_verify_attempts=3,
        retry_backoff_seconds=15.0,
        max_ledger_failures=10,
        max_declare_attempts=30,
        vllm_ready_timeout_seconds=900.0,
        runner_hostname="inf-ver-runner",
        preload_models=False,
        verification_timeout_seconds=120,
        max_concurrent_verifications_per_model=1,
        margin_clip=10.0,
        enable_activation_verification=False,
        vllm_url="http://vllm:8000/v1",
        vllm_api_key=None,
        runner_model_id=MODEL_ID,
        launch_sampling=launch_sampling,
        model=VerifierModelConfig(id="openai/gpt-oss-20b", display_name="gpt-oss-20b"),
    )
    values.update(overrides)
    return Settings(**values)


def make_item(
    *,
    event_id: str | None = None,
    prompt_token_ids: list[int] | None = [1, 2, 3],
    output_token_ids: list[int] | None = [7, 8],
    input_payload: dict[str, Any] | None = None,
    seed: int | None = 42,
) -> dict[str, Any]:
    """One /inference-events/unverified item, payloads encoded exactly as
    message-writer's transform.py writes them."""
    if input_payload is None:
        input_payload = (
            {"prompt_token_ids": prompt_token_ids}
            if prompt_token_ids is not None
            else {"prompt_text": "hello"}
        )
    return {
        "event": {
            "id": event_id or str(uuid.uuid4()),
            "session_id": "chat-1",
            "ts": "2026-01-01T00:00:00Z",
            "input_raw_logits": canonical_b64(input_payload),
            "output_raw_logits": canonical_b64(
                {"output_token_ids": output_token_ids, "output_logprobs": None}
            )
            if output_token_ids is not None
            else None,
            "input_text_representation": "hello",
            "output_text_representation": "world",
        },
        "model": {
            "model_name": "Qwen/Qwen3-8B",
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "seed": seed,
        },
    }


def make_response(
    request_id: str,
    status: VerificationStatus,
    *,
    error_code: ErrorCode | None = None,
    margins: list[float] | None = None,
) -> VerifyResponse:
    margins = margins or []
    return VerifyResponse(
        request_id=request_id,
        status=status,
        reason="test reason",
        error_code=error_code,
        match_target=0.1,
        metrics=TokenMetricSummary(
            token_count=len(margins),
            exact_match_count=len(margins),
            exact_match_ratio=1.0 if margins else 0.0,
            margins=margins,
            mean_margin=sum(margins) / len(margins) if margins else None,
        ),
        latency_ms=12,
        verifier_model_id="gpt-oss-20b",
        prover_output="world",
        verifier_output="world",
        prover_output_token_ids=[7, 8],
        verifier_output_token_ids=[7, 8],
    )
