"""Map ledger inference events to VerifyRequests and verdicts back to
verification_event payloads.

The ledger stores the tapped token payloads as canonical JSON bytes
(base64 over the API) written by message-writer: input is
``{"prompt_token_ids": [...]}`` (or ``{"prompt_text": ...}`` when the tap had
no ids) and output is ``{"output_token_ids": [...], "output_logprobs": [...]}``.
Missing or undecodable payloads leave the token-id fields None — the verifier
already turns that into UNVERIFIABLE / tokenization_mismatch, so failure
classification stays in one place.
"""

from __future__ import annotations

import base64
import json
import statistics
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import ModelRequest, SamplingConfig, VerifyMetadata, VerifyRequest, VerifyResponse
from .pod_identity import PodIdentity


def _decode_payload(raw_b64: str | None) -> dict[str, Any]:
    if not raw_b64:
        return {}
    try:
        decoded = json.loads(base64.b64decode(raw_b64))
    except (ValueError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _token_ids(payload: dict[str, Any], key: str) -> list[int] | None:
    ids = payload.get(key)
    if not isinstance(ids, list) or not all(isinstance(t, int) for t in ids):
        return None
    return ids


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def inference_event_to_verify_request(item: dict[str, Any]) -> VerifyRequest:
    """Build a VerifyRequest from one /inference-events/unverified item
    (``{"event": InferenceEventRead, "model": ModelRead}``)."""
    event = item["event"]
    model = item["model"]

    prompt_token_ids = _token_ids(
        _decode_payload(event.get("input_raw_logits")), "prompt_token_ids"
    )
    output_payload = _decode_payload(event.get("output_raw_logits"))
    output_token_ids = _token_ids(output_payload, "output_token_ids")

    # Capture-integrity fields (absent on events recorded by older taps).
    raw_usage = output_payload.get("usage_completion_tokens")
    usage_completion_tokens = raw_usage if isinstance(raw_usage, int) else None
    raw_constrained = output_payload.get("constrained_decoding")
    constrained_decoding = raw_constrained if isinstance(raw_constrained, bool) else None

    return VerifyRequest(
        request_id=event["id"],
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        model=ModelRequest(name=model["model_name"]),
        sampling_config=SamplingConfig(
            seed=model.get("seed"),
            temperature=model.get("temperature"),
            top_k=model.get("top_k"),
            top_p=model.get("top_p"),
            # Required non-None by the verifier's config check but never used
            # by it, and the ledger model row has no such value.
            max_output_tokens=len(output_token_ids) if output_token_ids else None,
        ),
        metadata=VerifyMetadata(
            user_prompt=event.get("input_text_representation"),
            prover_output=event.get("output_text_representation"),
        ),
        usage_completion_tokens=usage_completion_tokens,
        constrained_decoding=constrained_decoding,
    )


def build_verification_event_payload(
    request: VerifyRequest,
    response: VerifyResponse,
    hardware_id: str,
    verifier_model_id: str,
    pod_identity: PodIdentity | None = None,
    runner_concurrency: int = 1,
) -> dict[str, Any]:
    """Build the VerificationEventCreate JSON body for the ledger.

    ``request.request_id`` is the inference event's UUID (set by
    inference_event_to_verify_request); ``verifier_model_id`` is the ledger
    model row declared at startup.
    """
    metrics = response.metrics
    margins = list(metrics.margins)
    return {
        "id": str(uuid4()),
        "inference_event_id": request.request_id,
        "hardware_id": hardware_id,
        # GPU-attributable window: the bracket around the actual vLLM call
        # (Task: verifier.py). Fallback to build time only when no call ran —
        # started_at stays None then, so the enrichment poller skips it.
        "ts": (
            response.vllm_call_completed_at.isoformat()
            if response.vllm_call_completed_at
            else datetime.now(timezone.utc).isoformat()
        ),
        "started_at": (
            response.vllm_call_started_at.isoformat()
            if response.vllm_call_started_at
            else None
        ),
        "pod_name": pod_identity.pod_name if pod_identity else None,
        "node_name": pod_identity.node_name if pod_identity else None,
        "result": response.status.value,
        "result_detail": response.reason,
        "error_code": response.error_code.value if response.error_code else None,
        "verification_threshold": response.match_target,
        "verifier_model_id": verifier_model_id,
        "exact_match_level_pct": metrics.exact_match_ratio if metrics.token_count else None,
        "mean_logit_difference": metrics.mean_margin,
        "std_dev_logit_difference": statistics.pstdev(margins) if len(margins) >= 2 else None,
        "logit_difference_margins": (
            base64.b64encode(_canonical(margins)).decode("ascii") if margins else None
        ),
        "verifier_detail": {
            "prompt_token_count": len(request.prompt_token_ids or []),
            "output_token_count": len(request.output_token_ids or []),
            # The RUNNER_MODEL_CONCURRENCY in force for this drain — durable
            # per event so a run records what it actually ran with.
            "runner_concurrency": runner_concurrency,
            # Verifier-side timing; kept out of the table proper to stay lean.
            "latency_ms": response.latency_ms,
            "prover_output_text": response.prover_output,
            "verifier_output_text": response.verifier_output,
            "prover_output_token_ids": response.prover_output_token_ids,
            "verifier_output_token_ids": response.verifier_output_token_ids,
            "output_token_comparison": [
                item.model_dump(mode="json") for item in response.output_token_comparison
            ],
            "metrics": metrics.model_dump(mode="json"),
        },
    }
