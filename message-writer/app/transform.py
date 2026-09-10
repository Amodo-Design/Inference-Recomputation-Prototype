"""Transform a tap message into an inference_event payload for the ledger.

BYTEA columns are sent to the ledger as base64. We derive the input/output
"raw logits" from the captured token-id / logprob payloads: the raw bytes are a
canonical JSON serialisation, and the hashes are SHA-256 over those same bytes
(so a stored raw value always matches its hash).
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from typing import Any

from app.config import STORE_RAW_LOGITS
from app.schemas import TapMessage


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _sha256_b64(data: bytes) -> str:
    return _b64(hashlib.sha256(data).digest())


def build_inference_event(
    msg: TapMessage, *, model_id: str, hardware_id: str
) -> dict[str, Any]:
    # Input side: prefer the prompt token ids, fall back to the prompt text.
    if msg.request.prompt_token_ids is not None:
        input_bytes = _canonical({"prompt_token_ids": msg.request.prompt_token_ids})
    else:
        input_bytes = _canonical({"prompt_text": msg.request.prompt_text})

    # Output side: token ids + logprobs together.
    output_payload: dict[str, Any] = {
        "output_token_ids": msg.response.output_token_ids,
        "output_logprobs": msg.response.output_logprobs,
    }
    # Capture-integrity fields, only when the tap reported them — keeps the
    # canonical bytes (and their hash) unchanged for taps that don't.
    if msg.response.usage_completion_tokens is not None:
        output_payload["usage_completion_tokens"] = msg.response.usage_completion_tokens
    if msg.response.constrained_decoding is not None:
        output_payload["constrained_decoding"] = msg.response.constrained_decoding
    output_bytes = _canonical(output_payload)

    ts = msg.tap.completed_at or msg.tap.received_at
    session_id = msg.session.session_id or msg.session.response_id or "unknown"
    started_at = msg.tap.received_at

    return {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "ts": ts,
        "started_at": started_at,
        "model_id": model_id,
        "hardware_id": hardware_id,
        "pod_name": msg.tap.pod_name,
        "node_name": msg.tap.node_name,
        "input_raw_logits": _b64(input_bytes) if STORE_RAW_LOGITS else None,
        "output_raw_logits": _b64(output_bytes) if STORE_RAW_LOGITS else None,
        "hash_input_raw_logits": _sha256_b64(input_bytes),
        "hash_output_raw_logits": _sha256_b64(output_bytes),
        "input_text_representation": msg.request.prompt_text,
        "output_text_representation": msg.response.output_text,
    }
