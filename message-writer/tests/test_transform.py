"""Pure tests for build_inference_event (app/transform.py).

Covers the tap->ledger encoding, the hash-matches-raw integrity invariant, the
fallback chains, the STORE_RAW_LOGITS toggle, and serialisation determinism.
"""

from __future__ import annotations

import base64
import hashlib
import uuid

import pytest

from app import transform
from app.schemas import TapMessage

# Golden hashes for the sample below (guard the canonical serialisation).
GOLDEN_INPUT_HASH = "aADKAup8Ea2J15xm5mWa1ECCmoLoJ4b260ZapI+74QY="
GOLDEN_OUTPUT_HASH = "3dSGJqI6ypoHkEUT/qHuPxEG6sZfiy1//k3pyTMTODo="

SAMPLE = {
    "tap": {
        "hostname": "h",
        "completed_at": "2026-01-01T00:00:00Z",
        "received_at": "2026-01-01T00:00:00Z",
    },
    "session": {"session_id": "chat-1", "response_id": "resp-1"},
    "model": {"name": "m"},
    "request": {"prompt_text": "hi", "prompt_token_ids": [1, 2, 3]},
    "response": {
        "output_text": "yo",
        "output_token_ids": [40, 41],
        "output_logprobs": None,
        "finish_reason": "stop",
        "tool_calls": False,
    },
}


def _msg(**tree) -> TapMessage:
    merged = {**SAMPLE, **tree}
    return TapMessage.model_validate(merged)


def _build(msg, model_id="MID", hardware_id="HID"):
    return transform.build_inference_event(msg, model_id=model_id, hardware_id=hardware_id)


def test_happy_path_fields_and_golden_hashes():
    event = _build(_msg())

    assert event["model_id"] == "MID"
    assert event["hardware_id"] == "HID"
    assert event["session_id"] == "chat-1"
    assert event["ts"] == "2026-01-01T00:00:00Z"
    assert event["input_text_representation"] == "hi"
    assert event["output_text_representation"] == "yo"
    assert event["hash_input_raw_logits"] == GOLDEN_INPUT_HASH
    assert event["hash_output_raw_logits"] == GOLDEN_OUTPUT_HASH
    # id is a fresh valid UUID.
    uuid.UUID(event["id"])


def test_hash_matches_stored_raw():
    """The stored raw bytes must hash to the stored hash (integrity)."""
    event = _build(_msg())
    for raw_key, hash_key in (
        ("input_raw_logits", "hash_input_raw_logits"),
        ("output_raw_logits", "hash_output_raw_logits"),
    ):
        raw = base64.b64decode(event[raw_key])
        expected = base64.b64encode(hashlib.sha256(raw).digest()).decode()
        assert event[hash_key] == expected


def test_input_falls_back_to_prompt_text_when_no_token_ids():
    event = _build(_msg(request={"prompt_text": "only text", "prompt_token_ids": None}))
    decoded = base64.b64decode(event["input_raw_logits"]).decode()
    assert "prompt_text" in decoded
    assert "only text" in decoded
    assert "prompt_token_ids" not in decoded


def test_session_id_fallback_chain():
    # session_id present -> used.
    assert _build(_msg())["session_id"] == "chat-1"
    # missing session_id -> response_id.
    e = _build(_msg(session={"response_id": "resp-9"}))
    assert e["session_id"] == "resp-9"
    # neither -> "unknown".
    e = _build(_msg(session={}))
    assert e["session_id"] == "unknown"


def test_ts_falls_back_to_received_at():
    e = _build(_msg(tap={"hostname": "h", "received_at": "2026-02-02T00:00:00Z"}))
    assert e["ts"] == "2026-02-02T00:00:00Z"


def test_store_raw_logits_toggle(monkeypatch):
    # Default (true): raw stored.
    e = _build(_msg())
    assert e["input_raw_logits"] is not None
    assert e["output_raw_logits"] is not None

    # Disabled: raw is None but hashes remain.
    monkeypatch.setattr(transform, "STORE_RAW_LOGITS", False)
    e = _build(_msg())
    assert e["input_raw_logits"] is None
    assert e["output_raw_logits"] is None
    assert e["hash_input_raw_logits"] == GOLDEN_INPUT_HASH
    assert e["hash_output_raw_logits"] == GOLDEN_OUTPUT_HASH


def test_deterministic_hashes_unique_ids():
    a = _build(_msg())
    b = _build(_msg())
    # Same content -> same hashes/raw.
    assert a["hash_input_raw_logits"] == b["hash_input_raw_logits"]
    assert a["output_raw_logits"] == b["output_raw_logits"]
    # But each event gets a fresh id.
    assert a["id"] != b["id"]
