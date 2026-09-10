"""Simple tests: an inference event stores and reads back, and BYTEA columns
survive the base64 round-trip and decode to the exact original bytes."""

from __future__ import annotations

import base64
import os
import uuid

import pytest

MODEL_ID = "11111111-1111-1111-1111-111111111111"
HARDWARE_ID = "22222222-2222-2222-2222-222222222222"


async def _seed_model_and_hardware(client) -> None:
    r = await client.post(
        "/models",
        json={"model_id": MODEL_ID, "model_name": "test-model", "seed": 7},
    )
    assert r.status_code == 201, r.text
    r = await client.post(
        "/hardware",
        json={"hardware_id": HARDWARE_ID, "hostname": "test-host"},
    )
    assert r.status_code == 201, r.text


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


async def test_inference_event_stores_and_bytea_decodes(client):
    await _seed_model_and_hardware(client)

    event_id = str(uuid.uuid4())
    hash_in = os.urandom(32)
    hash_out = os.urandom(32)
    raw_in = b"\x00\x01\x02\x03raw-input-logits"
    raw_out = b"\xff\xfe\xfd\xfcraw-output-logits"

    payload = {
        "id": event_id,
        "session_id": "chat-1",
        "ts": "2026-07-09T12:00:00Z",
        "model_id": MODEL_ID,
        "hardware_id": HARDWARE_ID,
        "input_raw_logits": _b64(raw_in),
        "output_raw_logits": _b64(raw_out),
        "hash_input_raw_logits": _b64(hash_in),
        "hash_output_raw_logits": _b64(hash_out),
        "input_text_representation": "hello",
        "output_text_representation": "world",
    }
    r = await client.post("/inference-events", json=payload)
    assert r.status_code == 201, r.text

    r = await client.get(f"/inference-events/{event_id}")
    assert r.status_code == 200, r.text
    got = r.json()

    # Scalars intact.
    assert got["session_id"] == "chat-1"
    assert got["model_id"] == MODEL_ID
    assert got["hardware_id"] == HARDWARE_ID
    assert got["input_text_representation"] == "hello"
    assert got["output_text_representation"] == "world"

    # BYTEA columns come back as base64 and decode to the exact original bytes.
    assert base64.b64decode(got["hash_input_raw_logits"]) == hash_in
    assert base64.b64decode(got["hash_output_raw_logits"]) == hash_out
    assert base64.b64decode(got["input_raw_logits"]) == raw_in
    assert base64.b64decode(got["output_raw_logits"]) == raw_out


async def test_opaque_binary_roundtrips_unmangled(client):
    """A large random blob (stand-in for a packed logit tensor) must survive
    byte-for-byte — proving BYTEA is not text/JSON-normalised."""
    await _seed_model_and_hardware(client)

    event_id = str(uuid.uuid4())
    blob = os.urandom(65536)  # ~64 KiB of arbitrary binary

    payload = {
        "id": event_id,
        "session_id": "chat-blob",
        "ts": "2026-07-09T12:00:00Z",
        "model_id": MODEL_ID,
        "hardware_id": HARDWARE_ID,
        "output_raw_logits": _b64(blob),
        "hash_input_raw_logits": _b64(os.urandom(32)),
        "hash_output_raw_logits": _b64(os.urandom(32)),
    }
    r = await client.post("/inference-events", json=payload)
    assert r.status_code == 201, r.text

    r = await client.get(f"/inference-events/{event_id}")
    assert r.status_code == 200
    decoded = base64.b64decode(r.json()["output_raw_logits"])
    assert decoded == blob
    assert len(decoded) == 65536


async def test_missing_inference_event_returns_404(client):
    r = await client.get(f"/inference-events/{uuid.uuid4()}")
    assert r.status_code == 404
