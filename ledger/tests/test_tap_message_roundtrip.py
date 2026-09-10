"""End-to-end value fidelity: a sample message from the inference system (the
tap message emitted by inf-proxy) is converted to the ledger's inference-event
shape, stored, and read back — and the token-id / logit / text values still
match after deserialisation.

The conversion mirrors message-writer/app/transform.py. It is duplicated here
(rather than imported) because message-writer also uses the package name `app`,
which would collide with the ledger's `app` package under test.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid

import pytest

# A representative tap message (see message-writer/app/schemas.py).
SAMPLE_TAP = {
    "tap": {
        "source": "inf-proxy",
        "hostname": "dummy-model",
        "endpoint": "/v1/chat/completions",
        "streamed": True,
        "received_at": "2026-07-09T12:00:00.000Z",
        "completed_at": "2026-07-09T12:00:03.500Z",
    },
    "session": {"session_id": "chat-xyz", "chat_id": "chat-xyz", "response_id": "resp-1"},
    "model": {"name": "openai/gpt-oss-20b"},
    "sampling": {"seed": 42, "temperature": 0.7},
    "request": {
        "prompt_text": "user: hello there",
        "prompt_token_ids": [1000, 1007, 1014, 1021],
    },
    "response": {
        "output_text": "Hello! This is a canned reply.",
        "output_token_ids": [20000, 20013, 20026],
        "output_logprobs": [
            {"token": "Hello", "logprob": -0.1},
            {"token": "!", "logprob": -0.4},
        ],
        "finish_reason": "stop",
        "tool_calls": False,
    },
}


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def tap_to_inference_event(tap: dict, *, model_id: str, hardware_id: str) -> dict:
    """Mirror of message-writer's build_inference_event encoding."""
    req, resp = tap["request"], tap["response"]
    input_bytes = _canonical({"prompt_token_ids": req["prompt_token_ids"]})
    output_bytes = _canonical(
        {"output_token_ids": resp["output_token_ids"], "output_logprobs": resp["output_logprobs"]}
    )
    return {
        "id": str(uuid.uuid4()),
        "session_id": tap["session"]["session_id"],
        "ts": tap["tap"]["completed_at"],
        "model_id": model_id,
        "hardware_id": hardware_id,
        "input_raw_logits": _b64(input_bytes),
        "output_raw_logits": _b64(output_bytes),
        "hash_input_raw_logits": _b64(hashlib.sha256(input_bytes).digest()),
        "hash_output_raw_logits": _b64(hashlib.sha256(output_bytes).digest()),
        "input_text_representation": req["prompt_text"],
        "output_text_representation": resp["output_text"],
    }


async def test_tap_message_roundtrips_through_ledger(client):
    tap = SAMPLE_TAP

    # 1. Model declares itself (as the dummy model does on startup). Pin
    #    started_at before the tap timestamp so resolution is deterministic
    #    regardless of wall-clock time.
    r = await client.post(
        "/model-deployments/declare",
        json={
            "hostname": tap["tap"]["hostname"],
            "model_name": tap["model"]["name"],
            "seed": tap["sampling"]["seed"],
            "temperature": tap["sampling"]["temperature"],
            "started_at": "2026-07-09T11:00:00Z",
        },
    )
    assert r.status_code == 201, r.text

    # 2. Resolve (hostname, ts) -> model_id + hardware_id (as the message writer does).
    r = await client.get(
        "/model-deployments/resolve",
        params={
            "hostname": tap["tap"]["hostname"],
            "ts": tap["tap"]["completed_at"],
            "model_name": tap["model"]["name"],
        },
    )
    assert r.status_code == 200, r.text
    resolution = r.json()
    assert resolution["model_name_matches"] is True

    # 3. Convert the tap message and store it.
    event = tap_to_inference_event(
        tap, model_id=resolution["model_id"], hardware_id=resolution["hardware_id"]
    )
    r = await client.post("/inference-events", json=event)
    assert r.status_code == 201, r.text

    # 4. Read back and confirm every value survived.
    r = await client.get(f"/inference-events/{event['id']}")
    assert r.status_code == 200, r.text
    got = r.json()

    assert got["session_id"] == tap["session"]["session_id"]
    assert got["model_id"] == resolution["model_id"]
    assert got["hardware_id"] == resolution["hardware_id"]
    assert got["input_text_representation"] == tap["request"]["prompt_text"]
    assert got["output_text_representation"] == tap["response"]["output_text"]

    # Decode BYTEA -> JSON and confirm the token ids / logprobs match the source.
    input_decoded = json.loads(base64.b64decode(got["input_raw_logits"]))
    output_decoded = json.loads(base64.b64decode(got["output_raw_logits"]))
    assert input_decoded["prompt_token_ids"] == tap["request"]["prompt_token_ids"]
    assert output_decoded["output_token_ids"] == tap["response"]["output_token_ids"]
    assert output_decoded["output_logprobs"] == tap["response"]["output_logprobs"]

    # Hashes are integrity-consistent with the decoded bytes.
    assert base64.b64decode(got["hash_input_raw_logits"]) == hashlib.sha256(
        base64.b64decode(got["input_raw_logits"])
    ).digest()
    assert base64.b64decode(got["hash_output_raw_logits"]) == hashlib.sha256(
        base64.b64decode(got["output_raw_logits"])
    ).digest()
