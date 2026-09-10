"""started_at / pod identity columns round-trip through the event CRUD."""

from __future__ import annotations

import base64
import os
import uuid

MODEL_ID = "22222222-2222-2222-2222-222222222222"
HW_ID = "33333333-3333-3333-3333-333333333333"


def _b64_32() -> str:
    return base64.b64encode(os.urandom(32)).decode()


async def _seed(client) -> None:
    await client.post("/models", json={"model_id": MODEL_ID, "model_name": "m-gpu"})
    await client.post(
        "/hardware", json={"hardware_id": HW_ID, "hostname": "gpu-node-1"}
    )


def _event_body(**extra) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "session_id": "s-gpu",
        "ts": "2026-07-30T00:00:10Z",
        "model_id": MODEL_ID,
        "hardware_id": HW_ID,
        "hash_input_raw_logits": _b64_32(),
        "hash_output_raw_logits": _b64_32(),
        **extra,
    }


async def test_inference_event_started_at_roundtrip(client):
    await _seed(client)
    body = _event_body(started_at="2026-07-30T00:00:00Z")
    r = await client.post("/inference-events", json=body)
    assert r.status_code == 201, r.text
    got = (await client.get(f"/inference-events/{body['id']}")).json()
    assert got["started_at"] == "2026-07-30T00:00:00Z"


async def test_started_at_is_optional(client):
    await _seed(client)
    body = _event_body()
    r = await client.post("/inference-events", json=body)
    assert r.status_code == 201, r.text
    got = (await client.get(f"/inference-events/{body['id']}")).json()
    assert got["started_at"] is None


async def test_pod_identity_roundtrip(client):
    await _seed(client)
    body = _event_body(
        started_at="2026-07-30T00:00:00Z",
        pod_name="kserve-qwen2-5-1-5b-instruct-abc12",
        node_name="gpu-node-1",
    )
    r = await client.post("/inference-events", json=body)
    assert r.status_code == 201, r.text
    got = (await client.get(f"/inference-events/{body['id']}")).json()
    assert got["pod_name"] == "kserve-qwen2-5-1-5b-instruct-abc12"
    assert got["node_name"] == "gpu-node-1"
