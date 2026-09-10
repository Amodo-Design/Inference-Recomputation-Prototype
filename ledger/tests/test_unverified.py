"""GET /inference-events/unverified — the verifier's polling endpoint:
newest-first anti-join with the joined model config riding along."""

from __future__ import annotations

import base64
import json
import uuid


def _canonical_b64(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


HARDWARE_ID = str(uuid.uuid4())
MODEL_ID = str(uuid.uuid4())


async def _seed_refs(client) -> None:
    r = await client.post(
        "/hardware", json={"hardware_id": HARDWARE_ID, "hostname": "prover-host"}
    )
    assert r.status_code == 201, r.text
    r = await client.post(
        "/models",
        json={
            "model_id": MODEL_ID,
            "model_name": "Qwen/Qwen3-8B",
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "seed": 42,
            "decoding_algorithm": "gumbel_max",
        },
    )
    assert r.status_code == 201, r.text


async def _post_inference_event(client, *, ts: str, token_ids: list[int]) -> str:
    event_id = str(uuid.uuid4())
    input_b64 = _canonical_b64({"prompt_token_ids": [1, 2, 3]})
    output_b64 = _canonical_b64({"output_token_ids": token_ids, "output_logprobs": None})
    r = await client.post(
        "/inference-events",
        json={
            "id": event_id,
            "session_id": "chat-1",
            "ts": ts,
            "model_id": MODEL_ID,
            "hardware_id": HARDWARE_ID,
            "input_raw_logits": input_b64,
            "output_raw_logits": output_b64,
            "hash_input_raw_logits": _canonical_b64({"h": "i"}),
            "hash_output_raw_logits": _canonical_b64({"h": "o"}),
            "input_text_representation": "hello",
            "output_text_representation": "world",
        },
    )
    assert r.status_code == 201, r.text
    return event_id


async def test_unverified_newest_first_with_model(client):
    await _seed_refs(client)
    # Created out of ts order on purpose: ordering must come from ts, not insert order.
    mid = await _post_inference_event(client, ts="2026-01-02T00:00:00Z", token_ids=[20])
    oldest = await _post_inference_event(client, ts="2026-01-01T00:00:00Z", token_ids=[10])
    newest = await _post_inference_event(client, ts="2026-01-03T00:00:00Z", token_ids=[30])

    r = await client.get("/inference-events/unverified")
    assert r.status_code == 200, r.text
    items = r.json()
    assert [item["event"]["id"] for item in items] == [newest, mid, oldest]

    head = items[0]
    assert head["model"]["model_name"] == "Qwen/Qwen3-8B"
    assert head["model"]["seed"] == 42
    assert head["model"]["top_k"] == 50
    # Payloads round-trip as base64 of the canonical JSON.
    decoded = json.loads(base64.b64decode(head["event"]["output_raw_logits"]))
    assert decoded["output_token_ids"] == [30]


async def test_unverified_honours_limit(client):
    await _seed_refs(client)
    for day in (1, 2, 3):
        await _post_inference_event(
            client, ts=f"2026-01-0{day}T00:00:00Z", token_ids=[day]
        )
    r = await client.get("/inference-events/unverified", params={"limit": 2})
    assert len(r.json()) == 2


async def test_verified_event_disappears(client):
    await _seed_refs(client)
    event_id = await _post_inference_event(
        client, ts="2026-01-01T00:00:00Z", token_ids=[1]
    )
    r = await client.post(
        "/verification-events",
        json={
            "id": str(uuid.uuid4()),
            "inference_event_id": event_id,
            "hardware_id": HARDWARE_ID,
            "ts": "2026-01-01T00:01:00Z",
            "result": "pass",
            "verifier_model_id": MODEL_ID,
        },
    )
    assert r.status_code == 201, r.text

    r = await client.get("/inference-events/unverified")
    assert r.json() == []


async def test_unverified_view_envelope_and_shape(client):
    await _seed_refs(client)
    mid = await _post_inference_event(client, ts="2026-01-02T00:00:00Z", token_ids=[20])
    oldest = await _post_inference_event(client, ts="2026-01-01T00:00:00Z", token_ids=[10])
    newest = await _post_inference_event(client, ts="2026-01-03T00:00:00Z", token_ids=[30])

    r = await client.get("/inference-events/unverified/view")
    assert r.status_code == 200, r.text
    page = r.json()
    assert page["total"] == 3
    assert page["limit"] == 25
    assert page["offset"] == 0
    assert [item["id"] for item in page["items"]] == [newest, mid, oldest]

    head = page["items"][0]
    assert head["model_name"] == "Qwen/Qwen3-8B"
    assert head["sampling_config"]["seed"] == 42
    assert head["input_text_representation"] == "hello"
    # The view is the light shape: no raw-logit payloads ride along.
    assert "input_raw_logits" not in head
    assert "output_raw_logits" not in head


async def test_unverified_view_pagination(client):
    await _seed_refs(client)
    ids = {}
    for day in (1, 2, 3):
        ids[day] = await _post_inference_event(
            client, ts=f"2026-01-0{day}T00:00:00Z", token_ids=[day]
        )

    r = await client.get(
        "/inference-events/unverified/view", params={"limit": 2, "offset": 2}
    )
    page = r.json()
    assert page["total"] == 3
    assert [item["id"] for item in page["items"]] == [ids[1]]


async def test_unverified_view_excludes_verified(client):
    await _seed_refs(client)
    verified = await _post_inference_event(
        client, ts="2026-01-01T00:00:00Z", token_ids=[1]
    )
    pending = await _post_inference_event(
        client, ts="2026-01-02T00:00:00Z", token_ids=[2]
    )
    r = await client.post(
        "/verification-events",
        json={
            "id": str(uuid.uuid4()),
            "inference_event_id": verified,
            "hardware_id": HARDWARE_ID,
            "ts": "2026-01-01T00:01:00Z",
            "result": "pass",
            "verifier_model_id": MODEL_ID,
        },
    )
    assert r.status_code == 201, r.text

    page = (await client.get("/inference-events/unverified/view")).json()
    assert page["total"] == 1
    assert [item["id"] for item in page["items"]] == [pending]


# ---------------------------------------------------------------------------
# model_id filter — the runner polls only its own model's pending events
# ---------------------------------------------------------------------------
MODEL_ID_B = str(uuid.uuid4())


async def _post_event_for_model(client, model_id: str, *, ts: str) -> str:
    event_id = str(uuid.uuid4())
    r = await client.post(
        "/inference-events",
        json={
            "id": event_id,
            "session_id": "chat-2",
            "ts": ts,
            "model_id": model_id,
            "hardware_id": HARDWARE_ID,
            "input_raw_logits": _canonical_b64({"prompt_token_ids": [1]}),
            "output_raw_logits": _canonical_b64(
                {"output_token_ids": [9], "output_logprobs": None}
            ),
            "hash_input_raw_logits": _canonical_b64({"h": "i"}),
            "hash_output_raw_logits": _canonical_b64({"h": "o"}),
            "input_text_representation": "hi",
            "output_text_representation": "yo",
        },
    )
    assert r.status_code == 201, r.text
    return event_id


async def _seed_second_model(client) -> None:
    r = await client.post(
        "/models",
        json={"model_id": MODEL_ID_B, "model_name": "other/model", "seed": 7},
    )
    assert r.status_code == 201, r.text


async def test_unverified_filters_by_model_id(client):
    await _seed_refs(client)
    await _seed_second_model(client)
    a = await _post_event_for_model(client, MODEL_ID, ts="2026-01-01T00:00:00Z")
    b = await _post_event_for_model(client, MODEL_ID_B, ts="2026-01-02T00:00:00Z")

    items = (
        await client.get(
            "/inference-events/unverified", params={"model_id": MODEL_ID_B}
        )
    ).json()
    assert [item["event"]["id"] for item in items] == [b]

    items = (
        await client.get(
            "/inference-events/unverified", params={"model_id": MODEL_ID}
        )
    ).json()
    assert [item["event"]["id"] for item in items] == [a]


async def test_unverified_view_filters_by_model_id(client):
    await _seed_refs(client)
    await _seed_second_model(client)
    await _post_event_for_model(client, MODEL_ID, ts="2026-01-01T00:00:00Z")
    b = await _post_event_for_model(client, MODEL_ID_B, ts="2026-01-02T00:00:00Z")

    page = (
        await client.get(
            "/inference-events/unverified/view", params={"model_id": MODEL_ID_B}
        )
    ).json()
    assert page["total"] == 1
    assert [item["id"] for item in page["items"]] == [b]


async def test_unverified_view_model_name_filter(client):
    await _seed_refs(client)
    await _seed_second_model(client)
    await _post_event_for_model(client, MODEL_ID, ts="2026-01-01T00:00:00Z")
    await _post_event_for_model(client, MODEL_ID_B, ts="2026-01-02T00:00:00Z")

    r = await client.get(
        "/inference-events/unverified/view", params={"model": "qwen"}
    )
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["model_name"] == "Qwen/Qwen3-8B"

    r = await client.get(
        "/inference-events/unverified/view", params={"model": "nope"}
    )
    assert r.json()["total"] == 0
