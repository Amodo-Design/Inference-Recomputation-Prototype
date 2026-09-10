"""UI-facing verification endpoints: /verification-events/view, /stats and the
DELETE routes, plus the extended verification_event round-trip."""

from __future__ import annotations

import base64
import json
import uuid


def _canonical_b64(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


HARDWARE_ID = str(uuid.uuid4())
MODEL_ID = str(uuid.uuid4())
# Separate model row for the verifier's declared model, to prove the view
# joins the model table twice (prover vs verifier).
VERIFIER_MODEL_ID = str(uuid.uuid4())


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
        },
    )
    assert r.status_code == 201, r.text
    r = await client.post(
        "/models",
        json={
            "model_id": VERIFIER_MODEL_ID,
            "model_name": "gpt-oss-20b",
            "decoding_algorithm": "gumbel_max",
        },
    )
    assert r.status_code == 201, r.text


async def _seed_verification(
    client,
    *,
    ts: str,
    result: str,
    result_detail: str = "ok",
    match_level: float | None = 0.9,
    margins: list[float] | None = None,
) -> tuple[str, str]:
    """Create one inference_event + one verification_event; returns their ids."""
    inference_id = str(uuid.uuid4())
    r = await client.post(
        "/inference-events",
        json={
            "id": inference_id,
            "session_id": "chat-1",
            "ts": ts,
            "model_id": MODEL_ID,
            "hardware_id": HARDWARE_ID,
            "hash_input_raw_logits": _canonical_b64({"h": "i"}),
            "hash_output_raw_logits": _canonical_b64({"h": "o"}),
            "input_text_representation": "what is a llama?",
            "output_text_representation": "a camelid",
        },
    )
    assert r.status_code == 201, r.text

    verification_id = str(uuid.uuid4())
    r = await client.post(
        "/verification-events",
        json={
            "id": verification_id,
            "inference_event_id": inference_id,
            "hardware_id": HARDWARE_ID,
            "ts": ts,
            "result": result,
            "result_detail": result_detail,
            "error_code": "internal_error" if result == "unverifiable" else None,
            "verification_threshold": 0.1,
            "verifier_model_id": VERIFIER_MODEL_ID,
            "exact_match_level_pct": match_level,
            "mean_logit_difference": 0.02,
            "logit_difference_margins": (
                _canonical_b64(margins) if margins is not None else None
            ),
            "verifier_detail": {
                "prompt_token_count": 3,
                "output_token_count": 2,
                "latency_ms": 100,
                "output_token_comparison": [],
            },
        },
    )
    assert r.status_code == 201, r.text
    return inference_id, verification_id


async def test_extended_fields_round_trip(client):
    await _seed_refs(client)
    _, verification_id = await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="pass", margins=[0.1, 0.2]
    )
    r = await client.get(f"/verification-events/{verification_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "pass"
    assert body["verification_threshold"] == 0.1
    assert body["verifier_model_id"] == VERIFIER_MODEL_ID
    assert body["verifier_detail"]["prompt_token_count"] == 3


async def test_view_shape_ordering_and_margins(client):
    await _seed_refs(client)
    await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="pass", margins=[0.1, 0.2]
    )
    await _seed_verification(client, ts="2026-01-02T00:00:00Z", result="fail")

    r = await client.get("/verification-events/view")
    assert r.status_code == 200, r.text
    page = r.json()
    assert page["total"] == 2
    assert [item["result"] for item in page["items"]] == ["fail", "pass"]  # ts desc

    oldest = page["items"][1]
    assert oldest["model_name"] == "Qwen/Qwen3-8B"
    assert oldest["verifier_model_id"] == VERIFIER_MODEL_ID
    assert oldest["verifier_model_name"] == "gpt-oss-20b"  # joined verifier model row
    assert oldest["sampling_config"]["seed"] == 42
    assert oldest["difr_margins"] == [0.1, 0.2]
    assert oldest["session_id"] == "chat-1"
    assert oldest["input_text_representation"] == "what is a llama?"
    assert oldest["verifier_detail"]["output_token_count"] == 2


async def test_view_filters(client):
    await _seed_refs(client)
    await _seed_verification(client, ts="2026-01-01T00:00:00Z", result="pass")
    await _seed_verification(
        client, ts="2026-01-02T00:00:00Z", result="unverifiable", result_detail="vLLM down"
    )

    by_result = await client.get(
        "/verification-events/view", params={"result": "unverifiable"}
    )
    assert [i["result"] for i in by_result.json()["items"]] == ["unverifiable"]

    by_model = await client.get("/verification-events/view", params={"model": "qwen"})
    assert by_model.json()["total"] == 2
    # Matches on the verifier's model name too (joined via verifier_model_id).
    by_verifier = await client.get(
        "/verification-events/view", params={"model": "gpt-oss"}
    )
    assert by_verifier.json()["total"] == 2
    no_model = await client.get("/verification-events/view", params={"model": "nope"})
    assert no_model.json()["total"] == 0

    by_search = await client.get(
        "/verification-events/view", params={"search": "vllm down"}
    )
    assert by_search.json()["total"] == 1

    paged = await client.get(
        "/verification-events/view", params={"limit": 1, "offset": 1}
    )
    body = paged.json()
    assert body["total"] == 2
    assert len(body["items"]) == 1


async def test_stats(client):
    await _seed_refs(client)
    await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="pass", match_level=1.0
    )
    await _seed_verification(
        client, ts="2026-01-02T00:00:00Z", result="fail", match_level=0.5
    )
    await _seed_verification(
        client, ts="2026-01-03T00:00:00Z", result="unverifiable", match_level=None
    )

    r = await client.get("/verification-events/stats")
    body = r.json()
    assert body["total"] == 3
    assert body["pass_count"] == 1
    assert body["fail_count"] == 1
    assert body["unverifiable_count"] == 1
    assert body["average_match_level"] == 0.75  # NULLs excluded from avg


async def test_delete_one_is_complete(client):
    """Single delete removes the verification event AND its inference event —
    nothing re-queues."""
    await _seed_refs(client)
    inference_id, verification_id = await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="pass"
    )

    r = await client.delete(f"/verification-events/{verification_id}")
    assert r.status_code == 204

    assert (await client.get("/inference-events/unverified")).json() == []
    assert (await client.get(f"/inference-events/{inference_id}")).status_code == 404

    missing = await client.delete(f"/verification-events/{uuid.uuid4()}")
    assert missing.status_code == 404


async def test_delete_all_is_complete(client):
    await _seed_refs(client)
    await _seed_verification(client, ts="2026-01-01T00:00:00Z", result="pass")
    await _seed_verification(client, ts="2026-01-02T00:00:00Z", result="fail")

    r = await client.delete("/verification-events")
    assert r.json() == {"deleted": 2}
    assert (await client.get("/verification-events/view")).json()["total"] == 0
    # Complete delete: the inference events are gone too, so nothing re-queues.
    assert (await client.get("/inference-events/unverified")).json() == []


async def test_delete_filtered(client):
    """DELETE with filters removes only the matching subset."""
    await _seed_refs(client)
    pass_inf, _ = await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="pass"
    )
    fail_inf, _ = await _seed_verification(
        client, ts="2026-01-02T00:00:00Z", result="fail"
    )

    r = await client.delete("/verification-events", params={"result": "fail"})
    assert r.json() == {"deleted": 1}

    page = (await client.get("/verification-events/view")).json()
    assert page["total"] == 1
    assert page["items"][0]["result"] == "pass"
    assert (await client.get(f"/inference-events/{fail_inf}")).status_code == 404
    assert (await client.get(f"/inference-events/{pass_inf}")).status_code == 200

    r = await client.delete("/verification-events", params={"result": "fail"})
    assert r.json() == {"deleted": 0}


async def test_delete_keeps_inference_event_with_other_verifications(client):
    """Orphan rule: an inference event referenced by another verification
    event survives the delete of one of them."""
    await _seed_refs(client)
    inference_id, first_ver = await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="fail"
    )
    # Second verification event against the SAME inference event.
    second_ver = str(uuid.uuid4())
    r = await client.post(
        "/verification-events",
        json={
            "id": second_ver,
            "inference_event_id": inference_id,
            "hardware_id": HARDWARE_ID,
            "ts": "2026-01-02T00:00:00Z",
            "result": "pass",
            "verifier_model_id": VERIFIER_MODEL_ID,
        },
    )
    assert r.status_code == 201, r.text

    r = await client.delete(f"/verification-events/{first_ver}")
    assert r.status_code == 204
    # The pass verification still references the inference event: kept.
    assert (await client.get(f"/inference-events/{inference_id}")).status_code == 200

    r = await client.delete(f"/verification-events/{second_ver}")
    assert r.status_code == 204
    # Last reference gone: the inference event goes too.
    assert (await client.get(f"/inference-events/{inference_id}")).status_code == 404


async def test_stats_filters(client):
    await _seed_refs(client)
    await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="pass", match_level=1.0
    )
    await _seed_verification(
        client, ts="2026-01-02T00:00:00Z", result="fail", match_level=0.5,
        result_detail="vLLM down",
    )

    # model filter matches prover and verifier model names (ilike).
    by_model = await client.get(
        "/verification-events/stats", params={"model": "qwen"}
    )
    assert by_model.json()["total"] == 2
    no_model = await client.get(
        "/verification-events/stats", params={"model": "nope"}
    )
    assert no_model.json()["total"] == 0

    by_search = await client.get(
        "/verification-events/stats", params={"search": "vllm down"}
    )
    body = by_search.json()
    assert body["total"] == 1
    assert body["fail_count"] == 1
    assert body["pass_count"] == 0


async def test_replay_one_requeues_event(client):
    """Replay = the OLD delete semantics: verification event gone, inference
    event kept and visible in the unverified queue again."""
    await _seed_refs(client)
    inference_id, verification_id = await _seed_verification(
        client, ts="2026-01-01T00:00:00Z", result="fail"
    )

    r = await client.post(f"/verification-events/{verification_id}/replay")
    assert r.status_code == 204

    unverified = (await client.get("/inference-events/unverified")).json()
    assert [item["event"]["id"] for item in unverified] == [inference_id]
    assert (await client.get(f"/inference-events/{inference_id}")).status_code == 200

    missing = await client.post(f"/verification-events/{uuid.uuid4()}/replay")
    assert missing.status_code == 404


async def test_replay_filtered(client):
    await _seed_refs(client)
    await _seed_verification(client, ts="2026-01-01T00:00:00Z", result="pass")
    fail_inf, _ = await _seed_verification(
        client, ts="2026-01-02T00:00:00Z", result="fail"
    )

    r = await client.post("/verification-events/replay", params={"result": "fail"})
    assert r.json() == {"replayed": 1}

    page = (await client.get("/verification-events/view")).json()
    assert page["total"] == 1
    assert page["items"][0]["result"] == "pass"
    unverified = (await client.get("/inference-events/unverified")).json()
    assert [item["event"]["id"] for item in unverified] == [fail_inf]
