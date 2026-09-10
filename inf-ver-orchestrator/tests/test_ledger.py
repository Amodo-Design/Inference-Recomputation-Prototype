"""Pending-queue grouping against a mocked ledger view endpoint."""

from __future__ import annotations

import httpx
import respx

from app.ledger import LedgerClient

BASE = "http://ledger-api:8000"


def view_item(model_id: str, *, name="Org/Model", threshold=0.1, seed=42) -> dict:
    return {
        "id": "e-1",
        "session_id": "s",
        "ts": "2026-01-01T00:00:00Z",
        "model_id": model_id,
        "model_name": name,
        "sampling_config": {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "seed": seed,
            "decoding_algorithm": "gumbel_max",
        },
        "verification_threshold": threshold,
        "input_text_representation": "hi",
        "output_text_representation": "yo",
        "hardware_id": "hw",
    }


def page(items: list[dict], total: int, limit=500, offset=0) -> dict:
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@respx.mock
def test_groups_pending_by_model():
    items = [
        view_item("m-a"),
        view_item("m-b", name="Org/Other", threshold=None, seed=7),
        view_item("m-a"),
    ]
    respx.get(f"{BASE}/inference-events/unverified/view").mock(
        return_value=httpx.Response(200, json=page(items, total=3))
    )
    client = LedgerClient(BASE)

    groups = {g.model_id: g for g in client.pending_by_model()}

    assert groups["m-a"].pending == 2
    assert groups["m-a"].eligible
    assert groups["m-a"].seed == 42
    assert groups["m-b"].pending == 1
    assert not groups["m-b"].eligible  # NULL threshold = paused
    assert groups["m-b"].verification_threshold is None


@respx.mock
def test_pages_through_full_queue():
    first = [view_item("m-a")] * 2
    second = [view_item("m-a")]
    route = respx.get(f"{BASE}/inference-events/unverified/view")
    route.side_effect = [
        httpx.Response(200, json=page(first, total=3)),
        httpx.Response(200, json=page(second, total=3, offset=2)),
    ]
    client = LedgerClient(BASE)

    groups = client.pending_by_model()

    assert len(route.calls) == 2
    assert route.calls[1].request.url.params["offset"] == "2"
    assert groups[0].pending == 3


@respx.mock
def test_empty_queue():
    respx.get(f"{BASE}/inference-events/unverified/view").mock(
        return_value=httpx.Response(200, json=page([], total=0))
    )
    assert LedgerClient(BASE).pending_by_model() == []
