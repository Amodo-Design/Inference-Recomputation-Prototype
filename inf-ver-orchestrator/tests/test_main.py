"""main.py wiring: apply_action's seed kinds and reconcile_once's kube calls
(pure logic against FakeKube — no cluster, no HTTP)."""

from __future__ import annotations

import unittest.mock

import httpx
import respx

from app.ledger import LedgerClient
from app.main import apply_action, reconcile_once
from app.reconciler import Action, JobState, ObservedJob

from helpers import FakeKube, make_model, make_settings

BASE = "http://ledger-api:8000"


def _view_item(model_id: str) -> dict:
    return {
        "id": "e-1",
        "session_id": "s",
        "ts": "2026-01-01T00:00:00Z",
        "model_id": model_id,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "sampling_config": {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "seed": 42,
            "decoding_algorithm": "gumbel_max",
        },
        "verification_threshold": 0.1,
        "input_text_representation": "hi",
        "output_text_representation": "yo",
        "hardware_id": "hw",
    }


def _page(items: list[dict]) -> dict:
    return {"items": items, "total": len(items), "limit": 500, "offset": 0}


def test_jobs_partitioned_by_prefix():
    """Seed Jobs never masquerade as runner Jobs in the reconciler input."""
    kube = FakeKube(
        jobs={
            "m1": ObservedJob(name="inf-ver-runner-m1", state=JobState.ACTIVE),
        },
        seed_jobs={
            "m1": ObservedJob(name="seed-m1", state=JobState.SUCCEEDED),
        },
    )
    assert set(kube.list_managed_jobs()) == {"m1"}
    assert kube.list_managed_jobs()["m1"].name.startswith("inf-ver-runner-")
    assert kube.list_managed_seed_jobs()["m1"].name.startswith("seed-")


def test_apply_action_seed_kinds():
    kube = FakeKube()
    settings = make_settings(model_cache_pvc="hf-model-cache")
    apply_action(
        Action("create_seed", "m1", model=make_model(model_id="m1")), kube, settings
    )
    assert any(j["metadata"]["name"].startswith("seed-") for j in kube.created_jobs)
    apply_action(Action("delete_seed", "m1", name="seed-m1"), kube, settings)
    assert "seed-m1" in kube.deleted_jobs


@respx.mock
def test_reconcile_once_gates_new_model_behind_seed_when_cache_enabled():
    """With ORCH_MODEL_CACHE_PVC set, a brand-new eligible model gets a seed
    Job instead of an immediate pair — proving reconcile_once really passes
    kube.list_managed_seed_jobs() and the cache flag into decide()."""
    respx.get(f"{BASE}/inference-events/unverified/view").mock(
        return_value=httpx.Response(200, json=_page([_view_item("m1")]))
    )
    ledger = LedgerClient(BASE)
    kube = FakeKube()
    settings = make_settings(model_cache_pvc="hf-model-cache")

    actions = reconcile_once(ledger, kube, settings)

    assert [a.kind for a in actions] == ["create_seed"]
    assert any(j["metadata"]["name"].startswith("seed-") for j in kube.created_jobs)
    assert not kube.created_llmisvcs


@respx.mock
def test_reconcile_once_spawns_pair_directly_when_cache_disabled():
    respx.get(f"{BASE}/inference-events/unverified/view").mock(
        return_value=httpx.Response(200, json=_page([_view_item("m1")]))
    )
    ledger = LedgerClient(BASE)
    kube = FakeKube()
    settings = make_settings(model_cache_pvc=None)

    actions = reconcile_once(ledger, kube, settings)

    assert set(a.kind for a in actions) == {"create_vllm", "create_job"}


@respx.mock
def test_reconcile_passes_exempt_set():
    # Anchor note: the brief's snippet assumed a `_settings`/
    # `_ledger_with_no_pending()` fixture pair; this file instead uses
    # `make_settings` (helpers.py) plus a respx-mocked LedgerClient. Adapted
    # to that existing fixture pattern per the brief's fallback note.
    from app.config import PlacementOverride

    respx.get(f"{BASE}/inference-events/unverified/view").mock(
        return_value=httpx.Response(200, json=_page([]))
    )
    ledger = LedgerClient(BASE)
    kube = FakeKube()
    settings = make_settings(
        model_cache_pvc="hf-model-cache",
        vllm_placement_overrides={
            "cached/model": PlacementOverride(cache_pvc="pvc-x"),
            "uncached/model": PlacementOverride(),
        },
    )

    captured: dict = {}

    def fake_decide(pending, jobs, vllms, **kwargs):
        captured.update(kwargs)
        return []

    with unittest.mock.patch("app.main.decide", side_effect=fake_decide):
        reconcile_once(ledger, kube, settings)

    assert captured["cache_exempt_models"] == frozenset({"uncached/model"})
