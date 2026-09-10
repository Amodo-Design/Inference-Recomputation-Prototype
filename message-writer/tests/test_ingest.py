"""Endpoint tests for POST /inferences (app/main.py).

The ledger boundary is monkeypatched, so these exercise the orchestration
branches with no network or DB.
"""

from __future__ import annotations

import httpx
import pytest

from app import main as mw

FULL_MSG = {
    "tap": {"hostname": "dummy-model", "completed_at": "2026-01-01T00:00:00Z"},
    "session": {"session_id": "chat-1"},
    "model": {"name": "m"},
    "request": {"prompt_text": "hi", "prompt_token_ids": [1, 2, 3]},
    "response": {"output_text": "yo", "output_token_ids": [40, 41]},
}

RESOLUTION = {
    "deployment_id": "DID",
    "model_id": "MID",
    "hardware_id": "HID",
    "model_name": "m",
    "model_name_matches": True,
}


def _patch_ledger(monkeypatch, *, resolution=RESOLUTION, resolve_exc=None, create_exc=None):
    created = []

    async def fake_resolve(hostname, ts, model_name):
        if resolve_exc:
            raise resolve_exc
        return resolution

    async def fake_create(payload):
        if create_exc:
            raise create_exc
        created.append(payload)
        return {"id": payload["id"]}

    monkeypatch.setattr(mw, "resolve", fake_resolve)
    monkeypatch.setattr(mw, "create_inference_event", fake_create)
    return created


async def test_happy_path_writes_and_returns_ids(client, monkeypatch):
    created = _patch_ledger(monkeypatch)
    r = await client.post("/inferences", json=FULL_MSG)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["model_id"] == "MID"
    assert body["hardware_id"] == "HID"
    assert body["model_name_matches"] is True
    # The event written to the ledger carried the resolved ids.
    assert len(created) == 1
    assert created[0]["model_id"] == "MID"
    assert created[0]["hardware_id"] == "HID"
    assert created[0]["id"] == body["inference_event_id"]


async def test_missing_hostname_422(client, monkeypatch):
    _patch_ledger(monkeypatch)
    msg = {**FULL_MSG, "tap": {"completed_at": "2026-01-01T00:00:00Z"}}
    r = await client.post("/inferences", json=msg)
    assert r.status_code == 422


async def test_missing_timestamp_422(client, monkeypatch):
    _patch_ledger(monkeypatch)
    msg = {**FULL_MSG, "tap": {"hostname": "dummy-model"}}
    r = await client.post("/inferences", json=msg)
    assert r.status_code == 422


async def test_no_active_deployment_422(client, monkeypatch):
    _patch_ledger(monkeypatch, resolution=None)
    r = await client.post("/inferences", json=FULL_MSG)
    assert r.status_code == 422
    assert "no model deployment" in r.json()["detail"]


async def test_resolve_failure_502(client, monkeypatch):
    _patch_ledger(monkeypatch, resolve_exc=httpx.ConnectError("boom"))
    r = await client.post("/inferences", json=FULL_MSG)
    assert r.status_code == 502


async def test_ledger_write_failure_502(client, monkeypatch):
    _patch_ledger(monkeypatch, create_exc=httpx.ConnectError("boom"))
    r = await client.post("/inferences", json=FULL_MSG)
    assert r.status_code == 502


async def test_name_mismatch_still_writes(client, monkeypatch):
    resolution = {**RESOLUTION, "model_name_matches": False}
    created = _patch_ledger(monkeypatch, resolution=resolution)
    r = await client.post("/inferences", json=FULL_MSG)
    assert r.status_code == 201
    assert r.json()["model_name_matches"] is False
    assert len(created) == 1  # a mismatch is a flag, not a block
