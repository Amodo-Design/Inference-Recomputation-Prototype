"""CRUD-factory behaviour (app/routers/__init__.py) and the transactional
boundary: duplicate-PK / FK conflicts return 409 and leave nothing persisted."""

from __future__ import annotations

import base64
import os
import uuid

import pytest

MODEL_ID = "11111111-1111-1111-1111-111111111111"


def _b64_32() -> str:
    return base64.b64encode(os.urandom(32)).decode()


async def test_duplicate_primary_key_returns_409(client):
    body = {"model_id": MODEL_ID, "model_name": "m"}
    assert (await client.post("/models", json=body)).status_code == 201
    dup = await client.post("/models", json=body)
    assert dup.status_code == 409, dup.text


async def test_fk_violation_returns_409(client):
    # No model/hardware exist -> the FKs fail.
    event = {
        "id": str(uuid.uuid4()),
        "session_id": "s",
        "ts": "2026-01-01T00:00:00Z",
        "model_id": str(uuid.uuid4()),
        "hardware_id": str(uuid.uuid4()),
        "hash_input_raw_logits": _b64_32(),
        "hash_output_raw_logits": _b64_32(),
    }
    r = await client.post("/inference-events", json=event)
    assert r.status_code == 409, r.text


async def test_failed_write_persists_nothing(client):
    # A 409 must roll back cleanly — no partial inference_event row survives.
    event = {
        "id": str(uuid.uuid4()),
        "session_id": "s",
        "ts": "2026-01-01T00:00:00Z",
        "model_id": str(uuid.uuid4()),
        "hardware_id": str(uuid.uuid4()),
        "hash_input_raw_logits": _b64_32(),
        "hash_output_raw_logits": _b64_32(),
    }
    assert (await client.post("/inference-events", json=event)).status_code == 409
    listing = await client.get("/inference-events")
    assert listing.status_code == 200
    assert listing.json() == []


async def test_list_pagination(client):
    for i in range(3):
        r = await client.post(
            "/hardware-owners",
            json={"owner_id": str(uuid.uuid4()), "organisation_name": f"org-{i}"},
        )
        assert r.status_code == 201

    page1 = await client.get("/hardware-owners", params={"limit": 2, "offset": 0})
    page2 = await client.get("/hardware-owners", params={"limit": 2, "offset": 2})
    assert len(page1.json()) == 2
    assert len(page2.json()) == 1


async def test_get_missing_returns_404(client):
    r = await client.get(f"/models/{uuid.uuid4()}")
    assert r.status_code == 404
