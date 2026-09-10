"""Resolution edge cases (/model-deployments/resolve).

The resolve query uses a half-open interval [started_at, ended_at): started is
inclusive, ended is exclusive. These tests pin that contract and the 404 /
mismatch branches.
"""

from __future__ import annotations

import pytest

T_BEFORE = "2025-12-31T23:00:00Z"
T0 = "2026-01-01T00:00:00Z"
T_MID = "2026-01-01T00:30:00Z"
T1 = "2026-01-01T01:00:00Z"
T_LATE = "2026-01-01T02:00:00Z"


async def _declare(client, hostname, name, started_at, **params):
    body = {"hostname": hostname, "model_name": name, "started_at": started_at, **params}
    r = await client.post("/model-deployments/declare", json=body)
    assert r.status_code == 201, r.text
    return r.json()


async def _resolve(client, hostname, ts, model_name=None):
    params = {"hostname": hostname, "ts": ts}
    if model_name is not None:
        params["model_name"] = model_name
    return await client.get("/model-deployments/resolve", params=params)


async def test_resolve_within_open_window(client):
    dep = await _declare(client, "h", "m", T0)
    r = await _resolve(client, "h", T_MID)
    assert r.status_code == 200, r.text
    assert r.json()["model_id"] == dep["model_id"]


async def test_resolve_at_started_at_is_inclusive(client):
    dep = await _declare(client, "h", "m", T0)
    r = await _resolve(client, "h", T0)
    assert r.status_code == 200
    assert r.json()["model_id"] == dep["model_id"]


async def test_resolve_before_any_deployment_404(client):
    await _declare(client, "h", "m", T0)
    r = await _resolve(client, "h", T_BEFORE)
    assert r.status_code == 404


async def test_resolve_unknown_hostname_404(client):
    await _declare(client, "h", "m", T0)
    r = await _resolve(client, "ghost", T_MID)
    assert r.status_code == 404


async def test_resolve_handover_interval_boundaries(client):
    a = await _declare(client, "h", "model-a", T0, seed=1)
    b = await _declare(client, "h", "model-b", T1, seed=2)
    assert a["model_id"] != b["model_id"]

    # Before the boundary -> old model.
    r = await _resolve(client, "h", T_MID)
    assert r.json()["model_id"] == a["model_id"]
    # Exactly at the boundary (== prior.ended_at == new.started_at) -> new model
    # (ended_at is exclusive, started_at inclusive).
    r = await _resolve(client, "h", T1)
    assert r.json()["model_id"] == b["model_id"]
    # After -> new model.
    r = await _resolve(client, "h", T_LATE)
    assert r.json()["model_id"] == b["model_id"]


async def test_resolve_multi_host_disambiguation(client):
    a = await _declare(client, "host-a", "m", T0)
    b = await _declare(client, "host-b", "m", T0)
    assert a["model_id"] == b["model_id"]  # same config -> shared model

    ra = await _resolve(client, "host-a", T_MID)
    rb = await _resolve(client, "host-b", T_MID)
    assert ra.json()["hardware_id"] == a["hardware_id"]
    assert rb.json()["hardware_id"] == b["hardware_id"]
    assert ra.json()["hardware_id"] != rb.json()["hardware_id"]


async def test_resolve_model_name_matches_flag(client):
    await _declare(client, "h", "m1", T0)

    assert (await _resolve(client, "h", T_MID, "m1")).json()["model_name_matches"] is True
    assert (await _resolve(client, "h", T_MID, "other")).json()["model_name_matches"] is False
    # Omitted -> treated as a match (no cross-check requested).
    assert (await _resolve(client, "h", T_MID)).json()["model_name_matches"] is True
