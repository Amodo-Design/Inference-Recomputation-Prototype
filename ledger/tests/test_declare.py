"""Declaration invariants (/model-deployments/declare): hardware dedup, config
dedup, multi-host sharing, handover, and the one-active-per-host rule."""

from __future__ import annotations

import uuid

import pytest

T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-01T01:00:00Z"


async def _declare(client, hostname, name, started_at=None, **params):
    body = {"hostname": hostname, "model_name": name, **params}
    if started_at is not None:
        body["started_at"] = started_at
    r = await client.post("/model-deployments/declare", json=body)
    return r


async def _list(client, path):
    r = await client.get(path)
    assert r.status_code == 200, r.text
    return r.json()


async def test_hardware_deduped_by_hostname(client):
    assert (await _declare(client, "h", "model-a", T0, seed=1)).status_code == 201
    assert (await _declare(client, "h", "model-b", T1, seed=2)).status_code == 201
    hardware = await _list(client, "/hardware")
    assert len(hardware) == 1


async def test_model_deduped_by_config(client):
    # Identical config on two hosts -> one model row.
    await _declare(client, "host-a", "m", T0, seed=5)
    await _declare(client, "host-b", "m", T0, seed=5)
    assert len(await _list(client, "/models")) == 1
    # A changed param -> a second model row.
    await _declare(client, "host-c", "m", T0, seed=6)
    assert len(await _list(client, "/models")) == 2


async def test_same_model_many_hosts(client):
    await _declare(client, "host-a", "m", T0, seed=1)
    await _declare(client, "host-b", "m", T0, seed=1)
    assert len(await _list(client, "/models")) == 1
    assert len(await _list(client, "/hardware")) == 2
    assert len(await _list(client, "/model-deployments")) == 2


async def test_handover_leaves_one_active_per_host(client):
    a = (await _declare(client, "h", "model-a", T0, seed=1)).json()
    b = (await _declare(client, "h", "model-b", T1, seed=2)).json()

    deployments = await _list(client, "/model-deployments")
    assert len(deployments) == 2
    active = [d for d in deployments if d["ended_at"] is None]
    assert len(active) == 1
    assert active[0]["deployment_id"] == b["deployment_id"]
    # The prior deployment was closed.
    prior = next(d for d in deployments if d["deployment_id"] == a["deployment_id"])
    assert prior["ended_at"] is not None


async def test_started_at_defaulted_when_omitted(client):
    r = await _declare(client, "h", "m", started_at=None, seed=1)
    assert r.status_code == 201
    body = r.json()
    assert body["started_at"] is not None
    assert body["ended_at"] is None


async def test_hardware_metadata_not_updated_on_redeclare(client):
    # Documents current behaviour: hardware is matched by hostname and reused
    # as-is; metadata from a later declare does NOT overwrite the original.
    await _declare(client, "h", "model-a", T0, seed=1, gpu_product_id="gpu-1")
    await _declare(client, "h", "model-b", T1, seed=2, gpu_product_id="gpu-2")
    hardware = await _list(client, "/hardware")
    assert len(hardware) == 1
    assert hardware[0]["gpu_product_id"] == "gpu-1"


async def test_declare_with_unknown_owner_id_conflicts(client):
    # owner_id must reference an existing hardware_owner; a bogus one fails the
    # FK when the hardware row is created.
    r = await _declare(
        client, "h", "m", T0, seed=1, owner_id=str(uuid.uuid4())
    )
    assert r.status_code == 409, r.text
