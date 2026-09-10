"""Per-model verification threshold: a mutable, nullable field on the model
row — deliberately NOT part of the model's identity hash (the threshold is an
operator decision about strictness, not part of what the model *is*)."""

from __future__ import annotations

import uuid

T0 = "2026-01-01T00:00:00Z"


async def _declare(client, hostname, name, **params):
    r = await client.post(
        "/model-deployments/declare",
        json={"hostname": hostname, "model_name": name, "started_at": T0, **params},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _get_model(client, model_id):
    r = await client.get(f"/models/{model_id}")
    assert r.status_code == 200, r.text
    return r.json()


async def test_threshold_defaults_to_null(client):
    dep = await _declare(client, "h1", "m", seed=42)
    model = await _get_model(client, dep["model_id"])
    assert model["verification_threshold"] is None


async def test_patch_sets_changes_and_clears_threshold(client):
    dep = await _declare(client, "h1", "m", seed=42)
    model_id = dep["model_id"]

    r = await client.patch(
        f"/models/{model_id}/verification-threshold",
        json={"verification_threshold": 0.1},
    )
    assert r.status_code == 200, r.text
    assert r.json()["verification_threshold"] == 0.1

    r = await client.patch(
        f"/models/{model_id}/verification-threshold",
        json={"verification_threshold": 0.05},
    )
    assert r.status_code == 200, r.text
    assert (await _get_model(client, model_id))["verification_threshold"] == 0.05

    r = await client.patch(
        f"/models/{model_id}/verification-threshold",
        json={"verification_threshold": None},
    )
    assert r.status_code == 200, r.text
    assert (await _get_model(client, model_id))["verification_threshold"] is None


async def test_patch_unknown_model_404s(client):
    r = await client.patch(
        f"/models/{uuid.uuid4()}/verification-threshold",
        json={"verification_threshold": 0.1},
    )
    assert r.status_code == 404


async def test_threshold_is_not_part_of_model_identity(client):
    # Declare, set a threshold, re-declare the identical config: must reuse
    # the SAME model row (same id), keeping the threshold that was set.
    dep1 = await _declare(client, "h1", "m", seed=42)
    await client.patch(
        f"/models/{dep1['model_id']}/verification-threshold",
        json={"verification_threshold": 0.2},
    )
    dep2 = await _declare(client, "h2", "m", seed=42)
    assert dep2["model_id"] == dep1["model_id"]

    models = (await client.get("/models")).json()
    assert len(models) == 1
    assert models[0]["verification_threshold"] == 0.2


async def test_delta_max_defaults_to_null(client):
    dep = await _declare(client, "h1", "m", seed=42)
    model = await _get_model(client, dep["model_id"])
    assert model["delta_max"] is None


async def test_patch_sets_changes_and_clears_delta_max(client):
    dep = await _declare(client, "h1", "m", seed=42)
    model_id = dep["model_id"]

    r = await client.patch(f"/models/{model_id}/delta-max", json={"delta_max": 10.0})
    assert r.status_code == 200, r.text
    assert r.json()["delta_max"] == 10.0

    r = await client.patch(f"/models/{model_id}/delta-max", json={"delta_max": 5.0})
    assert r.status_code == 200, r.text
    assert (await _get_model(client, model_id))["delta_max"] == 5.0

    r = await client.patch(f"/models/{model_id}/delta-max", json={"delta_max": None})
    assert r.status_code == 200, r.text
    assert (await _get_model(client, model_id))["delta_max"] is None


async def test_patch_delta_max_unknown_model_404s(client):
    r = await client.patch(
        f"/models/{uuid.uuid4()}/delta-max", json={"delta_max": 10.0}
    )
    assert r.status_code == 404


async def test_delta_max_is_not_part_of_model_identity(client):
    dep1 = await _declare(client, "h1", "m", seed=42)
    await client.patch(f"/models/{dep1['model_id']}/delta-max", json={"delta_max": 7.5})
    dep2 = await _declare(client, "h2", "m", seed=42)
    assert dep2["model_id"] == dep1["model_id"]
    models = (await client.get("/models")).json()
    assert len(models) == 1
    assert models[0]["delta_max"] == 7.5


async def test_threshold_and_delta_max_are_independent(client):
    dep = await _declare(client, "h1", "m", seed=42)
    model_id = dep["model_id"]
    await client.patch(
        f"/models/{model_id}/verification-threshold",
        json={"verification_threshold": 0.3},
    )
    await client.patch(f"/models/{model_id}/delta-max", json={"delta_max": 5.0})
    model = await _get_model(client, model_id)
    assert model["verification_threshold"] == 0.3
    assert model["delta_max"] == 5.0

    # Clearing one leaves the other alone.
    await client.patch(f"/models/{model_id}/delta-max", json={"delta_max": None})
    model = await _get_model(client, model_id)
    assert model["verification_threshold"] == 0.3
    assert model["delta_max"] is None
