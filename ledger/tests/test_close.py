"""Deployment close reporting + owner attribution on declare.

Taps close their prover's deployment on pod shutdown; runners close their
own on exit. Owners arrive by name ("prover" / "verifier") and are upserted.
"""

from __future__ import annotations

T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-01T01:00:00Z"


async def _declare(client, hostname, name="m", **params):
    r = await client.post(
        "/model-deployments/declare",
        json={"hostname": hostname, "model_name": name, "started_at": T0, **params},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def test_close_sets_ended_at_on_active_deployment(client):
    dep = await _declare(client, "h1", seed=42)
    r = await client.post(
        "/model-deployments/close", json={"hostname": "h1", "ended_at": T1}
    )
    assert r.status_code == 200, r.text
    closed = r.json()
    assert closed["deployment_id"] == dep["deployment_id"]
    assert closed["ended_at"] is not None

    # The deployment is no longer active: closing again 404s.
    r = await client.post("/model-deployments/close", json={"hostname": "h1"})
    assert r.status_code == 404


async def test_close_defaults_ended_at_to_now(client):
    await _declare(client, "h1", seed=42)
    r = await client.post("/model-deployments/close", json={"hostname": "h1"})
    assert r.status_code == 200, r.text
    assert r.json()["ended_at"] is not None


async def test_close_unknown_hostname_404s(client):
    r = await client.post("/model-deployments/close", json={"hostname": "nope"})
    assert r.status_code == 404


async def test_close_then_redeclare_opens_fresh_deployment(client):
    dep1 = await _declare(client, "h1", seed=42)
    await client.post("/model-deployments/close", json={"hostname": "h1"})
    dep2 = await _declare(client, "h1", seed=42)
    assert dep2["deployment_id"] != dep1["deployment_id"]
    assert dep2["ended_at"] is None
    # Same config, same model row.
    assert dep2["model_id"] == dep1["model_id"]


async def test_declare_with_owner_name_creates_and_links_owner(client):
    dep = await _declare(client, "tap-host", seed=42, owner_name="prover")

    owners = (await client.get("/hardware-owners")).json()
    assert [o["organisation_name"] for o in owners] == ["prover"]
    assert owners[0]["is_trusted"] is False

    hw = (await client.get(f"/hardware/{dep['hardware_id']}")).json()
    assert hw["owner_id"] == owners[0]["owner_id"]


async def test_owner_name_is_reused_not_duplicated(client):
    await _declare(client, "tap-1", seed=42, owner_name="prover")
    await _declare(client, "tap-2", seed=42, owner_name="prover")
    await _declare(client, "runner-1", name="m", owner_name="verifier")

    owners = (await client.get("/hardware-owners")).json()
    names = sorted(o["organisation_name"] for o in owners)
    assert names == ["prover", "verifier"]


async def test_owner_name_backfills_existing_hardware(client):
    # First declaration without an owner, later one with: the hardware row
    # gains the owner link.
    dep = await _declare(client, "h1", seed=42)
    hw = (await client.get(f"/hardware/{dep['hardware_id']}")).json()
    assert hw["owner_id"] is None

    await _declare(client, "h1", seed=42, owner_name="prover")
    hw = (await client.get(f"/hardware/{dep['hardware_id']}")).json()
    assert hw["owner_id"] is not None
