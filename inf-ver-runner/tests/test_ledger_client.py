"""LedgerClient request/response behaviour against a mocked ledger API."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
import respx

from app.ledger_client import LedgerClient

from helpers import make_settings

BASE = "http://ledger-api:8000"


@pytest.fixture
async def client():
    c = LedgerClient(make_settings())
    yield c
    await c.aclose()


@respx.mock
async def test_declare(client):
    deployment = {
        "deployment_id": str(uuid.uuid4()),
        "model_id": str(uuid.uuid4()),
        "hardware_id": str(uuid.uuid4()),
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": None,
    }
    route = respx.post(f"{BASE}/model-deployments/declare").mock(
        return_value=httpx.Response(201, json=deployment)
    )

    result = await client.declare("inf-ver-runner", "openai/gpt-oss-20b")

    assert result == deployment
    body = json.loads(route.calls.last.request.content)
    assert body == {
        "hostname": "inf-ver-runner",
        "model_name": "openai/gpt-oss-20b",
        "decoding_algorithm": "gumbel_max",
        "owner_name": "verifier",
    }


@respx.mock
async def test_declare_raises_on_error(client):
    respx.post(f"{BASE}/model-deployments/declare").mock(
        return_value=httpx.Response(409)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.declare("inf-ver-runner", "openai/gpt-oss-20b")


@respx.mock
async def test_fetch_unverified(client):
    route = respx.get(f"{BASE}/inference-events/unverified").mock(
        return_value=httpx.Response(200, json=[{"event": {}, "model": {}}])
    )
    items = await client.fetch_unverified(limit=5)
    assert items == [{"event": {}, "model": {}}]
    params = route.calls.last.request.url.params
    assert params["limit"] == "5"
    assert "model_id" not in params


@respx.mock
async def test_fetch_unverified_scoped_to_model(client):
    model_id = str(uuid.uuid4())
    route = respx.get(f"{BASE}/inference-events/unverified").mock(
        return_value=httpx.Response(200, json=[])
    )
    await client.fetch_unverified(limit=5, model_id=model_id)
    assert route.calls.last.request.url.params["model_id"] == model_id


@respx.mock
async def test_get_model(client):
    model_id = str(uuid.uuid4())
    row = {"model_id": model_id, "verification_threshold": 0.1}
    respx.get(f"{BASE}/models/{model_id}").mock(
        return_value=httpx.Response(200, json=row)
    )
    assert await client.get_model(model_id) == row


@respx.mock
async def test_get_model_raises_on_missing(client):
    model_id = str(uuid.uuid4())
    respx.get(f"{BASE}/models/{model_id}").mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_model(model_id)


@respx.mock
async def test_create_verification_event_raises_on_error(client):
    respx.post(f"{BASE}/verification-events").mock(return_value=httpx.Response(409))
    with pytest.raises(httpx.HTTPStatusError):
        await client.create_verification_event({"id": "x"})


@respx.mock
async def test_health(client):
    respx.get(f"{BASE}/health").mock(return_value=httpx.Response(200, json={}))
    assert await client.health() is True
    respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("down"))
    assert await client.health() is False


@respx.mock
async def test_close_deployment(client):
    route = respx.post(f"{BASE}/model-deployments/close").mock(
        return_value=httpx.Response(200, json={"deployment_id": str(uuid.uuid4())})
    )
    assert await client.close_deployment("inf-ver-runner-x") is True
    assert json.loads(route.calls.last.request.content) == {
        "hostname": "inf-ver-runner-x"
    }


@respx.mock
async def test_close_deployment_tolerates_404_and_errors(client):
    respx.post(f"{BASE}/model-deployments/close").mock(
        return_value=httpx.Response(404)
    )
    assert await client.close_deployment("inf-ver-runner-x") is False

    respx.post(f"{BASE}/model-deployments/close").mock(
        side_effect=httpx.ConnectError("down")
    )
    assert await client.close_deployment("inf-ver-runner-x") is False
