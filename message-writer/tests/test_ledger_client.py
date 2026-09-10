"""Tests for the ledger HTTP boundary (app/ledger_client.py).

Uses an httpx MockTransport so no ledger is needed: verifies 404->None,
other non-2xx -> raise, and 2xx -> parsed json.
"""

from __future__ import annotations

import httpx
import pytest

from app import ledger_client


def _install(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ledger")
    monkeypatch.setattr(ledger_client, "_client", client)


async def test_resolve_returns_json_on_200(monkeypatch):
    def handler(request):
        assert request.url.path == "/model-deployments/resolve"
        assert request.url.params["hostname"] == "h"
        assert request.url.params["model_name"] == "m"
        return httpx.Response(200, json={"model_id": "MID", "hardware_id": "HID"})

    _install(monkeypatch, handler)
    result = await ledger_client.resolve("h", "2026-01-01T00:00:00Z", "m")
    assert result == {"model_id": "MID", "hardware_id": "HID"}


async def test_resolve_returns_none_on_404(monkeypatch):
    _install(monkeypatch, lambda req: httpx.Response(404, json={"detail": "nope"}))
    assert await ledger_client.resolve("h", "2026-01-01T00:00:00Z", None) is None


async def test_resolve_omits_model_name_when_absent(monkeypatch):
    def handler(request):
        assert "model_name" not in request.url.params
        return httpx.Response(200, json={})

    _install(monkeypatch, handler)
    await ledger_client.resolve("h", "2026-01-01T00:00:00Z", None)


async def test_resolve_raises_on_500(monkeypatch):
    _install(monkeypatch, lambda req: httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        await ledger_client.resolve("h", "2026-01-01T00:00:00Z", None)


async def test_create_inference_event_returns_json_on_201(monkeypatch):
    def handler(request):
        assert request.url.path == "/inference-events"
        return httpx.Response(201, json={"id": "abc"})

    _install(monkeypatch, handler)
    assert await ledger_client.create_inference_event({"id": "abc"}) == {"id": "abc"}


async def test_create_inference_event_raises_on_409(monkeypatch):
    _install(monkeypatch, lambda req: httpx.Response(409, json={"detail": "dup"}))
    with pytest.raises(httpx.HTTPStatusError):
        await ledger_client.create_inference_event({"id": "abc"})
