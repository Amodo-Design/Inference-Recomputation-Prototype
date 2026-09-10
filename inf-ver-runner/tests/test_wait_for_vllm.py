"""Startup gate: no events are consumed until the paired vLLM is ready."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.readiness import wait_for_vllm

from helpers import make_settings

MODELS_URL = "http://vllm:8000/v1/models"


@respx.mock
async def test_waits_through_cold_start_then_ready():
    route = respx.get(MODELS_URL)
    route.side_effect = [
        httpx.ConnectError("still booting"),
        httpx.Response(503),
        httpx.Response(200, json={"data": []}),
    ]
    settings = make_settings(poll_interval_seconds=0.001)

    assert await wait_for_vllm(settings) is True
    assert len(route.calls) == 3


@respx.mock
async def test_gives_up_after_timeout():
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("never up"))
    settings = make_settings(
        poll_interval_seconds=0.001, vllm_ready_timeout_seconds=0.05
    )

    assert await wait_for_vllm(settings) is False


@respx.mock
async def test_sends_api_key_when_configured():
    route = respx.get(MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    settings = make_settings(vllm_api_key="sekret")

    assert await wait_for_vllm(settings) is True
    assert route.calls.last.request.headers["Authorization"] == "Bearer sekret"
