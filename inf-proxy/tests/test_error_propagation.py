"""Upstream errors must be relayed, never converted into empty SSE streams.

Regression test for the bug where a streaming request rejected by the model
(e.g. vLLM's 400 for unsupported `tools`) was returned as an HTTP 200
`text/event-stream` containing only `data: [DONE]` — masking the error from
the client (Open WebUI rendered an empty chat response) and from the ledger.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import main as app_main


class FakeUpstreamResponse:
    def __init__(self, status: int, body: dict | str):
        self.status = status
        self._body = body
        self.released = False

    async def json(self):
        if isinstance(self._body, dict):
            return self._body
        raise ValueError("not json")

    async def text(self):
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    def release(self):
        self.released = True


class FakeSession:
    def __init__(self, resp: FakeUpstreamResponse):
        self._resp = resp
        self.last_payload = None

    async def post(self, url, data=None, headers=None, timeout=None):
        self.last_payload = json.loads(data)
        return self._resp


@pytest.fixture
def client():
    return TestClient(app_main.app)


def _fake_get_session(resp):
    async def get_session():
        return FakeSession(resp)

    return get_session


def test_streaming_request_relays_upstream_error_status_and_body(client, monkeypatch):
    error_body = {
        "error": {
            "message": '"auto" tool choice requires --enable-auto-tool-choice',
            "type": "BadRequestError",
            "code": 400,
        }
    }
    upstream = FakeUpstreamResponse(400, error_body)
    monkeypatch.setattr(app_main, "get_session", _fake_get_session(upstream))

    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert r.status_code == 400
    assert r.json() == error_body
    assert "text/event-stream" not in r.headers.get("content-type", "")
    assert upstream.released


def test_streaming_request_relays_non_json_error_body(client, monkeypatch):
    upstream = FakeUpstreamResponse(502, "upstream exploded")
    monkeypatch.setattr(app_main, "get_session", _fake_get_session(upstream))

    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert r.status_code == 502
    assert "upstream exploded" in r.json()["error"]["message"]
    assert upstream.released
