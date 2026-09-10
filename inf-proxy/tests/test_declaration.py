"""Tests for the tap's startup self-declaration to the ledger."""

from __future__ import annotations

import asyncio

import pytest

from app import declaration


# ---------------------------------------------------------------------------
# build_declaration
# ---------------------------------------------------------------------------
def _clear_sampling_config(monkeypatch, **overrides):
    for name in (
        "INF_PROXY_SEED",
        "INF_PROXY_TEMPERATURE",
        "INF_PROXY_TOP_K",
        "INF_PROXY_TOP_P",
    ):
        monkeypatch.setattr(declaration, name, overrides.get(name))


def test_build_declaration_includes_hostname_model_and_full_sampling_config(monkeypatch):
    monkeypatch.setattr(declaration, "MODEL_HOSTNAME", "kserve-qwen")
    _clear_sampling_config(
        monkeypatch,
        INF_PROXY_SEED=42,
        INF_PROXY_TEMPERATURE=1.0,
        INF_PROXY_TOP_K=50,
        INF_PROXY_TOP_P=0.95,
    )
    monkeypatch.setattr(declaration, "INF_PROXY_DECODING_ALGORITHM", None)

    payload = declaration.build_declaration("Qwen/Qwen2.5-7B-Instruct")

    assert payload["hostname"] == "kserve-qwen"
    assert payload["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["seed"] == 42
    assert payload["temperature"] == 1.0
    assert payload["top_k"] == 50
    assert payload["top_p"] == 0.95
    assert "decoding_algorithm" not in payload


def test_build_declaration_omits_unconfigured_sampling_fields(monkeypatch):
    monkeypatch.setattr(declaration, "MODEL_HOSTNAME", "h")
    # Optional fields the tap has no knowledge of must be absent, not null-y
    # guesses: model_id is derived from the config server-side.
    _clear_sampling_config(monkeypatch)
    monkeypatch.setattr(declaration, "INF_PROXY_DECODING_ALGORITHM", None)

    payload = declaration.build_declaration("m")

    assert "seed" not in payload
    assert "temperature" not in payload
    assert "top_k" not in payload
    assert "top_p" not in payload


def test_build_declaration_includes_decoding_algorithm_when_set(monkeypatch):
    monkeypatch.setattr(declaration, "MODEL_HOSTNAME", "h")
    monkeypatch.setattr(declaration, "INF_PROXY_SEED", None)
    monkeypatch.setattr(declaration, "INF_PROXY_DECODING_ALGORITHM", "gumbel_max")

    payload = declaration.build_declaration("m")

    assert payload["decoding_algorithm"] == "gumbel_max"


# ---------------------------------------------------------------------------
# parse_model_name
# ---------------------------------------------------------------------------
def test_parse_model_name_takes_first_model_id():
    body = {"object": "list", "data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}, {"id": "other"}]}
    assert declaration.parse_model_name(body) == "Qwen/Qwen2.5-7B-Instruct"


@pytest.mark.parametrize("body", [{}, {"data": []}, {"data": [{}]}, None])
def test_parse_model_name_handles_malformed_bodies(body):
    assert declaration.parse_model_name(body) is None


# ---------------------------------------------------------------------------
# declare_until_success — retries then stops after first success
# ---------------------------------------------------------------------------
def test_declare_until_success_retries_then_succeeds(monkeypatch):
    attempts = []

    async def fake_attempt():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("ledger not up yet")
        return {"deployment_id": "d", "model_id": "m", "hardware_id": "h"}

    monkeypatch.setattr(declaration, "_attempt_declaration", fake_attempt)
    monkeypatch.setattr(declaration, "INF_PROXY_DECLARE_RETRY_SECONDS", 0)

    result = asyncio.run(
        declaration.declare_until_success()
    )

    assert len(attempts) == 3
    assert result["deployment_id"] == "d"


def test_declare_until_success_gives_up_after_max_attempts(monkeypatch):
    async def fake_attempt():
        raise RuntimeError("never up")

    monkeypatch.setattr(declaration, "_attempt_declaration", fake_attempt)
    monkeypatch.setattr(declaration, "INF_PROXY_DECLARE_RETRY_SECONDS", 0)

    result = asyncio.run(
        declaration.declare_until_success(max_attempts=4)
    )

    assert result is None


# ---------------------------------------------------------------------------
# owner attribution + close_deployment
# ---------------------------------------------------------------------------
def test_build_declaration_reports_prover_owner(monkeypatch):
    monkeypatch.setattr(declaration, "MODEL_HOSTNAME", "h")
    _clear_sampling_config(monkeypatch)
    monkeypatch.setattr(declaration, "INF_PROXY_DECODING_ALGORITHM", None)
    monkeypatch.setattr(declaration, "INF_PROXY_OWNER_NAME", "prover")

    assert declaration.build_declaration("m")["owner_name"] == "prover"


def test_build_declaration_omits_owner_when_disabled(monkeypatch):
    monkeypatch.setattr(declaration, "MODEL_HOSTNAME", "h")
    _clear_sampling_config(monkeypatch)
    monkeypatch.setattr(declaration, "INF_PROXY_DECODING_ALGORITHM", None)
    monkeypatch.setattr(declaration, "INF_PROXY_OWNER_NAME", None)

    assert "owner_name" not in declaration.build_declaration("m")


def test_close_deployment_noop_when_declaration_disabled(monkeypatch):
    monkeypatch.setattr(declaration, "INF_PROXY_DECLARE", False)
    assert asyncio.run(declaration.close_deployment()) is False


def test_close_deployment_posts_hostname_and_survives_errors(monkeypatch):
    # The close call happens during pod teardown: any failure must be
    # swallowed. With no ledger listening the POST raises internally and
    # close_deployment reports False instead of raising.
    monkeypatch.setattr(declaration, "INF_PROXY_DECLARE", True)
    monkeypatch.setattr(declaration, "MODEL_HOSTNAME", "h")
    monkeypatch.setattr(
        declaration, "INF_PROXY_LEDGER_API_URL", "http://127.0.0.1:1"
    )
    assert asyncio.run(declaration.close_deployment()) is False


def test_close_deployment_posts_to_close_endpoint(monkeypatch):
    calls = {}

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeSession:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, json=None):
            calls["url"] = url
            calls["json"] = json
            return FakeResponse()

    monkeypatch.setattr(declaration, "INF_PROXY_DECLARE", True)
    monkeypatch.setattr(declaration, "MODEL_HOSTNAME", "kserve-qwen")
    monkeypatch.setattr(declaration, "INF_PROXY_LEDGER_API_URL", "http://ledger")
    monkeypatch.setattr(declaration.aiohttp, "ClientSession", FakeSession)

    assert asyncio.run(declaration.close_deployment()) is True
    assert calls["url"] == "http://ledger/model-deployments/close"
    assert calls["json"] == {"hostname": "kserve-qwen"}
