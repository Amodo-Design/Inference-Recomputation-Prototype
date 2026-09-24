"""Content-based LLM classification."""

from __future__ import annotations

import json

from app.filter import classify, is_verifier_traffic, unexpected_reason
from app.http_stream import parse_exchanges
from tests.helpers import http_request, http_response_json


def _exchange(
    path: str,
    body: dict | None,
    method_override: bytes | None = None,
    host: str | None = None,
):
    raw_body = json.dumps(body).encode() if body is not None else b"not json"
    raw = http_request(path, raw_body)
    if host:
        raw = raw.replace(b"host: 10.0.0.2:8080", f"host: {host}".encode(), 1)
    if method_override:
        raw = raw.replace(b"POST", method_override, 1)
    return parse_exchanges(raw, http_response_json(b"{}"))[0]


def test_chat_completions_behind_gateway_prefix_classifies():
    ex = _exchange("/infver/gpt-oss-120b/v1/chat/completions", {"model": "openai/gpt-oss-120b"})
    c = classify(ex)
    assert c is not None
    assert c.endpoint == "/v1/chat/completions"
    assert c.model_name == "openai/gpt-oss-120b"


def test_legacy_completions_classifies():
    ex = _exchange("/v1/completions", {"model": "m", "prompt": "hi"})
    assert classify(ex).endpoint == "/v1/completions"


def test_non_llm_paths_rejected():
    assert classify(_exchange("/metrics", {"model": "m"})) is None
    assert classify(_exchange("/v1/models", {"model": "m"})) is None
    assert classify(_exchange("/api/v1/chats", {"model": "m"})) is None


def test_verifier_replay_rejected():
    """The verifier's own replays are not inference for this tap to find."""
    ex = _exchange(
        "/v1/completions",
        {"model": "openai/gpt-oss-120b", "prompt": [1, 2, 3], "max_tokens": 1},
        host="verify-openai-gpt-oss-120b-kserve-workload-svc:8000",
    )
    assert classify(ex) is None


def test_production_hostnames_still_classify():
    """Exclusion must not swallow the serving path it exists to watch."""
    for host in (
        "gpt-oss-120b-kserve-workload-svc:8000",
        "10.244.2.20:8000",
        "llm-gateway-istio.infver.svc.cluster.local",
    ):
        ex = _exchange("/v1/chat/completions", {"model": "m"}, host=host)
        assert classify(ex) is not None, host


def test_is_verifier_traffic_matching():
    assert is_verifier_traffic("verify-qwen2-5-7b-instruct-kserve-workload-svc:8000")
    assert is_verifier_traffic("VERIFY-Model.infver.svc.cluster.local")  # case-insensitive
    assert not is_verifier_traffic("verifier-lookalike:8000")  # prefix, not substring
    assert not is_verifier_traffic("gpt-oss-120b-kserve:8000")
    assert not is_verifier_traffic(None)


def test_get_and_bodyless_rejected():
    assert classify(_exchange("/v1/chat/completions", {"model": "m"}, b"GET")) is None
    assert classify(_exchange("/v1/chat/completions", None)) is None
    assert classify(_exchange("/v1/chat/completions", {"no_model": True})) is None


# --- the exchange whitelist -------------------------------------------------
#
# `classify` says whether an exchange is inference, which is what decides
# whether to emit. `unexpected_reason` answers the different question the
# tapped link needs: whether it is allowed to be here at all. Before 0.6.0
# anything that was not inference was dropped in silence, and since the
# frame-level policy can only bound a TCP stream to two addresses and a port,
# that silence was where bytes could hide.


def test_inference_is_never_unexpected():
    exchange = _exchange("/v1/chat/completions", {"model": "m", "messages": []})
    assert classify(exchange) is not None
    assert unexpected_reason(exchange) is None


def test_allowed_endpoints_pass_whatever_they_carry():
    """Permissive about shape, strict about identity: these are vLLM's own
    liveness and discovery, and validating their contents buys nothing."""
    for path in ("/v1/models", "/health", "/metrics", "/ping"):
        exchange = _exchange(path, None, method_override=b"GET")
        assert unexpected_reason(exchange) is None, path


def test_a_path_nobody_declared_is_a_finding():
    reason = unexpected_reason(_exchange("/exfil", {"data": "x"}))
    assert reason is not None
    assert "/exfil" in reason


def test_the_verifiers_own_replays_are_excluded_not_unaccounted():
    """Deliberately skipped by classify, so it must not read as unexpected."""
    exchange = _exchange(
        "/v1/chat/completions",
        {"model": "m", "messages": []},
        host="verify-gpt-oss-120b-kserve-workload-svc:8000",
    )
    assert classify(exchange) is None
    assert unexpected_reason(exchange) is None


def test_a_real_endpoint_carrying_something_else_is_distinguished():
    """The case a prover would reach for: the right path, no model."""
    reason = unexpected_reason(_exchange("/v1/chat/completions", {"not_a_model": 1}))
    assert reason is not None
    assert "carried no model" in reason


def test_a_get_to_an_inference_endpoint_is_a_finding():
    """Wrong method rather than wrong body — still not inference, still
    something nobody declared, so it must not vanish."""
    reason = unexpected_reason(
        _exchange("/v1/chat/completions", {"model": "m"}, method_override=b"GET")
    )
    assert reason is not None
