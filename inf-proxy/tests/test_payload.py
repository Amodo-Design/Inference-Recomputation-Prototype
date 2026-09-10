"""Sampling config enforcement: the tap pins every request to the declared
seed, temperature, top_k and top_p.

The declared model config is the verification contract: the verifier
re-executes deterministically using exactly these four values read back off
the ledger's model row (component-setup.md, "Sampling config must be declared
truthfully"). Any field left null makes every event
`unverifiable/sampling_config_missing`; any field that doesn't match how the
model actually sampled makes verification (correctly) fail. So the tap
OVERRIDES client-supplied values for every one of these fields rather than
deferring to them, exactly as it already did for seed alone.
"""

from __future__ import annotations

from app import payload as payload_mod


def _set(monkeypatch, **overrides):
    for name in (
        "INF_PROXY_SEED",
        "INF_PROXY_TEMPERATURE",
        "INF_PROXY_TOP_K",
        "INF_PROXY_TOP_P",
    ):
        monkeypatch.setattr(payload_mod, name, overrides.get(name))


def test_all_configured_fields_injected_when_request_has_none(monkeypatch):
    _set(
        monkeypatch,
        INF_PROXY_SEED=42,
        INF_PROXY_TEMPERATURE=1.0,
        INF_PROXY_TOP_K=50,
        INF_PROXY_TOP_P=0.95,
    )
    p = {"model": "m", "messages": []}
    payload_mod.inject_sampling_config(p)
    assert p["seed"] == 42
    assert p["temperature"] == 1.0
    assert p["top_k"] == 50
    assert p["top_p"] == 0.95


def test_all_configured_fields_override_client_values(monkeypatch):
    _set(
        monkeypatch,
        INF_PROXY_SEED=42,
        INF_PROXY_TEMPERATURE=1.0,
        INF_PROXY_TOP_K=50,
        INF_PROXY_TOP_P=0.95,
    )
    p = {
        "model": "m",
        "messages": [],
        "seed": 1234,
        "temperature": 0.2,
        "top_k": 5,
        "top_p": 0.5,
    }
    payload_mod.inject_sampling_config(p)
    assert p["seed"] == 42
    assert p["temperature"] == 1.0
    assert p["top_k"] == 50
    assert p["top_p"] == 0.95


def test_unconfigured_fields_left_untouched(monkeypatch):
    # Only seed configured — the four knobs are independent; an unset one
    # must not be forced, and must not appear if the client omitted it.
    _set(monkeypatch, INF_PROXY_SEED=42)
    p = {"model": "m", "messages": [], "temperature": 0.2}
    payload_mod.inject_sampling_config(p)
    assert p["seed"] == 42
    assert p["temperature"] == 0.2
    assert "top_k" not in p
    assert "top_p" not in p


def test_nothing_configured_passes_request_through_unchanged(monkeypatch):
    _set(monkeypatch)
    p = {"model": "m", "messages": [], "seed": 1234, "temperature": 0.2, "top_k": 5, "top_p": 0.5}
    payload_mod.inject_sampling_config(p)
    assert p == {
        "model": "m",
        "messages": [],
        "seed": 1234,
        "temperature": 0.2,
        "top_k": 5,
        "top_p": 0.5,
    }

    q = {"model": "m", "messages": []}
    payload_mod.inject_sampling_config(q)
    assert q == {"model": "m", "messages": []}
