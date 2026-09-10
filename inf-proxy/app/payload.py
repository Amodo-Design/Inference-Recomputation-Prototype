"""Upstream request shaping.

The tap injects output-*reporting* flags so the model returns token IDs and
logprobs for the ledger. It does NOT otherwise alter generation.

The exception is the sampling config the verifier depends on — seed,
temperature, top_k, top_p. When any of INF_PROXY_SEED/_TEMPERATURE/_TOP_K/
_TOP_P is configured, the proxy PINS every request to it — including
overriding client-supplied values — because the declared model config (which
includes these fields) is the verification contract: an inference generated
under different values could never verify against it. Each field is an
independent knob; one left unset passes through unchanged.
"""

from __future__ import annotations

import logging

from app.config import (
    INF_PROXY_LOGPROBS,
    INF_PROXY_SEED,
    INF_PROXY_TEMPERATURE,
    INF_PROXY_TOP_K,
    INF_PROXY_TOP_LOGPROBS,
    INF_PROXY_TOP_P,
)

log = logging.getLogger("proxy.payload")


def add_tap_report_options(payload: dict, object_kind: str = "chat") -> dict:
    """Ask vLLM-compatible providers to include prompt/output token IDs (and
    optionally logprobs). Reporting-only: nothing here alters generation, and
    an explicit client value always wins (setdefault)."""
    payload.setdefault("return_token_ids", True)

    # Streams only carry the provider's token accounting (usage) in a final
    # chunk when asked. The tap needs usage.completion_tokens to assert the
    # captured token ids are COMPLETE (see token_capture_incomplete) - a
    # partial capture (e.g. gpt-oss harmony on vLLM < v0.11.1 omitting
    # analysis-channel tokens) is silent without it.
    if payload.get("stream"):
        opts = payload.setdefault("stream_options", {})
        if isinstance(opts, dict):
            opts.setdefault("include_usage", True)

    if INF_PROXY_LOGPROBS:
        if object_kind == "chat":
            payload.setdefault("logprobs", True)
            payload.setdefault("top_logprobs", INF_PROXY_TOP_LOGPROBS)
        else:
            # Legacy completions: logprobs is an integer count.
            payload.setdefault("logprobs", INF_PROXY_TOP_LOGPROBS)
    return payload


def _pin(payload: dict, field: str, configured_value) -> None:
    if configured_value is None:
        return
    client_value = payload.get(field)
    if client_value is not None and client_value != configured_value:
        log.info(
            "Overriding client-supplied %s %s with declared %s %s",
            field,
            client_value,
            field,
            configured_value,
        )
    payload[field] = configured_value


def inject_sampling_config(payload: dict) -> dict:
    """Pin the request to the configured sampling config (seed, temperature,
    top_k, top_p) — one field at a time, each independently optional.

    Every configured field is enforced on EVERY request — a client-supplied
    value is overridden (and the override logged), because the declared
    model config is the verification contract and an inference generated
    under different values is unverifiable against it. A field left
    unconfigured is a no-op for that field. Because the tap reads sampling
    from this same payload, the enforced values are both used upstream and
    captured in the ledger event.
    """
    _pin(payload, "seed", INF_PROXY_SEED)
    _pin(payload, "temperature", INF_PROXY_TEMPERATURE)
    _pin(payload, "top_k", INF_PROXY_TOP_K)
    _pin(payload, "top_p", INF_PROXY_TOP_P)
    return payload


def detect_constrained_decoding(payload: dict) -> bool:
    """True when the request constrains generation beyond the declared
    sampling contract - a grammar/mask the verifier cannot replay.

    Forced tool choice: ``tool_choice`` naming a function (dict) or
    ``"required"`` (``"auto"``/``"none"`` leave sampling unconstrained).
    Structured output: ``response_format`` of any type except plain text.
    """
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) or tool_choice == "required":
        return True
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (None, "text"):
        return True
    return False
