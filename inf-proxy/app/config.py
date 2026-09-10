"""Proxy configuration, read from the environment.

The proxy is a passive **tap**: it forwards inference traffic to the upstream
model untouched and copies each inference to the Message writer. It does not
alter sampling; it only injects output-*reporting* flags (return_token_ids,
logprobs) so the token IDs / logprobs come back for the ledger.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_int_or_none(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


# --- Upstream model (OpenAI-compatible) ---
INF_PROXY_MODEL_BASE_URL = os.getenv("INF_PROXY_MODEL_BASE_URL", "http://dummy-model:8000/v1").rstrip("/")
INF_PROXY_MODEL_API_KEY = os.getenv("INF_PROXY_MODEL_API_KEY") or None
INF_PROXY_APP_PORT = _as_int(os.getenv("INF_PROXY_APP_PORT"), 9000)

# Hostname the proxy forwards to. This is the ledger resolution key and MUST
# match the `hostname` a model declares at startup (POST /model-deployments/
# declare). Override with INF_PROXY_MODEL_HOSTNAME; otherwise derived from the
# upstream base URL.
MODEL_HOSTNAME = os.getenv("INF_PROXY_MODEL_HOSTNAME") or urlparse(INF_PROXY_MODEL_BASE_URL).hostname or ""

# --- Message writer (the tap destination) ---
INF_PROXY_TAP_ENABLED = _as_bool(os.getenv("INF_PROXY_TAP_ENABLED"), True)
INF_PROXY_MESSAGE_WRITER_URL = os.getenv(
    "INF_PROXY_MESSAGE_WRITER_URL", "http://message-writer:8100"
).rstrip("/")
INF_PROXY_MESSAGE_WRITER_PATH = os.getenv("INF_PROXY_MESSAGE_WRITER_PATH", "/inferences")
INF_PROXY_MESSAGE_WRITER_TIMEOUT = _as_int(os.getenv("INF_PROXY_MESSAGE_WRITER_TIMEOUT"), 30)

# --- Output reporting (does not change generation) ---
# Force the model to return token IDs (always) and, optionally, logprobs.
INF_PROXY_LOGPROBS = _as_bool(os.getenv("INF_PROXY_LOGPROBS"), True)
INF_PROXY_TOP_LOGPROBS = _as_int(os.getenv("INF_PROXY_TOP_LOGPROBS"), 5)

# --- Sampling enforcement (for verification) ---
# The verifier re-executes an inference deterministically using seed,
# temperature, top_k and top_p read back off the ledger's declared model row.
# Any of the four left null makes every event unverifiable/
# sampling_config_missing; any that doesn't match how the model actually
# sampled makes verification (correctly) fail. So each configured field is
# PINNED on every request (client-supplied values are overridden), keeping
# actual generation and the declared contract in lockstep. Each is an
# independent knob: unset (or blank) -> that field passes through unchanged.
INF_PROXY_SEED = _as_int_or_none(os.getenv("INF_PROXY_SEED"))
INF_PROXY_TEMPERATURE = _as_float_or_none(os.getenv("INF_PROXY_TEMPERATURE"))
INF_PROXY_TOP_K = _as_int_or_none(os.getenv("INF_PROXY_TOP_K"))
INF_PROXY_TOP_P = _as_float_or_none(os.getenv("INF_PROXY_TOP_P"))

# --- Startup self-declaration to the ledger (opt-in) ---
# When enabled, the tap declares the model deployment it fronts to the ledger
# at startup (POST /model-deployments/declare), replacing the manual declare
# step: hostname = MODEL_HOSTNAME, model_name discovered from the upstream
# /v1/models (or INF_PROXY_MODEL_NAME), seed = INF_PROXY_SEED. Off by default
# so setups where the model self-declares (e.g. the dummy model) don't end up
# with two competing declarations for one hostname.
INF_PROXY_DECLARE = _as_bool(os.getenv("INF_PROXY_DECLARE"), False)
# Owner recorded on the declared hardware ("prover" for taps; the runners
# declare "verifier"). Blank disables owner attribution.
INF_PROXY_OWNER_NAME = os.getenv("INF_PROXY_OWNER_NAME", "prover") or None
INF_PROXY_LEDGER_API_URL = os.getenv(
    "INF_PROXY_LEDGER_API_URL", "http://ledger-api:8000"
).rstrip("/")
# Skips upstream discovery when set (useful if /v1/models is unavailable).
INF_PROXY_MODEL_NAME = os.getenv("INF_PROXY_MODEL_NAME") or None
# Recorded on the declared model config (part of the derived model_id).
INF_PROXY_DECODING_ALGORITHM = os.getenv("INF_PROXY_DECODING_ALGORITHM") or None
INF_PROXY_DECLARE_RETRY_SECONDS = _as_int(os.getenv("INF_PROXY_DECLARE_RETRY_SECONDS"), 5)

# --- Pod identity from Downward API (Kubernetes only) ---
# Injected by Kubernetes Downward API in cluster deployments; absent in
# docker-compose. Stamped on every tap message for enrichment downstream.
INF_PROXY_POD_NAME = os.getenv("INF_PROXY_POD_NAME") or None
INF_PROXY_NODE_NAME = os.getenv("INF_PROXY_NODE_NAME") or None

PROXY_VERSION = "0.1.0"
