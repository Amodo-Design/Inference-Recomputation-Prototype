"""LLM traffic identification.

Content-based: an exchange is LLM inference traffic if it is a POST to an
OpenAI-compatible completions endpoint whose JSON body carries `model`. This
classifies traffic by what it *is*, so it survives pod IP churn, gateway path
prefixes, and CNI encapsulation, and needs no packet marking.

Traffic addressed to a verifier vLLM is excluded — see
FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES in config.py for why the verifier's own
replays are not inference this tap should be finding.

Fallback (only if this proves insufficient in practice): have the
`verify-tap` sidecar — already in the request path and already ours — stamp a
header such as `X-INFVER-TAP: <hostname>` on forwarded requests, and match it
here post-reassembly. Deliberately NOT implemented until content-based
filtering has been shown wanting; marking packets makes the capture depend on
a non-standard serving path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import FRAME_PROCESSOR_ALLOWED_ENDPOINTS, FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES
from app.http_stream import Exchange

log = logging.getLogger("frameprocessor.filter")

# The gateway prefixes routed paths (e.g. /infver/gpt-oss-120b/v1/...), so
# match on the suffix. Order matters: check the longer one first.
LLM_ENDPOINTS = ("/v1/chat/completions", "/v1/completions")


@dataclass(frozen=True)
class LlmClassification:
    endpoint: str  # normalised: which LLM_ENDPOINTS entry matched
    model_name: str
    request_body: dict[str, Any]


def is_verifier_traffic(request_host: str | None) -> bool:
    """True for a request addressed to a verifier vLLM (see config)."""
    if not request_host or not FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES:
        return False
    # Host is "name:port"; IPv6 literals are bracketed, so split off the port
    # only from the tail. Neither form can match a DNS-name prefix anyway.
    host = request_host.strip().lower().rsplit(":", 1)[0].strip("[]")
    return host.startswith(FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES)


def classify(exchange: Exchange) -> LlmClassification | None:
    """LlmClassification if this exchange is LLM inference traffic, else None."""
    request = exchange.request
    if request.method != "POST" or not request.path:
        return None

    if is_verifier_traffic(request.headers.get("host")):
        log.debug("Skipping verifier traffic to %s", request.headers.get("host"))
        return None

    path = request.path.split("?", 1)[0].rstrip("/")
    endpoint = next((e for e in LLM_ENDPOINTS if path.endswith(e)), None)
    if endpoint is None:
        return None

    body = request.json()
    if body is None:
        return None
    model_name = body.get("model")
    if not isinstance(model_name, str) or not model_name:
        return None

    return LlmClassification(endpoint=endpoint, model_name=model_name, request_body=body)


def unexpected_reason(exchange: Exchange) -> str | None:
    """Why this exchange should not be on the tapped link, or None if it may be.

    `classify` answers "is this inference?" and collapses several different
    reasons into one `None`: not a POST, addressed to a verifier, wrong path,
    no model in the body. Only emitting ever cared about that distinction, so
    everything which was not inference was dropped in silence.

    On a link whose whole claim is that all traffic is accounted for, that
    silence is the gap. The frame-level whitelist in policy.py can bound a TCP
    stream to two addresses and one port, but it cannot say what travels
    inside — the shape of an exchange only exists after reassembly. So an
    exchange that reassembles and classifies as nothing is precisely where
    bytes would hide. This names it instead.

    Deliberately strict about identity and permissive about shape: a request
    whose path is on the allowed list passes whatever its method or body,
    because those endpoints are vLLM's own liveness and discovery and
    validating their contents buys nothing. What does not pass is a path
    nobody declared.
    """
    request = exchange.request

    if classify(exchange) is not None:
        return None
    # The verifier's own replays are excluded deliberately, not unaccounted.
    if is_verifier_traffic(request.headers.get("host")):
        return None

    path = (request.path or "").split("?", 1)[0].rstrip("/")
    if path and any(path.endswith(allowed) for allowed in FRAME_PROCESSOR_ALLOWED_ENDPOINTS):
        return None

    if not path:
        # Parsed far enough to become an Exchange but carries no addressable
        # request line.
        return "exchange with no request path"

    method = request.method or "?"
    if any(path.endswith(endpoint) for endpoint in LLM_ENDPOINTS):
        # The right path but not inference — no model in the body, or a method
        # other than POST. Worth distinguishing, because it is the case a
        # prover would reach for: a real endpoint carrying something else.
        return f"{method} {path} matched an inference endpoint but carried no model"
    return f"{method} {path} is not inference and not an allowed endpoint"
