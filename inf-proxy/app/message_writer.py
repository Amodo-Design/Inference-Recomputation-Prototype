"""Message-writer client (the tap destination).

Builds the tap message from captured inference data and sends it to the Message
writer. Sending is best-effort and must never affect the client response: errors
are swallowed and logged, and sends are scheduled as background tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from app.config import (
    INF_PROXY_MESSAGE_WRITER_PATH,
    INF_PROXY_MESSAGE_WRITER_TIMEOUT,
    INF_PROXY_MESSAGE_WRITER_URL,
    INF_PROXY_MODEL_BASE_URL,
    INF_PROXY_LOGPROBS,
    INF_PROXY_TOP_LOGPROBS,
    INF_PROXY_TAP_ENABLED,
    INF_PROXY_POD_NAME,
    INF_PROXY_NODE_NAME,
    MODEL_HOSTNAME,
    PROXY_VERSION,
)

log = logging.getLogger("proxy.message_writer")

_session: aiohttp.ClientSession | None = None

# Strong references to in-flight fire-and-forget tap sends. Without this, the
# event loop only holds a weak reference to a bare create_task() result and may
# garbage-collect it before it runs (esp. when scheduled from a cancelled
# streaming generator on client disconnect).
_pending_taps: set[asyncio.Task] = set()


def spawn_tap(message: dict[str, Any]) -> None:
    """Schedule send_tap_message as a tracked background task.

    Safe to call from a request handler or from the finally block of a
    (possibly cancelled) streaming generator: the created task is independent of
    the caller's cancellation and is kept referenced until it completes.
    """
    try:
        task = asyncio.create_task(send_tap_message(message))
    except RuntimeError:
        # No running loop (should not happen in a request context) — drop.
        log.error("Cannot schedule tap send: no running event loop")
        return
    _pending_taps.add(task)
    task.add_done_callback(_pending_taps.discard)


async def get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close_session() -> None:
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_tap_message(
    *,
    endpoint: str,
    streamed: bool,
    received_at: str,
    completed_at: str,
    metadata: dict[str, Any],
    model_name: str | None,
    sampling: dict[str, Any],
    prompt_text: str | None,
    prompt_token_ids: list[int] | None,
    output_text: str | None,
    output_token_ids: list[int] | None,
    output_logprobs: Any,
    finish_reason: str | None,
    tool_calls: bool,
    response_id: str | None,
    usage_completion_tokens: int | None = None,
    constrained_decoding: bool = False,
) -> dict[str, Any]:
    """Assemble the tap message (schema: message-writer/app/schemas.py)."""
    return {
        "tap": {
            "source": "inf-proxy",
            "proxy_version": PROXY_VERSION,
            "hostname": MODEL_HOSTNAME,
            "upstream_base_url": INF_PROXY_MODEL_BASE_URL,
            "endpoint": endpoint,
            "streamed": streamed,
            "received_at": received_at,
            "completed_at": completed_at,
            "pod_name": INF_PROXY_POD_NAME,
            "node_name": INF_PROXY_NODE_NAME,
        },
        "session": {
            "session_id": metadata.get("chat_id") or response_id,
            "user_id": metadata.get("user_id"),
            "chat_id": metadata.get("chat_id"),
            "message_id": metadata.get("message_id"),
            "response_id": response_id,
        },
        "model": {
            "name": model_name,
            "revision": None,
            "tokenizer_revision": None,
        },
        "sampling": {
            **sampling,
            "reporting_flags_added": {
                "return_token_ids": True,
                "logprobs": INF_PROXY_LOGPROBS,
                "top_logprobs": INF_PROXY_TOP_LOGPROBS if INF_PROXY_LOGPROBS else None,
            },
        },
        "request": {
            "prompt_text": prompt_text,
            "prompt_token_ids": prompt_token_ids,
        },
        "response": {
            "output_text": output_text,
            "output_token_ids": output_token_ids,
            "output_logprobs": output_logprobs,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
            # Capture-integrity fields: the provider's own completion-token
            # count (from usage) and whether the request used constrained
            # decoding (forced tool_choice / response_format grammar). The
            # verifier refuses to verify when the capture is incomplete or
            # generation was masked beyond the sampling contract.
            "usage_completion_tokens": usage_completion_tokens,
            "constrained_decoding": constrained_decoding,
        },
    }


async def send_tap_message(message: dict[str, Any]) -> None:
    """POST the tap message to the Message writer. Best-effort: never raises."""
    if not INF_PROXY_TAP_ENABLED:
        return
    url = f"{INF_PROXY_MESSAGE_WRITER_URL}{INF_PROXY_MESSAGE_WRITER_PATH}"
    try:
        session = await get_session()
        async with session.post(
            url,
            data=json.dumps(message),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=INF_PROXY_MESSAGE_WRITER_TIMEOUT),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                log.error("Message writer returned %s: %s", resp.status, body[:500])
            else:
                log.info(
                    "Tapped inference sent to message writer (session=%s status=%s)",
                    message.get("session", {}).get("session_id"),
                    resp.status,
                )
    except Exception as exc:
        # A tap failure must never affect the client — log and move on.
        log.error("Failed to send tap message to %s: %s", url, exc)
