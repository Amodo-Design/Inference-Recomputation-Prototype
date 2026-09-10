"""Streaming relay with end-of-stream tap.

Forwards every SSE chunk to the client live and untouched. While relaying it
captures the top-level prompt_token_ids and the per-choice output token ids,
logprobs, and text. When the relay ends — normal completion, upstream error, or
client disconnect — it assembles the tap message in a `finally` block and sends
it to the Message writer as a background task, so the client is never delayed,
modified, or blocked, and no streamed inference is dropped on early disconnect.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import aiohttp

from app.capture import extract_prompt_text, sampling_config
from app.message_writer import build_tap_message, spawn_tap, utc_now_iso
from app.payload import detect_constrained_decoding

log = logging.getLogger("proxy.streaming")


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def _first_token_list(data: Any, keys: tuple[str, ...]) -> list[int] | None:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if _is_int_list(value):
                return value
        for value in data.values():
            found = _first_token_list(value, keys)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _first_token_list(item, keys)
            if found:
                return found
    return None


def _has_tool_calls(choice: dict[str, Any]) -> bool:
    delta = choice.get("delta") or {}
    message = choice.get("message") or {}
    return bool(delta.get("tool_calls") or message.get("tool_calls"))


def _chunk_text(choice: dict[str, Any], object_kind: str) -> str | None:
    if object_kind == "text":
        text = choice.get("text")
        return text if isinstance(text, str) else None
    delta = choice.get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else None


async def stream_and_tap(
    upstream: aiohttp.ClientResponse,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    endpoint: str,
    object_kind: str,
    received_at: str,
) -> AsyncIterator[bytes]:
    prompt_ids: list[int] | None = None
    output_ids: list[int] = []
    usage_completion_tokens = None
    output_logprobs: list[Any] = []
    output_text_parts: list[str] = []
    response_id: str | None = None
    model: str | None = payload.get("model")
    finish_reason: str | None = None
    saw_tool_calls = False

    completed = False
    try:
        async for raw in upstream.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                # Forward anything we can't parse untouched.
                yield f"data: {data_str}\n\n".encode("utf-8")
                continue

            if response_id is None and data.get("id"):
                response_id = data["id"]
            if data.get("model"):
                model = data["model"]
            if prompt_ids is None:
                prompt_ids = _first_token_list(data, ("prompt_token_ids", "input_token_ids"))
            usage = data.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                usage_completion_tokens = usage["completion_tokens"]

            choices = data.get("choices") or []
            if choices:
                choice0 = choices[0]
                tids = _first_token_list(
                    choice0, ("token_ids", "output_token_ids", "completion_token_ids")
                )
                if tids:
                    output_ids.extend(tids)
                lp = choice0.get("logprobs")
                if isinstance(lp, dict) and isinstance(lp.get("content"), list):
                    output_logprobs.extend(lp["content"])
                elif lp:
                    output_logprobs.append(lp)
                text = _chunk_text(choice0, object_kind)
                if text:
                    output_text_parts.append(text)
                if _has_tool_calls(choice0):
                    saw_tool_calls = True
                if choice0.get("finish_reason"):
                    finish_reason = choice0["finish_reason"]

            # Forward live, untouched.
            yield f"data: {json.dumps(data)}\n\n".encode("utf-8")

        # Natural end of stream — terminate exactly as the model would have.
        completed = True
        yield b"data: [DONE]\n\n"
    finally:
        # Emit the tap whether the stream completed normally, errored, or the
        # client disconnected mid-stream (which throws GeneratorExit/CancelledError
        # into us at a `yield`). Running this in `finally` is what makes streamed
        # captures reliable — the previous post-loop version was skipped entirely
        # on early client disconnect. Skip only if we captured nothing at all.
        if prompt_ids or output_ids or output_text_parts:
            message = build_tap_message(
                endpoint=endpoint,
                streamed=True,
                received_at=received_at,
                completed_at=utc_now_iso(),
                metadata=metadata,
                model_name=model,
                sampling=sampling_config(payload),
                prompt_text=extract_prompt_text(payload),
                prompt_token_ids=prompt_ids,
                output_text="".join(output_text_parts) or None,
                output_token_ids=output_ids or None,
                output_logprobs=output_logprobs or None,
                finish_reason=finish_reason,
                tool_calls=saw_tool_calls,
                response_id=response_id,
                usage_completion_tokens=usage_completion_tokens,
                constrained_decoding=detect_constrained_decoding(payload),
            )
            spawn_tap(message)
            log.info(
                "Tap queued (stream) session=%s output_tokens=%s completed=%s",
                message["session"]["session_id"],
                len(output_ids),
                completed,
            )
