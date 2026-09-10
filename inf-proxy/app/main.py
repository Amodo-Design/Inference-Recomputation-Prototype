"""Inference tap proxy.

OpenAI-compatible chat/inference endpoints that forward to the upstream model
and relay the response **untouched**, while copying each completed inference to
the Message writer (a passive network-tap replacement). Model-metadata routes
(/v1/models, etc.) are served by nginx straight from the model and never reach
this app.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.capture import (
    extract_prompt_output_token_ids,
    extract_prompt_text,
    sampling_config,
)
from app.config import INF_PROXY_MODEL_API_KEY, INF_PROXY_MODEL_BASE_URL
from app.declaration import close_deployment, spawn_declaration
from app.logging_setup import configure_logging
from app.message_writer import (
    build_tap_message,
    close_session,
    get_session,
    spawn_tap,
    utc_now_iso,
)
from app.payload import (
    add_tap_report_options,
    detect_constrained_decoding,
    inject_sampling_config,
)
from app.streaming import _has_tool_calls, stream_and_tap

configure_logging("inf-proxy")
log = logging.getLogger("proxy")

app = FastAPI(title="Inference Tap Proxy", version="0.1.0")

# Streamed generations can run long; don't impose a total deadline, only a
# connect timeout.
_MODEL_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30)


@app.on_event("startup")
async def _startup() -> None:
    # Best-effort background self-declaration to the ledger (opt-in via
    # INF_PROXY_DECLARE); never blocks or fails serving.
    spawn_declaration()


@app.on_event("shutdown")
async def _shutdown() -> None:
    # Report the prover's deployment as ended before dropping connections
    # (best-effort: SIGKILL or a dead ledger just leaves it open, and the
    # next declare on this hostname closes it).
    await close_deployment()
    await close_session()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _metadata_from_headers(request: Request) -> dict[str, str | None]:
    h = request.headers
    return {
        "user_id": h.get("X-OpenWebUI-User-Id"),
        "chat_id": h.get("X-OpenWebUI-Chat-Id"),
        "message_id": h.get("X-OpenWebUI-Message-Id"),
    }


def _upstream_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if INF_PROXY_MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {INF_PROXY_MODEL_API_KEY}"
    return headers


def _response_fields(data: dict[str, Any], object_kind: str) -> dict[str, Any]:
    """Pull output text / logprobs / finish_reason from a non-streamed response."""
    choices = data.get("choices") or []
    choice0 = choices[0] if choices else {}
    if object_kind == "text":
        output_text = choice0.get("text")
    else:
        output_text = (choice0.get("message") or {}).get("content")
    lp = choice0.get("logprobs")
    if isinstance(lp, dict) and isinstance(lp.get("content"), list):
        output_logprobs: Any = lp["content"]
    else:
        output_logprobs = lp or None
    return {
        "output_text": output_text if isinstance(output_text, str) else None,
        "output_logprobs": output_logprobs,
        "finish_reason": choice0.get("finish_reason"),
        "tool_calls": _has_tool_calls(choice0) if choice0 else False,
    }


async def _handle(request: Request, upstream_path: str, object_kind: str, public_endpoint: str):
    received_at = utc_now_iso()
    body = await request.json()
    metadata = _metadata_from_headers(request)
    client_wants_stream = bool(body.get("stream", False))

    payload = dict(body)
    add_tap_report_options(payload, object_kind=object_kind)
    inject_sampling_config(payload)

    session = await get_session()
    url = f"{INF_PROXY_MODEL_BASE_URL}{upstream_path}"

    if client_wants_stream:
        payload["stream"] = True
        resp = await session.post(
            url, data=json.dumps(payload), headers=_upstream_headers(), timeout=_MODEL_TIMEOUT
        )

        if resp.status >= 400:
            # The model rejected the request (e.g. vLLM's 400 for unsupported
            # `tools`). Never convert this into an SSE stream: the error body
            # has no `data:` lines, so streaming it would produce an empty
            # 200 stream ending in [DONE] — masking the failure from both the
            # client and the ledger. Relay status + body like the
            # non-streaming path does.
            try:
                try:
                    error_body: Any = await resp.json()
                except Exception:
                    error_body = {
                        "error": {"message": (await resp.text())[:2000], "code": resp.status}
                    }
            finally:
                resp.release()
            log.warning(
                "Upstream rejected streaming %s request with %d: %.500s",
                public_endpoint,
                resp.status,
                error_body,
            )
            return JSONResponse(status_code=resp.status, content=error_body)

        async def gen():
            try:
                async for chunk in stream_and_tap(
                    resp, payload, metadata, public_endpoint, object_kind, received_at
                ):
                    yield chunk
            finally:
                resp.release()

        return StreamingResponse(gen(), media_type="text/event-stream")

    # Non-streaming: relay the model's JSON as-is, then tap in the background.
    payload["stream"] = False
    async with session.post(
        url, data=json.dumps(payload), headers=_upstream_headers(), timeout=_MODEL_TIMEOUT
    ) as resp:
        status = resp.status
        data = await resp.json()

    if status >= 400:
        return JSONResponse(data, status_code=status)

    prompt_ids, output_ids = extract_prompt_output_token_ids(data)
    fields = _response_fields(data, object_kind)
    message = build_tap_message(
        endpoint=public_endpoint,
        streamed=False,
        received_at=received_at,
        completed_at=utc_now_iso(),
        metadata=metadata,
        model_name=data.get("model") or payload.get("model"),
        sampling=sampling_config(payload),
        prompt_text=extract_prompt_text(payload),
        prompt_token_ids=prompt_ids,
        output_text=fields["output_text"],
        output_token_ids=output_ids,
        output_logprobs=fields["output_logprobs"],
        finish_reason=fields["finish_reason"],
        tool_calls=fields["tool_calls"],
        response_id=data.get("id"),
        usage_completion_tokens=(
            data["usage"].get("completion_tokens")
            if isinstance(data.get("usage"), dict)
            and isinstance(data["usage"].get("completion_tokens"), int)
            else None
        ),
        constrained_decoding=detect_constrained_decoding(payload),
    )
    spawn_tap(message)
    return JSONResponse(data)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle(request, "/chat/completions", "chat", "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle(request, "/completions", "text", "/v1/completions")
