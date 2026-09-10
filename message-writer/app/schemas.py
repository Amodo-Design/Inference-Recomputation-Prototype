"""Tap-message schema.

Everything is optional/lenient so a slightly-varying tap message still parses;
the transform fills sensible fallbacks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TapInfo(BaseModel):
    source: str | None = None
    proxy_version: str | None = None
    hostname: str | None = None
    upstream_base_url: str | None = None
    endpoint: str | None = None
    streamed: bool | None = None
    received_at: str | None = None
    completed_at: str | None = None
    pod_name: str | None = None
    node_name: str | None = None


class Session(BaseModel):
    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    message_id: str | None = None
    response_id: str | None = None


class ModelInfo(BaseModel):
    name: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None


class RequestData(BaseModel):
    prompt_text: str | None = None
    prompt_token_ids: list[int] | None = None


class ResponseData(BaseModel):
    output_text: str | None = None
    output_token_ids: list[int] | None = None
    output_logprobs: Any = None
    finish_reason: str | None = None
    tool_calls: bool | None = None
    # Capture-integrity fields from the tap: the provider's own
    # completion-token count and whether generation was constrained beyond
    # the sampling contract. Flow into the output payload blob for the
    # verifier's pre-checks.
    usage_completion_tokens: int | None = None
    constrained_decoding: bool | None = None


class TapMessage(BaseModel):
    tap: TapInfo = TapInfo()
    session: Session = Session()
    model: ModelInfo = ModelInfo()
    sampling: Any = None
    request: RequestData = RequestData()
    response: ResponseData = ResponseData()
