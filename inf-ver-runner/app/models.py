from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"


class ErrorCode(str, Enum):
    UNSUPPORTED_MODEL = "unsupported_model"
    TOKENIZATION_MISMATCH = "tokenization_mismatch"
    SAMPLING_CONFIG_MISSING = "sampling_config_missing"
    VERIFICATION_TIMEOUT = "verification_timeout"
    VERIFIER_UNHEALTHY = "verifier_unhealthy"
    INTERNAL_ERROR = "internal_error"
    INCONCLUSIVE = "inconclusive"
    # The tap captured fewer output token ids than the provider reported
    # generating (usage.completion_tokens): teacher-forcing the partial
    # sequence would verify against the wrong context (e.g. gpt-oss harmony
    # responses on vLLM < v0.11.1 omit analysis-channel tokens).
    TOKEN_CAPTURE_INCOMPLETE = "token_capture_incomplete"
    # The request used constrained decoding (forced tool_choice or a
    # response_format grammar): generation was masked beyond the declared
    # seed/temperature/top_k/top_p contract, so the Gumbel replay does not
    # apply.
    CONSTRAINED_DECODING = "constrained_decoding"


class SamplingConfig(BaseModel):
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    dtype: str | None = None
    quantization: str | None = None


class ModelRequest(BaseModel):
    name: str
    model_id: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None


class VerifyMetadata(BaseModel):
    # Plain-text rendering of the submitted prompt, recorded for display only.
    prompt_text: str | None = None
    user_prompt: str | None = None
    prover_output: str | None = None
    verifier_output: str | None = None
    open_webui_user_id: str | None = None
    open_webui_chat_id: str | None = None
    open_webui_message_id: str | None = None
    open_webui_response_id: str | None = None
    provider_url_label: str | None = None
    requested_at: datetime | None = None
    completed_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] | None = None
    model: ModelRequest
    sampling_config: SamplingConfig
    metadata: VerifyMetadata = Field(default_factory=VerifyMetadata)
    # Capture-integrity fields from the tap (None on events recorded before
    # the tap reported them): the provider's own completion-token count and
    # whether the request used constrained decoding.
    usage_completion_tokens: int | None = None
    constrained_decoding: bool | None = None


class TokenMetricSummary(BaseModel):
    token_count: int
    exact_match_count: int
    exact_match_ratio: float
    margins: list[float] = Field(default_factory=list)
    min_probability: float | None = None
    mean_probability: float | None = None
    mean_margin: float | None = None
    max_margin: float | None = None
    max_logit_rank: float | None = None
    max_gumbel_rank: float | None = None


class OutputTokenComparison(BaseModel):
    index: int
    prover_token_id: int
    verifier_token_id: int
    prover_text: str
    verifier_text: str
    exact_match: bool
    margin: float | None = None


class VerifyResponse(BaseModel):
    request_id: str
    status: VerificationStatus
    reason: str
    error_code: ErrorCode | None = None
    match_target: float | None = None
    metrics: TokenMetricSummary
    latency_ms: int
    # Bracket around the actual vLLM completions call — the GPU-attributable
    # span, excluding the surrounding tokenizer/tensor Python work. None when
    # verification never reached the vLLM call (unverifiable pre-checks,
    # errors). The enrichment poller integrates DCGM series over exactly
    # this window.
    vllm_call_started_at: datetime | None = None
    vllm_call_completed_at: datetime | None = None
    verifier_model_id: str | None = None
    prover_output: str | None = None
    verifier_output: str | None = None
    prover_output_token_ids: list[int] = Field(default_factory=list)
    verifier_output_token_ids: list[int] = Field(default_factory=list)
    output_token_comparison: list[OutputTokenComparison] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    id: str
    display_name: str
    revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str
    quantization: str | None = None
    ready: bool
    max_model_len: int
    max_logprobs: int

