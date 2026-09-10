"""Pydantic v2 response schemas for the `/analysis` endpoint (copied
verbatim from ledger/app/schemas.py)."""

import datetime
import uuid

from pydantic import BaseModel


class GpuActivityRead(BaseModel):
    """Joined GPU-activity fields for one side of an analysis row."""

    window_s: float
    sample_count: int
    tensor_active_time_s: float | None
    sm_occupancy_mean: float | None
    pipe_activity_s: dict[str, float] | None
    concurrent_events: int | None


class AnalysisRow(BaseModel):
    """Compact per-event row for the Analysis tab — numeric/categorical
    fields only, no prompts/outputs/margins/logits."""

    id: uuid.UUID
    ts: datetime.datetime
    result: str | None
    mean_logit_difference: float | None
    exact_match_level_pct: float | None
    verification_threshold: float | None
    prover_model: str
    verifier_model: str
    temperature: float | None
    top_k: int | None
    top_p: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None

    # GPU activity for each side, when the enrichment poller has run.
    prover_gpu: GpuActivityRead | None = None
    verify_gpu: GpuActivityRead | None = None


class AnalysisPage(BaseModel):
    items: list[AnalysisRow]
    total: int
    truncated: bool
