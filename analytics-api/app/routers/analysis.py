"""GET /analysis — moved from ledger's
verification_view.py:analysis (byte-compatible response shape/params).

Joins verification_event + inference_event + model + enrichment_gpu_activity
into the flat, numeric/categorical shape the Analysis tab renders."""

import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Text, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.database import get_session
from app.models import EnrichmentGpuActivity, InferenceEvent, Model, VerificationEvent
from app.schemas import AnalysisPage, AnalysisRow, GpuActivityRead

router = APIRouter(tags=["analysis"])

# The prover's model comes via inference_event.model_id; the verifier's own
# declared model via verification_event.verifier_model_id — same table, twice.
VerifierModel = aliased(Model)

# enrichment_gpu_activity is keyed by (event_type, event_id) — one row for
# the prover's inference event, one for the verifier's verification event.
ProverGpu = aliased(EnrichmentGpuActivity)
VerifyGpu = aliased(EnrichmentGpuActivity)


def _gpu_read(row) -> GpuActivityRead | None:
    if row is None:
        return None
    return GpuActivityRead(
        window_s=row.window_s,
        sample_count=row.sample_count,
        tensor_active_time_s=row.tensor_active_time_s,
        sm_occupancy_mean=row.sm_occupancy_mean,
        pipe_activity_s=row.pipe_activity_s,
        concurrent_events=row.concurrent_events,
    )


def _filters(result: str | None, model: str | None, search: str | None):
    """The subset of ledger's `_view_filters` the `/analysis` endpoint
    reaches: result/model/search clauses (view's own `id`/`model_name`
    dedupe search-column set — no output_text/session columns needed)."""
    clauses = []
    if result:
        clauses.append(VerificationEvent.result == result)
    if model:
        pattern = f"%{model}%"
        clauses.append(
            or_(
                Model.model_name.ilike(pattern),
                VerifierModel.model_name.ilike(pattern),
            )
        )
    if search:
        pattern = f"%{search}%"
        clauses.append(
            or_(
                cast(VerificationEvent.inference_event_id, Text).ilike(pattern),
                VerificationEvent.result_detail.ilike(pattern),
                InferenceEvent.input_text_representation.ilike(pattern),
            )
        )
    return clauses


def _joined(stmt):
    return (
        stmt.join(
            InferenceEvent, VerificationEvent.inference_event_id == InferenceEvent.id
        )
        .join(Model, InferenceEvent.model_id == Model.model_id)
        .join(VerifierModel, VerificationEvent.verifier_model_id == VerifierModel.model_id)
    )


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@router.get("/analysis", response_model=AnalysisPage)
async def analysis(
    result: str | None = None,
    model: str | None = None,
    prover_model: str | None = None,
    verifier_model: str | None = None,
    search: str | None = None,
    from_ts: datetime.datetime | None = None,
    to_ts: datetime.datetime | None = None,
    inference_from_ts: datetime.datetime | None = None,
    inference_to_ts: datetime.datetime | None = None,
    limit: int = Query(50000, ge=1, le=100000),
    session: AsyncSession = Depends(get_session),
):
    """Compact per-event rows for the Analysis tab — numeric/categorical
    fields only, no prompts/outputs/margins/logits. ts DESC so a truncated
    fetch keeps the newest events.

    from_ts/to_ts filter on the VERIFICATION timestamp; inference_from_ts/
    inference_to_ts filter on the INFERENCE timestamp. The latter is how a
    benchmarking run's cohort is defined (run window × inference ts) —
    verifications drain after the run, so a verification-ts window never
    matches the run's own events."""
    clauses = _filters(result, model, search)
    if prover_model:
        clauses.append(Model.model_name.ilike(f"%{prover_model}%"))
    if verifier_model:
        clauses.append(VerifierModel.model_name.ilike(f"%{verifier_model}%"))
    if from_ts is not None:
        clauses.append(VerificationEvent.ts >= from_ts)
    if to_ts is not None:
        clauses.append(VerificationEvent.ts <= to_ts)
    if inference_from_ts is not None:
        clauses.append(InferenceEvent.ts >= inference_from_ts)
    if inference_to_ts is not None:
        clauses.append(InferenceEvent.ts <= inference_to_ts)

    total = await session.scalar(
        _joined(select(func.count(VerificationEvent.id))).where(*clauses)
    )
    rows = await session.execute(
        _joined(
            select(VerificationEvent, Model, VerifierModel, ProverGpu, VerifyGpu)
        )
        .outerjoin(
            ProverGpu,
            (ProverGpu.event_type == "inference")
            & (ProverGpu.event_id == VerificationEvent.inference_event_id),
        )
        .outerjoin(
            VerifyGpu,
            (VerifyGpu.event_type == "verification")
            & (VerifyGpu.event_id == VerificationEvent.id),
        )
        .where(*clauses)
        .order_by(VerificationEvent.ts.desc())
        .limit(limit)
    )
    items = []
    for ver, mdl, verifier_mdl, prover_gpu, verify_gpu in rows.all():
        detail = ver.verifier_detail or {}
        items.append(
            AnalysisRow(
                id=ver.id,
                ts=ver.ts,
                result=ver.result,
                mean_logit_difference=ver.mean_logit_difference,
                exact_match_level_pct=ver.exact_match_level_pct,
                verification_threshold=ver.verification_threshold,
                prover_model=mdl.model_name,
                verifier_model=verifier_mdl.model_name,
                temperature=mdl.temperature,
                top_k=mdl.top_k,
                top_p=mdl.top_p,
                prompt_tokens=_int_or_none(detail.get("prompt_token_count")),
                output_tokens=_int_or_none(detail.get("output_token_count")),
                latency_ms=_float_or_none(detail.get("latency_ms")),
                prover_gpu=_gpu_read(prover_gpu),
                verify_gpu=_gpu_read(verify_gpu),
            )
        )
    return AnalysisPage(items=items, total=total or 0, truncated=(total or 0) > len(items))
