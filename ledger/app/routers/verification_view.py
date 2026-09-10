"""UI-facing read (and delete) endpoints over verification events.

These join verification_event + inference_event + model into the flat shape
the results UI renders. Registered before the verification_event CRUD router
so the literal /view and /stats paths are not captured by GET /{item_id}.
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import Text, cast, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.database import get_session
from app.models import InferenceEvent, Model, VerificationEvent
from app.schemas import (
    VerificationEventView,
    VerificationEventViewPage,
    VerificationStats,
)

router = APIRouter(prefix="/verification-events", tags=["verification_view"])

# The prover's model comes via inference_event.model_id; the verifier's own
# declared model via verification_event.verifier_model_id — same table, twice.
VerifierModel = aliased(Model)


def _decode_margins(raw: bytes | None) -> list[float]:
    if not raw:
        return []
    try:
        margins = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return []
    if not isinstance(margins, list):
        return []
    return [float(m) for m in margins if isinstance(m, (int, float))]


def _view_filters(result: str | None, model: str | None, search: str | None):
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


@router.get("/view", response_model=VerificationEventViewPage)
async def view(
    limit: int = Query(25, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    result: str | None = None,
    model: str | None = None,
    search: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    clauses = _view_filters(result, model, search)

    total = await session.scalar(
        _joined(select(func.count(VerificationEvent.id))).where(*clauses)
    )
    rows = await session.execute(
        _joined(select(VerificationEvent, InferenceEvent, Model, VerifierModel))
        .where(*clauses)
        .order_by(VerificationEvent.ts.desc())
        .limit(limit)
        .offset(offset)
    )

    items = [
        VerificationEventView(
            id=ver.id,
            inference_event_id=ver.inference_event_id,
            ts=ver.ts,
            result=ver.result,
            result_detail=ver.result_detail,
            error_code=ver.error_code,
            verification_threshold=ver.verification_threshold,
            verifier_model_id=ver.verifier_model_id,
            verifier_model_name=verifier_mdl.model_name,
            model_name=mdl.model_name,
            sampling_config={
                "temperature": mdl.temperature,
                "top_k": mdl.top_k,
                "top_p": mdl.top_p,
                "seed": mdl.seed,
                "decoding_algorithm": mdl.decoding_algorithm,
            },
            exact_match_level_pct=ver.exact_match_level_pct,
            mean_logit_difference=ver.mean_logit_difference,
            std_dev_logit_difference=ver.std_dev_logit_difference,
            difr_margins=_decode_margins(ver.logit_difference_margins),
            session_id=inf.session_id,
            input_text_representation=inf.input_text_representation,
            output_text_representation=inf.output_text_representation,
            verifier_detail=ver.verifier_detail,
        )
        for ver, inf, mdl, verifier_mdl in rows.all()
    ]
    return VerificationEventViewPage(
        items=items, total=total or 0, limit=limit, offset=offset
    )


@router.get("/stats", response_model=VerificationStats)
async def stats(
    model: str | None = None,
    search: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Aggregate tiles for the UI. Accepts the view's model/search filters —
    but never `result`: the pass/fail/unverifiable breakdown IS the stats."""
    clauses = _view_filters(None, model, search)
    row = (
        await session.execute(
            _joined(
                select(
                    func.count(VerificationEvent.id),
                    func.count(VerificationEvent.id).filter(
                        VerificationEvent.result == "pass"
                    ),
                    func.count(VerificationEvent.id).filter(
                        VerificationEvent.result == "fail"
                    ),
                    func.count(VerificationEvent.id).filter(
                        VerificationEvent.result == "unverifiable"
                    ),
                    func.avg(VerificationEvent.exact_match_level_pct),
                )
            ).where(*clauses)
        )
    ).one()
    return VerificationStats(
        total=row[0],
        pass_count=row[1],
        fail_count=row[2],
        unverifiable_count=row[3],
        average_match_level=float(row[4]) if row[4] is not None else None,
    )


async def _delete_orphaned_inference_events(
    session: AsyncSession, inference_ids: set[uuid.UUID]
) -> None:
    """Delete the given inference events unless another verification event
    still references them (e.g. a second verification of the same event
    that did not match the delete filter)."""
    if not inference_ids:
        return
    still_referenced = set(
        (
            await session.execute(
                select(VerificationEvent.inference_event_id).where(
                    VerificationEvent.inference_event_id.in_(inference_ids)
                )
            )
        )
        .scalars()
        .all()
    )
    orphaned = inference_ids - still_referenced
    if orphaned:
        await session.execute(
            delete(InferenceEvent).where(InferenceEvent.id.in_(orphaned))
        )


@router.post("/replay")
async def replay_all(
    result: str | None = None,
    model: str | None = None,
    search: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Re-verify every event matching the filters: delete only the
    verification events; their inference events then look unverified again
    and the runners pick them back up. The original results are not kept."""
    clauses = _view_filters(result, model, search)
    matching_ids = _joined(select(VerificationEvent.id)).where(*clauses).scalar_subquery()
    replayed = (
        await session.execute(
            delete(VerificationEvent).where(VerificationEvent.id.in_(matching_ids))
        )
    ).rowcount
    return {"replayed": replayed}


@router.post("/{item_id}/replay", status_code=status.HTTP_204_NO_CONTENT)
async def replay_one(item_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """Re-verify one event (delete only its verification event)."""
    obj = await session.get(VerificationEvent, item_id)
    if obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="verification_event not found"
        )
    await session.delete(obj)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("")
async def delete_all(
    result: str | None = None,
    model: str | None = None,
    search: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """COMPLETE delete of every verification event matching the filters
    (all of them when no filters are given): the underlying inference
    events go too, so nothing re-queues. Use the replay endpoints for the
    old delete-to-re-verify behaviour."""
    clauses = _view_filters(result, model, search)
    matching_ids = _joined(select(VerificationEvent.id)).where(*clauses).scalar_subquery()
    deleted = (
        await session.execute(
            delete(VerificationEvent)
            .where(VerificationEvent.id.in_(matching_ids))
            .returning(VerificationEvent.inference_event_id)
        )
    ).scalars().all()
    await _delete_orphaned_inference_events(session, set(deleted))
    return {"deleted": len(deleted)}


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_one(item_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """COMPLETE delete of one verification event and (unless another
    verification event references it) its inference event."""
    obj = await session.get(VerificationEvent, item_id)
    if obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="verification_event not found"
        )
    inference_id = obj.inference_event_id
    await session.delete(obj)
    await session.flush()
    await _delete_orphaned_inference_events(session, {inference_id})
    return Response(status_code=status.HTTP_204_NO_CONTENT)
