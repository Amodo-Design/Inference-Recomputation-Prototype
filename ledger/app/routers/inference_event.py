import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import InferenceEvent, Model, VerificationEvent
from app.routers import make_crud_router
from app.schemas import (
    InferenceEventCreate,
    InferenceEventRead,
    ModelRead,
    UnverifiedInferenceEvent,
    UnverifiedInferenceEventView,
    UnverifiedInferenceEventViewPage,
)

# Literal sub-paths must be registered before the CRUD router, whose
# GET /{item_id} would otherwise swallow them (see main.py include order).
extra_router = APIRouter(prefix="/inference-events", tags=["inference_event"])


def _unverified(
    stmt,
    model_id: uuid.UUID | None = None,
    model_name: str | None = None,
):
    """Anti-join: inference events with no verification_event row yet,
    optionally scoped to one model — by id (a runner polls only its own
    model) or by name pattern (the UI's model filter)."""
    stmt = (
        stmt.join(Model, InferenceEvent.model_id == Model.model_id)
        .outerjoin(
            VerificationEvent,
            VerificationEvent.inference_event_id == InferenceEvent.id,
        )
        .where(VerificationEvent.id.is_(None))
    )
    if model_id is not None:
        stmt = stmt.where(InferenceEvent.model_id == model_id)
    if model_name:
        stmt = stmt.where(Model.model_name.ilike(f"%{model_name}%"))
    return stmt


@extra_router.get("/unverified/view", response_model=UnverifiedInferenceEventViewPage)
async def view_unverified(
    limit: int = Query(25, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    model_id: uuid.UUID | None = Query(None),
    model: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """Paginated awaiting-verification feed for the UI — unlike /unverified it
    skips the raw-logit payloads, so it is safe to page through at any size."""
    total = await session.scalar(
        _unverified(select(func.count(InferenceEvent.id)), model_id, model)
    )
    rows = await session.execute(
        _unverified(select(InferenceEvent, Model), model_id, model)
        .order_by(InferenceEvent.ts.desc())
        .limit(limit)
        .offset(offset)
    )
    items = [
        UnverifiedInferenceEventView(
            id=event.id,
            session_id=event.session_id,
            ts=event.ts,
            model_id=event.model_id,
            model_name=model.model_name,
            sampling_config={
                "temperature": model.temperature,
                "top_k": model.top_k,
                "top_p": model.top_p,
                "seed": model.seed,
                "decoding_algorithm": model.decoding_algorithm,
            },
            verification_threshold=model.verification_threshold,
            input_text_representation=event.input_text_representation,
            output_text_representation=event.output_text_representation,
            hardware_id=event.hardware_id,
        )
        for event, model in rows.all()
    ]
    return UnverifiedInferenceEventViewPage(
        items=items, total=total or 0, limit=limit, offset=offset
    )


@extra_router.get("/unverified", response_model=list[UnverifiedInferenceEvent])
async def list_unverified(
    limit: int = Query(10, ge=1, le=100),
    model_id: uuid.UUID | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """Inference events with no verification_event row yet, newest first.

    Each item carries the joined model row so the verifier gets the sampling
    config and model name in the same call. Raw-logit payloads ride along as
    base64, so keep `limit` small.
    """
    result = await session.execute(
        _unverified(select(InferenceEvent, Model), model_id)
        .order_by(InferenceEvent.ts.desc())
        .limit(limit)
    )
    return [
        UnverifiedInferenceEvent(
            event=InferenceEventRead.model_validate(event),
            model=ModelRead.model_validate(model),
        )
        for event, model in result.all()
    ]


router = make_crud_router(
    prefix="/inference-events",
    tag="inference_event",
    orm_model=InferenceEvent,
    pk_attr="id",
    create_schema=InferenceEventCreate,
    read_schema=InferenceEventRead,
)
