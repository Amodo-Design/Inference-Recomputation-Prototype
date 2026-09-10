import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Model
from app.routers import make_crud_router
from app.schemas import ModelCreate, ModelDeltaMaxUpdate, ModelRead, ModelThresholdUpdate

router = make_crud_router(
    prefix="/models",
    tag="model",
    orm_model=Model,
    pk_attr="model_id",
    create_schema=ModelCreate,
    read_schema=ModelRead,
)


@router.patch("/{model_id}/verification-threshold", response_model=ModelRead)
async def set_verification_threshold(
    model_id: uuid.UUID,
    payload: ModelThresholdUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Set (or clear, with null) this model's verification pass mark.

    The threshold is a mutable operator setting, deliberately excluded from
    the model_id identity hash — updating it never creates a new model row.
    NULL pauses verification for the model: the orchestrator will not spawn
    runners for it, and its pending events wait (they verify retroactively
    once a threshold is set).
    """
    model = await session.get(Model, model_id)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {model_id}",
        )
    model.verification_threshold = payload.verification_threshold
    await session.flush()
    await session.refresh(model)
    return model


@router.patch("/{model_id}/delta-max", response_model=ModelRead)
async def set_delta_max(
    model_id: uuid.UUID,
    payload: ModelDeltaMaxUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Set (or clear, with null) this model's margin cap (difr delta max).

    Per-token logit-difference margins are clipped to this value by the
    runner, and it substitutes for the infinite margin when the prover's
    token falls outside the verifier's reconstructed candidate set. Like the
    threshold it is mutable and outside the identity hash; NULL means the
    runner's built-in default (10.0).
    """
    model = await session.get(Model, model_id)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {model_id}",
        )
    model.delta_max = payload.delta_max
    await session.flush()
    await session.refresh(model)
    return model
