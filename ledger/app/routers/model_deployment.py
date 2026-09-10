import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.identity import derive_model_id
from app.models import Hardware, HardwareOwner, Model, ModelDeployment
from app.schemas import (
    ModelDeclarationRequest,
    ModelDeploymentCloseRequest,
    ModelDeploymentRead,
    ModelResolution,
)

router = APIRouter(prefix="/model-deployments", tags=["model_deployment"])


@router.post(
    "/declare",
    response_model=ModelDeploymentRead,
    status_code=status.HTTP_201_CREATED,
)
async def declare(
    payload: ModelDeclarationRequest,
    session: AsyncSession = Depends(get_session),
):
    """Register a model start. The ledger owns all ids: model_id is derived
    from the config, hardware is matched by hostname, deployment_id is
    generated. Closes the host's active deployment and opens a new one."""

    started_at = payload.started_at or datetime.datetime.now(datetime.timezone.utc)

    try:
        # 1. Hardware: match by unique hostname, otherwise create.
        hardware = (
            await session.execute(
                select(Hardware).where(Hardware.hostname == payload.hostname)
            )
        ).scalar_one_or_none()
        if hardware is None:
            hardware = Hardware(
                hardware_id=uuid.uuid4(),
                hostname=payload.hostname,
                gpu_product_id=payload.gpu_product_id,
                cpu_product_id=payload.cpu_product_id,
                owner_id=payload.owner_id,
                gpu_firmware_version=payload.gpu_firmware_version,
            )
            session.add(hardware)
            await session.flush()

        # 1b. Owner by name (convenience for self-declaring processes: the
        #     verify-taps send "prover", the runners "verifier"). Upserted by
        #     organisation_name and linked to the hardware.
        if payload.owner_name:
            owner = (
                await session.execute(
                    select(HardwareOwner).where(
                        HardwareOwner.organisation_name == payload.owner_name
                    )
                )
            ).scalar_one_or_none()
            if owner is None:
                owner = HardwareOwner(
                    owner_id=uuid.uuid4(),
                    organisation_name=payload.owner_name,
                    is_trusted=False,
                )
                session.add(owner)
                await session.flush()
            hardware.owner_id = owner.owner_id

        # 2. Model config: id is derived from the config, so an identical config
        #    reuses the existing row ("do we already have this model?" for free).
        model_id = derive_model_id(
            model_name=payload.model_name,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
            seed=payload.seed,
            decoding_algorithm=payload.decoding_algorithm,
        )
        model = await session.get(Model, model_id)
        if model is None:
            model = Model(
                model_id=model_id,
                model_name=payload.model_name,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                seed=payload.seed,
                decoding_algorithm=payload.decoding_algorithm,
            )
            session.add(model)
            await session.flush()

        # 3. Close the host's currently-active deployment (if any).
        await session.execute(
            update(ModelDeployment)
            .where(
                ModelDeployment.hardware_id == hardware.hardware_id,
                ModelDeployment.ended_at.is_(None),
            )
            .values(ended_at=started_at)
        )

        # 4. Open the new deployment.
        deployment = ModelDeployment(
            deployment_id=uuid.uuid4(),
            model_id=model.model_id,
            hardware_id=hardware.hardware_id,
            started_at=started_at,
            ended_at=None,
        )
        session.add(deployment)
        await session.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Could not declare deployment: {exc.orig}",
        ) from exc

    await session.refresh(deployment)
    return deployment


@router.get("/resolve", response_model=ModelResolution)
async def resolve(
    hostname: str = Query(..., description="Destination hostname seen by the proxy"),
    ts: datetime.datetime = Query(..., description="Inference timestamp (ISO8601)"),
    model_name: str | None = Query(
        None, description="Model name seen by the proxy, for cross-check"
    ),
    session: AsyncSession = Depends(get_session),
):
    """Resolve a tapped inference event to (model_id, hardware_id) via the
    deployment active on `hostname` at `ts`."""

    hardware = (
        await session.execute(
            select(Hardware).where(Hardware.hostname == hostname)
        )
    ).scalar_one_or_none()
    if hardware is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No hardware registered for hostname '{hostname}'",
        )

    deployment = (
        await session.execute(
            select(ModelDeployment)
            .where(
                ModelDeployment.hardware_id == hardware.hardware_id,
                ModelDeployment.started_at <= ts,
                or_(
                    ModelDeployment.ended_at.is_(None),
                    ModelDeployment.ended_at > ts,
                ),
            )
            .order_by(ModelDeployment.started_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active model deployment on '{hostname}' at {ts.isoformat()}",
        )

    model = await session.get(Model, deployment.model_id)
    return ModelResolution(
        deployment_id=deployment.deployment_id,
        model_id=deployment.model_id,
        hardware_id=hardware.hardware_id,
        model_name=model.model_name,
        model_name_matches=(model_name is None or model.model_name == model_name),
    )


@router.get("", response_model=list[ModelDeploymentRead])
async def list_deployments(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(ModelDeployment)
        .order_by(ModelDeployment.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


@router.get("/{deployment_id}", response_model=ModelDeploymentRead)
async def get_deployment(
    deployment_id: str,
    session: AsyncSession = Depends(get_session),
):
    deployment = await session.get(ModelDeployment, deployment_id)
    if deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="model_deployment not found",
        )
    return deployment


@router.post("/close", response_model=ModelDeploymentRead)
async def close(
    payload: ModelDeploymentCloseRequest,
    session: AsyncSession = Depends(get_session),
):
    """Register a model stop: sets ended_at on the host's ACTIVE deployment.

    Mirrors declare's hostname resolution. Best-effort callers (taps on pod
    shutdown, runners on exit) get a 404 if the host is unknown or nothing is
    active — e.g. a re-declare already closed it — which they may ignore.
    """
    hardware = (
        await session.execute(
            select(Hardware).where(Hardware.hostname == payload.hostname)
        )
    ).scalar_one_or_none()
    if hardware is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No hardware with hostname {payload.hostname}",
        )
    deployment = (
        await session.execute(
            select(ModelDeployment).where(
                ModelDeployment.hardware_id == hardware.hardware_id,
                ModelDeployment.ended_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active deployment on {payload.hostname}",
        )
    deployment.ended_at = payload.ended_at or datetime.datetime.now(
        datetime.timezone.utc
    )
    await session.flush()
    await session.refresh(deployment)
    return deployment
