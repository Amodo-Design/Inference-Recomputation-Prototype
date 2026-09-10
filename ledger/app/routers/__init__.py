from typing import Type

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, get_session


def make_crud_router(
    *,
    prefix: str,
    tag: str,
    orm_model: Type[Base],
    pk_attr: str,
    create_schema: Type[BaseModel],
    read_schema: Type[BaseModel],
) -> APIRouter:
    """Build a router exposing Create + Read (list / get-by-id) for a table."""

    router = APIRouter(prefix=prefix, tags=[tag])
    pk_column = getattr(orm_model, pk_attr)

    @router.post("", response_model=read_schema, status_code=status.HTTP_201_CREATED)
    async def create(
        payload: create_schema,  # type: ignore[valid-type]
        session: AsyncSession = Depends(get_session),
    ):
        obj = orm_model(**payload.model_dump())
        session.add(obj)
        try:
            await session.flush()
        except IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Could not create {tag}: {exc.orig}",
            ) from exc
        await session.refresh(obj)
        return obj

    @router.get("", response_model=list[read_schema])
    async def list_items(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        session: AsyncSession = Depends(get_session),
    ):
        result = await session.execute(
            select(orm_model).order_by(pk_column).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    @router.get("/{item_id}", response_model=read_schema)
    async def get_item(
        item_id: str,
        session: AsyncSession = Depends(get_session),
    ):
        obj = await session.get(orm_model, item_id)
        if obj is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{tag} not found",
            )
        return obj

    return router
