"""Capture windows: frame-processor's account of the tapped link, one row per window.

Written explicitly rather than through the CRUD factory because a window and
its findings are one record and are written in one transaction, the row id is
derived from the window's natural key rather than supplied, and the tapped
node is resolved from a hostname the way inference events are.
"""

from __future__ import annotations

import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import CaptureFinding, CaptureWindow, Hardware, InferenceEvent
from app.schemas import (
    CaptureFindingRead,
    CaptureWindowCreate,
    CaptureWindowDetail,
    CaptureWindowPage,
    CaptureWindowRead,
    InferenceEventCapture,
)

router = APIRouter(prefix="/capture-windows", tags=["capture_window"])
findings_router = APIRouter(prefix="/capture-findings", tags=["capture_window"])
# Registered before the inference_event CRUD router in main.py, whose
# GET /{item_id} would otherwise not matter here (different depth) but keeps
# the literal-before-parameter convention the other routers follow.
capture_router = APIRouter(prefix="/inference-events", tags=["capture_window"])

# Window ids are uuid5 of the natural key, so the same window posted twice —
# a retry after a crash, a duplicate reporter — is one row, and the second
# post is a 409 the reporter treats as delivered.
_WINDOW_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "infver.capture_window")


def window_id_for(capture_host: str, ifaces: str, window_start: datetime.datetime) -> uuid.UUID:
    start = window_start.astimezone(datetime.timezone.utc).isoformat()
    return uuid.uuid5(_WINDOW_NAMESPACE, f"{capture_host}|{ifaces}|{start}")


@router.post("", response_model=CaptureWindowRead, status_code=status.HTTP_201_CREATED)
async def create_window(
    payload: CaptureWindowCreate,
    session: AsyncSession = Depends(get_session),
):
    """Record one window and its findings. 409 if the window is already there."""
    if payload.window_end <= payload.window_start:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="window_end must be after window_start",
        )
    window_id = window_id_for(payload.capture_host, payload.ifaces, payload.window_start)
    if await session.get(CaptureWindow, window_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"capture window {window_id} already recorded",
        )

    # The tapped node, by the hostname its deployment declared. Unknown is
    # allowed: the window is still evidence about the link, it just cannot be
    # joined to inference events until the hostname is declared.
    hardware_id = None
    if payload.tapped_hostname:
        hardware_id = (
            await session.execute(
                select(Hardware.hardware_id).where(Hardware.hostname == payload.tapped_hostname)
            )
        ).scalar_one_or_none()

    window = CaptureWindow(
        window_id=window_id,
        capture_host=payload.capture_host,
        ifaces=payload.ifaces,
        hardware_id=hardware_id,
        window_start=payload.window_start,
        window_end=payload.window_end,
        tap_version=payload.tap_version,
        process_epoch=payload.process_epoch,
        observed=payload.observed,
        classified=payload.classified,
        kernel_dropped=payload.kernel_dropped,
        truncated=payload.truncated,
        errors=payload.errors,
        finding_count=payload.finding_count,
        finding_groups=payload.finding_groups,
        finding_groups_overflow=payload.finding_groups_overflow,
        complete=payload.complete,
        classes=payload.classes,
        pins=payload.pins,
    )
    session.add(window)
    try:
        # The window row first: no relationship() ties the two models, so the
        # unit of work would not otherwise know the findings depend on it.
        await session.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Could not record capture window: {exc.orig}",
        ) from exc
    for finding in payload.findings:
        session.add(
            CaptureFinding(
                finding_id=uuid.uuid4(),
                window_id=window_id,
                kind=finding.kind,
                frame_class=finding.frame_class,
                direction=finding.direction,
                source_mac=finding.source_mac,
                count=finding.count,
                first_ts=finding.first_ts,
                last_ts=finding.last_ts,
                samples=finding.samples,
            )
        )
    try:
        await session.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Could not record capture window: {exc.orig}",
        ) from exc
    await session.refresh(window)
    return window


@router.get("", response_model=CaptureWindowPage)
async def list_windows(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    capture_host: str | None = Query(None),
    complete: bool | None = Query(None, description="filter on the complete flag"),
    since: datetime.datetime | None = Query(None, description="window_end >= since"),
    until: datetime.datetime | None = Query(None, description="window_start <= until"),
    session: AsyncSession = Depends(get_session),
):
    """Windows newest first, with the filters the UI needs.

    ``since`` and ``until`` together select the windows overlapping a span —
    an inference's started_at..ts — which is how the UI lands on exactly the
    windows that decided an event's capture status.
    """
    stmt = select(CaptureWindow)
    count_stmt = select(func.count(CaptureWindow.window_id))
    if capture_host:
        stmt = stmt.where(CaptureWindow.capture_host == capture_host)
        count_stmt = count_stmt.where(CaptureWindow.capture_host == capture_host)
    if complete is not None:
        stmt = stmt.where(CaptureWindow.complete.is_(complete))
        count_stmt = count_stmt.where(CaptureWindow.complete.is_(complete))
    if since is not None:
        stmt = stmt.where(CaptureWindow.window_end >= since)
        count_stmt = count_stmt.where(CaptureWindow.window_end >= since)
    if until is not None:
        stmt = stmt.where(CaptureWindow.window_start <= until)
        count_stmt = count_stmt.where(CaptureWindow.window_start <= until)
    total = await session.scalar(count_stmt)
    rows = await session.execute(
        stmt.order_by(CaptureWindow.window_start.desc()).limit(limit).offset(offset)
    )
    return CaptureWindowPage(
        items=[CaptureWindowRead.model_validate(w) for w in rows.scalars().all()],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/latest", response_model=CaptureWindowRead)
async def latest_window(
    capture_host: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """The newest window: what a restarting tap seeds its pins from, and
    what an alert compares against the clock — a latest window older than a
    few window lengths means the tap is not running."""
    stmt = select(CaptureWindow)
    if capture_host:
        stmt = stmt.where(CaptureWindow.capture_host == capture_host)
    window = (
        await session.execute(stmt.order_by(CaptureWindow.window_start.desc()).limit(1))
    ).scalar_one_or_none()
    if window is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no capture windows")
    return window


@router.get("/{window_id}", response_model=CaptureWindowDetail)
async def get_window(
    window_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    window = await session.get(CaptureWindow, window_id)
    if window is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="capture window not found")
    findings = (
        await session.execute(
            select(CaptureFinding)
            .where(CaptureFinding.window_id == window_id)
            .order_by(CaptureFinding.count.desc())
        )
    ).scalars().all()
    detail = CaptureWindowDetail.model_validate(
        {
            **CaptureWindowRead.model_validate(window).model_dump(),
            "findings": [CaptureFindingRead.model_validate(f) for f in findings],
        }
    )
    return detail


@findings_router.get("", response_model=list[CaptureFindingRead])
async def list_findings(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    kind: str | None = Query(None),
    frame_class: str | None = Query(None),
    since: datetime.datetime | None = Query(None, description="last_ts >= since"),
    session: AsyncSession = Depends(get_session),
):
    """Findings across windows, newest first — the non-whitelisted traffic
    report."""
    stmt = select(CaptureFinding)
    if kind:
        stmt = stmt.where(CaptureFinding.kind == kind)
    if frame_class:
        stmt = stmt.where(CaptureFinding.frame_class == frame_class)
    if since is not None:
        stmt = stmt.where(CaptureFinding.last_ts >= since)
    rows = await session.execute(
        stmt.order_by(CaptureFinding.last_ts.desc()).limit(limit).offset(offset)
    )
    return list(rows.scalars().all())


@capture_router.get("/{item_id}/capture", response_model=InferenceEventCapture)
async def inference_event_capture(
    item_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Was the link fully accounted for while this inference crossed it."""
    if await session.get(InferenceEvent, item_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="inference_event not found")
    row = (
        await session.execute(
            text(
                "SELECT inference_event_id, capture_status, windows, span_seconds, "
                "covered_seconds, kernel_dropped, findings "
                "FROM inference_event_capture WHERE inference_event_id = :id"
            ),
            {"id": item_id},
        )
    ).mappings().one()
    return InferenceEventCapture(
        inference_event_id=row["inference_event_id"],
        capture_status=row["capture_status"],
        windows=int(row["windows"]),
        span_seconds=float(row["span_seconds"]),
        covered_seconds=float(row["covered_seconds"]),
        kernel_dropped=int(row["kernel_dropped"]),
        findings=int(row["findings"]),
    )
