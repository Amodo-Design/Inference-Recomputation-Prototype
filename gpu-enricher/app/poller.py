"""Background GPU-activity enrichment.

Every cycle: find events (both tables) that have a started_at, finished
longer than the settle delay ago, and have no gpu_activity row yet; run one
bounded Prometheus range query per metric over exactly [started_at, ts];
integrate; write the row. Events with zero samples still get a row
(sample_count=0) so they are never re-queried — the UI excludes them at
display time. Selector precedence: exact pod > node captured on the event >
legacy hardware.hostname (unreliable for prover rows, whose hostname is a
deployment-name string — recorded anyway, flagged by its low sample count).

Runs only when PROMETHEUS_URL is set; docker-compose never starts it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EnrichmentGpuActivity, Hardware, InferenceEvent, VerificationEvent
from app.prometheus import (
    HEADLINE_METRIC,
    METRICS,
    OCCUPANCY_METRIC,
    PrometheusClient,
    integrate,
    mean,
)

log = logging.getLogger("gpu_enricher.poller")

_EVENT_TABLES = (("inference", InferenceEvent), ("verification", VerificationEvent))


def _selector(event, hostname: str | None) -> str | None:
    if event.pod_name:
        return f'{{pod="{event.pod_name}"}}'
    if event.node_name:
        return f'{{node="{event.node_name}"}}'
    if hostname:
        return f'{{node="{hostname}"}}'
    return None


async def _hostname(session: AsyncSession, hardware_id) -> str | None:
    return await session.scalar(
        select(Hardware.hostname).where(Hardware.hardware_id == hardware_id)
    )


async def _count_overlaps(session: AsyncSession, event, event_type: str) -> int | None:
    """Events in either table overlapping this one's window on the same pod
    (node when this event has no pod). None when there is nothing to match on."""
    if event.pod_name:
        attr = "pod_name"
        value = event.pod_name
    elif event.node_name:
        attr = "node_name"
        value = event.node_name
    else:
        return None
    total = 0
    for other_type, table in _EVENT_TABLES:
        clauses = [
            getattr(table, attr) == value,
            table.started_at.is_not(None),
            table.started_at < event.ts,
            table.ts > event.started_at,
        ]
        if other_type == event_type:
            clauses.append(table.id != event.id)
        total += await session.scalar(
            select(func.count(table.id)).where(*clauses)
        ) or 0
    return total


async def _enrich_event(
    session: AsyncSession, prom: PrometheusClient, event, event_type: str
) -> int:
    """Enrich one event; returns 1 if a row was actually inserted, 0 if the
    insert-or-ignore hit an existing row (switchover overlap with another
    enricher writing the same key concurrently)."""
    hostname = await _hostname(session, event.hardware_id)
    selector = _selector(event, hostname)
    window_s = max(0.0, (event.ts - event.started_at).total_seconds())

    tensor_active = None
    occupancy_mean = None
    pipe: dict[str, float] = {}
    sample_count = 0
    if selector is not None:
        for metric in METRICS:
            values = await prom.range_values(
                metric, selector, event.started_at, event.ts
            )
            if metric == HEADLINE_METRIC:
                sample_count = len(values)
                tensor_active = integrate(values)
            elif metric == OCCUPANCY_METRIC:
                occupancy_mean = mean(values)
            else:
                busy = integrate(values)
                if busy is not None:
                    pipe[metric] = busy

    stmt = (
        pg_insert(EnrichmentGpuActivity)
        .values(
            event_type=event_type,
            event_id=event.id,
            window_s=window_s,
            sample_count=sample_count,
            tensor_active_time_s=tensor_active,
            sm_occupancy_mean=occupancy_mean,
            pipe_activity_s=pipe or None,
            concurrent_events=await _count_overlaps(session, event, event_type),
            enriched_at=datetime.now(timezone.utc),
        )
        .on_conflict_do_nothing(index_elements=["event_type", "event_id"])
        # rowcount is unreliable for INSERT ... ON CONFLICT DO NOTHING under
        # the async psycopg driver (it reports -1); RETURNING the primary
        # key and counting the rows actually handed back is the reliable
        # way to tell "inserted" from "conflict skipped".
        .returning(EnrichmentGpuActivity.event_id)
    )
    result = await session.execute(stmt)
    return len(result.fetchall())


async def enrich_once(
    session: AsyncSession,
    prom: PrometheusClient,
    now: datetime,
    delay_s: float,
    batch: int,
) -> int:
    """One enrichment pass; returns how many gpu_activity rows were written."""
    written = 0
    cutoff = now - timedelta(seconds=delay_s)
    for event_type, table in _EVENT_TABLES:
        enriched = select(EnrichmentGpuActivity.event_id).where(
            EnrichmentGpuActivity.event_type == event_type
        )
        events = (
            (
                await session.execute(
                    select(table)
                    .where(
                        table.started_at.is_not(None),
                        table.ts <= cutoff,
                        table.id.not_in(enriched),
                    )
                    .order_by(table.ts)
                    .limit(batch)
                )
            )
            .scalars()
            .all()
        )
        for event in events:
            written += await _enrich_event(session, prom, event, event_type)
    await session.commit()
    return written


async def run_poller(
    session_factory,
    prom: PrometheusClient,
    interval_s: float,
    delay_s: float,
    batch: int,
) -> None:
    log.info("GPU enrichment poller started (interval=%ss)", interval_s)
    while True:
        try:
            async with session_factory() as session:
                written = await enrich_once(
                    session, prom, datetime.now(timezone.utc), delay_s, batch
                )
            if written:
                log.info("Enriched %s events", written)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Prometheus flaps or DB hiccups must not kill the loop.
            log.exception("GPU enrichment cycle failed; will retry")
        await asyncio.sleep(interval_s)
