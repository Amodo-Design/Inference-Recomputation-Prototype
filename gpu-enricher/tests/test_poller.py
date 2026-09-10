"""Enrichment poller: selection (started_at set, past the delay, not yet
enriched), pod->node->hostname selector fallback, overlap counting, 0-sample
rows, and insert-or-ignore idempotency across switchover overlap.

The enricher's app.models maps only the columns the poller reads, so event
rows here are seeded with raw SQL against the full ledger-owned schema
(loaded into the test container by conftest) rather than via that ORM model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import EnrichmentGpuActivity, Hardware
from app.poller import _enrich_event, enrich_once

NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _clean_slate(engine):
    # The engine/database are session-scoped and shared with other test
    # modules, which may leave eligible-but-unenriched event rows behind.
    # enrich_once has no module boundary, so without this those leftover
    # rows would get swept up by this module's enrich_once calls and
    # pollute FakeProm.calls. Truncate once, before this module's tests
    # start; persistence *within* this module across its own tests is the
    # anti-join behaviour under test and must not be disturbed.
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE TABLE inference_event, verification_event, enrichment_gpu_activity"
            " RESTART IDENTITY CASCADE"
        )


class FakeProm:
    """Returns a fixed ratio series and records the selectors queried."""

    def __init__(self, values=None):
        self.values = values if values is not None else [0.5, 0.5, 0.5]
        self.calls: list[tuple[str, str]] = []

    async def range_values(self, metric, selector, start, end):
        self.calls.append((metric, selector))
        return self.values


def _ids():
    return uuid.uuid4(), uuid.uuid4()


async def _seed(session, *, pod=None, node=None, started_offset_s=-1.0):
    """Insert a minimal, valid inference_event row (plus its model/hardware
    parents) via raw SQL — app.models.InferenceEvent only maps the columns
    the poller reads, so it cannot satisfy the full table's NOT NULL
    columns (session_id, hash_*, model_id) on its own."""
    model_id, hw_id = _ids()
    await session.execute(
        text("INSERT INTO model (model_id, model_name) VALUES (:id, :name)"),
        {"id": model_id, "name": f"m-{model_id.hex[:6]}"},
    )
    # hardware.hostname is globally unique and rows persist across tests
    # (session-scoped engine/database) — reuse the existing "gpu-node-1"
    # row instead of inserting a duplicate.
    existing_hw_id = await session.scalar(
        select(Hardware.hardware_id).where(Hardware.hostname == "gpu-node-1")
    )
    if existing_hw_id is not None:
        hw_id = existing_hw_id
    else:
        await session.execute(
            text("INSERT INTO hardware (hardware_id, hostname) VALUES (:id, :hostname)"),
            {"id": hw_id, "hostname": "gpu-node-1"},
        )
    event_id = uuid.uuid4()
    ts = NOW - timedelta(seconds=60)
    started_at = ts + timedelta(seconds=started_offset_s)
    await session.execute(
        text(
            "INSERT INTO inference_event "
            "(id, session_id, ts, started_at, model_id, hash_input_raw_logits, "
            "hash_output_raw_logits, hardware_id, pod_name, node_name) "
            "VALUES (:id, :session_id, :ts, :started_at, :model_id, :hash_in, "
            ":hash_out, :hardware_id, :pod, :node)"
        ),
        {
            "id": event_id,
            "session_id": "s",
            "ts": ts,
            "started_at": started_at,
            "model_id": model_id,
            "hash_in": b"h" * 32,
            "hash_out": b"h" * 32,
            "hardware_id": hw_id,
            "pod": pod,
            "node": node,
        },
    )
    await session.commit()
    return SimpleNamespace(
        id=event_id, ts=ts, started_at=started_at, pod_name=pod, node_name=node, hardware_id=hw_id
    )


async def _set_ts(session, event_id: uuid.UUID, *, ts=None, started_at=...) -> None:
    """Mutate a seeded event's timing columns (mirrors the original test's
    direct attribute assignment on the ORM instance)."""
    if ts is not None:
        await session.execute(
            text("UPDATE inference_event SET ts = :ts WHERE id = :id"),
            {"ts": ts, "id": event_id},
        )
    if started_at is not ...:
        await session.execute(
            text("UPDATE inference_event SET started_at = :started_at WHERE id = :id"),
            {"started_at": started_at, "id": event_id},
        )
    await session.commit()


@pytest.fixture
def sessionmaker(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


async def test_pod_selector_preferred(sessionmaker):
    prom = FakeProm()
    async with sessionmaker() as session:
        event = await _seed(session, pod="kserve-q-abc", node="gpu-node-1")
        written = await enrich_once(session, prom, now=NOW, delay_s=10.0, batch=10)
        assert written >= 1
        row = await session.get(EnrichmentGpuActivity, ("inference", event.id))
        assert row is not None
        assert row.sample_count == 3
        assert row.tensor_active_time_s == pytest.approx(0.3)
        assert row.concurrent_events == 0
    assert all('pod="kserve-q-abc"' in sel for _, sel in prom.calls)


async def test_node_fallback_then_hostname(sessionmaker):
    prom = FakeProm()
    async with sessionmaker() as session:
        event = await _seed(session, pod=None, node="gpu-node-2")
        await enrich_once(session, prom, now=NOW, delay_s=10.0, batch=10)
        assert (await session.get(EnrichmentGpuActivity, ("inference", event.id))) is not None
    assert all('node="gpu-node-2"' in sel for _, sel in prom.calls)

    prom2 = FakeProm()
    async with sessionmaker() as session:
        legacy = await _seed(session, pod=None, node=None)
        await enrich_once(session, prom2, now=NOW, delay_s=10.0, batch=10)
        assert (await session.get(EnrichmentGpuActivity, ("inference", legacy.id))) is not None
    assert all('node="gpu-node-1"' in sel for _, sel in prom2.calls)


async def test_zero_samples_still_writes_row(sessionmaker):
    prom = FakeProm(values=[])
    async with sessionmaker() as session:
        event = await _seed(session, pod="p1")
        await enrich_once(session, prom, now=NOW, delay_s=10.0, batch=10)
        row = await session.get(EnrichmentGpuActivity, ("inference", event.id))
        assert row.sample_count == 0
        assert row.tensor_active_time_s is None


async def test_respects_delay_and_skips_missing_started_at(sessionmaker):
    prom = FakeProm()
    async with sessionmaker() as session:
        # Too recent: ts within the delay window.
        recent = await _seed(session, pod="p2")
        await _set_ts(session, recent.id, ts=NOW - timedelta(seconds=1))
        # No started_at at all.
        no_start = await _seed(session, pod="p3")
        await _set_ts(session, no_start.id, started_at=None)
        await enrich_once(session, prom, now=NOW, delay_s=10.0, batch=10)
        assert (await session.get(EnrichmentGpuActivity, ("inference", recent.id))) is None
        assert (await session.get(EnrichmentGpuActivity, ("inference", no_start.id))) is None


async def test_concurrent_events_counted_on_same_pod(sessionmaker):
    prom = FakeProm()
    async with sessionmaker() as session:
        a = await _seed(session, pod="shared-pod")
        b = await _seed(session, pod="shared-pod")  # same window shape -> overlaps
        await enrich_once(session, prom, now=NOW, delay_s=10.0, batch=10)
        row_a = await session.get(EnrichmentGpuActivity, ("inference", a.id))
        row_b = await session.get(EnrichmentGpuActivity, ("inference", b.id))
        assert row_a.concurrent_events == 1
        assert row_b.concurrent_events == 1


async def test_enrich_is_insert_or_ignore(sessionmaker):
    """Seed one enrichable event, enrich it, then force the same primary key
    to be inserted again directly (simulating the old ledger poller and the
    new enricher both running during the switchover overlap): the second
    write must neither raise nor duplicate the row."""
    prom = FakeProm()
    async with sessionmaker() as session:
        event = await _seed(session, pod="switchover-pod")
        written = await enrich_once(session, prom, now=NOW, delay_s=10.0, batch=10)
        assert written == 1
        row = await session.get(EnrichmentGpuActivity, ("inference", event.id))
        assert row is not None

    # Re-enriching the same event directly (bypassing enrich_once's
    # anti-join, which would normally skip it) exercises the insert-or-ignore
    # conflict path: the poller's "Enriched N events" count must reflect
    # actual inserts, not attempts, so a second write of an already-enriched
    # event must report 0 rows written.
    async with sessionmaker() as session:
        rows_written = await _enrich_event(session, prom, event, "inference")
        await session.commit()
        assert rows_written == 0

    async with sessionmaker() as session:
        stmt = (
            pg_insert(EnrichmentGpuActivity)
            .values(
                event_type="inference",
                event_id=event.id,
                window_s=999.0,
                sample_count=999,
                tensor_active_time_s=None,
                sm_occupancy_mean=None,
                pipe_activity_s=None,
                concurrent_events=None,
                enriched_at=NOW,
            )
            .on_conflict_do_nothing(index_elements=["event_type", "event_id"])
        )
        await session.execute(stmt)
        await session.commit()

        count = await session.scalar(
            text(
                "SELECT count(*) FROM enrichment_gpu_activity"
                " WHERE event_type = 'inference' AND event_id = :id"
            ),
            {"id": event.id},
        )
        assert count == 1
        # The conflicting insert was ignored — original values still stand.
        row = await session.get(EnrichmentGpuActivity, ("inference", event.id))
        assert row.sample_count == 3
