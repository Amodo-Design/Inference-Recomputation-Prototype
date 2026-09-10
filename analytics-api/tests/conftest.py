"""Test harness: an ephemeral Postgres (testcontainers) loaded with the
ledger's core event-table schema (read-only, ledger-owned), the
gpu-enricher's enrichment_gpu_activity DDL (read-only, gpu-enricher-owned),
and the benchmarking tables (read-only, prompt-runner-owned — needed by
Task 4's cohort/economics endpoints).

Uses NullPool so connections are never cached across event loops — the
session-scoped engine is safe to use from each function-scoped test loop.

analytics-api has no write routes, so seeding is done directly against the
schema via the helpers below rather than through CRUD posts.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

import app.main as main_module
from app.database import get_session
from app.main import app
from app.models import EnrichmentGpuActivity

# One repo, deliberate: the event tables are ledger-owned and the enrichment
# table is gpu-enricher-owned; this service only reads them, so the test
# schema is loaded from their own sql/ dirs (never modified by this service).
LEDGER_SQL_DIR = Path(__file__).resolve().parent.parent.parent / "ledger" / "sql"
ENRICHER_SQL_DIR = Path(__file__).resolve().parent.parent.parent / "gpu-enricher" / "sql"

# Benchmarking tables: copied from prompt-runner/app/db.py's DDL constant
# (prompt-runner is the sole writer; this service only reads them).
BENCHMARKING_DDL = """
CREATE TABLE IF NOT EXISTS benchmarking_run (
    id             TEXT PRIMARY KEY,
    state          TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    error          TEXT,
    settings       JSONB NOT NULL,
    total_requests INTEGER NOT NULL DEFAULT 0,
    completed      INTEGER NOT NULL DEFAULT 0,
    failed         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS benchmarking_run_result (
    run_id           TEXT NOT NULL REFERENCES benchmarking_run(id) ON DELETE CASCADE,
    request_index    INTEGER NOT NULL,
    prompt_id        TEXT NOT NULL,
    sampling_id      TEXT NOT NULL,
    model            TEXT NOT NULL,
    status_code      INTEGER,
    stream_completed BOOLEAN NOT NULL,
    elapsed_s        DOUBLE PRECISION NOT NULL,
    error            TEXT,
    PRIMARY KEY (run_id, request_index)
);
"""

TABLES = (
    "inference_event, verification_event, model_deployment, model, "
    "hardware, hardware_owner, enrichment_gpu_activity, "
    "benchmarking_run_result, benchmarking_run"
)


async def _load_schema(engine) -> None:
    async with engine.begin() as conn:
        # Same order Postgres applies init scripts in: sorted by filename.
        for script in sorted(LEDGER_SQL_DIR.glob("*.sql")):
            await conn.exec_driver_sql(script.read_text())
        for script in sorted(ENRICHER_SQL_DIR.glob("*.sql")):
            await conn.exec_driver_sql(script.read_text())
        await conn.exec_driver_sql(BENCHMARKING_DDL)


@pytest.fixture(scope="session")
def engine():
    with PostgresContainer(
        "postgres:16", username="ledger", password="ledger", dbname="ledger"
    ) as pg:
        host = pg.get_container_host_ip()
        port = pg.get_exposed_port(5432)
        url = f"postgresql+psycopg://ledger:ledger@{host}:{port}/ledger"
        eng = create_async_engine(url, poolclass=NullPool)
        asyncio.run(_load_schema(eng))

        TestSession = async_sessionmaker(eng, expire_on_commit=False)

        async def _override():
            async with TestSession() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_session] = _override
        # main.health() references these module globals directly (not via
        # Depends), so point the module's own bindings at the test container
        # too.
        main_module.engine = eng
        main_module.AsyncSessionLocal = TestSession
        try:
            yield eng
        finally:
            app.dependency_overrides.clear()
            asyncio.run(eng.dispose())


@pytest_asyncio.fixture
async def client(engine):
    # Isolate each test: wipe all rows before it runs.
    async with engine.begin() as conn:
        await conn.exec_driver_sql(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def session(engine):
    """A plain session against the test container, for direct seeding.
    Depend on `client` first in a test's parameter list so the truncate it
    performs runs before this session seeds any rows."""
    TestSession = async_sessionmaker(engine, expire_on_commit=False)
    async with TestSession() as s:
        yield s


_model_ids: dict[str, uuid.UUID] = {}
_hardware_id = uuid.uuid4()


async def _ensure_model(session: AsyncSession, model_name: str) -> uuid.UUID:
    if model_name not in _model_ids:
        _model_ids[model_name] = uuid.uuid4()
    model_id = _model_ids[model_name]
    await session.execute(
        text(
            "INSERT INTO model (model_id, model_name) VALUES (:id, :name) "
            "ON CONFLICT (model_id) DO NOTHING"
        ),
        {"id": model_id, "name": model_name},
    )
    return model_id


async def _ensure_hardware(session: AsyncSession) -> uuid.UUID:
    await session.execute(
        text(
            "INSERT INTO hardware (hardware_id, hostname) VALUES (:id, :hostname) "
            "ON CONFLICT (hardware_id) DO NOTHING"
        ),
        {"id": _hardware_id, "hostname": f"h-{_hardware_id}"},
    )
    return _hardware_id


async def seed_event_pair(
    session: AsyncSession,
    *,
    model_name: str = "m",
    inf_ts,
    ver_ts,
    inf_started=None,
    ver_started=None,
    pod_name: str | None = None,
    ver_pod_name: str | None = None,
    result: str = "pass",
    verifier_detail: dict | None = None,
) -> tuple[uuid.UUID, uuid.UUID]:
    """INSERT a model (idempotent), one inference_event and one
    verification_event; returns (inference_id, verification_id)."""
    model_id = await _ensure_model(session, model_name)
    hardware_id = await _ensure_hardware(session)

    inf_id = uuid.uuid4()
    ver_id = uuid.uuid4()

    await session.execute(
        text(
            """
            INSERT INTO inference_event (
                id, session_id, ts, started_at, model_id, hardware_id,
                hash_input_raw_logits, hash_output_raw_logits, pod_name
            ) VALUES (
                :id, :session_id, :ts, :started_at, :model_id, :hardware_id,
                :hash_in, :hash_out, :pod_name
            )
            """
        ),
        {
            "id": inf_id,
            "session_id": "s",
            "ts": inf_ts,
            "started_at": inf_started,
            "model_id": model_id,
            "hardware_id": hardware_id,
            "hash_in": b"\x00" * 32,
            "hash_out": b"\x00" * 32,
            "pod_name": pod_name,
        },
    )
    await session.execute(
        text(
            """
            INSERT INTO verification_event (
                id, inference_event_id, hardware_id, ts, started_at, result,
                verifier_model_id, verifier_detail, pod_name
            ) VALUES (
                :id, :inference_event_id, :hardware_id, :ts, :started_at, :result,
                :verifier_model_id, CAST(:verifier_detail AS JSONB), :pod_name
            )
            """
        ),
        {
            "id": ver_id,
            "inference_event_id": inf_id,
            "hardware_id": hardware_id,
            "ts": ver_ts,
            "started_at": ver_started,
            "result": result,
            "verifier_model_id": model_id,
            "verifier_detail": (
                None if verifier_detail is None else json.dumps(verifier_detail)
            ),
            "pod_name": ver_pod_name,
        },
    )
    await session.commit()
    return inf_id, ver_id


async def seed_enrichment_row(
    session: AsyncSession,
    event_id: uuid.UUID,
    *,
    event_type: str = "verification",
    window_s: float = 5.0,
    sample_count: int = 5,
    tensor_active_time_s: float | None = None,
    concurrent_events: int | None = None,
) -> None:
    """INSERT one enrichment_gpu_activity row (gpu-enricher-owned; this
    service only reads it) for the given inference/verification event id."""
    session.add(
        EnrichmentGpuActivity(
            event_type=event_type,
            event_id=event_id,
            window_s=window_s,
            sample_count=sample_count,
            tensor_active_time_s=tensor_active_time_s,
            concurrent_events=concurrent_events,
            enriched_at=datetime.datetime.now(datetime.timezone.utc),
        )
    )
    await session.commit()


async def seed_run(
    session: AsyncSession,
    run_id: str,
    *,
    state: str = "completed",
    started_at=None,
    finished_at=None,
    settings: dict | None = None,
) -> None:
    """INSERT one benchmarking_run row (Task 4's cohort/economics routes
    read this table; prompt-runner is its sole writer in production)."""
    await session.execute(
        text(
            """
            INSERT INTO benchmarking_run (
                id, state, created_at, started_at, finished_at, settings
            ) VALUES (
                :id, :state, now(), :started_at, :finished_at, CAST(:settings AS JSONB)
            )
            """
        ),
        {
            "id": run_id,
            "state": state,
            "started_at": started_at,
            "finished_at": finished_at,
            "settings": json.dumps(settings if settings is not None else {"models": []}),
        },
    )
    await session.commit()
