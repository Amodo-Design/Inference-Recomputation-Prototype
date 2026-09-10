"""Test harness: an ephemeral Postgres (testcontainers) loaded with the
ledger's core event-table schema (read-only, ledger-owned) plus this
service's own enrichment_gpu_activity DDL.

Uses NullPool so connections are never cached across event loops — the
session-scoped engine is safe to use from each function-scoped test loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

import app.main as main_module
from app.database import get_session
from app.main import app

# One repo, deliberate: the event tables are ledger-owned and the enricher
# only reads them, so the test schema is loaded from ledger/sql (never
# modified by this service).
LEDGER_SQL_DIR = Path(__file__).resolve().parent.parent.parent / "ledger" / "sql"
ENRICHER_SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

TABLES = (
    "inference_event, verification_event, model_deployment, model, "
    "hardware, hardware_owner, enrichment_gpu_activity"
)


async def _load_schema(engine) -> None:
    async with engine.begin() as conn:
        # Same order Postgres applies init scripts in: sorted by filename.
        for script in sorted(LEDGER_SQL_DIR.glob("*.sql")):
            await conn.exec_driver_sql(script.read_text())
        for script in sorted(ENRICHER_SQL_DIR.glob("*.sql")):
            await conn.exec_driver_sql(script.read_text())


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
        # main.health()/startup() reference these module globals directly
        # (not via Depends), so point the module's own bindings at the
        # test container too.
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
