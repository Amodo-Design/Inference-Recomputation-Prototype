"""Test harness: an ephemeral Postgres (testcontainers) loaded with the real
schema, with the app's `get_session` dependency pointed at it.

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

from app.database import get_session
from app.main import app

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
TABLES = "inference_event, verification_event, model_deployment, model, hardware, hardware_owner"


async def _load_schema(engine) -> None:
    # Same order Postgres applies the init scripts in: sorted by filename.
    async with engine.begin() as conn:
        for script in sorted(SQL_DIR.glob("*.sql")):
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
