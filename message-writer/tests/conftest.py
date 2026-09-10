"""Test fixtures. No database or ledger needed — the ledger boundary
(`resolve` / `create_inference_event`) is monkeypatched in the endpoint tests,
and the transform is pure."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
