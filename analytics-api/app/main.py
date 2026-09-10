"""analytics-api — read-only cross-domain reads (design ruling: the one
service allowed to SELECT across the ledger, enrichment_ and benchmarking_
table families; it never writes anything)."""

from fastapi import FastAPI
from sqlalchemy import text

from app.database import AsyncSessionLocal
from app.logging_setup import configure_logging
from app.routers import analysis, runs

configure_logging("analytics-api")

app = FastAPI(
    title="Inference Verification Analytics API",
    description="Read-only analysis, cohort and run-economics queries.",
    version="0.1.0",
)
app.include_router(analysis.router)
app.include_router(runs.router)


@app.get("/health")
async def health():
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ok"}
