"""gpu-enricher — GPU measurement service.

Owns enrichment_gpu_activity (startup DDL creates it) and the DCGM/
Prometheus enrichment poller moved out of ledger-api; Task 2 adds the
domain-neutral pod-window aggregate endpoint. Talks to the shared Postgres
directly (design ruling): reads the event tables, writes only its own table.
"""

import asyncio
import pathlib

from fastapi import FastAPI
from sqlalchemy import text

from app.config import settings
from app.database import AsyncSessionLocal, engine
from app.logging_setup import configure_logging
from app.window import router as window_router

configure_logging("gpu-enricher")

DDL_PATH = pathlib.Path(__file__).resolve().parent.parent / "sql" / "enrichment_gpu_activity.sql"

app = FastAPI(
    title="Inference Verification GPU Enricher",
    description="DCGM/Prometheus GPU measurement: per-event enrichment poller + pod-window aggregates.",
    version="0.1.0",
)

app.include_router(window_router)


@app.get("/health")
async def health():
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.on_event("startup")
async def startup() -> None:
    async with engine.begin() as conn:
        await conn.exec_driver_sql(DDL_PATH.read_text())
    if not settings.prometheus_url:
        return
    from app.poller import run_poller
    from app.prometheus import PrometheusClient

    app.state.prom = PrometheusClient(settings.prometheus_url)
    app.state.poller = asyncio.create_task(
        run_poller(
            AsyncSessionLocal,
            app.state.prom,
            settings.gpu_poll_interval_seconds,
            settings.gpu_enrich_delay_seconds,
            settings.gpu_poll_batch,
        )
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "poller", None)
    if task is not None:
        task.cancel()
    prom = getattr(app.state, "prom", None)
    if prom is not None:
        await prom.aclose()
