from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.logging_setup import configure_logging
from app.routers import (
    hardware,
    hardware_owner,
    inference_event,
    model,
    model_deployment,
    verification_event,
    verification_view,
)

configure_logging("ledger")

app = FastAPI(
    title="Verification Ledger API",
    description="Read + Create access layer over the verification ledger schema.",
    version="0.1.0",
)

app.include_router(hardware_owner.router)
app.include_router(hardware.router)
app.include_router(model.router)
app.include_router(model_deployment.router)
# Literal sub-paths (/unverified, /view, /stats) must be registered before the
# CRUD routers, whose GET /{item_id} would otherwise capture them.
app.include_router(inference_event.extra_router)
app.include_router(inference_event.router)
app.include_router(verification_view.router)
app.include_router(verification_event.router)


@app.get("/health", tags=["health"])
async def health(session: AsyncSession = Depends(get_session)):
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}
