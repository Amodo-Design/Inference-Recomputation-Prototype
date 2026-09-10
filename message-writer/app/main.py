"""Message writer.

Receives tap messages from the inf-proxy (the network-tap replacement),
resolves each to a ledger model_id/hardware_id via the deployment active on the
tapped hostname at the inference time, hashes the token-id/logprob payloads, and
writes an inference_event to the Ledger Access Layer.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException, status

from app.ledger_client import close_client, create_inference_event, resolve
from app.logging_setup import configure_logging
from app.schemas import TapMessage
from app.transform import build_inference_event

configure_logging("message-writer")
log = logging.getLogger("message_writer")

app = FastAPI(title="Inference Verification Message Writer", version="0.1.0")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await close_client()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/inferences", status_code=status.HTTP_201_CREATED)
async def ingest(msg: TapMessage):
    hostname = msg.tap.hostname
    ts = msg.tap.completed_at or msg.tap.received_at
    if not hostname or not ts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="tap message missing hostname or timestamp",
        )

    # Resolve the inference to a declared model deployment.
    try:
        resolution = await resolve(hostname, ts, msg.model.name)
    except httpx.HTTPError as exc:
        log.error("Ledger resolve failed (hostname=%s ts=%s): %s", hostname, ts, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ledger resolve failed: {exc}",
        ) from exc
    if resolution is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"no model deployment active on '{hostname}' at {ts}",
        )
    if resolution.get("model_name_matches") is False:
        log.warning(
            "model name mismatch: proxy saw %r, deployment declares %r (hostname=%s)",
            msg.model.name, resolution.get("model_name"), hostname,
        )

    event = build_inference_event(
        msg,
        model_id=resolution["model_id"],
        hardware_id=resolution["hardware_id"],
    )

    try:
        await create_inference_event(event)
    except httpx.HTTPError as exc:
        log.error(
            "Ledger write failed (event=%s session=%s): %s",
            event["id"], event["session_id"], exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ledger write failed: {exc}",
        ) from exc

    log.info(
        "Wrote inference_event %s (session=%s model_id=%s)",
        event["id"], event["session_id"], event["model_id"],
    )
    return {
        "inference_event_id": event["id"],
        "model_id": event["model_id"],
        "hardware_id": event["hardware_id"],
        "model_name_matches": resolution.get("model_name_matches"),
    }
