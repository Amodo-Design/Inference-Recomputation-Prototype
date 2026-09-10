"""Thin async client for the Ledger Access Layer."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import LEDGER_API_URL, LEDGER_TIMEOUT

log = logging.getLogger("message_writer.ledger")

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=LEDGER_API_URL, timeout=LEDGER_TIMEOUT)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


async def resolve(hostname: str, ts: str, model_name: str | None) -> dict[str, Any] | None:
    """Resolve (hostname, ts) to model_id + hardware_id via the ledger.

    Returns the resolution dict, or None if no deployment matched (404)."""
    params = {"hostname": hostname, "ts": ts}
    if model_name:
        params["model_name"] = model_name
    resp = await get_client().get("/model-deployments/resolve", params=params)
    if resp.status_code == 404:
        log.warning("resolve 404: %s", resp.text[:300])
        return None
    resp.raise_for_status()
    return resp.json()


async def create_inference_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Write an inference_event; raises on a non-2xx ledger response."""
    resp = await get_client().post("/inference-events", json=payload)
    resp.raise_for_status()
    return resp.json()
