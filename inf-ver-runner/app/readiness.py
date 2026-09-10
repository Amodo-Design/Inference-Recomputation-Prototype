"""Startup gate for the paired vLLM (kept free of torch imports so tests
can exercise it without pulling in app.verifier)."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .config import Settings

log = logging.getLogger(__name__)


async def wait_for_vllm(settings: Settings) -> bool:
    """Block until the paired vLLM answers /models, or the timeout passes.

    A runner Job is spawned together with its vLLM, whose cold start (weight
    download + load) takes minutes. Consuming events before it is up would
    burn the per-event retry budget into wrong `unverifiable` verdicts, so
    nothing is fetched until the server is actually ready.
    """
    url = f"{settings.vllm_url.rstrip('/')}/models"
    headers = (
        {"Authorization": f"Bearer {settings.vllm_api_key}"}
        if settings.vllm_api_key
        else {}
    )
    deadline = time.monotonic() + settings.vllm_ready_timeout_seconds
    attempt = 0
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    log.info("vLLM ready at %s (attempt %s)", url, attempt)
                    return True
                reason: object = f"HTTP {resp.status_code}"
            except httpx.HTTPError as exc:
                reason = exc
            if attempt == 1 or attempt % 12 == 0:
                log.info(
                    "Waiting for vLLM at %s (attempt %s): %s", url, attempt, reason
                )
            await asyncio.sleep(settings.poll_interval_seconds)
    log.error(
        "vLLM at %s not ready after %.0fs",
        url,
        settings.vllm_ready_timeout_seconds,
    )
    return False
