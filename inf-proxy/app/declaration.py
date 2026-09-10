"""Startup self-declaration of the fronted model to the ledger.

The tap sits directly in front of the model, so it is the natural owner of
the deployment declaration (`POST /model-deployments/declare`): it knows the
ledger resolution hostname, it injects the sampling seed the declaration must
match, and its lifecycle *is* the model's lifecycle in Kubernetes (sidecar).

Declaration is best-effort and must never affect serving: it runs as a
background task, retries until the ledger accepts, and gives up quietly after
a bounded number of attempts (the tap keeps serving; events simply fail to
resolve until someone declares — exactly the pre-existing behaviour).

Re-declaration on restart is safe by design: the ledger derives model_id from
the config (identical config -> same row) and re-opens the host's deployment.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from app.config import (
    INF_PROXY_DECLARE,
    INF_PROXY_DECLARE_RETRY_SECONDS,
    INF_PROXY_DECODING_ALGORITHM,
    INF_PROXY_LEDGER_API_URL,
    INF_PROXY_MODEL_API_KEY,
    INF_PROXY_MODEL_BASE_URL,
    INF_PROXY_MODEL_NAME,
    INF_PROXY_OWNER_NAME,
    INF_PROXY_SEED,
    INF_PROXY_TEMPERATURE,
    INF_PROXY_TOP_K,
    INF_PROXY_TOP_P,
    MODEL_HOSTNAME,
)

log = logging.getLogger("proxy.declaration")

_DECLARE_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Strong reference to the background declaration task (see message_writer's
# _pending_taps for why bare create_task() results must be kept alive).
_declaration_task: asyncio.Task | None = None


def parse_model_name(models_body: Any) -> str | None:
    """First model id from an OpenAI `GET /v1/models` response, else None."""
    try:
        return models_body["data"][0]["id"] or None
    except (TypeError, KeyError, IndexError):
        return None


def build_declaration(model_name: str) -> dict[str, Any]:
    """Declaration payload. Only fields the tap actually knows are sent —
    the ledger derives model_id from the config, so absent params must be
    omitted rather than guessed."""
    payload: dict[str, Any] = {
        "hostname": MODEL_HOSTNAME,
        "model_name": model_name,
    }
    if INF_PROXY_SEED is not None:
        payload["seed"] = INF_PROXY_SEED
    if INF_PROXY_TEMPERATURE is not None:
        payload["temperature"] = INF_PROXY_TEMPERATURE
    if INF_PROXY_TOP_K is not None:
        payload["top_k"] = INF_PROXY_TOP_K
    if INF_PROXY_TOP_P is not None:
        payload["top_p"] = INF_PROXY_TOP_P
    if INF_PROXY_DECODING_ALGORITHM is not None:
        payload["decoding_algorithm"] = INF_PROXY_DECODING_ALGORITHM
    if INF_PROXY_OWNER_NAME:
        payload["owner_name"] = INF_PROXY_OWNER_NAME
    return payload


async def _discover_model_name(session: aiohttp.ClientSession) -> str | None:
    headers = {}
    if INF_PROXY_MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {INF_PROXY_MODEL_API_KEY}"
    async with session.get(
        f"{INF_PROXY_MODEL_BASE_URL}/models", headers=headers
    ) as resp:
        resp.raise_for_status()
        return parse_model_name(await resp.json())


async def _attempt_declaration() -> dict[str, Any]:
    """One declaration attempt: discover the model name (unless configured),
    then declare. Raises on any failure so the caller can retry."""
    async with aiohttp.ClientSession(timeout=_DECLARE_TIMEOUT) as session:
        model_name = INF_PROXY_MODEL_NAME or await _discover_model_name(session)
        if not model_name:
            raise RuntimeError(
                "could not discover model name from upstream /v1/models "
                "(set INF_PROXY_MODEL_NAME to skip discovery)"
            )
        async with session.post(
            f"{INF_PROXY_LEDGER_API_URL}/model-deployments/declare",
            json=build_declaration(model_name),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


async def declare_until_success(max_attempts: int | None = None) -> dict[str, Any] | None:
    """Retry declaration until it succeeds (or max_attempts is exhausted).

    Returns the deployment JSON on success, None on give-up. Never raises.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            deployment = await _attempt_declaration()
        except Exception as exc:  # noqa: BLE001 — by design: keep serving
            if max_attempts is not None and attempt >= max_attempts:
                log.error(
                    "Ledger declaration gave up after %d attempts: %s "
                    "(events will not resolve until the model is declared)",
                    attempt,
                    exc,
                )
                return None
            log.warning(
                "Ledger declaration attempt %d failed, retrying in %ss: %s",
                attempt,
                INF_PROXY_DECLARE_RETRY_SECONDS,
                exc,
            )
            await asyncio.sleep(INF_PROXY_DECLARE_RETRY_SECONDS)
            continue
        log.info(
            "Declared model deployment hostname=%s model_id=%s hardware_id=%s deployment_id=%s",
            MODEL_HOSTNAME,
            deployment.get("model_id"),
            deployment.get("hardware_id"),
            deployment.get("deployment_id"),
        )
        return deployment


async def close_deployment() -> bool:
    """Report the prover's shutdown: sets ended_at on this host's active
    deployment. Best-effort — one attempt on a short timeout (the pod is
    being torn down); a 404 means a re-declare already closed it. Never
    raises. No-op unless declaration is enabled (same opt-in)."""
    if not INF_PROXY_DECLARE:
        return False
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.post(
                f"{INF_PROXY_LEDGER_API_URL}/model-deployments/close",
                json={"hostname": MODEL_HOSTNAME},
            ) as resp:
                if resp.status == 200:
                    log.info("Closed model deployment hostname=%s", MODEL_HOSTNAME)
                    return True
                log.warning(
                    "Deployment close returned %s for hostname=%s",
                    resp.status,
                    MODEL_HOSTNAME,
                )
                return False
    except Exception as exc:  # noqa: BLE001 — shutdown path, by design
        log.warning("Deployment close failed for hostname=%s: %s", MODEL_HOSTNAME, exc)
        return False


def spawn_declaration() -> None:
    """Kick off the startup declaration in the background (no-op unless
    INF_PROXY_DECLARE is enabled)."""
    global _declaration_task
    if not INF_PROXY_DECLARE:
        log.info("Startup declaration disabled (INF_PROXY_DECLARE not set)")
        return
    _declaration_task = asyncio.create_task(declare_until_success())
