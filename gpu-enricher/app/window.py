"""Domain-neutral pod-window GPU aggregates, computed live from Prometheus
with the same integrator the poller uses. The enricher knows nothing about
runs — callers (analytics-api) own the mapping from run → pod pattern."""

from __future__ import annotations

import datetime

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.prometheus import (
    HEADLINE_METRIC,
    METRICS,
    OCCUPANCY_METRIC,
    integrate,
    mean,
)

router = APIRouter(tags=["window"])

MAX_WINDOW = datetime.timedelta(hours=6)
MAX_PATTERN_CHARS = 512


def _has_unsafe_chars(pattern: str) -> bool:
    """pod_pattern is spliced directly into a PromQL string literal
    (`{pod=~"<pattern>"}`); a bare `"` would break out of that literal and a
    control character has no business in a pod-name regex either."""
    return any(ch == '"' or ord(ch) < 32 for ch in pattern)


@router.get("/window-activity")
async def window_activity(
    request: Request,
    pod_pattern: str = Query(...),
    from_: datetime.datetime = Query(..., alias="from"),
    to: datetime.datetime = Query(...),
):
    prom = getattr(request.app.state, "prom", None)
    if prom is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PROMETHEUS_URL is not configured",
        )
    if not pod_pattern or len(pod_pattern) > MAX_PATTERN_CHARS:
        raise HTTPException(status_code=400, detail=f"pod_pattern must be 1..{MAX_PATTERN_CHARS} chars")
    if _has_unsafe_chars(pod_pattern):
        raise HTTPException(status_code=400, detail="pod_pattern contains an unsafe character")
    if to <= from_:
        raise HTTPException(status_code=400, detail="to must be after from")
    if to - from_ > MAX_WINDOW:
        raise HTTPException(status_code=400, detail=f"window exceeds {MAX_WINDOW}")

    selector = f'{{pod=~"{pod_pattern}"}}'
    busy = None
    occupancy = None
    pipe: dict[str, float] = {}
    sample_count = 0
    try:
        for metric in METRICS:
            values = await prom.range_values(metric, selector, from_, to)
            if metric == HEADLINE_METRIC:
                sample_count = len(values)
                busy = integrate(values)
            elif metric == OCCUPANCY_METRIC:
                occupancy = mean(values)
            else:
                v = integrate(values)
                if v is not None:
                    pipe[metric] = v
    except httpx.HTTPStatusError as exc:
        body_snippet = exc.response.text[:200]
        raise HTTPException(
            status_code=502,
            detail=f"prometheus rejected query: {body_snippet}",
        ) from exc
    return {
        "busy_seconds": busy,
        "sm_occupancy_mean": occupancy,
        "pipe_activity_s": pipe,
        "sample_count": sample_count,
        "window_s": (to - from_).total_seconds(),
    }
