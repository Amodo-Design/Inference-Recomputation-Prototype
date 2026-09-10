"""Bounded Prometheus range queries + integration for GPU enrichment.

Each DCGM sample is a utilisation ratio (0..1) averaged over the exporter's
collection interval; integrating the ratio over an event's window
(rectangle rule, sample * step) yields busy-SECONDS — comparable across
events of different lengths. A pod holding several GPUs returns one series
per GPU; they are summed per timestamp (total busy-time across the pod's
devices).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import httpx

METRICS: list[str] = [
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP64_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
    "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE",
    "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE",
]
HEADLINE_METRIC = "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"
OCCUPANCY_METRIC = "DCGM_FI_PROF_SM_OCCUPANCY"
STEP_SECONDS = 0.2
# Prometheus rejects query_range over 11,000 points; 2000s @ 200ms = 10,000.
MAX_CHUNK_SECONDS = 2000


def integrate(values: list[float], step_s: float = STEP_SECONDS) -> float | None:
    """Rectangle-rule integral of a ratio series -> busy-seconds."""
    if not values:
        return None
    return sum(values) * step_s


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


class PrometheusClient:
    def __init__(
        self,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, transport=transport, timeout=timeout
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def range_values(
        self, metric: str, selector: str, start: datetime, end: datetime
    ) -> list[float]:
        """Fetch the series at STEP_SECONDS resolution, chunking the window.

        Prometheus caps query_range at 11,000 points, which at a 200ms step
        is ~36.7 minutes — long windows (e.g. a run's multi-hour verification
        drain) would 400 outright. Chunks of MAX_CHUNK_SECONDS stay safely
        under the cap while preserving full resolution; per-chunk series sums
        are merged by timestamp (overlapping boundary samples overwrite with
        the identical value, never double-count).
        """
        merged: dict[float, float] = {}
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(
                chunk_start + timedelta(seconds=MAX_CHUNK_SECONDS), end
            )
            resp = await self._client.get(
                "/api/v1/query_range",
                params={
                    "query": f"{metric}{selector}",
                    "start": chunk_start.timestamp(),
                    "end": chunk_end.timestamp(),
                    "step": f"{int(STEP_SECONDS * 1000)}ms",
                },
            )
            resp.raise_for_status()
            result = resp.json()["data"]["result"]
            # Sum across series per timestamp WITHIN the chunk: a pod with N
            # GPUs has N series.
            by_ts: dict[float, float] = defaultdict(float)
            for series in result:
                for ts, value in series.get("values", []):
                    by_ts[float(ts)] += float(value)
            merged.update(by_ts)
            chunk_start = chunk_end
        return [merged[ts] for ts in sorted(merged)]
