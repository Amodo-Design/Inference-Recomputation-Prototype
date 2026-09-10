"""GET /runs/{run_id}/cohort and GET /runs/{run_id}/economics — first-class
run views layered over the ledger's event tables and the benchmarking_run
table (prompt-runner-owned, read-only here).

Cohort = per-allocation-model inference events whose ts falls inside the
run's [started_at, finished_at ?? now] window, LEFT JOINed to their
verification event (an event with no verification yet contributes zero
tokens and does not count toward verified/pass/fail/unverifiable).

Economics adds two independently-degrading sides per model, both fetched
from gpu-enricher as a pod-window GPU integral (never a per-event sum, since
verify concurrency > 1 means per-event tensor_active_time_s double-counts
GPU time shared across concurrently-running verifications):
  - prover: over the cohort's own DISTINCT inference_event.pod_name values,
    windowed to the run's own [started_at, finished_at ?? now].
  - verify: over the cohort's own DISTINCT verification_event.pod_name
    values, windowed to the drain window
    [min(verification started_at), max(verification ts)] — the span the
    verify pods were actually busy draining this cohort's verifications,
    which outlives the run window whenever verification queues past
    finished_at.
  Also reports a per-run concurrency block: "configured" (max observed
  verifier_detail.runner_concurrency) vs. "observed_mean"/"observed_max"
  (from enrichment_gpu_activity.concurrent_events — this counts OTHER
  concurrently-running events, so observed ≈ configured - 1 is expected).
  Any enricher failure (non-200, connect error, its 6h window cap) degrades
  either side to busy_seconds=None — this route never 5xxs because the
  enricher had a bad day.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models import BenchmarkingRun, EnrichmentGpuActivity, InferenceEvent, Model, VerificationEvent

log = logging.getLogger("analytics_api.runs")

router = APIRouter(prefix="/runs", tags=["runs"])


async def _run_or_404(session: AsyncSession, run_id: str) -> BenchmarkingRun:
    run = await session.get(BenchmarkingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


def _window(run: BenchmarkingRun) -> tuple[datetime.datetime | None, datetime.datetime]:
    return run.started_at, run.finished_at or datetime.datetime.now(datetime.timezone.utc)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: float | None, denominator: float | int | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


async def _cohort_raw_rows(
    session: AsyncSession, model_name: str, frm: datetime.datetime | None, to: datetime.datetime
) -> list[Any]:
    """One grouped-by-nothing query over inference_event JOIN model (name
    match) LEFT JOIN verification_event LEFT JOIN enrichment_gpu_activity
    (for concurrent_events only — busy_seconds itself comes from the
    gpu-enricher pod-window call, not this join), filtered ts BETWEEN
    window. Aggregation happens in Python (matches /analysis's
    verifier_detail.get() pattern — no JSONB-in-SQL extraction needed)."""
    if frm is None:
        return []
    stmt = (
        select(
            InferenceEvent.id,
            InferenceEvent.pod_name,
            InferenceEvent.ts,
            InferenceEvent.started_at,
            VerificationEvent.id,
            VerificationEvent.pod_name,
            VerificationEvent.ts,
            VerificationEvent.started_at,
            VerificationEvent.result,
            VerificationEvent.verifier_detail,
            EnrichmentGpuActivity.concurrent_events,
        )
        .select_from(InferenceEvent)
        .join(Model, InferenceEvent.model_id == Model.model_id)
        .outerjoin(VerificationEvent, VerificationEvent.inference_event_id == InferenceEvent.id)
        .outerjoin(
            EnrichmentGpuActivity,
            (EnrichmentGpuActivity.event_type == "verification")
            & (EnrichmentGpuActivity.event_id == VerificationEvent.id),
        )
        .where(
            Model.model_name == model_name,
            InferenceEvent.ts >= frm,
            InferenceEvent.ts <= to,
        )
    )
    return (await session.execute(stmt)).all()


def _aggregate(rows: list[Any]) -> dict[str, Any]:
    """Reduce the raw joined rows to every count/sum both endpoints need.
    Distinguishes 'no matching rows' (None) from 'rows summed to zero'
    (0.0) for every optional sum — that distinction is what lets economics
    degrade nulls instead of misreporting a real zero."""
    seen_inf_ids: set = set()
    pods: set[str] = set()
    verify_pods: set[str] = set()
    prover_wall_clock = 0.0
    have_prover_wall_clock = False
    output_tokens = 0
    prompt_tokens = 0
    verified_count = 0
    pass_count = 0
    fail_count = 0
    unverifiable_count = 0
    verify_wall_clock = 0.0
    have_verify_wall_clock = False
    drain_from: datetime.datetime | None = None
    drain_to: datetime.datetime | None = None
    configured: int | None = None
    observed: list[int] = []

    for (
        inf_id,
        pod_name,
        inf_ts,
        inf_started,
        ver_id,
        ver_pod_name,
        ver_ts,
        ver_started,
        result,
        detail,
        concurrent_events,
    ) in rows:
        if inf_id not in seen_inf_ids:
            seen_inf_ids.add(inf_id)
            if pod_name:
                pods.add(pod_name)
            if inf_started is not None:
                prover_wall_clock += (inf_ts - inf_started).total_seconds()
                have_prover_wall_clock = True

        if ver_id is not None:
            verified_count += 1
            if result == "pass":
                pass_count += 1
            elif result == "fail":
                fail_count += 1
            elif result == "unverifiable":
                unverifiable_count += 1

            d = detail or {}
            prompt_tokens += _int_or_none(d.get("prompt_token_count")) or 0
            output_tokens += _int_or_none(d.get("output_token_count")) or 0

            if ver_started is not None:
                verify_wall_clock += (ver_ts - ver_started).total_seconds()
                have_verify_wall_clock = True

            if ver_pod_name:
                verify_pods.add(ver_pod_name)
            if ver_started is not None:
                drain_from = ver_started if drain_from is None else min(drain_from, ver_started)
            drain_to = ver_ts if drain_to is None else max(drain_to, ver_ts)
            rc = _int_or_none(d.get("runner_concurrency"))
            if rc is not None:
                configured = rc if configured is None else max(configured, rc)
            if concurrent_events is not None:
                observed.append(concurrent_events)

    return {
        "event_count": len(seen_inf_ids),
        "pods": pods,
        "prover_wall_clock_s": prover_wall_clock if have_prover_wall_clock else None,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "verified_count": verified_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unverifiable_count": unverifiable_count,
        "verify_wall_clock_s": verify_wall_clock if have_verify_wall_clock else None,
        "verify_pods": verify_pods,
        "drain_window": (drain_from, drain_to),
        "configured_concurrency": configured,
        "observed_concurrency": observed,
    }


@router.get("/{run_id}/cohort")
async def cohort(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _run_or_404(session, run_id)
    frm, to = _window(run)

    models_out = []
    for alloc in (run.settings or {}).get("models", []):
        model_name = alloc["model"]
        rows = await _cohort_raw_rows(session, model_name, frm, to)
        agg = _aggregate(rows)
        models_out.append(
            {
                "model": model_name,
                "event_count": agg["event_count"],
                "verified_count": agg["verified_count"],
                "pass_count": agg["pass_count"],
                "fail_count": agg["fail_count"],
                "unverifiable_count": agg["unverifiable_count"],
                "prompt_tokens": agg["prompt_tokens"],
                "output_tokens": agg["output_tokens"],
            }
        )

    return {
        "run_id": run.id,
        "window": {"from": frm, "to": to},
        "models": models_out,
    }


async def _window_activity(
    request: Request, pods: list[str], frm: datetime.datetime, to: datetime.datetime
) -> tuple[float | None, int | None]:
    """Query gpu-enricher for the pod-window GPU integral via gpu-enricher —
    used for both sides (prover over the run window and its own pods,
    verify over the drain window and its own pods). Any failure (non-200,
    connect error, its 6h window cap 400) degrades to (None, None) — never
    raised past this function."""
    # re.escape backslash-escapes regex metacharacters (including "-"), but
    # gpu-enricher splices this straight into a PromQL string literal, which
    # Go then unquotes — a lone "\-" is not a valid Go escape sequence and
    # Prometheus 400s the whole query. Doubling the backslashes here makes
    # them survive that string-literal layer intact.
    pattern = "^(" + "|".join(re.escape(p).replace("\\", "\\\\") for p in sorted(pods)) + ")$"
    transport = getattr(request.app.state, "enricher_transport", None)
    try:
        async with httpx.AsyncClient(
            base_url=settings.gpu_enricher_url, transport=transport, timeout=30
        ) as hc:
            resp = await hc.get(
                "/window-activity",
                params={
                    "pod_pattern": pattern,
                    "from": frm.isoformat(),
                    "to": to.isoformat(),
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"gpu-enricher returned non-object JSON: {type(data)!r}")
            return _float_or_none(data.get("busy_seconds")), _int_or_none(data.get("sample_count"))
    except (httpx.HTTPError, ValueError) as exc:
        # A total enricher outage (non-200, connect error, non-JSON/garbage
        # 200 body) must degrade to nulls, never 5xx this route — but it
        # should still be visible in the logs.
        log.warning("gpu-enricher window-activity call failed, degrading to nulls: %s", exc)
    return None, None


@router.get("/{run_id}/economics")
async def economics(run_id: str, request: Request, session: AsyncSession = Depends(get_session)):
    run = await _run_or_404(session, run_id)
    if run.started_at is None:
        raise HTTPException(status_code=409, detail="run has not started")
    frm, to = _window(run)

    models_out = []
    for alloc in (run.settings or {}).get("models", []):
        model_name = alloc["model"]
        rows = await _cohort_raw_rows(session, model_name, frm, to)
        agg = _aggregate(rows)
        pods = sorted(agg["pods"])

        prover_busy: float | None = None
        prover_sample_count: int | None = None
        if pods:
            prover_busy, prover_sample_count = await _window_activity(request, pods, frm, to)

        verify_pods = sorted(agg["verify_pods"])
        drain_from, drain_to = agg["drain_window"]
        verify_busy: float | None = None
        verify_sample_count: int | None = None
        if verify_pods and drain_from is not None and drain_to is not None and drain_to > drain_from:
            verify_busy, verify_sample_count = await _window_activity(
                request, verify_pods, drain_from, drain_to
            )
        observed = agg["observed_concurrency"]
        concurrency = {
            "configured": agg["configured_concurrency"],
            "observed_mean": (sum(observed) / len(observed)) if observed else None,
            "observed_max": max(observed) if observed else None,
        }

        output_tokens = agg["output_tokens"]
        prompt_plus_output = agg["prompt_tokens"] + agg["output_tokens"]

        models_out.append(
            {
                "model": model_name,
                "prover": {
                    "busy_seconds": prover_busy,
                    "wall_clock_s": agg["prover_wall_clock_s"],
                    "output_tokens": output_tokens,
                    "busy_per_token": _safe_div(prover_busy, output_tokens),
                    "wall_clock_per_token": _safe_div(agg["prover_wall_clock_s"], output_tokens),
                    "sample_count": prover_sample_count,
                    "method": "pod-window integral",
                    "pods": pods,
                },
                "verify": {
                    "busy_seconds": verify_busy,
                    "wall_clock_s": agg["verify_wall_clock_s"],
                    "prompt_plus_output_tokens": prompt_plus_output,
                    "busy_per_token": _safe_div(verify_busy, prompt_plus_output),
                    "wall_clock_per_token": _safe_div(agg["verify_wall_clock_s"], prompt_plus_output),
                    "event_count": agg["verified_count"],
                    "sample_count": verify_sample_count,
                    "method": "pod-window integral",
                    "pods": verify_pods,
                    "window": {"from": drain_from, "to": drain_to},
                    "concurrency": concurrency,
                },
            }
        )

    return {
        "run_id": run.id,
        "window": {"from": frm, "to": to},
        "models": models_out,
    }
