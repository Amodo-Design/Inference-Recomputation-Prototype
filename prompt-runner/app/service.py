"""HTTP service around the vendored Open WebUI prompt runner.

Runs the deterministic PromptBench prompt suite against a tapped prover model
(through Open WebUI, so every response is captured as an inference event and
lands in the verification pipeline). One run at a time: verification numbers
are only comparable when the suite is the sole traffic on the model.

Run state is durable now: queued/running/completed/error/cancelled/interrupted
live in the benchmarking_run table (see app.db.RunStore), with a single
QueueWorker thread claiming and executing the oldest queued run at a time. A
pod restart forgets nothing except the currently-executing run's live log
(the terminal counters and any flushed results survive).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import threading
import time
import urllib.request
import uuid
from collections import deque
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import allocation, core
from .db import RunStore

log = logging.getLogger(__name__)

OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://open-webui:8080")
OPEN_WEBUI_API_KEY = os.getenv("OPEN_WEBUI_API_KEY", "")
MAX_LOG_LINES = 500
DATABASE_URL = os.getenv("DATABASE_URL", "")
QUEUE_POLL_SECONDS = float(os.getenv("QUEUE_POLL_SECONDS", "2.0"))
SET_FINISHED_MAX_ATTEMPTS = 5
SET_FINISHED_RETRY_SECONDS = 2.0
FLUSH_RETRY_SECONDS = 10.0


class ModelAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    percent: int = Field(ge=1, le=100)
    concurrency: int = Field(ge=1, le=32)


class RunSettings(BaseModel):
    """Mirror of the reference script's CLI arguments (openwebui-url and
    api-key come from the service environment instead), with the target
    model generalised to a percentage allocation across several models.

    The script's sampling knobs (seed/temperature/top-k/top-p and the sweep
    grid) are deliberately NOT exposed: the verify-taps pin every model's
    sampling config server-side, so request-level values can never take
    effect in this stack. The vendored core still receives the script
    defaults so payload shape and sampling_id labels stay identical."""

    # Reject unknown fields so removed knobs (seed, and the dropped legacy
    # model/concurrency pair) fail loudly instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    models: list[ModelAllocation] = Field(min_length=1)
    promptbench_preset: str = "verification-v1"
    prompt_count: int = Field(20, ge=1)
    start_prompt: int = Field(1, ge=1)
    max_tokens: int = Field(10000, ge=1)
    timeout: float = Field(120.0, gt=0)
    continue_on_error: bool = False
    skip_preflight: bool = False

    @model_validator(mode="after")
    def _check_allocations(self) -> "RunSettings":
        names = [a.model for a in self.models]
        if len(set(names)) != len(names):
            raise ValueError("duplicate models in allocation")
        total = sum(a.percent for a in self.models)
        if total != 100:
            raise ValueError(f"model percentages must sum to exactly 100 (got {total})")
        return self

    def to_namespace(self, model: str, concurrency: int) -> argparse.Namespace:
        """The vendored core functions read CLI-style attributes — one
        namespace per model allocation."""
        return argparse.Namespace(
            openwebui_url=OPENWEBUI_URL,
            api_key=OPEN_WEBUI_API_KEY,
            model=model,
            concurrency=concurrency,
            # Fixed at the reference script's defaults: the taps override
            # sampling anyway (see class docstring).
            seed=42,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            sampling_sweep=False,
            temperature_values=None,
            top_p_values=None,
            top_k_values=None,
            **self.model_dump(exclude={"models"}),
        )


class Run:
    """Live (in-memory) state of the currently executing run. Queue states
    and durable history live in the benchmarking_run table; this object only
    exists while its run is running."""

    def __init__(self, run_id: str, settings: RunSettings) -> None:
        self.id = run_id
        self.settings = settings
        self.state = "running"  # running -> completed | error
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.finished_at: str | None = None
        self.total_requests = 0
        self.completed = 0
        self.failed = 0
        self.error: str | None = None
        self.log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.results: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def log_line(self, line: str) -> None:
        with self._lock:
            self.log.append(line)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "total_requests": self.total_requests,
                "completed": self.completed,
                "failed": self.failed,
                "error": self.error,
                "settings": self.settings.model_dump(),
            }

    def detail(self) -> dict[str, Any]:
        data = self.summary()
        with self._lock:
            data["log"] = list(self.log)
            data["results"] = list(self.results)
        return data


def new_run_id() -> str:
    return f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def _resolve_api_key() -> str:
    """An explicit OPEN_WEBUI_API_KEY wins; otherwise self-authenticate.

    With WEBUI_AUTH=false Open WebUI still requires a bearer token on its
    API, but an empty-credentials signin returns the auto-admin's JWT — so
    the service fetches one per run instead of requiring an operator-issued
    key."""
    if OPEN_WEBUI_API_KEY:
        return OPEN_WEBUI_API_KEY
    request = urllib.request.Request(
        f"{OPENWEBUI_URL}/api/v1/auths/signin",
        data=json.dumps({"email": "", "password": ""}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        token = json.loads(response.read()).get("token")
    if not token:
        raise RuntimeError("Open WebUI signin returned no token")
    return token


def _run_slice(
    run: Run,
    args: argparse.Namespace,
    prompt_runs: list[Any],
    total: int,
    on_result: Callable[[dict], None] | None,
) -> None:
    """Today's scheduling loop, scoped to one model allocation's contiguous
    slice. Runs on its own thread; all slices execute in parallel, each at
    its allocation's concurrency. continue_on_error stops scheduling within
    THIS slice only (other models' traffic is independent)."""
    pending = iter(prompt_runs)
    active: dict[cf.Future, Any] = {}
    stop_scheduling = False

    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        while True:
            while not stop_scheduling and len(active) < max(1, args.concurrency):
                try:
                    prompt_run = next(pending)
                except StopIteration:
                    break
                future = executor.submit(
                    core._run_single_prompt,
                    prompt_run,
                    args,
                    run.id,
                    args.api_key,
                    args.openwebui_url,
                    args.timeout,
                )
                active[future] = prompt_run

            if not active:
                break

            done, _ = cf.wait(active.keys(), return_when=cf.FIRST_COMPLETED)
            for future in done:
                prompt_run = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 — mirror the script
                    result = core.RunResult(
                        index=prompt_run.index,
                        prompt_id=prompt_run.prompt.prompt_id,
                        sampling_id=prompt_run.sampling.sampling_id,
                        request_url=core._build_url(args.openwebui_url, "/api/chat/completions"),
                        status_code=None,
                        stream_completed=False,
                        elapsed_s=0.0,
                        error=str(exc),
                    )
                ok = result.stream_completed and (result.status_code or 0) < 400
                row = {
                    "request_index": result.index,
                    "prompt_id": result.prompt_id,
                    "sampling_id": result.sampling_id,
                    "model": args.model,
                    "status": "ok" if ok else "failed",
                    "status_code": result.status_code,
                    "stream_completed": result.stream_completed,
                    "elapsed_s": round(result.elapsed_s, 2),
                    "error": result.error,
                }
                with run._lock:
                    run.completed += 1
                    if not ok:
                        run.failed += 1
                    run.results.append(row)
                if on_result is not None:
                    on_result(row)
                suffix = "" if ok else f"; stream_completed={result.stream_completed}: {result.error}"
                run.log_line(
                    f"[{result.index + 1}/{total}] {args.model} {result.prompt_id} {result.sampling_id} "
                    f"-> {'ok' if ok else 'failed'} ({result.status_code or '-'}; "
                    f"{result.elapsed_s:.1f}s){suffix}"
                )
                if not ok and not args.continue_on_error:
                    stop_scheduling = True
            if stop_scheduling and not active:
                break


def execute_run(run: Run, on_result: Callable[[dict], None] | None = None) -> None:
    """Load the suite once, split it across the model allocations, and run
    every allocation's slice in parallel (one worker pool per model at its
    own concurrency — user ruling: parallel, not sequential)."""
    settings = run.settings
    run.state = "running"
    try:
        api_key = _resolve_api_key()
        first = settings.models[0]
        base_args = settings.to_namespace(model=first.model, concurrency=first.concurrency)
        base_args.api_key = api_key
        prompts, source_label = core._load_prompt_suite(base_args)
        sampling_configs = core._sampling_configs(base_args)
        all_runs = core._prompt_runs(prompts, sampling_configs)
        prompt_runs, selection_label = core._select_prompt_runs(all_runs, settings.start_prompt)
        run.total_requests = len(prompt_runs)
        slices = allocation.allocate(prompt_runs, settings.models)
        if not settings.skip_preflight:
            for alloc in settings.models:
                args = settings.to_namespace(model=alloc.model, concurrency=alloc.concurrency)
                args.api_key = api_key
                core._preflight(args, prompts)
        alloc_label = ", ".join(
            f"{alloc.model}={len(sl)} ({alloc.percent}%, conc {alloc.concurrency})"
            for alloc, sl in zip(settings.models, slices)
        )
        run.log_line(
            f"Run {run.id}: {len(prompts)} prompts from {source_label}; "
            f"{len(sampling_configs)} sampling configs; {selection_label}; allocation: {alloc_label}"
        )
    except Exception as exc:
        run.state = "error"
        run.error = str(exc)
        run.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        run.log_line(f"Run setup failed: {exc}")
        return

    total = len(prompt_runs)
    with cf.ThreadPoolExecutor(max_workers=max(1, len(settings.models))) as models_pool:
        futures = []
        for alloc, slice_runs in zip(settings.models, slices):
            if not slice_runs:
                continue
            args = settings.to_namespace(model=alloc.model, concurrency=alloc.concurrency)
            args.api_key = api_key
            futures.append(models_pool.submit(_run_slice, run, args, slice_runs, total, on_result))
        for future in cf.as_completed(futures):
            future.result()  # propagate slice-thread crashes

    run.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    run.state = "completed"
    run.log_line(
        f"Completed {run.completed - run.failed}/{run.total_requests} selected "
        f"requests ({run.failed} failed)."
    )


def _set_finished_with_retry(store: RunStore, run_id: str, state: str, error: str | None) -> bool:
    """set_finished with bounded retries: a transient DB blip must not leave
    the row stuck 'running' forever (claim_next never claims again while any
    row is 'running'). Returns True once persisted; logs loudly and returns
    False if every attempt fails."""
    last_exc: Exception | None = None
    for attempt in range(1, SET_FINISHED_MAX_ATTEMPTS + 1):
        try:
            store.set_finished(run_id, state, error)
            return True
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning(
                "set_finished attempt %d/%d failed for %s: %s",
                attempt, SET_FINISHED_MAX_ATTEMPTS, run_id, exc,
            )
            if attempt < SET_FINISHED_MAX_ATTEMPTS:
                time.sleep(SET_FINISHED_RETRY_SECONDS)
    log.error(
        "could not persist final state for %s after %d attempts; row will remain "
        "'running' until manually reconciled or the pod restarts: %s",
        run_id, SET_FINISHED_MAX_ATTEMPTS, last_exc,
    )
    return False


class BufferedRunWriter:
    """Streams result rows and counter updates to the store, buffering
    through DB outages: the run keeps executing, rows accumulate in
    `pending`, and every subsequent write retries the backlog first (spec:
    'results buffer in memory and flush when the DB returns'). Never raises
    into the run loop."""

    def __init__(self, store: RunStore, run_id: str) -> None:
        self._store = store
        self._run_id = run_id
        self.pending: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._cooldown_until: float | None = None

    def on_result(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.pending.append(row)
            if self._cooldown_until is not None and time.monotonic() < self._cooldown_until:
                return  # still cooling down from a recent failure; just buffer
            self._flush_locked()

    def update_progress(self, run: Run) -> None:
        try:
            self._store.update_progress(
                run.id, run.total_requests, run.completed, run.failed
            )
        except Exception as exc:  # noqa: BLE001 — DB outage must not kill the run
            log.warning("progress update failed for %s: %s", run.id, exc)

    def final_flush(self) -> None:
        with self._lock:
            self._flush_locked()  # ignores the cooldown: always tries once
        if self.pending:
            log.error(
                "run %s: %d result rows never flushed (DB unreachable); "
                "the startup interrupted-sweep reconciles state on next boot",
                self._run_id, len(self.pending),
            )

    def _flush_locked(self) -> None:
        try:
            self._store.add_results(self._run_id, self.pending)
            self.pending.clear()
            self._cooldown_until = None
        except Exception as exc:  # noqa: BLE001
            self._cooldown_until = time.monotonic() + FLUSH_RETRY_SECONDS
            log.warning("result flush failed for %s (%d rows buffered): %s",
                        self._run_id, len(self.pending), exc)


class QueueWorker:
    """Single-executor queue: claims the oldest queued run whenever nothing
    is running, executes it to completion, repeats. One instance, one
    thread; `tick()` is the synchronous unit tests drive directly."""

    def __init__(self, store: RunStore, poll_interval: float = QUEUE_POLL_SECONDS) -> None:
        self._store = store
        self._poll = poll_interval
        self.current: Run | None = None

    def start(self) -> None:
        threading.Thread(target=self._loop, name="queue-worker", daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                if not self.tick():
                    time.sleep(self._poll)
            except Exception:  # noqa: BLE001 — the worker must never die
                log.exception("queue worker tick failed")
                time.sleep(self._poll)

    def tick(self) -> bool:
        claimed = self._store.claim_next()
        if claimed is None:
            if self.current is None:
                # Single-replica invariant: if this worker isn't executing
                # anything, any row still 'running' in the DB is an orphan
                # (its owning process died between claiming and finishing
                # without ever reaching set_finished) and is silently
                # wedging the queue — claim_next() never claims while any
                # row is 'running'. Sweep it so the queue can move again.
                swept = self._store.sweep_orphaned_running()
                if swept:
                    log.warning("swept %d orphaned running run(s) to interrupted", swept)
            return False
        run_id = claimed["id"]
        try:
            settings = RunSettings.model_validate(claimed["settings"])
        except Exception as exc:  # noqa: BLE001 — stale/corrupt queued settings
            self._store.set_error(run_id, f"settings no longer valid: {exc}")
            return True

        run = Run(run_id, settings)
        self.current = run
        writer = BufferedRunWriter(self._store, run_id)
        try:
            def on_result(row: dict[str, Any]) -> None:
                writer.on_result(row)
                writer.update_progress(run)

            execute_run(run, on_result=on_result)
        except Exception as exc:  # noqa: BLE001 — belt and braces
            run.state = "error"
            run.error = str(exc)
        finally:
            writer.final_flush()
            writer.update_progress(run)
            _set_finished_with_retry(self._store, run_id, run.state, run.error)
            self.current = None
        return True


store: RunStore | None = None
worker: QueueWorker | None = None

app = FastAPI(title="Inference Verification Prompt Runner", version="0.2.0")


@app.on_event("startup")
def _startup() -> None:
    global store, worker
    if not DATABASE_URL:
        log.error("DATABASE_URL is not set — runs cannot be queued (503 on POST /runs)")
        return
    store = RunStore(DATABASE_URL)
    store.init_schema()
    swept = store.sweep_interrupted()
    if swept:
        log.warning("marked %d mid-flight run(s) as interrupted", swept)
    worker = QueueWorker(store)
    worker.start()


def _require_store() -> RunStore:
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="persistence unavailable: DATABASE_URL is not configured",
        )
    return store


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", status_code=status.HTTP_201_CREATED)
async def create_run(settings: RunSettings) -> dict[str, Any]:
    return _require_store().insert_queued(new_run_id(), settings.model_dump())


VALID_RUN_STATES = frozenset(
    ("queued", "running", "completed", "error", "cancelled", "interrupted")
)


@app.get("/runs")
async def list_runs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    states: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Paged run listing, newest first. `states` is a comma-separated state
    filter (e.g. "queued,running" for the live sections, or the terminal
    states for the history view); `model` matches runs whose allocation
    includes that model."""
    state_list: list[str] | None = None
    if states:
        state_list = [s.strip() for s in states.split(",") if s.strip()]
        unknown = set(state_list) - VALID_RUN_STATES
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown states: {sorted(unknown)}",
            )
    s = _require_store()
    return {
        "items": s.list_runs(limit=limit, offset=offset, states=state_list, model=model),
        "total": s.count_runs(states=state_list, model=model),
        "limit": limit,
        "offset": offset,
    }


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    s = _require_store()
    row = s.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No run {run_id}")
    live = worker.current if worker else None
    if live is not None and live.id == run_id:
        # The live object is fresher than the periodic DB flush.
        data = {**row, **live.summary(), "created_at": row["created_at"]}
        data["log"] = list(live.log)
        data["results"] = list(live.results)
        return data
    row["results"] = s.get_results(run_id)
    row["log"] = []
    return row


@app.delete("/runs")
async def delete_history() -> dict[str, Any]:
    """Delete ALL terminal runs (the durable history). Queued runs keep
    their place; the running run is untouched. The ledger's verification
    events from deleted runs always survive — only the run windows (and so
    the analysis tab's run-picker entries) are lost."""
    return {"deleted": _require_store().delete_finished()}


@app.delete("/runs/{run_id}")
async def cancel_or_delete_run(run_id: str) -> dict[str, Any]:
    """Queued runs are cancelled (row kept as history); terminal runs are
    deleted outright (result rows cascade); the running run is neither."""
    s = _require_store()
    if s.cancel_queued(run_id):
        return {"id": run_id, "state": "cancelled"}
    if s.delete_run(run_id):
        return {"id": run_id, "deleted": True}
    if s.get_run(run_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No run {run_id}")
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="the running run cannot be cancelled or deleted",
    )
