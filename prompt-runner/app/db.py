"""Direct Postgres persistence for benchmarking runs.

The prompt-runner owns the benchmarking_-prefixed tables end to end: it
creates them at startup and is the only writer (design ruling: services
integrate through the shared database, each owning its prefixed tables —
never via the ledger API). Same database and credentials as the ledger;
separation is by table prefix.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

DDL = """
CREATE TABLE IF NOT EXISTS benchmarking_run (
    id             TEXT PRIMARY KEY,
    state          TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    error          TEXT,
    settings       JSONB NOT NULL,
    total_requests INTEGER NOT NULL DEFAULT 0,
    completed      INTEGER NOT NULL DEFAULT 0,
    failed         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS benchmarking_run_result (
    run_id           TEXT NOT NULL REFERENCES benchmarking_run(id) ON DELETE CASCADE,
    request_index    INTEGER NOT NULL,
    prompt_id        TEXT NOT NULL,
    sampling_id      TEXT NOT NULL,
    model            TEXT NOT NULL,
    status_code      INTEGER,
    stream_completed BOOLEAN NOT NULL,
    elapsed_s        DOUBLE PRECISION NOT NULL,
    error            TEXT,
    PRIMARY KEY (run_id, request_index)
);
"""


def normalize_url(url: str) -> str:
    """The cluster secret stores a SQLAlchemy URL (postgresql+psycopg://);
    raw psycopg wants plain postgresql://."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


class RunStore:
    """All benchmarking SQL in one place. Short-lived connection per call:
    write volume is one row per finished prompt plus state transitions, so
    pooling would be dead weight."""

    def __init__(self, url: str) -> None:
        self._url = normalize_url(url)

    def _connect(self) -> psycopg.Connection:
        # connect_timeout bounds the TCP handshake: a blackholed DB (e.g.
        # network partition, not "connection refused") would otherwise
        # block for the OS TCP timeout (minutes), holding
        # BufferedRunWriter's lock the whole time.
        return psycopg.connect(self._url, row_factory=dict_row, connect_timeout=5)

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(DDL)
            conn.commit()

    def insert_queued(self, run_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO benchmarking_run (id, state, created_at, settings) "
                "VALUES (%s, 'queued', now(), %s) RETURNING *",
                (run_id, Json(settings)),
            ).fetchone()
            conn.commit()
            return row

    def claim_next(self) -> dict[str, Any] | None:
        """Oldest queued -> running, atomically. SKIP LOCKED keeps this
        correct even though the deployment runs a single replica."""
        with self._connect() as conn:
            row = conn.execute(
                """
                UPDATE benchmarking_run SET state = 'running', started_at = now()
                WHERE id = (
                    SELECT id FROM benchmarking_run WHERE state = 'queued'
                    AND NOT EXISTS (SELECT 1 FROM benchmarking_run WHERE state = 'running')
                    ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
            ).fetchone()
            conn.commit()
            return row

    def set_finished(self, run_id: str, state: str, error: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE benchmarking_run SET state = %s, error = %s, finished_at = now() WHERE id = %s",
                (state, error, run_id),
            )
            conn.commit()

    def set_error(self, run_id: str, error: str) -> None:
        self.set_finished(run_id, "error", error)

    def update_progress(self, run_id: str, total_requests: int, completed: int, failed: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE benchmarking_run SET total_requests = %s, completed = %s, failed = %s WHERE id = %s",
                (total_requests, completed, failed, run_id),
            )
            conn.commit()

    def add_results(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._connect() as conn:
            conn.cursor().executemany(
                "INSERT INTO benchmarking_run_result "
                "(run_id, request_index, prompt_id, sampling_id, model, status_code, stream_completed, elapsed_s, error) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    (run_id, r["request_index"], r["prompt_id"], r["sampling_id"], r["model"],
                     r["status_code"], r["stream_completed"], r["elapsed_s"], r["error"])
                    for r in rows
                ],
            )
            conn.commit()

    @staticmethod
    def _run_filters(states: list[str] | None, model: str | None) -> tuple[str, list[Any]]:
        """WHERE clause + params shared by list_runs/count_runs. The model
        filter matches runs whose settings.models allocation includes the
        model (JSONB containment on the array element)."""
        clauses: list[str] = []
        params: list[Any] = []
        if states:
            clauses.append("state = ANY(%s)")
            params.append(states)
        if model:
            clauses.append("settings->'models' @> jsonb_build_array(jsonb_build_object('model', %s::text))")
            params.append(model)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def list_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        states: list[str] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        where, params = self._run_filters(states, model)
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM benchmarking_run{where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (*params, limit, offset),
            ).fetchall()

    def count_runs(self, states: list[str] | None = None, model: str | None = None) -> int:
        where, params = self._run_filters(states, model)
        with self._connect() as conn:
            return conn.execute(
                f"SELECT count(*) AS n FROM benchmarking_run{where}", tuple(params)
            ).fetchone()["n"]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM benchmarking_run WHERE id = %s", (run_id,)
            ).fetchone()

    def get_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM benchmarking_run_result WHERE run_id = %s ORDER BY request_index",
                (run_id,),
            ).fetchall()

    def cancel_queued(self, run_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE benchmarking_run SET state = 'cancelled', finished_at = now() "
                "WHERE id = %s AND state = 'queued' RETURNING id",
                (run_id,),
            ).fetchone()
            conn.commit()
            return row is not None

    TERMINAL_STATES = ("completed", "error", "cancelled", "interrupted")

    def delete_run(self, run_id: str) -> bool:
        """Delete one TERMINAL run (results cascade via the FK). Queued runs
        are cancelled, not deleted (cancel_queued); the running run is never
        touched. Returns False when the row is absent or non-terminal."""
        with self._connect() as conn:
            row = conn.execute(
                "DELETE FROM benchmarking_run WHERE id = %s AND state = ANY(%s) RETURNING id",
                (run_id, list(self.TERMINAL_STATES)),
            ).fetchone()
            conn.commit()
            return row is not None

    def delete_finished(self) -> int:
        """Delete ALL terminal runs (the durable history); queued and
        running rows untouched. Returns how many runs were deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM benchmarking_run WHERE state = ANY(%s)",
                (list(self.TERMINAL_STATES),),
            )
            conn.commit()
            return cur.rowcount

    def sweep_interrupted(self) -> int:
        """Rows still 'running' at startup mean the pod died mid-run.
        Partial verification data would mislead an auto-retry — re-queueing
        is a deliberate user act (spec ruling)."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE benchmarking_run SET state = 'interrupted', finished_at = now() WHERE state = 'running'"
            )
            conn.commit()
            return cur.rowcount

    def sweep_orphaned_running(self) -> int:
        """Same fix as sweep_interrupted, but for runtime self-healing
        rather than only at startup: the caller (QueueWorker.tick) must only
        call this when it isn't itself executing a run, since the
        single-replica invariant means any 'running' row it sees in that
        case belongs to a process that died before reaching set_finished
        (e.g. after every retry attempt was exhausted) — a wedge that would
        otherwise block claim_next forever."""
        return self.sweep_interrupted()
