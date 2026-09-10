"""RunStore against a real Postgres (testcontainers). Sync psycopg — no
event-loop machinery needed (the ledger's async conftest is NOT reused;
these tables belong to the prompt-runner alone)."""

from __future__ import annotations

import pytest
from testcontainers.postgres import PostgresContainer

from app.db import RunStore, normalize_url

SETTINGS = {
    "models": [{"model": "m-a", "percent": 100, "concurrency": 2}],
    "promptbench_preset": "verification-v1",
    "prompt_count": 4,
}


@pytest.fixture(scope="module")
def store():
    with PostgresContainer("postgres:16", username="ledger", password="ledger", dbname="ledger") as pg:
        url = f"postgresql://ledger:ledger@{pg.get_container_host_ip()}:{pg.get_exposed_port(5432)}/ledger"
        s = RunStore(url)
        s.init_schema()
        yield s


@pytest.fixture(autouse=True)
def clean(store):
    with store._connect() as conn:
        conn.execute("DELETE FROM benchmarking_run_result")
        conn.execute("DELETE FROM benchmarking_run")
        conn.commit()


def test_normalize_url_strips_sqlalchemy_driver():
    assert (
        normalize_url("postgresql+psycopg://ledger:ledger@infver-db:5432/ledger")
        == "postgresql://ledger:ledger@infver-db:5432/ledger"
    )
    assert normalize_url("postgresql://x@y/z") == "postgresql://x@y/z"


def test_init_schema_is_idempotent(store):
    store.init_schema()  # second call must not raise


def test_insert_claim_finish_lifecycle(store):
    row = store.insert_queued("run-1", SETTINGS)
    assert row["state"] == "queued" and row["settings"]["prompt_count"] == 4

    claimed = store.claim_next()
    assert claimed["id"] == "run-1" and claimed["state"] == "running"
    assert claimed["started_at"] is not None

    # Nothing else queued and run-1 is running: no further claim.
    assert store.claim_next() is None

    store.update_progress("run-1", total_requests=4, completed=2, failed=1)
    store.set_finished("run-1", "completed")
    final = store.get_run("run-1")
    assert final["state"] == "completed" and final["finished_at"] is not None
    assert final["completed"] == 2 and final["failed"] == 1


def test_claim_is_fifo_and_blocked_by_running(store):
    store.insert_queued("run-a", SETTINGS)
    store.insert_queued("run-b", SETTINGS)
    assert store.claim_next()["id"] == "run-a"
    assert store.claim_next() is None  # run-a still running
    store.set_finished("run-a", "completed")
    assert store.claim_next()["id"] == "run-b"


def test_results_roundtrip_and_idempotent_flush(store):
    store.insert_queued("run-1", SETTINGS)
    rows = [
        {"request_index": 0, "prompt_id": "p-0", "sampling_id": "s", "model": "m-a",
         "status_code": 200, "stream_completed": True, "elapsed_s": 1.5, "error": None},
        {"request_index": 1, "prompt_id": "p-1", "sampling_id": "s", "model": "m-b",
         "status_code": 500, "stream_completed": False, "elapsed_s": 0.2, "error": "boom"},
    ]
    store.add_results("run-1", rows)
    store.add_results("run-1", rows)  # re-flush must not raise or duplicate
    got = store.get_results("run-1")
    assert [r["request_index"] for r in got] == [0, 1]
    assert got[1]["error"] == "boom" and got[0]["model"] == "m-a"


def test_cancel_only_touches_queued(store):
    store.insert_queued("run-1", SETTINGS)
    assert store.cancel_queued("run-1") is True
    assert store.get_run("run-1")["state"] == "cancelled"
    assert store.cancel_queued("run-1") is False  # already cancelled

    store.insert_queued("run-2", SETTINGS)
    store.claim_next()
    assert store.cancel_queued("run-2") is False  # running, not cancellable


def test_sweep_marks_running_as_interrupted(store):
    store.insert_queued("run-1", SETTINGS)
    store.insert_queued("run-2", SETTINGS)
    store.claim_next()
    assert store.sweep_interrupted() == 1
    assert store.get_run("run-1")["state"] == "interrupted"
    assert store.get_run("run-2")["state"] == "queued"  # untouched


def test_list_runs_newest_first(store):
    store.insert_queued("run-1", SETTINGS)
    store.insert_queued("run-2", SETTINGS)
    ids = [r["id"] for r in store.list_runs()]
    assert ids == ["run-2", "run-1"]
    assert "settings" in store.list_runs()[0]


def test_delete_run_terminal_only_with_cascade(store):
    store.insert_queued("run-1", SETTINGS)
    # Queued: protected.
    assert store.delete_run("run-1") is False
    # Running: protected.
    store.claim_next()
    assert store.delete_run("run-1") is False
    # Terminal: deleted, results cascade.
    store.set_finished("run-1", "completed")
    store.add_results("run-1", [
        {"request_index": 0, "prompt_id": "p-0", "sampling_id": "s", "model": "m-a",
         "status_code": 200, "stream_completed": True, "elapsed_s": 1.0, "error": None},
    ])
    assert store.delete_run("run-1") is True
    assert store.get_run("run-1") is None
    assert store.get_results("run-1") == []
    assert store.delete_run("run-1") is False  # already gone


def test_delete_finished_spares_queue_and_running(store):
    for i, state in enumerate(["completed", "error", "cancelled", "interrupted"]):
        store.insert_queued(f"run-{i}", SETTINGS)
        with store._connect() as conn:
            conn.execute("UPDATE benchmarking_run SET state = %s WHERE id = %s", (state, f"run-{i}"))
            conn.commit()
    store.insert_queued("run-q", SETTINGS)
    store.insert_queued("run-r", SETTINGS)
    with store._connect() as conn:
        conn.execute("UPDATE benchmarking_run SET state = 'running' WHERE id = 'run-r'")
        conn.commit()

    assert store.delete_finished() == 4
    remaining = {r["id"] for r in store.list_runs()}
    assert remaining == {"run-q", "run-r"}
    assert store.delete_finished() == 0
