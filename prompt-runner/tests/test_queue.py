"""Queue worker + persisted API against a real Postgres, with the prompt
suite and per-prompt HTTP call stubbed exactly like test_service.py."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

import app.core as core
import app.service as service
from app.db import RunStore

from test_service import make_prompts, stub_single_prompt  # reuse stubs


@pytest.fixture(scope="module")
def store():
    with PostgresContainer("postgres:16", username="ledger", password="ledger", dbname="ledger") as pg:
        url = f"postgresql://ledger:ledger@{pg.get_container_host_ip()}:{pg.get_exposed_port(5432)}/ledger"
        s = RunStore(url)
        s.init_schema()
        yield s


@pytest.fixture(autouse=True)
def clean(store, monkeypatch):
    with store._connect() as conn:
        conn.execute("DELETE FROM benchmarking_run_result")
        conn.execute("DELETE FROM benchmarking_run")
        conn.commit()
    monkeypatch.setattr(core, "_load_prompt_suite", lambda args: (make_prompts(3), "stub suite"))
    monkeypatch.setattr(service, "_resolve_api_key", lambda: "stub-token")
    monkeypatch.setattr(service, "store", store)
    monkeypatch.setattr(service, "worker", service.QueueWorker(store))


PAYLOAD = {
    "models": [{"model": "m", "percent": 100, "concurrency": 1}],
    "skip_preflight": True,
}


def test_post_enqueues_and_tick_executes(store, monkeypatch):
    stub_single_prompt(monkeypatch)
    client = TestClient(service.app)

    created = client.post("/runs", json=PAYLOAD)
    assert created.status_code == 201, created.text
    run_id = created.json()["id"]
    assert created.json()["state"] == "queued"

    # No 409: a second POST queues behind the first.
    second = client.post("/runs", json=PAYLOAD)
    assert second.status_code == 201
    assert second.json()["state"] == "queued"

    assert service.worker.tick() is True  # claims + executes run 1
    row = store.get_run(run_id)
    assert row["state"] == "completed"
    assert row["completed"] == 3 and row["failed"] == 0
    results = store.get_results(run_id)
    assert len(results) == 3 and all(r["model"] == "m" for r in results)

    assert service.worker.tick() is True  # run 2
    assert service.worker.tick() is False  # queue empty


def test_get_runs_and_detail_from_db(store, monkeypatch):
    stub_single_prompt(monkeypatch, fail_ids={"p-0"})
    client = TestClient(service.app)
    run_id = client.post("/runs", json=PAYLOAD).json()["id"]
    service.worker.tick()

    listed = client.get("/runs").json()["items"]
    assert [r["id"] for r in listed] == [run_id]
    assert listed[0]["state"] == "completed"  # failures are counters, not a state
    assert listed[0]["failed"] >= 1
    assert "results" not in listed[0]

    detail = client.get(f"/runs/{run_id}").json()
    assert len(detail["results"]) >= 1
    assert detail["log"] == []  # not the live run any more

    assert client.get("/runs/nope").status_code == 404


def test_cancel_queued_delete_terminal(store, monkeypatch):
    stub_single_prompt(monkeypatch)
    client = TestClient(service.app)
    run_id = client.post("/runs", json=PAYLOAD).json()["id"]

    # Queued -> cancelled (row kept as history).
    cancelled = client.delete(f"/runs/{run_id}")
    assert cancelled.status_code == 200
    assert cancelled.json() == {"id": run_id, "state": "cancelled"}
    assert store.get_run(run_id)["state"] == "cancelled"
    assert service.worker.tick() is False  # nothing left to claim

    # Terminal -> deleted outright, results cascade.
    done_id = client.post("/runs", json=PAYLOAD).json()["id"]
    service.worker.tick()
    assert len(store.get_results(done_id)) > 0
    deleted = client.delete(f"/runs/{done_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": done_id, "deleted": True}
    assert store.get_run(done_id) is None
    assert store.get_results(done_id) == []

    assert client.delete("/runs/nope").status_code == 404


def test_delete_running_run_is_409(store, monkeypatch):
    stub_single_prompt(monkeypatch)
    client = TestClient(service.app)
    run_id = client.post("/runs", json=PAYLOAD).json()["id"]
    claimed = store.claim_next()
    assert claimed["id"] == run_id
    assert client.delete(f"/runs/{run_id}").status_code == 409
    assert store.get_run(run_id)["state"] == "running"


def test_delete_all_clears_history_only(store, monkeypatch):
    stub_single_prompt(monkeypatch)
    client = TestClient(service.app)

    # Two terminal runs (one completed, one cancelled)...
    done_id = client.post("/runs", json=PAYLOAD).json()["id"]
    service.worker.tick()
    cancelled_id = client.post("/runs", json=PAYLOAD).json()["id"]
    client.delete(f"/runs/{cancelled_id}")
    # ...one running, one queued behind it.
    running_id = client.post("/runs", json=PAYLOAD).json()["id"]
    store.claim_next()
    queued_id = client.post("/runs", json=PAYLOAD).json()["id"]

    r = client.delete("/runs")
    assert r.status_code == 200
    assert r.json() == {"deleted": 2}
    assert store.get_run(done_id) is None
    assert store.get_run(cancelled_id) is None
    assert store.get_run(running_id)["state"] == "running"
    assert store.get_run(queued_id)["state"] == "queued"


def test_delete_all_503_without_store(monkeypatch):
    monkeypatch.setattr(service, "store", None)
    monkeypatch.setattr(service, "worker", None)
    client = TestClient(service.app)
    assert client.delete("/runs").status_code == 503


def test_invalid_settings_at_claim_time_marks_error(store, monkeypatch):
    stub_single_prompt(monkeypatch)
    client = TestClient(service.app)
    run_id = client.post("/runs", json=PAYLOAD).json()["id"]
    # Corrupt the stored settings (simulates e.g. schema drift between
    # queueing and claiming).
    with store._connect() as conn:
        conn.execute(
            "UPDATE benchmarking_run SET settings = '{\"nonsense\": true}' WHERE id = %s",
            (run_id,),
        )
        conn.commit()
    assert service.worker.tick() is True
    row = store.get_run(run_id)
    assert row["state"] == "error"
    assert row["error"]


def test_post_returns_503_without_store(monkeypatch):
    monkeypatch.setattr(service, "store", None)
    monkeypatch.setattr(service, "worker", None)
    client = TestClient(service.app)
    r = client.post("/runs", json=PAYLOAD)
    assert r.status_code == 503


def test_set_finished_retries_transient_failure(store, monkeypatch):
    """A set_finished call that fails once (transient DB blip) must not
    leave the row stuck 'running' forever: the retry helper covers it."""
    stub_single_prompt(monkeypatch)
    monkeypatch.setattr(service, "SET_FINISHED_RETRY_SECONDS", 0.0)  # don't slow the test
    client = TestClient(service.app)
    run_id = client.post("/runs", json=PAYLOAD).json()["id"]

    real_set_finished = store.set_finished
    calls = {"n": 0}

    def flaky(run_id_, state, error=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient DB blip")
        return real_set_finished(run_id_, state, error)

    monkeypatch.setattr(store, "set_finished", flaky)
    assert service.worker.tick() is True
    assert calls["n"] == 2  # first attempt raised, retry succeeded
    assert store.get_run(run_id)["state"] == "completed"


def test_tick_self_heals_orphaned_running_row(store, monkeypatch):
    """Single-replica invariant: if the worker isn't executing anything and
    a row is still 'running', that row's owning process must have died
    before reaching set_finished. tick() must sweep it to 'interrupted' so
    claim_next can claim the next queued run."""
    stub_single_prompt(monkeypatch)
    client = TestClient(service.app)

    orphan_id = client.post("/runs", json=PAYLOAD).json()["id"]
    queued_id = client.post("/runs", json=PAYLOAD).json()["id"]

    claimed = store.claim_next()  # promotes orphan_id to 'running' directly
    assert claimed["id"] == orphan_id
    service.worker.current = None  # simulate the owning process dying unnoticed

    assert service.worker.tick() is False  # claim_next blocked by 'running' row
    assert store.get_run(orphan_id)["state"] == "interrupted"

    assert service.worker.tick() is True  # queued run can now be claimed
    row = store.get_run(queued_id)
    assert row["state"] == "completed"


def test_buffered_writer_survives_db_outage(store):
    writer = service.BufferedRunWriter(store, "run-x")
    store.insert_queued("run-x", PAYLOAD)

    row = {"request_index": 0, "prompt_id": "p-0", "sampling_id": "s", "model": "m",
           "status_code": 200, "stream_completed": True, "elapsed_s": 1.0, "error": None}

    # Break the DB: writes buffer instead of raising.
    good_url = store._url
    store._url = "postgresql://nobody:nope@127.0.0.1:1/void"
    writer.on_result(row)          # must not raise
    assert len(writer.pending) == 1

    # DB returns, but the failure cooldown is still active: on_result
    # buffers without re-attempting the backlog per row.
    store._url = good_url
    writer.on_result({**row, "request_index": 1})
    assert len(writer.pending) == 2

    # final_flush ignores the cooldown and always tries once.
    writer.final_flush()
    assert writer.pending == []
    assert len(store.get_results("run-x")) == 2


def test_buffered_writer_cooldown_skips_reattempt_per_row(store, monkeypatch):
    """After a failed flush, on_result must not re-attempt the DB for every
    subsequent row within FLUSH_RETRY_SECONDS — it just buffers. Verified
    directly against add_results call counts (not just end-state) so the
    cooldown itself, not merely eventual success, is under test."""
    monkeypatch.setattr(service, "FLUSH_RETRY_SECONDS", 10.0)
    store.insert_queued("run-y", PAYLOAD)
    writer = service.BufferedRunWriter(store, "run-y")

    calls = {"n": 0}
    real_add_results = store.add_results

    def flaky(run_id, rows):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_add_results(run_id, rows)

    monkeypatch.setattr(store, "add_results", flaky)

    row = {"request_index": 0, "prompt_id": "p-0", "sampling_id": "s", "model": "m",
           "status_code": 200, "stream_completed": True, "elapsed_s": 1.0, "error": None}
    writer.on_result(row)  # attempt 1: fails, enters cooldown
    assert calls["n"] == 1

    writer.on_result({**row, "request_index": 1})  # cooling down: no attempt
    writer.on_result({**row, "request_index": 2})  # still cooling down
    assert calls["n"] == 1
    assert len(writer.pending) == 3

    writer.final_flush()  # ignores cooldown: attempt 2 succeeds
    assert calls["n"] == 2
    assert writer.pending == []
    assert len(store.get_results("run-y")) == 3


def test_list_runs_paging_states_and_model_filter(store, monkeypatch):
    stub_single_prompt(monkeypatch)
    client = TestClient(service.app)

    # Three completed runs across two models, one queued behind a claim.
    payload_b = {**PAYLOAD, "models": [{"model": "m-b", "percent": 100, "concurrency": 1}]}
    ids = []
    for body in (PAYLOAD, payload_b, PAYLOAD):
        ids.append(client.post("/runs", json=body).json()["id"])
        service.worker.tick()
    queued_id = client.post("/runs", json=PAYLOAD).json()["id"]

    # Paging over the full set: newest first, envelope carries total.
    page1 = client.get("/runs", params={"limit": 2}).json()
    assert page1["total"] == 4 and page1["limit"] == 2 and page1["offset"] == 0
    assert [r["id"] for r in page1["items"]] == [queued_id, ids[2]]
    page2 = client.get("/runs", params={"limit": 2, "offset": 2}).json()
    assert [r["id"] for r in page2["items"]] == [ids[1], ids[0]]

    # State filter: live vs terminal.
    live = client.get("/runs", params={"states": "queued,running"}).json()
    assert [r["id"] for r in live["items"]] == [queued_id] and live["total"] == 1
    hist = client.get("/runs", params={"states": "completed,error,cancelled,interrupted"}).json()
    assert hist["total"] == 3

    # Model filter matches allocation membership.
    only_b = client.get("/runs", params={"model": "m-b"}).json()
    assert [r["id"] for r in only_b["items"]] == [ids[1]] and only_b["total"] == 1
    both = client.get("/runs", params={"model": "m"}).json()
    assert both["total"] == 3  # exact match on "m", not substring of "m-b"

    # Unknown state names are rejected loudly.
    assert client.get("/runs", params={"states": "queued,bogus"}).status_code == 422
