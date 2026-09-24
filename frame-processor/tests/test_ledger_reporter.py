"""Window rows reach the ledger without ever waiting on it."""

from __future__ import annotations

import threading
import time

import requests

from app.ledger_reporter import LedgerReporter


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _Session:
    """Scripted responses; records every post."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.posts: list[tuple[str, dict]] = []
        self.closed = False
        self.seen = threading.Event()

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        self.seen.set()
        outcome = self.script.pop(0) if self.script else _Response(201)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


def _wait(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def _reporter(session: _Session, **kw) -> LedgerReporter:
    kw.setdefault("retries", 2)
    kw.setdefault("drain_seconds", 1.0)
    return LedgerReporter("http://ledger-api:8000/", session_factory=lambda: session, sleep=lambda s: None, **kw)


ROW = {"window_start": "2026-09-02T10:00:00+00:00", "window_end": "2026-09-02T10:05:00+00:00", "complete": True, "finding_count": 0}


def test_a_row_is_posted_to_the_capture_windows_endpoint():
    session = _Session([_Response(201)])
    reporter = _reporter(session)
    assert reporter.submit(ROW)
    stats = reporter.close()
    assert session.posts == [("http://ledger-api:8000/capture-windows", ROW)]
    assert (stats.submitted, stats.delivered, stats.failed) == (1, 1, 0)
    assert session.closed


def test_a_duplicate_counts_as_delivered():
    """409 means the ledger already has it — a retry after a crash, not a loss."""
    session = _Session([_Response(409)])
    reporter = _reporter(session)
    reporter.submit(ROW)
    stats = reporter.close()
    assert (stats.delivered, stats.duplicates, stats.failed) == (0, 1, 0)


def test_server_errors_and_unreachable_ledgers_are_retried():
    session = _Session([_Response(503), requests.ConnectionError("down"), _Response(201)])
    reporter = _reporter(session, retries=3)
    reporter.submit(ROW)
    stats = reporter.close()
    assert len(session.posts) == 3
    assert stats.delivered == 1 and stats.failed == 0


def test_a_rejected_row_is_not_retried():
    session = _Session([_Response(422, "window_end must be after window_start")])
    reporter = _reporter(session)
    reporter.submit(ROW)
    stats = reporter.close()
    assert len(session.posts) == 1
    assert stats.failed == 1


def test_retries_give_up_and_say_so():
    session = _Session([_Response(500)] * 10)
    reporter = _reporter(session, retries=2)
    reporter.submit(ROW)
    stats = reporter.close()
    assert len(session.posts) == 3
    assert stats.failed == 1


def test_a_full_queue_drops_rather_than_blocks_capture():
    gate = threading.Event()

    class _Blocking(_Session):
        def post(self, url, json=None, timeout=None):
            gate.wait(5.0)
            return super().post(url, json=json, timeout=timeout)

    session = _Blocking([])
    reporter = _reporter(session, queue_size=1)
    reporter.submit(ROW)  # taken by the worker, blocked in post
    _wait(lambda: gate.is_set() or True)
    time.sleep(0.05)
    assert reporter.submit(ROW)  # fills the queue
    assert not reporter.submit(ROW), "third row: queue full, dropped, never blocked"
    gate.set()
    stats = reporter.close()
    assert stats.dropped == 1
    assert stats.submitted == 3


def test_close_is_idempotent_and_refuses_new_rows():
    session = _Session([])
    reporter = _reporter(session)
    reporter.close()
    reporter.close()
    try:
        reporter.submit(ROW)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a closed reporter accepted a row")
