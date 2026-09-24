"""Deliver capture windows to the ledger without ever blocking capture.

The same shape as `emitter.EmitterPool`, for the same reason: HTTP to the
ledger must never sit on the path that drains the packet ring, so windows are
handed to a thread through a bounded queue and delivered from there. The
volume is tiny — one row per window per link — so one thread is plenty, and
the queue exists to bound memory during an outage rather than for throughput.

A window is posted straight to ledger-api rather than through the Message
writer. The writer exists to turn a tap message into an inference event and
resolve its ids; a window needs neither, and the ledger derives the row's id
from the window's natural key, so a retry after a crash is idempotent: 409
means the row is already there and counts as delivered.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

log = logging.getLogger("frameprocessor.ledger_reporter")

_STOP = object()


@dataclass(frozen=True)
class ReporterStats:
    submitted: int
    delivered: int
    duplicates: int
    failed: int
    dropped: int
    queued: int


class LedgerReporter:
    """One thread posting window rows to ``POST {ledger_url}/capture-windows``."""

    def __init__(
        self,
        ledger_url: str,
        *,
        path: str = "/capture-windows",
        timeout: float = 10.0,
        retries: int = 3,
        queue_size: int = 64,
        drain_seconds: float = 5.0,
        session_factory: Callable[[], requests.Session] = requests.Session,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = f"{ledger_url.rstrip('/')}{path}"
        self._timeout = timeout
        self._retries = max(0, retries)
        self._drain_seconds = drain_seconds
        self._session_factory = session_factory
        self._sleep = sleep
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(1, queue_size))
        self._lock = threading.Lock()
        self._submitted = self._delivered = self._duplicates = 0
        self._failed = self._dropped = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker, name="frame-processor-ledger-reporter", daemon=True
        )
        self._thread.start()

    def _count(self, name: str) -> int:
        with self._lock:
            value = getattr(self, name) + 1
            setattr(self, name, value)
            return value

    def submit(self, row: dict[str, Any]) -> bool:
        """Queue one window row; never waits. False when the queue was full."""
        with self._lock:
            if self._closed:
                raise RuntimeError("LedgerReporter is closed")
            self._submitted += 1
        try:
            self._queue.put_nowait(row)
            return True
        except queue.Full:
            dropped = self._count("_dropped")
            if dropped == 1 or dropped % 10 == 0:
                log.error(
                    "Ledger reporter queue full; dropped %d window row(s) total — "
                    "window %s is now missing from the ledger",
                    dropped,
                    row.get("window_start"),
                )
            return False

    def _post_once(self, session: requests.Session, row: dict[str, Any]) -> str:
        """'delivered' | 'duplicate' | 'retry' | 'failed'."""
        try:
            response = session.post(self.url, json=row, timeout=self._timeout)
        except requests.RequestException as exc:
            log.warning("Ledger unreachable at %s: %s", self.url, exc)
            return "retry"
        if response.status_code == 409:
            return "duplicate"
        if response.status_code < 300:
            return "delivered"
        if response.status_code >= 500:
            log.warning("Ledger returned %s for a window row; will retry", response.status_code)
            return "retry"
        log.error(
            "Ledger rejected window %s with %s: %s",
            row.get("window_start"),
            response.status_code,
            response.text[:500],
        )
        return "failed"

    def _deliver(self, session: requests.Session, row: dict[str, Any]) -> None:
        for attempt in range(self._retries + 1):
            outcome = self._post_once(session, row)
            if outcome == "delivered":
                self._count("_delivered")
                log.info(
                    "Recorded capture window %s → %s (%s, %d finding(s))",
                    row.get("window_start"),
                    row.get("window_end"),
                    "complete" if row.get("complete") else "INCOMPLETE",
                    row.get("finding_count", 0),
                )
                return
            if outcome == "duplicate":
                self._count("_duplicates")
                return
            if outcome == "failed":
                break
            if attempt < self._retries:
                self._sleep(min(30.0, 2.0**attempt))
        self._count("_failed")
        log.error(
            "Gave up on capture window %s after %d attempt(s); it is missing from the ledger",
            row.get("window_start"),
            self._retries + 1,
        )

    def _worker(self) -> None:
        session = self._session_factory()
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                try:
                    self._deliver(session, item)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001 -- the thread must outlive one bad row
                    self._count("_failed")
                    log.exception("Ledger reporter failed on a window row")
        finally:
            session.close()

    @property
    def stats(self) -> ReporterStats:
        with self._lock:
            return ReporterStats(
                self._submitted,
                self._delivered,
                self._duplicates,
                self._failed,
                self._dropped,
                self._queue.qsize(),
            )

    def close(self) -> ReporterStats:
        """Stop accepting rows, drain briefly, and return the final counters."""
        with self._lock:
            already_closed = self._closed
            self._closed = True
        if already_closed:
            return self.stats
        deadline = time.monotonic() + self._drain_seconds
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()) + 1.0)
        stats = self.stats
        if stats.queued:
            log.error(
                "Ledger reporter stopped with %d window row(s) undelivered", stats.queued
            )
        return stats
