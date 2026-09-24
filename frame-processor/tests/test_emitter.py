"""Message-writer I/O must never stall the packet capture loop."""

from __future__ import annotations

import threading

from app.emitter import EmitterPool


class FakeResponse:
    def __init__(self, status_code: int = 201, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(
        self,
        *,
        status_code: int = 201,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.status_code = status_code
        self.started = started
        self.release = release
        self.messages = []
        self.closed = False

    def post(self, _url, *, json, timeout):
        assert timeout > 0
        self.messages.append(json)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=2), "test did not release blocked emitter"
        return FakeResponse(self.status_code, "failure" if self.status_code >= 400 else "")

    def close(self) -> None:
        self.closed = True


def _message(index: int) -> dict:
    return {"session": {"session_id": f"s-{index}"}}


def test_pool_drains_messages_and_reuses_one_session_per_worker():
    sessions = []

    def factory():
        session = FakeSession()
        sessions.append(session)
        return session

    pool = EmitterPool(enabled=True, workers=1, queue_size=8, session_factory=factory)
    for index in range(3):
        assert pool.submit(_message(index))

    stats = pool.close()

    assert stats.submitted == 3
    assert stats.delivered == 3
    assert stats.failed == 0
    assert stats.dropped == 0
    assert len(sessions) == 1
    assert sessions[0].messages == [_message(0), _message(1), _message(2)]
    assert sessions[0].closed


def test_every_delivered_inference_is_logged_as_it_lands(caplog):
    """One INFO line per tapped inference, naming it, at the moment it lands.

    This is the operator's live view of a running tap: an event appearing is
    how you know the thing works. An every-N summary cannot say which
    inference arrived, and says nothing at all until N of them have.
    """
    session = FakeSession()
    pool = EmitterPool(
        enabled=True, workers=1, queue_size=8, session_factory=lambda: session
    )

    with caplog.at_level("INFO", logger="frameprocessor.emitter"):
        for index in range(3):
            assert pool.submit(_message(index))
        pool.close()

    sent = [r.getMessage() for r in caplog.records if "Tapped inference sent" in r.getMessage()]
    # One line each, naming which inference it was and how many have landed. A
    # single worker delivers in submission order, so this is exact rather than
    # a set comparison, and it pins the count to the line it belongs to.
    assert sent == [
        f"Tapped inference sent (session=s-{index} status=201) [{index + 1} delivered]"
        for index in range(3)
    ]


def test_the_running_count_is_shared_across_workers(caplog):
    """The total counts inferences, not each thread's share of them.

    Every worker increments the one pool counter under its lock, so the
    numbers on the lines form 1..N with no repeats however many threads
    delivered them — which is the only property that makes a tail of this
    log readable as progress.
    """
    pool = EmitterPool(
        enabled=True, workers=4, queue_size=32, session_factory=FakeSession
    )

    with caplog.at_level("INFO", logger="frameprocessor.emitter"):
        for index in range(12):
            assert pool.submit(_message(index))
        stats = pool.close()

    counts = sorted(
        int(record.getMessage().split("[")[1].split(" ")[0])
        for record in caplog.records
        if "Tapped inference sent" in record.getMessage()
    )
    assert counts == list(range(1, 13))
    assert stats.delivered == 12


def test_a_backlogged_queue_warns_once_rather_than_per_message(caplog):
    """The depth warning is a signal, not a running commentary.

    It fires while there is still headroom to act on — a drop only tells you
    records are already lost — and then stays quiet, because the condition
    persists and repeating it would be its own flood.
    """
    release = threading.Event()
    session = FakeSession(release=release)
    pool = EmitterPool(
        enabled=True, workers=1, queue_size=4, session_factory=lambda: session
    )

    with caplog.at_level("WARNING", logger="frameprocessor.emitter"):
        # The worker blocks on the first message, so the rest pile up.
        for index in range(4):
            pool.submit(_message(index))
        release.set()
        pool.close()

    warnings = [r for r in caplog.records if "Emitter queue" in r.getMessage()]
    assert len(warnings) == 1
    assert "not keeping up" in warnings[0].getMessage()


def test_a_healthy_queue_says_nothing_about_depth(caplog):
    session = FakeSession()
    pool = EmitterPool(
        enabled=True, workers=1, queue_size=64, session_factory=lambda: session
    )

    with caplog.at_level("WARNING", logger="frameprocessor.emitter"):
        assert pool.submit(_message(0))
        pool.close()

    assert [r for r in caplog.records if "Emitter queue" in r.getMessage()] == []


def test_submit_does_not_wait_for_slow_message_writer():
    started, release = threading.Event(), threading.Event()
    session = FakeSession(started=started, release=release)
    pool = EmitterPool(
        enabled=True, workers=1, queue_size=2, session_factory=lambda: session
    )
    try:
        assert pool.submit(_message(1))
        assert started.wait(timeout=1)
        # The worker is blocked in HTTP, but capture can continue handing off.
        assert pool.submit(_message(2))
    finally:
        release.set()

    stats = pool.close()
    assert stats.delivered == 2


def test_full_queue_drops_explicitly_instead_of_blocking_capture(caplog):
    started, release = threading.Event(), threading.Event()
    session = FakeSession(started=started, release=release)
    pool = EmitterPool(
        enabled=True, workers=1, queue_size=1, session_factory=lambda: session
    )
    try:
        assert pool.submit(_message(1))
        assert started.wait(timeout=1)  # first is in flight
        assert pool.submit(_message(2))  # second occupies the sole queue slot
        assert pool.submit(_message(3)) is False
    finally:
        release.set()

    stats = pool.close()
    assert stats.submitted == 2
    assert stats.delivered == 2
    assert stats.dropped == 1
    assert "queue full" in caplog.text.lower()


def test_http_failure_is_counted_and_does_not_kill_worker():
    session = FakeSession(status_code=503)
    pool = EmitterPool(
        enabled=True, workers=1, queue_size=4, session_factory=lambda: session
    )
    assert pool.submit(_message(1))
    assert pool.submit(_message(2))

    stats = pool.close()

    assert stats.failed == 2
    assert stats.delivered == 0


def test_disabled_pool_preserves_pcap_stdout_mode(capsys):
    pool = EmitterPool(enabled=False)
    assert pool.submit(_message(1))

    stats = pool.close()

    assert '"session_id": "s-1"' in capsys.readouterr().out
    assert stats.submitted == 1
    assert stats.delivered == 1


def test_session_factory_failure_drains_instead_of_hanging_close():
    """A worker that cannot open a session must still record completions.

    close() waits on the queue, so a worker that exits before its first
    task_done() leaves shutdown waiting on completions nobody will ever record.
    Draining-and-failing keeps the process able to exit.
    """
    def explode() -> FakeSession:
        raise RuntimeError("no sockets today")

    pool = EmitterPool(
        enabled=True, workers=2, queue_size=8, session_factory=explode
    )
    assert pool.submit(_message(1))
    assert pool.submit(_message(2))

    stats = pool.close()  # must return rather than block forever

    assert stats.submitted == 2
    assert stats.failed == 2
    assert stats.delivered == 0


def test_close_gives_up_on_an_undrainable_queue(caplog):
    """An unreachable Message writer must not hold shutdown open.

    Queue.join() has no timeout, so without a deadline a stuck writer outlasts
    any pod grace period and SIGKILL discards the backlog anyway.
    """
    started, release = threading.Event(), threading.Event()
    session = FakeSession(started=started, release=release)
    pool = EmitterPool(
        enabled=True,
        workers=1,
        queue_size=8,
        drain_seconds=0.2,
        session_factory=lambda: session,
    )
    try:
        assert pool.submit(_message(1))
        assert started.wait(timeout=1)  # wedged mid-post
        assert pool.submit(_message(2))

        stats = pool.close()  # bounded, so this returns

        assert stats.submitted == 2
        assert "did not drain" in caplog.text.lower()
    finally:
        release.set()
