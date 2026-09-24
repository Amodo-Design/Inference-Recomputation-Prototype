"""The health check's socket behaviour — what it tells the local kernel.

The frame on the wire is covered by `test_beats.py`, which imports this
tool's payloads so the wire contract and the tap's declaration cannot drift.
What is left is everything that never reaches the wire but decides what else
does: the flag that stops a UDP-only flow provoking ARP forever.

That one is worth a test rather than a comment because getting it wrong is
invisible. Send without it and the check works perfectly while quietly adding
~9,900 ARP frames a day to the link it exists to account for; assert it
unconditionally and the kernel holds a stale address against a peer that has
gone, on evidence nobody had.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tapped_link_health as health  # noqa: E402

PEER = "192.0.2.1"
PORT = 9999
LINUX_MSG_CONFIRM = 0x800  # the value on the hosts this runs on


class FakeSocket:
    """Records what was sent and with which flags; replies on demand."""

    def __init__(self, replies: list[bytes | None], stop_after: int) -> None:
        self.sent: list[tuple[bytes, int]] = []
        self._replies = replies
        self._stop_after = stop_after
        self.closed = False

    def settimeout(self, _seconds): ...

    def sendto(self, data, flags, address):
        self.sent.append((data, flags))
        if len(self.sent) >= self._stop_after:
            health._stop = True
        return len(data)

    def recvfrom(self, _size):
        if not self._replies:
            raise socket.timeout
        reply = self._replies.pop(0)
        if reply is None:
            raise socket.timeout
        return reply, (PEER, PORT)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(health, "_stop", False)
    # The hosts this deploys to are Linux; the developer's machine may not be,
    # and there the constant is 0. Pin it so the test means the same thing
    # in both places.
    monkeypatch.setattr(health, "MSG_CONFIRM", LINUX_MSG_CONFIRM)
    yield
    health._stop = False


class FakeClock:
    """A monotonic clock that advances one tick per reading.

    The send loop interleaves sending and receiving by comparing the clock
    against its next deadline, so a real clock would make which of the two
    happens on a given pass depend on how fast the machine is. With a tick of
    1 and an interval of 3 the pass order is fixed: send, receive, send,
    receive, receive, send.
    """

    def __init__(self, tick: float = 1.0) -> None:
        self.now = 0.0
        self.tick = tick

    def monotonic(self) -> float:
        current, self.now = self.now, self.now + self.tick
        return current


def run_send(replies: list[bytes | None], sends: int, monkeypatch) -> FakeSocket:
    sock = FakeSocket(replies, stop_after=sends)
    monkeypatch.setattr(health, "_socket", lambda *a, **k: sock)
    monkeypatch.setattr(health, "time", FakeClock())
    health.send(PEER, PORT, "192.0.2.2", interval=3.0)
    return sock


def flags_of(sock: FakeSocket) -> list[int]:
    return [flag for _data, flag in sock.sent]


# --- the sender -------------------------------------------------------------


def test_the_first_request_is_not_confirmed(monkeypatch):
    """Nothing has come back yet, so there is nothing to vouch for. It costs
    one ARP exchange at startup and buys the flag its meaning."""
    sock = run_send([health.REPLY], sends=1, monkeypatch=monkeypatch)
    assert flags_of(sock) == [0]


def test_a_request_is_confirmed_once_a_reply_has_come_back(monkeypatch):
    sock = run_send([health.REPLY, health.REPLY, None], sends=3,
                    monkeypatch=monkeypatch)
    assert flags_of(sock) == [0, LINUX_MSG_CONFIRM, LINUX_MSG_CONFIRM]


def test_a_beacon_nobody_answers_is_never_confirmed(monkeypatch):
    """The far end has gone. Claiming forward progress here would keep the
    kernel holding an address on evidence that stopped arriving — and would
    hide the failure from the one layer meant to report it."""
    sock = run_send([None] * 20, sends=3, monkeypatch=monkeypatch)
    assert flags_of(sock) == [0, 0, 0]


def test_confirmation_lapses_when_the_replies_stop(monkeypatch):
    """One answered round trip does not vouch for every send after it."""
    sock = run_send([health.REPLY, None, None], sends=3, monkeypatch=monkeypatch)
    assert flags_of(sock) == [0, LINUX_MSG_CONFIRM, 0]


def test_only_the_expected_reply_counts_as_an_answer(monkeypatch):
    """Anything else on the port is not evidence of anything."""
    sock = run_send([b"not the reply", None, None], sends=2, monkeypatch=monkeypatch)
    assert flags_of(sock) == [0, 0]


def test_the_payload_is_the_same_every_time(monkeypatch):
    """The pin depends on it; a flag must never leak into the datagram."""
    sock = run_send([health.REPLY, health.REPLY, None], sends=3,
                    monkeypatch=monkeypatch)
    assert {data for data, _flag in sock.sent} == {health.REQUEST}


# --- the responder ----------------------------------------------------------


def test_a_reply_is_confirmed_by_the_request_that_prompted_it(monkeypatch):
    """Weaker evidence than the sender's, and it cannot mislead: no request
    arrives from a peer that is gone, so there is no reply to flag."""
    sock = FakeSocket([health.REQUEST, health.REQUEST], stop_after=2)
    monkeypatch.setattr(health, "_socket", lambda *a, **k: sock)
    health.respond("192.0.2.1", PORT)
    assert flags_of(sock) == [LINUX_MSG_CONFIRM, LINUX_MSG_CONFIRM]
    assert {data for data, _flag in sock.sent} == {health.REPLY}


def test_only_the_expected_request_is_answered(monkeypatch):
    """A responder that echoed whatever arrived would be a way to put chosen
    bytes on the link, which is the opposite of what a declared flow is for."""
    sock = FakeSocket([b"anything else", health.REQUEST], stop_after=1)
    monkeypatch.setattr(health, "_socket", lambda *a, **k: sock)
    health.respond("192.0.2.1", PORT)
    assert [data for data, _flag in sock.sent] == [health.REPLY]
