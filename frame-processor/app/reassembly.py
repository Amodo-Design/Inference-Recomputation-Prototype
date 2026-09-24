"""TCP stream reassembly: segments → per-connection ordered byte streams.

Segments arrive in any order (and duplicated —
a tap sees retransmissions); this module rebuilds, per connection and per
direction, the contiguous byte stream the endpoints exchanged.

Scaffold simplifications, deliberate and documented:
- exact-duplicate segments are dropped, but partially overlapping
  retransmissions are not trimmed (rare on a healthy LAN; revisit with drop
  counters before trusting production captures);
- no receive-window or checksum handling;
- sequence numbers wrap at 2**32 (handled).

A connection is *complete* when both directions have seen FIN or either saw
RST; `idle_flush()` force-completes silent connections, because a passive tap
has no guarantee of ever seeing a clean close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app import config
from app.capture import TcpSegment

log = logging.getLogger("frameprocessor.reassembly")

_SEQ_MOD = 2**32

Endpoint = tuple[str, int]
ConnectionKey = tuple[Endpoint, Endpoint]


def _connection_key(left: Endpoint, right: Endpoint) -> ConnectionKey:
    """Stable key shared by both directions without allocating a frozenset."""
    return (left, right) if left <= right else (right, left)


@dataclass
class DirectionStream:
    """One direction of a connection: ordered payload bytes + timing.

    `retain` keeps every byte in `data` for callers that parse the stream in
    one go (pcap replay, tests). Live capture parses incrementally from what
    `feed()` returns and leaves it off, so a long-lived connection — the
    gateway's metrics scraping runs to megabytes a minute — does not hold its
    whole history in memory waiting for a flush that may never come.
    """

    retain: bool = True
    next_seq: int | None = None  # seq we expect next; None until first segment
    data: bytearray = field(default_factory=bytearray)
    pending: dict[int, bytes] = field(default_factory=dict)  # out-of-order, by seq
    first_ts: float | None = None
    last_ts: float | None = None
    fin_seq: int | None = None  # seq just past the FIN's data; None = no FIN yet
    # Abandoned: a hole this direction will never fill (see _abandon).
    lost: bool = False
    _stranded: int = 0  # bytes held in `pending`, tracked rather than summed

    def feed(self, seg: TcpSegment) -> bytes:
        """Absorb one segment; return the bytes that just became contiguous."""
        self.first_ts = seg.ts if self.first_ts is None else self.first_ts
        self.last_ts = seg.ts

        if self.next_seq is None:
            # First segment seen for this direction. A SYN consumes one
            # sequence number; data starts at seq+1.
            self.next_seq = (seg.seq + 1) % _SEQ_MOD if seg.syn else seg.seq

        if seg.fin:
            self.fin_seq = (seg.seq + len(seg.payload)) % _SEQ_MOD
        if not seg.payload:
            return b""
        if self.lost:
            # Nothing can be delivered in order again, so buffering only costs
            # memory. Dropped here, cheaply, rather than stranded.
            return b""
        # A retransmission of bytes already handed downstream. Rejected on
        # arrival because it is O(1) here and a whole-dict scan below.
        if _before(seg.seq, self.next_seq):
            return b""

        # The overwhelmingly common case is an in-order segment with no gap.
        # Hand its immutable payload straight to the parser: putting it into
        # ``pending`` only to pop it again used to add a dict mutation, a
        # bytearray copy and a final bytes copy to every TCP data packet.
        if seg.seq == self.next_seq:
            if not self.pending:
                self.next_seq = (self.next_seq + len(seg.payload)) % _SEQ_MOD
                if self.retain:
                    self.data.extend(seg.payload)
                return seg.payload

            fresh = self._release_pending(seg.payload)
            if self._stranded > config.FRAME_PROCESSOR_MAX_STRANDED_BYTES:
                self._abandon()
            return fresh

        # Only genuinely out-of-order data belongs in the pending map.
        held = self.pending.get(seg.seq)
        if held is None or len(held) < len(seg.payload):
            self.pending[seg.seq] = seg.payload
            self._stranded += len(seg.payload) - (len(held) if held else 0)
        if self._stranded > config.FRAME_PROCESSOR_MAX_STRANDED_BYTES:
            self._abandon()
        return b""

    def _release_pending(self, payload: bytes) -> bytes:
        """Accept an in-order payload and release anything it unblocks.

        The no-pending case is returned directly by ``feed``. A join is
        needed only on this rare path where filling a hole releases one or
        more previously held segments.
        """
        started_at = self.next_seq
        assert started_at is not None

        chunks = [payload]
        self.next_seq = (started_at + len(payload)) % _SEQ_MOD
        while self.next_seq in self.pending:
            held = self.pending.pop(self.next_seq)
            self._stranded -= len(held)
            chunks.append(held)
            self.next_seq = (self.next_seq + len(held)) % _SEQ_MOD

        fresh = b"".join(chunks)
        if self.retain:
            self.data.extend(fresh)

        # Stale retransmissions of bytes just consumed — only reachable when
        # next_seq moved, since that is the only thing that can leave a
        # pending entry behind it. Guarded because the scan is O(pending) and
        # a stalled direction never advances: running it per segment anyway is
        # what turned one dropped frame into a tap-wide collapse, the cost
        # rising with the backlog it could never shrink.
        if self.pending and self.next_seq != started_at:
            for seq in [s for s in self.pending if _before(s, self.next_seq)]:
                self._stranded -= len(self.pending.pop(seq))
        return fresh

    def _abandon(self) -> None:
        """Give up on a direction stalled behind a hole nothing will fill.

        No attempt is made to resynchronise. Restarting mid-stream hands the
        parser bytes from the middle of a message it never saw the start of,
        which it cannot recover from — so this only bounds the memory and
        makes the loss visible. The connection stays until it closes or goes
        idle, and whatever it was carrying is lost from here on.
        """
        log.warning(
            "Abandoning a stalled direction: %d bytes stranded behind a missing "
            "segment at %s — capture loss, so every inference after it on this "
            "connection is lost too",
            self._stranded,
            self.next_seq,
        )
        self.pending.clear()
        self._stranded = 0
        self.lost = True

    @property
    def has_gap(self) -> bool:
        return bool(self.pending) or self.lost

    @property
    def closed(self) -> bool:
        """FIN seen AND every byte before it reassembled.

        A tap sees segments out of order, so the FIN routinely arrives before
        the data it follows — the direction is only closed once next_seq has
        caught up with the FIN's sequence number.
        """
        if self.fin_seq is None or self.next_seq is None:
            return False
        return not _before(self.next_seq, self.fin_seq)


def _before(a: int, b: int) -> bool:
    """True if seq a is before b, mod 2**32."""
    return 0 < (b - a) % _SEQ_MOD < _SEQ_MOD // 2


@dataclass
class Connection:
    """Both directions of a TCP connection.

    `initiator` is the client: taken from the handshake where one was captured
    (see ConnectionTable._roles), else the endpoint seen first. Nothing
    downstream re-derives this — http_stream assigns h11's roles straight from
    it — so a wrong guess costs the whole connection.
    """

    initiator: Endpoint
    responder: Endpoint
    directions: dict[Endpoint, DirectionStream] = field(default_factory=dict)
    rst: bool = False
    retain: bool = True
    # Slot for the layer above to hang per-connection parser state on, so it
    # lives and dies with the connection. Reassembly never looks inside it —
    # main.py owns what goes here, keeping this module pure TCP.
    parser: Any = None

    def stream(self, sender: Endpoint) -> DirectionStream:
        try:
            return self.directions[sender]
        except KeyError:
            self.directions[sender] = DirectionStream(retain=self.retain)
            return self.directions[sender]

    @property
    def client_data(self) -> bytes:
        return bytes(self.stream(self.initiator).data)

    @property
    def server_data(self) -> bytes:
        return bytes(self.stream(self.responder).data)

    @property
    def first_ts(self) -> float | None:
        stamps = [d.first_ts for d in self.directions.values() if d.first_ts is not None]
        return min(stamps) if stamps else None

    @property
    def last_ts(self) -> float | None:
        stamps = [d.last_ts for d in self.directions.values() if d.last_ts is not None]
        return max(stamps) if stamps else None

    @property
    def complete(self) -> bool:
        if self.rst:
            return True
        if len(self.directions) != 2:
            return False
        for direction in self.directions.values():
            if not direction.closed:
                return False
        return True

    @property
    def has_gap(self) -> bool:
        return any(d.has_gap for d in self.directions.values())


@dataclass
class Delivery:
    """What one segment yielded: newly contiguous bytes, and whether that
    segment ended the connection."""

    conn: Connection
    from_client: bool
    data: bytes
    ts: float
    completed: bool


class ConnectionTable:
    """Feed segments in, get each segment's newly ordered bytes out."""

    # Remembered closed connections, so late retransmissions arriving after
    # the close don't resurrect a stub connection. Bounded FIFO.
    _CLOSED_MEMORY = 1024

    def __init__(self, retain: bool = True) -> None:
        self._table: dict[ConnectionKey, Connection] = {}
        self._closed: dict[ConnectionKey, None] = {}
        self._retain = retain

    @staticmethod
    def _roles(seg: TcpSegment, sender: Endpoint, receiver: Endpoint) -> tuple[Endpoint, Endpoint]:
        """Which endpoint is the client, from the segment that opened the table.

        A SYN with no ACK is a client opening; a SYN-ACK is a server
        answering. Either one names both roles outright, which is what makes
        this safe when frames arrive from two capture sockets and the order
        the reader drains them no longer implies who spoke first — getting it
        backwards hands h11 a response to parse as a request, killing both
        directions of the connection with nothing recovered.

        Without a handshake there is nothing in TCP to go on and first-seen
        wins, as before: a capture that joined mid-connection cannot be
        resynchronised anyway.
        """
        if seg.syn and seg.ack:
            return receiver, sender
        return sender, receiver

    def feed(self, seg: TcpSegment) -> Delivery | None:
        """Feed one segment; None if it was dropped as a late retransmit."""
        sender: Endpoint = (seg.src, seg.sport)
        receiver: Endpoint = (seg.dst, seg.dport)
        key = _connection_key(sender, receiver)

        conn = self._table.get(key)
        if conn is None:
            if key in self._closed:
                if not seg.syn:
                    return None  # late retransmit of a closed connection
                del self._closed[key]  # port pair reused by a fresh connection
            initiator, responder = self._roles(seg, sender, receiver)
            conn = Connection(initiator=initiator, responder=responder, retain=self._retain)
            self._table[key] = conn

        fresh = conn.stream(sender).feed(seg)
        if seg.rst:
            conn.rst = True

        completed = conn.complete
        if completed:
            del self._table[key]
            self._closed[key] = None
            while len(self._closed) > self._CLOSED_MEMORY:
                self._closed.pop(next(iter(self._closed)))
            if conn.has_gap:
                log.warning(
                    "Connection %s completed with reassembly gap(s) — capture drops?", key
                )
        return Delivery(
            conn=conn,
            from_client=sender == conn.initiator,
            data=fresh,
            ts=seg.ts,
            completed=completed,
        )

    def idle_flush(self, now: float, idle_timeout: float) -> list[Connection]:
        """Force-complete connections silent for longer than idle_timeout."""
        flushed = []
        for key in [k for k, c in self._table.items() if c.last_ts and now - c.last_ts > idle_timeout]:
            flushed.append(self._table.pop(key))
        return flushed

    def drain(self) -> list[Connection]:
        """Flush everything (end of a pcap replay)."""
        remaining = list(self._table.values())
        self._table.clear()
        return remaining
