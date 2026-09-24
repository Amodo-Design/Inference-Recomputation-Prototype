"""Reassembly: ordered streams out of unordered, duplicated segments."""

from __future__ import annotations

import random
import time
from unittest import mock

from app import config
from app.capture import TcpSegment
from app.reassembly import ConnectionTable, DirectionStream
from tests.helpers import CLIENT, PATH_MSS, SERVER, conversation


def _run(segments):
    table = ConnectionTable()
    completed = []
    for seg in segments:
        delivery = table.feed(seg)
        if delivery is not None and delivery.completed:
            completed.append(delivery.conn)
    completed.extend(table.drain())
    return completed


def test_in_order_conversation_reassembles():
    client_bytes = b"C" * 4000  # spans multiple segments
    server_bytes = b"S" * 9000
    completed = _run(conversation(client_bytes, server_bytes))

    assert len(completed) == 1
    conn = completed[0]
    assert conn.complete and not conn.has_gap
    assert conn.initiator == CLIENT and conn.responder == SERVER
    assert conn.client_data == client_bytes
    assert conn.server_data == server_bytes
    assert conn.first_ts < conn.last_ts


def test_out_of_order_and_duplicated_segments_reassemble():
    client_bytes = bytes(range(256)) * 40
    server_bytes = b"the quick brown fox " * 700
    segments = conversation(client_bytes, server_bytes)

    # A tap sees reordering and retransmissions: duplicate some payload
    # segments and shuffle everything after the SYNs (kept first so stream
    # origins are known). A duplicate can land after the close — it must be
    # recognised as a late retransmit, not a new connection.
    syns, rest = segments[:2], segments[2:]
    rng = random.Random(42)
    rest += rng.sample(rest, k=5)  # duplicates
    rng.shuffle(rest)

    completed = _run(syns + rest)
    assert len(completed) == 1
    conn = completed[0]
    assert conn.client_data == client_bytes
    assert conn.server_data == server_bytes
    assert not conn.has_gap


def test_idle_flush_completes_silent_connections():
    segments = conversation(b"hello", b"world")
    no_fins = [s for s in segments if not s.fin]  # capture never saw the close

    table = ConnectionTable()
    for seg in no_fins:
        assert not table.feed(seg).completed

    flushed = table.idle_flush(now=no_fins[-1].ts + 999, idle_timeout=120)
    assert len(flushed) == 1
    assert flushed[0].client_data == b"hello"
    assert flushed[0].server_data == b"world"


def test_feed_returns_newly_contiguous_bytes_in_order():
    """Live parsing consumes what each segment released, so out-of-order
    arrivals must surface only once their gap is filled — and then in full."""
    segments = conversation(b"abcdefghij", b"", mtu=2)
    payloads = [s for s in segments if s.payload]
    held_back = payloads[1]  # arrives late, blocking everything behind it

    table = ConnectionTable()
    delivered = b""
    for seg in [s for s in segments if s is not held_back]:
        delivery = table.feed(seg)
        if delivery is not None:
            delivered += delivery.data
    assert delivered == b"ab"  # stalled at the gap

    delivered += table.feed(held_back).data
    assert delivered == b"abcdefghij"  # gap filled, the rest released at once


def test_retain_false_keeps_no_history():
    """Live capture parses as it goes; holding the bytes too would mean a
    busy connection accumulating its whole history until the flush."""
    table = ConnectionTable(retain=False)
    seen = b""
    for seg in conversation(b"hello there", b"world"):
        delivery = table.feed(seg)
        if delivery is not None and delivery.from_client:
            seen += delivery.data

    assert seen == b"hello there"  # every byte was delivered exactly once
    flushed = table.drain()
    assert not flushed or flushed[0].client_data == b""  # but none was kept


# --- Capture loss: a hole no retransmission will ever fill ---------------
#
# A dropped frame is lost at the tap, not on the wire: the endpoints are
# having a healthy conversation and never learn an observer missed a packet,
# so nothing is resent and every later byte on that direction strands behind
# the hole. These cover what that must cost — bounded memory, constant time,
# and a direction that says so — rather than the collapse it used to cause.

# One segment as measured on the tapped path, so a backlog here is sized in
# the same units a real one is.
_SEGMENT = b"x" * PATH_MSS
# Bigger than any backlog these tests build: lets the cost of a stall be
# measured on its own, with the bound that would otherwise cut it short off.
_NO_BOUND = 1 << 40
# Enough feeds to time without the measurement itself perturbing the backlog.
_TIMED_FEEDS = 200
# A linear scan over 200x the backlog would cost ~200x. The bar is set well
# below that and well above measurement noise, because this is a timing test.
_MAX_COST_RATIO = 5


def _seg(seq, payload=_SEGMENT):
    return TcpSegment(ts=0.0, src=CLIENT[0], sport=CLIENT[1], dst=SERVER[0], dport=SERVER[1],
                      seq=seq, syn=False, ack=True, fin=False, rst=False, payload=payload)


def _stall(count, bound=_NO_BOUND):
    """A direction holding `count` segments behind one missing segment.

    Returns the stream and the sequence just past the backlog, so a caller
    feeding it more adds to the backlog rather than re-feeding what is already
    held — which exercises nothing, since a duplicate key is not reinserted.
    """
    stream = DirectionStream(retain=False)
    stream.feed(_seg(0))
    first_after_hole = 2 * len(_SEGMENT)  # the segment at len(_SEGMENT) never arrives
    with mock.patch.object(config, "FRAME_PROCESSOR_MAX_STRANDED_BYTES", bound):
        for i in range(count):
            stream.feed(_seg(first_after_hole + i * len(_SEGMENT)))
    return stream, first_after_hole + count * len(_SEGMENT)


def test_a_stalled_direction_costs_the_same_however_much_it_holds():
    """The cost of one segment must not scale with the backlog behind it.

    It used to: the stale-retransmission sweep scanned the whole pending dict
    on every segment, so a stalled direction slowed as it grew — and the
    slower it ran the more frames the kernel dropped, which stalled more
    directions. Measured on the tapped link, that spiral took the whole tap
    from ~99k segments/sec to ~140, against ~860 arriving.

    The bound stays off throughout, so this measures the scan alone rather
    than the stream being abandoned out from under the timer.
    """
    def cost(backlog):
        stream, next_free = _stall(backlog)
        with mock.patch.object(config, "FRAME_PROCESSOR_MAX_STRANDED_BYTES", _NO_BOUND):
            start = time.perf_counter()
            for i in range(_TIMED_FEEDS):
                stream.feed(_seg(next_free + i * len(_SEGMENT)))
            elapsed = time.perf_counter() - start
        assert not stream.lost  # the bound must not be what made this cheap
        return elapsed

    small, large = cost(100), cost(100 * 200)
    assert large < small * _MAX_COST_RATIO, (
        f"cost grew with the backlog: {small:.4f}s -> {large:.4f}s"
    )


def test_a_stalled_direction_is_abandoned_rather_than_held_forever():
    stream, next_free = _stall(count=10_000, bound=1024 * 1024)

    assert stream.lost
    assert stream.pending == {}  # memory released, not held for the process's life
    assert stream.has_gap  # still reported as damaged, not as clean

    # And it stays released: nothing can be delivered in order again, so
    # later segments are dropped on arrival instead of stranding too.
    stream.feed(_seg(next_free))
    assert stream.pending == {}


def test_an_in_order_stream_never_strands_or_abandons():
    stream = DirectionStream(retain=False)
    for i in range(2000):
        assert stream.feed(_seg(i * len(_SEGMENT))) == _SEGMENT
    assert not stream.lost and not stream.has_gap and stream.pending == {}


def test_in_order_fast_path_returns_the_original_payload_without_buffering():
    """The live hot path hands immutable segment bytes straight downstream."""
    payload = bytes(bytearray(b"not-an-interned-payload"))
    stream = DirectionStream(retain=False)

    delivered = stream.feed(_seg(0, payload))

    assert delivered is payload
    assert stream.pending == {}
    assert stream.data == b""


def test_gap_filler_joins_itself_with_every_newly_contiguous_segment():
    stream = DirectionStream(retain=False)
    first = b"aa"
    gap = b"bb"
    held_one = b"cc"
    held_two = b"dd"

    assert stream.feed(_seg(0, first)) is first
    assert stream.feed(_seg(6, held_two)) == b""
    assert stream.feed(_seg(4, held_one)) == b""
    assert stream.feed(_seg(2, gap)) == gap + held_one + held_two
    assert stream.pending == {}
    assert not stream.has_gap


def test_retransmission_of_delivered_bytes_is_ignored():
    """Rejected on arrival now, where it costs one comparison rather than a
    scan of everything pending."""
    stream = DirectionStream(retain=False)
    assert stream.feed(_seg(0)) == _SEGMENT
    assert stream.feed(_seg(len(_SEGMENT))) == _SEGMENT
    assert stream.feed(_seg(0)) == b""  # already handed downstream
    assert stream.pending == {}
    assert not stream.has_gap
