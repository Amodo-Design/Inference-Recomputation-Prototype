"""Capturing a hardware tap: two monitor ports feeding one pipeline.

A passive tap on a full-duplex link presents one monitor port per direction,
so live capture runs one `RingCapture` per interface. AF_PACKET is Linux only,
so the ring is exercised here over hand-built block bytes — the block walk and
the socket setup are separate concerns, and only the walk carries logic worth
testing.

The direction tests are the important ones: two interfaces mean arrival order
no longer tells you who opened the connection, and guessing wrong costs the
whole connection rather than degrading it.
"""

from __future__ import annotations

import dataclasses
import json
import struct

from app import capture
from app.capture import RingCapture
from app.config import _parse_ifaces
from app.main import process_segments
from app.policy import PACKET_OUTGOING
from app.reassembly import ConnectionTable
from tests.helpers import (
    CLIENT,
    SERVER,
    conversation,
    http_request,
    http_response_json,
    ip_tcp,
    ip_udp,
    vxlan_frame,
)


class FakeStatsSocket:
    def __init__(self, samples=()) -> None:
        self.samples = list(samples)
        self.closed = False

    def getsockopt(self, *_args):
        return self.samples.pop(0)

    def fileno(self):
        return 123

    def close(self):
        self.closed = True


_RING_BLOCK_SIZE = 4096
_RING_PACKET_AT = 64
_RING_MAC_AT = 48


def _align(value: int, alignment: int = 16) -> int:
    return (value + alignment - 1) // alignment * alignment


def _fake_ring(block_count: int = 2, *, stats=()) -> RingCapture:
    """A RingCapture with real block bytes and no Linux socket dependency."""
    source = RingCapture.__new__(RingCapture)
    source.iface = "tap-test"
    source.received = 0
    source.drops = 0
    source.freeze_q_count = 0
    source.truncated_packets = 0
    source.malformed_blocks = 0
    source.blocks_released = 0
    source._block_size = _RING_BLOCK_SIZE
    source._block_count = block_count
    source._block_index = 0
    source._active_packets = 0
    source._active_packet_at = 0
    source._closed = False
    source._ring = bytearray(_RING_BLOCK_SIZE * block_count)
    source._sock = FakeStatsSocket(stats)
    return source


def _fill_ring_block(
    source: RingCapture,
    block: int,
    packets: list[tuple[bytes, int, int]],
    *,
    original_lengths: list[int] | None = None,
    packet_statuses: list[int] | None = None,
    mac_at: int = _RING_MAC_AT,
    pkttypes: list[int] | None = None,
) -> None:
    """Write the subset of block/tpacket3 headers RingCapture consumes.

    ``mac_at`` defaults to a frame offset *inside* where a real kernel puts
    the sockaddr_ll, which is how most of these tests were written; pass the
    kernel's real layout (>= TPACKET3_HDRLEN) with ``pkttypes`` to exercise
    the direction read.
    """
    base = block * source._block_size
    ring = source._ring
    struct.pack_into("=I", ring, base + capture._TPACKET_BLOCK_STATUS_AT, capture._TP_STATUS_USER)
    struct.pack_into("=I", ring, base + capture._TPACKET_BLOCK_NUM_PKTS_AT, len(packets))
    struct.pack_into("=I", ring, base + capture._TPACKET_BLOCK_FIRST_PKT_AT, _RING_PACKET_AT)

    packet_at = base + _RING_PACKET_AT
    for index, (frame, sec, nsec) in enumerate(packets):
        stride = _align(mac_at + len(frame))
        next_offset = stride if index + 1 < len(packets) else 0
        original_len = (
            original_lengths[index] if original_lengths is not None else len(frame)
        )
        status = packet_statuses[index] if packet_statuses is not None else 0
        struct.pack_into(
            "=6I",
            ring,
            packet_at,
            next_offset,
            sec,
            nsec,
            len(frame),
            original_len,
            status,
        )
        struct.pack_into("=H", ring, packet_at + capture._TPACKET3_MAC_AT, mac_at)
        if pkttypes is not None:
            ring[packet_at + capture._TPACKET3_PKTTYPE_AT] = pkttypes[index]
        frame_at = packet_at + mac_at
        ring[frame_at : frame_at + len(frame)] = frame
        packet_at += stride
    assert packet_at <= base + source._block_size


# --- TPACKET_V3 block-ring capture ---


def _wire(seg) -> bytes:
    return vxlan_frame(ip_tcp(seg))


def test_ring_drains_every_ready_block_and_uses_kernel_timestamps():
    source = _fake_ring(2)
    segments = conversation(b"request", b"response")
    _fill_ring_block(
        source,
        0,
        [(_wire(segments[0]), 1700000001, 125_000_000)],
    )
    _fill_ring_block(
        source,
        1,
        [(_wire(segments[2]), 1700000002, 750_000_000)],
    )

    decoded = source.read_segments()

    assert [seg.ts for seg in decoded] == [1700000001.125, 1700000002.75]
    assert [seg.payload for seg in decoded] == [b"", b"request"]
    assert struct.unpack_from("=I", source._ring, capture._TPACKET_BLOCK_STATUS_AT)[0] == 0
    assert struct.unpack_from(
        "=I", source._ring, _RING_BLOCK_SIZE + capture._TPACKET_BLOCK_STATUS_AT
    )[0] == 0
    assert source._block_index == 0


class _RecordingObserver:
    def __init__(self) -> None:
        self.seen: list[tuple[bytes, str | None]] = []

    def observe(self, data, ts, direction=None):
        self.seen.append((bytes(data), direction))
        return "ipv4-tcp"

    def note_truncated(self, original_len, captured_len, ts=0.0):
        pass


def test_ring_hands_the_observer_the_kernel_direction():
    """sll_pkttype follows the tpacket3_hdr; PACKET_OUTGOING means we sent it.

    That bit is what lets the link whitelist tell the tapped node's frames
    from this host's without trusting a source MAC the peer wrote.
    """
    source = _fake_ring(1)
    observer = _RecordingObserver()
    source.attach_observer(observer)
    segments = conversation(b"request", b"response")
    # 96 is where a real V3 ring puts the frame: TPACKET_ALIGN(68 + 16) - 14
    # rounded up by the kernel's reserve; anything >= 68 leaves the
    # sockaddr_ll intact.
    _fill_ring_block(
        source,
        0,
        [(_wire(segments[0]), 1, 0), (_wire(segments[1]), 2, 0)],
        mac_at=96,
        pkttypes=[PACKET_OUTGOING, 0],
    )

    source.read_segments()

    assert [direction for _frame, direction in observer.seen] == ["in", "out"]
    assert [frame for frame, _direction in observer.seen] == [
        _wire(segments[0]),
        _wire(segments[1]),
    ]


def test_ring_gives_no_direction_when_the_frame_overlaps_the_sockaddr():
    """The legacy fake layout puts the frame at 48, over the sockaddr_ll: the
    walk must not read frame bytes as a packet type."""
    source = _fake_ring(1)
    observer = _RecordingObserver()
    source.attach_observer(observer)
    segments = conversation(b"request", b"response")
    _fill_ring_block(source, 0, [(_wire(segments[0]), 1, 0)])

    source.read_segments()

    assert [direction for _frame, direction in observer.seen] == [None]


def test_ring_rejects_udp_in_place_then_continues_to_next_block(monkeypatch):
    source = _fake_ring(2)
    udp = vxlan_frame(
        ip_udp(b"x" * 64, sport=41000, dport=5201, src="10.244.1.1", dst="10.244.2.2")
    )
    tcp = conversation(b"request", b"response")[2]
    _fill_ring_block(source, 0, [(udp, 1, 0)])
    _fill_ring_block(source, 1, [(_wire(tcp), 2, 0)])

    def should_not_decode(*_args, **_kwargs):
        raise AssertionError("common ring traffic reached dpkt")

    monkeypatch.setattr(capture, "_decode", should_not_decode)
    decoded = source.read_segments()

    assert [seg.payload for seg in decoded] == [b"request"]
    assert source._block_index == 0


def test_ring_block_budget_bounds_a_turn_even_when_no_tcp_is_returned():
    source = _fake_ring(3)
    udp = vxlan_frame(
        ip_udp(b"x" * 64, sport=41000, dport=5201, src="10.244.1.1", dst="10.244.2.2")
    )
    for block in range(3):
        _fill_ring_block(source, block, [(udp, block + 1, 0)])

    assert source.read_segments(max_blocks=2) == []
    assert source.blocks_released == 2
    assert source._block_index == 2
    # The third ready block is left for the next receiver turn, allowing the
    # caller to dispatch candidates and check stop/statistics between turns.
    assert struct.unpack_from(
        "=I", source._ring, 2 * _RING_BLOCK_SIZE + capture._TPACKET_BLOCK_STATUS_AT
    )[0] == capture._TP_STATUS_USER

    assert source.read_segments(max_blocks=2) == []
    assert source.blocks_released == 3


def test_ring_default_turn_is_bounded_under_a_continuously_ready_ring():
    budget = capture.DEFAULT_RING_READ_BLOCKS
    # One more ready block than a turn may release, so the bound is what stops
    # the walk rather than the ring running out.
    source = _fake_ring(budget + 1)
    udp = vxlan_frame(
        ip_udp(b"x" * 64, sport=41000, dport=5201, src="10.244.1.1", dst="10.244.2.2")
    )
    for block in range(budget + 1):
        _fill_ring_block(source, block, [(udp, block + 1, 0)])

    assert source.read_segments() == []
    assert source.blocks_released == budget
    assert source._block_index == budget
    # The block beyond the budget is untouched, still owned by userspace.
    assert struct.unpack_from(
        "=I", source._ring, budget * _RING_BLOCK_SIZE + capture._TPACKET_BLOCK_STATUS_AT
    )[0] == capture._TP_STATUS_USER


def test_ring_limit_resumes_inside_a_block_without_retaining_views():
    source = _fake_ring(1)
    segments = conversation(b"request", b"response")
    candidates = [segments[0], segments[2], segments[3]]
    _fill_ring_block(
        source,
        0,
        [(_wire(seg), 10 + index, index) for index, seg in enumerate(candidates)],
    )

    first = source.read_segments(limit=1)
    assert len(first) == 1
    assert source._active_packets == 2
    assert struct.unpack_from("=I", source._ring, capture._TPACKET_BLOCK_STATUS_AT)[0] == 1

    second = source.read_segments(limit=1)
    third = source.read_segments(limit=1)

    assert [seg.payload for seg in first + second + third] == [b"", b"request", b"response"]
    assert source._active_packets == 0
    assert struct.unpack_from("=I", source._ring, capture._TPACKET_BLOCK_STATUS_AT)[0] == 0
    assert all(type(seg.payload) is bytes for seg in first + second + third)


def test_ring_drops_and_reports_a_truncated_slot(caplog):
    source = _fake_ring(1)
    frame = _wire(conversation(b"request", b"response")[2])
    _fill_ring_block(source, 0, [(frame, 1, 0)], original_lengths=[len(frame) + 100])

    with caplog.at_level("WARNING", logger="frameprocessor.capture"):
        assert source.read_segments() == []

    assert source.truncated_packets == 1
    assert "truncated" in caplog.text
    assert struct.unpack_from("=I", source._ring, capture._TPACKET_BLOCK_STATUS_AT)[0] == 0


def test_ring_copy_status_is_treated_as_truncation(caplog):
    source = _fake_ring(1)
    frame = _wire(conversation(b"request", b"response")[2])
    _fill_ring_block(
        source,
        0,
        [(frame, 1, 0)],
        packet_statuses=[capture._TP_STATUS_COPY],
    )

    with caplog.at_level("WARNING", logger="frameprocessor.capture"):
        assert source.read_segments() == []

    assert source.truncated_packets == 1


def test_ring_packet_statistics_include_interval_and_cumulative_totals():
    source = _fake_ring(
        stats=[struct.pack("=III", 1000, 7, 2), struct.pack("=III", 500, 3, 1)]
    )

    assert source.poll_stats() == capture.PacketStats(1000, 7, 2)
    assert source.poll_stats() == capture.PacketStats(500, 3, 1)
    assert (source.received, source.drops, source.freeze_q_count) == (1500, 10, 3)


def test_ring_releases_a_malformed_block_instead_of_jamming(caplog):
    source = _fake_ring(1)
    struct.pack_into(
        "=I", source._ring, capture._TPACKET_BLOCK_STATUS_AT, capture._TP_STATUS_USER
    )
    struct.pack_into("=I", source._ring, capture._TPACKET_BLOCK_NUM_PKTS_AT, 1)
    struct.pack_into("=I", source._ring, capture._TPACKET_BLOCK_FIRST_PKT_AT, 2)

    with caplog.at_level("ERROR", logger="frameprocessor.capture"):
        assert source.read_segments() == []

    assert source.malformed_blocks == 1
    assert "malformed" in caplog.text
    assert struct.unpack_from("=I", source._ring, capture._TPACKET_BLOCK_STATUS_AT)[0] == 0


def test_ring_rejects_invalid_kernel_timestamp_and_unaligned_packet_stride(caplog):
    frame = _wire(conversation(b"request", b"response")[2])

    bad_time = _fake_ring(1)
    _fill_ring_block(bad_time, 0, [(frame, 1, 1_000_000_000)])
    with caplog.at_level("ERROR", logger="frameprocessor.capture"):
        assert bad_time.read_segments() == []
    assert bad_time.malformed_blocks == 1

    bad_stride = _fake_ring(1)
    _fill_ring_block(bad_stride, 0, [(frame, 1, 0), (frame, 2, 0)])
    struct.pack_into(
        "=I",
        bad_stride._ring,
        _RING_PACKET_AT + capture._TPACKET3_NEXT_OFFSET_AT,
        49,  # V3's variable-length records are 8-byte aligned
    )
    with caplog.at_level("ERROR", logger="frameprocessor.capture"):
        assert bad_stride.read_segments() == []
    assert bad_stride.malformed_blocks == 1


def test_ring_geometry_is_validated_before_opening_a_socket():
    try:
        RingCapture._validate_geometry(4097, 1, 2048, 64)
    except ValueError as exc:
        assert "page size" in str(exc)
    else:  # pragma: no cover - assertion message is clearer than pytest.raises here
        raise AssertionError("misaligned block_size accepted")

    try:
        RingCapture._validate_geometry(capture.mmap.PAGESIZE, 1, 64, 64)
    except ValueError as exc:
        assert "at least 68" in str(exc)
    else:
        raise AssertionError("frame smaller than TPACKET3_HDRLEN accepted")

    RingCapture._validate_geometry(capture.mmap.PAGESIZE, 2, 2048, 64)


# --- interface configuration ---


def test_iface_list_is_parsed_and_deduplicated():
    """Two sockets on one interface would double-count every inference."""
    assert _parse_ifaces("mon0,mon1") == ("mon0", "mon1")
    assert _parse_ifaces(" mon0 , mon1 ") == ("mon0", "mon1")
    assert _parse_ifaces("mon0,mon0") == ("mon0",)
    assert _parse_ifaces("mon0") == ("mon0",)
    assert _parse_ifaces("") == ()
    assert _parse_ifaces(None) == ()


# --- direction, when arrival order no longer implies it ---


def _syn_ack_first(segments):
    """Reorder so the server's SYN-ACK is drained before the client's SYN.

    Exactly what happens when the two halves of a handshake arrive on
    different capture sockets and select() reports both ready at once.
    """
    syn_ack = next(s for s in segments if s.syn and s.ack)
    return [syn_ack] + [s for s in segments if s is not syn_ack]


def test_the_handshake_names_the_client_whatever_the_arrival_order():
    table = ConnectionTable()
    segments = _syn_ack_first(conversation(b"request", b"response"))

    deliveries = [table.feed(seg) for seg in segments]
    conn = deliveries[0].conn

    assert conn.initiator == CLIENT
    assert conn.responder == SERVER
    # The client's own segments must still read as client-side.
    assert deliveries[1].from_client is True
    assert deliveries[0].from_client is False


def test_an_inference_survives_the_handshake_arriving_out_of_order():
    """Roles reversed would hand h11 a response to parse as a request, and
    kill both directions with nothing recovered."""
    body = {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "q"}]}
    request = http_request("/v1/chat/completions", json.dumps(body).encode())
    response = http_response_json(
        json.dumps(
            {"id": "cmpl-1", "model": "openai/gpt-oss-120b",
             "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        ).encode()
    )
    segments = conversation(request, response)

    in_order = list(process_segments(iter(segments)))
    reordered = list(process_segments(iter(_syn_ack_first(segments))))

    assert len(in_order) == 1
    # Both halves parsed, not just the request: a reversed role would kill the
    # response direction and leave output_text empty.
    assert in_order[0]["response"]["output_text"] == "hi"
    assert reordered == in_order


def test_a_mid_connection_join_still_falls_back_to_first_seen():
    """With no handshake there is nothing in TCP to go on; behaviour is unchanged."""
    table = ConnectionTable()
    mid = [s for s in conversation(b"request", b"response") if not s.syn]

    conn = table.feed(mid[0]).conn

    assert conn.initiator == CLIENT


def test_a_bare_syn_from_the_server_is_not_read_as_a_client_open():
    """Only SYN+ACK identifies a server; a lone SYN is always a client open."""
    table = ConnectionTable()
    syn = next(s for s in conversation(b"q", b"r") if s.syn and not s.ack)
    server_syn = dataclasses.replace(
        syn, src=SERVER[0], sport=SERVER[1], dst=CLIENT[0], dport=CLIENT[1]
    )

    assert table.feed(server_syn).conn.initiator == SERVER
