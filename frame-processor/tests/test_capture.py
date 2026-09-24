"""Frame decoding, including the CNI's VXLAN encapsulation.

A tap on a physical link sees pod traffic only after the CNI has wrapped it,
so the decisive test is that the pipeline cannot tell the two vantage points
apart: the same conversation, tunnelled, must produce the same tap message.

The rest guard the two ways decapsulation could go wrong — swallowing bare
TCP that already worked, or stripping eight bytes off something that only
looked like a tunnel.
"""

from __future__ import annotations

import dataclasses
import json
import mmap
import socket
import struct

import dpkt

from app import capture, config

# Frame geometry these fixtures hand-place bytes at. Spelled out rather than
# imported from app.capture for the same reason tests/helpers.py does: a wrong
# offset in the code under test must not be able to hide behind a matching one
# here.
_ETHERTYPE_AT = 12  # past both 6-byte MAC addresses
_IPV6_HEADER_LEN = 40  # fixed, before any extension headers
from app.capture import Frame, TcpSegment, decode_tcp
from app.main import process_segments
from tests.helpers import (
    NODE_A,
    NODE_B,
    PATH_MSS,
    as_vxlan_frames,
    coalesced_vxlan_frame,
    padded_frame,
    conversation,
    ethernet,
    http_request,
    http_response_json,
    ip_tcp,
    ip_udp,
    vxlan_frame,
)

POD_CLIENT = ("10.244.0.10", 44444)
POD_SERVER = ("10.244.2.20", 8000)
REQUEST = b"POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\n\r\n"


def _segment(payload: bytes = REQUEST, src=POD_CLIENT, dst=POD_SERVER) -> TcpSegment:
    return TcpSegment(
        ts=1700000000.0, src=src[0], sport=src[1], dst=dst[0], dport=dst[1],
        seq=1000, syn=False, ack=True, fin=False, rst=False, payload=payload,
    )


def _decode(data: bytes) -> TcpSegment | None:
    return decode_tcp(Frame(ts=1700000000.0, data=data))


# --- decapsulation is additive ---


def test_bare_tcp_still_decodes():
    """The tapped link carries plain TCP too — image pulls, kubelet, NodePort."""
    seg = _decode(ethernet(ip_tcp(_segment())))

    assert seg is not None
    assert (seg.src, seg.sport) == POD_CLIENT
    assert (seg.dst, seg.dport) == POD_SERVER
    assert seg.payload == REQUEST


def test_vxlan_yields_the_inner_addresses_not_the_outer_ones():
    """The TCP header is inner, so its addresses must be inner to match."""
    seg = _decode(vxlan_frame(ip_tcp(_segment())))

    assert seg is not None
    assert (seg.src, seg.sport) == POD_CLIENT
    assert (seg.dst, seg.dport) == POD_SERVER
    assert seg.src not in (NODE_A, NODE_B)
    assert seg.payload == REQUEST


def test_iana_tunnel_port_decodes_as_well_as_flannels():
    assert _decode(vxlan_frame(ip_tcp(_segment()), port=4789)) is not None


def test_tcp_flags_and_sequence_survive_decapsulation():
    syn = dataclasses.replace(_segment(payload=b""), syn=True)
    seg = _decode(vxlan_frame(ip_tcp(syn)))

    assert seg is not None and seg.syn and seg.seq == 1000


# --- and does not fire on things that merely resemble a tunnel ---


def test_plain_udp_is_still_ignored():
    assert _decode(ethernet(ip_udp(b"\x00\x01", sport=33333, dport=53))) is None


def test_tunnel_port_without_the_vni_flag_is_ignored():
    """Some other service on 8472 must not have eight bytes lopped off."""
    assert _decode(vxlan_frame(ip_tcp(_segment()), vxlan_flags=0x00)) is None


def test_truncated_tunnel_payload_is_ignored():
    assert _decode(ethernet(ip_udp(b"\x08\x00\x00", sport=39999, dport=8472))) is None


def test_tunnel_carrying_non_tcp_is_ignored():
    inner = ip_udp(b"\x00\x01", sport=33333, dport=53, src="10.244.0.10", dst="10.96.0.10")
    assert _decode(vxlan_frame(inner)) is None


def test_fast_classifier_rejects_inner_udp_before_dpkt(monkeypatch):
    """High-rate VXLAN/UDP is classified from raw headers, without dpkt."""
    inner = ip_udp(b"x" * 64, sport=40000, dport=5201, src="10.244.1.10", dst="10.244.2.10")
    frame = Frame(ts=1700000000.0, data=vxlan_frame(inner))

    monkeypatch.setattr(config, "FRAME_PROCESSOR_FAST_CLASSIFY", True)

    def should_not_decode(*_args, **_kwargs):
        raise AssertionError("definitely-non-TCP frame reached dpkt")

    monkeypatch.setattr(capture, "_decode", should_not_decode)
    assert decode_tcp(frame) is None


def test_common_tcp_path_decodes_without_dpkt(monkeypatch):
    """Relevant VXLAN/TCP is decoded once, not classified then handed to dpkt."""
    frame = Frame(ts=1700000000.125, data=vxlan_frame(ip_tcp(_segment())))

    monkeypatch.setattr(config, "FRAME_PROCESSOR_FAST_CLASSIFY", True)

    def should_not_decode(*_args, **_kwargs):
        raise AssertionError("common IPv4/VXLAN/TCP frame reached dpkt")

    monkeypatch.setattr(capture, "_decode", should_not_decode)
    seg = decode_tcp(frame)

    assert seg is not None
    assert seg.ts == frame.ts
    assert seg.src == POD_CLIENT[0]
    assert seg.payload == REQUEST


def test_raw_decoder_accepts_a_ring_memoryview_and_copies_payload():
    raw = bytearray(vxlan_frame(ip_tcp(_segment())))
    seg = capture.decode_tcp_bytes(memoryview(raw), 1700000000.5)

    assert seg is not None
    assert type(seg.payload) is bytes
    before = seg.payload
    raw[:] = b"\x00" * len(raw)
    assert seg.payload == before


def test_candidate_decoder_drops_pure_acks_but_keeps_tcp_lifecycle():
    ack = padded_frame(dataclasses.replace(_segment(payload=b""), ack=True))
    syn = ethernet(ip_tcp(dataclasses.replace(_segment(payload=b""), syn=True)))
    fin = ethernet(ip_tcp(dataclasses.replace(_segment(payload=b""), fin=True)))

    assert capture.decode_tcp_bytes(ack, 1.0, candidate_only=True) is None
    assert capture.decode_tcp_bytes(syn, 2.0, candidate_only=True).syn
    assert capture.decode_tcp_bytes(fin, 3.0, candidate_only=True).fin


def test_candidate_decoder_rejects_configured_ports_before_copy(monkeypatch):
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PORTS", frozenset({9000}))
    monkeypatch.setattr(config, "FRAME_PROCESSOR_EXCLUDE_PORTS", frozenset())
    frame = vxlan_frame(ip_tcp(_segment()))  # inner destination is 8000

    assert capture.decode_tcp_bytes(frame, 1.0, candidate_only=True) is None
    # The public decoder retains its old unfiltered contract.
    assert capture.decode_tcp_bytes(frame, 1.0) is not None

    monkeypatch.setattr(config, "FRAME_PROCESSOR_PORTS", frozenset({8000}))
    monkeypatch.setattr(config, "FRAME_PROCESSOR_EXCLUDE_PORTS", frozenset({8000}))
    assert capture.decode_tcp_bytes(frame, 1.0, candidate_only=True) is None


def test_fast_classifier_can_be_disabled(monkeypatch):
    """The optimisation is an operational switch, not a hard coverage rule."""
    inner = ip_udp(b"x", sport=40000, dport=5201)
    frame = Frame(ts=1700000000.0, data=vxlan_frame(inner))
    called = False

    def decoded(*_args, **_kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(config, "FRAME_PROCESSOR_FAST_CLASSIFY", False)
    monkeypatch.setattr(capture, "_decode", decoded)
    assert decode_tcp(frame) is None
    assert called


def test_fast_path_falls_back_to_dpkt_on_ipv6(monkeypatch):
    """An optimisation must never silently reject a layout it cannot prove.

    Asserted against the decoder production actually calls. A helper-level
    assertion would keep passing if the real fast path started dropping
    unfamiliar layouts outright, which is the failure worth guarding.
    """
    # Bare Ethernet carrying IPv6: two zeroed MAC addresses, the IPv6
    # EtherType, then a zeroed 40-byte IPv6 header. Extension headers are
    # deliberately left to dpkt rather than guessed by the raw IPv4 fast path,
    # so this must reach it.
    ipv6 = b"\x00" * _ETHERTYPE_AT + b"\x86\xdd" + b"\x00" * _IPV6_HEADER_LEN
    reached = False

    def fallback(*args, **kwargs):
        nonlocal reached
        reached = True
        return None

    monkeypatch.setattr(config, "FRAME_PROCESSOR_FAST_CLASSIFY", True)
    monkeypatch.setattr(capture, "_decode", fallback)
    assert capture.decode_tcp_bytes(ipv6, 1.0) is None
    assert reached, "IPv6 must fail open into the full decoder, not be dropped"


def test_fast_path_tolerates_a_truncated_frame(monkeypatch):
    """A short frame yields nothing, and must not raise out of the hot loop."""
    monkeypatch.setattr(config, "FRAME_PROCESSOR_FAST_CLASSIFY", True)
    assert capture.decode_tcp_bytes(b"short", 1.0) is None
    assert decode_tcp(Frame(ts=1.0, data=b"short")) is None


def test_fast_path_walks_vlan_tags_without_losing_tcp(monkeypatch):
    bare = vxlan_frame(ip_tcp(_segment()))
    # Insert an 802.1Q TPID + TCI ahead of the original EtherType, which a tag
    # displaces rather than replaces.
    tagged = bare[:_ETHERTYPE_AT] + b"\x81\x00\x00\x01" + bare[_ETHERTYPE_AT:]

    monkeypatch.setattr(config, "FRAME_PROCESSOR_FAST_CLASSIFY", True)
    seg = capture.decode_tcp_bytes(tagged, 1.0)

    assert seg is not None
    assert (seg.src, seg.sport) == POD_CLIENT
    assert seg.payload == REQUEST
    assert _decode(tagged) is not None  # and the fallback agrees


def test_nesting_stops_after_one_layer():
    """A tunnel inside a tunnel is malformed here; it must not recurse."""
    once = ip_udp(
        b"\x08\x00\x00\x00\x00\x00\x01\x00" + vxlan_frame(ip_tcp(_segment())),
        sport=39999,
        dport=8472,
    )
    assert _decode(ethernet(once)) is None


def test_decapsulation_can_be_disabled(monkeypatch):
    """A tap that already sees bare TCP should not be second-guessing UDP."""
    monkeypatch.setattr(config, "FRAME_PROCESSOR_DECAP_PORTS", frozenset())

    assert _decode(vxlan_frame(ip_tcp(_segment()))) is None
    assert _decode(ethernet(ip_tcp(_segment()))) is not None


def test_port_prefilter_sees_the_inner_ports(monkeypatch):
    """FRAME_PROCESSOR_PORTS filters after decapsulation, so it means inner ports.

    The tunnel's own port is not what anyone would configure here, and the
    outer header has no TCP ports at all.
    """
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PORTS", frozenset({8000}))
    request = http_request("/v1/chat/completions", json.dumps({"model": "m", "prompt": "p"}).encode())
    response = http_response_json(json.dumps({"id": "c-1", "choices": [{"text": "hi"}]}).encode())
    segments = conversation(request, response)

    # conversation() talks to port 8080; nothing should survive the pre-filter.
    tunnelled = [s for s in (decode_tcp(f) for f in as_vxlan_frames(segments)) if s is not None]
    assert tunnelled, "decap must still yield segments; the pre-filter runs later"
    assert list(process_segments(iter(tunnelled))) == []

    monkeypatch.setattr(config, "FRAME_PROCESSOR_PORTS", frozenset({8080}))
    assert len(list(process_segments(iter(tunnelled)))) == 1


def test_excluded_ports_are_dropped_whatever_the_allowlist_says(monkeypatch):
    """The denylist wins, so one heavy irrelevant talker can be removed
    without enumerating everything worth keeping.

    This is the safer of the two knobs: an allowlist that misses a port drops
    inference silently, while this can only ever drop what it was named.
    """
    request = http_request("/v1/chat/completions", json.dumps({"model": "m", "prompt": "p"}).encode())
    response = http_response_json(json.dumps({"id": "c-1", "choices": [{"text": "hi"}]}).encode())
    segments = conversation(request, response)  # port 8080

    # Baseline: no filtering at all, the exchange is tapped.
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PORTS", None)
    monkeypatch.setattr(config, "FRAME_PROCESSOR_EXCLUDE_PORTS", frozenset())
    assert len(list(process_segments(iter(segments)))) == 1

    # Excluded even though the allowlist explicitly admits it.
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PORTS", frozenset({8080}))
    monkeypatch.setattr(config, "FRAME_PROCESSOR_EXCLUDE_PORTS", frozenset({8080}))
    assert list(process_segments(iter(segments))) == []

    # An unrelated exclusion (e.g. the ledger's Postgres) leaves it alone.
    monkeypatch.setattr(config, "FRAME_PROCESSOR_EXCLUDE_PORTS", frozenset({5432}))
    assert len(list(process_segments(iter(segments)))) == 1


# --- the raw decoder's header constants ---


def test_raw_decoder_flag_bits_match_dpkts():
    """Two decoders spell the same four bits; this is what keeps them equal.

    _decode reads TCP flags through dpkt.tcp.TH_*; _decode_tcp_raw carries its
    own copies so the fast path need not import dpkt for four integers. They
    must agree, and nothing else in the suite would notice if they stopped.
    """
    assert (capture._TH_FIN, capture._TH_SYN, capture._TH_RST, capture._TH_ACK) == (
        dpkt.tcp.TH_FIN,
        dpkt.tcp.TH_SYN,
        dpkt.tcp.TH_RST,
        dpkt.tcp.TH_ACK,
    )


def test_raw_decoder_header_offsets_match_a_hand_built_packet():
    """The named offsets are checked against a packet dpkt built, not restated.

    A wrong offset here does not raise — it silently misreads every frame on
    the link — so the constants are anchored to an independent encoder.
    """
    ip = dpkt.ip.IP(
        src=socket.inet_aton("10.1.2.3"),
        dst=socket.inet_aton("10.4.5.6"),
        p=dpkt.ip.IP_PROTO_TCP,
        data=dpkt.tcp.TCP(sport=1234, dport=8080),
    )
    raw = bytes(ip)

    assert struct.unpack_from("!H", raw, capture._IPV4_TOTAL_LEN_AT)[0] == len(raw)
    assert raw[capture._IPV4_PROTOCOL_AT] == dpkt.ip.IP_PROTO_TCP
    src_at, dst_at = capture._IPV4_SRC_AT, capture._IPV4_DST_AT
    assert raw[src_at : src_at + capture._IPV4_ADDR_LEN] == socket.inet_aton("10.1.2.3")
    assert raw[dst_at : dst_at + capture._IPV4_ADDR_LEN] == socket.inet_aton("10.4.5.6")

    tcp = raw[capture._MIN_IPV4_HEADER_LEN :]
    data_offset = tcp[capture._TCP_DATA_OFFSET_AT] >> capture._TCP_DATA_OFFSET_SHIFT
    assert data_offset * capture._HEADER_WORD_LEN == capture._MIN_TCP_HEADER_LEN
    assert struct.unpack_from("!H", tcp, capture._UDP_DPORT_AT)[0] == 8080  # same slot

    # Both fragment shapes must trip the mask, and an unfragmented packet with
    # DF set must not — DF lives in the three flag bits the mask discards.
    def fragment_bits(**fields) -> int:
        raw = bytes(dpkt.ip.IP(data=b"", **fields))
        word = struct.unpack_from("!H", raw, capture._IPV4_FLAGS_FRAGMENT_AT)[0]
        return word & capture._IPV4_FRAGMENT_MASK

    assert fragment_bits(mf=1)  # a first or middle fragment
    assert fragment_bits(offset=185)  # a later fragment, carrying no TCP header
    assert not fragment_bits(df=1)  # merely "do not fragment", not a fragment
    assert not fragment_bits()


# --- the shared port/candidate predicate ---


def test_exclusion_beats_the_allowlist_in_the_shared_predicate():
    """Every stage applies this one function, so the rule is asserted once."""
    allow_8080 = frozenset({8080})
    assert capture.ports_allowed(51000, 8080, allow_8080, frozenset())
    assert not capture.ports_allowed(51000, 8080, allow_8080, allow_8080)
    # Either endpoint matching is enough, in both directions.
    assert capture.ports_allowed(8080, 51000, allow_8080, frozenset())
    assert not capture.ports_allowed(51000, 9090, allow_8080, frozenset())
    # No allowlist means everything not explicitly excluded.
    assert capture.ports_allowed(51000, 9090, None, frozenset())
    assert not capture.ports_allowed(51000, 5432, None, frozenset({5432}))


def test_only_segments_carrying_state_are_candidates():
    """A pure ACK advances nothing; SYN/FIN/RST count even when empty."""
    base = dict(
        ts=1.0, src="10.0.0.1", sport=51000, dst="10.0.0.2", dport=8080, seq=1,
        syn=False, ack=False, fin=False, rst=False, payload=b"",
    )
    assert capture.carries_reassembly_state(TcpSegment(**{**base, "payload": b"x"}))
    assert capture.carries_reassembly_state(TcpSegment(**{**base, "syn": True}))
    assert capture.carries_reassembly_state(TcpSegment(**{**base, "fin": True}))
    assert capture.carries_reassembly_state(TcpSegment(**{**base, "rst": True}))
    assert not capture.carries_reassembly_state(TcpSegment(**{**base, "ack": True}))
    # An empty segment with no flags at all carries nothing either, whether or
    # not ACK happens to be set — the receiver and the ring agree on this.
    assert not capture.carries_reassembly_state(TcpSegment(**base))


# --- coalesced frames: one frame, many tunnel packets ---


def test_coalesced_frame_keeps_the_whole_payload():
    """Offload merges segments and leaves BOTH length fields describing the
    first one. Honouring them cost 1398 bytes of a 6990-byte frame, and the
    sequence number then advanced short, stranding the rest of the response for
    the life of the connection — measured at 20-30% of frames on a host-NIC
    capture, ~20MB per capture."""
    merged_segments = 5  # what offload had merged in the frame this was measured from
    body = b"".join(bytes([ord("A") + i % 26]) * PATH_MSS for i in range(merged_segments))
    frame = coalesced_vxlan_frame(_segment(payload=body), segment_size=PATH_MSS)

    seg = _decode(frame)

    assert seg is not None
    assert len(seg.payload) == len(body), "payload trimmed to the stale length field"
    assert seg.payload == body
    assert (seg.src, seg.sport) == POD_CLIENT


def test_an_honest_frame_is_unaffected():
    seg = _decode(vxlan_frame(ip_tcp(_segment())))

    assert seg is not None and seg.payload == REQUEST


def test_ethernet_padding_never_reaches_the_payload():
    """A sub-60-byte packet is padded, so the frame legitimately exceeds its IP
    length. Reading past the header there would append zeros to the payload."""
    seg = _decode(padded_frame(_segment(payload=b"tiny")))

    assert seg is not None
    assert seg.payload == b"tiny", "padding leaked into the payload"


def test_a_pure_ack_stays_empty():
    seg = _decode(padded_frame(dataclasses.replace(_segment(payload=b""), ack=True)))

    assert seg is not None and seg.payload == b""


# --- fragmentation is refused out loud, not silently ---


def test_fragmented_outer_packet_is_dropped_and_counted(caplog):
    before = capture._fragmented

    with caplog.at_level("WARNING", logger="frameprocessor.capture"):
        assert _decode(vxlan_frame(ip_tcp(_segment()), more_fragments=True)) is None

    assert capture._fragmented == before + 1
    if before == 0:
        assert "fragmented" in caplog.text.lower()


# --- the differential: the two vantage points must agree ---


def test_tunnelled_conversation_produces_an_identical_tap_message():
    """The same inference, seen on cni0 and seen on the wire, must match.

    This is what makes decapsulation trustworthy: not that it parses, but
    that nothing downstream can tell which vantage point produced it.
    """
    body = {"model": "openai/gpt-oss-120b", "prompt": [1, 2, 3], "max_tokens": 4,
            "temperature": 0.7, "seed": 11}
    request = http_request("/v1/chat/completions", json.dumps(body).encode())
    response = http_response_json(
        json.dumps(
            {"id": "cmpl-1", "model": "openai/gpt-oss-120b",
             "choices": [{"text": "hello", "finish_reason": "stop"}]}
        ).encode()
    )
    segments = conversation(request, response)

    bare = list(process_segments(iter(segments)))
    tunnelled = list(
        process_segments(
            seg for seg in (decode_tcp(f) for f in as_vxlan_frames(segments)) if seg is not None
        )
    )

    assert len(bare) == 1
    assert tunnelled == bare


# --- ring geometry the kernel will actually accept ---


def test_ring_geometry_env_values_are_snapped_to_kernel_rules():
    """A plausible typo must not become two spawn tracebacks and no capture.

    setsockopt runs in the child, so a clamp that only bounds the magnitude lets
    the parent log a healthy startup line while every receiver dies mapping its
    ring. Page size differs by platform, so validity is asserted through the
    kernel's own rules rather than against fixed numbers.
    """
    frame = config._as_ring_frame_size("2000", 2048)
    block = config._as_ring_block_size("2000000", 1024 * 1024, frame)

    capture.RingCapture._validate_geometry(block, 128, frame, 10)
    assert block >= 2000000  # snapped up, never silently shrunk
    assert block % mmap.PAGESIZE == 0
    assert block % frame == 0


def test_ring_frame_size_is_aligned_and_never_below_one_header():
    assert config._as_ring_frame_size("100", 2048) % 16 == 0
    assert config._as_ring_frame_size("100", 2048) >= 100
    assert config._as_ring_frame_size("8", 2048) >= 68
    assert config._as_ring_frame_size("garbage", 2048) == 2048


def test_ring_geometry_defaults_are_valid():
    capture.RingCapture._validate_geometry(
        config.FRAME_PROCESSOR_RING_BLOCK_SIZE,
        config.FRAME_PROCESSOR_RING_BLOCK_COUNT,
        config.FRAME_PROCESSOR_RING_FRAME_SIZE,
        config.FRAME_PROCESSOR_RING_RETIRE_TIMEOUT_MS,
    )
