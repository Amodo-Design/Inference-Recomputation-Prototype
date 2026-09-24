"""Synthetic TCP segments and frames for tests — no pcap files needed."""

from __future__ import annotations

import socket
import struct

import dpkt

from app.capture import Frame, TcpSegment

CLIENT = ("10.0.0.1", 51000)
SERVER = ("10.0.0.2", 8080)

# The nodes either end of the tapped link, carrying the overlay between them.
NODE_A = "192.168.1.11"
NODE_B = "192.168.1.12"
_MAC_A = b"\xaa\xaa\xaa\xaa\xaa\xaa"
_MAC_B = b"\xbb\xbb\xbb\xbb\xbb\xbb"
_VXLAN_FLAG_VNI = 0x08
_VXLAN_HEADER_LEN = 8

# One inner TCP segment, as measured on the tapped path — the unit offload
# coalesces, so it is both the size a stale length field reports and the step
# a payload is built in.
PATH_MSS = 1398

# Frame geometry for the fixtures below, which reach into a built frame and
# corrupt its length fields by hand.
#
# Spelled out here rather than imported from app.capture on purpose: these
# offsets are how the fixture *breaks* a frame, so sharing constants with the
# code under test would let a wrong value there hide behind a matching one here.
_ETH_HEADER_LEN = 14
_UDP_HEADER_LEN = 8
_HEADER_WORD_LEN = 4  # IPv4's IHL and TCP's data offset both count 32-bit words
_IPV4_IHL_MASK = 0x0F
_IPV4_TOTAL_LEN_AT = 2  # bytes into an IPv4 header
_UDP_LEN_AT = 4  # bytes into a UDP header
_TCP_DATA_OFFSET_AT = 12  # bytes into a TCP header; its high nibble holds it
_MIN_ETH_FRAME_LEN = 60  # a NIC zero-pads any frame shorter than this


def conversation(
    client_bytes: bytes,
    server_bytes: bytes,
    *,
    mtu: int = 1400,
    ts0: float = 1700000000.0,
) -> list[TcpSegment]:
    """A full synthetic connection: SYN, client payload, server payload, FINs.

    Payloads are split into mtu-sized segments; timestamps tick 1ms apart so
    received_at < completed_at.
    """
    segments: list[TcpSegment] = []
    ts = ts0

    def seg(src, dst, seq, payload=b"", syn=False, ack=False, fin=False):
        nonlocal ts
        ts += 0.001
        return TcpSegment(
            ts=ts, src=src[0], sport=src[1], dst=dst[0], dport=dst[1],
            seq=seq, syn=syn, ack=ack, fin=fin, rst=False, payload=payload,
        )

    segments.append(seg(CLIENT, SERVER, 1000, syn=True))
    segments.append(seg(SERVER, CLIENT, 5000, syn=True, ack=True))

    seq = 1001
    for i in range(0, len(client_bytes), mtu):
        chunk = client_bytes[i : i + mtu]
        segments.append(seg(CLIENT, SERVER, seq, chunk))
        seq += len(chunk)
    client_fin_seq = seq

    seq = 5001
    for i in range(0, len(server_bytes), mtu):
        chunk = server_bytes[i : i + mtu]
        segments.append(seg(SERVER, CLIENT, seq, chunk))
        seq += len(chunk)

    segments.append(seg(CLIENT, SERVER, client_fin_seq, fin=True))
    segments.append(seg(SERVER, CLIENT, seq, fin=True))
    return segments


def keepalive(
    exchanges: list[tuple[bytes, bytes]],
    *,
    mtu: int = 1400,
    ts0: float = 1700000000.0,
    gap: float = 1.0,
) -> list[TcpSegment]:
    """A pooled connection that never closes: SYNs, then request/response
    pairs back to back, no FIN — what the gateway actually holds open.

    `gap` seconds pass between exchanges, so per-exchange timestamps are
    distinguishable from connection-level ones.
    """
    segments: list[TcpSegment] = []
    ts = ts0

    def seg(src, dst, seq, payload=b"", syn=False, ack=False):
        nonlocal ts
        ts += 0.001
        return TcpSegment(
            ts=ts, src=src[0], sport=src[1], dst=dst[0], dport=dst[1],
            seq=seq, syn=syn, ack=ack, fin=False, rst=False, payload=payload,
        )

    segments.append(seg(CLIENT, SERVER, 1000, syn=True))
    segments.append(seg(SERVER, CLIENT, 5000, syn=True, ack=True))

    client_seq, server_seq = 1001, 5001
    for request, response in exchanges:
        for i in range(0, len(request), mtu):
            chunk = request[i : i + mtu]
            segments.append(seg(CLIENT, SERVER, client_seq, chunk))
            client_seq += len(chunk)
        for i in range(0, len(response), mtu):
            chunk = response[i : i + mtu]
            segments.append(seg(SERVER, CLIENT, server_seq, chunk))
            server_seq += len(chunk)
        ts += gap
    return segments


# --- Raw frames, for the capture layer ---


def ethernet(payload, *, src: bytes = _MAC_A, dst: bytes = _MAC_B) -> bytes:
    """Wrap an IP packet in an ethernet frame, as it appears on the wire."""
    return bytes(
        dpkt.ethernet.Ethernet(src=src, dst=dst, type=dpkt.ethernet.ETH_TYPE_IP, data=payload)
    )


def ip_tcp(seg: TcpSegment):
    """One TcpSegment rebuilt as the IPv4/TCP packet it was decoded from."""
    flags = 0
    if seg.syn:
        flags |= dpkt.tcp.TH_SYN
    if seg.ack:
        flags |= dpkt.tcp.TH_ACK
    if seg.fin:
        flags |= dpkt.tcp.TH_FIN
    if seg.rst:
        flags |= dpkt.tcp.TH_RST
    if seg.payload:
        flags |= dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK
    tcp = dpkt.tcp.TCP(sport=seg.sport, dport=seg.dport, seq=seg.seq, flags=flags)
    tcp.data = seg.payload
    return dpkt.ip.IP(
        src=socket.inet_aton(seg.src),
        dst=socket.inet_aton(seg.dst),
        p=dpkt.ip.IP_PROTO_TCP,
        data=tcp,
    )


def ip_udp(payload: bytes, *, sport: int, dport: int, src: str = NODE_A, dst: str = NODE_B):
    """A UDP packet between two node addresses."""
    udp = dpkt.udp.UDP(sport=sport, dport=dport)
    udp.data = payload
    udp.ulen = len(udp)
    return dpkt.ip.IP(
        src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=dpkt.ip.IP_PROTO_UDP, data=udp
    )


def vxlan_frame(
    inner_ip,
    *,
    port: int = 8472,
    vxlan_flags: int = _VXLAN_FLAG_VNI,
    vni: int = 1,
    more_fragments: bool = False,
) -> bytes:
    """`inner_ip` sealed in VXLAN over UDP, exactly as the CNI sends it."""
    header = struct.pack(
        "!B3xBBBx", vxlan_flags, (vni >> 16) & 0xFF, (vni >> 8) & 0xFF, vni & 0xFF
    )
    outer = ip_udp(header + ethernet(inner_ip), sport=39999, dport=port)
    if more_fragments:
        outer.mf = 1
    return ethernet(outer)


def _ipv4_header_len(frame: bytes, at: int) -> int:
    """Bytes of the IPv4 header starting at `at`, read from its IHL nibble."""
    return (frame[at] & _IPV4_IHL_MASK) * _HEADER_WORD_LEN


def coalesced_vxlan_frame(seg: TcpSegment, *, segment_size: int = PATH_MSS) -> bytes:
    """A GRO-coalesced tunnel frame, as a host NIC actually delivers one.

    Receive offload merges consecutive segments of the inner TCP stream into one
    frame but rewrites neither length field: the outer UDP `ulen` and the inner
    IP total length both still describe the *first* segment only. The bytes are
    all present; only the headers lie. `seg.payload` supplies the whole
    coalesced payload and `segment_size` is what the stale headers will claim.
    """
    frame = bytearray(vxlan_frame(ip_tcp(seg)))

    # Walk in to the two length fields. Both IPv4 header lengths are read from
    # the frame rather than assumed, since IHL varies with options; every other
    # header on this synthetic path is fixed-size.
    outer_udp_at = _ETH_HEADER_LEN + _ipv4_header_len(frame, _ETH_HEADER_LEN)
    inner_ip_at = outer_udp_at + _UDP_HEADER_LEN + _VXLAN_HEADER_LEN + _ETH_HEADER_LEN
    inner_ip_header_len = _ipv4_header_len(frame, inner_ip_at)
    tcp_at = inner_ip_at + inner_ip_header_len
    tcp_header_len = (frame[tcp_at + _TCP_DATA_OFFSET_AT] >> 4) * _HEADER_WORD_LEN

    # Stale both length fields back to a single segment, leaving the bytes intact.
    inner_declares = inner_ip_header_len + tcp_header_len + segment_size
    outer_declares = _UDP_HEADER_LEN + _VXLAN_HEADER_LEN + _ETH_HEADER_LEN + inner_declares
    struct.pack_into("!H", frame, outer_udp_at + _UDP_LEN_AT, outer_declares)
    struct.pack_into("!H", frame, inner_ip_at + _IPV4_TOTAL_LEN_AT, inner_declares)
    return bytes(frame)


def padded_frame(seg: TcpSegment, *, total_len: int = _MIN_ETH_FRAME_LEN) -> bytes:
    """A short packet padded to the ethernet minimum, with trailing zero bytes.

    The legitimate case where a frame is longer than its IP header claims —
    which must NOT be mistaken for offload, or the padding lands in the payload.
    """
    frame = bytearray(ethernet(ip_tcp(seg)))
    frame.extend(b"\x00" * max(0, total_len - len(frame)))
    return bytes(frame)


def as_vxlan_frames(segments: list[TcpSegment], **kwargs) -> list[Frame]:
    """A synthetic conversation as the wire carries it: every segment tunnelled.

    Timestamps carry over, so a tap message built from these is comparable
    key-for-key with one built from the segments directly.
    """
    return [Frame(ts=seg.ts, data=vxlan_frame(ip_tcp(seg), **kwargs)) for seg in segments]


def http_request(path: str, body: bytes, headers: dict[str, str] | None = None) -> bytes:
    lines = [f"POST {path} HTTP/1.1", "host: 10.0.0.2:8080", f"content-length: {len(body)}",
             "content-type: application/json"]
    for name, value in (headers or {}).items():
        lines.append(f"{name}: {value}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + body


def http_response_json(body: bytes) -> bytes:
    head = (
        "HTTP/1.1 200 OK\r\n"
        f"content-length: {len(body)}\r\n"
        "content-type: application/json\r\n\r\n"
    )
    return head.encode() + body


def http_response_sse(events: list[bytes]) -> bytes:
    """A chunked text/event-stream response, one SSE event per chunk."""
    head = (
        "HTTP/1.1 200 OK\r\n"
        "transfer-encoding: chunked\r\n"
        "content-type: text/event-stream\r\n\r\n"
    )
    chunks = b""
    for event in events + [b"data: [DONE]\n\n"]:
        frame = b"data: " + event + b"\n\n" if not event.startswith(b"data:") else event
        chunks += f"{len(frame):x}\r\n".encode() + frame + b"\r\n"
    chunks += b"0\r\n\r\n"
    return head.encode() + chunks
