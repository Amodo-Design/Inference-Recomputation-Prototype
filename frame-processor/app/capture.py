"""Frame sources and TCP segment decoding.

Two sources, one shape: `pcap_frames()` replays a capture file and
`RingCapture` reads an interface through a TPACKET_V3 ring (Linux only).
Everything downstream consumes `TcpSegment`s, so the pipeline neither knows
nor cares whether frames were mirrored by hardware, sniffed on the node, or
replayed from a fixture — which is what lets the whole pipeline be developed
and tested offline.

Decoding sees through one layer of VXLAN, because a tap on a physical link
watches pod traffic *after* the CNI has encapsulated it — on the wire it is
UDP between node addresses with the inference request sealed inside. Both
vantage points therefore hand the same thing downstream: a bare TCP segment
carrying pod addresses, whether it was seen on `cni0` before encapsulation
or on the wire after.
"""

from __future__ import annotations

import logging
import mmap
import socket
import struct
from dataclasses import dataclass
from typing import Iterator, Protocol

import dpkt

from app import config
from app.policy import direction_from_pkttype

log = logging.getLogger("frameprocessor.capture")

# AF_PACKET socket options. Spelled out rather than taken from `socket`,
# which only defines them on Linux — this module imports on any platform so
# the decoding half can be developed and tested off the tapped node.
_SOL_PACKET = 263
_PACKET_ADD_MEMBERSHIP = 1
_PACKET_MR_PROMISC = 1
_PACKET_RX_RING = 5
_PACKET_STATISTICS = 6
_PACKET_VERSION = 10
_SO_TIMESTAMPNS = 35  # also the SCM_ type of the ancillary message it returns
_ETH_P_ALL = 0x0003
_SNAPLEN = 65535  # whole frames, including jumbo

# PACKET_MMAP/TPACKET_V3.  These values come from linux/if_packet.h rather
# than ``socket`` for the same portability reason as the options above:
# capture is Linux-only, but the decoder and its tests must import on macOS.
_TPACKET_V3 = 2
_TP_STATUS_KERNEL = 0
_TP_STATUS_USER = 1
_TP_STATUS_COPY = 1 << 1
_TPACKET_ALIGNMENT = 16
# V3 packs variable-sized records within a block at its own 8-byte alignment
# (``V3_ALIGNMENT`` in net/packet/af_packet.c).  Ring frame geometry remains
# subject to the public 16-byte TPACKET_ALIGNMENT above.
_TPACKET3_RECORD_ALIGNMENT = 8
_TPACKET_BLOCK_STATUS_AT = 8
_TPACKET_BLOCK_NUM_PKTS_AT = 12
_TPACKET_BLOCK_FIRST_PKT_AT = 16
_TPACKET3_NEXT_OFFSET_AT = 0
_TPACKET3_SEC_AT = 4
_TPACKET3_NSEC_AT = 8
_TPACKET3_SNAPLEN_AT = 12
_TPACKET3_LEN_AT = 16
_TPACKET3_STATUS_AT = 20
_TPACKET3_MAC_AT = 24
_TPACKET3_HEADER_MIN_LEN = 28
# TPACKET_ALIGN(sizeof(tpacket3_hdr)==48) + sizeof(sockaddr_ll)==20.
# The kernel rejects a configured frame size below this ABI header length.
_TPACKET3_HDRLEN = 68
# The sockaddr_ll the kernel writes after each tpacket3_hdr: sll_pkttype is
# its eleventh byte (family u16, protocol u16, ifindex s32, hatype u16). It
# says whether this host sent the frame or received it from the wire, which is
# the one statement about direction the peer cannot forge.
_TPACKET3_SLL_AT = 48
_SLL_PKTTYPE_AT = 10
_TPACKET3_PKTTYPE_AT = _TPACKET3_SLL_AT + _SLL_PKTTYPE_AT
DEFAULT_RING_READ_BLOCKS = 8

# VXLAN seals the inner ethernet frame behind an 8-byte header. Bit 3 of the
# first byte (0x08) is the "VNI present" flag, set on every valid VXLAN
# packet; checking it stops an unrelated UDP service sharing the port from
# having eight bytes removed and the remainder misread as ethernet.
_VXLAN_HEADER_LEN = 8
_VXLAN_FLAG_VNI = 0x08

# --- Ethernet framing ---
# An untagged header is two 6-byte addresses followed by a 2-byte EtherType.
_ETH_ADDR_LEN = 6
_ETHERTYPE_LEN = 2
_ETHERTYPE_OFFSET = 2 * _ETH_ADDR_LEN  # 12: past both addresses
_ETH_HEADER_LEN = _ETHERTYPE_OFFSET + _ETHERTYPE_LEN  # 14

# A VLAN tag displaces the EtherType instead of replacing it: 2-byte TPID,
# 2-byte tag control info, then the EtherType again — so each tag adds four
# bytes and moves the EtherType four further in. Two TPIDs turn up here:
# 802.1Q for a single tag, 802.1ad for the outer tag of a QinQ stack.
_VLAN_TAG_LEN = 4
# Within a tag: the 2-byte tag control info, then the EtherType it displaced.
_VLAN_TCI_LEN = 2
_VLAN_TPIDS = (b"\x81\x00", b"\x88\xa8")
_VLAN_TPID_VALUES = (0x8100, 0x88A8)
_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_IPV6 = 0x86DD

# --- IPv4 and transport framing ---
# An IPv4 header's first byte packs the version into the high nibble and the
# header length into the low one. That length, and TCP's data offset, are both
# counted in 32-bit words, so both scale by _HEADER_WORD_LEN.
_IPV4_VERSION = 4
_IPV4_VERSION_SHIFT = 4
_IPV4_IHL_MASK = 0x0F
_HEADER_WORD_LEN = 4
_MIN_IPV4_HEADER_LEN = 5 * _HEADER_WORD_LEN  # 20; a valid IHL is never lower
_UDP_HEADER_LEN = 8
_IPPROTO_TCP = 6
_IPPROTO_UDP = 17

# Field offsets within an IPv4 header, for the single-pass raw decoder. It
# reads fields straight out of the frame rather than building an object graph,
# so these are the only thing standing between it and silently misreading
# every packet on the link — named, because a bare `ip_at + 9` is not a claim
# anyone can check against RFC 791 at review speed.
_IPV4_TOTAL_LEN_AT = 2
_IPV4_FLAGS_FRAGMENT_AT = 6
_IPV4_PROTOCOL_AT = 9
_IPV4_SRC_AT = 12
_IPV4_DST_AT = 16
_IPV4_ADDR_LEN = 4
# The flags/fragment word is three flag bits (reserved, DF, MF) then a 13-bit
# offset. The mask keeps MF and the offset and discards reserved and DF: a set
# MF *or* a non-zero offset means this packet is a fragment, which
# _note_fragment explains is unusable here, while DF says only that the sender
# forbade fragmentation and is not itself a fragment.
_IPV4_FRAGMENT_MASK = 0x3FFF

# Field offsets within a UDP header (RFC 768).
_UDP_DPORT_AT = 2

# Field offsets within a TCP header (RFC 9293). _MIN_TCP_HEADER_LEN happens to
# equal _MIN_IPV4_HEADER_LEN; they are separate names because they are separate
# facts about separate protocols, and substituting one for the other would
# work by coincidence until a header grew.
_MIN_TCP_HEADER_LEN = 20
_TCP_DATA_OFFSET_AT = 12  # high nibble, counted in 32-bit words
_TCP_DATA_OFFSET_SHIFT = 4
_TCP_FLAGS_AT = 13

# TCP control bits, matching dpkt.tcp.TH_*. Spelled out so the raw decoder does
# not import dpkt for four constants on a path whose whole purpose is not
# touching dpkt; _decode uses dpkt's own names for the same bits.
_TH_FIN = 0x01
_TH_SYN = 0x02
_TH_RST = 0x04
_TH_ACK = 0x10

# How far a frame may exceed the end its IP total length declares before the
# length is treated as stale rather than the frame as padded — see _tcp_payload.
#
# Two things make the excess legitimate: ethernet zero-pads any frame under 60
# bytes, and some captures keep the 4-byte FCS. Worst case is 30 bytes (a
# header-only 20-byte IP packet declares the frame ends at 34, gets padded to
# 60, and carries an FCS to 64). A coalesced frame, by contrast, exceeds by at
# least one MSS — ~1398 bytes on this path. Any threshold between 30 and 1398
# separates the two, so this sits mid-valley rather than on either edge.
_MAX_TRAILER = 60

# Fragmented IPv4 is unusable and worth saying so out loud — see _note_fragment.
_fragmented = 0


@dataclass(frozen=True)
class Frame:
    ts: float  # capture timestamp (epoch seconds)
    data: bytes  # raw link-layer frame


@dataclass(frozen=True)
class TcpSegment:
    ts: float
    src: str
    sport: int
    dst: str
    dport: int
    seq: int
    syn: bool
    # ACK distinguishes a client's opening SYN from a server's SYN-ACK, which
    # is the only thing on the wire that names the two roles outright. It
    # matters once frames arrive from two capture sockets, where whichever
    # socket the reader drains first no longer implies who spoke first.
    ack: bool
    fin: bool
    rst: bool
    payload: bytes


@dataclass(frozen=True)
class PacketStats:
    """One self-resetting ``PACKET_STATISTICS`` sample.

    ``received`` and ``dropped`` are the kernel's counts since the preceding
    read.  Capture sources also retain cumulative totals on their
    ``received`` and ``drops`` attributes, which is much less error-prone for
    a long-running receiver process than making every caller accumulate them.
    """

    received: int
    dropped: int
    freeze_q_count: int = 0


class FrameObserver(Protocol):
    """Sees every frame, including the ones TCP decoding throws away.

    Structural typing rather than an import, so this module keeps knowing
    nothing about accounting: `app.accounting.FrameAccountant` satisfies it,
    and so does a test double. The contract is that neither method raises —
    they run inside the packet-ring walk, where an exception costs the block
    and every frame in it, and a validator that takes down the capture it is
    validating has made the problem worse rather than better.

    ``observe`` is handed a memoryview into the ring, which is released as
    soon as the walk moves on. An implementation that needs to keep bytes must
    copy them.

    ``direction`` is `app.policy.DIRECTION_OUT` / `DIRECTION_IN` when the ring
    record said which way the frame went (see ``_TPACKET3_PKTTYPE_AT``), and
    None when the source had no such bit — a pcap replay, or a ring whose
    frame offset leaves no room for the sockaddr_ll.
    """

    def observe(
        self, data: bytes | memoryview, ts: float, direction: str | None = None
    ) -> str: ...

    def note_truncated(
        self, original_len: int, captured_len: int, ts: float = 0.0
    ) -> None: ...


def pcap_frames(path: str) -> Iterator[Frame]:
    """Replay frames from a pcap/pcapng file."""
    with open(path, "rb") as fh:
        try:
            reader = dpkt.pcap.UniversalReader(fh)
        except AttributeError:  # older dpkt: classic pcap only
            reader = dpkt.pcap.Reader(fh)
        for ts, data in reader:
            yield Frame(ts=ts, data=data)


def default_route_iface() -> str:
    """The interface carrying the default route (Linux)."""
    with open("/proc/net/route") as fh:
        for line in fh.readlines()[1:]:
            fields = line.split()
            if len(fields) >= 2 and fields[1] == "00000000":
                return fields[0]
    raise RuntimeError("No default-route interface found in /proc/net/route")


class RingCapture:
    """One Linux interface captured through a ``TPACKET_V3`` block ring.

    The hot path does not make a syscall for every mirrored frame. The kernel
    retires blocks of frames into one shared memory mapping; ``read_segments``
    walks ready blocks in bounded turns, rejects irrelevant traffic in place,
    and copies only TCP segment fields and payloads before returning ownership
    of each block to the kernel.

    The optional ``limit`` bounds *returned TCP segments*, not inspected
    frames.  High-rate UDP is still drained completely.  If the limit lands in
    the middle of a block, only integer offsets are retained and the next call
    resumes that block; no memoryview escapes this method. ``max_blocks`` also
    bounds a call independently of protocol mix, so a ring continuously
    refilled by UDP cannot starve dispatch, statistics, or shutdown checks.
    """

    # Class-level default so a ring built without ``__init__`` — which the
    # capture tests do, to exercise the block walk without a Linux socket —
    # still has an observer attribute. The walk checks it per frame, and an
    # AttributeError there would cost a whole block.
    _observer: FrameObserver | None = None

    def __init__(
        self,
        iface: str,
        *,
        block_size: int = 1 << 20,
        block_count: int = 64,
        frame_size: int = 2048,
        retire_timeout_ms: int = 64,
        observer: FrameObserver | None = None,
    ) -> None:
        self._validate_geometry(block_size, block_count, frame_size, retire_timeout_ms)
        self.iface = iface
        self._observer = observer
        self.received = 0
        self.drops = 0
        self.freeze_q_count = 0
        self.truncated_packets = 0
        self.malformed_blocks = 0
        self.blocks_released = 0
        self._block_size = block_size
        self._block_count = block_count
        self._block_index = 0
        self._active_packets = 0
        self._active_packet_at = 0
        self._closed = False
        self._ring: mmap.mmap | None = None

        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(_ETH_P_ALL))
        self._sock = sock
        try:
            # PACKET_VERSION must precede PACKET_RX_RING. Binding before the
            # allocation prevents unrelated interfaces filling the new ring
            # during the remaining setup.
            sock.setsockopt(_SOL_PACKET, _PACKET_VERSION, struct.pack("=I", _TPACKET_V3))
            sock.bind((iface, 0))
            self._set_promiscuous()
            frame_count = block_size * block_count // frame_size
            request = struct.pack(
                "=7I",
                block_size,
                block_count,
                frame_size,
                frame_count,
                retire_timeout_ms,
                0,  # private bytes at the start of each block
                0,  # no optional feature fields requested
            )
            sock.setsockopt(_SOL_PACKET, _PACKET_RX_RING, request)
            self._ring = mmap.mmap(
                sock.fileno(),
                block_size * block_count,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            sock.setblocking(False)
        except BaseException:
            if self._ring is not None:
                self._ring.close()
            sock.close()
            raise

    @staticmethod
    def _validate_geometry(
        block_size: int,
        block_count: int,
        frame_size: int,
        retire_timeout_ms: int,
    ) -> None:
        if block_size <= 0 or block_size % mmap.PAGESIZE:
            raise ValueError("block_size must be a positive multiple of the system page size")
        if block_count <= 0:
            raise ValueError("block_count must be positive")
        if frame_size < _TPACKET3_HDRLEN or frame_size % _TPACKET_ALIGNMENT:
            raise ValueError("frame_size must be at least 68 and 16-byte aligned")
        if block_size % frame_size:
            raise ValueError("block_size must be an exact multiple of frame_size")
        if retire_timeout_ms <= 0:
            raise ValueError("retire_timeout_ms must be positive")

    def _set_promiscuous(self) -> None:
        mreq = struct.pack(
            "iHH8s", socket.if_nametoindex(self.iface), _PACKET_MR_PROMISC, 0, b""
        )
        self._sock.setsockopt(_SOL_PACKET, _PACKET_ADD_MEMBERSHIP, mreq)

    def attach_observer(self, observer: FrameObserver | None) -> None:
        """Set the per-frame observer after construction.

        Receiver processes build their capture source through an injectable
        factory, and the integration tests inject single-argument ones. Rather
        than force every factory to grow a parameter, the observer is attached
        here once the source exists.
        """
        self._observer = observer

    def fileno(self) -> int:
        return self._sock.fileno()

    def poll_stats(self) -> PacketStats:
        """Read and accumulate Linux's self-resetting V3 socket counters."""
        raw = self._sock.getsockopt(_SOL_PACKET, _PACKET_STATISTICS, 12)
        received, dropped = struct.unpack_from("=II", raw)
        freeze_q_count = struct.unpack_from("=I", raw, 8)[0] if len(raw) >= 12 else 0
        self.received += received
        self.drops += dropped
        self.freeze_q_count += freeze_q_count
        return PacketStats(received, dropped, freeze_q_count)

    def _note_truncated(self, original_len: int, captured_len: int) -> None:
        self.truncated_packets += 1
        if self.truncated_packets == 1 or self.truncated_packets % 1000 == 0:
            log.warning(
                "%s packet ring truncated %d packet(s), latest %d -> %d bytes; "
                "increase ring/frame geometry or disable NIC offload",
                self.iface,
                self.truncated_packets,
                original_len,
                captured_len,
            )

    def _release_block(self, ring: memoryview, block_at: int) -> None:
        # All segment payloads have been copied by this point.  A plain aligned
        # native write is the userspace side of the TPACKET ownership handoff.
        # CPython does not expose an explicit memory barrier. The deployed
        # image is explicitly linux/amd64, whose strong memory ordering makes
        # this safe and matches the store in the kernel's packet_mmap example.
        # A future ARM image must add a native release fence before this store.
        struct.pack_into("=I", ring, block_at + _TPACKET_BLOCK_STATUS_AT, _TP_STATUS_KERNEL)
        self.blocks_released += 1
        self._active_packets = 0
        self._active_packet_at = 0
        self._block_index = (self._block_index + 1) % self._block_count

    def _malformed_block(self, ring: memoryview, block_at: int, reason: str) -> None:
        self.malformed_blocks += 1
        log.error("%s malformed TPACKET_V3 block: %s", self.iface, reason)
        self._release_block(ring, block_at)

    def read_segments(
        self,
        limit: int | None = None,
        *,
        max_blocks: int = DEFAULT_RING_READ_BLOCKS,
    ) -> list[TcpSegment]:
        """Drain ready ring blocks and return copied reassembly candidates.

        Timestamps come from each ``tpacket3_hdr`` and therefore represent
        kernel receive time, not the later moment this Python process happened
        to walk the block.  Pure ACKs and configured-out ports are discarded
        before address rendering/payload copying; SYN/FIN/RST packets remain
        so connection role and lifetime tracking stay correct. At most
        ``max_blocks`` are released per call even if the producer refills the
        ring faster than Python drains it.
        """
        if self._closed or self._ring is None:
            return []
        if limit is not None and limit <= 0:
            return []
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")

        segments: list[TcpSegment] = []
        released_at_start = self.blocks_released
        ring = memoryview(self._ring)
        try:
            while (
                (limit is None or len(segments) < limit)
                and self.blocks_released - released_at_start < max_blocks
            ):
                block_at = self._block_index * self._block_size
                block_end = block_at + self._block_size

                if self._active_packets == 0:
                    status = struct.unpack_from(
                        "=I", ring, block_at + _TPACKET_BLOCK_STATUS_AT
                    )[0]
                    if not status & _TP_STATUS_USER:
                        break
                    packet_count = struct.unpack_from(
                        "=I", ring, block_at + _TPACKET_BLOCK_NUM_PKTS_AT
                    )[0]
                    first_packet = struct.unpack_from(
                        "=I", ring, block_at + _TPACKET_BLOCK_FIRST_PKT_AT
                    )[0]
                    if packet_count == 0:
                        self._release_block(ring, block_at)
                        continue
                    if not (
                        _TPACKET3_HEADER_MIN_LEN
                        <= first_packet
                        <= self._block_size - _TPACKET3_HEADER_MIN_LEN
                    ) or first_packet % _TPACKET3_RECORD_ALIGNMENT:
                        self._malformed_block(
                            ring, block_at, f"invalid first-packet offset {first_packet}"
                        )
                        continue
                    self._active_packets = packet_count
                    self._active_packet_at = block_at + first_packet

                packet_at = self._active_packet_at
                if packet_at < block_at or packet_at + _TPACKET3_HEADER_MIN_LEN > block_end:
                    self._malformed_block(
                        ring, block_at, f"packet header outside block at {packet_at - block_at}"
                    )
                    continue

                next_offset, sec, nsec, snaplen, original_len, packet_status = (
                    struct.unpack_from("=6I", ring, packet_at)
                )
                if nsec >= 1_000_000_000:
                    self._malformed_block(
                        ring, block_at, f"invalid kernel timestamp nanoseconds {nsec}"
                    )
                    continue
                mac_offset = struct.unpack_from(
                    "=H", ring, packet_at + _TPACKET3_MAC_AT
                )[0]
                frame_at = packet_at + mac_offset
                frame_end = frame_at + snaplen

                self._active_packets -= 1
                if self._active_packets:
                    next_packet_at = packet_at + next_offset
                    if (
                        next_offset < _TPACKET3_HEADER_MIN_LEN
                        or next_offset % _TPACKET3_RECORD_ALIGNMENT
                        or next_packet_at + _TPACKET3_HEADER_MIN_LEN > block_end
                    ):
                        self._malformed_block(
                            ring, block_at, f"invalid next-packet offset {next_offset}"
                        )
                        continue
                    self._active_packet_at = next_packet_at

                if (
                    packet_status & _TP_STATUS_COPY
                    or snaplen < original_len
                    or frame_at < packet_at + _TPACKET3_HEADER_MIN_LEN
                    or frame_end > block_end
                ):
                    self._note_truncated(original_len, snaplen)
                    if self._observer is not None:
                        self._observer.note_truncated(
                            original_len, snaplen, sec + nsec / 1e9
                        )
                else:
                    frame = ring[frame_at:frame_end]
                    try:
                        # Every frame is accounted for here, before the TCP
                        # decode decides it is uninteresting. This is the only
                        # place that can happen: the decoder rejects non-TCP
                        # without copying — deliberately, for speed — so by the
                        # time anything downstream sees a segment, the ~99% of
                        # this link that is not TCP has already gone. The
                        # observer reads the same zero-copy view of the ring.
                        if self._observer is not None:
                            # Only when the frame sits past the ABI header, so
                            # the byte read really is the sockaddr_ll and not
                            # the frame's own first bytes.
                            direction = None
                            if frame_at >= packet_at + _TPACKET3_HDRLEN:
                                direction = direction_from_pkttype(
                                    ring[packet_at + _TPACKET3_PKTTYPE_AT]
                                )
                            self._observer.observe(frame, sec + nsec / 1e9, direction)
                        if config.FRAME_PROCESSOR_FAST_CLASSIFY:
                            segment = decode_tcp_bytes(
                                frame, sec + nsec / 1e9, candidate_only=True
                            )
                        else:
                            segment = _decode(bytes(frame), sec + nsec / 1e9)
                            if segment is not None and not _configured_candidate(segment):
                                segment = None
                    finally:
                        frame.release()
                    if segment is not None:
                        segments.append(segment)

                if self._active_packets == 0:
                    self._release_block(ring, block_at)

            return segments
        finally:
            ring.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ring is not None:
            self._ring.close()
            self._ring = None
        self._sock.close()


def _note_fragment() -> None:
    """Record a dropped IPv4 fragment, loudly enough to be noticed.

    A later fragment carries no transport header at all, and a first fragment
    carries one with a truncated payload — which would reassemble into a gap
    no downstream stage can see. Both are dropped, but silently dropping them
    is exactly the failure a tap must not have, so the count is logged: on
    this path fragments mean the overlay MTU is wrong (VXLAN adds 50 bytes),
    not that traffic is unusual.
    """
    global _fragmented
    _fragmented += 1
    if _fragmented == 1 or _fragmented % 1000 == 0:
        log.warning(
            "Dropped %d fragmented IPv4 packet(s) — inner headers span frames; "
            "check the overlay MTU",
            _fragmented,
        )


def _ethernet_header_len(data: bytes) -> int:
    """Bytes of ethernet header, counting any 802.1Q / 802.1ad tags.

    Read from the wire rather than inferred from dpkt, because the offset of
    the payload is what matters here and dpkt's handling of stacked tags varies
    by version.

    The EtherType is the header's final field, so it always occupies the two
    bytes ending at `header_len` — and a tag displaces it rather than replacing
    it. So while a TPID is sitting where the EtherType belongs, step over that
    tag and look again. A truncated frame needs no bounds check: a slice past
    the end comes back short, and a short slice matches no TPID.
    """
    header_len = _ETH_HEADER_LEN
    while data[header_len - _ETHERTYPE_LEN:header_len] in _VLAN_TPIDS:
        header_len += _VLAN_TAG_LEN
    return header_len


# A private tri-state result distinguishes "proved irrelevant" (None) from
# "layout not covered by the common path" (fallback to dpkt).  This is how the
# optimisation remains fail-open without running the old classifier and then
# walking the exact same headers a second time.
_RAW_FALLBACK = object()


def ports_allowed(
    sport: int,
    dport: int,
    include_ports: frozenset[int] | None,
    exclude_ports: frozenset[int],
) -> bool:
    """The coarse port pre-filter, shared by every stage that applies it.

    Exclusion wins: a port named there is never reassembled however the
    allowlist is set, so a heavy known-irrelevant talker can be dropped
    without having to enumerate everything worth keeping.

    Takes ports rather than a segment because the raw decoder tests them
    before it has built one — rejecting on the ports is what lets it skip
    rendering addresses and copying a payload out of a packet ring.
    """
    if exclude_ports and (sport in exclude_ports or dport in exclude_ports):
        return False
    return include_ports is None or sport in include_ports or dport in include_ports


def carries_reassembly_state(segment: TcpSegment) -> bool:
    """Whether a segment is worth reassembling at all.

    A pure ACK advances no sequence space and supplies no role/lifetime
    information. SYN/FIN/RST remain even without a payload.
    """
    return bool(segment.payload or segment.syn or segment.fin or segment.rst)


def is_reassembly_candidate(
    segment: TcpSegment,
    include_ports: frozenset[int] | None,
    exclude_ports: frozenset[int],
) -> bool:
    """Both tests at once, for callers that need no reason for the reject."""
    return ports_allowed(
        segment.sport, segment.dport, include_ports, exclude_ports
    ) and carries_reassembly_state(segment)


def _configured_candidate(segment: TcpSegment) -> bool:
    """``is_reassembly_candidate`` against this process's own configuration."""
    return is_reassembly_candidate(
        segment, config.FRAME_PROCESSOR_PORTS, config.FRAME_PROCESSOR_EXCLUDE_PORTS
    )


def _decode_tcp_raw(
    data: bytes | memoryview,
    ts: float,
    *,
    candidate_only: bool,
) -> TcpSegment | None | object:
    """Single-pass decoder for the common Ethernet/IPv4[/VXLAN]/TCP path.

    No dpkt objects, intermediate frame copies, or classifier re-walks are
    created.  A memoryview may point straight into a TPACKET block; the only
    bytes retained from it are the final TCP payload and four-byte addresses.
    Ambiguous layouts return ``_RAW_FALLBACK`` and are decoded by dpkt so this
    optimisation cannot silently narrow protocol coverage.
    """
    size = len(data)
    layer_at = 0
    tunnelled = False

    # There are at most two ethernet/IP pairs: outer and VXLAN inner. Keeping
    # the walk in one loop avoids classifying a frame and then parsing it again.
    while True:
        if size < layer_at + _ETH_HEADER_LEN:
            return _RAW_FALLBACK
        ethertype = struct.unpack_from("!H", data, layer_at + _ETHERTYPE_OFFSET)[0]
        ip_at = layer_at + _ETH_HEADER_LEN
        while ethertype in _VLAN_TPID_VALUES:
            if size < ip_at + _VLAN_TAG_LEN:
                return _RAW_FALLBACK
            ethertype = struct.unpack_from("!H", data, ip_at + _VLAN_TCI_LEN)[0]
            ip_at += _VLAN_TAG_LEN

        if ethertype == _ETHERTYPE_IPV6:
            return _RAW_FALLBACK
        if ethertype != _ETHERTYPE_IPV4:
            return None
        if size < ip_at + _MIN_IPV4_HEADER_LEN:
            return _RAW_FALLBACK
        version_ihl = data[ip_at]
        if version_ihl >> _IPV4_VERSION_SHIFT != _IPV4_VERSION:
            return _RAW_FALLBACK
        ip_header_len = (version_ihl & _IPV4_IHL_MASK) * _HEADER_WORD_LEN
        if ip_header_len < _MIN_IPV4_HEADER_LEN or size < ip_at + ip_header_len:
            return _RAW_FALLBACK
        ip_total_len = struct.unpack_from("!H", data, ip_at + _IPV4_TOTAL_LEN_AT)[0]
        if ip_total_len < ip_header_len:
            return _RAW_FALLBACK
        fragment_word = struct.unpack_from(
            "!H", data, ip_at + _IPV4_FLAGS_FRAGMENT_AT
        )[0]
        if fragment_word & _IPV4_FRAGMENT_MASK:
            _note_fragment()
            return None

        protocol = data[ip_at + _IPV4_PROTOCOL_AT]
        transport_at = ip_at + ip_header_len
        if protocol == _IPPROTO_UDP:
            if tunnelled or size < transport_at + _UDP_HEADER_LEN:
                return _RAW_FALLBACK if not tunnelled else None
            dport = struct.unpack_from("!H", data, transport_at + _UDP_DPORT_AT)[0]
            if dport not in config.FRAME_PROCESSOR_DECAP_PORTS:
                return None
            vxlan_at = transport_at + _UDP_HEADER_LEN
            if size < vxlan_at + _VXLAN_HEADER_LEN:
                return _RAW_FALLBACK
            if not data[vxlan_at] & _VXLAN_FLAG_VNI:
                return None
            layer_at = vxlan_at + _VXLAN_HEADER_LEN
            tunnelled = True
            continue
        if protocol != _IPPROTO_TCP:
            return None

        if size < transport_at + _MIN_TCP_HEADER_LEN:
            return _RAW_FALLBACK
        sport, dport, seq = struct.unpack_from("!HHI", data, transport_at)
        tcp_header_len = (
            data[transport_at + _TCP_DATA_OFFSET_AT] >> _TCP_DATA_OFFSET_SHIFT
        ) * _HEADER_WORD_LEN
        if (
            tcp_header_len < _MIN_TCP_HEADER_LEN
            or size < transport_at + tcp_header_len
        ):
            return _RAW_FALLBACK
        flags = data[transport_at + _TCP_FLAGS_AT]
        syn = bool(flags & _TH_SYN)
        ack = bool(flags & _TH_ACK)
        fin = bool(flags & _TH_FIN)
        rst = bool(flags & _TH_RST)

        payload_at = transport_at + tcp_header_len
        declared_end = ip_at + ip_total_len
        # Small excess is ethernet padding/FCS; a large excess is a GRO frame
        # whose IP total length still describes only the first merged segment.
        if size - declared_end > _MAX_TRAILER:
            payload_end = size
        else:
            payload_end = min(size, declared_end)
        if payload_end < payload_at:
            return _RAW_FALLBACK

        if candidate_only:
            if not ports_allowed(
                sport, dport, config.FRAME_PROCESSOR_PORTS, config.FRAME_PROCESSOR_EXCLUDE_PORTS
            ):
                return None
            if payload_end == payload_at and not (syn or fin or rst):
                return None

        # Address and payload conversion happen only after every cheap reject.
        # bytes() is intentional: a TPACKET block is returned to the kernel
        # before read_segments() returns and no view into it may survive.
        src_at = ip_at + _IPV4_SRC_AT
        dst_at = ip_at + _IPV4_DST_AT
        src = socket.inet_ntop(
            socket.AF_INET, bytes(data[src_at : src_at + _IPV4_ADDR_LEN])
        )
        dst = socket.inet_ntop(
            socket.AF_INET, bytes(data[dst_at : dst_at + _IPV4_ADDR_LEN])
        )
        payload = bytes(data[payload_at:payload_end])
        return TcpSegment(
            ts=ts,
            src=src,
            sport=sport,
            dst=dst,
            dport=dport,
            seq=seq,
            syn=syn,
            ack=ack,
            fin=fin,
            rst=rst,
            payload=payload,
        )


def decode_tcp_bytes(
    data: bytes | memoryview,
    ts: float,
    *,
    candidate_only: bool = False,
) -> TcpSegment | None:
    """Decode raw frame bytes, using dpkt only for uncommon/ambiguous layouts.

    ``candidate_only`` is intended for receiver processes: it also rejects
    configured-out ports and pure ACKs before copying data from a packet ring.
    The default preserves the established ``decode_tcp(Frame(...))`` contract.
    """
    decoded = _decode_tcp_raw(data, ts, candidate_only=candidate_only)
    if decoded is not _RAW_FALLBACK:
        return decoded  # type: ignore[return-value]
    copied = bytes(data)
    segment = _decode(copied, ts)
    if candidate_only and segment is not None and not _configured_candidate(segment):
        return None
    return segment


def _udp_payload(data: bytes) -> bytes | None:
    """Everything after the outer UDP header, taken from the frame itself.

    Deliberately not `bytes(udp.data)`: dpkt trims that to the length the UDP
    header declares, and on a capture taken at a host NIC that length is a lie.
    Receive offload coalesces several tunnel packets into one frame without
    updating the outer header, so it still reports the first packet alone.
    This affects a substantial fraction of frames on such a capture, and the
    inference responses inside them are silently discarded. The bytes are all
    there; only the length field is stale.

    None when the outer header is not IPv4/UDP, in which case the caller falls
    back to dpkt's view.
    """
    eth_len = _ethernet_header_len(data)
    if len(data) < eth_len + _MIN_IPV4_HEADER_LEN:
        return None
    if data[eth_len] >> _IPV4_VERSION_SHIFT != _IPV4_VERSION:
        return None  # not IPv4; IPv6 extension headers make this arithmetic unsafe
    ip_header_len = (data[eth_len] & _IPV4_IHL_MASK) * _HEADER_WORD_LEN
    payload_at = eth_len + ip_header_len + _UDP_HEADER_LEN
    return data[payload_at:] if len(data) > payload_at else None


def _vxlan_payload(udp_payload: bytes) -> bytes:
    """The inner ethernet frame inside a VXLAN packet; empty if it isn't one."""
    if len(udp_payload) <= _VXLAN_HEADER_LEN:
        return b""
    if not udp_payload[0] & _VXLAN_FLAG_VNI:
        return b""
    return udp_payload[_VXLAN_HEADER_LEN:]


def _tcp_payload(data: bytes, ip, tcp) -> bytes:
    """The TCP payload, preferring the frame's own length over the IP header's.

    Receive offload coalesces several segments into one frame and updates
    neither the outer UDP length nor the inner IP total length — both keep
    reporting the first segment alone. dpkt honours them, so `bytes(tcp.data)`
    returns 1398 bytes of a frame carrying 6990, and the sequence number then
    advances short, stranding every byte that follows for the life of the
    connection. On a host-NIC capture this affects a substantial fraction of
    frames.

    Two shapes make a frame legitimately longer than its IP header claims —
    ethernet padding on a sub-60-byte packet, and a trailing FCS — so the frame
    only wins when the excess is larger than both can account for.
    """
    if not isinstance(ip, dpkt.ip.IP):
        return bytes(tcp.data)  # IPv6 extension headers make this arithmetic unsafe
    eth_len = _ethernet_header_len(data)
    declared_end = eth_len + ip.len  # where the IP total length says the frame ends
    if len(data) - declared_end <= _MAX_TRAILER:
        return bytes(tcp.data)
    # ip.hl and tcp.off both count 32-bit words, so they scale the same way.
    payload_at = eth_len + (ip.hl + tcp.off) * _HEADER_WORD_LEN
    return data[payload_at:] if len(data) > payload_at else b""


def decode_tcp(frame: Frame) -> TcpSegment | None:
    """Decode ethernet(/VLAN)/IPv4|IPv6/TCP, through one VXLAN layer if present.

    None for anything that is not TCP once decapsulated.
    """
    if config.FRAME_PROCESSOR_FAST_CLASSIFY:
        return decode_tcp_bytes(frame.data, frame.ts)
    return _decode(frame.data, frame.ts)


def _decode(data: bytes, ts: float, tunnelled: bool = False) -> TcpSegment | None:
    try:
        eth = dpkt.ethernet.Ethernet(data)
    except (dpkt.dpkt.UnpackError, Exception):
        return

    ip = eth.data
    # dpkt unwraps 802.1Q itself (Ethernet.data skips the VLAN tag), so ip is
    # already the network-layer payload — or a Tag object we can step through.
    while isinstance(ip, dpkt.ethernet.VLANtag8021Q):
        ip = ip.data

    if isinstance(ip, dpkt.ip.IP):
        if ip.mf or ip.offset:
            _note_fragment()
            return None
        family = socket.AF_INET
    elif isinstance(ip, dpkt.ip6.IP6):
        family = socket.AF_INET6
    else:
        return None

    transport_header = ip.data
    # Decapsulate before rendering addresses: on an overlay link most frames
    # are tunnel packets whose outer addresses are then thrown away.
    if (
        isinstance(transport_header, dpkt.udp.UDP)
        and transport_header.dport in config.FRAME_PROCESSOR_DECAP_PORTS
    ):
        if tunnelled:
            return None  # one tunnel layer is all this path carries
        # Not bytes(transport_header.data): dpkt trims that to the outer UDP
        # length, which offload leaves stale. The frame is the honest source.
        payload = _udp_payload(data)
        if payload is None:
            payload = bytes(transport_header.data)
        return _decode(_vxlan_payload(payload), ts, tunnelled=True)

    tcp = transport_header
    if not isinstance(tcp, dpkt.tcp.TCP):
        return None

    src = socket.inet_ntop(family, ip.src)
    dst = socket.inet_ntop(family, ip.dst)
    return TcpSegment(
        ts=ts,
        src=src,
        sport=tcp.sport,
        dst=dst,
        dport=tcp.dport,
        seq=tcp.seq,
        syn=bool(tcp.flags & dpkt.tcp.TH_SYN),
        ack=bool(tcp.flags & dpkt.tcp.TH_ACK),
        fin=bool(tcp.flags & dpkt.tcp.TH_FIN),
        rst=bool(tcp.flags & dpkt.tcp.TH_RST),
        payload=_tcp_payload(data, ip, tcp),
    )
