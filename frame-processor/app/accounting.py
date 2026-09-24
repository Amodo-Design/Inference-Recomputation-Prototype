"""Frame-level accounting — every frame on the link lands in exactly one bucket.

The rest of frame-processor answers "which frames are inference?" and discards the
remainder. This module answers the complementary question the tapped-link
filtering scheme needs: "can we account for *everything* that crossed the
link?" — because a prover that can hide bytes in traffic nobody classified is
not constrained by a tap at all.

Three mechanisms, in order of how much of the link they cover:

**Pin and compare.** Most of a quiet link is byte-identical, frame after
frame: switch loop-detection beacons, LLDP advertisements, spanning-tree
BPDUs, DHCPv6 solicits, IGMP membership reports. Measured over a long idle
window, each sender of these typically emits exactly one distinct payload. So
each `(source MAC, class)` pair is pinned to the payload
first seen, and every later frame must match it byte for byte. Residual
capacity under that rule is zero, and a change is a finding rather than a
silent pass.

Pins are compared against the retained bytes, not a digest. A hash would
introduce a collision question for no gain — these payloads are 46 to 512
bytes and there are single digits of distinct pins, so an equality test is
both exact and a C-level `memcmp`. Digests are computed only for reporting.

**What gets pinned is the transport payload, not the frame.** The IPv4
identification field increments on every packet a host sends, and UDP source
ports rotate per datagram on some senders. Fingerprinting whole frames would
therefore make constant traffic look variable and reduce the rule to noise.
`_pin_region` skips the headers whose variance is expected.

That default is right for traffic nobody controls and wrong for traffic we
emit on purpose. On the tapped-link health check (`tools/tapped_link_health.py`) the payload pin
leaves sixteen header bytes checked by nothing at all — DSCP, total length,
identification, flags, TTL, both checksums, the UDP source port and the UDP
length — which on a thirty-second cadence is tens of kilobytes a day of
residual capacity in the one flow that could have had none. A class named in
`whole_frame_classes` is pinned from byte zero instead, so the declaration
becomes "this exact byte sequence" rather than "these payload bytes". It is
opt-in per class because it is only sound where the sender is configured to
make every header field deterministic.

**Structural validation.** A pin cannot help on the first frame from a sender
nobody has seen. Structural rules hold regardless: an LLDP TLV chain must sum
to exactly the payload length with nothing after the terminator, and the
Ethernet padding on any frame short enough to need it must be zero. Both are
places a well-formed-looking frame can carry payload past where a parser
stops looking.

**Reject by default.** Anything that matches no known class is counted as
`unclassified` and raises a finding. This is the property that makes the
whole scheme's "unknown traffic is flagged" claim true rather than hopeful,
and it is why the terminal rule is reject and not allow: an allow-by-default
parser silently absorbs exactly the traffic it exists to catch.

**The link whitelist.** Classification says what a frame is; `app.policy`
says whether it is allowed to be on this link going the way it was going.
With a `LinkPolicy` attached, every frame is also checked against the
per-direction allowed classes, the two addresses on the link, the ports the
tapped node may serve, and the payload digests declared for pinned flows.
Without one, this module behaves as it did before the policy existed.

Nothing here decides anything. It observes, counts, and reports — the tap is
passive, and so is this.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

from app.policy import (
    DIRECTION_IN,
    DIRECTION_OUT,
    DIRECTION_UNKNOWN,
    LinkPolicy,
    payload_digest,
)
from app.windows import (
    DEFAULT_MAX_GROUPS,
    DEFAULT_SAMPLE_FRAME_BYTES,
    KIND_CAPTURE_GAP,
    WindowFindingGroup,
    WindowReport,
    window_start_for,
)

log = logging.getLogger("frameprocessor.accounting")

# Bytes of a pinned payload carried on a window row, so a pin can be re-seeded
# from the ledger after a restart and compared as bytes rather than a digest.
_PIN_PAYLOAD_RECORDED_BYTES = 512

# --- link layer ---
_ETH_HEADER_LEN = 14
_ETHERTYPE_LEN = 2
_VLAN_TAG_LEN = 4
_VLAN_TPIDS = (b"\x81\x00", b"\x88\xa8")

# The smallest frame Ethernet puts on the wire. Anything whose real content is
# shorter is padded to reach it, and that padding is where a parser stops
# looking while the bytes keep travelling.
_ETH_MIN_FRAME_LEN = 60

_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_ARP = 0x0806
_ETHERTYPE_IPV6 = 0x86DD
_ETHERTYPE_LLDP = 0x88CC
_ETHERTYPE_REALTEK = 0x8899
_ETHERTYPE_PROFINET = 0x8892
_ETHERTYPE_MPLS = 0x8847
_ETHERTYPE_PPPOE_DISC = 0x8863
_ETHERTYPE_PPPOE_SESS = 0x8864
_ETHERTYPE_FLOW_CONTROL = 0x8808

# An EtherType field below the minimum frame length is an 802.3 length field
# instead, and what follows is an LLC header rather than a protocol payload.
_ETHERTYPE_IS_LENGTH_BELOW = 0x0600
_LLC_HEADER_LEN = 3
_LLC_SAP_STP = 0x42  # DSAP and SSAP both, for a bridge PDU

# --- IPv4 ---
_IPV4_IHL_MASK = 0x0F
_HEADER_WORD_LEN = 4
_MIN_IPV4_HEADER_LEN = 20
_IPV4_TOTAL_LEN_AT = 2
_IPV4_IDENT_AT = 4
_IPV4_FLAGS_AT = 6
_IPV4_PROTOCOL_AT = 9
_IPV4_CHECKSUM_AT = 10
# The IPv4 flags/fragment word: one reserved bit, Don't Fragment, More
# Fragments, then a 13-bit offset.
_IPV4_FLAG_RESERVED = 0x8000
_IPV4_FLAG_MORE = 0x2000
_IPV4_FRAG_OFFSET_MASK = 0x1FFF
# TCP header, from the start of the transport (Classification.pin_at).
_TCP_OFFSET_RESERVED_AT = 12  # high nibble data offset, low nibble reserved
_TCP_RESERVED_MASK = 0x0F
_TCP_FLAGS_AT = 13
_TCP_FLAG_URG = 0x20
_TCP_URGENT_AT = 18
_TCP_MIN_HEADER_LEN = 20
# ARP over Ethernet/IPv4, offsets from the start of the ARP payload.
_ARP_HTYPE_ETHERNET = 1
_ARP_HLEN_AT = 4
_ARP_PLEN_AT = 5
_ARP_OPCODE_AT = 6
_ARP_SENDER_HW_AT = 8
_ARP_TARGET_HW_AT = 18
_ARP_IPV4_LEN = 28
_ARP_REQUEST = 1
_ARP_REPLY = 2

# --- IPv6 ---
_IPV6_HEADER_LEN = 40
_IPV6_PAYLOAD_LEN_AT = 4
_IPV6_NEXT_HEADER_AT = 6
# Extension headers that carry (next-header, length-in-8-octet-units-minus-1).
_IPV6_EXT_HEADERS = frozenset({0, 43, 60, 135, 139, 140})
_IPV6_FRAGMENT_HEADER = 44
_IPV6_NO_NEXT_HEADER = 59

_IPPROTO_ICMP = 1
_IPPROTO_IGMP = 2
_IPPROTO_TCP = 6
_IPPROTO_UDP = 17
_IPPROTO_ICMPV6 = 58

_UDP_HEADER_LEN = 8
# Above this, a port is an ephemeral source rather than a service being
# addressed — so for an unnamed datagram the *other* port is the informative
# one. Some senders open a fresh ephemeral port per datagram, which would
# otherwise mint a new class every frame.
_EPHEMERAL_PORT_FLOOR = 49152

# --- well-known UDP ports, by what they mean on this link ---
_UDP_CLASSES = {
    53: "dns",
    67: "dhcp",
    68: "dhcp",
    123: "ntp",
    546: "dhcpv6",
    547: "dhcpv6",
    1900: "ssdp",
    4789: "vxlan",
    5353: "mdns",
}

# --- protocol families, by class name ---
CLASS_TCP = "ipv4-tcp"
CLASS_TCP6 = "ipv6-tcp"
CLASS_UNCLASSIFIED = "unclassified"
CLASS_TRUNCATED = "truncated"
CLASS_MALFORMED = "malformed"

# Pinning is the default, so a class nobody thought about is still held to
# something. These are the exceptions — flows whose payload is *supposed* to
# differ frame to frame, where a pin would raise a finding per packet and say
# nothing.
#
# SSDP and mDNS are the ones worth explaining. Both are text protocols that
# announce several device and service types per cycle, each NOTIFY carrying a
# different NT and USN, so length and content legitimately vary. That variance
# is precisely why SSDP tends to be the largest attributable side channel on a
# link rather than a constant: it needs field-level validation
# against an allowed header set, which pin-and-compare cannot provide. Leaving
# it unpinned is a statement that it is not yet covered, not that it is safe.
UNPINNED_CLASSES = frozenset(
    {
        "arp",
        "ssdp",
        "mdns",
        "dns",
        "ntp",
        "icmp",
        "vxlan",
        CLASS_TCP,
        CLASS_TCP6,
        CLASS_TRUNCATED,
        CLASS_MALFORMED,
        CLASS_UNCLASSIFIED,
    }
)

# Findings are aggregated by (kind, class) rather than stored one per frame.
# A single flow that never holds still would otherwise fill the store and
# crowd out the one finding from elsewhere that mattered — which is the
# failure mode that makes a validator worse than useless. This many samples
# are kept per group; the rest are counted.
DEFAULT_SAMPLES_PER_GROUP = 3


def ethernet_header_len(data: bytes | memoryview) -> int:
    """Bytes of Ethernet header, counting any 802.1Q / 802.1ad tags.

    The EtherType is the header's final field, so it always occupies the two
    bytes ending at ``header_len`` — and a tag displaces it rather than
    replacing it. While a TPID sits where the EtherType belongs, step over
    that tag and look again. A truncated frame needs no bounds check: a slice
    past the end comes back short, and a short slice matches no TPID.

    Mirrors ``capture._ethernet_header_len``. Kept separate because that one
    is on the TCP hot path and this one has to stay valid for every frame on
    the link, including the ones that decoder discards.
    """
    header_len = _ETH_HEADER_LEN
    while bytes(data[header_len - _ETHERTYPE_LEN : header_len]) in _VLAN_TPIDS:
        header_len += _VLAN_TAG_LEN
    return header_len


@dataclass(frozen=True)
class Classification:
    """Where a frame belongs, and which of its bytes are meant to be stable."""

    name: str
    # Offset the pin starts at: past the headers whose variance is expected
    # (IPv4 identification, UDP source port), so a constant payload reads as
    # constant. Equal to len(frame) when there is nothing left to pin.
    pin_at: int
    # Offset the protocol's own content ends at, when the protocol declares it.
    # Everything from here to the end of the frame is Ethernet padding and
    # must be zero. ``None`` when the protocol does not say.
    content_end: int | None = None
    # Byte ranges whose value the sending application does not choose: the
    # kernel assigns them, or they are computed from the rest of the frame.
    # Only whole-frame pinning consults these — the default pin starts past
    # the headers and never reaches them. See `FrameAccountant._pinned`.
    derived: tuple[tuple[int, int], ...] = ()


@dataclass
class Pin:
    """Every distinct payload a `(source, class)` pair has been seen carrying.

    A set rather than a single value, because "one payload or a finding" is
    the wrong model for a real link. Measured over an idle window, most senders
    emit exactly one payload — but a device may answer discovery with two
    payloads in strict alternation, or emit a management datagram that is
    different every frame. Collapsing those three cases into
    pass/fail throws away the number that actually characterises the flow.

    So a finding is raised per *new variant*, not per frame. Findings then
    scale with variety instead of volume: a constant sender is silent, a
    two-variant sender raises one, and an unbounded one raises enough to be
    unmistakable while `distinct` states the real figure.
    """

    variants: dict[bytes, int] = field(default_factory=dict)
    first_seen: float = 0.0
    frames: int = 0
    # Distinct payloads seen, including any beyond what is retained.
    distinct: int = 0
    # Set once the retained set is full: this flow has stopped being a set of
    # known payloads and become a stream of new ones.
    overflowed: bool = False
    # The digest the link policy said this flow must carry, when it said one.
    # With a declaration the first frame is checked rather than trusted.
    declared: str | None = None
    # True when the pin covers the frame from byte zero rather than from the
    # end of the headers — see `FrameAccountant._pin_at`.
    whole_frame: bool = False

    @property
    def constant(self) -> bool:
        return self.distinct == 1

    @property
    def digest(self) -> str:
        """Short BLAKE2b of the first payload seen, for logs and the ledger."""
        if not self.variants:
            return ""
        first = next(iter(self.variants))
        return hashlib.blake2b(first, digest_size=8).hexdigest()

    @property
    def payload_bytes(self) -> int:
        return len(next(iter(self.variants))) if self.variants else 0


# Retained distinct payloads per (source, class). Past this the flow is
# recorded as overflowed rather than remembered frame by frame — a sender that
# never repeats itself must not be able to grow this without bound.
DEFAULT_MAX_VARIANTS = 32

def _iso_ts(ts: float) -> str:
    """A timestamp a beat finding can carry into a log or a ledger row."""
    if not ts:
        return "never"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


# Findings a declared cadence can raise.
KIND_BEAT_MISSED = "beat-missed"
KIND_BEAT_UNSCHEDULED = "beat-unscheduled"

# A silence this many beats long is treated as the schedule having lost its
# place rather than as that many individual misses — a replayed capture or a
# stepped clock can put the two arbitrarily far apart, and a finding claiming
# two million missed beats says less than one saying the schedule resynced.
BEAT_RESYNC_LIMIT = 1000


@dataclass
class Beat:
    """The declared cadence of one `(source, class)`, and how it has held.

    A pin says what a flow carries; a beat says how often. Only the second can
    be failed by a flow that has stopped entirely, which is the case this
    exists for: on a link whose only other traffic is inference, no frames is
    as consistent with an idle prover as with a blind tap, and nothing else in
    this module can tell those apart.

    ``next_due`` is the schedule, and it only ever moves forward. Both the
    frame path and the timer path advance it through the same helper, so a
    beat that goes missing during a five-minute window and is noticed when the
    window closes cannot also be counted again when traffic resumes.
    """

    interval: float
    tolerance: float
    source: str
    frame_class: str
    # When the next beat is expected. 0.0 means the schedule has no anchor
    # yet — a replay, where process start says nothing about the capture.
    next_due: float = 0.0
    first_ts: float = 0.0
    last_ts: float = 0.0
    beats: int = 0
    missed: int = 0
    unscheduled: int = 0
    resyncs: int = 0
    # Set when beats have been reported missing and none has arrived since.
    # The next one to turn up re-anchors the phase instead of being judged
    # against a schedule that has already been charged for its absence.
    awaiting_return: bool = False
    # The gaps actually seen, so a declaration can be checked against the
    # cadence the beacon really keeps rather than the one it was meant to.
    min_gap: float = 0.0
    max_gap: float = 0.0
    # Which way the last beat went, so a missing leg names the port that
    # went dark rather than just the flow.
    direction: str = DIRECTION_UNKNOWN

    @property
    def armed(self) -> bool:
        return self.next_due > 0.0

    @property
    def mean_gap(self) -> float:
        """Mean interval actually observed, across every beat seen."""
        return 0.0 if self.beats < 2 else (self.last_ts - self.first_ts) / (self.beats - 1)


@dataclass(frozen=True)
class Finding:
    """Something that did not match what the link is supposed to carry."""

    kind: str  # pin-changed | unclassified | padding-nonzero | tlv-mismatch | error
    frame_class: str
    source: str  # source MAC, rendered
    detail: str
    ts: float
    # Which way the frame was going (app.policy), when the accountant knew.
    direction: str = DIRECTION_UNKNOWN


@dataclass
class ClassTotals:
    frames: int = 0
    frame_bytes: int = 0
    # Split by link direction (app.policy): what the tapped node emitted,
    # what it was sent, and frames whose direction nobody could say. The
    # three sum to ``frames``.
    frames_out: int = 0
    frames_in: int = 0
    frames_unknown: int = 0


@dataclass
class FindingGroup:
    """Every finding of one kind, for one class, folded into one row.

    A flow that never holds still produces a finding per frame. Storing those
    individually buries everything else; storing only a count loses the detail
    that makes one actionable. So: a full count, the set of sources involved,
    and the first few in full.
    """

    kind: str
    frame_class: str
    count: int = 0
    sources: set[str] = field(default_factory=set)
    samples: list[Finding] = field(default_factory=list)


@dataclass(frozen=True)
class AccountingSnapshot:
    """A point-in-time account of the link.

    ``balanced`` is the property the whole module exists to establish: every
    frame observed was placed in exactly one class. It is arithmetic rather
    than a judgement, which is what makes it worth putting in front of a
    reviewer.
    """

    observed: int
    classified: int
    classes: dict[str, ClassTotals]
    pins: dict[str, Pin]
    beats: dict[str, Beat]
    finding_groups: dict[tuple[str, str], FindingGroup]
    kernel_dropped: int
    truncated: int
    errors: int

    @property
    def balanced(self) -> bool:
        return self.observed == self.classified

    @property
    def findings(self) -> int:
        """Total findings raised, across every group."""
        return sum(group.count for group in self.finding_groups.values())

    @property
    def complete(self) -> bool:
        """True when the account covers the whole link for this window.

        Kernel drops are disqualifying and not merely noted. A window that
        lost frames cannot support a claim about what crossed the link,
        however clean the frames that did arrive look — and a prover able to
        induce congestion would exfiltrate in exactly that gap.
        """
        return self.balanced and self.kernel_dropped == 0 and self.findings == 0

    def by_direction(self) -> dict[str, ClassTotals]:
        """Frame totals folded across classes, one entry per direction."""
        out = ClassTotals()
        inbound = ClassTotals()
        unknown = ClassTotals()
        for totals in self.classes.values():
            out.frames += totals.frames_out
            inbound.frames += totals.frames_in
            unknown.frames += totals.frames_unknown
        return {DIRECTION_OUT: out, DIRECTION_IN: inbound, DIRECTION_UNKNOWN: unknown}


def classify(data: bytes | memoryview) -> Classification:
    """Which class a frame belongs to, and which of its bytes should be stable.

    Every return path names a class. There is no path that declines to answer,
    because a frame nobody classified is the thing this module exists to stop
    happening quietly.
    """
    size = len(data)
    if size < _ETH_HEADER_LEN:
        return Classification(CLASS_MALFORMED, size)

    eth_at = ethernet_header_len(data)
    if size < eth_at:
        return Classification(CLASS_MALFORMED, size)

    ethertype = struct.unpack_from("!H", data, eth_at - _ETHERTYPE_LEN)[0]

    # 802.3: the field is a length, and an LLC header follows. Spanning tree
    # lives here, which is why this is checked before the EtherType table.
    if ethertype < _ETHERTYPE_IS_LENGTH_BELOW:
        return _classify_llc(data, eth_at, ethertype)

    if ethertype == _ETHERTYPE_IPV4:
        return _classify_ipv4(data, eth_at)
    if ethertype == _ETHERTYPE_IPV6:
        return _classify_ipv6(data, eth_at)
    if ethertype == _ETHERTYPE_ARP:
        # ARP over Ethernet/IPv4 is 28 bytes; the rest of a minimum frame is
        # padding. Read the address lengths rather than assuming, so a
        # non-IPv4 ARP is not accused of carrying 18 bytes of payload.
        content_end = None
        if size >= eth_at + 8:
            hlen, plen = data[eth_at + 4], data[eth_at + 5]
            content_end = eth_at + 8 + 2 * (hlen + plen)
        return Classification("arp", eth_at, content_end)
    if ethertype == _ETHERTYPE_LLDP:
        return Classification("lldp", eth_at)
    if ethertype == _ETHERTYPE_REALTEK:
        # Named rather than folded into eth-<hex> below because switch silicon
        # emits it unprompted and a reader should not have to look the number
        # up. It is a registered EtherType emitted by common switch hardware.
        return Classification("realtek", eth_at)
    if ethertype in (
        _ETHERTYPE_PROFINET,
        _ETHERTYPE_MPLS,
        _ETHERTYPE_PPPOE_DISC,
        _ETHERTYPE_PPPOE_SESS,
        _ETHERTYPE_FLOW_CONTROL,
    ):
        return Classification(f"eth-{ethertype:04x}", eth_at)

    return Classification("eth-other", eth_at)


def _classify_llc(data: bytes | memoryview, eth_at: int, length: int) -> Classification:
    """802.3 LLC: spanning tree, or something else worth naming as LLC."""
    if len(data) < eth_at + _LLC_HEADER_LEN:
        return Classification(CLASS_MALFORMED, len(data))
    dsap, ssap = data[eth_at], data[eth_at + 1]
    if dsap == _LLC_SAP_STP and ssap == _LLC_SAP_STP:
        # The 802.3 length field counts the LLC header and the PDU, so it is
        # the protocol's own statement of where its content ends.
        content_end = eth_at + length if length else None
        return Classification("stp", eth_at, content_end)
    return Classification("llc-other", eth_at)


def _ones_complement(data: bytes | bytearray) -> int:
    """The internet checksum of ``data`` (RFC 1071), as it appears on the wire."""
    if len(data) % 2:
        data = bytes(data) + b"\x00"
    total = sum(struct.unpack(f"!{len(data) // 2}H", bytes(data)))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _classify_ipv4(data: bytes | memoryview, eth_at: int) -> Classification:
    size = len(data)
    if size < eth_at + _MIN_IPV4_HEADER_LEN:
        return Classification(CLASS_MALFORMED, size)

    ihl = (data[eth_at] & _IPV4_IHL_MASK) * _HEADER_WORD_LEN
    if ihl < _MIN_IPV4_HEADER_LEN or size < eth_at + ihl:
        return Classification(CLASS_MALFORMED, size)

    total_len = struct.unpack_from("!H", data, eth_at + _IPV4_TOTAL_LEN_AT)[0]
    # The IP header carries a per-packet identification counter, so the pin
    # starts after it. Where total length is credible it also tells us where
    # the datagram ends and Ethernet padding begins.
    content_end = eth_at + total_len if _MIN_IPV4_HEADER_LEN <= total_len <= size - eth_at else None
    transport_at = eth_at + ihl
    protocol = data[eth_at + _IPV4_PROTOCOL_AT]
    # The two fields a sender does not get to decide. Identification is
    # assigned by the sending kernel, and in practice it moves on every
    # datagram even with DF set, which is why a whole-frame pin that covered
    # it would raise a mismatch per frame and say nothing. The header
    # checksum is a function of the rest of the header, so pinning it would
    # only restate what the other bytes already say; excluded, it is verified
    # arithmetically instead (`_check_ipv4_checksum`), which is strictly
    # stronger than a pin because it holds for a frame nobody has seen before.
    derived = (
        (eth_at + _IPV4_IDENT_AT, eth_at + _IPV4_IDENT_AT + 2),
        (eth_at + _IPV4_CHECKSUM_AT, eth_at + _IPV4_CHECKSUM_AT + 2),
    )

    if protocol == _IPPROTO_TCP:
        return Classification(CLASS_TCP, transport_at, content_end, derived)
    if protocol == _IPPROTO_UDP:
        return _classify_udp(data, transport_at, content_end, tcp_class=CLASS_TCP, derived=derived)
    if protocol == _IPPROTO_IGMP:
        return Classification("igmp", transport_at, content_end, derived)
    if protocol == _IPPROTO_ICMP:
        return Classification("icmp", transport_at, content_end, derived)
    return Classification(f"ipv4-proto-{protocol}", transport_at, content_end, derived)


def _classify_ipv6(data: bytes | memoryview, eth_at: int) -> Classification:
    """IPv6, walking extension headers to reach the transport.

    The TCP decoder deliberately refuses IPv6 because the arithmetic is unsafe
    without this walk. It is needed here: SSDP v6, ICMPv6 and DHCPv6 are a
    large share of the inbound frames on the tapped link.
    """
    size = len(data)
    if size < eth_at + _IPV6_HEADER_LEN:
        return Classification(CLASS_MALFORMED, size)

    payload_len = struct.unpack_from("!H", data, eth_at + _IPV6_PAYLOAD_LEN_AT)[0]
    content_end = eth_at + _IPV6_HEADER_LEN + payload_len
    if content_end > size:
        content_end = None

    next_header = data[eth_at + _IPV6_NEXT_HEADER_AT]
    at = eth_at + _IPV6_HEADER_LEN
    # Bounded: a crafted chain of extension headers must not spin here.
    for _ in range(8):
        if next_header in _IPV6_EXT_HEADERS:
            if size < at + 2:
                return Classification(CLASS_MALFORMED, size)
            next_header, at = data[at], at + (data[at + 1] + 1) * 8
            if at > size:
                return Classification(CLASS_MALFORMED, size)
            continue
        if next_header == _IPV6_FRAGMENT_HEADER:
            # A fragment cannot be classified beyond this point without
            # reassembly, and saying so is better than guessing.
            return Classification("ipv6-fragment", at, content_end)
        break
    else:
        return Classification(CLASS_MALFORMED, size)

    if next_header == _IPPROTO_TCP:
        return Classification(CLASS_TCP6, at, content_end)
    if next_header == _IPPROTO_UDP:
        return _classify_udp(data, at, content_end, tcp_class=CLASS_TCP6)
    if next_header == _IPPROTO_ICMPV6:
        return Classification("icmpv6", at, content_end)
    if next_header == _IPV6_NO_NEXT_HEADER:
        return Classification("ipv6-no-next", at, content_end)
    return Classification(f"ipv6-proto-{next_header}", at, content_end)


def _classify_udp(
    data: bytes | memoryview,
    udp_at: int,
    content_end: int | None,
    *,
    tcp_class: str,
    derived: tuple[tuple[int, int], ...] = (),
) -> Classification:
    """Name a UDP datagram by its port, and pin past the UDP header.

    The source port is excluded from the pin on purpose: senders that open a
    fresh ephemeral port per datagram are common, and including it would make
    an otherwise byte-identical payload look different every time.
    """
    if len(data) < udp_at + _UDP_HEADER_LEN:
        return Classification(CLASS_MALFORMED, len(data))
    sport, dport = struct.unpack_from("!HH", data, udp_at)
    payload_at = udp_at + _UDP_HEADER_LEN
    name = _UDP_CLASSES.get(dport) or _UDP_CLASSES.get(sport)
    if name is None:
        # Named by port rather than lumped into one "other" bucket. Two
        # unnamed services on this link behave completely differently — one
        # is 508 bytes of zero padding that never changes, the other is the
        # only inbound flow whose payload differs every frame — and a shared
        # bucket would average them into a class that describes neither.
        service = dport if dport < _EPHEMERAL_PORT_FLOOR else sport
        return Classification(f"udp-{service}", payload_at, content_end, derived)
    if name == "vxlan":
        # VXLAN carries an inner frame the TCP path already decapsulates. It
        # is named rather than followed here: this module accounts for what
        # crossed the link, and the outer datagram is what did.
        return Classification("vxlan", payload_at, content_end, derived)
    return Classification(name, payload_at, content_end, derived)


def _lldp_tlv_end(payload: bytes | memoryview) -> int | None:
    """Offset the LLDP TLV chain ends at, or None if it is malformed.

    Each TLV is a 7-bit type and 9-bit length packed into two bytes, and the
    chain ends at a type-0 terminator. A chain that ends anywhere other than
    the end of the payload means bytes are travelling past where a reader
    stops — which is the whole reason to walk it.
    """
    at, size = 0, len(payload)
    while at + 2 <= size:
        header = struct.unpack_from("!H", payload, at)[0]
        kind, length = header >> 9, header & 0x1FF
        at += 2 + length
        if at > size:
            return None
        if kind == 0:
            return at
    return None


class _Window:
    """The account of one window while it is open. Becomes a WindowReport."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end
        self.observed = 0
        self.classified = 0
        self.kernel_dropped = 0
        self.truncated = 0
        self.errors = 0
        self.finding_count = 0
        self.groups_overflow = 0
        self.classes: dict[str, ClassTotals] = {}
        self.groups: dict[tuple[str, str, str, str], WindowFindingGroup] = {}


class FrameAccountant:
    """Counts and validates every frame, so the total can be reconciled.

    One instance per capture source. Not thread-safe and not shared between
    processes: the receiver processes each own theirs, and their snapshots are
    summed, which keeps this off any lock on the hot path.

    With ``window_seconds`` set, the same account is also cut into aligned
    windows (app.windows): frames are bucketed by their kernel timestamp,
    `roll()` closes windows the wall clock has moved past — so an idle window
    still closes, with zero frames — and each closed window is handed back as
    a `WindowReport` for the ledger. Pins are deliberately NOT per window: a
    pin is a statement about a sender across time, and resetting it every
    five minutes would let a slowly changing payload through unremarked.
    """

    def __init__(
        self,
        *,
        unpinned_classes: Iterable[str] = UNPINNED_CLASSES,
        whole_frame_classes: Iterable[str] = (),
        allow_ip_options: bool = False,
        allow_fragments: bool = False,
        samples_per_group: int = DEFAULT_SAMPLES_PER_GROUP,
        max_variants: int = DEFAULT_MAX_VARIANTS,
        on_finding: Callable[[Finding], None] | None = None,
        policy: LinkPolicy | None = None,
        iface: str = "",
        window_seconds: float | None = None,
        window_grace: float = 5.0,
        max_window_groups: int = DEFAULT_MAX_GROUPS,
        sample_frame_bytes: int = DEFAULT_SAMPLE_FRAME_BYTES,
        process_epoch: float | None = None,
        fixed_direction: str | None = None,
    ) -> None:
        self._unpinned = frozenset(unpinned_classes)
        # The one direction this capture source carries, when it carries only
        # one. A tap's monitor port does; this host's own end of a link does
        # not, and neither does a replay.
        self._fixed_direction = fixed_direction
        self._whole_frame = frozenset(whole_frame_classes)
        # Properties of the LINK, declared rather than assumed. Everything
        # else these rules compare against is a property of the protocol.
        self._allow_ip_options = allow_ip_options
        self._allow_fragments = allow_fragments
        self._samples_per_group = samples_per_group
        self._max_variants = max_variants
        self._on_finding = on_finding
        # The link whitelist, or None for classification-only accounting.
        self._policy = policy
        self.observed = 0
        self.classified = 0
        self.errors = 0
        self.truncated = 0
        self.kernel_dropped = 0
        self._classes: dict[str, ClassTotals] = {}
        self._pins: dict[str, Pin] = {}
        self._groups: dict[tuple[str, str], FindingGroup] = {}
        self._beats: dict[str, Beat] = {}

        # Windowing. ``_ctx_*`` is the frame being observed right now, so the
        # bookkeeping helpers below know which window, direction and bytes a
        # count or finding belongs to without every call site passing them.
        self.iface = iface
        self._window_seconds = window_seconds if window_seconds and window_seconds > 0 else None
        self._window_grace = max(0.0, window_grace)
        self._max_window_groups = max(1, max_window_groups)
        self._sample_frame_bytes = max(0, sample_frame_bytes)
        self.process_epoch = (
            process_epoch if process_epoch is not None else time.time()
        )
        self._window: _Window | None = None
        self._window_end_of_last_closed = 0.0
        self._windows_opened = 0
        self._closed_windows: list[WindowReport] = []
        self._ctx_window: _Window | None = None
        self._ctx_direction: str = DIRECTION_UNKNOWN
        self._ctx_frame: bytes | memoryview = b""
        self._arm_beats()

    # --- beats ------------------------------------------------------------

    def _arm_beats(self) -> None:
        """Build the declared schedules and set when each is first expected.

        A live capture anchors on process start: the first beat is due one
        interval in, so a beacon that was already dead when the tap came up
        is reported rather than waited for indefinitely. A replay has no
        meaningful epoch (``process_epoch=0.0``) and anchors on its first
        observed beat instead — there, "the beacon never started" is a
        statement about the recording, not about the link.

        A beat is a property of the LINK, but this object is per capture
        source, and a tap's monitor port carries one direction only. So a
        declared beat whose sender cannot appear on this interface is not
        armed here: arming it would report a leg that is arriving perfectly
        well on the other port as missing, once per interval, forever. Each
        interface therefore watches its own leg, and the parent merges both
        into one window row — the same split the frame counts already use.
        """
        if self._policy is None:
            return
        tolerance = max(0.0, self._policy.beat_tolerance)
        skipped = []
        for key, interval in self._policy.declared_beats.items():
            source, _, frame_class = key.partition("/")
            if self._fixed_direction is not None:
                carried = self._policy.direction_for_mac(source)
                if carried != DIRECTION_UNKNOWN and carried != self._fixed_direction:
                    skipped.append(key)
                    continue
            self._beats[key] = Beat(
                interval=interval,
                tolerance=tolerance,
                source=source,
                frame_class=frame_class,
                next_due=self.process_epoch + interval if self.process_epoch > 0 else 0.0,
            )
        if self._policy.declared_beats:
            # Logged because a beat nobody armed and a beat nobody declared
            # look identical afterwards, and a mistyped MAC would silently
            # arm nothing anywhere.
            log.info(
                "%s beats armed: %s%s",
                self.iface or "accountant",
                ", ".join(f"{k}@{b.interval:g}s" for k, b in sorted(self._beats.items()))
                or "none",
                f"; not on this interface: {', '.join(sorted(skipped))}" if skipped else "",
            )

    def _check_beat(
        self, source: str, name: str, ts: float, direction: str | None
    ) -> None:
        """Judge one frame of a declared flow against its cadence."""
        beat = self._beats.get(f"{source}/{name}")
        if beat is None:
            return
        beat.direction = direction or DIRECTION_UNKNOWN
        if beat.beats:
            gap = ts - beat.last_ts
            beat.min_gap = gap if beat.beats == 1 else min(beat.min_gap, gap)
            beat.max_gap = max(beat.max_gap, gap)
        # Every schedule, not just this one: when a round trip loses one leg,
        # the surviving leg is the frame that arrives, and sweeping only its
        # own schedule would leave the stopped one for the next timer tick.
        # Reached only on a declared flow's frames, so the hot path is unaffected.
        # Doing it here as well as on the timer is also what keeps the two
        # paths from both counting the same absence.
        self._sweep_in(self._ctx_window, ts)
        if beat.beats == 0:
            # First beat seen: it sets the phase rather than being judged
            # against a phase nobody knew yet.
            beat.first_ts = ts
            beat.awaiting_return = False
        elif beat.awaiting_return:
            # The beacon is back. Where it resumes is its own business — the
            # absence has already been reported, and judging the first frame
            # after it would charge twice for one gap.
            beat.awaiting_return = False
        elif ts < beat.next_due - beat.tolerance:
            beat.unscheduled += 1
            self._note(
                Finding(
                    KIND_BEAT_UNSCHEDULED,
                    name,
                    source,
                    f"{beat.next_due - ts:.1f}s before it was due; "
                    f"declared every {beat.interval:g}s",
                    ts,
                    beat.direction,
                )
            )
            # The schedule keeps its phase: an extra frame must not be able to
            # drag the cadence along behind it and so stop being extra.
            return
        beat.beats += 1
        beat.last_ts = ts
        beat.next_due = ts + beat.interval

    def _sweep_beat(self, beat: Beat, until: float) -> None:
        """Advance one schedule to ``until``, reporting what never arrived.

        A beat due at ``d`` is missing once ``until`` has passed ``d`` by more
        than the tolerance, so the count is how many due times fit in the
        overdue span — not how many intervals it spans, which counts the beat
        that is merely late as already missed.
        """
        if not beat.armed or beat.next_due + beat.tolerance >= until:
            return
        overdue = until - beat.tolerance - beat.next_due
        missed = math.ceil(overdue / beat.interval)
        beat.awaiting_return = True
        if missed > BEAT_RESYNC_LIMIT:
            beat.resyncs += 1
            beat.next_due = until + beat.interval
            self._note(
                Finding(
                    KIND_BEAT_MISSED,
                    beat.frame_class,
                    beat.source,
                    f"silent for {overdue:.0f}s, past {BEAT_RESYNC_LIMIT} beats; "
                    "schedule resynchronised",
                    until,
                    beat.direction,
                )
            )
            return
        beat.missed += missed
        due = beat.next_due
        beat.next_due += missed * beat.interval
        # Reported with the time of the first beat that went missing: that is
        # the window the gap belongs to, and the one a verifier reading a row
        # will be asking about.
        # One finding for the run, not one per beat: the count is the useful
        # figure and a finding per beat would bury everything else in a window.
        self._note(
            Finding(
                KIND_BEAT_MISSED,
                beat.frame_class,
                beat.source,
                f"{missed} beat(s) missed; declared every {beat.interval:g}s, "
                f"due {_iso_ts(due)}, "
                + (
                    f"nothing since {_iso_ts(beat.last_ts)}"
                    if beat.last_ts
                    else "never seen"
                ),
                due,
                beat.direction,
            )
        )

    def _sweep_beats(self, until: float) -> None:
        """Advance every schedule to ``until``. Idempotent: ``next_due`` only
        moves forward, so calling this from both the timer and a window close
        cannot double-count one absence."""
        for beat in self._beats.values():
            self._sweep_beat(beat, until)

    # --- observation -----------------------------------------------------

    def observe(
        self, data: bytes | memoryview, ts: float, direction: str | None = None
    ) -> str:
        """Account for one frame. Returns the class it was placed in.

        ``direction`` is the kernel's word on which way the frame went, when
        the source had one. Without it, and with a policy attached, direction
        is read from the source MAC instead — good enough for a replay, and
        the reason DIRECTION_UNKNOWN exists.

        Never raises. This runs inside the packet-ring walk, where an
        exception would cost the block and every frame in it — a validator
        that takes down the capture it is validating has made things worse.
        """
        self.observed += 1
        if direction is None and self._policy is not None:
            try:
                direction = self._policy.direction_from_source(data)
            except Exception:  # pragma: no cover - defensive
                direction = DIRECTION_UNKNOWN
        self._ctx_window = self._window_for(ts)
        self._ctx_direction = direction or DIRECTION_UNKNOWN
        self._ctx_frame = data
        try:
            try:
                classification = classify(data)
            except Exception as exc:  # pragma: no cover - defensive
                self.errors += 1
                self._note(
                    Finding("error", CLASS_MALFORMED, "", f"classify failed: {exc!r}", ts)
                )
                self._count(CLASS_MALFORMED, len(data), direction)
                return CLASS_MALFORMED

            name = classification.name
            self._count(name, len(data), direction)

            try:
                self._validate(data, classification, ts, direction)
            except Exception as exc:  # pragma: no cover - defensive
                self.errors += 1
                self._note(Finding("error", name, self._source(data), repr(exc), ts))
            return name
        finally:
            # The ring view is released as soon as the walk moves on; never
            # keep a reference to it past this call.
            self._ctx_frame = b""
            self._ctx_window = None

    def note_truncated(self, original_len: int, captured_len: int, ts: float = 0.0) -> None:
        """A frame the ring could not hand over whole.

        Counted as observed and classified so the arithmetic still balances —
        the frame is accounted for, as one that cannot be validated. It is
        also a finding, because a link carrying frames too large to capture
        cannot be claimed as fully parsed.
        """
        self.observed += 1
        self.truncated += 1
        self._ctx_window = self._window_for(ts)
        self._ctx_direction = DIRECTION_UNKNOWN
        self._ctx_frame = b""
        try:
            if self._ctx_window is not None:
                self._ctx_window.truncated += 1
            self._count(CLASS_TRUNCATED, captured_len)
            self._note(
                Finding(
                    "truncated",
                    CLASS_TRUNCATED,
                    "",
                    f"{original_len} -> {captured_len} bytes",
                    ts,
                )
            )
        finally:
            self._ctx_window = None

    def note_kernel_dropped(self, dropped: int) -> None:
        """Frames the kernel never handed over, since the last time this was
        called. Disqualifies the window they fell in — the open one."""
        if dropped > 0:
            self.kernel_dropped += dropped
            if self._window is not None:
                self._window.kernel_dropped += dropped

    # --- windows -------------------------------------------------------------

    @property
    def windowed(self) -> bool:
        return self._window_seconds is not None

    def _window_for(self, ts: float) -> _Window | None:
        """The open window a frame at ``ts`` belongs to, rolling as needed."""
        if self._window_seconds is None:
            return None
        start = window_start_for(ts, self._window_seconds)
        if self._window is None:
            self._open(start)
        elif start >= self._window.end:
            # Frames arrive in kernel order, so a later bucket means this one
            # is over. A frame stamped before the open window (the walk
            # crossed a boundary between stamp and read) is counted where we
            # are.
            self._advance_to(start)
        return self._window

    def _advance_to(self, start: float) -> None:
        """Close the open window and open the one at ``start``.

        Empty buckets in between are opened and closed in turn, so the ledger
        sees the heartbeat rather than a hole — up to a point. A jump of more
        than two windows is not idleness, it is a clock that disagrees with
        the frames (a replayed capture, a stepped wall clock), and minting
        hundreds of empty rows to paper over it would be a lie about
        coverage. Those windows are left missing, which reads as uncovered.
        """
        assert self._window is not None and self._window_seconds is not None
        if start - self._window.end > 2 * self._window_seconds:
            log.warning(
                "%s window clock jumped %.0fs (from %.0f to %.0f); windows between "
                "are not reported",
                self.iface or "accountant",
                start - self._window.end,
                self._window.end,
                start,
            )
            self._close()
            self._open(start)
            return
        while self._window is None or start >= self._window.end:
            if self._window is not None:
                self._close()
            self._open(self._window_end_of_last_closed if self._window is None else start)

    def _open(self, start: float) -> None:
        assert self._window_seconds is not None
        self._window = _Window(start, start + self._window_seconds)
        self._windows_opened += 1
        if self._windows_opened == 1 and self.process_epoch > start:
            # The first window did not see its beginning: whatever crossed
            # the link before this process started is unaccounted, and the
            # window has to say so rather than pass as clean.
            self._note_in_window(
                self._window,
                Finding(
                    KIND_CAPTURE_GAP,
                    "capture",
                    "",
                    f"capture started {self.process_epoch - start:.0f}s into the window",
                    self.process_epoch,
                ),
            )

    def _close(self, until: float | None = None) -> None:
        """Close the open window. ``until`` bounds how far its beats are
        judged — the window's own end, except at shutdown, where a beat due
        after the capture stopped was never missed, only unobserved."""
        window = self._window
        assert window is not None
        # Sweep before the report is built, with the window this one is being
        # filed against: a beat due at 12:03 belongs to the 12:00 window, not
        # to whichever window happens to be open when its absence is noticed.
        # Empty windows opened and closed by ``_advance_to`` come through here
        # too, so a long silence taints every window it spans rather than one.
        self._sweep_in(window, window.end if until is None else min(until, window.end))
        self._window_end_of_last_closed = window.end
        self._closed_windows.append(self._report(window))
        self._window = None

    def _sweep_in(self, window: _Window | None, until: float) -> None:
        """Sweep every schedule to ``until``, filing findings on ``window``.

        The findings this raises have no frame behind them — even when a frame
        is what prompted the sweep, it is not the frame that went missing — so
        the sample bytes are cleared and the direction comes from the beat
        rather than from whatever was last observed.
        """
        if not self._beats:
            return
        saved = (self._ctx_window, self._ctx_direction, self._ctx_frame)
        self._ctx_window, self._ctx_direction, self._ctx_frame = (
            window,
            DIRECTION_UNKNOWN,
            b"",
        )
        try:
            self._sweep_beats(until)
        finally:
            self._ctx_window, self._ctx_direction, self._ctx_frame = saved

    def _report(self, window: _Window) -> WindowReport:
        return WindowReport(
            iface=self.iface,
            window_start=window.start,
            window_end=window.end,
            process_epoch=self.process_epoch,
            observed=window.observed,
            classified=window.classified,
            kernel_dropped=window.kernel_dropped,
            truncated=window.truncated,
            errors=window.errors,
            classes={
                name: {
                    "frames": t.frames,
                    "bytes": t.frame_bytes,
                    "out": t.frames_out,
                    "in": t.frames_in,
                    "unknown": t.frames_unknown,
                }
                for name, t in window.classes.items()
            },
            pins={
                key: {
                    "digest": pin.digest,
                    "declared": pin.declared,
                    "frames": pin.frames,
                    "distinct": pin.distinct,
                    "payload_b64": (
                        base64.b64encode(
                            next(iter(pin.variants))[:_PIN_PAYLOAD_RECORDED_BYTES]
                        ).decode("ascii")
                        if pin.variants
                        else None
                    ),
                }
                for key, pin in self._pins.items()
            },
            groups=list(window.groups.values()),
            groups_overflow=window.groups_overflow,
            finding_count=window.finding_count,
        )

    def roll(self, now: float) -> list[WindowReport]:
        """Close every window the wall clock has moved past; return them.

        Called from the receiver loop on a timer. This is what makes an idle
        window close: frames alone cannot end a window nothing arrives in.
        For the same reason it is also where declared beats are checked
        against the clock — a beacon that has stopped produces no frame to
        notice it by.
        """
        if self._window_seconds is None:
            # Unwindowed, but a declared beat still has to be checked against
            # the clock — a flow that stopped is invisible to every path that
            # starts from a frame.
            self._sweep_in(None, now)
            return []
        if self._window is None:
            self._open(window_start_for(now, self._window_seconds))
        elif now >= self._window.end + self._window_grace:
            self._advance_to(window_start_for(now - self._window_grace, self._window_seconds))
        # Then the open window, so a beacon that stops is reported within a
        # beat of stopping rather than when the window it stopped in closes.
        self._sweep_in(self._window, now)
        return self.take_closed()

    def finish(self, now: float) -> list[WindowReport]:
        """Close the open window at shutdown.

        A window the clock had not yet finished is marked as cut short: what
        crossed the link between ``now`` and its end is unaccounted, and the
        row has to say so rather than pass as clean. A window whose end has
        already passed simply closes.
        """
        if self._window is None:
            self._sweep_in(None, now)
            return self.take_closed()
        if now < self._window.end:
            self._note_in_window(
                self._window,
                Finding(
                    KIND_CAPTURE_GAP,
                    "capture",
                    "",
                    f"capture stopped {self._window.end - now:.0f}s before the window ended",
                    now,
                ),
            )
        self._close(until=now)
        return self.take_closed()

    def take_closed(self) -> list[WindowReport]:
        closed, self._closed_windows = self._closed_windows, []
        return closed

    # --- validation ------------------------------------------------------

    def _validate(
        self,
        data: bytes | memoryview,
        classification: Classification,
        ts: float,
        direction: str | None = None,
    ) -> None:
        name = classification.name

        if name == CLASS_UNCLASSIFIED or name.endswith("-other"):
            self._note(
                Finding(
                    "unclassified",
                    name,
                    self._source(data),
                    f"no rule matched, {len(data)} bytes",
                    ts,
                )
            )
        if name == CLASS_MALFORMED:
            self._note(
                Finding("malformed", name, self._source(data), f"{len(data)} bytes", ts)
            )
            return

        self._check_padding(data, classification, ts)
        self._check_ipv4_header(data, classification, ts)
        if name in (CLASS_TCP, CLASS_TCP6):
            self._check_tcp_header(data, classification, ts)
        if name == "lldp":
            self._check_lldp(data, classification, ts)
        if name == "arp":
            self._check_arp(data, classification, ts)
        # Pinned unless the class is a declared exception — so a protocol
        # nobody anticipated is still held to something rather than waved past.
        if name not in self._unpinned:
            self._check_pin(data, classification, ts)
        # And, for a flow declared to keep a cadence, whether it kept it. This
        # is the only check here a flow can fail by not arriving.
        if self._beats:
            self._check_beat(self._source(data), name, ts, direction)
        # Then the whitelist: is this class, from this sender, between these
        # addresses, allowed to be travelling this way at all.
        if self._policy is not None:
            source = self._source(data)
            for kind, detail in self._policy.check(
                data, name, classification.pin_at, direction or DIRECTION_UNKNOWN
            ):
                self._note(Finding(kind, name, source, detail, ts))

    def _check_padding(
        self, data: bytes | memoryview, classification: Classification, ts: float
    ) -> None:
        """Everything past the protocol's declared content must be zero.

        Only checked where the protocol states its own length, and only on
        frames short enough for padding to be the explanation. A longer frame
        with trailing bytes is a parsing disagreement, not padding, and saying
        "padding" about it would be wrong.
        """
        end = classification.content_end
        if end is None or end >= len(data):
            return
        if len(data) > _ETH_MIN_FRAME_LEN:
            self._note(
                Finding(
                    "trailing-bytes",
                    classification.name,
                    self._source(data),
                    f"{len(data) - end} bytes past declared content in a "
                    f"{len(data)}-byte frame",
                    ts,
                )
            )
            return
        if any(data[end:]):
            self._note(
                Finding(
                    "padding-nonzero",
                    classification.name,
                    self._source(data),
                    f"{len(data) - end} padding bytes, "
                    f"0x{bytes(data[end:]).hex()}",
                    ts,
                )
            )

    def _check_lldp(
        self, data: bytes | memoryview, classification: Classification, ts: float
    ) -> None:
        payload = data[classification.pin_at :]
        end = _lldp_tlv_end(payload)
        if end is None:
            self._note(
                Finding(
                    "tlv-mismatch",
                    "lldp",
                    self._source(data),
                    "TLV chain has no terminator or overruns the frame",
                    ts,
                )
            )
            return
        if end != len(payload):
            self._note(
                Finding(
                    "tlv-mismatch",
                    "lldp",
                    self._source(data),
                    f"{len(payload) - end} bytes after End-of-LLDPDU",
                    ts,
                )
            )

    def _check_arp(
        self, data: bytes | memoryview, classification: Classification, ts: float
    ) -> None:
        """ARP is not pinned, so every field is constrained individually.

        On a segment with many hosts a pin would be wrong — a router resolving
        several neighbours emits a different target each time. On a link with
        exactly two ends nothing varies at all: the addresses are the two
        declared ones, the opcode is a request or a reply, and every remaining
        field is fixed by the protocol. So each is held to the one value it
        can legitimately have, which reaches the same place a pin would
        without needing to learn anything or declare a digest.

        The four descriptors come first, and a wrong one stops the rest. They
        are not decoration: `hlen` and `plen` say where every later field
        begins, so reading them without checking them meant a frame could
        declare a length nobody validated and walk the identity and address
        rules off their offsets. Both used to be read and neither was checked.
        """
        source = self._source(data)
        at = classification.pin_at
        if len(data) < at + 8:
            return

        # Hardware type, protocol type, and the two address lengths. Ethernet
        # and IPv4 are the only combination this link carries.
        htype, ptype = struct.unpack_from("!HH", data, at)
        hlen, plen = data[at + _ARP_HLEN_AT], data[at + _ARP_PLEN_AT]
        wrong = []
        if htype != _ARP_HTYPE_ETHERNET:
            wrong.append(f"hardware type {htype}, expected 1 (Ethernet)")
        if ptype != _ETHERTYPE_IPV4:
            wrong.append(f"protocol type {ptype:#06x}, expected 0x0800 (IPv4)")
        if hlen != 6:
            wrong.append(f"hardware address length {hlen}, expected 6")
        if plen != 4:
            wrong.append(f"protocol address length {plen}, expected 4")
        if wrong:
            self._note(Finding("arp-shape", "arp", source, "; ".join(wrong), ts))
            # Every offset below is derived from hlen and plen. With either
            # wrong there is nothing sound left to read.
            return

        opcode = struct.unpack_from("!H", data, at + _ARP_OPCODE_AT)[0]
        if opcode not in (_ARP_REQUEST, _ARP_REPLY):
            self._note(Finding("arp-opcode", "arp", source, f"opcode {opcode}", ts))
            return

        if len(data) < at + _ARP_IPV4_LEN:
            return

        claimed = bytes(data[at + _ARP_SENDER_HW_AT : at + _ARP_SENDER_HW_AT + 6])
        actual = bytes(data[6:12])
        if claimed != actual:
            self._note(
                Finding(
                    "arp-source-mismatch",
                    "arp",
                    source,
                    f"claims {claimed.hex(':')} but was sent from {actual.hex(':')}",
                    ts,
                )
            )

        # The target hardware address: six bytes the protocol says are ignored
        # in a request, and which nothing checked. Both cases are nonetheless
        # fully determined — a request does not know the address it is asking
        # for, and a reply is telling one host in particular.
        target_hw = bytes(data[at + _ARP_TARGET_HW_AT : at + _ARP_TARGET_HW_AT + 6])
        if opcode == _ARP_REQUEST and target_hw != bytes(6):
            self._note(
                Finding(
                    "arp-target-hw",
                    "arp",
                    source,
                    f"request naming target {target_hw.hex(':')}; the field is "
                    "ignored on a request and every implementation zeroes it",
                    ts,
                )
            )
        elif opcode == _ARP_REPLY and target_hw != bytes(data[:6]):
            self._note(
                Finding(
                    "arp-target-hw",
                    "arp",
                    source,
                    f"reply naming target {target_hw.hex(':')} but addressed to "
                    f"{bytes(data[:6]).hex(':')}",
                    ts,
                )
            )

    def _pinned(self, data: bytes | memoryview, classification: Classification) -> bytes:
        """The bytes this class is held to, and where they start.

        Normally the transport payload: the pin begins past the headers,
        because on general traffic the IPv4 identification and a rotating UDP
        source port would make a constant flow look variable and reduce the
        rule to noise.

        A declared whole-frame class is held to the frame from byte zero
        instead, which matters for a flow that exists to be constant — pinning
        only the payload leaves DSCP, total length, flags, TTL, the source
        port and the UDP length checked by nothing at all.

        Except that a sender does not choose every byte. The identification
        field is assigned by the sending kernel and in practice moves on
        every datagram even with DF set; the header checksum is
        computed from the rest of the header. Pinning those would fail once
        per frame and restate the others respectively, so they are masked to
        zero here and covered another way: `_check_ipv4_checksum` verifies the
        checksum arithmetically, which is stronger than a pin because it holds
        for a frame nobody has seen before, and identification is left as the
        one field a whole-frame pin does not constrain.

        Masked rather than spliced out, so the stored variant still lines up
        with the frame it came from when someone reads it back off a row.
        """
        if classification.name not in self._whole_frame:
            return bytes(data[classification.pin_at :])
        frame = bytearray(data)
        for start, end in classification.derived:
            if 0 <= start < end <= len(frame):
                frame[start:end] = bytes(end - start)
        return bytes(frame)

    def _check_ipv4_header(
        self, data: bytes | memoryview, classification: Classification, ts: float
    ) -> None:
        """The IPv4 header fields that should never vary on this link.

        Both are places a frame can carry bytes past where anything looks.

        **Options.** The header length is read only to find where the
        transport begins, and any value from 5 to 15 words is accepted — so up
        to forty bytes of arbitrary option data rides in every packet,
        examined by nothing. Whether they are legitimate is a property of the
        link (`allow_ip_options`), not of IPv4: nothing on a point-to-point
        segment has any use for them, but IGMP elsewhere carries Router Alert.

        **Fragmentation.** Likewise declared (`allow_fragments`): a link with
        one MTU and no hop between its ends never fragments, so More Fragments
        or a non-zero offset is evasion or a fault — and a reassembly this
        module does not perform is one nobody is checking. The reserved bit is
        not part of that judgement; it has no meaning under any configuration.
        """
        eth_at = ethernet_header_len(data)
        if len(data) < eth_at + _MIN_IPV4_HEADER_LEN:
            return
        if data[eth_at] >> 4 != 4:
            return
        ihl = (data[eth_at] & _IPV4_IHL_MASK) * _HEADER_WORD_LEN
        if ihl > _MIN_IPV4_HEADER_LEN and not self._allow_ip_options:
            options = bytes(data[eth_at + _MIN_IPV4_HEADER_LEN : eth_at + ihl])
            self._note(
                Finding(
                    "ipv4-options",
                    classification.name,
                    self._source(data),
                    f"{len(options)} bytes of IPv4 options, examined by nothing: "
                    f"{options[:16].hex()}",
                    ts,
                )
            )
        flags = struct.unpack_from("!H", data, eth_at + _IPV4_FLAGS_AT)[0]
        detail = []
        if flags & _IPV4_FLAG_RESERVED:
            # Not covered by the fragmentation setting: the reserved bit has
            # no meaning under any configuration.
            detail.append("reserved bit set")
        if not self._allow_fragments:
            if flags & _IPV4_FLAG_MORE:
                detail.append("More Fragments set")
            if flags & _IPV4_FRAG_OFFSET_MASK:
                detail.append(f"fragment offset {flags & _IPV4_FRAG_OFFSET_MASK}")
        if detail:
            self._note(
                Finding(
                    "ipv4-fragmented",
                    classification.name,
                    self._source(data),
                    ", ".join(detail),
                    ts,
                )
            )

    def _check_tcp_header(
        self, data: bytes | memoryview, classification: Classification, ts: float
    ) -> None:
        """The two TCP header fields that carry no meaning at all.

        The four bits below the data offset are reserved and the RFC requires
        them to be zero. The urgent pointer is sixteen bits the receiver is
        instructed to IGNORE unless the URG flag is set — so on a stream that
        never sets URG, it is a field both endpoints agree means nothing,
        present in every segment. Requiring zero costs nothing and is the
        cheapest sixteen bits on the link.
        """
        at = classification.pin_at
        if len(data) < at + _TCP_MIN_HEADER_LEN:
            return
        reserved = data[at + _TCP_OFFSET_RESERVED_AT] & _TCP_RESERVED_MASK
        if reserved:
            self._note(
                Finding(
                    "tcp-reserved",
                    classification.name,
                    self._source(data),
                    f"reserved bits {reserved:#06b} set, must be zero",
                    ts,
                )
            )
        urgent = struct.unpack_from("!H", data, at + _TCP_URGENT_AT)[0]
        if urgent and not data[at + _TCP_FLAGS_AT] & _TCP_FLAG_URG:
            self._note(
                Finding(
                    "tcp-urgent",
                    classification.name,
                    self._source(data),
                    f"urgent pointer {urgent:#06x} with URG clear — the field is "
                    "ignored, so it carries nothing but whatever was put in it",
                    ts,
                )
            )

    def _check_ipv4_checksum(
        self, data: bytes | memoryview, classification: Classification, ts: float
    ) -> None:
        """The IPv4 header checksum must be the one the header implies.

        Only for whole-frame classes, and only because the pin excludes it:
        an excluded field that nothing else checks is two free bytes. Verified
        rather than pinned because arithmetic covers a frame nobody has seen,
        where a pin can only compare against one that has.

        Deliberately not applied to every IPv4 frame. A capture taken on a
        sending host sees checksums before the NIC computes them — TX offload
        leaves them zero — so a replay of such a file would raise a finding
        per frame. On the tap the frames are post-NIC and the field is real.
        """
        eth_at = ethernet_header_len(data)
        if len(data) < eth_at + _MIN_IPV4_HEADER_LEN:
            return
        if struct.unpack_from("!H", data, eth_at)[0] >> 12 != 4:
            return  # not IPv4; nothing to check
        ihl = (data[eth_at] & _IPV4_IHL_MASK) * _HEADER_WORD_LEN
        if ihl < _MIN_IPV4_HEADER_LEN or len(data) < eth_at + ihl:
            return
        header = bytearray(data[eth_at : eth_at + ihl])
        stated = struct.unpack_from("!H", header, _IPV4_CHECKSUM_AT)[0]
        header[_IPV4_CHECKSUM_AT : _IPV4_CHECKSUM_AT + 2] = b"\x00\x00"
        if _ones_complement(header) != stated:
            self._note(
                Finding(
                    "ipv4-checksum",
                    classification.name,
                    self._source(data),
                    f"header checksum {stated:#06x}, header implies "
                    f"{_ones_complement(header):#06x}",
                    ts,
                )
            )

    def _check_pin(
        self, data: bytes | memoryview, classification: Classification, ts: float
    ) -> None:
        source = self._source(data)
        key = f"{source}/{classification.name}"
        whole_frame = classification.name in self._whole_frame
        if whole_frame:
            self._check_ipv4_checksum(data, classification, ts)
        pin = self._pins.get(key)
        if pin is None:
            declared = self._policy.declared_digest(key) if self._policy else None
            pin = self._pins[key] = Pin(
                first_seen=ts, declared=declared, whole_frame=whole_frame
            )
            if self._policy is not None and declared is None:
                # A constant flow the whitelist did not mention. It will still
                # be pinned to whatever this frame carries, but the point of
                # declaring pins is that nothing gets to establish itself.
                self._note(
                    Finding(
                        "pin-undeclared",
                        classification.name,
                        source,
                        f"pinned class with no declared payload; "
                        f"first seen carrying {payload_digest(self._pinned(data, classification))}",
                        ts,
                    )
                )

        pin.frames += 1
        payload = self._pinned(data, classification)
        seen = pin.variants.get(payload)
        if seen is not None:
            pin.variants[payload] = seen + 1
            return

        pin.distinct += 1
        if len(pin.variants) < self._max_variants:
            pin.variants[payload] = 1
        else:
            pin.overflowed = True

        if pin.declared is not None:
            # Declared: every distinct payload, the first included, is judged
            # against the declaration rather than against each other. A second
            # variant necessarily fails this, so it is one finding, not two.
            digest = payload_digest(payload)
            if digest != pin.declared:
                self._note(
                    Finding(
                        "pin-mismatch",
                        classification.name,
                        source,
                        f"payload {digest} ({len(payload)} bytes) does not match "
                        f"declared {pin.declared}",
                        ts,
                    )
                )
            return

        if pin.distinct == 1:
            return  # the first payload establishes the pin; not a finding

        self._note(
            Finding(
                "variant-added",
                classification.name,
                source,
                f"payload #{pin.distinct} for this sender "
                f"({len(payload)} bytes; first pin {pin.digest} at "
                f"{pin.first_seen:.3f})",
                ts,
            )
        )

    # --- bookkeeping -----------------------------------------------------

    def _count(self, name: str, size: int, direction: str | None = None) -> None:
        for classes in (self._classes, self._ctx_window.classes if self._ctx_window else None):
            if classes is None:
                continue
            totals = classes.get(name)
            if totals is None:
                totals = classes[name] = ClassTotals()
            totals.frames += 1
            totals.frame_bytes += size
            if direction == DIRECTION_OUT:
                totals.frames_out += 1
            elif direction == DIRECTION_IN:
                totals.frames_in += 1
            else:
                totals.frames_unknown += 1
        self.classified += 1
        if self._ctx_window is not None:
            self._ctx_window.observed += 1
            self._ctx_window.classified += 1

    def _note(self, finding: Finding) -> None:
        if finding.direction == DIRECTION_UNKNOWN and self._ctx_direction != DIRECTION_UNKNOWN:
            finding = dataclasses.replace(finding, direction=self._ctx_direction)
        key = (finding.kind, finding.frame_class)
        group = self._groups.get(key)
        if group is None:
            group = self._groups[key] = FindingGroup(finding.kind, finding.frame_class)
        group.count += 1
        if finding.source:
            group.sources.add(finding.source)
        if len(group.samples) < self._samples_per_group:
            group.samples.append(finding)
            # Logged once per sample, not once per frame: the count is in the
            # snapshot, and a warning per frame is how a log stops being read.
            log.warning(
                "%s: %s from %s — %s",
                finding.kind,
                finding.frame_class,
                finding.source or "?",
                finding.detail,
            )
        if self._ctx_window is not None:
            if finding.kind == "error":
                self._ctx_window.errors += 1
            self._note_in_window(self._ctx_window, finding, self._ctx_frame)
        if self._on_finding is not None:
            try:
                self._on_finding(finding)
            except Exception:  # pragma: no cover - defensive
                self.errors += 1

    def _note_in_window(
        self, window: _Window, finding: Finding, frame: bytes | memoryview = b""
    ) -> None:
        """File a finding on a window: by (kind, class, direction, sender),
        bounded in groups per window and in retained samples per group."""
        window.finding_count += 1
        key = (finding.kind, finding.frame_class, finding.direction, finding.source)
        group = window.groups.get(key)
        if group is None:
            if len(window.groups) >= self._max_window_groups:
                # Variety past the cap: the window is tainted regardless, and
                # the count says how much was not kept in detail.
                window.groups_overflow += 1
                return
            group = window.groups[key] = WindowFindingGroup(
                finding.kind, finding.frame_class, finding.direction, finding.source
            )
        sample = b""
        if frame and len(group.samples) < self._samples_per_group and self._sample_frame_bytes:
            sample = bytes(frame[: self._sample_frame_bytes])
        group.add(finding.ts, finding.detail, sample, samples_per_group=self._samples_per_group)

    @staticmethod
    def _source(data: bytes | memoryview) -> str:
        if len(data) < 12:
            return ""
        return bytes(data[6:12]).hex(":")

    def snapshot(self) -> AccountingSnapshot:
        return AccountingSnapshot(
            observed=self.observed,
            classified=self.classified,
            classes={
                name: ClassTotals(
                    t.frames, t.frame_bytes, t.frames_out, t.frames_in, t.frames_unknown
                )
                for name, t in self._classes.items()
            },
            pins={
                key: Pin(
                    dict(pin.variants),
                    pin.first_seen,
                    pin.frames,
                    pin.distinct,
                    pin.overflowed,
                    pin.declared,
                    pin.whole_frame,
                )
                for key, pin in self._pins.items()
            },
            beats={key: dataclasses.replace(beat) for key, beat in self._beats.items()},
            finding_groups={
                key: FindingGroup(
                    group.kind,
                    group.frame_class,
                    group.count,
                    set(group.sources),
                    list(group.samples),
                )
                for key, group in self._groups.items()
            },
            kernel_dropped=self.kernel_dropped,
            truncated=self.truncated,
            errors=self.errors,
        )
