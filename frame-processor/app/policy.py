"""Link policy — what the tapped link is allowed to carry, in each direction.

`accounting.py` answers "what is this frame?" for every frame on the link.
This module answers the next question, "is it allowed to be here?", and the
answer depends on which way the frame was going. The tapped node is one
machine on a point-to-point link, so everything it emits is one of a short
list of things with a fixed shape, and anything else is a finding. The list
for the other direction is shorter still, because the other end is us.

**Direction comes from the capture, not the frame.** A frame's source MAC is
whatever the sender wrote, so a policy that reads direction from it can be
talked out of applying. Where direction really comes from depends on where
the capture sits. On a hardware tap each monitor port carries exactly one
direction, so the interface *is* the direction and it is declared per port
(`FRAME_PROCESSOR_IFACE_DIRECTION`, applied by `FixedDirection`). On this host's
own end of a point-to-point link, the TPACKET_V3 ring records for every
frame whether the host sent it (`PACKET_OUTGOING`) or received it from the
wire, and that bit cannot be forged by the peer. Either way the source MAC
becomes something to check against the direction rather than the thing that
decides it.

The kernel bit is wrong on a monitor port — a port that never transmits sees
nothing as outgoing, so both directions would read as ``out`` — which is
why the per-interface declaration exists and takes precedence.

The two directions are named from the tapped node's point of view, because
that is the node whose traffic is being constrained:

- ``out`` — the tapped node emitted it; it arrived here from the wire.
- ``in``  — the tapped node received it; this host sent it.

A pcap replay has no kernel bit, so replay falls back to the source MAC.
That is fine for developing rules against a stored window and is why
``unknown`` exists: a MAC that is neither end of the link.

**Declared pins.** `accounting.py` pins a constant flow to the first payload
it sees. On an isolated link with one peer, the constant flows are few enough
to be written down, and a declared digest means the first frame is checked
rather than trusted. A pinned class that turns up from a sender with no
declaration is itself a finding — the whitelist is meant to be complete.

Nothing here decides anything either. It reports; the tap is passive.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import struct
from dataclasses import dataclass, field

log = logging.getLogger("frameprocessor.policy")

# --- direction ---
DIRECTION_OUT = "out"  # emitted by the tapped node
DIRECTION_IN = "in"  # received by the tapped node (sent by this host)
DIRECTION_UNKNOWN = "unknown"  # replay only: source MAC is neither end
DIRECTIONS = (DIRECTION_OUT, DIRECTION_IN, DIRECTION_UNKNOWN)

# sockaddr_ll.sll_pkttype values (linux/if_packet.h). Everything that is not
# PACKET_OUTGOING arrived from the wire — unicast to us, broadcast, multicast,
# or addressed to somebody else and seen because the socket is promiscuous.
PACKET_OUTGOING = 4




def direction_from_pkttype(pkttype: int | None) -> str | None:
    """Kernel packet type to link direction; None when the ring had none.

    Only meaningful on this host's own end of the link. A tap's monitor port
    never transmits, so there every frame reads as received; use
    `FixedDirection` for those.
    """
    if pkttype is None:
        return None
    return DIRECTION_IN if pkttype == PACKET_OUTGOING else DIRECTION_OUT


class FixedDirection:
    """An observer adapter that stamps every frame with one direction.

    For a monitor port on a hardware tap, where the interface itself says
    which way the traffic was going and the ring's packet type says nothing
    useful. Wraps the accountant the receiver would otherwise attach directly;
    the accountant never learns whether its direction came from here or from
    the kernel, which keeps it one implementation for both capture layouts.
    """

    def __init__(self, inner, direction: str) -> None:
        if direction not in (DIRECTION_IN, DIRECTION_OUT):
            raise ValueError(f"direction must be 'in' or 'out', not {direction!r}")
        self._inner = inner
        self.direction = direction

    def observe(self, data, ts: float, direction: str | None = None) -> str:
        # The caller's direction is ignored on purpose: on a monitor port it
        # is the kernel bit, and the kernel bit is the thing that is wrong.
        return self._inner.observe(data, ts, self.direction)

    def note_truncated(self, original_len: int, captured_len: int, ts: float = 0.0) -> None:
        self._inner.note_truncated(original_len, captured_len, ts)

    def __getattr__(self, name: str):
        # snapshot(), note_kernel_dropped() and the counters live on the
        # accountant; expose them so callers holding the adapter need not know.
        return getattr(self._inner, name)


# --- link layer, duplicated from accounting.py on purpose ---
# accounting imports this module, so this module cannot import accounting.
_ETH_HEADER_LEN = 14
_ETHERTYPE_LEN = 2
_VLAN_TAG_LEN = 4
_VLAN_TPIDS = (b"\x81\x00", b"\x88\xa8")
_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_ARP = 0x0806
_IPV4_SRC_AT = 12
_IPV4_DST_AT = 16
_ARP_OPCODE_AT = 6
_ARP_REQUEST = 1
_ARP_REPLY = 2
_BROADCAST_MAC = b"\xff" * 6
_ARP_HLEN_AT = 4
_ARP_PLEN_AT = 5
_ARP_SENDER_IP_AT = 14  # for hlen 6, plen 4
_ARP_TARGET_IP_AT = 24
_ARP_IPV4_LEN = 28

# Field offsets within a TCP header (RFC 9293), for the dial-out check. Only
# the flags byte is needed; duplicated from capture.py for the same reason as
# the ethernet constants above — accounting imports this module, so it cannot
# import back the other way.
_TCP_FLAGS_AT = 13
_TH_SYN = 0x02
_TH_ACK = 0x10

CLASS_TCP = "ipv4-tcp"
CLASS_TCP6 = "ipv6-tcp"
CLASS_ARP = "arp"

# Classes the "not allowed" rule does not repeat itself about: each already
# raises its own finding in accounting (malformed, unclassified), so a second
# finding for the same frame would count one problem twice.
_ALREADY_FLAGGED = frozenset({"malformed", "truncated", "unclassified"})


def _eth_at(data: bytes | memoryview) -> int:
    header_len = _ETH_HEADER_LEN
    while bytes(data[header_len - _ETHERTYPE_LEN : header_len]) in _VLAN_TPIDS:
        header_len += _VLAN_TAG_LEN
    return header_len


def _mac(value: str | None) -> bytes | None:
    if not value or not value.strip():
        return None
    return bytes.fromhex(value.strip().replace(":", "").replace("-", ""))


def _ip(value: str | None) -> bytes | None:
    if not value or not value.strip():
        return None
    return ipaddress.ip_address(value.strip()).packed


def _render_mac(raw: bytes) -> str:
    return bytes(raw).hex(":")


def _render_ip(raw: bytes) -> str:
    return str(ipaddress.ip_address(bytes(raw)))


def _csv(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(p.strip() for p in value.split(",") if p.strip())


def _ports(value: str | None) -> frozenset[int]:
    ports: set[int] = set()
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            try:
                ports.add(int(part))
            except ValueError:
                log.warning("ignoring non-numeric port %r in FRAME_PROCESSOR_PEER_PORTS", part)
    return frozenset(ports)


def _parse_declarations(value: str | None, want: str) -> dict[str, str]:
    """``mac/class=value`` pairs, comma-separated, keyed as accounting keys them.

    The grammar every per-flow declaration uses, so a pin and a beat are
    written the same way and a reader who has learned one has learned both.
    ``want`` is only for the warning on a line that does not parse.
    """
    declared: dict[str, str] = {}
    for pair in (value or "").split(","):
        key, sep, setting = pair.strip().partition("=")
        if not sep or not key or not setting:
            continue
        source, slash, name = key.partition("/")
        if not slash or not source or not name:
            log.warning("ignoring malformed declaration %r (want %s)", pair, want)
            continue
        declared[f"{_render_mac(_mac(source) or b'')}/{name.strip()}"] = setting.strip().lower()
    return declared


def parse_declared_pins(value: str | None) -> dict[str, str]:
    """``mac/class=digest`` pairs: the payload a pinned flow must carry.

    e.g. ``00:00:5e:00:53:01/udp-9999=0123456789abcdef``. The digest is the
    16-hex-character BLAKE2b that `Pin.digest` reports, so a declaration can
    be copied straight out of a `tools/account.py` run.
    """
    return _parse_declarations(value, "mac/class=digest")


def parse_declared_beats(value: str | None) -> dict[str, float]:
    """``mac/class=seconds`` pairs: the cadence a flow must keep.

    e.g. ``00:00:5e:00:53:01/arp=30``. A pin says what a flow carries and a
    beat says how often it carries it — the second is the only one of the two
    that a flow which has stopped entirely can fail, which is the whole
    reason for declaring it.
    """
    beats: dict[str, float] = {}
    for key, setting in _parse_declarations(value, "mac/class=seconds").items():
        try:
            seconds = float(setting)
        except ValueError:
            log.warning("ignoring beat %s=%r: not a number of seconds", key, setting)
            continue
        if seconds <= 0:
            log.warning("ignoring beat %s=%s: cadence must be positive", key, setting)
            continue
        beats[key] = seconds
    return beats


def payload_digest(payload: bytes) -> str:
    """The digest a declaration is compared against — same as `Pin.digest`."""
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


@dataclass(frozen=True)
class LinkPolicy:
    """The whitelist for one point-to-point link.

    Every field is optional except the peer MAC, which is what says "the
    policy is on". Rules whose inputs are missing are skipped, so the policy
    can be tightened one setting at a time and never fails closed on a link
    it was not told enough about.
    """

    peer_mac: bytes  # the tapped node's NIC
    local_mac: bytes | None = None  # this host's end of the link
    peer_ip: bytes | None = None
    local_ip: bytes | None = None
    peer_ports: frozenset[int] = frozenset()  # TCP ports the tapped node may serve
    allow_out: frozenset[str] = frozenset()  # classes the tapped node may emit
    allow_in: frozenset[str] = frozenset()  # classes it may be sent
    declared_pins: dict[str, str] = field(default_factory=dict)
    # `mac/class` → seconds: flows declared to appear on a fixed cadence.
    declared_beats: dict[str, float] = field(default_factory=dict)
    beat_tolerance: float = 5.0
    # Destination MACs permitted beyond the two ends and broadcast. Empty is
    # right for a link nothing else speaks on; a deployment where some group
    # address is legitimately in use names it rather than needing the rule
    # changed.
    extra_destinations: frozenset[bytes] = frozenset()
    # Whether the broadcast address is a permitted destination. False suits a
    # segment whose two addresses are both configured; a shared one needs it.
    allow_broadcast: bool = False

    @classmethod
    def from_config(cls, iface: str | None = None) -> LinkPolicy | None:
        """Build from the FRAME_PROCESSOR_* environment, or None when it is off.

        With an interface that captures on this host's own end of the link,
        the local MAC is read from sysfs when not configured: it is this
        host's own address, so there is nothing to trust it about, and asking
        the operator to copy it is a chance to copy it wrong. A monitor port
        (one with a declared direction) is skipped: its address is not this
        host's address on the tapped link, and using it would make every
        frame this host sent look like a stranger's.
        """
        from app import config

        peer_mac = _mac(config.FRAME_PROCESSOR_PEER_MAC)
        if peer_mac is None:
            return None
        local_mac = _mac(config.FRAME_PROCESSOR_LOCAL_MAC)
        if local_mac is None and iface and iface not in config.FRAME_PROCESSOR_IFACE_DIRECTIONS:
            local_mac = _sysfs_mac(iface)
        if local_mac is None and config.FRAME_PROCESSOR_IFACE_DIRECTIONS:
            log.warning(
                "FRAME_PROCESSOR_LOCAL_MAC is unset and capture is on monitor ports; "
                "frames this host sends will not be checked against a MAC"
            )
        return cls(
            peer_mac=peer_mac,
            local_mac=local_mac,
            peer_ip=_ip(config.FRAME_PROCESSOR_PEER_IP),
            local_ip=_ip(config.FRAME_PROCESSOR_LOCAL_IP),
            peer_ports=_ports(config.FRAME_PROCESSOR_PEER_PORTS),
            allow_out=_csv(config.FRAME_PROCESSOR_ALLOW_OUT),
            allow_in=_csv(config.FRAME_PROCESSOR_ALLOW_IN),
            declared_pins=parse_declared_pins(config.FRAME_PROCESSOR_EXPECTED_PINS),
            declared_beats=parse_declared_beats(config.FRAME_PROCESSOR_EXPECTED_BEATS),
            beat_tolerance=max(0.0, config.FRAME_PROCESSOR_BEAT_TOLERANCE),
            allow_broadcast=config.FRAME_PROCESSOR_ALLOW_BROADCAST,
            extra_destinations=frozenset(
                m for m in (_mac(v) for v in config.FRAME_PROCESSOR_EXTRA_DESTINATIONS)
                if m is not None
            ),
        )

    def describe(self) -> str:
        return (
            f"peer={_render_mac(self.peer_mac)}"
            f"{'/' + _render_ip(self.peer_ip) if self.peer_ip else ''}"
            f" local={_render_mac(self.local_mac) if self.local_mac else '?'}"
            f"{'/' + _render_ip(self.local_ip) if self.local_ip else ''}"
            f" peer_ports={sorted(self.peer_ports) or 'any'}"
            f" allow_out={sorted(self.allow_out) or 'nothing'}"
            f" allow_in={sorted(self.allow_in) or 'nothing'}"
            f" declared_pins={len(self.declared_pins)}"
            f"{' extra_dst=' + ','.join(sorted(_render_mac(m) for m in self.extra_destinations)) if self.extra_destinations else ''}"
            f" beats={','.join(f'{k}@{v:g}s' for k, v in sorted(self.declared_beats.items())) or 'none'}"
        )

    # --- direction ---------------------------------------------------------

    def direction_from_source(self, data: bytes | memoryview) -> str:
        """Replay fallback: which end of the link wrote this source MAC."""
        if len(data) < 12:
            return DIRECTION_UNKNOWN
        src = bytes(data[6:12])
        if src == self.peer_mac:
            return DIRECTION_OUT
        if self.local_mac is not None and src == self.local_mac:
            return DIRECTION_IN
        return DIRECTION_UNKNOWN

    # --- pins -------------------------------------------------------------

    def declared_digest(self, key: str) -> str | None:
        return self.declared_pins.get(key)

    # --- beats ------------------------------------------------------------

    def declared_interval(self, key: str) -> float | None:
        """Seconds this `mac/class` is declared to appear on, or None."""
        return self.declared_beats.get(key)

    def direction_for_mac(self, rendered: str) -> str:
        """Which way a frame from this address travels, from its MAC alone.

        Unlike `direction_from_source` this is about a DECLARATION rather than
        an observed frame, so reading the address is sound: nobody is claiming
        anything, the operator wrote it down. Used to decide which monitor
        port can see a declared beat at all.
        """
        if rendered == _render_mac(self.peer_mac):
            return DIRECTION_OUT
        if self.local_mac is not None and rendered == _render_mac(self.local_mac):
            return DIRECTION_IN
        return DIRECTION_UNKNOWN

    # --- the rules ----------------------------------------------------------

    def check(
        self,
        data: bytes | memoryview,
        name: str,
        pin_at: int,
        direction: str,
    ) -> list[tuple[str, str]]:
        """Every way this frame departs from the whitelist, as (kind, detail).

        ``name`` and ``pin_at`` are the accountant's classification; for
        ``ipv4-tcp`` the pin offset is the start of the TCP header, which is
        where the ports are.
        """
        findings: list[tuple[str, str]] = []
        if len(data) < _ETH_HEADER_LEN:
            return findings
        src = bytes(data[6:12])

        # Who is speaking, and from which side. The kernel said which side;
        # the MAC had better agree.
        if direction == DIRECTION_UNKNOWN:
            findings.append(
                ("mac-unknown", f"source {_render_mac(src)} is neither end of the link")
            )
        else:
            expected = self.peer_mac if direction == DIRECTION_OUT else self.local_mac
            if expected is not None and src != expected:
                findings.append(
                    (
                        "mac-unknown",
                        f"{direction} frame from {_render_mac(src)}, "
                        f"expected {_render_mac(expected)}",
                    )
                )

        # Is this kind of frame allowed to travel this way at all.
        if name not in _ALREADY_FLAGGED and not name.endswith("-other"):
            if direction == DIRECTION_UNKNOWN:
                allowed = self.allow_out | self.allow_in
            elif direction == DIRECTION_OUT:
                allowed = self.allow_out
            else:
                allowed = self.allow_in
            if name not in allowed:
                findings.append(("class-not-allowed", f"{name} is not allowed {direction}"))

        # And who it is addressed to. Only the SOURCE address was ever
        # checked, which left six bytes per frame that nothing looked at.
        #
        # On a segment with two NICs reaching each other by a static route,
        # every frame has exactly one address it can be going to. Broadcast is
        # not an exception to that so much as a less specific name for the
        # same host, which is why it is refused unless declared: the only
        # thing that would use it is a resolution for an address already
        # configured at both ends. A group address is likewise permitted only
        # by name (`extra_destinations`), because on a two-node segment
        # nothing has a group to speak to.
        #
        # A wrong destination is not harmless because the peer's NIC would
        # drop it: the frame still crossed the link, the tap still saw it, and
        # so did anything else on the segment.
        dst = bytes(data[:6])
        wanted = self.local_mac if direction == DIRECTION_OUT else self.peer_mac
        if direction != DIRECTION_UNKNOWN and wanted is not None:
            permitted = {wanted, *self.extra_destinations}
            if self.allow_broadcast:
                permitted.add(_BROADCAST_MAC)
            if dst not in permitted:
                findings.append(
                    (
                        "mac-destination",
                        f"{direction} frame to {_render_mac(dst)}, expected "
                        + " or ".join(sorted(_render_mac(m) for m in permitted)),
                    )
                )

        eth_at = _eth_at(data)
        if len(data) < eth_at:
            return findings
        ethertype = struct.unpack_from("!H", data, eth_at - _ETHERTYPE_LEN)[0]

        if ethertype == _ETHERTYPE_IPV4:
            findings.extend(self._check_ipv4(data, eth_at, name, pin_at, direction))
        elif ethertype == _ETHERTYPE_ARP and name == CLASS_ARP:
            findings.extend(self._check_arp(data, pin_at, direction))
        return findings

    def _endpoints(self, direction: str) -> tuple[bytes | None, bytes | None]:
        """(expected source IP, expected destination IP) for a direction."""
        if direction == DIRECTION_OUT:
            return self.peer_ip, self.local_ip
        if direction == DIRECTION_IN:
            return self.local_ip, self.peer_ip
        return None, None

    def _check_ipv4(
        self,
        data: bytes | memoryview,
        eth_at: int,
        name: str,
        pin_at: int,
        direction: str,
    ) -> list[tuple[str, str]]:
        findings: list[tuple[str, str]] = []
        if len(data) < eth_at + _IPV4_DST_AT + 4:
            return findings
        src_ip = bytes(data[eth_at + _IPV4_SRC_AT : eth_at + _IPV4_SRC_AT + 4])
        dst_ip = bytes(data[eth_at + _IPV4_DST_AT : eth_at + _IPV4_DST_AT + 4])

        want_src, want_dst = self._endpoints(direction)
        if want_src is not None and want_dst is not None:
            if src_ip != want_src or dst_ip != want_dst:
                findings.append(
                    (
                        "ip-outside-link",
                        f"{_render_ip(src_ip)} -> {_render_ip(dst_ip)} {direction}, "
                        f"expected {_render_ip(want_src)} -> {_render_ip(want_dst)}",
                    )
                )

        if name == CLASS_TCP and self.peer_ports and len(data) >= pin_at + 4:
            sport, dport = struct.unpack_from("!HH", data, pin_at)
            # The tapped node serves; so its port is the source going out and
            # the destination coming in. Anything else is a connection to or
            # from a service nobody declared.
            peer_port = sport if direction == DIRECTION_OUT else dport
            if direction != DIRECTION_UNKNOWN and peer_port not in self.peer_ports:
                findings.append(
                    (
                        "tcp-port-unexpected",
                        f"{_render_ip(src_ip)}:{sport} -> {_render_ip(dst_ip)}:{dport} "
                        f"{direction}; tapped node may only use "
                        f"{sorted(self.peer_ports)}",
                    )
                )

        # A bare SYN says who OPENED the connection, which is the one thing a
        # port cannot: a node dialling out from its own service port satisfies
        # the rule above while still being the one that initiated. The tapped
        # node only ever serves, so a SYN it sent is a connection it started.
        if name == CLASS_TCP and direction == DIRECTION_OUT and len(data) >= pin_at + 14:
            flags = data[pin_at + _TCP_FLAGS_AT]
            if flags & _TH_SYN and not flags & _TH_ACK:
                sport, dport = struct.unpack_from("!HH", data, pin_at)
                findings.append(
                    (
                        "tcp-dial-out",
                        f"tapped node opened a connection to "
                        f"{_render_ip(dst_ip)}:{dport} from port {sport}; it "
                        f"should only ever accept them",
                    )
                )
        return findings

    def _check_arp(
        self, data: bytes | memoryview, at: int, direction: str
    ) -> list[tuple[str, str]]:
        """ARP may only ever be about — and addressed to — the two ends."""
        findings: list[tuple[str, str]] = []
        want_dst_mac = self.local_mac if direction == DIRECTION_OUT else self.peer_mac
        want_sender, want_target = self._endpoints(direction)
        if want_dst_mac is None:
            return findings

        # A reply always knows the host it is answering, so it is never
        # addressed to everybody even where broadcast is permitted. The
        # general rule above cannot say this: it does not read the opcode.
        if (
            self.allow_broadcast
            and len(data) >= at + _ARP_OPCODE_AT + 2
            and struct.unpack_from("!H", data, at + _ARP_OPCODE_AT)[0] == _ARP_REPLY
            and bytes(data[:6]) == _BROADCAST_MAC
        ):
            findings.append(
                ("mac-destination", "ARP reply addressed to broadcast; a reply "
                                    "knows the host it answers")
            )

        if want_sender is None or want_target is None:
            return findings
        if len(data) < at + _ARP_IPV4_LEN:
            return findings
        if data[at + _ARP_HLEN_AT] != 6 or data[at + _ARP_PLEN_AT] != 4:
            return findings  # not IPv4-over-Ethernet ARP; accounting's shape rules apply
        sender_ip = bytes(data[at + _ARP_SENDER_IP_AT : at + _ARP_SENDER_IP_AT + 4])
        target_ip = bytes(data[at + _ARP_TARGET_IP_AT : at + _ARP_TARGET_IP_AT + 4])
        if sender_ip != want_sender or target_ip != want_target:
            findings.append(
                (
                    "arp-address-unexpected",
                    f"sender {_render_ip(sender_ip)} target {_render_ip(target_ip)} "
                    f"{direction}, expected {_render_ip(want_sender)} -> "
                    f"{_render_ip(want_target)}",
                )
            )
        return findings


def _sysfs_mac(iface: str) -> bytes | None:
    try:
        with open(f"/sys/class/net/{iface}/address", encoding="ascii") as fh:
            return _mac(fh.read())
    except (OSError, ValueError):
        return None
