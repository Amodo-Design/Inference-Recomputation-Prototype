"""frame-processor configuration, read from the environment.

The frame tap is fully passive: it never sits in the serving path, it only
observes frames (live from an interface, or replayed from a pcap file) and
emits tap messages. Nothing here can affect a client request.
"""

from __future__ import annotations

import logging
import math
import mmap
import os
import socket

# TPACKET_V3 geometry rules the kernel enforces when a ring is mapped. The
# authority is RingCapture._validate_geometry; these two ABI constants are
# repeated here rather than imported because capture.py imports this module.
#
# They matter at *this* layer because the values have to be valid before a
# child process maps its ring. A clamp that only bounds the magnitude lets a
# plausible-looking typo through, so the parent logs a healthy startup line and
# the children then die with two spawn tracebacks and capture nothing.
_TPACKET_MIN_FRAME_SIZE = 68  # sizeof(struct tpacket3_hdr)
_TPACKET_ALIGNMENT = 16


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _align_up(value: int, multiple: int) -> int:
    remainder = value % multiple
    return value if not remainder else value + (multiple - remainder)


def _as_ring_frame_size(value: str | None, default: int) -> int:
    """TPACKET frame geometry: at least one header, and 16-byte aligned."""
    return _align_up(
        max(_TPACKET_MIN_FRAME_SIZE, _as_int(value, default)), _TPACKET_ALIGNMENT
    )


def _as_ring_block_size(value: str | None, default: int, frame_size: int) -> int:
    """Block geometry: a positive multiple of BOTH the page size and frame_size.

    Rounded up to their least common multiple, which satisfies each rule at
    once — page alignment alone is not enough when frame_size is not itself a
    power-of-two divisor of a page.
    """
    step = math.lcm(mmap.PAGESIZE, frame_size)
    return _align_up(max(step, _as_int(value, default)), step)


def _parse_ports(value: str | None) -> frozenset[int] | None:
    """Comma-separated TCP ports; None (match everything) when unset/blank."""
    if not value or not value.strip():
        return None
    ports: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            try:
                ports.add(int(part))
            except ValueError:
                continue
    return frozenset(ports) or None


def _parse_ifaces(value: str | None) -> tuple[str, ...]:
    """Comma-separated capture interfaces; empty means auto-detect one.

    Deduplicated, because binding two sockets to one interface would deliver
    every frame twice and file every inference twice with it.
    """
    if not value or not value.strip():
        return ()
    return tuple(dict.fromkeys(p.strip() for p in value.split(",") if p.strip()))


def _parse_iface_directions(value: str | None) -> dict[str, str]:
    """`iface=in|out` pairs, comma-separated; which way each monitor port faces.

    e.g. FRAME_PROCESSOR_IFACE_DIRECTION="<monitor-iface-a>=in,<monitor-iface-b>=out". Values are
    named from the tapped node's side: ``out`` is what it emits, ``in`` what
    it is sent. Anything else is dropped with a warning rather than guessed,
    because a wrong direction silently turns every frame into a finding.
    """
    directions: dict[str, str] = {}
    if not value:
        return directions
    for pair in value.split(","):
        iface, sep, direction = pair.strip().partition("=")
        iface, direction = iface.strip(), direction.strip().lower()
        if not sep or not iface:
            continue
        if direction not in ("in", "out"):
            logging.getLogger("frameprocessor.config").warning(
                "ignoring FRAME_PROCESSOR_IFACE_DIRECTION %r: want <iface>=in or <iface>=out", pair
            )
            continue
        directions[iface] = direction
    return directions


def _parse_tunnel_ports(value: str | None, default: frozenset[int]) -> frozenset[int]:
    """Comma-separated UDP ports; explicit blank disables decapsulation."""
    if value is None:
        return default
    return _parse_ports(value) or frozenset()


def _parse_classes(value: str | None) -> frozenset[str]:
    """Comma-separated accounting class names, lower-cased; blank = empty."""
    return frozenset(p.strip().lower() for p in (value or "").split(",") if p.strip())


def _parse_prefixes(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated lower-cased prefixes; explicit blank disables the list."""
    if value is None:
        return default
    return tuple(p.strip().lower() for p in value.split(",") if p.strip())


def _parse_hostname_map(value: str | None) -> dict[str, str]:
    """`model=hostname` pairs, comma-separated.

    e.g. FRAME_PROCESSOR_HOSTNAME_MAP="openai/gpt-oss-120b=kserve-gpt-oss-120b"
    """
    mapping: dict[str, str] = {}
    if not value:
        return mapping
    for pair in value.split(","):
        model, sep, hostname = pair.strip().partition("=")
        if sep and model and hostname:
            mapping[model] = hostname
    return mapping


# --- Frame source ---
FRAME_PROCESSOR_MODE = os.getenv("FRAME_PROCESSOR_MODE", "pcap").lower()  # pcap | live
FRAME_PROCESSOR_PCAP_PATH = os.getenv("FRAME_PROCESSOR_PCAP_PATH") or None
# Live capture interfaces, comma-separated; empty = auto-detect the
# default-route interface.
#
# A hardware tap on a full-duplex link cannot do better than one monitor port
# per direction — there is no point on the link where both directions exist as
# one signal — so tapping it means capturing two interfaces at once. Both feed
# one connection table, because an exchange is only complete when its request
# and its response have both been seen, and those arrive on different ports.
# Which port carries which direction is a property of how the cables were run
# and nothing here depends on knowing.
FRAME_PROCESSOR_IFACES = _parse_ifaces(os.getenv("FRAME_PROCESSOR_IFACE"))

# Live capture reads Linux's TPACKET_V3 block ring: one wake-up is amortised
# across a whole mmap'd block, and definitely-irrelevant traffic dies in place
# without allocating a Python bytes/Frame object per packet.
#
# Frame size is resolved first: the block size has to be a multiple of it, so
# snapping the block against a stale frame size would reintroduce the mismatch.
FRAME_PROCESSOR_RING_FRAME_SIZE = _as_ring_frame_size(
    os.getenv("FRAME_PROCESSOR_RING_FRAME_SIZE"), 2048
)
FRAME_PROCESSOR_RING_BLOCK_SIZE = _as_ring_block_size(
    os.getenv("FRAME_PROCESSOR_RING_BLOCK_SIZE"), 1024 * 1024, FRAME_PROCESSOR_RING_FRAME_SIZE
)
FRAME_PROCESSOR_RING_BLOCK_COUNT = max(
    2, _as_int(os.getenv("FRAME_PROCESSOR_RING_BLOCK_COUNT"), 128)
)
FRAME_PROCESSOR_RING_RETIRE_TIMEOUT_MS = max(
    1, _as_int(os.getenv("FRAME_PROCESSOR_RING_RETIRE_TIMEOUT_MS"), 10)
)

# Live traffic is deliberately a process topology, not a thread topology.
# CPython's GIL serialises header parsing, reassembly, h11 and JSON work; two
# receiver processes drain the two physical directions independently and the
# inner connection key assigns both directions to one stateful flow worker.
FRAME_PROCESSOR_FLOW_WORKERS = max(1, _as_int(os.getenv("FRAME_PROCESSOR_FLOW_WORKERS"), 4))
FRAME_PROCESSOR_SEGMENT_QUEUE_SIZE = max(
    1, _as_int(os.getenv("FRAME_PROCESSOR_SEGMENT_QUEUE_SIZE"), 4096)
)
FRAME_PROCESSOR_MESSAGE_QUEUE_SIZE = max(
    1, _as_int(os.getenv("FRAME_PROCESSOR_MESSAGE_QUEUE_SIZE"), 256)
)

# Reject frames that are provably unable to yield a TCP segment with a small,
# bounds-checked header walk before asking dpkt to construct the complete
# outer and inner protocol object graphs.  This is userspace classification,
# not a socket filter: every mirrored frame still reaches frame-processor.  The
# classifier fails open on layouts it cannot prove, preserving coverage.
FRAME_PROCESSOR_FAST_CLASSIFY = _as_bool(os.getenv("FRAME_PROCESSOR_FAST_CLASSIFY"), True)

# Account for every frame, not just the TCP the pipeline reconstructs.
#
# On by default. The tapped-link filtering scheme needs a statement about
# *all* traffic on the link — a prover that can hide bytes in a class nobody
# counted is not constrained by a tap — and the TCP decoder discards most
# frames before anything downstream sees them. Classification is cheap
# relative to a link's frame rate, so the flag exists to turn accounting off
# while diagnosing it, not because it is expected to be too expensive.
FRAME_PROCESSOR_ACCOUNT_ALL = _as_bool(os.getenv("FRAME_PROCESSOR_ACCOUNT_ALL"), True)

# --- Link whitelist (app/policy.py) ---
# What the tapped link is allowed to carry, per direction. Off unless the
# peer MAC is set, because every rule below is relative to that one node:
# "out" is what it emits, "in" is what it is sent, and anything on the link
# that is not one of the allowed classes going the allowed way is a finding.
#
# Where direction comes from depends on where the capture sits:
#
# - On a hardware tap, each monitor port carries one direction, and which
#   port carries which is a property of how the fibres were run. It is
#   declared here, per interface, and nothing in the frame is consulted. A
#   monitor port never transmits, so the kernel's packet type would call
#   every frame on it "received" and label both directions ``out``.
# - On this host's own end of a point-to-point link (no tap), the kernel's
#   packet type is
#   authoritative: PACKET_OUTGOING means this host sent it, anything else
#   arrived from the wire — and the peer cannot forge that bit by writing a
#   different source MAC. Interfaces not listed below use it.
# - A pcap replay has neither and falls back to the source MAC.
#
# A minimal configuration names the two ends, the classes each direction may
# carry, and the cadence of any flow declared constant. The link a deployment
# actually runs belongs in its own notes, not here.
#   FRAME_PROCESSOR_IFACE_DIRECTION=<iface>=in,<iface>=out
#   FRAME_PROCESSOR_PEER_MAC=<tapped node>   FRAME_PROCESSOR_LOCAL_MAC=<capture host>
#   FRAME_PROCESSOR_PEER_IP=<tapped node>    FRAME_PROCESSOR_LOCAL_IP=<capture host>
#   FRAME_PROCESSOR_ALLOW_OUT=arp,ipv4-tcp   FRAME_PROCESSOR_ALLOW_IN=arp,ipv4-tcp
# Whole-frame pin digests can only come from a capture, because they cover
# header fields the sender does not choose; tools/account.py reports them.
# On a tap the local MAC must be given: the monitor port's own address is not
# this host's address on the tapped link. On a host-end capture it is read
# from sysfs for the capture interface when unset.
FRAME_PROCESSOR_IFACE_DIRECTIONS = _parse_iface_directions(os.getenv("FRAME_PROCESSOR_IFACE_DIRECTION"))
FRAME_PROCESSOR_PEER_MAC = os.getenv("FRAME_PROCESSOR_PEER_MAC") or None
FRAME_PROCESSOR_LOCAL_MAC = os.getenv("FRAME_PROCESSOR_LOCAL_MAC") or None
FRAME_PROCESSOR_PEER_IP = os.getenv("FRAME_PROCESSOR_PEER_IP") or None
FRAME_PROCESSOR_LOCAL_IP = os.getenv("FRAME_PROCESSOR_LOCAL_IP") or None
# TCP ports the tapped node may serve: its source port going out, the
# destination port coming in. Empty = any port.
FRAME_PROCESSOR_PEER_PORTS = os.getenv("FRAME_PROCESSOR_PEER_PORTS") or None
# Frame classes (accounting.py names) allowed in each direction. Empty means
# nothing is allowed, which is the honest default once the policy is on: a
# link nobody has described has no traffic that is known to belong there.
FRAME_PROCESSOR_ALLOW_OUT = os.getenv("FRAME_PROCESSOR_ALLOW_OUT") or None
FRAME_PROCESSOR_ALLOW_IN = os.getenv("FRAME_PROCESSOR_ALLOW_IN") or None
# `mac/class=digest` pairs: the payload a pinned flow must carry, declared up
# front instead of learned from whatever frame arrives first. The digest is
# the one `python3 tools/account.py` prints in its pins table.
FRAME_PROCESSOR_EXPECTED_PINS = os.getenv("FRAME_PROCESSOR_EXPECTED_PINS") or None

# Classes to pin that accounting.py leaves unpinned by default. Its
# UNPINNED_CLASSES list is written for a general segment; a point-to-point
# link can be stricter. ARP is the case this exists for: the default reason
# for leaving it unpinned is that "a router resolving several neighbours
# emits a different target each time", which is untrue on a /30 with exactly
# two hosts — there an ARP request is byte-identical every time it is sent.
#
# The health check does not need this. An unnamed UDP port is pinned already; this
# is a separate tightening of the traffic the data path emits anyway.

# Classes pinned from byte zero rather than from the end of the headers.
#
# The default skips the headers because the IPv4 identification field and a
# rotating UDP source port would make ordinary constant traffic look variable.
# That reasoning does not apply to a flow we emit on purpose: on the 66-byte
# link beacon, pinning only the payload leaves DSCP, total length,
# identification, flags, TTL, both checksums, the source port and the UDP
# length — sixteen bytes — checked by nothing, twice every thirty seconds.
#
# Only sound where the sender makes every header field deterministic, which is
# what tools/tapped_link_health.py sets its socket options for.
#
# Two IPv4 fields are excluded, because the sender does not choose them: the
# identification, which the kernel assigns, and the header checksum, which
# follows from the rest of the header and is verified arithmetically instead.
# Identification is therefore the one field a whole-frame pin leaves free —
# on the tapped link it was measured moving on every datagram despite DF being
# set, which is exactly why the digest must come from a real capture
# (`python3 tools/account.py`) and never from a guess.
FRAME_PROCESSOR_PIN_WHOLE_FRAME = _parse_classes(os.getenv("FRAME_PROCESSOR_PIN_WHOLE_FRAME"))

# --- what the structural rules compare against (app/accounting.py) ---
# The rules below hold a frame to the one value each field can legitimately
# have. Most of those values are properties of the PROTOCOL — an Ethernet ARP
# has a hardware address length of 6, broadcast is ff:ff:ff:ff:ff:ff, the TCP
# urgent pointer is ignored when URG is clear — and are not configurable,
# because they are not opinions about this deployment.
#
# These two are properties of a LINK, so they are declared rather than
# assumed. Both default to the stricter reading, which is right for a
# point-to-point segment with one MTU and no router in the middle.

# IPv4 options. Refused by default: the header length is otherwise read only
# to find the transport, so any value from 5 to 15 words admits up to forty
# bytes nothing examines. Set true on a link where they are legitimate — IGMP
# carries the Router Alert option, for instance.
FRAME_PROCESSOR_ALLOW_IP_OPTIONS = _as_bool(os.getenv("FRAME_PROCESSOR_ALLOW_IP_OPTIONS"), False)

# IPv4 fragmentation. Refused by default: a link with one MTU and no hop
# between its ends never fragments, so More Fragments or a non-zero offset is
# evasion or a fault — and a reassembly frame-processor does not perform is one
# nobody is checking. Set true on a segment where the MTUs differ.
FRAME_PROCESSOR_ALLOW_FRAGMENTS = _as_bool(os.getenv("FRAME_PROCESSOR_ALLOW_FRAGMENTS"), False)

# Additional destination MACs a frame may carry, beyond the far end of the
# link. Empty is right for a segment with two NICs and nothing else on it; a
# deployment where a group address is legitimately in use — LLDP's reserved
# multicast, say — names it here rather than needing the rule changed.
FRAME_PROCESSOR_EXTRA_DESTINATIONS = _parse_classes(os.getenv("FRAME_PROCESSOR_EXTRA_DESTINATIONS"))

# Whether a frame may be addressed to the broadcast address.
#
# Refused by default, which is a statement about this link rather than about
# Ethernet: there are two NICs on the segment and each reaches the other by a
# static route, so a frame addressed to everybody is addressed to one host by
# a less specific name. ARP is the only thing that would do it, and it only
# does so when resolving an address it does not yet know — which on a link
# whose two addresses are configured means either a cold cache or something
# that should be looked at.
#
# Set true on a shared segment, where resolution genuinely has to ask. Note
# that permanent neighbour entries at both ends remove the case entirely, and
# with them the link carries no ARP at all.
FRAME_PROCESSOR_ALLOW_BROADCAST = _as_bool(os.getenv("FRAME_PROCESSOR_ALLOW_BROADCAST"), False)

FRAME_PROCESSOR_PIN_CLASSES = _parse_classes(os.getenv("FRAME_PROCESSOR_PIN_CLASSES"))

# --- Beats (app/accounting.py) ---
# `mac/class=seconds` pairs: a flow declared to appear on a fixed cadence.
# Missing beats are a finding, and so are extra ones.
#
# This is what makes a silent link readable. Everything else here checks
# frames that arrived; nothing checks that any arrived at all, and on a link
# whose only other traffic is inference, "no frames" is exactly as consistent
# with "nobody asked for an inference" as with "the tap has gone blind". A
# declared beat removes the ambiguity: the link is asserted to carry
# something every N seconds, so absence becomes evidence rather than silence.
#
# The datagram is not emitted from here — the tap is passive. It is a fixed
# UDP datagram the tapped node sends on a timer and this host answers
# (tools/tapped_link_health.py); each leg is one declaration, and because the request
# and the reply land on different monitor ports, a missing leg says which port
# went dark. Declare the class as `udp-<port>`, which is what the accountant
# names it, and give both ends the same port — see the tool's docstring.
FRAME_PROCESSOR_EXPECTED_BEATS = os.getenv("FRAME_PROCESSOR_EXPECTED_BEATS") or None
# How far from its declared cadence a beat may land before it is judged. It
# bounds both directions: outside this a beat is late (and the ones it
# skipped are missed) or early (and unscheduled). Set it from the jitter the
# beacon actually has — a systemd timer needs AccuracySec set or it will
# drift by up to a minute on its own.
FRAME_PROCESSOR_BEAT_TOLERANCE = _as_float(os.getenv("FRAME_PROCESSOR_BEAT_TOLERANCE"), 5.0)

# Seconds between kernel drop-counter polls. Capture loss corrupts inferences
# without failing anything, so it is reported on a timer rather than waited for.
FRAME_PROCESSOR_STATS_INTERVAL = _as_int(os.getenv("FRAME_PROCESSOR_STATS_INTERVAL"), 30)
# Coarse pre-filter: only reassemble segments touching these TCP ports. The
# authoritative LLM filter is content-based (filter.py); this just keeps
# obvious noise out of the reassembler. Unset = reassemble everything.
FRAME_PROCESSOR_PORTS = _parse_ports(os.getenv("FRAME_PROCESSOR_PORTS"))

# Ports never reassembled, whatever the allowlist above says — exclusion wins.
#
# Both settings keep noise out of the reassembler, but they fail in opposite
# directions, and only one of them fails safely. An allowlist that goes stale
# is silent in the direction that matters: inference served on a port nobody
# added is dropped before reassembly, and the tap runs on indefinitely with
# nothing to show for it. A denylist cannot lose an inference it was not
# explicitly told to discard — the worst it does is fail to exclude noise,
# which costs throughput rather than coverage. Prefer this one unless the
# traffic mix on the link is genuinely known and stable.
#
# The case it exists for: a bulk, non-HTTP service sharing the tapped link,
# such as a database. Every frame of it would be decoded, reassembled, and
# handed to an HTTP parser that was never going to make sense of it — easily
# outweighing the inference traffic itself. Excluding that port removes the
# load without ever putting an inference at risk. Placing such a service off
# the tapped link is the better fix where it is possible.
FRAME_PROCESSOR_EXCLUDE_PORTS = _parse_ports(os.getenv("FRAME_PROCESSOR_EXCLUDE_PORTS")) or frozenset()

# --- Overlay decapsulation ---
# Pod traffic that crosses a node boundary is encapsulated by the CNI, so a
# tap on a physical link sees UDP between node addresses with the inference
# request sealed inside — not the TCP the pipeline needs. Decoding sees
# through one such layer, keyed on the tunnel's destination port (VXLAN's
# source port is a hash of the inner packet, so only the destination is
# stable; both directions send *to* this port).
#
# Defaults cover flannel (8472, the port Linux shipped before the standard
# settled) and the IANA assignment (4789) that Cilium and others use. Geneve
# is a different header, not another port: 6081 belongs here only once
# _vxlan_payload knows the difference.
#
# Blank disables decapsulation, for a tap that already sees bare TCP.
FRAME_PROCESSOR_DECAP_PORTS = _parse_tunnel_ports(
    os.getenv("FRAME_PROCESSOR_DECAP_PORTS"), frozenset({4789, 8472})
)

# --- Verifier traffic ---
# The point of the tap is to find inference traffic to SEND to the verifier,
# so the verifier's own replays are not inference this tap should record:
# tapping them would file replays as fresh events against the production
# deployment, and every event the tap files is another event needing
# verification. The verifier pair is scheduled onto whichever node has the
# GPU — often this one — so the exclusion cannot be positional.
#
# Matched on the request's Host header, which the runner sets from
# RUNNER_VLLM_URL (verify-<model-slug>-kserve-workload-svc, see
# inf-ver-orchestrator/app/naming.py). Host is part of the request, so this
# stays content-based like the rest of filter.py and survives pod IP churn.
FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES = _parse_prefixes(
    os.getenv("FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES"), ("verify-",)
)

# --- Exchange whitelist ---
# Path suffixes that may cross the tapped link without being inference.
#
# The link whitelist in policy.py can only say "TCP between these two
# addresses, with the tapped node's end on 8000". It cannot say what is inside
# that stream, because a single frame does not tell you: the shape of an HTTP
# exchange only exists after reassembly. So without this list, a prover could
# open a connection to its own port 8000, send anything at all, and satisfy
# every frame-level rule while nothing reported it.
#
# The rule this enables is the strong one: EVERY exchange reassembled on the
# link must be either inference (filter.classify), the verifier's own replay
# (FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES above), or a path named here. Anything else
# is a finding. That inverts the previous default, where an exchange that was
# not inference was silently discarded.
#
# Keep it short and keep it justified — every entry is a path a prover may use
# with contents nobody validates. These four are what vLLM's own liveness and
# discovery need: Open WebUI polls /v1/models to populate its model list, and
# the container healthcheck hits /health.
FRAME_PROCESSOR_ALLOWED_ENDPOINTS = _parse_prefixes(
    os.getenv("FRAME_PROCESSOR_ALLOWED_ENDPOINTS"),
    ("/health", "/v1/models", "/metrics", "/ping"),
)

# --- Reassembly ---
# A connection with no traffic for this long is flushed and processed with
# whatever was captured (streams have no reliable FIN guarantee on a tap).
#
# This is NOT the emission path: exchanges are assembled incrementally and
# tapped the moment their response ends, so nothing waits on this timer.
# Its only remaining jobs are salvaging an exchange the capture cut in half
# and releasing state for connections that vanished without a close.
#
# So longer is safer. Flushing early truncates an inference that was merely
# slow — a long prefill, a queued request — and pops the connection, leaving
# the bytes that follow to start a stream h11 can never resynchronise. There
# is no latency to win back by shortening it.
FRAME_PROCESSOR_IDLE_TIMEOUT = _as_int(os.getenv("FRAME_PROCESSOR_IDLE_TIMEOUT"), 120)

# The old live loop scanned every active connection after every accepted TCP
# segment.  At high packet rates that is O(packets * connections), even though
# a 120-second expiry does not need packet-granular checks.  Sweep on capture
# time at this cadence instead.
FRAME_PROCESSOR_IDLE_SWEEP_INTERVAL = max(
    1, _as_int(os.getenv("FRAME_PROCESSOR_IDLE_SWEEP_INTERVAL"), 1)
)

# Bytes one direction may hold behind a missing segment before the tap gives
# up on it.
#
# A capture drop leaves a hole no retransmission can fill: the sender and
# receiver are having a healthy conversation and never learn an observer
# missed a packet, so every later byte on that direction strands behind the
# hole forever. Unbounded, one lost frame costs the whole connection's memory
# for the life of the process, which over a long run is unbounded growth.
#
# Generous on purpose: legitimate reordering within one direction spans a
# handful of segments on a LAN, and receive offload coalesces up to 64KB into
# a single frame, so 2MiB is dozens of frames' headroom. Set it too low and a
# merely-reordered stream is abandoned mid-inference.
FRAME_PROCESSOR_MAX_STRANDED_BYTES = _as_int(
    os.getenv("FRAME_PROCESSOR_MAX_STRANDED_BYTES"), 2 * 1024 * 1024
)

# --- Emit (the tap destination) ---
# Off by default: reconstructed tap messages are printed to stdout, which is
# what development/replay wants. Turn on to POST to the Message writer.
FRAME_PROCESSOR_EMIT_ENABLED = _as_bool(os.getenv("FRAME_PROCESSOR_EMIT_ENABLED"), False)
FRAME_PROCESSOR_MESSAGE_WRITER_URL = os.getenv(
    "FRAME_PROCESSOR_MESSAGE_WRITER_URL", "http://message-writer:8100"
).rstrip("/")
FRAME_PROCESSOR_MESSAGE_WRITER_PATH = os.getenv("FRAME_PROCESSOR_MESSAGE_WRITER_PATH", "/inferences")
FRAME_PROCESSOR_MESSAGE_WRITER_TIMEOUT = _as_int(os.getenv("FRAME_PROCESSOR_MESSAGE_WRITER_TIMEOUT"), 30)

# Completed inference messages leave the capture/parser loop through a
# bounded queue.  Worker threads are appropriate here: requests is network
# I/O, so it releases the GIL while the CPU-heavy capture path keeps running.
# Each worker owns a persistent Session and therefore reuses HTTP connections.
FRAME_PROCESSOR_EMIT_WORKERS = max(1, _as_int(os.getenv("FRAME_PROCESSOR_EMIT_WORKERS"), 8))
FRAME_PROCESSOR_EMIT_QUEUE_SIZE = max(
    1, _as_int(os.getenv("FRAME_PROCESSOR_EMIT_QUEUE_SIZE"), 256)
)
# Seconds close() will spend draining the queue before giving up on it.
#
# Bounded because the worst case otherwise exceeds any pod grace period and
# loses the backlog anyway: a full queue of 256 against a 30s writer timeout
# across 8 workers is ~16 minutes, and SIGKILL arrives long before that. The
# default leaves room for LivePipeline.stop() inside a 30s termination grace,
# and whatever is still queued is logged rather than waited on.
FRAME_PROCESSOR_EMIT_DRAIN_SECONDS = max(
    0, _as_int(os.getenv("FRAME_PROCESSOR_EMIT_DRAIN_SECONDS"), 5)
)

# --- Ledger resolution ---
# tap.hostname is the Message writer's resolution key and must match what the
# model deployment declared. The wire only carries the model name, so the
# mapping is configuration. Unmapped models fall back to "kserve-<basename>".
FRAME_PROCESSOR_HOSTNAME_MAP = _parse_hostname_map(os.getenv("FRAME_PROCESSOR_HOSTNAME_MAP"))

# --- Capture windows → ledger (app/windows.py, app/ledger_reporter.py) ---
# The account of the link is cut into wall-clock-aligned windows and every
# closed window is posted to the ledger as one row, findings included. The
# row exists when nothing happened: a window with zero frames is still
# reported, so a window that is MISSING means the tap was blind for it —
# the one failure a findings-only record could never show.
#
# Off unless the ledger URL is set. Windows are still cut and logged without
# it, so the rest of the machinery is exercised either way.
FRAME_PROCESSOR_LEDGER_URL = (os.getenv("FRAME_PROCESSOR_LEDGER_URL") or "").rstrip("/") or None
FRAME_PROCESSOR_LEDGER_TIMEOUT = max(1, _as_int(os.getenv("FRAME_PROCESSOR_LEDGER_TIMEOUT"), 10))
# Where the capture ran. On a hostNetwork pod this is the node's name.
FRAME_PROCESSOR_CAPTURE_HOST = os.getenv("FRAME_PROCESSOR_CAPTURE_HOST") or socket.gethostname()
# The tapped node as the ledger knows it — the hostname its model deployment
# declared — so a window joins to inference events on hardware_id rather than
# on a string. Defaults to the sole FRAME_PROCESSOR_HOSTNAME_MAP target when there
# is exactly one, which on a single-model link is the same node.
FRAME_PROCESSOR_TAPPED_HOSTNAME = os.getenv("FRAME_PROCESSOR_TAPPED_HOSTNAME") or (
    next(iter(FRAME_PROCESSOR_HOSTNAME_MAP.values())) if len(FRAME_PROCESSOR_HOSTNAME_MAP) == 1 else None
)
# Window length. 0 turns windowing off entirely (cumulative account only).
FRAME_PROCESSOR_WINDOW_SECONDS = max(0, _as_int(os.getenv("FRAME_PROCESSOR_WINDOW_SECONDS"), 300))
# How long past a window's end a receiver waits before closing it, so a frame
# stamped just inside the boundary but walked just after still lands in it.
FRAME_PROCESSOR_WINDOW_GRACE_SECONDS = max(0, _as_int(os.getenv("FRAME_PROCESSOR_WINDOW_GRACE_SECONDS"), 5))
# How long past a window's end the parent waits before writing its row, so
# flow workers — which reach an exchange some time after the frames that
# carried it — can still attribute their findings to the right window.
FRAME_PROCESSOR_WINDOW_FLUSH_SECONDS = max(
    0, _as_int(os.getenv("FRAME_PROCESSOR_WINDOW_FLUSH_SECONDS"), 15)
)
# Bounds on what one window row can hold. Findings aggregate by (kind, class,
# direction, sender), so rows scale with VARIETY rather than volume — but an
# unknown UDP port mints a class per port, so variety itself has to be
# capped. Past the cap a window is still tainted and the excess is counted;
# only detail is lost. Samples keep the first bytes of the offending frame.
FRAME_PROCESSOR_WINDOW_MAX_GROUPS = max(1, _as_int(os.getenv("FRAME_PROCESSOR_WINDOW_MAX_GROUPS"), 64))
FRAME_PROCESSOR_SAMPLE_FRAME_BYTES = max(0, _as_int(os.getenv("FRAME_PROCESSOR_SAMPLE_FRAME_BYTES"), 2048))

# Stamped into every tap message as tap.proxy_version, so the ledger records
# what the tap could see when an event was filed. Bump it whenever a change
# alters what is captured or how an inference is rendered, because events
# stamped with different versions are not necessarily comparable.
FRAME_PROCESSOR_VERSION = "0.7.1"
