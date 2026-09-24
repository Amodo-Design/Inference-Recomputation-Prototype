"""Declared beats: the check a flow fails by not arriving.

Every other rule in `accounting.py` judges a frame that turned up. On a link
whose only other traffic is inference, that leaves the failure that matters
most unreadable: no frames is exactly as consistent with an idle
prover as with a capture that has gone blind. A beat is the declaration that
closes it — this flow appears every N seconds, so absence is evidence.

The properties under test: a kept cadence is silent, a broken one is a finding
on the window it was due in, silence is caught with no frame to notice it by,
an absence is counted once however it is noticed, and nothing invents a miss
across a boundary the capture was not running for.

The flow under test is the one `tools/tapped_link_health.py` sends — a fixed
UDP datagram each way on a fixed port. Its payloads are imported rather than
restated, so the wire contract and the declaration cannot drift apart.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

from app.accounting import (
    KIND_BEAT_MISSED,
    KIND_BEAT_UNSCHEDULED,
    UNPINNED_CLASSES,
    FrameAccountant,
    _ones_complement,
)
from app.policy import DIRECTION_IN, DIRECTION_OUT, LinkPolicy, payload_digest

PEER = bytes.fromhex("020000000001")  # the tapped node: sends the request
LOCAL = bytes.fromhex("020000000002")  # this host: sends the reply
PEER_MAC = "02:00:00:00:00:01"
LOCAL_MAC = "02:00:00:00:00:02"

W = 300.0
T = 1_700_000_100.0  # a window boundary
BEAT = 30.0


PORT = 9999
CLASS = f"udp-{PORT}"

# From the beacon itself, not copied: the payload it sends and the payload the
# accountant is told to expect are one fact, and a test that restated it could
# keep passing while the two drifted apart.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from tapped_link_health import REPLY, REQUEST  # noqa: E402

# 14 Ethernet + 20 IPv4 + 8 UDP, then the payload.
FRAME_BYTES = 42 + len(REQUEST)


def beacon(
    src: bytes = PEER,
    payload: bytes = REQUEST,
    *,
    dport: int = PORT,
    ttl: int = 64,
    ident: int = 0,
    tos: int = 0,
) -> bytes:
    """One beacon datagram, as it looks on the wire.

    Both ends bind the same port on purpose — see `test_a_reply_to_a_low_
    source_port_would_mint_a_class_per_beat`. The header fields the beacon
    sets explicitly are parameters here so a test can vary one and nothing
    else, which is how the two pin regions are told apart.
    """
    body = struct.pack("!HHHH", PORT, dport, 8 + len(payload), 0) + payload
    header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, tos, 20 + len(body), ident, 0x4000, ttl, 17, 0, bytes(4), bytes(4),
    )
    # A real checksum, because a whole-frame pin excludes the field and
    # verifies it arithmetically instead — a fake one would fire that check on
    # every test that pins a frame.
    header = header[:10] + struct.pack("!H", _ones_complement(header)) + header[12:]
    # Addressed to the other end, as a real datagram on a two-host link is.
    return _peer_of(src) + src + struct.pack("!H", 0x0800) + header + body


def _peer_of(src: bytes) -> bytes:
    """The address at the other end of a two-host link."""
    return LOCAL if src == PEER else PEER


def masked(frame: bytes) -> bytes:
    """The frame as a whole-frame pin holds it: the two IPv4 fields the sender
    does not choose zeroed out. This is what `tools/account.py` prints a digest of,
    and therefore what goes into FRAME_PROCESSOR_EXPECTED_PINS."""
    out = bytearray(frame)
    out[18:20] = b"\x00\x00"   # identification — the kernel's
    out[24:26] = b"\x00\x00"   # header checksum — implied by the rest
    return bytes(out)


def arp(src: bytes = PEER, *, opcode: int = 1) -> bytes:
    """An ARP frame — the data path's own traffic, not the beacon."""
    # A request is broadcast and does not know the address it is asking for;
    # a reply goes back to the other end and names it.
    dst = _peer_of(src)
    target_hw = bytes(6) if opcode == 1 else dst
    body = (
        struct.pack("!HHBBH", 1, 0x0800, 6, 4, opcode)
        + src + bytes(4)
        + target_hw + bytes(4)
    )
    frame = dst + src + struct.pack("!H", 0x0806) + body
    return frame + bytes(max(0, 60 - len(frame)))


def policy(
    beats: dict[str, float] | None = None,
    tolerance: float = 5.0,
    whole_frame: bool = False,
    extra_destinations: frozenset = frozenset(),
    allow_broadcast: bool = False,
) -> LinkPolicy:
    pins = (
        {
            f"{PEER_MAC}/{CLASS}": payload_digest(masked(beacon(PEER, REQUEST))),
            f"{LOCAL_MAC}/{CLASS}": payload_digest(masked(beacon(LOCAL, REPLY))),
        }
        if whole_frame
        else {
            f"{PEER_MAC}/{CLASS}": payload_digest(REQUEST),
            f"{LOCAL_MAC}/{CLASS}": payload_digest(REPLY),
        }
    )
    return LinkPolicy(
        peer_mac=PEER,
        local_mac=LOCAL,
        # As the link is actually declared: ARP, the inference TCP, and the
        # health check.
        allow_out=frozenset({"arp", "ipv4-tcp", CLASS}),
        allow_in=frozenset({"arp", "ipv4-tcp", CLASS}),
        declared_pins=pins,
        extra_destinations=extra_destinations,
        allow_broadcast=allow_broadcast,
        declared_beats={f"{PEER_MAC}/{CLASS}": BEAT} if beats is None else beats,
        beat_tolerance=tolerance,
    )


def accountant(**kw) -> FrameAccountant:
    kw.setdefault("policy", policy())
    kw.setdefault("iface", "tap0")
    kw.setdefault("window_seconds", W)
    kw.setdefault("process_epoch", T)
    return FrameAccountant(**kw)


def kinds(acct: FrameAccountant) -> set[str]:
    return {kind for kind, _class in acct.snapshot().finding_groups}


def beat_of(acct: FrameAccountant, mac: str = PEER_MAC):
    return acct.snapshot().beats[f"{mac}/{CLASS}"]


# --- a cadence that is kept ---------------------------------------------------


def test_a_beacon_on_its_declared_cadence_raises_nothing():
    acct = accountant()
    for n in range(10):
        acct.observe(beacon(), T + 5 + n * BEAT, DIRECTION_OUT)
    acct.roll(T + 5 + 9 * BEAT)
    assert kinds(acct) == set()
    assert beat_of(acct).beats == 10
    assert beat_of(acct).missed == 0


def beats_at(acct: FrameAccountant, gaps: list[float], start: float = T + 5) -> float:
    """Drive one beacon through ``gaps``, returning the last timestamp.

    Gaps rather than absolute times because that is what the check measures:
    a timer firing 30s after the last one drifts against the wall clock and
    is still keeping its cadence, which is the shape real beacons have.
    """
    ts = start
    acct.observe(beacon(), ts, DIRECTION_OUT)
    for gap in gaps:
        ts += gap
        acct.observe(beacon(), ts, DIRECTION_OUT)
    return ts


def test_jitter_inside_the_tolerance_is_on_time():
    acct = accountant()
    beats_at(acct, [30, 33.9, 25.5, 32, 34.9])
    assert kinds(acct) == set()
    assert beat_of(acct).beats == 6


def test_the_gaps_actually_seen_are_measured():
    """So a declaration can be checked against the cadence the beacon keeps,
    rather than the one whoever wrote the config believed it kept."""
    acct = accountant(policy=policy(tolerance=20.0))
    beats_at(acct, [12, 40, 30])
    beat = beat_of(acct)
    assert beat.beats == 4
    assert round(beat.min_gap, 1) == 12.0
    assert round(beat.max_gap, 1) == 40.0
    assert round(beat.mean_gap, 1) == 27.3
    assert kinds(acct) == set(), "all three gaps are inside a 20s tolerance"


def test_a_flow_nobody_declared_is_not_tracked():
    acct = accountant()
    acct.observe(arp(src=LOCAL, opcode=2), T + 5, DIRECTION_IN)
    assert f"{LOCAL_MAC}/arp" not in acct.snapshot().beats
    assert kinds(acct) == set()


# --- a cadence that is broken -------------------------------------------------


def test_a_skipped_beat_is_reported_when_the_beacon_comes_back():
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.observe(beacon(), T + 5 + 3 * BEAT, DIRECTION_OUT)  # two beats missing
    groups = acct.snapshot().finding_groups
    assert (KIND_BEAT_MISSED, CLASS) in groups
    assert beat_of(acct).missed == 2
    assert "2 beat(s) missed" in groups[(KIND_BEAT_MISSED, CLASS)].samples[0].detail


def test_total_silence_is_caught_on_the_clock_with_no_frame_to_notice_it_by():
    """The failure this exists for: nothing arrives, so nothing on the frame
    path can fire, and the log and the window would both read clean."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.roll(T + 5 + 4 * BEAT)  # wall clock moves on; the link does not
    assert (KIND_BEAT_MISSED, CLASS) in acct.snapshot().finding_groups
    assert beat_of(acct).missed == 3


def test_a_beacon_that_never_starts_is_reported_rather_than_waited_for():
    """A live capture anchors on process start, so a beacon already dead when
    the tap came up is a finding rather than an absence of evidence."""
    acct = accountant()
    acct.roll(T + BEAT + 6)
    assert (KIND_BEAT_MISSED, CLASS) in acct.snapshot().finding_groups
    assert "never seen" in [
        f.detail
        for f in acct.snapshot().finding_groups[(KIND_BEAT_MISSED, CLASS)].samples
    ][0]


def test_a_missed_beat_carries_the_direction_its_leg_was_going():
    """A two-port tap sees each leg on a different interface, so which leg
    stopped is which monitor port went dark."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.roll(T + 5 + 3 * BEAT)
    report = acct.take_closed() or acct.finish(T + 5 + 3 * BEAT)
    groups = [g for g in report[0].groups if g.kind == KIND_BEAT_MISSED]
    assert groups and groups[0].direction == DIRECTION_OUT


# --- one link, two monitor ports -----------------------------------------------


def two_port_accountants():
    """A tap's two receivers, as the live pipeline builds them: one accountant
    each, one direction each, both handed the same declarations."""
    both = {f"{PEER_MAC}/{CLASS}": BEAT, f"{LOCAL_MAC}/{CLASS}": BEAT}
    return (
        accountant(iface="mon1", policy=policy(beats=both),
                   fixed_direction=DIRECTION_OUT),
        accountant(iface="mon0", policy=policy(beats=both),
                   fixed_direction=DIRECTION_IN),
    )


def test_a_monitor_port_does_not_watch_for_the_leg_it_cannot_see():
    """The port carrying the request never carries the reply — it is on the
    other fibre. Arming both declarations on both accountants reports a leg
    that is arriving perfectly well as missing, once per interval, forever."""
    out_port, in_port = two_port_accountants()
    assert set(out_port.snapshot().beats) == {f"{PEER_MAC}/{CLASS}"}
    assert set(in_port.snapshot().beats) == {f"{LOCAL_MAC}/{CLASS}"}


def test_neither_port_invents_a_missing_beat_while_both_legs_flow():
    out_port, in_port = two_port_accountants()
    for n in range(6):
        out_port.observe(beacon(PEER, REQUEST), T + 5 + n * BEAT, DIRECTION_OUT)
        in_port.observe(beacon(LOCAL, REPLY), T + 5 + n * BEAT + 0.0004, DIRECTION_IN)
        out_port.roll(T + 5 + n * BEAT + 1)
        in_port.roll(T + 5 + n * BEAT + 1)
    assert kinds(out_port) == set(), "the request leg is on time"
    assert kinds(in_port) == set(), "so is the reply leg"


def test_the_port_that_can_see_a_stopped_leg_still_reports_it():
    """Narrowing what each port watches must not narrow what it catches."""
    out_port, in_port = two_port_accountants()
    out_port.observe(beacon(PEER, REQUEST), T + 5, DIRECTION_OUT)
    in_port.observe(beacon(LOCAL, REPLY), T + 5.0004, DIRECTION_IN)
    out_port.roll(T + 5 + 4 * BEAT)          # the request leg goes quiet
    in_port.observe(beacon(LOCAL, REPLY), T + 5 + BEAT, DIRECTION_IN)
    assert (KIND_BEAT_MISSED, CLASS) in out_port.snapshot().finding_groups
    assert beat_of(out_port).missed == 3


def test_a_capture_with_no_fixed_direction_arms_every_declaration():
    """This host's own end of a link, and any replay, sees both directions on
    one source — so scoping by direction would be wrong there."""
    both = {f"{PEER_MAC}/{CLASS}": BEAT, f"{LOCAL_MAC}/{CLASS}": BEAT}
    acct = accountant(policy=policy(beats=both))     # no fixed_direction
    assert set(acct.snapshot().beats) == set(both)


# --- an extra beat ------------------------------------------------------------


def test_a_frame_arriving_inside_the_cadence_is_unscheduled():
    """"Every 30s" is a two-sided claim. An extra datagram is a frame on the
    link nobody declared, and its timing is a channel; a beacon that only
    ever checked for absence would wave it through."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.observe(beacon(), T + 12, DIRECTION_OUT)  # 7s later: far too early
    assert (KIND_BEAT_UNSCHEDULED, CLASS) in acct.snapshot().finding_groups
    assert beat_of(acct).unscheduled == 1


def test_an_unscheduled_frame_does_not_drag_the_schedule_with_it():
    """Otherwise a flow could walk its own cadence forward one early frame at
    a time and never be early again."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.observe(beacon(), T + 12, DIRECTION_OUT)  # unscheduled
    acct.observe(beacon(), T + 35, DIRECTION_OUT)  # still on the original phase
    beat = beat_of(acct)
    assert (beat.beats, beat.unscheduled, beat.missed) == (2, 1, 0)


# --- counted once, on the right window ----------------------------------------


def test_a_missed_beat_is_filed_on_the_window_it_was_due_in():
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    # Silent from T+5 to well into the second window, then back.
    acct.observe(beacon(), T + W + 65, DIRECTION_OUT)
    first = acct.take_closed()[0]
    assert not first.complete
    missed = [g for g in first.groups if g.kind == KIND_BEAT_MISSED]
    assert missed, "the beats due inside the first window belong to it"
    assert missed[0].last_ts < first.window_end


def test_a_long_silence_taints_every_window_it_spans():
    """A findings-only record cannot tell a clean window from a blind one, so
    one tainted window and three clean ones would be the wrong answer."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    reports = acct.roll(T + 3 * W + 10)
    assert len(reports) == 3
    assert all(not r.complete for r in reports), [r.finding_count for r in reports]


def test_an_absence_is_counted_once_however_it_is_noticed():
    """The timer path and the frame path advance the same schedule, so a
    window closing over a silence and traffic resuming afterwards cannot both
    charge for it."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.roll(T + W + 10)  # noticed by the clock first
    acct.observe(beacon(), T + W + 20, DIRECTION_OUT)  # then the beacon returns
    # T+5 to T+W+20 is 315s: ten beats due, one arrived at the end.
    assert beat_of(acct).missed == 10
    assert beat_of(acct).beats == 2


def test_a_capture_that_stops_does_not_owe_the_beats_it_never_saw():
    """A beat due after the tap stopped was not missed, it was unobserved —
    and the capture-gap finding already says the window was cut short."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    reports = acct.finish(T + 40)
    assert beat_of(acct).missed == 0
    assert reports[0].finding_count == 1, "the capture gap, and nothing else"


# --- replay -------------------------------------------------------------------


def test_a_replay_anchors_on_its_first_beat_rather_than_on_process_start():
    """`process_epoch=0.0` means the file's own timeline is the record; when
    the beacon started is a fact about the recording, not about the link."""
    acct = accountant(process_epoch=0.0, window_seconds=None)
    acct.observe(beacon(), T + 900, DIRECTION_OUT)  # long after any process start
    acct.observe(beacon(), T + 900 + BEAT, DIRECTION_OUT)
    assert kinds(acct) == set()
    assert beat_of(acct).beats == 2


def test_a_wild_gap_resynchronises_rather_than_claiming_millions_of_misses():
    acct = accountant(process_epoch=0.0, window_seconds=None)
    acct.observe(beacon(), 1.0, DIRECTION_OUT)
    acct.observe(beacon(), T, DIRECTION_OUT)  # decades later
    beat = beat_of(acct)
    assert beat.resyncs == 1
    assert beat.missed == 0
    assert "resynchronised" in (
        acct.snapshot().finding_groups[(KIND_BEAT_MISSED, CLASS)].samples[0].detail
    )


# --- the beacon on the wire ---------------------------------------------------


def test_both_legs_of_the_round_trip_are_one_class_told_apart_by_sender():
    """Which is what lets each leg be declared separately: the request and the
    reply land on different monitor ports, so a missing one names the port."""
    acct = accountant(
        policy=policy(
            beats={f"{PEER_MAC}/{CLASS}": BEAT, f"{LOCAL_MAC}/{CLASS}": BEAT}
        )
    )
    for n in range(4):
        acct.observe(beacon(PEER, REQUEST), T + 5 + n * BEAT, DIRECTION_OUT)
        acct.observe(beacon(LOCAL, REPLY), T + 5 + n * BEAT + 0.0004, DIRECTION_IN)
    snap = acct.snapshot()
    assert kinds(acct) == set()
    assert snap.classes[CLASS].frames_out == snap.classes[CLASS].frames_in == 4
    assert beat_of(acct, PEER_MAC).beats == beat_of(acct, LOCAL_MAC).beats == 4
    # Two senders, two constant payloads, and neither is the other's.
    assert snap.pins[f"{PEER_MAC}/{CLASS}"].digest == payload_digest(REQUEST)
    assert snap.pins[f"{LOCAL_MAC}/{CLASS}"].digest == payload_digest(REPLY)


def test_one_leg_stopping_is_reported_against_that_leg_alone():
    """The half of the round trip that proves delivery: the request keeps
    arriving, so the sender and its monitor port are fine, and only the
    answer has stopped."""
    acct = accountant(
        policy=policy(
            beats={f"{PEER_MAC}/{CLASS}": BEAT, f"{LOCAL_MAC}/{CLASS}": BEAT}
        )
    )
    for n in range(6):
        acct.observe(beacon(PEER, REQUEST), T + 5 + n * BEAT, DIRECTION_OUT)
        if n < 2:
            acct.observe(beacon(LOCAL, REPLY), T + 5 + n * BEAT + 0.0004, DIRECTION_IN)
    assert beat_of(acct, PEER_MAC).missed == 0
    assert beat_of(acct, LOCAL_MAC).missed == 3
    group = acct.snapshot().finding_groups[(KIND_BEAT_MISSED, CLASS)]
    assert group.sources == {LOCAL_MAC}


def test_a_reply_to_a_low_source_port_would_mint_a_class_per_beat():
    """Why both ends bind the same port. An unnamed datagram is named for
    whichever port is not ephemeral, so a reply addressed back to a low source
    port is classified under *that* port — a new class every beat, none of
    them the declared one, and the beat silently never seen."""
    acct = accountant()
    acct.observe(beacon(LOCAL, REPLY, dport=33000), T + 5, DIRECTION_IN)
    assert "udp-33000" in acct.snapshot().classes
    assert CLASS not in acct.snapshot().classes


def test_the_payload_is_long_enough_to_need_no_ethernet_padding():
    """A shorter one would be padded to the 60-byte minimum, and padding is
    separately required to be zero — a rule worth not depending on here.
    Derived, not hardcoded, so changing the payload cannot silently drop the
    frame under the minimum."""
    assert len(beacon()) == FRAME_BYTES >= 60
    assert "padding-nonzero" not in kinds(
        _observed(accountant(), beacon(), T + 5, DIRECTION_OUT)
    )


def _observed(acct, frame, ts, direction):
    acct.observe(frame, ts, direction)
    return acct


# --- what the pin covers ------------------------------------------------------


def test_pinning_the_payload_alone_leaves_the_headers_free():
    """The gap whole-frame pinning closes. Same payload, different TTL: the
    frame is not the one that was declared and nothing says so."""
    acct = accountant()
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.observe(beacon(ttl=17), T + 5 + BEAT, DIRECTION_OUT)
    assert kinds(acct) == set()
    assert acct.snapshot().pins[f"{PEER_MAC}/{CLASS}"].constant


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"ttl": 17}, id="ttl"),
        pytest.param({"tos": 0x28}, id="dscp"),
    ],
)
def test_a_whole_frame_pin_holds_the_header_bytes_the_sender_chooses(changed):
    """A payload pin sees none of these: same payload, different frame."""
    acct = accountant(whole_frame_classes={CLASS}, policy=policy(whole_frame=True))
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    acct.observe(beacon(**changed), T + 5 + BEAT, DIRECTION_OUT)
    assert ("pin-mismatch", CLASS) in acct.snapshot().finding_groups


def test_identification_is_the_one_field_a_whole_frame_pin_does_not_hold():
    """Not an oversight — the sending kernel assigns it, and on the tapped
    link it was measured moving on every datagram even with DF set. Pinned, it
    would raise a mismatch per frame and say nothing about anything."""
    acct = accountant(whole_frame_classes={CLASS}, policy=policy(whole_frame=True))
    for n, ident in enumerate((0x0000, 0x3C86, 0xBEEF)):
        acct.observe(beacon(ident=ident), T + 5 + n * BEAT, DIRECTION_OUT)
    assert kinds(acct) == set()
    assert acct.snapshot().pins[f"{PEER_MAC}/{CLASS}"].constant


def test_a_header_checksum_that_the_header_does_not_imply_is_a_finding():
    """What replaces pinning the checksum. Arithmetic beats a pin here: it
    holds for a frame nobody has seen before, where a pin can only compare
    against one that has."""
    acct = accountant(whole_frame_classes={CLASS}, policy=policy(whole_frame=True))
    frame = bytearray(beacon())
    frame[24:26] = b"\xba\xad"
    acct.observe(bytes(frame), T + 5, DIRECTION_OUT)
    assert ("ipv4-checksum", CLASS) in acct.snapshot().finding_groups


def test_the_checksum_is_only_checked_where_the_pin_excludes_it():
    """A capture taken on a sending host sees checksums before the NIC fills
    them in, so applying this to every IPv4 frame would raise a finding per
    frame on a replay of one."""
    acct = accountant(policy=policy())      # payload pin: checksum is not excluded
    frame = bytearray(beacon())
    frame[24:26] = b"\xba\xad"
    acct.observe(bytes(frame), T + 5, DIRECTION_OUT)
    assert ("ipv4-checksum", CLASS) not in acct.snapshot().finding_groups


def test_a_whole_frame_pin_covers_the_frame_from_byte_zero():
    acct = accountant(whole_frame_classes={CLASS}, policy=policy(whole_frame=True))
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    pin = acct.snapshot().pins[f"{PEER_MAC}/{CLASS}"]
    assert pin.whole_frame
    assert pin.payload_bytes == FRAME_BYTES, "the whole frame, not just the payload"
    assert pin.declared == pin.digest


def test_a_class_not_named_keeps_the_default_pin_region():
    """Whole-frame pinning is opt-in per class: it is only sound where the
    sender was configured to make every header field deterministic."""
    acct = accountant(whole_frame_classes={"udp-1234"}, policy=policy())
    acct.observe(beacon(), T + 5, DIRECTION_OUT)
    pin = acct.snapshot().pins[f"{PEER_MAC}/{CLASS}"]
    assert not pin.whole_frame and pin.payload_bytes == len(REQUEST)


# --- pinning ARP --------------------------------------------------------------


def test_arp_can_be_pinned_on_a_link_where_it_is_constant():
    """`UNPINNED_CLASSES` exempts ARP because a router resolving several
    neighbours varies its target every frame. On a /30 with two hosts it does
    not vary at all, and the exemption is the wrong default."""
    acct = accountant(unpinned_classes=UNPINNED_CLASSES - {"arp"})
    acct.observe(arp(), T + 5, DIRECTION_OUT)
    acct.observe(arp(), T + 5 + BEAT, DIRECTION_OUT)
    pin = acct.snapshot().pins[f"{PEER_MAC}/arp"]
    assert pin.constant and pin.frames == 2


# --- the header fields that carry nothing ---------------------------------------
#
# Each of these is a place a frame could hold bytes that no rule examined. They
# live here rather than in test_accounting because the case that motivated them
# is this link: on a point-to-point segment carrying one service, every one of
# these fields has exactly one legitimate value, and none of them had a rule.


def tcp_frame(*, ihl: int = 5, ipopts: bytes = b"", flags_frag: int = 0x4000,
              reserved: int = 0, urgent: int = 0, tcp_flags: int = 0x18) -> bytes:
    """One inference response segment, with the fields under test parameterised."""
    body = struct.pack("!HHIIBBHHH", 8000, 51234, 1, 1,
                       (5 << 4) | reserved, tcp_flags, 501, 0, urgent) + b"payload"
    header = struct.pack("!BBHHHBBH4s4s", 0x40 | ihl, 0, 20 + len(ipopts) + len(body),
                         1, flags_frag, 64, 6, 0, bytes(4), bytes(4))
    return _peer_of(PEER) + PEER + struct.pack("!H", 0x0800) + header + ipopts + body


def test_ipv4_options_are_a_finding():
    """The largest single channel in an inference segment: the header length is
    read to find the transport and never constrained, so up to forty bytes ride
    in every packet, examined by nothing."""
    acct = accountant(policy=policy(beats={}))
    acct.observe(tcp_frame(ihl=15, ipopts=b"\xde\xad\xbe\xef" * 10), T + 5, DIRECTION_OUT)
    assert ("ipv4-options", "ipv4-tcp") in acct.snapshot().finding_groups


def test_a_header_with_no_options_is_not():
    acct = accountant(policy=policy(beats={}))
    acct.observe(tcp_frame(), T + 5, DIRECTION_OUT)
    assert "ipv4-options" not in kinds(acct)


@pytest.mark.parametrize(
    "flags_frag, why",
    [
        pytest.param(0xC000, "reserved bit", id="reserved-bit"),
        pytest.param(0x6000, "More Fragments", id="more-fragments"),
        pytest.param(0x4001, "fragment offset", id="fragment-offset"),
    ],
)
def test_fragmentation_and_the_reserved_bit_are_findings(flags_frag, why):
    """A link with one MTU and no router between its ends never fragments, so
    this is evasion or a fault — and a reassembly nobody performs is a
    reassembly nobody is checking."""
    acct = accountant(policy=policy(beats={}))
    acct.observe(tcp_frame(flags_frag=flags_frag), T + 5, DIRECTION_OUT)
    assert ("ipv4-fragmented", "ipv4-tcp") in acct.snapshot().finding_groups


def test_the_tcp_reserved_bits_must_be_zero():
    acct = accountant(policy=policy(beats={}))
    acct.observe(tcp_frame(reserved=0x0F), T + 5, DIRECTION_OUT)
    assert ("tcp-reserved", "ipv4-tcp") in acct.snapshot().finding_groups


def test_an_urgent_pointer_with_urg_clear_is_a_finding():
    """Sixteen bits the receiver is instructed to ignore, in every segment."""
    acct = accountant(policy=policy(beats={}))
    acct.observe(tcp_frame(urgent=0xBEEF), T + 5, DIRECTION_OUT)
    assert ("tcp-urgent", "ipv4-tcp") in acct.snapshot().finding_groups


def test_an_urgent_pointer_with_urg_set_is_not():
    """The field means something then. Flagging it would be flagging TCP."""
    acct = accountant(policy=policy(beats={}))
    acct.observe(tcp_frame(urgent=0xBEEF, tcp_flags=0x38), T + 5, DIRECTION_OUT)
    assert "tcp-urgent" not in kinds(acct)


def test_a_frame_addressed_to_neither_end_is_a_finding():
    """Six bytes that nothing looked at. A wrong destination is not harmless
    because the peer's NIC would drop it — the frame still crossed the link."""
    acct = accountant(policy=policy(beats={}))
    stray = bytes.fromhex("020000000001") + tcp_frame()[6:]
    acct.observe(stray, T + 5, DIRECTION_OUT)
    assert ("mac-destination", "ipv4-tcp") in acct.snapshot().finding_groups


def test_tcp_addressed_to_a_group_address_is_a_finding():
    acct = accountant(policy=policy(beats={}))
    acct.observe(b"\x01\x80\xc2\x00\x00\x0e" + tcp_frame()[6:], T + 5, DIRECTION_OUT)
    assert ("mac-destination", "ipv4-tcp") in acct.snapshot().finding_groups


def test_a_declared_group_address_is_permitted():
    """LLDP's reserved multicast, on a link where LLDP is running. Nothing on
    this one speaks to a group, so it is permitted by name rather than by
    being a group address at all."""
    lldp_dst = bytes.fromhex("0180c200000e")
    acct = accountant(policy=policy(beats={}, extra_destinations=frozenset({lldp_dst})),
                      unpinned_classes=UNPINNED_CLASSES)
    acct.observe(lldp_dst + PEER + struct.pack("!H", 0x8899) + bytes(46),
                 T + 5, DIRECTION_OUT)
    assert "mac-destination" not in kinds(acct)


# --- ARP: every field, held to the one value it can have ------------------------
#
# On a segment with many hosts ARP genuinely varies and a pin would be wrong.
# On a link with exactly two ends nothing varies at all, so each field is held
# to what the protocol and the topology determine — which reaches the same
# place a pin would, without learning anything or declaring a digest.


def arp_frame(*, dst=None, src=PEER, op=1, htype=1, ptype=0x0800, hlen=6, plen=4,
              sha=None, tha=None) -> bytes:
    if dst is None:
        # Unicast to the peer, which is what resolution on this link is: two
        # NICs, both addresses configured, nobody else to ask.
        dst = _peer_of(src)
    if tha is None:
        tha = bytes(6) if op == 1 else dst
    body = (struct.pack("!HHBBH", htype, ptype, hlen, plen, op)
            + (sha or src) + bytes(4) + tha + bytes(4))
    f = dst + src + struct.pack("!H", 0x0806) + body
    return f + bytes(max(0, 60 - len(f)))


def arp_accountant():
    return accountant(policy=policy(beats={}), unpinned_classes=UNPINNED_CLASSES)


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"htype": 0xBEEF}, id="hardware-type"),
        pytest.param({"ptype": 0xBEEF}, id="protocol-type"),
        pytest.param({"hlen": 7}, id="hardware-address-length"),
        pytest.param({"plen": 5}, id="protocol-address-length"),
    ],
)
def test_the_arp_shape_descriptors_are_checked_not_merely_read(changed):
    """hlen and plen say where every later field begins. Reading them without
    checking them meant a frame could declare a length nobody validated and
    walk the identity and address rules off their offsets."""
    acct = arp_accountant()
    acct.observe(arp_frame(**changed), T + 5, DIRECTION_OUT)
    assert ("arp-shape", "arp") in acct.snapshot().finding_groups


def test_a_wrong_shape_stops_the_rules_that_depend_on_it():
    """Rather than reading the later fields from offsets that are no longer
    where they are claimed to be."""
    acct = arp_accountant()
    acct.observe(arp_frame(hlen=7, sha=bytes(6)), T + 5, DIRECTION_OUT)
    assert "arp-source-mismatch" not in kinds(acct), "nothing sound left to read"


def test_the_target_hardware_address_in_a_request_must_be_zero():
    """Six bytes the protocol says are IGNORED on a request — which is exactly
    what makes them attractive, and what nothing checked."""
    acct = arp_accountant()
    acct.observe(arp_frame(tha=bytes.fromhex("deadbeefcafe")), T + 5, DIRECTION_OUT)
    assert ("arp-target-hw", "arp") in acct.snapshot().finding_groups


def test_a_reply_must_name_the_host_it_is_addressed_to():
    acct = arp_accountant()
    acct.observe(arp_frame(src=LOCAL, op=2, tha=b"\xff" * 6), T + 5, DIRECTION_IN)
    assert ("arp-target-hw", "arp") in acct.snapshot().finding_groups


@pytest.mark.parametrize(
    "dst, why",
    [
        pytest.param(bytes.fromhex("01005e010203"), "not a link address", id="other-multicast"),
        pytest.param(bytes.fromhex("020000000001"), "neither end of the link", id="stranger"),
    ],
)
def test_an_arp_request_must_go_to_the_link_or_to_everyone(dst, why):
    acct = arp_accountant()
    acct.observe(arp_frame(dst=dst), T + 5, DIRECTION_OUT)
    assert ("mac-destination", "arp") in acct.snapshot().finding_groups, why


def test_a_unicast_probe_is_what_resolution_looks_like_here():
    """Re-validating a known neighbour sends unicast probes to that address —
    ucast_solicit, three by default — and on a link whose two addresses are
    both configured that is all the ARP there is."""
    acct = arp_accountant()
    acct.observe(arp_frame(dst=LOCAL), T + 5, DIRECTION_OUT)
    assert kinds(acct) == set()


def test_a_broadcast_request_is_a_finding_unless_declared():
    """Two NICs reaching each other by a static route: a frame addressed to
    everybody is addressed to one host by a less specific name. The only thing
    that would send one is a resolution for an address already configured."""
    acct = arp_accountant()
    acct.observe(arp_frame(dst=b"\xff" * 6), T + 5, DIRECTION_OUT)
    assert ("mac-destination", "arp") in acct.snapshot().finding_groups


def test_a_shared_segment_can_declare_broadcast():
    acct = accountant(policy=policy(beats={}, allow_broadcast=True),
                      unpinned_classes=UNPINNED_CLASSES)
    acct.observe(arp_frame(dst=b"\xff" * 6), T + 5, DIRECTION_OUT)
    assert kinds(acct) == set()


def test_an_honest_exchange_raises_nothing():
    acct = arp_accountant()
    acct.observe(arp_frame(), T + 5, DIRECTION_OUT)
    acct.observe(arp_frame(src=LOCAL, op=2), T + 5.001, DIRECTION_IN)
    assert kinds(acct) == set()


# --- what the rules compare against ---------------------------------------------
#
# Most of it is the protocol's: an Ethernet ARP has a six-byte hardware
# address, broadcast is ff:ff:ff:ff:ff:ff, the urgent pointer means nothing
# with URG clear. Those are not opinions about a deployment and are not
# configurable. These two are properties of a LINK, so they are declared.


def test_ip_options_can_be_declared_legitimate():
    """IGMP carries the Router Alert option. A link where that is expected
    should not need the rule rebuilt to say so."""
    acct = accountant(policy=policy(beats={}), allow_ip_options=True)
    acct.observe(tcp_frame(ihl=15, ipopts=b"\x94\x04\x00\x00" * 10), T + 5, DIRECTION_OUT)
    assert "ipv4-options" not in kinds(acct)


def test_fragments_can_be_declared_legitimate():
    acct = accountant(policy=policy(beats={}), allow_fragments=True)
    acct.observe(tcp_frame(flags_frag=0x6000), T + 5, DIRECTION_OUT)
    assert "ipv4-fragmented" not in kinds(acct)


def test_the_reserved_bit_is_not_part_of_that_judgement():
    """It has no meaning under any configuration, so allowing fragments does
    not allow it."""
    acct = accountant(policy=policy(beats={}), allow_fragments=True)
    acct.observe(tcp_frame(flags_frag=0xC000), T + 5, DIRECTION_OUT)
    assert ("ipv4-fragmented", "ipv4-tcp") in acct.snapshot().finding_groups


def test_an_extra_destination_can_be_declared():
    """A deployment where some group address is legitimately in use names it,
    rather than needing the destination rule changed."""
    stray = bytes.fromhex("01005e010203")
    acct = accountant(policy=policy(beats={}, extra_destinations=frozenset({stray})))
    acct.observe(arp_frame(dst=stray), T + 5, DIRECTION_OUT)
    assert "mac-destination" not in kinds(acct)
