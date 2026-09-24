"""Frame accounting: classification, pins, structural rules, and the balance.

The invariant these tests exist to protect is arithmetic — every frame
observed is placed in exactly one class — because that is the only thing that
turns the parser into a statement about the whole link rather than about the
frames it happened to recognise.
"""

from __future__ import annotations

import struct

import pytest

from app.accounting import (
    CLASS_MALFORMED,
    CLASS_TCP,
    CLASS_TCP6,
    FrameAccountant,
    classify,
    ethernet_header_len,
)

SRC = bytes.fromhex("aabbccddeeff")
DST = bytes.fromhex("112233445566")


def eth(ethertype: int, payload: bytes = b"", *, src: bytes = SRC, tags: int = 0) -> bytes:
    header = DST + src
    for _ in range(tags):
        header += struct.pack("!HH", 0x8100, 0x0064)
    return header + struct.pack("!H", ethertype) + payload


def ipv4(protocol: int, payload: bytes = b"", *, ident: int = 0x1234) -> bytes:
    total = 20 + len(payload)
    header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, total, ident, 0, 64, protocol, 0,
        bytes(4), bytes(4),
    )
    return eth(0x0800, header + payload)


def ipv6(next_header: int, payload: bytes = b"", *, ext: bytes = b"") -> bytes:
    body = ext + payload
    header = struct.pack("!IHBB", 0x60000000, len(body), next_header, 64) + bytes(32)
    return eth(0x86DD, header + body)


def udp(sport: int, dport: int, payload: bytes) -> bytes:
    return struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload


def pad_to_min(frame: bytes, fill: bytes = b"\x00") -> bytes:
    return frame + fill * max(0, 60 - len(frame))


def arp_body(*, opcode: int = 1, sender: bytes = SRC) -> bytes:
    """A well-formed ARP payload — the opcode and sender rules are checked."""
    return (
        struct.pack("!HHBBH", 1, 0x0800, 6, 4, opcode)
        + sender + bytes(4)
        + bytes(6) + bytes(4)
    )


# --- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    "frame, expected",
    [
        (eth(0x0806, bytes(28)), "arp"),
        (eth(0x88CC, b"\x00\x00"), "lldp"),
        (eth(0x8899, bytes(32)), "realtek"),
        (ipv4(6, bytes(20)), CLASS_TCP),
        (ipv4(2, bytes(8)), "igmp"),
        (ipv4(1, bytes(8)), "icmp"),
        (ipv4(17, udp(1900, 1900, b"NOTIFY")), "ssdp"),
        (ipv4(17, udp(1234, 123, b"x" * 40)), "ntp"),
        (ipv4(17, udp(5000, 7788, b"x" * 8)), "udp-7788"),
        (ipv6(58, bytes(8)), "icmpv6"),
        (ipv6(17, udp(546, 547, b"x")), "dhcpv6"),
        (ipv6(6, bytes(20)), CLASS_TCP6),
        (b"\x00" * 8, CLASS_MALFORMED),
    ],
)
def test_classify_names_each_family(frame, expected):
    assert classify(frame).name == expected


def test_stp_is_recognised_through_the_llc_header():
    # An 802.3 length field rather than an EtherType, then DSAP/SSAP 0x42.
    bpdu = b"\x42\x42\x03" + bytes(35)
    frame = DST + SRC + struct.pack("!H", len(bpdu)) + bpdu
    assert classify(frame).name == "stp"


def test_vlan_tags_do_not_hide_the_protocol():
    """A tag displaces the EtherType; it must be followed, not tripped over."""
    assert ethernet_header_len(eth(0x88CC, tags=0)) == 14
    assert ethernet_header_len(eth(0x88CC, tags=1)) == 18
    assert ethernet_header_len(eth(0x88CC, tags=2)) == 22
    assert classify(eth(0x88CC, b"\x00\x00", tags=2)).name == "lldp"


def test_ipv6_extension_headers_are_walked_to_the_transport():
    hop_by_hop = bytes([58, 0]) + bytes(6)  # next=ICMPv6, one 8-octet unit
    assert classify(ipv6(0, bytes(8), ext=hop_by_hop)).name == "icmpv6"


def test_an_extension_header_chain_cannot_spin_forever():
    """A chain that only ever points at itself must terminate as malformed."""
    # Repeated hop-by-hop headers, more than the walk will follow.
    chain = (bytes([0, 0]) + bytes(6)) * 12
    assert classify(ipv6(0, b"", ext=chain)).name == CLASS_MALFORMED


def test_unnamed_udp_is_named_by_its_service_port_not_the_ephemeral_one():
    # A sender using a fresh ephemeral source port per datagram must not mint
    # a new class every frame.
    assert classify(ipv4(17, udp(51000, 9999, b"x"))).name == "udp-9999"
    assert classify(ipv4(17, udp(9999, 51000, b"x"))).name == "udp-9999"


def test_an_unknown_ethertype_is_still_given_a_class():
    assert classify(eth(0x1234, bytes(40))).name == "eth-other"


# --- the balance ------------------------------------------------------------


def test_every_observed_frame_is_classified_exactly_once():
    accountant = FrameAccountant()
    frames = [
        eth(0x0806, bytes(28)),
        eth(0x88CC, b"\x00\x00"),
        ipv4(6, bytes(20)),
        ipv6(58, bytes(8)),
        b"\x00" * 4,  # malformed
        eth(0x1234, bytes(40)),  # unknown
    ]
    for frame in frames:
        accountant.observe(frame, 1.0)
    snapshot = accountant.snapshot()
    assert snapshot.observed == len(frames)
    assert snapshot.classified == len(frames)
    assert snapshot.balanced
    assert sum(t.frames for t in snapshot.classes.values()) == len(frames)


def test_a_truncated_frame_still_balances_but_is_never_complete():
    accountant = FrameAccountant()
    accountant.note_truncated(9000, 2048, 1.0)
    snapshot = accountant.snapshot()
    assert snapshot.balanced, "a frame we could not read is still a frame we saw"
    assert snapshot.truncated == 1
    assert not snapshot.complete


def test_kernel_drops_disqualify_the_window_even_when_nothing_else_is_wrong():
    accountant = FrameAccountant()
    accountant.observe(eth(0x8899, bytes(32)), 1.0)
    assert accountant.snapshot().complete
    accountant.note_kernel_dropped(1)
    snapshot = accountant.snapshot()
    assert snapshot.balanced, "drops do not unbalance what we did see"
    assert not snapshot.complete, "but they do mean we cannot claim the link"


def test_observe_never_raises_into_the_ring_walk(monkeypatch):
    """An exception here would cost the block and every frame in it."""
    monkeypatch.setattr(
        "app.accounting.classify", lambda data: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    accountant = FrameAccountant()
    assert accountant.observe(eth(0x8899, bytes(32)), 1.0) == CLASS_MALFORMED
    snapshot = accountant.snapshot()
    assert snapshot.errors == 1
    assert snapshot.balanced


# --- pins -------------------------------------------------------------------


def test_a_constant_sender_pins_and_raises_nothing():
    accountant = FrameAccountant()
    frame = eth(0x8899, bytes(32))
    for _ in range(50):
        accountant.observe(frame, 1.0)
    pin = accountant.snapshot().pins["aa:bb:cc:dd:ee:ff/realtek"]
    assert pin.frames == 50
    assert pin.distinct == 1
    assert pin.constant
    assert accountant.snapshot().findings == 0


def test_a_second_payload_is_one_finding_not_one_per_frame():
    """Findings scale with variety, not volume — else one noisy flow buries
    everything else."""
    accountant = FrameAccountant()
    a, b = eth(0x8899, bytes(32)), eth(0x8899, b"\x01" + bytes(31))
    for _ in range(100):
        accountant.observe(a, 1.0)
        accountant.observe(b, 1.0)
    snapshot = accountant.snapshot()
    assert snapshot.findings == 1
    assert snapshot.pins["aa:bb:cc:dd:ee:ff/realtek"].distinct == 2


def test_the_variant_set_is_bounded_for_a_sender_that_never_repeats():
    accountant = FrameAccountant(max_variants=4)
    for i in range(40):
        accountant.observe(eth(0x8899, struct.pack("!I", i) + bytes(28)), 1.0)
    pin = accountant.snapshot().pins["aa:bb:cc:dd:ee:ff/realtek"]
    assert len(pin.variants) == 4, "retained payloads must not grow without bound"
    assert pin.distinct == 40, "but the true count is still reported"
    assert pin.overflowed


def test_the_pin_ignores_the_ipv4_identification_counter():
    """Otherwise every IP-carried flow looks variable and the rule is noise."""
    accountant = FrameAccountant()
    for ident in range(1, 31):
        accountant.observe(ipv4(2, b"\x22" + bytes(7), ident=ident), 1.0)
    snapshot = accountant.snapshot()
    assert snapshot.pins["aa:bb:cc:dd:ee:ff/igmp"].distinct == 1
    assert snapshot.findings == 0


def test_classes_that_are_meant_to_vary_are_not_pinned():
    accountant = FrameAccountant()
    for i in range(20):
        body = f"NOTIFY * HTTP/1.1\r\nUSN: {i}\r\n".encode()
        accountant.observe(ipv4(17, udp(1900, 1900, body)), 1.0)
    snapshot = accountant.snapshot()
    assert snapshot.classes["ssdp"].frames == 20
    assert snapshot.findings == 0, "SSDP varies by design; pinning it says nothing"
    assert not any(key.endswith("/ssdp") for key in snapshot.pins)


# --- structural rules -------------------------------------------------------


def test_nonzero_ethernet_padding_is_a_finding():
    accountant = FrameAccountant()
    clean = pad_to_min(eth(0x0806, arp_body()))
    accountant.observe(clean, 1.0)
    assert accountant.snapshot().findings == 0

    accountant = FrameAccountant()
    smuggled = pad_to_min(eth(0x0806, arp_body()), fill=b"\xff")
    accountant.observe(smuggled, 1.0)
    groups = accountant.snapshot().finding_groups
    assert ("padding-nonzero", "arp") in groups


def test_bytes_after_end_of_lldpdu_are_a_finding():
    # Chassis ID TLV (type 1, length 2), then the terminator.
    chain = struct.pack("!H", (1 << 9) | 2) + b"\x04\x00" + b"\x00\x00"
    accountant = FrameAccountant()
    accountant.observe(eth(0x88CC, chain), 1.0)
    assert accountant.snapshot().findings == 0

    accountant = FrameAccountant()
    accountant.observe(eth(0x88CC, chain + b"hidden payload"), 1.0)
    assert ("tlv-mismatch", "lldp") in accountant.snapshot().finding_groups


def test_an_lldp_chain_with_no_terminator_is_a_finding():
    chain = struct.pack("!H", (1 << 9) | 2) + b"\x04\x00"
    accountant = FrameAccountant()
    accountant.observe(eth(0x88CC, chain), 1.0)
    assert ("tlv-mismatch", "lldp") in accountant.snapshot().finding_groups


def test_arp_claiming_a_hardware_address_it_is_not_sending_from_is_a_finding():
    """The rule that stops ARP being a way to speak as somebody else."""
    accountant = FrameAccountant()
    accountant.observe(pad_to_min(eth(0x0806, arp_body())), 1.0)
    assert accountant.snapshot().findings == 0

    spoofed = arp_body(sender=DST)
    accountant = FrameAccountant()
    accountant.observe(pad_to_min(eth(0x0806, spoofed)), 1.0)
    assert ("arp-source-mismatch", "arp") in accountant.snapshot().finding_groups


def test_an_arp_opcode_that_is_neither_request_nor_reply_is_a_finding():
    accountant = FrameAccountant()
    accountant.observe(pad_to_min(eth(0x0806, arp_body(opcode=9))), 1.0)
    assert ("arp-opcode", "arp") in accountant.snapshot().finding_groups


def test_an_unknown_ethertype_raises_a_finding_rather_than_passing_quietly():
    accountant = FrameAccountant()
    accountant.observe(eth(0x1234, bytes(40)), 1.0)
    snapshot = accountant.snapshot()
    assert ("unclassified", "eth-other") in snapshot.finding_groups
    assert not snapshot.complete, "reject by default is the whole point"


def test_a_memoryview_is_accepted_without_being_retained():
    """The ring releases its view as soon as the walk moves on."""
    accountant = FrameAccountant()
    frame = bytearray(pad_to_min(eth(0x8899, bytes(32))))
    view = memoryview(frame)
    accountant.observe(view, 1.0)
    view.release()  # would raise if the accountant still held a slice
    assert accountant.snapshot().pins["aa:bb:cc:dd:ee:ff/realtek"].distinct == 1
