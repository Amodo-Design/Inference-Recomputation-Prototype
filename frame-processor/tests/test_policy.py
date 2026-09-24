"""The link whitelist: which frames may cross the tapped link, and which way.

The tapped node is one machine on a point-to-point link, so the whitelist is
short and directional. These tests build one such link — the tapped node
serving on port 8000, the capture host at the other end — and check that a
clean exchange raises nothing while each way of departing from it raises
exactly the finding that names the departure.
"""

from __future__ import annotations

import struct

import pytest

from app import policy as policy_module
from app.accounting import FrameAccountant
from app.config import _parse_iface_directions
from app.policy import (
    DIRECTION_IN,
    DIRECTION_OUT,
    DIRECTION_UNKNOWN,
    PACKET_OUTGOING,
    FixedDirection,
    LinkPolicy,
    direction_from_pkttype,
    parse_declared_beats,
    parse_declared_pins,
    payload_digest,
)

PEER_MAC = bytes.fromhex("020000000003")  # the tapped node's NIC
LOCAL_MAC = bytes.fromhex("020000000002")  # the capture host's end of the link
OTHER_MAC = bytes.fromhex("deadbeef0001")
PEER_IP = bytes([192, 0, 2, 2])
LOCAL_IP = bytes([192, 0, 2, 1])
OTHER_IP = bytes([192, 168, 50, 40])
LLDP_MCAST = bytes.fromhex("0180c200000e")
BROADCAST = b"\xff" * 6

LLDP_PAYLOAD = (
    struct.pack("!H", (1 << 9) | 7) + b"\x04" + PEER_MAC  # chassis id
    + struct.pack("!H", (2 << 9) | 7) + b"\x03" + PEER_MAC  # port id
    + struct.pack("!H", (3 << 9) | 2) + b"\x00\x78"  # ttl
    + b"\x00\x00"  # end of lldpdu
)


def eth(src: bytes, dst: bytes, ethertype: int, payload: bytes) -> bytes:
    return dst + src + struct.pack("!H", ethertype) + payload


def ipv4(src_mac: bytes, dst_mac: bytes, src_ip: bytes, dst_ip: bytes, proto: int, payload: bytes) -> bytes:
    header = struct.pack(
        "!BBHHHBBH4s4s", 0x45, 0, 20 + len(payload), 1, 0x4000, 63, proto, 0, src_ip, dst_ip
    )
    return eth(src_mac, dst_mac, 0x0800, header + payload)


def tcp(sport: int, dport: int, payload: bytes = b"", flags: int = 0x18) -> bytes:
    """Default flags are PSH|ACK — mid-stream data. 0x02 is a bare SYN."""
    return struct.pack("!HHIIBBHHH", sport, dport, 1, 1, 5 << 4, flags, 64, 0, 0) + payload


def udp(sport: int, dport: int, payload: bytes) -> bytes:
    return struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload


def arp(src_mac: bytes, dst_mac: bytes, *, opcode: int, sender_ip: bytes, target_ip: bytes, target_mac: bytes = bytes(6)) -> bytes:
    body = (
        struct.pack("!HHBBH", 1, 0x0800, 6, 4, opcode)
        + src_mac + sender_ip + target_mac + target_ip
    )
    frame = eth(src_mac, dst_mac, 0x0806, body)
    return frame + bytes(max(0, 60 - len(frame)))


def lldp(src_mac: bytes = PEER_MAC, payload: bytes = LLDP_PAYLOAD) -> bytes:
    return eth(src_mac, LLDP_MCAST, 0x88CC, payload)


# An example link, spelled the way the manifest would spell it: LLDP running
# at both ends and ARP resolving by broadcast. Both are declared rather than
# assumed, because the strict defaults reject both. Declaring them here keeps
# this suite testing what it says it tests: that a clean link raises nothing,
# for the link as declared.
LINK = LinkPolicy(
    peer_mac=PEER_MAC,
    local_mac=LOCAL_MAC,
    peer_ip=PEER_IP,
    local_ip=LOCAL_IP,
    peer_ports=frozenset({8000}),
    allow_out=frozenset({"lldp", "arp", "ipv4-tcp"}),
    allow_in=frozenset({"arp", "ipv4-tcp"}),
    declared_pins={f"{PEER_MAC.hex(':')}/lldp": payload_digest(LLDP_PAYLOAD)},
    allow_broadcast=True,
    extra_destinations=frozenset({bytes.fromhex("0180c200000e")}),  # LLDP
)


def findings(accountant: FrameAccountant) -> dict[str, int]:
    return {
        kind: group.count
        for (kind, _cls), group in accountant.snapshot().finding_groups.items()
    }


def request_and_response() -> list[tuple[bytes, str]]:
    """One HTTP exchange on the inference port, with the kernel's direction."""
    return [
        (ipv4(LOCAL_MAC, PEER_MAC, LOCAL_IP, PEER_IP, 6, tcp(38506, 8000, b"GET /v1/models")), DIRECTION_IN),
        (ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 6, tcp(8000, 38506, b"HTTP/1.1 200 OK")), DIRECTION_OUT),
    ]


# --- direction --------------------------------------------------------------


def test_the_kernel_bit_names_the_direction():
    assert direction_from_pkttype(PACKET_OUTGOING) == DIRECTION_IN, "we sent it → tapped node receives it"
    assert direction_from_pkttype(0) == DIRECTION_OUT  # PACKET_HOST
    assert direction_from_pkttype(1) == DIRECTION_OUT  # PACKET_BROADCAST
    assert direction_from_pkttype(2) == DIRECTION_OUT  # PACKET_MULTICAST
    assert direction_from_pkttype(None) is None


def test_replay_falls_back_to_the_source_mac():
    assert LINK.direction_from_source(lldp()) == DIRECTION_OUT
    assert LINK.direction_from_source(arp(LOCAL_MAC, BROADCAST, opcode=1, sender_ip=LOCAL_IP, target_ip=PEER_IP)) == DIRECTION_IN
    assert LINK.direction_from_source(lldp(src_mac=OTHER_MAC)) == DIRECTION_UNKNOWN


def test_the_kernel_direction_beats_the_mac_and_the_mac_is_then_checked():
    """A frame that arrived from the wire is an OUT frame whatever it says.

    The peer writing our MAC as its source does not make its frame ours; it
    makes it a frame from the wire with the wrong MAC on it.
    """
    accountant = FrameAccountant(policy=LINK)
    spoofed = ipv4(LOCAL_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 6, tcp(8000, 40000))
    accountant.observe(spoofed, 1.0, DIRECTION_OUT)
    assert findings(accountant) == {"mac-unknown": 1}
    assert accountant.snapshot().classes["ipv4-tcp"].frames_out == 1


def test_a_monitor_port_stamps_its_declared_direction_over_the_kernel_bit():
    """A tap's monitor port never transmits, so the kernel calls everything on
    it "received" and the pkttype route would label both directions OUT. The
    interface is the direction; the declaration wins."""
    accountant = FrameAccountant(policy=LINK)
    toward_gig2 = FixedDirection(accountant, DIRECTION_IN)
    from_gig2 = FixedDirection(accountant, DIRECTION_OUT)
    request, response = request_and_response()
    # What the kernel would say on a monitor port: PACKET_HOST → "out".
    toward_gig2.observe(request[0], 1.0, direction_from_pkttype(0))
    from_gig2.observe(response[0], 1.0, direction_from_pkttype(0))
    snapshot = accountant.snapshot()
    assert snapshot.findings == 0, snapshot.finding_groups
    assert snapshot.classes["ipv4-tcp"].frames_in == 1
    assert snapshot.classes["ipv4-tcp"].frames_out == 1
    # The adapter is transparent for everything but direction.
    toward_gig2.note_truncated(9000, 2048, 2.0)
    assert toward_gig2.snapshot().truncated == 1


def test_a_monitor_port_direction_must_be_one_of_the_two():
    with pytest.raises(ValueError):
        FixedDirection(FrameAccountant(), DIRECTION_UNKNOWN)


def test_monitor_port_directions_are_declared_per_interface():
    assert _parse_iface_directions("mon0=in, mon1=OUT") == {
        "mon0": "in",
        "mon1": "out",
    }
    assert _parse_iface_directions("mon0=sideways,mon1=out") == {"mon1": "out"}
    assert _parse_iface_directions(None) == {}
    assert _parse_iface_directions("") == {}


# --- the clean link ---------------------------------------------------------


def test_the_link_as_measured_raises_nothing():
    accountant = FrameAccountant(policy=LINK)
    for _ in range(3):
        accountant.observe(lldp(), 1.0, DIRECTION_OUT)
    accountant.observe(arp(PEER_MAC, BROADCAST, opcode=1, sender_ip=PEER_IP, target_ip=LOCAL_IP), 2.0, DIRECTION_OUT)
    accountant.observe(arp(LOCAL_MAC, PEER_MAC, opcode=2, sender_ip=LOCAL_IP, target_ip=PEER_IP, target_mac=PEER_MAC), 2.0, DIRECTION_IN)
    for frame, direction in request_and_response():
        accountant.observe(frame, 3.0, direction)
    snapshot = accountant.snapshot()
    assert snapshot.findings == 0, snapshot.finding_groups
    assert snapshot.complete
    assert snapshot.by_direction()[DIRECTION_OUT].frames == 5
    assert snapshot.by_direction()[DIRECTION_IN].frames == 2
    assert snapshot.by_direction()[DIRECTION_UNKNOWN].frames == 0


def test_replaying_the_clean_link_without_kernel_directions_also_raises_nothing():
    accountant = FrameAccountant(policy=LINK)
    accountant.observe(lldp(), 1.0)
    for frame, _direction in request_and_response():
        accountant.observe(frame, 3.0)
    snapshot = accountant.snapshot()
    assert snapshot.findings == 0
    assert snapshot.classes["ipv4-tcp"].frames_out == 1
    assert snapshot.classes["ipv4-tcp"].frames_in == 1


def test_without_a_policy_the_accountant_behaves_as_before():
    accountant = FrameAccountant()
    accountant.observe(ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 17, udp(5000, 9999, b"x" * 8)), 1.0)
    accountant.observe(lldp(src_mac=OTHER_MAC), 1.0)
    assert accountant.snapshot().findings == 0
    assert accountant.snapshot().classes["udp-9999"].frames_unknown == 1


# --- every way of departing from it ----------------------------------------


def test_a_class_the_tapped_node_may_not_emit_is_a_finding():
    accountant = FrameAccountant(policy=LINK)
    datagram = ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 17, udp(5000, 9999, b"x" * 8))
    accountant.observe(datagram, 1.0, DIRECTION_OUT)
    assert findings(accountant) == {"class-not-allowed": 1, "pin-undeclared": 1}


def test_the_whitelist_is_directional():
    """LLDP is allowed out of the tapped node and not into it."""
    accountant = FrameAccountant(policy=LINK)
    accountant.observe(lldp(src_mac=LOCAL_MAC), 1.0, DIRECTION_IN)
    assert "class-not-allowed" in findings(accountant)


def test_a_third_mac_on_a_point_to_point_link_is_a_finding():
    accountant = FrameAccountant(policy=LINK)
    accountant.observe(lldp(src_mac=OTHER_MAC), 1.0, DIRECTION_OUT)
    assert findings(accountant)["mac-unknown"] == 1
    accountant.observe(lldp(src_mac=OTHER_MAC), 2.0)  # replay: neither end
    assert findings(accountant)["mac-unknown"] == 2


def test_an_address_outside_the_link_is_a_finding():
    accountant = FrameAccountant(policy=LINK)
    accountant.observe(ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, OTHER_IP, 6, tcp(8000, 40000)), 1.0, DIRECTION_OUT)
    assert findings(accountant) == {"ip-outside-link": 1}


def test_a_connection_from_the_tapped_node_is_a_finding():
    """Only the server port is allowed: the tapped node answers, it never dials."""
    accountant = FrameAccountant(policy=LINK)
    accountant.observe(ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 6, tcp(51000, 22)), 1.0, DIRECTION_OUT)
    assert findings(accountant) == {"tcp-port-unexpected": 1}
    accountant.observe(ipv4(LOCAL_MAC, PEER_MAC, LOCAL_IP, PEER_IP, 6, tcp(51000, 22)), 1.0, DIRECTION_IN)
    assert findings(accountant) == {"tcp-port-unexpected": 2}


def test_arp_about_anything_but_the_two_link_addresses_is_a_finding():
    accountant = FrameAccountant(policy=LINK)
    probe = arp(PEER_MAC, BROADCAST, opcode=1, sender_ip=PEER_IP, target_ip=OTHER_IP)
    accountant.observe(probe, 1.0, DIRECTION_OUT)
    assert findings(accountant) == {"arp-address-unexpected": 1}


# --- declared pins ----------------------------------------------------------


def test_a_declared_pin_is_checked_on_the_first_frame_not_learned_from_it():
    accountant = FrameAccountant(policy=LINK)
    changed = LLDP_PAYLOAD[:-2] + struct.pack("!H", (4 << 9) | 1) + b"X" + b"\x00\x00"
    accountant.observe(lldp(payload=changed), 1.0, DIRECTION_OUT)
    assert findings(accountant) == {"pin-mismatch": 1}
    pin = accountant.snapshot().pins[f"{PEER_MAC.hex(':')}/lldp"]
    assert pin.declared == payload_digest(LLDP_PAYLOAD)
    assert pin.digest != pin.declared


def test_a_second_payload_under_a_declaration_is_one_finding_not_two():
    accountant = FrameAccountant(policy=LINK)
    accountant.observe(lldp(), 1.0, DIRECTION_OUT)
    changed = LLDP_PAYLOAD[:-2] + struct.pack("!H", (4 << 9) | 1) + b"X" + b"\x00\x00"
    accountant.observe(lldp(payload=changed), 2.0, DIRECTION_OUT)
    assert findings(accountant) == {"pin-mismatch": 1}


def test_a_pinned_flow_nobody_declared_is_a_finding_once():
    accountant = FrameAccountant(policy=LINK)
    beacon = eth(PEER_MAC, BROADCAST, 0x8899, bytes(32))
    for _ in range(10):
        accountant.observe(beacon, 1.0, DIRECTION_OUT)
    assert findings(accountant) == {"class-not-allowed": 10, "pin-undeclared": 1}


# --- configuration ----------------------------------------------------------


def test_declarations_are_keyed_the_way_the_accountant_keys_pins():
    declared = parse_declared_pins(
        "02:00:00:00:00:03/lldp=0123456789ABCDEF, garbage, 02-00-00-00-00-02/realtek=0011223344556677"
    )
    assert declared == {
        "02:00:00:00:00:03/lldp": "0123456789abcdef",
        "02:00:00:00:00:02/realtek": "0011223344556677",
    }


def test_beats_are_declared_with_the_same_grammar_as_pins():
    """One grammar for both, so a reader who has learned one has learned both."""
    assert parse_declared_beats(
        "02:00:00:00:00:01/arp=30, 02-00-00-00-00-02/arp=30.5"
    ) == {
        "02:00:00:00:00:01/arp": 30.0,
        "02:00:00:00:00:02/arp": 30.5,
    }


def test_a_cadence_that_is_not_a_positive_number_of_seconds_is_dropped():
    """Rather than guessed at: a zero or negative interval would either divide
    by zero or declare a beat that can never be late."""
    assert parse_declared_beats("aa:bb:cc:dd:ee:ff/arp=soon") == {}
    assert parse_declared_beats("aa:bb:cc:dd:ee:ff/arp=0") == {}
    assert parse_declared_beats("aa:bb:cc:dd:ee:ff/arp=-30") == {}


def test_the_policy_is_off_until_a_peer_is_named(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "FRAME_PROCESSOR_PEER_MAC", None)
    assert LinkPolicy.from_config() is None


def test_the_policy_reads_the_manifest_shaped_environment(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "FRAME_PROCESSOR_PEER_MAC", "02:00:00:00:00:03")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_LOCAL_MAC", None)
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PEER_IP", "192.0.2.2")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_LOCAL_IP", "192.0.2.1")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PEER_PORTS", "8000, 8001")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_ALLOW_OUT", "lldp,arp,ipv4-tcp")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_ALLOW_IN", "arp,ipv4-tcp")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_EXPECTED_PINS", "02:00:00:00:00:03/lldp=0123456789abcdef")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_IFACE_DIRECTIONS", {})
    monkeypatch.setattr(policy_module, "_sysfs_mac", lambda iface: LOCAL_MAC if iface == "mon2" else None)

    built = LinkPolicy.from_config(iface="mon2")
    assert built is not None
    assert built.peer_mac == PEER_MAC
    assert built.local_mac == LOCAL_MAC, "read from sysfs for the capture interface"
    assert built.peer_ip == PEER_IP and built.local_ip == LOCAL_IP
    assert built.peer_ports == frozenset({8000, 8001})
    assert built.allow_out == frozenset({"lldp", "arp", "ipv4-tcp"})
    assert built.allow_in == frozenset({"arp", "ipv4-tcp"})
    assert built.declared_digest("02:00:00:00:00:03/lldp") == "0123456789abcdef"
    assert "peer=02:00:00:00:00:03/192.0.2.2" in built.describe()


def test_a_monitor_port_never_lends_its_own_mac_as_the_local_one(monkeypatch):
    """A monitor port's address is not this host's address on the tapped link;
    reading it would make every frame this host sent look like a stranger's."""
    from app import config

    monitor_mac = bytes.fromhex("0000aabbccdd")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PEER_MAC", "02:00:00:00:00:01")
    monkeypatch.setattr(config, "FRAME_PROCESSOR_LOCAL_MAC", None)
    monkeypatch.setattr(config, "FRAME_PROCESSOR_IFACE_DIRECTIONS", {"mon0": "in", "mon1": "out"})
    monkeypatch.setattr(policy_module, "_sysfs_mac", lambda iface: monitor_mac)

    built = LinkPolicy.from_config(iface="mon0")
    assert built is not None
    assert built.local_mac is None, "unset rather than wrong"

    monkeypatch.setattr(config, "FRAME_PROCESSOR_LOCAL_MAC", "02:00:00:00:00:02")
    assert LinkPolicy.from_config(iface="mon0").local_mac == LOCAL_MAC


@pytest.mark.parametrize("value", ["", "   ", None])
def test_blank_settings_mean_unset(value):
    assert policy_module._mac(value) is None
    assert policy_module._ip(value) is None
    assert policy_module._csv(value) == frozenset()
    assert policy_module._ports(value) == frozenset()


# --- dial-out ---------------------------------------------------------------
#
# The port rule catches a connection whose ends are wrong, but a node dialling
# out FROM its own service port satisfies it while still being the initiator.
# A bare SYN is the only frame that says who opened the connection, and the
# tapped node only ever serves.


def test_a_syn_from_the_tapped_node_is_a_dial_out_finding():
    accountant = FrameAccountant(policy=LINK)
    syn = ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 6, tcp(8000, 9999, flags=0x02))
    accountant.observe(syn, 1.0, DIRECTION_OUT)
    assert findings(accountant).get("tcp-dial-out") == 1


def test_a_syn_ack_from_the_tapped_node_is_normal_serving():
    """Answering a connection is exactly what it is there to do."""
    accountant = FrameAccountant(policy=LINK)
    syn_ack = ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 6, tcp(8000, 40000, flags=0x12))
    accountant.observe(syn_ack, 1.0, DIRECTION_OUT)
    assert "tcp-dial-out" not in findings(accountant)


def test_mid_stream_data_from_the_tapped_node_is_not_a_dial_out():
    accountant = FrameAccountant(policy=LINK)
    data = ipv4(PEER_MAC, LOCAL_MAC, PEER_IP, LOCAL_IP, 6, tcp(8000, 40000, b"hello"))
    accountant.observe(data, 1.0, DIRECTION_OUT)
    assert "tcp-dial-out" not in findings(accountant)


def test_a_syn_towards_the_tapped_node_is_not_a_dial_out():
    """This host opening a connection to it is the normal case."""
    accountant = FrameAccountant(policy=LINK)
    syn = ipv4(LOCAL_MAC, PEER_MAC, LOCAL_IP, PEER_IP, 6, tcp(40000, 8000, flags=0x02))
    accountant.observe(syn, 1.0, DIRECTION_IN)
    assert "tcp-dial-out" not in findings(accountant)
