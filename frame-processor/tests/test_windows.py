"""Capture windows: the account cut into rows, and the rows merged per link.

The property under test is that a window row means what a verifier will read
it to mean. A window with zero frames exists; a window an interface never
reported is incomplete; a finding lands in the window its frame fell in; a
process that started or stopped mid-window says so; and none of the caps
that keep a row bounded can make a dirty window read as clean.
"""

from __future__ import annotations

import base64
import struct

from app.accounting import FrameAccountant
from app.policy import DIRECTION_IN, DIRECTION_OUT, LinkPolicy
from app.windows import (
    CLASS_EXCHANGE,
    KIND_CAPTURE_GAP,
    ExchangeFinding,
    WindowMerger,
    WindowReport,
    window_start_for,
)

PEER = bytes.fromhex("020000000003")
LOCAL = bytes.fromhex("020000000002")
W = 300.0
T = 1_700_000_100.0  # a window boundary: 5666667 * 300


def eth(src: bytes, ethertype: int, payload: bytes) -> bytes:
    return bytes(6) + src + struct.pack("!H", ethertype) + payload


def beacon(src: bytes = PEER, body: bytes = bytes(32)) -> bytes:
    return eth(src, 0x8899, body)  # realtek: pinned class


def accountant(**kw) -> FrameAccountant:
    kw.setdefault("iface", "tap0")
    kw.setdefault("window_seconds", W)
    kw.setdefault("process_epoch", T)
    return FrameAccountant(**kw)


# --- cutting -------------------------------------------------------------------


def test_window_starts_are_aligned():
    assert window_start_for(T + 17, W) == T
    assert window_start_for(T + 299.9, W) == T
    assert window_start_for(T + 300, W) == T + 300


def test_frames_are_counted_in_the_window_their_timestamp_falls_in():
    acct = accountant()
    acct.observe(beacon(), T + 1)
    acct.observe(beacon(), T + 2)
    acct.observe(beacon(), T + 301)  # next window: closes the first
    reports = acct.take_closed()
    assert len(reports) == 1
    first = reports[0]
    assert (first.window_start, first.window_end) == (T, T + W)
    assert first.observed == first.classified == 2
    assert first.classes["realtek"]["frames"] == 2
    assert first.complete
    # The cumulative account is untouched by windowing.
    assert acct.snapshot().observed == 3


def test_an_idle_window_still_closes_with_zero_frames():
    """The heartbeat: a row for a window nothing crossed is what separates
    "the link was quiet" from "the tap was down"."""
    acct = accountant()
    acct.observe(beacon(), T + 1)
    reports = acct.roll(T + 2 * W + 10)  # wall clock two windows on, past grace
    assert [(r.window_start, r.observed) for r in reports] == [(T, 1), (T + W, 0)]
    assert reports[1].complete


def test_roll_respects_the_grace_period():
    acct = accountant(window_grace=5.0)
    acct.observe(beacon(), T + 1)
    assert acct.roll(T + W + 4) == [], "not yet: a late frame could still belong here"
    assert len(acct.roll(T + W + 5)) == 1


def test_a_late_frame_lands_in_the_open_window_rather_than_reopening_a_closed_one():
    acct = accountant()
    acct.observe(beacon(), T + 1)
    acct.roll(T + W + 10)
    acct.observe(beacon(), T + 299)  # stamped in the closed window
    reports = acct.finish(T + W + 20)
    assert reports[0].observed == 1
    assert reports[0].window_start == T + W


def test_a_clock_jump_does_not_mint_hundreds_of_empty_rows():
    acct = accountant()
    acct.observe(beacon(), T + 1)
    acct.observe(beacon(), T + 1000 * W)  # a replayed capture, or a stepped clock
    reports = acct.take_closed()
    assert len(reports) == 1, "the windows between are left missing, not invented"
    assert acct.finish(T + 1000 * W + 1)[0].window_start == T + 1000 * W


# --- what a window says about itself --------------------------------------------


def test_findings_land_in_their_window_grouped_by_kind_class_direction_and_sender():
    acct = accountant()
    acct.observe(beacon(), T + 1, DIRECTION_OUT)
    acct.observe(beacon(body=bytes(31) + b"\x01"), T + 2, DIRECTION_OUT)  # variant → finding
    acct.observe(beacon(body=bytes(31) + b"\x02"), T + 3, DIRECTION_OUT)
    acct.observe(beacon(), T + 400, DIRECTION_OUT)  # clean, next window
    first, = acct.take_closed()
    assert first.finding_count == 2
    assert not first.complete
    assert len(first.groups) == 1
    group = first.groups[0]
    assert group.key == ("variant-added", "realtek", "out", PEER.hex(":"))
    assert group.count == 2
    assert (group.first_ts, group.last_ts) == (T + 2, T + 3)
    second = acct.roll(T + 2 * W + 10)[0]
    assert second.finding_count == 0 and second.complete


def test_samples_keep_the_offending_frames_bytes_bounded():
    acct = accountant(sample_frame_bytes=20, samples_per_group=2)
    for i in range(5):
        acct.observe(beacon(body=bytes(31) + bytes([i])), T + i)
    report = acct.finish(T + 10)[0]
    group = report.groups[0]
    assert group.count == 4  # first payload establishes the pin
    assert len(group.samples) == 2
    assert all(len(sample.frame) == 20 for sample in group.samples)
    assert group.samples[0].frame == beacon(body=bytes(31) + b"\x01")[:20]


def test_the_group_cap_taints_the_window_and_counts_what_it_did_not_keep():
    """Unknown UDP mints a class per port; variety itself has to be bounded."""
    acct = accountant(max_window_groups=3, policy=LinkPolicy(peer_mac=PEER, allow_out=frozenset()))

    def udp_frame(port: int) -> bytes:
        header = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 28, 1, 0, 64, 17, 0, bytes(4), bytes(4))
        return eth(PEER, 0x0800, header + struct.pack("!HHHH", 5000, port, 8, 0))

    for port in range(9000, 9010):
        acct.observe(udp_frame(port), T + 1, DIRECTION_OUT)
    report = acct.roll(T + W + 10)[0]
    assert len(report.groups) == 3
    assert report.groups_overflow > 0
    # Each frame is a class nobody allowed AND a pinned flow nobody declared.
    assert report.finding_count == 20, "every finding is counted, capped or not"
    assert not report.complete


def test_kernel_drops_taint_the_open_window():
    acct = accountant()
    acct.observe(beacon(), T + 1)
    acct.note_kernel_dropped(4)
    acct.observe(beacon(), T + W + 1)
    first, = acct.take_closed()
    assert first.kernel_dropped == 4 and not first.complete
    assert acct.finish(T + W + 2)[0].kernel_dropped == 0


def test_truncated_frames_are_on_the_window_too():
    acct = accountant()
    acct.note_truncated(9000, 2048, T + 1)
    report = acct.roll(T + W + 10)[0]
    assert report.truncated == 1 and report.observed == 1 and report.finding_count == 1


def test_a_process_that_starts_mid_window_says_so():
    acct = accountant(process_epoch=T + 120)
    acct.observe(beacon(), T + 130)
    report = acct.finish(T + 140)[0]
    gap, = report.groups  # both gaps share a key: one group, two samples
    assert gap.kind == KIND_CAPTURE_GAP and gap.count == 2
    details = sorted(sample.detail for sample in gap.samples)
    assert details[0].startswith("capture started 120s into the window")
    assert details[1].startswith("capture stopped 160s before the window ended")
    assert not report.complete


def test_a_window_the_clock_has_finished_is_not_cut_short_by_shutdown():
    acct = accountant()
    acct.observe(beacon(), T + 1)
    report = acct.finish(T + W + 1)[0]
    assert report.finding_count == 0 and report.complete


def test_a_process_that_starts_on_the_boundary_is_not_a_gap():
    acct = accountant(process_epoch=T)
    acct.observe(beacon(), T + 1)
    acct.observe(beacon(), T + W + 1)
    first, = acct.take_closed()
    assert first.finding_count == 0


def test_pins_travel_on_every_window_with_their_payload():
    acct = accountant()
    acct.observe(beacon(), T + 1)
    report = acct.finish(T + 2)[0]
    pin = report.pins[f"{PEER.hex(':')}/realtek"]
    assert pin["frames"] == 1 and pin["distinct"] == 1
    assert base64.b64decode(pin["payload_b64"]) == bytes(32)


def test_without_window_seconds_nothing_is_cut():
    acct = FrameAccountant()
    acct.observe(beacon(), T + 1)
    assert not acct.windowed
    assert acct.roll(T + 10 * W) == []
    assert acct.finish(T + 10 * W) == []


# --- merging in the parent ------------------------------------------------------


def merger(**kw) -> WindowMerger:
    kw.setdefault("ifaces", ["mon0", "mon1"])
    kw.setdefault("window_seconds", W)
    kw.setdefault("flush_delay", 15.0)
    kw.setdefault("capture_host", "gpu-node-1")
    kw.setdefault("tapped_hostname", "kserve-gpt-oss-120b")
    kw.setdefault("tap_version", "0.6.0")
    return WindowMerger(**kw)


def report(iface: str, start: float = T, **kw) -> WindowReport:
    kw.setdefault("observed", 3)
    kw.setdefault("classified", 3)
    return WindowReport(iface=iface, window_start=start, window_end=start + W, process_epoch=T, **kw)


def test_two_monitor_ports_become_one_row_about_the_link():
    m = merger()
    m.add_report(report("mon0", classes={"ipv4-tcp": {"frames": 3, "bytes": 300, "out": 0, "in": 3, "unknown": 0}}))
    m.add_report(report("mon1", classes={"ipv4-tcp": {"frames": 3, "bytes": 900, "out": 3, "in": 0, "unknown": 0}}))
    assert m.flush(T + W + 14) == [], "not before the flush delay"
    rows = m.flush(T + W + 15)
    assert len(rows) == 1
    row = rows[0]
    assert row["ifaces"] == "mon0,mon1"
    assert row["observed"] == 6
    assert row["classes"]["ipv4-tcp"] == {"frames": 6, "bytes": 1200, "out": 3, "in": 3, "unknown": 0}
    assert row["complete"] is True
    assert row["window_start"] == "2023-11-14T22:15:00+00:00"
    assert row["tapped_hostname"] == "kserve-gpt-oss-120b"
    assert row["findings"] == []


def test_an_interface_that_never_reports_is_the_finding():
    m = merger()
    m.add_report(report("mon0"))
    row, = m.flush(T + W + 15)
    assert row["complete"] is False
    assert row["finding_count"] == 1
    assert row["findings"][0]["kind"] == KIND_CAPTURE_GAP
    assert "mon1" in row["findings"][0]["samples"][0]["detail"]


def test_exchange_findings_are_filed_on_the_window_their_timestamp_falls_in():
    m = merger()
    m.add_report(report("mon0"))
    m.add_report(report("mon1"))
    m.add_exchange_finding(ExchangeFinding("unexpected-exchange", "GET /admin", T + 100, "192.0.2.2:8000"))
    m.add_exchange_finding(ExchangeFinding("unexpected-exchange", "GET /admin", T + 200, "192.0.2.2:8000"))
    row, = m.flush(T + W + 15)
    assert row["complete"] is False
    assert row["finding_count"] == 2
    group, = row["findings"]
    assert group["frame_class"] == CLASS_EXCHANGE
    assert group["count"] == 2
    assert group["samples"][0]["detail"] == "192.0.2.2:8000: GET /admin"


def test_a_finding_that_arrives_after_its_window_was_written_is_not_lost():
    m = merger(ifaces=["tap0"])
    m.add_report(report("tap0"))
    m.flush(T + W + 15)
    late = ExchangeFinding("non-http-stream", "garbage", T + 50, "192.0.2.2:8000")
    m.add_exchange_finding(late, now=T + W + 20)
    m.add_report(report("tap0", start=T + W))
    row, = m.flush(T + 2 * W + 15)
    assert row["window_start"] == "2023-11-14T22:20:00+00:00"
    assert row["finding_count"] == 1, "filed on the next window rather than dropped"
    assert row["findings"][0]["first_ts"] == "2023-11-14T22:15:50+00:00", "with its real time"


def test_finish_writes_whatever_is_pending():
    m = merger(ifaces=["tap0"])
    m.add_report(report("tap0"))
    m.add_report(report("tap0", start=T + W))
    rows = m.finish()
    assert [r["window_start"][11:16] for r in rows] == ["22:15", "22:20"]
    assert m.windows_flushed == 2


def test_a_report_for_a_window_already_written_is_refused_loudly(caplog):
    m = merger(ifaces=["tap0"])
    m.add_report(report("tap0"))
    m.flush(T + W + 15)
    with caplog.at_level("ERROR"):
        m.add_report(report("tap0"))
    assert "after it was flushed" in caplog.text
    assert m.finish() == []


def test_the_merged_row_caps_groups_across_sources():
    m = merger(ifaces=["tap0"], max_groups=2)
    for i in range(5):
        m.add_exchange_finding(ExchangeFinding(f"kind-{i}", "x", T + i, "a:1"))
    m.add_report(report("tap0"))
    row, = m.finish()
    assert row["finding_groups"] == 2
    assert row["finding_groups_overflow"] == 3
    assert row["finding_count"] == 5
