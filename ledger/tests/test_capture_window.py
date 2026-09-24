"""Capture windows (sql/004): frame-processor's account of the tapped link as rows,
and the capture status each inference event derives from them."""

from __future__ import annotations

import base64
import hashlib
import uuid

T0 = "2026-09-02T10:00:00Z"
T1 = "2026-09-02T10:05:00Z"
T2 = "2026-09-02T10:10:00Z"
T3 = "2026-09-02T10:15:00Z"

HOST = "gpu-node-1"
IFACES = "mon0,mon1"
TAPPED = "kserve-gpt-oss-120b"


def _window(start: str, end: str, *, complete: bool = True, findings=(), **overrides) -> dict:
    body = {
        "capture_host": HOST,
        "ifaces": IFACES,
        "tapped_hostname": TAPPED,
        "window_start": start,
        "window_end": end,
        "tap_version": "0.6.0",
        "process_epoch": "2026-09-02T09:00:00Z",
        "observed": 12,
        "classified": 12,
        "kernel_dropped": 0,
        "truncated": 0,
        "errors": 0,
        "finding_count": sum(f["count"] for f in findings),
        "finding_groups": len(findings),
        "finding_groups_overflow": 0,
        "complete": complete,
        "classes": {"lldp": {"frames": 10, "bytes": 1980, "out": 10, "in": 0, "unknown": 0}},
        "pins": {
            "02:00:00:00:00:01/lldp": {
                "digest": "3ba2cbea96d953c8",
                "declared": "3ba2cbea96d953c8",
                "frames": 10,
                "distinct": 1,
                "payload_b64": base64.b64encode(b"\x02\x07\x04" + bytes(6)).decode(),
            }
        },
        "findings": list(findings),
    }
    body.update(overrides)
    return body


def _finding(kind="class-not-allowed", frame_class="udp-9999", count=3, direction="out"):
    return {
        "kind": kind,
        "frame_class": frame_class,
        "direction": direction,
        "source_mac": "02:00:00:00:00:01",
        "count": count,
        "first_ts": "2026-09-02T10:01:00Z",
        "last_ts": "2026-09-02T10:04:00Z",
        "samples": [{"ts": "2026-09-02T10:01:00Z", "detail": "udp-9999 is not allowed out", "frame_b64": "AAECAw=="}],
    }


async def _declare(client, hostname=TAPPED):
    r = await client.post(
        "/model-deployments/declare",
        json={"hostname": hostname, "model_name": "openai/gpt-oss-120b", "seed": 42, "started_at": "2026-09-01T00:00:00Z"},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _inference(client, deployment, *, started_at: str | None, ts: str) -> str:
    digest = base64.b64encode(hashlib.sha256(ts.encode()).digest()).decode()
    body = {
        "id": str(uuid.uuid4()),
        "session_id": "chat-1",
        "ts": ts,
        "started_at": started_at,
        "model_id": deployment["model_id"],
        "hardware_id": deployment["hardware_id"],
        "hash_input_raw_logits": digest,
        "hash_output_raw_logits": digest,
    }
    r = await client.post("/inference-events", json=body)
    assert r.status_code == 201, r.text
    return body["id"]


# --- writing -----------------------------------------------------------------


async def test_a_window_is_recorded_with_its_findings_and_resolved_to_the_tapped_node(client):
    deployment = await _declare(client)
    r = await client.post("/capture-windows", json=_window(T0, T1, complete=False, findings=[_finding()]))
    assert r.status_code == 201, r.text
    window = r.json()
    assert window["hardware_id"] == deployment["hardware_id"], "joined on the declared hostname"
    assert window["complete"] is False
    assert window["finding_count"] == 3

    detail = (await client.get(f"/capture-windows/{window['window_id']}")).json()
    assert len(detail["findings"]) == 1
    finding = detail["findings"][0]
    assert finding["kind"] == "class-not-allowed"
    assert finding["direction"] == "out"
    assert finding["samples"][0]["frame_b64"] == "AAECAw=="

    listed = (await client.get("/capture-findings?kind=class-not-allowed")).json()
    assert [f["finding_id"] for f in listed] == [finding["finding_id"]]
    assert (await client.get("/capture-findings?kind=pin-mismatch")).json() == []


async def test_the_same_window_cannot_be_recorded_twice(client):
    first = await client.post("/capture-windows", json=_window(T0, T1))
    assert first.status_code == 201
    second = await client.post("/capture-windows", json=_window(T0, T1, observed=99))
    assert second.status_code == 409, "a retry after a crash must not double-record"
    page = (await client.get("/capture-windows")).json()
    assert page["total"] == 1
    assert page["items"][0]["observed"] == 12, "the first record stands"


async def test_an_unknown_tapped_hostname_is_kept_but_unjoined(client):
    r = await client.post("/capture-windows", json=_window(T0, T1, tapped_hostname="nobody-declared-this"))
    assert r.status_code == 201, r.text
    assert r.json()["hardware_id"] is None


async def test_a_window_with_no_frames_is_still_a_row(client):
    """The heartbeat: an idle window is evidence the tap was watching."""
    r = await client.post("/capture-windows", json=_window(T0, T1, observed=0, classified=0, classes={}, pins={}))
    assert r.status_code == 201, r.text
    assert r.json()["complete"] is True


async def test_a_window_that_ends_before_it_starts_is_rejected(client):
    r = await client.post("/capture-windows", json=_window(T1, T0))
    assert r.status_code == 422


# --- reading -----------------------------------------------------------------


async def test_latest_and_listing_are_newest_first(client):
    for start, end in ((T0, T1), (T2, T3), (T1, T2)):
        assert (await client.post("/capture-windows", json=_window(start, end))).status_code == 201
    latest = (await client.get(f"/capture-windows/latest?capture_host={HOST}")).json()
    assert latest["window_start"].startswith("2026-09-02T10:10")
    page = (await client.get("/capture-windows?limit=2")).json()
    assert page["total"] == 3
    assert [w["window_start"][11:16] for w in page["items"]] == ["10:10", "10:05"]
    assert (await client.get("/capture-windows/latest?capture_host=elsewhere")).status_code == 404
    incomplete = (await client.get("/capture-windows?complete=false")).json()
    assert incomplete["total"] == 0


# --- capture status per inference ---------------------------------------------


async def test_capture_status_names_each_way_an_inference_can_be_covered(client):
    deployment = await _declare(client)

    complete_a = await _inference(client, deployment, started_at="2026-09-02T10:01:00Z", ts="2026-09-02T10:02:00Z")
    complete_b = await _inference(client, deployment, started_at="2026-09-02T10:04:00Z", ts="2026-09-02T10:06:00Z")  # spans two windows
    tainted = await _inference(client, deployment, started_at="2026-09-02T10:11:00Z", ts="2026-09-02T10:12:00Z")
    partial = await _inference(client, deployment, started_at="2026-09-02T10:14:00Z", ts="2026-09-02T10:16:00Z")  # runs past the last window
    uncovered = await _inference(client, deployment, started_at="2026-09-02T11:00:00Z", ts="2026-09-02T11:01:00Z")
    no_start = await _inference(client, deployment, started_at=None, ts="2026-09-02T10:03:00Z")  # zero-length span

    for body in (
        _window(T0, T1),
        _window(T1, T2),
        _window(T2, T3, complete=False, findings=[_finding(count=2)], kernel_dropped=7),
    ):
        assert (await client.post("/capture-windows", json=body)).status_code == 201

    async def status(event_id: str) -> dict:
        r = await client.get(f"/inference-events/{event_id}/capture")
        assert r.status_code == 200, r.text
        return r.json()

    assert (await status(complete_a))["capture_status"] == "complete"
    two = await status(complete_b)
    assert two["capture_status"] == "complete"
    assert two["windows"] == 2
    assert two["covered_seconds"] == two["span_seconds"] == 120.0

    bad = await status(tainted)
    assert bad["capture_status"] == "tainted"
    assert bad["kernel_dropped"] == 7
    assert bad["findings"] == 2

    cut = await status(partial)
    assert cut["capture_status"] == "partial"
    assert cut["covered_seconds"] == 60.0 and cut["span_seconds"] == 120.0

    assert (await status(uncovered))["capture_status"] == "uncovered"
    assert (await status(no_start))["capture_status"] == "complete"

    assert (await client.get(f"/inference-events/{uuid.uuid4()}/capture")).status_code == 404


async def test_the_verification_view_carries_capture_status_for_its_page(client):
    """The UI's Capture pill reads this: one join, not one call per row."""
    deployment = await _declare(client)
    event = await _inference(client, deployment, started_at="2026-09-02T10:01:00Z", ts="2026-09-02T10:02:00Z")
    r = await client.post(
        "/verification-events",
        json={
            "id": str(uuid.uuid4()),
            "inference_event_id": event,
            "hardware_id": deployment["hardware_id"],
            "ts": "2026-09-02T10:30:00Z",
            "result": "pass",
            "verifier_model_id": deployment["model_id"],
        },
    )
    assert r.status_code == 201, r.text

    view = (await client.get("/verification-events/view")).json()
    assert view["items"][0]["capture_status"] == "uncovered"
    assert view["items"][0]["capture_findings"] == 0

    assert (
        await client.post("/capture-windows", json=_window(T0, T1, complete=False, findings=[_finding(count=2)]))
    ).status_code == 201
    view = (await client.get("/verification-events/view")).json()
    assert view["items"][0]["capture_status"] == "tainted"
    assert view["items"][0]["capture_findings"] == 2
    assert view["items"][0]["result"] == "pass", "the verdict is untouched by the link"
    # The inference's own span rides along, because capture status follows
    # when the inference ran, not when it was verified.
    assert view["items"][0]["inference_started_at"].startswith("2026-09-02T10:01:00")
    assert view["items"][0]["inference_ts"].startswith("2026-09-02T10:02:00")


async def test_since_and_until_select_the_windows_overlapping_a_span(client):
    for start, end in ((T0, T1), (T1, T2), (T2, T3)):
        assert (await client.post("/capture-windows", json=_window(start, end))).status_code == 201
    # An inference from 10:04 to 10:06 touches the first two windows only.
    page = (await client.get("/capture-windows?since=2026-09-02T10:04:00Z&until=2026-09-02T10:06:00Z")).json()
    assert [w["window_start"][11:16] for w in page["items"]] == ["10:05", "10:00"]
    assert page["total"] == 2
    # A zero-length span inside one window selects that window alone.
    page = (await client.get("/capture-windows?since=2026-09-02T10:12:00Z&until=2026-09-02T10:12:00Z")).json()
    assert [w["window_start"][11:16] for w in page["items"]] == ["10:10"]


async def test_windows_for_another_host_do_not_cover_this_inference(client):
    deployment = await _declare(client)
    await _declare(client, hostname="some-other-node")
    event = await _inference(client, deployment, started_at="2026-09-02T10:01:00Z", ts="2026-09-02T10:02:00Z")
    assert (
        await client.post("/capture-windows", json=_window(T0, T1, tapped_hostname="some-other-node"))
    ).status_code == 201
    r = await client.get(f"/inference-events/{event}/capture")
    assert r.json()["capture_status"] == "uncovered"
