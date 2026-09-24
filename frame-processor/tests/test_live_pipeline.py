"""Process topology and overload semantics for the live capture path."""

from __future__ import annotations

import dataclasses
import multiprocessing
import queue
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.capture import TcpSegment
from app.live_pipeline import (
    FlowWorkerStats,
    LivePipeline,
    ReceiverStats,
    _receiver_main,
    _route_segment,
    _worker_main,
    flow_shard,
)
from tests.helpers import CLIENT, SERVER, conversation, http_request, http_response_json


def _segment(
    *,
    src: tuple[str, int] = CLIENT,
    dst: tuple[str, int] = SERVER,
    seq: int = 1,
    payload: bytes = b"data",
    ack: bool = True,
) -> TcpSegment:
    return TcpSegment(
        ts=1700000000.0,
        src=src[0],
        sport=src[1],
        dst=dst[0],
        dport=dst[1],
        seq=seq,
        syn=False,
        ack=ack,
        fin=False,
        rst=False,
        payload=payload,
    )


def _reverse(segment: TcpSegment) -> TcpSegment:
    return dataclasses.replace(
        segment,
        src=segment.dst,
        sport=segment.dport,
        dst=segment.src,
        dport=segment.sport,
    )


def _counters() -> dict[str, int]:
    return {
        "segments_seen": 0,
        "excluded": 0,
        "ack_only": 0,
        "dispatched": 0,
        "queue_dropped": 0,
        "errors": 0,
    }


def test_symmetric_flow_hash_is_direction_independent_and_stable():
    forward = _segment()
    reverse = _reverse(forward)

    assert flow_shard(forward, 17) == flow_shard(reverse, 17)
    # A golden value: 17 is a shard count no default would produce by accident,
    # and 2 is what BLAKE2s gives for this endpoint pair. It catches accidental
    # replacement with Python's process-randomised hash(), which agrees with
    # itself inside one process and so would pass the symmetry check above
    # while splitting the two directions across spawned receivers.
    assert flow_shard(forward, 17) == 2


def test_both_directions_are_routed_to_the_same_worker():
    worker_queues = [queue.Queue() for _ in range(7)]
    counters = _counters()
    forward = _segment()

    _route_segment("tap0", forward, worker_queues, None, frozenset(), counters)
    _route_segment("tap1", _reverse(forward), worker_queues, None, frozenset(), counters)

    occupied = [index for index, work in enumerate(worker_queues) if not work.empty()]
    assert occupied == [flow_shard(forward, len(worker_queues))]
    assert worker_queues[occupied[0]].get_nowait() == forward
    assert worker_queues[occupied[0]].get_nowait() == _reverse(forward)
    assert counters["dispatched"] == 2


def test_full_segment_queue_drops_without_blocking_and_counts_loss():
    work = queue.Queue(maxsize=1)
    counters = _counters()

    _route_segment("tap0", _segment(seq=1), [work], None, frozenset(), counters)
    _route_segment("tap0", _segment(seq=2), [work], None, frozenset(), counters)

    assert counters["segments_seen"] == 2
    assert counters["dispatched"] == 1
    assert counters["queue_dropped"] == 1
    assert work.get_nowait().seq == 1


class _StaticCapture:
    def __init__(self, segments: tuple[TcpSegment, ...]) -> None:
        self._segments = segments
        self._read = False
        self.closed = False
        self.received = len(segments)
        self.drops = 3
        self.freeze_q_count = 2
        self.truncated_packets = 1
        self.malformed_blocks = 0

    def read_segments(
        self, limit: int | None = None, *, max_blocks: int = 8
    ) -> list[TcpSegment] | None:
        del limit, max_blocks
        if self._read:
            return None
        self._read = True
        return list(self._segments)

    def poll_stats(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _StaticCaptureFactory:
    segments: tuple[TcpSegment, ...]

    def __call__(self, _iface: str) -> _StaticCapture:
        return _StaticCapture(self.segments)


@dataclass(frozen=True)
class _SplitCaptureFactory:
    client_segments: tuple[TcpSegment, ...]
    server_segments: tuple[TcpSegment, ...]

    def __call__(self, iface: str) -> _StaticCapture:
        chosen = self.client_segments if iface == "tap-request" else self.server_segments
        return _StaticCapture(chosen)


class _NeverStopped:
    @staticmethod
    def is_set() -> bool:
        return False


class _FakeProcess:
    def __init__(self, context, *, name: str, **_kwargs) -> None:
        self._context = context
        self.name = name
        self.exitcode = None
        self._alive = False

    def start(self) -> None:
        self._context.starts += 1
        if self._context.starts == self._context.fail_on_start:
            raise RuntimeError("process start failed")
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False
        # How multiprocessing reports a child killed by a signal.
        self.exitcode = -signal.SIGTERM

    def join(self, timeout=None) -> None:
        del timeout


class _FailingStartContext:
    def __init__(self, fail_on_start: int) -> None:
        self.fail_on_start = fail_on_start
        self.starts = 0
        self.processes: list[_FakeProcess] = []

    @staticmethod
    def Event():
        return threading.Event()

    @staticmethod
    def Queue(maxsize: int = 0):
        return queue.Queue(maxsize=maxsize)

    def Process(self, **kwargs):
        process = _FakeProcess(self, **kwargs)
        self.processes.append(process)
        return process


def test_partial_process_start_is_rolled_back_and_stop_does_not_restart():
    context = _FailingStartContext(fail_on_start=2)
    pipeline = LivePipeline(
        ifaces=("tap0",),
        emitter=_CollectingEmitter(),
        worker_count=2,
        mp_context=context,
    )

    try:
        pipeline.start()
    except RuntimeError as exc:
        assert str(exc) == "process start failed"
    else:
        raise AssertionError("start should propagate the child startup failure")

    assert context.starts == 2
    assert all(not process.is_alive() for process in context.processes)
    pipeline.stop()  # idempotent cleanup, not a second startup attempt
    assert context.starts == 2


def test_receiver_reports_overflow_and_capture_counters():
    segments = (_segment(seq=1), _segment(seq=2), _segment(seq=3))
    work = queue.Queue(maxsize=1)
    reports: queue.Queue[Any] = queue.Queue()

    _receiver_main(
        "tap0",
        [work],
        reports,
        _NeverStopped(),
        _StaticCaptureFactory(segments),
        None,
        frozenset(),
        0.01,
        30.0,
    )

    report = reports.get_nowait()
    assert isinstance(report, ReceiverStats)
    assert report.segments_seen == 3
    assert report.dispatched == 1
    assert report.queue_dropped == 2
    assert report.kernel_received == 3
    assert report.kernel_dropped == 3
    assert report.kernel_freezes == 2
    assert report.truncated_packets == 1


class _RecordingProcessor:
    def __init__(self) -> None:
        self.fed: list[int] = []
        self.finished = False

    def feed(self, segment: TcpSegment):
        self.fed.append(segment.seq)
        yield {"seq": segment.seq}

    def sweep(self, _now: float):
        return iter(())

    def finish(self):
        self.finished = True
        yield {"finished": True}


def test_worker_sentinel_drains_accepted_segments_before_finish():
    segments: queue.Queue[Any] = queue.Queue()
    messages: queue.Queue[Any] = queue.Queue(maxsize=8)
    reports: queue.Queue[Any] = queue.Queue()
    processor = _RecordingProcessor()
    for seq in (1, 2, 3):
        segments.put(_segment(seq=seq))
    segments.put(None)

    _worker_main(0, segments, messages, reports, lambda: processor, 0.01)

    assert processor.fed == [1, 2, 3]
    assert processor.finished is True
    assert [messages.get_nowait() for _ in range(4)] == [
        {"seq": 1},
        {"seq": 2},
        {"seq": 3},
        {"finished": True},
    ]
    report = reports.get_nowait()
    assert report == FlowWorkerStats(worker_id=0, segments_processed=3, messages_built=4)


def test_worker_output_overflow_is_explicit():
    segments: queue.Queue[Any] = queue.Queue()
    messages: queue.Queue[Any] = queue.Queue(maxsize=1)
    reports: queue.Queue[Any] = queue.Queue()
    for seq in (1, 2):
        segments.put(_segment(seq=seq))
    segments.put(None)

    _worker_main(0, segments, messages, reports, _RecordingProcessor, 0.01)

    report = reports.get_nowait()
    assert report.messages_built == 3  # two feeds plus finish
    assert report.queue_dropped == 2
    assert messages.get_nowait() == {"seq": 1}


class _CollectingEmitter:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def submit(self, message: dict[str, Any]) -> bool:
        self.messages.append(message)
        return True


class _FailingCaptureFactory:
    def __call__(self, _iface: str):
        raise RuntimeError("capture startup failed")


class _FailingProcessor:
    def feed(self, _segment: TcpSegment):
        raise RuntimeError("flow processing failed")

    def sweep(self, _now: float):
        return iter(())

    def finish(self):
        return iter(())


def test_receiver_failure_stops_the_pipeline_instead_of_running_half_duplex():
    pipeline = LivePipeline(
        ifaces=("tap0", "tap1"),
        emitter=_CollectingEmitter(),
        worker_count=1,
        capture_factory=_FailingCaptureFactory(),
        mp_context=multiprocessing.get_context("spawn"),
    )

    stats = pipeline.run()

    assert stats.receiver_process_failures == 2
    assert stats.receiver_errors == 2
    assert all(not child.is_alive() for child in pipeline.receivers + pipeline.workers)


def test_flow_worker_failure_stops_the_pipeline_and_is_reported():
    pipeline = LivePipeline(
        ifaces=("tap0",),
        emitter=_CollectingEmitter(),
        worker_count=1,
        capture_factory=_StaticCaptureFactory((_segment(),)),
        processor_factory=_FailingProcessor,
        mp_context=multiprocessing.get_context("spawn"),
    )

    stats = pipeline.run()

    assert stats.worker_process_failures == 1
    assert stats.worker_errors == 1
    assert all(not child.is_alive() for child in pipeline.receivers + pipeline.workers)


def test_spawned_pipeline_gracefully_drains_a_complete_exchange():
    request_body = b'{"model":"openai/gpt-oss-120b","prompt":"hello"}'
    response_body = (
        b'{"id":"cmpl-process","model":"openai/gpt-oss-120b",'
        b'"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}'
    )
    segments = tuple(
        conversation(
            http_request("/v1/chat/completions", request_body),
            http_response_json(response_body),
        )
    )
    emitter = _CollectingEmitter()
    client_segments = tuple(segment for segment in segments if segment.src == CLIENT[0])
    server_segments = tuple(segment for segment in segments if segment.src == SERVER[0])
    pipeline = LivePipeline(
        ifaces=("tap-request", "tap-response"),
        emitter=emitter,
        worker_count=2,
        segment_queue_size=64,
        message_queue_size=8,
        capture_factory=_SplitCaptureFactory(client_segments, server_segments),
        mp_context=multiprocessing.get_context("spawn"),
        stats_interval=0,
    )

    stats = pipeline.run()

    # Each receiver owns one physical half of the full-duplex tap. The stable
    # symmetric hash must converge both halves on one stateful flow worker.
    assert len(pipeline.receivers) == 2
    assert len(pipeline.workers) == 2
    assert all(not process.is_alive() for process in pipeline.receivers + pipeline.workers)
    assert stats.segments_seen == len(segments)
    assert stats.segment_queue_dropped == 0
    assert stats.message_queue_dropped == 0
    assert stats.messages_built == 1
    assert stats.emitter_accepted == 1
    assert len(emitter.messages) == 1
    assert emitter.messages[0]["response"]["output_text"] == "hi"


class _AliveProcess:
    """Minimal stand-in: alive until the sentinel loop gives up on it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.exitcode = None

    def is_alive(self) -> bool:
        return True


def test_one_unreachable_worker_does_not_starve_the_others_of_a_sentinel(caplog):
    """A blocked worker must be skipped individually, not end the loop.

    The sentinel is what makes a worker run processor.finish() and publish the
    exchanges it still holds, so abandoning the loop at the first full queue
    costs every later worker its whole ConnectionTable to a force-terminate.
    """
    pipeline = LivePipeline.__new__(LivePipeline)  # bypass process construction
    pipeline._sentinels_sent = False
    pipeline._workers = [_AliveProcess(f"flow-worker-{i}") for i in range(3)]
    # Worker 0's queue is permanently full; 1 and 2 have room.
    wedged: queue.Queue[Any] = queue.Queue(maxsize=1)
    wedged.put(_segment())
    reachable = [queue.Queue(maxsize=4), queue.Queue(maxsize=4)]
    pipeline._segment_queues = [wedged, *reachable]
    pipeline.poll = lambda timeout=0.0, limit=None: 0

    pipeline._send_worker_sentinels(deadline=time.monotonic() - 1)  # already expired

    assert [q.qsize() for q in reachable] == [1, 1], "later workers were starved"
    assert pipeline._sentinels_sent is False  # worker 0 genuinely missed its stop
    assert "flow-worker-0" in caplog.text


# --- capture windows through the process topology ------------------------------


class _ObservedCapture:
    """A source with an observer, like RingCapture: frames go to the observer
    with a direction, and nothing is returned as a TCP segment."""

    def __init__(self, frames: list[tuple[bytes, float, str]]) -> None:
        self._frames = frames
        self._observer = None
        self._read = False
        self.closed = False
        self.received = len(frames)
        self.drops = 0
        self.freeze_q_count = 0
        self.truncated_packets = 0
        self.malformed_blocks = 0

    def attach_observer(self, observer) -> None:
        self._observer = observer

    def read_segments(self, limit=None, *, max_blocks=8):
        if self._read:
            return None
        self._read = True
        for frame, ts, direction in self._frames:
            self._observer.observe(frame, ts, direction)
        return []

    def poll_stats(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_receiver_reports_capture_windows_before_its_final_stats(monkeypatch):
    from app import config
    from app.windows import WindowReport

    import types

    monkeypatch.setattr(config, "FRAME_PROCESSOR_ACCOUNT_ALL", True)
    monkeypatch.setattr(config, "FRAME_PROCESSOR_PEER_MAC", None)
    monkeypatch.setattr(config, "FRAME_PROCESSOR_WINDOW_SECONDS", 300)
    beacon = bytes(6) + bytes.fromhex("020000000003") + b"\x88\x99" + bytes(32)
    t0 = (time.time() // 300) * 300  # the open window, so the wall-clock roll leaves it alone
    # The accountant stamps its process epoch from its own clock at
    # construction; the frames below are stamped from the previous window, so
    # the "process" has to have started at that window's boundary for the
    # first window to be whole. live_pipeline's own clock stays real.
    monkeypatch.setattr("app.accounting.time", types.SimpleNamespace(time=lambda: t0 - 300))
    frames = [(beacon, t0 - 300 + 1, "out"), (beacon, t0 + 1, "out"), (beacon, t0 + 2, "in")]
    reports: queue.Queue[Any] = queue.Queue()

    _receiver_main(
        "tap0", [queue.Queue(maxsize=8)], reports, _NeverStopped(),
        lambda _iface: _ObservedCapture(frames), None, frozenset(), 0.01, 30.0,
    )

    items = []
    while True:
        try:
            items.append(reports.get_nowait())
        except queue.Empty:
            break
    windows = [item for item in items if isinstance(item, WindowReport)]
    assert isinstance(items[-1], ReceiverStats)
    assert [w.observed for w in windows] == [1, 2], "one per window, closed in order"
    assert windows[0].complete
    assert windows[1].classes["realtek"]["out"] == 1 and windows[1].classes["realtek"]["in"] == 1
    assert not windows[1].complete, "the last window was cut short by shutdown"
    assert windows[1].groups[0].kind == "capture-gap"
    assert items[-1].frames_observed == 3


class _LocalContext:
    """A multiprocessing context made of in-process queues, for the parent."""

    Event = threading.Event

    @staticmethod
    def Queue(maxsize=0):
        return queue.Queue(maxsize=maxsize)

    def Process(self, *args, **kwargs):  # pragma: no cover - never started here
        raise AssertionError("this test never starts a process")


def test_parent_merges_window_reports_and_exchange_findings_into_one_row():
    from app.windows import ExchangeFinding, WindowReport

    rows: list[dict] = []
    pipeline = LivePipeline(
        ifaces=("mon0", "mon1"),
        emitter=lambda message: True,
        mp_context=_LocalContext(),
        window_sink=rows.append,
        window_seconds=300.0,
        window_flush_delay=0.0,
        capture_host="gpu-node-1",
        tapped_hostname="kserve-gpt-oss-120b",
        tap_version="test",
    )
    start = 1_700_000_100.0
    for iface in ("mon0", "mon1"):
        pipeline._status_queue.put(
            WindowReport(iface=iface, window_start=start, window_end=start + 300, process_epoch=start, observed=2, classified=2)
        )
    pipeline._status_queue.put(ExchangeFinding("unexpected-exchange", "GET /admin", start + 10, "192.0.2.2:8000"))
    pipeline._status_queue.put(FlowWorkerStats(worker_id=0))

    pipeline._collect_reports()
    pipeline._flush_windows(now=start + 300 + 1)

    assert len(rows) == 1
    row = rows[0]
    assert row["ifaces"] == "mon0,mon1"
    assert row["observed"] == 4
    assert row["complete"] is False
    assert [f["kind"] for f in row["findings"]] == ["unexpected-exchange"]
    assert pipeline.stats.windows_reported == 1
    assert pipeline.stats.windows_rejected == 0
