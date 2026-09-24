"""Multi-process live capture, flow sharding, and parent-side emission.

The live path has two very different workloads:

* draining and classifying every mirrored packet; and
* maintaining ordered TCP/HTTP state for the small subset which is relevant.

One receiver process owns each tap interface.  Receivers route decoded
``TcpSegment`` objects to flow workers by a stable, direction-independent hash
of the *inner* endpoints.  Consequently both halves of one connection always
reach the same ``SegmentProcessor`` while unrelated connections can use
different CPU cores.  Completed tap messages travel back to the parent, where
the existing ``EmitterPool`` remains the sole owner of its HTTP threads.

Only bounded data queues sit between the stages.  A full queue is never
allowed to stop a receiver from draining its packet ring: the loss is counted
and logged instead.  Controlled shutdown is different -- receivers stop
first, workers drain every segment already accepted, and the parent drains
every completed message before returning.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import multiprocessing
import queue
import select
import signal
import struct
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Protocol

from app.capture import (
    DEFAULT_RING_READ_BLOCKS,
    TcpSegment,
    carries_reassembly_state,
    ports_allowed,
)
from app.windows import ExchangeFinding, WindowMerger, WindowReport

log = logging.getLogger("frameprocessor.live_pipeline")

_CHILD_LOG_FORMAT = "%(asctime)s %(processName)s %(name)s %(levelname)s %(message)s"
_QUEUE_STOP = None

# TCP candidates a receiver accepts from one read before it returns to the top
# of its loop to observe stop_event and the statistics timer. It bounds the
# transient list and the latency of a shutdown, not the ring: the block budget
# below is what bounds a turn when the traffic is all UDP and this never fires.
_RECEIVER_SEGMENT_LIMIT = 1024


class MessageSubmitter(Protocol):
    """The parent-side part of :class:`app.emitter.EmitterPool`."""

    def submit(self, message: dict[str, Any]) -> bool: ...


@dataclass(frozen=True)
class ReceiverStats:
    iface: str
    # Candidates returned by the capture source. RingCapture has already
    # rejected non-TCP, configured-out ports, and packets carrying no
    # reassembly state before this count; ``excluded``/``ack_only`` cover
    # injectable or fallback sources. ``ack_only`` is named for what it
    # almost always is, but counts any segment with no payload and no
    # SYN/FIN/RST.
    segments_seen: int = 0
    excluded: int = 0
    ack_only: int = 0
    dispatched: int = 0
    queue_dropped: int = 0
    kernel_received: int = 0
    kernel_dropped: int = 0
    kernel_freezes: int = 0
    truncated_packets: int = 0
    malformed_blocks: int = 0
    errors: int = 0
    # Frame accounting, when FRAME_PROCESSOR_ACCOUNT_ALL is on. ``frames_observed``
    # counts every frame the ring handed over, not just TCP candidates, so
    # comparing it with ``segments_seen`` shows how much of the link the
    # inference pipeline never sees.
    frames_observed: int = 0
    frames_classified: int = 0
    accounting_findings: int = 0


@dataclass(frozen=True)
class FlowWorkerStats:
    worker_id: int
    segments_processed: int = 0
    messages_built: int = 0
    queue_dropped: int = 0
    errors: int = 0


@dataclass(frozen=True)
class PipelineStats:
    """Aggregate counters returned after a complete pipeline shutdown."""

    segments_seen: int = 0
    excluded: int = 0
    ack_only: int = 0
    segments_dispatched: int = 0
    segment_queue_dropped: int = 0
    kernel_received: int = 0
    kernel_dropped: int = 0
    kernel_freezes: int = 0
    truncated_packets: int = 0
    malformed_blocks: int = 0
    receiver_errors: int = 0
    frames_observed: int = 0
    frames_classified: int = 0
    accounting_findings: int = 0
    segments_processed: int = 0
    messages_built: int = 0
    message_queue_dropped: int = 0
    worker_errors: int = 0
    emitter_accepted: int = 0
    emitter_rejected: int = 0
    emitter_errors: int = 0
    receiver_process_failures: int = 0
    worker_process_failures: int = 0
    forced_terminations: int = 0
    # Capture windows merged in the parent and handed to the window sink, and
    # the ones the sink refused (its queue was full).
    windows_reported: int = 0
    windows_rejected: int = 0


def _endpoint_bytes(address: str, port: int) -> bytes:
    """Canonical, unambiguous bytes for an IPv4/IPv6 endpoint."""
    try:
        ip = ipaddress.ip_address(address)
        # The prefix prevents an unusual textual fallback or a future address
        # family from aliasing an ordinary packed IP address.
        return bytes((ip.version,)) + ip.packed + struct.pack("!H", port)
    except ValueError:
        encoded = address.encode("utf-8", errors="surrogatepass")
        return b"\x00" + struct.pack("!H", len(encoded)) + encoded + struct.pack("!H", port)


def flow_key(segment: TcpSegment) -> bytes:
    """A stable symmetric key made from the decoded inner TCP endpoints."""
    left = _endpoint_bytes(segment.src, segment.sport)
    right = _endpoint_bytes(segment.dst, segment.dport)
    if right < left:
        left, right = right, left
    return struct.pack("!H", len(left)) + left + struct.pack("!H", len(right)) + right


def flow_shard(segment: TcpSegment, worker_count: int) -> int:
    """Choose a flow worker, identically in every process and direction.

    Python's built-in ``hash`` is deliberately randomised independently in
    spawned processes, so it cannot be used here.  BLAKE2s is stable and the
    eight-byte digest is ample for distributing connection state.
    """
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    digest = hashlib.blake2s(flow_key(segment), digest_size=8, person=b"ft-flow").digest()
    return int.from_bytes(digest, "big") % worker_count


def _log_at_powers_of_two(count: int) -> bool:
    """Make overload visible without turning logging into another overload."""
    return count > 0 and count & (count - 1) == 0


def _default_capture_factory(iface: str, observer=None):
    # Lazy so this module remains importable on non-Linux development hosts,
    # and so app.main can import LivePipeline without a circular dependency.
    from app import config
    from app.capture import RingCapture

    return RingCapture(
        iface,
        observer=observer,
        block_size=config.FRAME_PROCESSOR_RING_BLOCK_SIZE,
        block_count=config.FRAME_PROCESSOR_RING_BLOCK_COUNT,
        frame_size=config.FRAME_PROCESSOR_RING_FRAME_SIZE,
        retire_timeout_ms=config.FRAME_PROCESSOR_RING_RETIRE_TIMEOUT_MS,
    )


@dataclass(frozen=True)
class CaptureTotals:
    """One capture source's cumulative kernel counters."""

    received: int = 0
    dropped: int = 0
    freezes: int = 0
    truncated: int = 0
    malformed: int = 0


# Attribute names on the source, in CaptureTotals field order. Read with a
# default because the finite-iterable source used by integration tests keeps
# no counters at all.
_CAPTURE_COUNTERS = (
    "received",
    "drops",
    "freeze_q_count",
    "truncated_packets",
    "malformed_blocks",
)


def _capture_totals(source: Any) -> CaptureTotals:
    """Poll a capture source and return its cumulative kernel counters."""
    poll_stats = getattr(source, "poll_stats", None)
    if poll_stats is not None:
        poll_stats()
    return CaptureTotals(*(int(getattr(source, name, 0)) for name in _CAPTURE_COUNTERS))


def _route_segment(
    iface: str,
    segment: TcpSegment,
    worker_queues: list[Any],
    include_ports: frozenset[int] | None,
    exclude_ports: frozenset[int],
    counters: dict[str, int],
) -> None:
    counters["segments_seen"] += 1
    # Re-tested rather than trusted: RingCapture already applies both to what
    # it returns, but an injected or fallback source need not, and the cost of
    # being wrong is reassembling a port an operator excluded.
    if not ports_allowed(segment.sport, segment.dport, include_ports, exclude_ports):
        counters["excluded"] += 1
        return
    if not carries_reassembly_state(segment):
        counters["ack_only"] += 1
        return

    destination = flow_shard(segment, len(worker_queues))
    try:
        worker_queues[destination].put_nowait(segment)
    except queue.Full:
        counters["queue_dropped"] += 1
        dropped = counters["queue_dropped"]
        if _log_at_powers_of_two(dropped):
            log.error(
                "%s segment queue full; dropped %d decoded TCP segment(s) total",
                iface,
                dropped,
            )
    else:
        counters["dispatched"] += 1


def _let_the_parent_own_ctrl_c() -> None:
    """Ignore SIGINT in a child, because the parent coordinates shutdown.

    A spawned child inherits default SIGINT handling, so an operator's Ctrl-C
    raises KeyboardInterrupt inside it. Both child mains treat any BaseException
    as a failure — traceback logged, final publish skipped, nonzero exit — which
    turns a routine stop into lost reassembly state and a nonzero exit code for
    the whole service. The parent catches KeyboardInterrupt and brings the
    children down through stop_event and the queue sentinel, which is the path
    that runs processor.finish(). SIGTERM, the Kubernetes signal, only ever
    reaches the parent and is already handled there.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (OSError, ValueError):  # pragma: no cover — not the main thread
        pass


def _beats_summary(snapshot: Any, now: float) -> str:
    """The declared cadences, for the periodic log line.

    Rendered on every tick rather than only when something is wrong: a beat
    that is being kept is the one piece of positive evidence in this log that
    the link is up and the capture can see it, and a line that only appears
    on failure cannot be distinguished from a line nobody is writing.

    Age since the last beat rather than a running total of misses, because a
    total never recovers — one blip and the line reads alarming forever,
    which is the same way a cumulative frame count reads healthy forever.
    """
    if not snapshot.beats:
        return ""
    parts = []
    for key, beat in sorted(snapshot.beats.items()):
        if not beat.last_ts:
            state = "NEVER SEEN"
        else:
            age = now - beat.last_ts
            state = f"{age:.0f}s ago"
            if age > beat.interval + beat.tolerance:
                state = "LATE " + state
        if beat.unscheduled:
            state += f", {beat.unscheduled} unscheduled"
        parts.append(f"{key} {state}")
    return "; beats: " + ", ".join(parts)


def _receiver_main(
    iface: str,
    worker_queues: list[Any],
    status_queue: Any,
    stop_event: Any,
    capture_factory: Callable[[str], Any],
    include_ports: frozenset[int] | None,
    exclude_ports: frozenset[int],
    poll_interval: float,
    stats_interval: float,
) -> None:
    """Child target: drain one interface and route its decoded segments."""
    logging.basicConfig(level=logging.INFO, format=_CHILD_LOG_FORMAT)
    _let_the_parent_own_ctrl_c()
    source: Any = None
    counters = {
        "segments_seen": 0,
        "excluded": 0,
        "ack_only": 0,
        "dispatched": 0,
        "queue_dropped": 0,
        "errors": 0,
    }
    totals = CaptureTotals()
    # Kernel drops already handed to the accountant, so each stats tick
    # attributes only the fresh ones to the window that is open.
    dropped_noted = 0
    observed_noted = 0
    accountant = None
    accounted = None
    next_stats = time.monotonic() + stats_interval if stats_interval > 0 else float("inf")
    failure: BaseException | None = None

    try:
        source = capture_factory(iface)

        # Frame accounting is attached to the source rather than wrapped
        # around the segment stream, because by the time a segment exists the
        # frames this is meant to account for have already been discarded.
        from app import config as _config
        from app.accounting import UNPINNED_CLASSES, FrameAccountant
        from app.policy import FixedDirection, LinkPolicy

        accountant = None
        observer = None
        if _config.FRAME_PROCESSOR_ACCOUNT_ALL:
            policy = LinkPolicy.from_config(iface=iface)
            # Declared before the accountant is built, because a monitor port
            # carries one direction and the accountant needs that to know
            # which declared beats it could ever see.
            fixed = _config.FRAME_PROCESSOR_IFACE_DIRECTIONS.get(iface)
            accountant = FrameAccountant(
                policy=policy,
                iface=iface,
                fixed_direction=fixed,
                unpinned_classes=UNPINNED_CLASSES - _config.FRAME_PROCESSOR_PIN_CLASSES,
                whole_frame_classes=_config.FRAME_PROCESSOR_PIN_WHOLE_FRAME,
                allow_ip_options=_config.FRAME_PROCESSOR_ALLOW_IP_OPTIONS,
                allow_fragments=_config.FRAME_PROCESSOR_ALLOW_FRAGMENTS,
                window_seconds=_config.FRAME_PROCESSOR_WINDOW_SECONDS or None,
                window_grace=_config.FRAME_PROCESSOR_WINDOW_GRACE_SECONDS,
                max_window_groups=_config.FRAME_PROCESSOR_WINDOW_MAX_GROUPS,
                sample_frame_bytes=_config.FRAME_PROCESSOR_SAMPLE_FRAME_BYTES,
            )
            # A monitor port carries one direction and its ring's packet type
            # says nothing useful; the declared direction is stamped on every
            # frame instead. An interface with no declaration is this host's
            # own end of the link, where the kernel bit is right.
            observer = FixedDirection(accountant, fixed) if fixed else accountant
            # Logged because an unset policy looks exactly like a clean link:
            # no findings either way, and only one of them means anything.
            log.info(
                "%s link whitelist: %s; direction=%s",
                iface,
                policy.describe() if policy is not None else "off (FRAME_PROCESSOR_PEER_MAC unset)",
                f"fixed {fixed}" if fixed else "kernel packet type",
            )
        attach = getattr(source, "attach_observer", None)
        if observer is not None and attach is not None:
            attach(observer)
        elif accountant is not None:
            # An injected source with no attach point: say so once rather than
            # reporting a balanced account for frames nobody looked at.
            log.warning(
                "%s capture source takes no frame observer; accounting is off "
                "for this interface",
                iface,
            )
            accountant = None

        read_segments = getattr(source, "read_segments", None)

        if read_segments is None:
            # A finite iterable is useful for pcap-like integration tests.  A
            # real live source should implement the batch API below so it can
            # wake periodically and observe stop_event.
            batches: Iterable[Iterable[TcpSegment]] = ([segment] for segment in source)
            for batch in batches:
                if stop_event.is_set():
                    break
                for segment in batch:
                    _route_segment(
                        iface,
                        segment,
                        worker_queues,
                        include_ports,
                        exclude_ports,
                        counters,
                    )
        else:
            while not stop_event.is_set():
                # The inspected-block bound is load-bearing: a limit on
                # returned TCP candidates alone never fires during a pure-UDP
                # flood, when ready ring blocks can arrive continuously.
                batch = read_segments(
                    limit=_RECEIVER_SEGMENT_LIMIT,
                    max_blocks=DEFAULT_RING_READ_BLOCKS,
                )
                # ``None`` is the injectable finite-source EOF marker.  An
                # empty list means a live nonblocking poll found no full block.
                if batch is None:
                    break
                for segment in batch:
                    _route_segment(
                        iface,
                        segment,
                        worker_queues,
                        include_ports,
                        exclude_ports,
                        counters,
                    )

                if accountant is not None:
                    # Windows close on the wall clock, not on frames, so an
                    # idle link still produces its heartbeat rows.
                    for report in accountant.roll(time.time()):
                        status_queue.put(report)

                if time.monotonic() >= next_stats:
                    before = totals.dropped
                    totals = _capture_totals(source)
                    fresh = totals.dropped - before
                    if fresh:
                        log.warning(
                            "%s packet ring dropped +%d frame(s), %d total",
                            iface,
                            fresh,
                            totals.dropped,
                        )
                    if accountant is not None:
                        accountant.note_kernel_dropped(totals.dropped - dropped_noted)
                        dropped_noted = totals.dropped
                    if accountant is not None:
                        snapshot = accountant.snapshot()
                        # Rate as well as total: a link that died an hour ago
                        # reports the same cumulative figure forever, which
                        # reads as healthy to anyone watching the log.
                        fresh_frames = snapshot.observed - observed_noted
                        observed_noted = snapshot.observed
                        log.info(
                            "%s accounted +%d frame(s) (%d total) in %d class(es): %s%s%s",
                            iface,
                            fresh_frames,
                            snapshot.observed,
                            len(snapshot.classes),
                            ", ".join(
                                f"{name}={totals.frames}"
                                for name, totals in sorted(
                                    snapshot.classes.items(),
                                    key=lambda kv: -kv[1].frames,
                                )[:8]
                            ),
                            _beats_summary(snapshot, time.time()),
                            ""
                            if snapshot.findings == 0
                            else f" — {snapshot.findings} finding(s)",
                        )
                    next_stats = time.monotonic() + stats_interval

                if batch:
                    # There may be a retained mid-block cursor after the
                    # candidate limit. Drain it before waiting on fd readiness;
                    # select only speaks for newly retired blocks.
                    continue

                # RingCapture is nonblocking. Waiting only after an empty read
                # keeps an idle receiver at zero CPU while the timeout bounds
                # stop and statistics latency. A pure-UDP batch can also be
                # empty after releasing its eight-block work budget; in that
                # case a still-ready fd returns immediately.
                fileno = getattr(source, "fileno", None)
                if fileno is not None:
                    select.select([source], [], [], poll_interval)
                else:
                    wait = getattr(stop_event, "wait", None)
                    if wait is not None:
                        wait(poll_interval)
                    else:  # simple injected events used by unit tests
                        time.sleep(poll_interval)
    except BaseException as exc:  # noqa: BLE001 -- report child failure before exit
        counters["errors"] += 1
        failure = exc
        log.exception("Receiver process for %s failed", iface)
    finally:
        if source is not None:
            try:
                totals = _capture_totals(source)
            except Exception:  # noqa: BLE001 -- shutdown diagnostics are best effort
                counters["errors"] += 1
                log.exception("Could not read final packet-ring stats for %s", iface)
            close = getattr(source, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    counters["errors"] += 1
                    log.exception("Could not close capture source for %s", iface)

        if accountant is not None:
            # Only the drops not yet attributed to a window: the stats ticks
            # above already handed over everything up to ``dropped_noted``,
            # and poll_stats self-resets, so ``totals`` is cumulative.
            accountant.note_kernel_dropped(totals.dropped - dropped_noted)
            for report in accountant.finish(time.time()):
                status_queue.put(report)
            accounted = accountant.snapshot()
            log.info(
                "%s final account: observed=%d classified=%d balanced=%s "
                "complete=%s findings=%d",
                iface,
                accounted.observed,
                accounted.classified,
                accounted.balanced,
                accounted.complete,
                accounted.findings,
            )

        status_queue.put(
            ReceiverStats(
                iface=iface,
                segments_seen=counters["segments_seen"],
                excluded=counters["excluded"],
                ack_only=counters["ack_only"],
                dispatched=counters["dispatched"],
                queue_dropped=counters["queue_dropped"],
                kernel_received=totals.received,
                kernel_dropped=totals.dropped,
                kernel_freezes=totals.freezes,
                truncated_packets=totals.truncated,
                malformed_blocks=totals.malformed,
                errors=counters["errors"],
                frames_observed=accounted.observed if accounted else 0,
                frames_classified=accounted.classified if accounted else 0,
                accounting_findings=accounted.findings if accounted else 0,
            )
        )
    if failure is not None:
        raise failure


def _default_processor_factory():
    # Imported in the spawned flow worker, after app.main has finished loading
    # in the parent.  This avoids main -> live_pipeline -> main at import time.
    from app.main import SegmentProcessor

    return SegmentProcessor()


def _worker_main(
    worker_id: int,
    segment_queue: Any,
    message_queue: Any,
    status_queue: Any,
    processor_factory: Callable[[], Any],
    idle_poll_interval: float,
) -> None:
    """Child target: exclusively own TCP/HTTP state for a set of flows."""
    logging.basicConfig(level=logging.INFO, format=_CHILD_LOG_FORMAT)
    _let_the_parent_own_ctrl_c()
    processed = messages = dropped = errors = 0
    processor: Any = None
    failure: BaseException | None = None

    def publish(built: Iterable[dict[str, Any]]) -> None:
        nonlocal messages, dropped
        for message in built:
            messages += 1
            try:
                message_queue.put_nowait(message)
            except queue.Full:
                dropped += 1
                if _log_at_powers_of_two(dropped):
                    log.error(
                        "Flow worker %d message queue full; dropped %d completed "
                        "tap message(s) total",
                        worker_id,
                        dropped,
                    )

    try:
        processor = processor_factory()
        # Exchange-level findings are decided here, after reassembly, in a
        # different process from the accountant that counts frames. They ride
        # the status queue to the parent, which files them on the window
        # their timestamp falls in.
        from app import main as _main

        _main.set_finding_sink(status_queue.put)
        while True:
            try:
                item = segment_queue.get(timeout=idle_poll_interval)
            except queue.Empty:
                publish(processor.sweep(time.time()))
                continue
            if item is _QUEUE_STOP:
                break
            processed += 1
            publish(processor.feed(item))
        # The sentinel is inserted only after every receiver has stopped, so
        # reaching it proves all accepted segments ahead of it were consumed.
        publish(processor.finish())
    except BaseException as exc:  # noqa: BLE001 -- keep failure explicit in stats/logs
        errors += 1
        failure = exc
        log.exception("Flow worker %d failed", worker_id)
    finally:
        status_queue.put(
            FlowWorkerStats(
                worker_id=worker_id,
                segments_processed=processed,
                messages_built=messages,
                queue_dropped=dropped,
                errors=errors,
            )
        )
    if failure is not None:
        raise failure


class LivePipeline:
    """Parent-side supervisor for receivers and connection-sharded workers.

    ``emitter`` is deliberately retained only in this parent object and never
    appears in child-process arguments.  The default ``spawn`` context is safe
    even when an ``EmitterPool`` (and therefore HTTP worker threads) already
    exists before :meth:`start` is called.
    """

    def __init__(
        self,
        *,
        ifaces: Collection[str],
        emitter: MessageSubmitter | Callable[[dict[str, Any]], bool],
        worker_count: int = 4,
        segment_queue_size: int = 4096,
        message_queue_size: int = 256,
        include_ports: Collection[int] | None = None,
        exclude_ports: Collection[int] = (),
        capture_factory: Callable[[str], Any] = _default_capture_factory,
        processor_factory: Callable[[], Any] = _default_processor_factory,
        receiver_poll_interval: float = 0.1,
        worker_idle_poll_interval: float = 0.25,
        stats_interval: float = 30.0,
        mp_context: BaseContext | None = None,
        receiver_target: Callable[..., None] = _receiver_main,
        worker_target: Callable[..., None] = _worker_main,
        window_sink: Callable[[dict[str, Any]], Any] | None = None,
        window_seconds: float = 300.0,
        window_flush_delay: float = 15.0,
        capture_host: str = "",
        tapped_hostname: str | None = None,
        tap_version: str = "",
    ) -> None:
        names = tuple(dict.fromkeys(ifaces))
        if not names:
            raise ValueError("at least one capture interface is required")
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if segment_queue_size <= 0 or message_queue_size <= 0:
            raise ValueError("queue sizes must be positive")
        if receiver_poll_interval <= 0 or worker_idle_poll_interval <= 0:
            raise ValueError("poll intervals must be positive")

        self.ifaces = names
        self.worker_count = worker_count
        self._emitter = emitter
        self._capture_factory = capture_factory
        self._processor_factory = processor_factory
        self._include_ports = None if include_ports is None else frozenset(include_ports)
        self._exclude_ports = frozenset(exclude_ports)
        self._receiver_poll_interval = receiver_poll_interval
        self._worker_idle_poll_interval = worker_idle_poll_interval
        self._stats_interval = stats_interval
        self._context = mp_context or multiprocessing.get_context("spawn")
        self._receiver_target = receiver_target
        self._worker_target = worker_target

        self._stop_event = self._context.Event()
        self._segment_queues = [
            self._context.Queue(maxsize=segment_queue_size) for _ in range(worker_count)
        ]
        self._message_queue = self._context.Queue(maxsize=message_queue_size)
        # At most one final report per child; leaving this tiny control plane
        # unbounded ensures diagnostics cannot themselves be dropped.
        self._status_queue = self._context.Queue()
        self._receivers: list[Any] = []
        self._workers: list[Any] = []
        self._receiver_reports: dict[str, ReceiverStats] = {}
        self._worker_reports: dict[int, FlowWorkerStats] = {}
        # One row per window for the whole link: receivers' per-interface
        # accounts and workers' exchange findings meet here.
        self._window_sink = window_sink
        self._merger = (
            WindowMerger(
                ifaces=list(names),
                window_seconds=window_seconds,
                flush_delay=window_flush_delay,
                capture_host=capture_host,
                tapped_hostname=tapped_hostname,
                tap_version=tap_version,
            )
            if window_seconds > 0
            else None
        )
        self._windows_reported = 0
        self._windows_rejected = 0
        self._emitter_accepted = 0
        self._emitter_rejected = 0
        self._emitter_errors = 0
        self._forced_terminations = 0
        self._started = False
        self._closed = False
        self._sentinels_sent = False

    @property
    def receivers(self) -> tuple[Any, ...]:
        return tuple(self._receivers)

    @property
    def workers(self) -> tuple[Any, ...]:
        return tuple(self._workers)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("LivePipeline is closed")
        if self._started:
            return

        try:
            # Consumers first: a receiver can fill a small queue immediately
            # after it opens a busy tap interface.
            for worker_id, segment_queue in enumerate(self._segment_queues):
                process = self._context.Process(
                    target=self._worker_target,
                    args=(
                        worker_id,
                        segment_queue,
                        self._message_queue,
                        self._status_queue,
                        self._processor_factory,
                        self._worker_idle_poll_interval,
                    ),
                    name=f"frame-processor-flow-{worker_id + 1}",
                )
                process.start()
                self._workers.append(process)

            for iface in self.ifaces:
                process = self._context.Process(
                    target=self._receiver_target,
                    args=(
                        iface,
                        self._segment_queues,
                        self._status_queue,
                        self._stop_event,
                        self._capture_factory,
                        self._include_ports,
                        self._exclude_ports,
                        self._receiver_poll_interval,
                        self._stats_interval,
                    ),
                    name=f"frame-processor-rx-{iface}",
                )
                process.start()
                self._receivers.append(process)
        except BaseException:
            # Process.start itself can fail after earlier children are already
            # live (pickling error, PID/resource exhaustion). Never orphan the
            # partial topology or leave main.finally to accidentally start it.
            self._stop_event.set()
            self._terminate_alive(self._receivers + self._workers, "partially-started")
            self._closed = True
            raise
        self._started = True

    def _submit(self, message: dict[str, Any]) -> None:
        try:
            submit = getattr(self._emitter, "submit", self._emitter)
            accepted = submit(message)
        except Exception:  # noqa: BLE001 -- one callback must not stop draining
            self._emitter_errors += 1
            log.exception("Parent emitter callback failed")
            return
        if accepted is False:
            self._emitter_rejected += 1
        else:
            self._emitter_accepted += 1

    def _collect_reports(self) -> None:
        while True:
            try:
                report = self._status_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(report, ReceiverStats):
                self._receiver_reports[report.iface] = report
            elif isinstance(report, FlowWorkerStats):
                self._worker_reports[report.worker_id] = report
            elif isinstance(report, WindowReport):
                if self._merger is not None:
                    self._merger.add_report(report)
            elif isinstance(report, ExchangeFinding):
                if self._merger is not None:
                    self._merger.add_exchange_finding(report, time.time())
            else:
                log.error("Unknown live-pipeline status report: %r", report)

    def _deliver_windows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self._windows_reported += 1
            if self._window_sink is None:
                log.info(
                    "capture window %s → %s: %s, %d frame(s), %d finding(s) (no ledger configured)",
                    row["window_start"],
                    row["window_end"],
                    "complete" if row["complete"] else "INCOMPLETE",
                    row["observed"],
                    row["finding_count"],
                )
                continue
            try:
                accepted = self._window_sink(row)
            except Exception:  # noqa: BLE001 -- a sink failure must not stop draining
                self._windows_rejected += 1
                log.exception("Window sink failed")
                continue
            if accepted is False:
                self._windows_rejected += 1

    def _flush_windows(self, now: float) -> None:
        if self._merger is not None:
            self._deliver_windows(self._merger.flush(now))

    def poll(self, timeout: float = 0.0, limit: int | None = None) -> int:
        """Hand completed messages to the parent emitter and return the count."""
        if not self._started:
            raise RuntimeError("LivePipeline has not been started")
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative")
        if limit == 0:
            return 0
        handled = 0
        self._collect_reports()
        self._flush_windows(time.time())
        try:
            first = self._message_queue.get(timeout=timeout) if timeout else self._message_queue.get_nowait()
        except queue.Empty:
            return 0
        self._submit(first)
        handled = 1
        while limit is None or handled < limit:
            try:
                message = self._message_queue.get_nowait()
            except queue.Empty:
                break
            self._submit(message)
            handled += 1
        self._collect_reports()
        return handled

    def run(self) -> PipelineStats:
        """Run until interrupted, a finite source ends, or a receiver fails."""
        try:
            self.start()
            while any(process.is_alive() for process in self._receivers):
                self.poll(timeout=0.1)
                failed_receivers = [
                    process
                    for process in self._receivers
                    if process.exitcode not in (None, 0)
                ]
                failed_workers = [
                    process
                    for process in self._workers
                    if process.exitcode is not None
                ]
                if failed_receivers or failed_workers:
                    log.error(
                        "Child process failure; stopping the full-duplex pipeline: %s",
                        ", ".join(
                            f"{p.name}={p.exitcode}"
                            for p in failed_receivers + failed_workers
                        ),
                    )
                    break
        except KeyboardInterrupt:
            log.info("Stopping live capture")
        return self.stop()

    def _wait_for(self, processes: list[Any], deadline: float) -> None:
        while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
            self.poll(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
            for process in processes:
                process.join(timeout=0)

    def _terminate_alive(self, processes: list[Any], stage: str) -> None:
        for process in processes:
            if not process.is_alive():
                continue
            self._forced_terminations += 1
            log.error("Force-terminating stuck %s process %s", stage, process.name)
            process.terminate()
        for process in processes:
            process.join(timeout=1.0)

    def _send_worker_sentinels(self, deadline: float) -> None:
        """Ask every worker to finish, even when one of them cannot be reached.

        A worker whose segment queue is still full at the deadline is skipped
        individually rather than abandoning the loop. The sentinel is what makes
        a worker run ``processor.finish()`` and publish the exchanges it still
        holds, so returning early would cost every *later* worker its whole
        ConnectionTable to a force-terminate — workers whose queues have room
        are reached on their first attempt regardless of the deadline.
        """
        if self._sentinels_sent:
            return
        sent = 0
        for process, segment_queue in zip(self._workers, self._segment_queues):
            if not process.is_alive():
                sent += 1
                continue
            while process.is_alive():
                try:
                    segment_queue.put(_QUEUE_STOP, timeout=0.05)
                    sent += 1
                    break
                except queue.Full:
                    self.poll(timeout=0.0)
                    if time.monotonic() >= deadline:
                        log.error(
                            "Could not hand the stop sentinel to %s before the "
                            "deadline; its reassembly state is lost",
                            process.name,
                        )
                        break
        self._sentinels_sent = sent == len(self._workers)

    def stop(self, timeout: float = 30.0) -> PipelineStats:
        """Stop receivers, drain accepted work, and return final counters."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if self._closed:
            return self.stats
        if not self._started:
            self._closed = True
            return self.stats

        deadline = time.monotonic() + timeout
        self._stop_event.set()
        self._wait_for(self._receivers, deadline)
        self._terminate_alive(self._receivers, "receiver")

        self._send_worker_sentinels(deadline)
        self._wait_for(self._workers, deadline)
        self._terminate_alive(self._workers, "flow-worker")

        # Joined Queue producer processes have flushed their feeder threads,
        # so repeated nonblocking reads now consume the complete tail.
        while self.poll(timeout=0.0):
            pass
        self._collect_reports()
        if self._merger is not None:
            # Whatever is still pending is written now, flush delay or not:
            # the children have all reported, so nothing more is coming.
            self._deliver_windows(self._merger.finish())
        self._closed = True
        return self.stats

    @property
    def stats(self) -> PipelineStats:
        self._collect_reports()
        receivers = list(self._receiver_reports.values())
        workers = list(self._worker_reports.values())
        return PipelineStats(
            segments_seen=sum(item.segments_seen for item in receivers),
            excluded=sum(item.excluded for item in receivers),
            ack_only=sum(item.ack_only for item in receivers),
            segments_dispatched=sum(item.dispatched for item in receivers),
            segment_queue_dropped=sum(item.queue_dropped for item in receivers),
            kernel_received=sum(item.kernel_received for item in receivers),
            kernel_dropped=sum(item.kernel_dropped for item in receivers),
            kernel_freezes=sum(item.kernel_freezes for item in receivers),
            truncated_packets=sum(item.truncated_packets for item in receivers),
            malformed_blocks=sum(item.malformed_blocks for item in receivers),
            receiver_errors=sum(item.errors for item in receivers),
            frames_observed=sum(item.frames_observed for item in receivers),
            frames_classified=sum(item.frames_classified for item in receivers),
            accounting_findings=sum(item.accounting_findings for item in receivers),
            segments_processed=sum(item.segments_processed for item in workers),
            messages_built=sum(item.messages_built for item in workers),
            message_queue_dropped=sum(item.queue_dropped for item in workers),
            worker_errors=sum(item.errors for item in workers),
            emitter_accepted=self._emitter_accepted,
            emitter_rejected=self._emitter_rejected,
            emitter_errors=self._emitter_errors,
            receiver_process_failures=sum(
                process.exitcode not in (None, 0) for process in self._receivers
            ),
            worker_process_failures=sum(
                process.exitcode not in (None, 0) for process in self._workers
            ),
            forced_terminations=self._forced_terminations,
            windows_reported=self._windows_reported,
            windows_rejected=self._windows_rejected,
        )
