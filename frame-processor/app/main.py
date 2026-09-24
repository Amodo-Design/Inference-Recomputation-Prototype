"""frame-processor entrypoint: wire the pipeline and run it.

frames → decode → reassemble → parse HTTP → filter LLM → emit tap message

`process_segments()` is the whole pipeline behind the frame sources — tests
drive it directly with synthetic segments, `main()` just feeds it frames from
a pcap file or a live interface.
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import Any, Callable, Iterable, Iterator

from app import config
from app.capture import (
    TcpSegment,
    decode_tcp,
    default_route_iface,
    pcap_frames,
    ports_allowed,
)
from app.emitter import EmitterPool, build_tap_message
from app.filter import classify, unexpected_reason
from app.http_stream import Exchange, ExchangeAssembler
from app.reassembly import Connection, ConnectionTable
from app.windows import ExchangeFinding

log = logging.getLogger("frameprocessor.main")

# Where exchange-level findings go besides the log. A flow worker points this
# at its status queue so the parent can file them on a capture window; the
# pcap path leaves it unset and they are logged only.
_finding_sink: Callable[[ExchangeFinding], Any] | None = None


def set_finding_sink(sink: Callable[[ExchangeFinding], Any] | None) -> None:
    global _finding_sink
    _finding_sink = sink


def _report_exchange_finding(kind: str, detail: str, ts: float | None, observed_on: str) -> None:
    """An exchange the link should not have carried: log it, and file it."""
    log.warning("finding: %s on %s — %s", kind, observed_on, detail)
    if _finding_sink is None:
        return
    try:
        _finding_sink(ExchangeFinding(kind, detail, ts if ts is not None else 0.0, observed_on))
    except Exception:  # noqa: BLE001 -- reporting must never stop the pipeline
        log.exception("Could not file exchange finding %s", kind)


def _messages(conn: Connection, exchanges: list[Exchange]) -> Iterator[dict[str, Any]]:
    """Tap messages for the LLM exchanges among those just assembled.

    Exchanges that are not inference are REPORTED here rather than dropped.
    Dropping them is the one outcome a link claiming full accounting cannot
    afford: policy.py bounds a TCP stream
    to two addresses and one port but says nothing about its contents, so an
    exchange that reassembles and classifies as nothing is exactly where bytes
    would hide. See filter.unexpected_reason for what still passes silently.
    """
    observed_on = f"{conn.responder[0]}:{conn.responder[1]}"
    for exchange in exchanges:
        classification = classify(exchange)
        if classification is None:
            reason = unexpected_reason(exchange)
            if reason is not None:
                _report_exchange_finding(
                    "unexpected-exchange",
                    reason,
                    exchange.completed_at
                    if exchange.completed_at is not None
                    else (exchange.received_at if exchange.received_at is not None else conn.last_ts),
                    observed_on,
                )
            continue
        yield build_tap_message(
            exchange,
            classification,
            # Per-exchange packet timestamps. A connection carrying several
            # inferences gives each its own timing; connection-level stamps
            # would make them indistinguishable in the ledger.
            received_at=(
                exchange.received_at if exchange.received_at is not None else conn.first_ts
            ),
            completed_at=(
                exchange.completed_at if exchange.completed_at is not None else conn.last_ts
            ),
            observed_on=observed_on,
        )


def _close(conn: Connection, ts: float | None) -> Iterator[dict[str, Any]]:
    """Salvage whatever a connection was still holding when it ended."""
    if conn.parser is None:
        return
    exchanges = conn.parser.finish(ts)
    if conn.has_gap and exchanges:
        # A missing TCP segment makes request/response pairing after the gap
        # unprovable, and these exchanges are filed anyway — with whatever was
        # captured, which for a queued request is a prompt and no response.
        #
        # They were discarded for a while, on the grounds that an unverifiable
        # row is worse than none. That has it backwards: the ledger is the
        # record, and an inference that reached the tap and never reached the
        # ledger is indistinguishable from one that never happened.
        #
        # Filed, they land where they belong. inf-ver-runner rejects an event
        # with an empty token list on a pre-check and returns `unverifiable`
        # rather than attempting a replay, so these can never surface as a
        # false `fail` — the outcome that would actually mislead. They are
        # attributed to `tokenization_mismatch`, which is shared with real
        # tokenizer problems, so telling capture loss apart from those still
        # means correlating with the kernel drop counters.
        #
        # The cost is that a completion wave behind a gap becomes a wave of
        # emitter submissions; that is bounded by the emit queue, and its
        # counters and depth warning make the pressure visible.
        log.warning(
            "Emitting %d incomplete exchange(s) from a capture-tainted "
            "connection: a dropped frame left the response unprovable, so "
            "these file with whatever was captured",
            len(exchanges),
        )
    yield from _messages(conn, exchanges)


class SegmentProcessor:
    """Stateful per-flow-shard TCP/HTTP pipeline.

    The synchronous pcap path feeds this object directly. Live worker
    processes also own one each, which matters because their input queue can
    go quiet: ``sweep()`` lets a timer expire idle connections without
    inventing a packet just to drive the old generator's cleanup branch.
    """

    def __init__(self) -> None:
        self.table = ConnectionTable(retain=False)
        self.last_ts: float | None = None
        self.next_idle_sweep: float | None = None

    def feed(self, seg: TcpSegment) -> Iterator[dict[str, Any]]:
        """Process one decoded TCP segment and yield completed tap messages.

        The port pre-filter is the caller's job, because both callers already
        do it upstream: the live receiver rejects a segment before it is ever
        queued to a worker, and ``process_segments`` filters its own source.
        """
        self.last_ts = seg.ts

        delivery = self.table.feed(seg)
        if delivery is not None:
            conn = delivery.conn
            if conn.parser is None:
                conn.parser = ExchangeAssembler()
            if delivery.data:
                parse = (
                    conn.parser.feed_client
                    if delivery.from_client
                    else conn.parser.feed_server
                )
                yield from _messages(conn, parse(delivery.data, delivery.ts))
                if conn.parser.unparseable and not conn.parser.unparseable_reported:
                    # Bytes on this connection never parsed as HTTP at all.
                    # On a link whose only permitted TCP is inference, that is
                    # the smuggling case; one finding per connection.
                    conn.parser.unparseable_reported = True
                    _report_exchange_finding(
                        "non-http-stream",
                        "bytes on this connection never parsed as HTTP",
                        delivery.ts if delivery.ts is not None else seg.ts,
                        f"{conn.responder[0]}:{conn.responder[1]}",
                    )
            if delivery.completed:
                yield from _close(conn, delivery.ts)

        if self.next_idle_sweep is None:
            self.next_idle_sweep = seg.ts + config.FRAME_PROCESSOR_IDLE_SWEEP_INTERVAL
        elif seg.ts >= self.next_idle_sweep:
            yield from self.sweep(seg.ts)

    def sweep(self, now: float) -> Iterator[dict[str, Any]]:
        """Expire connections on a timer, even if this shard is packet-idle."""
        if self.next_idle_sweep is None:
            self.next_idle_sweep = now + config.FRAME_PROCESSOR_IDLE_SWEEP_INTERVAL
            return
        if now < self.next_idle_sweep:
            return
        for idle in self.table.idle_flush(now, config.FRAME_PROCESSOR_IDLE_TIMEOUT):
            yield from _close(idle, now)
        self.next_idle_sweep = now + config.FRAME_PROCESSOR_IDLE_SWEEP_INTERVAL

    def finish(self) -> Iterator[dict[str, Any]]:
        """Drain this shard at end-of-source or controlled shutdown."""
        if self.last_ts is None:
            return
        for conn in self.table.drain():
            yield from _close(conn, self.last_ts)


def process_segments(segments: Iterable[TcpSegment]) -> Iterator[dict[str, Any]]:
    """The full pipeline: TCP segments in, tap messages out.

    Exchanges are assembled as bytes arrive, so an inference is tapped when
    its response ends rather than when its connection does — on this path
    clients hold pooled connections open indefinitely, so waiting for the
    close (or the idle flush) would delay every verification behind it.
    """
    processor = SegmentProcessor()
    for seg in segments:
        if not ports_allowed(
            seg.sport, seg.dport, config.FRAME_PROCESSOR_PORTS, config.FRAME_PROCESSOR_EXCLUDE_PORTS
        ):
            continue
        yield from processor.feed(seg)
    yield from processor.finish()


def _segments(frames) -> Iterator[TcpSegment]:
    for frame in frames:
        seg = decode_tcp(frame)
        if seg is not None:
            yield seg


def _interrupt_on_sigterm(_signum, _frame) -> None:
    """Turn Kubernetes termination into the pipeline's graceful-stop path."""
    raise KeyboardInterrupt


def _run_live_multiprocess(emitter: EmitterPool):
    """Run the packet-ring/process topology and restore signal state after it."""
    # Imported after SegmentProcessor is defined, avoiding the worker factory's
    # intentional lazy app.main import becoming a module-level cycle.
    from app.ledger_reporter import LedgerReporter
    from app.live_pipeline import LivePipeline

    ifaces = config.FRAME_PROCESSOR_IFACES or (default_route_iface(),)
    # Capture windows go straight to the ledger when it is configured;
    # otherwise they are logged and the machinery still runs.
    reporter = (
        LedgerReporter(
            config.FRAME_PROCESSOR_LEDGER_URL,
            timeout=config.FRAME_PROCESSOR_LEDGER_TIMEOUT,
            drain_seconds=config.FRAME_PROCESSOR_EMIT_DRAIN_SECONDS,
        )
        if config.FRAME_PROCESSOR_LEDGER_URL
        else None
    )
    pipeline = LivePipeline(
        ifaces=ifaces,
        emitter=emitter,
        worker_count=config.FRAME_PROCESSOR_FLOW_WORKERS,
        segment_queue_size=config.FRAME_PROCESSOR_SEGMENT_QUEUE_SIZE,
        message_queue_size=config.FRAME_PROCESSOR_MESSAGE_QUEUE_SIZE,
        include_ports=config.FRAME_PROCESSOR_PORTS,
        exclude_ports=config.FRAME_PROCESSOR_EXCLUDE_PORTS,
        stats_interval=config.FRAME_PROCESSOR_STATS_INTERVAL,
        window_sink=reporter.submit if reporter is not None else None,
        window_seconds=config.FRAME_PROCESSOR_WINDOW_SECONDS,
        window_flush_delay=config.FRAME_PROCESSOR_WINDOW_FLUSH_SECONDS,
        capture_host=config.FRAME_PROCESSOR_CAPTURE_HOST,
        tapped_hostname=config.FRAME_PROCESSOR_TAPPED_HOSTNAME,
        tap_version=config.FRAME_PROCESSOR_VERSION,
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        return pipeline.run()
    finally:
        # run() normally stops and drains itself. This second call is
        # idempotent and covers an unexpected parent-side exception as well.
        try:
            pipeline.stop()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            if reporter is not None:
                stats = reporter.close()
                log.info(
                    "Ledger reporter: submitted=%d delivered=%d duplicates=%d "
                    "failed=%d dropped=%d undelivered=%d",
                    stats.submitted,
                    stats.delivered,
                    stats.duplicates,
                    stats.failed,
                    stats.dropped,
                    stats.queued,
                )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if config.FRAME_PROCESSOR_MODE == "pcap":
        if not config.FRAME_PROCESSOR_PCAP_PATH:
            log.error("FRAME_PROCESSOR_MODE=pcap requires FRAME_PROCESSOR_PCAP_PATH")
            return 2
    elif config.FRAME_PROCESSOR_MODE == "live":
        pass
    else:
        log.error("Unknown FRAME_PROCESSOR_MODE %r (want pcap|live)", config.FRAME_PROCESSOR_MODE)
        return 2

    # decap is logged because a mismatch with the CNI's tunnel port is
    # otherwise indistinguishable from there being no traffic: the tap keeps
    # running and simply never tags anything.
    log.info(
        "frame-processor %s starting: mode=%s iface=%s emit=%s ports=%s exclude=%s "
        "decap=%s flow_workers=%s ring=%sx%s/%sB retire=%sms segment_queue=%s "
        "message_queue=%s fast_classify=%s emit_workers=%s emit_queue=%s "
        "account_all=%s whitelist_peer=%s iface_directions=%s windows=%ss ledger=%s "
        "capture_host=%s tapped=%s",
        config.FRAME_PROCESSOR_VERSION,
        config.FRAME_PROCESSOR_MODE,
        ",".join(config.FRAME_PROCESSOR_IFACES) if config.FRAME_PROCESSOR_IFACES else "auto",
        config.FRAME_PROCESSOR_EMIT_ENABLED,
        sorted(config.FRAME_PROCESSOR_PORTS) if config.FRAME_PROCESSOR_PORTS else "all",
        sorted(config.FRAME_PROCESSOR_EXCLUDE_PORTS) if config.FRAME_PROCESSOR_EXCLUDE_PORTS else "none",
        sorted(config.FRAME_PROCESSOR_DECAP_PORTS) if config.FRAME_PROCESSOR_DECAP_PORTS else "off",
        config.FRAME_PROCESSOR_FLOW_WORKERS,
        config.FRAME_PROCESSOR_RING_BLOCK_COUNT,
        config.FRAME_PROCESSOR_RING_BLOCK_SIZE,
        config.FRAME_PROCESSOR_RING_FRAME_SIZE,
        config.FRAME_PROCESSOR_RING_RETIRE_TIMEOUT_MS,
        config.FRAME_PROCESSOR_SEGMENT_QUEUE_SIZE,
        config.FRAME_PROCESSOR_MESSAGE_QUEUE_SIZE,
        config.FRAME_PROCESSOR_FAST_CLASSIFY,
        config.FRAME_PROCESSOR_EMIT_WORKERS,
        config.FRAME_PROCESSOR_EMIT_QUEUE_SIZE,
        config.FRAME_PROCESSOR_ACCOUNT_ALL,
        config.FRAME_PROCESSOR_PEER_MAC or "unset (whitelist off)",
        config.FRAME_PROCESSOR_IFACE_DIRECTIONS or "kernel packet type",
        config.FRAME_PROCESSOR_WINDOW_SECONDS or "off",
        config.FRAME_PROCESSOR_LEDGER_URL or "off (log only)",
        config.FRAME_PROCESSOR_CAPTURE_HOST,
        config.FRAME_PROCESSOR_TAPPED_HOSTNAME or "unset",
    )

    emitter = EmitterPool()
    count = 0
    pipeline_stats = None
    try:
        if config.FRAME_PROCESSOR_MODE == "pcap":
            frames = pcap_frames(config.FRAME_PROCESSOR_PCAP_PATH)
            for message in process_segments(_segments(frames)):
                emitter.submit(message)
                count += 1
        else:
            pipeline_stats = _run_live_multiprocess(emitter)
            count = pipeline_stats.messages_built
    finally:
        emit_stats = emitter.close()

    if pipeline_stats is not None:
        log.info(
            "Live pipeline: kernel received=%d dropped=%d freezes=%d "
            "truncated=%d malformed_blocks=%d; TCP seen=%d dispatched=%d "
            "processed=%d segment_queue_dropped=%d; messages built=%d "
            "message_queue_dropped=%d emitter_accepted=%d emitter_rejected=%d; "
            "emitter_errors=%d receiver_errors=%d worker_errors=%d "
            "process_failures=%d/%d forced=%d",
            pipeline_stats.kernel_received,
            pipeline_stats.kernel_dropped,
            pipeline_stats.kernel_freezes,
            pipeline_stats.truncated_packets,
            pipeline_stats.malformed_blocks,
            pipeline_stats.segments_seen,
            pipeline_stats.segments_dispatched,
            pipeline_stats.segments_processed,
            pipeline_stats.segment_queue_dropped,
            pipeline_stats.messages_built,
            pipeline_stats.message_queue_dropped,
            pipeline_stats.emitter_accepted,
            pipeline_stats.emitter_rejected,
            pipeline_stats.emitter_errors,
            pipeline_stats.receiver_errors,
            pipeline_stats.worker_errors,
            pipeline_stats.receiver_process_failures,
            pipeline_stats.worker_process_failures,
            pipeline_stats.forced_terminations,
        )
    log.info(
        "Done: %d LLM exchange(s) tapped; emit submitted=%d delivered=%d "
        "failed=%d dropped=%d high_water=%d",
        count,
        emit_stats.submitted,
        emit_stats.delivered,
        emit_stats.failed,
        emit_stats.dropped,
        emit_stats.high_water,
    )
    if pipeline_stats is not None and any(
        (
            pipeline_stats.receiver_errors,
            pipeline_stats.worker_errors,
            pipeline_stats.emitter_errors,
            pipeline_stats.receiver_process_failures,
            pipeline_stats.worker_process_failures,
            pipeline_stats.forced_terminations,
        )
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
