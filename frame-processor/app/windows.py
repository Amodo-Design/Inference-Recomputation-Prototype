"""Capture windows — the unit in which the link's account becomes a record.

`accounting.py` keeps a running total for the life of a process, which is the
right shape for a log line and the wrong shape for a ledger: a verifier
looking at one inference wants to know whether the link was clean *while
that inference crossed it*, not since the tap last restarted. So the
accountant also cuts its account into fixed, wall-clock-aligned windows,
and every closed window becomes one row.

The row must exist when nothing happened. A findings-only record cannot tell
"clean" from "the capture was down" — and the capture being down while
inference continues is the failure that matters most. A window
with zero frames is therefore still reported, and a window that is *missing*
means the tap was blind for it.

Two kinds of message travel from child processes to the parent here:

- `WindowReport`: one interface's account of one window, from a receiver.
- `ExchangeFinding`: something a flow worker decided about a reassembled
  exchange (an undeclared path on the inference port, a stream that never
  parsed as HTTP). These are decided after reassembly, in a different process
  from the one that counted the frames, so they cannot be attributed to a
  frame class — they are attributed to the window their timestamp falls in.

`WindowMerger` in the parent folds both into one row per window: the
interfaces' counts summed (a two-port tap sees each direction on a different
port, and the row is about the link, not a port), the findings from every
source under one cap, and `complete` only if every interface reported and
every one of them was clean.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("frameprocessor.windows")

# Finding kinds this module raises itself, as opposed to relaying.
KIND_CAPTURE_GAP = "capture-gap"
# Frame class given to exchange-level findings: they are about a reassembled
# HTTP exchange rather than any one frame.
CLASS_EXCHANGE = "http-exchange"

DEFAULT_SAMPLES_PER_GROUP = 3
DEFAULT_MAX_GROUPS = 64
DEFAULT_SAMPLE_FRAME_BYTES = 2048


def window_start_for(ts: float, window_seconds: float) -> float:
    """The aligned start of the window containing ``ts``."""
    return (ts // window_seconds) * window_seconds


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class FindingSample:
    ts: float
    detail: str
    # First bytes of the offending frame, when there was one. Bounded by the
    # accountant (``sample_frame_bytes``); an exchange finding has none.
    frame: bytes = b""

    def to_payload(self) -> dict[str, Any]:
        return {
            "ts": _iso(self.ts),
            "detail": self.detail,
            "frame_b64": base64.b64encode(self.frame).decode("ascii") if self.frame else None,
        }


@dataclass
class WindowFindingGroup:
    """Every finding of one kind, class, direction and sender in one window."""

    kind: str
    frame_class: str
    direction: str
    source: str  # rendered MAC, or '' when the finding has no sender
    count: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    samples: list[FindingSample] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.kind, self.frame_class, self.direction, self.source)

    def add(self, ts: float, detail: str, frame: bytes, *, samples_per_group: int) -> None:
        if self.count == 0:
            self.first_ts = self.last_ts = ts
        else:
            self.first_ts = min(self.first_ts, ts)
            self.last_ts = max(self.last_ts, ts)
        self.count += 1
        if len(self.samples) < samples_per_group:
            self.samples.append(FindingSample(ts, detail, frame))

    def merge(self, other: WindowFindingGroup, *, samples_per_group: int) -> None:
        if self.count == 0:
            self.first_ts, self.last_ts = other.first_ts, other.last_ts
        else:
            self.first_ts = min(self.first_ts, other.first_ts)
            self.last_ts = max(self.last_ts, other.last_ts)
        self.count += other.count
        for sample in other.samples:
            if len(self.samples) >= samples_per_group:
                break
            self.samples.append(sample)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "frame_class": self.frame_class,
            "direction": self.direction,
            "source_mac": self.source or None,
            "count": self.count,
            "first_ts": _iso(self.first_ts),
            "last_ts": _iso(self.last_ts),
            "samples": [sample.to_payload() for sample in self.samples],
        }


@dataclass
class WindowReport:
    """One interface's account of one window. Picklable: it crosses a queue."""

    iface: str
    window_start: float
    window_end: float
    process_epoch: float
    observed: int = 0
    classified: int = 0
    kernel_dropped: int = 0
    truncated: int = 0
    errors: int = 0
    # {class: {"frames", "bytes", "out", "in", "unknown"}}
    classes: dict[str, dict[str, int]] = field(default_factory=dict)
    # {source/class: {"digest", "declared", "frames", "distinct", "payload_b64"}}
    pins: dict[str, dict[str, Any]] = field(default_factory=dict)
    groups: list[WindowFindingGroup] = field(default_factory=list)
    # Findings that arrived after the group cap was reached: counted, not kept.
    groups_overflow: int = 0
    finding_count: int = 0

    @property
    def balanced(self) -> bool:
        return self.observed == self.classified

    @property
    def complete(self) -> bool:
        return self.balanced and self.kernel_dropped == 0 and self.finding_count == 0


@dataclass(frozen=True)
class ExchangeFinding:
    """A flow worker's verdict on a reassembled exchange. Picklable."""

    kind: str  # unexpected-exchange | non-http-stream
    detail: str
    ts: float
    observed_on: str  # "<ip>:<port>" the exchange was addressed to


class _Pending:
    """Everything received so far about one window."""

    def __init__(self, window_start: float, window_end: float) -> None:
        self.window_start = window_start
        self.window_end = window_end
        self.reports: dict[str, WindowReport] = {}
        self.exchange_findings: list[ExchangeFinding] = []


class WindowMerger:
    """Parent-side: one row per window from many child reports.

    A window is flushed as soon as every interface has reported it and it is
    old enough for the workers' exchange findings to have caught up, or after
    ``flush_delay`` past its end regardless — an interface that never reports
    a window is itself the finding, not a reason to wait forever.
    """

    def __init__(
        self,
        *,
        ifaces: list[str],
        window_seconds: float,
        flush_delay: float,
        capture_host: str,
        tapped_hostname: str | None,
        tap_version: str,
        max_groups: int = DEFAULT_MAX_GROUPS,
        samples_per_group: int = DEFAULT_SAMPLES_PER_GROUP,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.ifaces = tuple(dict.fromkeys(ifaces))
        self.window_seconds = window_seconds
        self.flush_delay = flush_delay
        self.capture_host = capture_host
        self.tapped_hostname = tapped_hostname
        self.tap_version = tap_version
        self.max_groups = max_groups
        self.samples_per_group = samples_per_group
        self._pending: dict[float, _Pending] = {}
        self._last_flushed: float | None = None
        self.windows_flushed = 0

    # --- intake ------------------------------------------------------------

    def _pending_for(self, window_start: float) -> _Pending:
        pending = self._pending.get(window_start)
        if pending is None:
            pending = self._pending[window_start] = _Pending(
                window_start, window_start + self.window_seconds
            )
        return pending

    def add_report(self, report: WindowReport) -> None:
        if self._last_flushed is not None and report.window_start <= self._last_flushed:
            # A receiver reporting a window the parent already flushed means
            # the flush delay is too short for this machine. Say so once per
            # occurrence; the row is already written and cannot grow.
            log.error(
                "%s reported window %s after it was flushed; raise "
                "FRAME_PROCESSOR_WINDOW_FLUSH_SECONDS (%d findings, %d frames not recorded)",
                report.iface,
                _iso(report.window_start),
                report.finding_count,
                report.observed,
            )
            return
        self._pending_for(report.window_start).reports[report.iface] = report

    def add_exchange_finding(self, finding: ExchangeFinding, now: float | None = None) -> None:
        start = window_start_for(finding.ts, self.window_seconds)
        if self._last_flushed is not None and start <= self._last_flushed:
            # Late: the worker reached this exchange after its window was
            # written. Filed against the current window rather than dropped,
            # with the original time kept in the sample, so it is still on
            # the record — slightly misplaced beats absent.
            start = window_start_for(now if now is not None else finding.ts, self.window_seconds)
            start = max(start, self._last_flushed + self.window_seconds)
        self._pending_for(start).exchange_findings.append(finding)

    # --- output -------------------------------------------------------------

    def flush(self, now: float) -> list[dict[str, Any]]:
        """Rows for every window that is ready, oldest first."""
        ready = sorted(
            start
            for start, pending in self._pending.items()
            if now >= pending.window_end + self.flush_delay
        )
        return [self._emit(self._pending.pop(start)) for start in ready]

    def finish(self) -> list[dict[str, Any]]:
        """Rows for everything still pending, at shutdown."""
        starts = sorted(self._pending)
        return [self._emit(self._pending.pop(start)) for start in starts]

    def _emit(self, pending: _Pending) -> dict[str, Any]:
        self._last_flushed = (
            pending.window_start
            if self._last_flushed is None
            else max(self._last_flushed, pending.window_start)
        )
        self.windows_flushed += 1

        groups: dict[tuple[str, str, str, str], WindowFindingGroup] = {}
        overflow = 0
        finding_count = 0

        def take(group: WindowFindingGroup) -> None:
            nonlocal overflow
            existing = groups.get(group.key)
            if existing is not None:
                existing.merge(group, samples_per_group=self.samples_per_group)
            elif len(groups) < self.max_groups:
                groups[group.key] = WindowFindingGroup(
                    group.kind, group.frame_class, group.direction, group.source,
                    group.count, group.first_ts, group.last_ts,
                    list(group.samples[: self.samples_per_group]),
                )
            else:
                overflow += group.count

        observed = classified = dropped = truncated = errors = 0
        classes: dict[str, dict[str, int]] = {}
        pins: dict[str, dict[str, Any]] = {}
        epochs: list[float] = []
        for report in pending.reports.values():
            observed += report.observed
            classified += report.classified
            dropped += report.kernel_dropped
            truncated += report.truncated
            errors += report.errors
            finding_count += report.finding_count
            overflow += report.groups_overflow
            epochs.append(report.process_epoch)
            for name, totals in report.classes.items():
                merged = classes.setdefault(
                    name, {"frames": 0, "bytes": 0, "out": 0, "in": 0, "unknown": 0}
                )
                for field_name, value in totals.items():
                    merged[field_name] = merged.get(field_name, 0) + value
            for key, pin in report.pins.items():
                # Two monitor ports never see the same sender/class, so a
                # collision here is one interface being reported twice; keep
                # the first and let the frame count say so.
                pins.setdefault(key, pin)
            for group in report.groups:
                take(group)

        # An interface that did not report this window is the loudest finding
        # there is: for that direction, the tap was blind.
        for iface in self.ifaces:
            if iface not in pending.reports:
                gap = WindowFindingGroup(KIND_CAPTURE_GAP, "capture", "unknown", "")
                gap.add(
                    pending.window_start,
                    f"no account from {iface} for this window",
                    b"",
                    samples_per_group=self.samples_per_group,
                )
                finding_count += 1
                take(gap)

        for finding in pending.exchange_findings:
            group = WindowFindingGroup(finding.kind, CLASS_EXCHANGE, "unknown", "")
            group.add(
                finding.ts,
                f"{finding.observed_on}: {finding.detail}",
                b"",
                samples_per_group=self.samples_per_group,
            )
            finding_count += 1
            take(group)

        complete = (
            observed == classified
            and dropped == 0
            and finding_count == 0
            and len(pending.reports) == len(self.ifaces)
        )
        return {
            "capture_host": self.capture_host,
            "ifaces": ",".join(self.ifaces),
            "tapped_hostname": self.tapped_hostname,
            "window_start": _iso(pending.window_start),
            "window_end": _iso(pending.window_end),
            "tap_version": self.tap_version,
            # A replay has no process epoch (accountant built with 0.0).
            "process_epoch": _iso(min(epochs)) if epochs and min(epochs) > 0 else None,
            "observed": observed,
            "classified": classified,
            "kernel_dropped": dropped,
            "truncated": truncated,
            "errors": errors,
            "finding_count": finding_count,
            "finding_groups": len(groups),
            "finding_groups_overflow": overflow,
            "complete": complete,
            "classes": classes,
            "pins": pins,
            "findings": [group.to_payload() for group in groups.values()],
        }
