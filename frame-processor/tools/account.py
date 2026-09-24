#!/usr/bin/env python3
"""Account for every frame in one or more capture files.

    python3 tools/account.py captures/in-*.pcap

Not the service. frame-processor's runtime path is `app.main` → `live_pipeline` →
the accountant, and nothing under `app/` imports this file: it runs only when
someone invokes it. That is why it lives here rather than beside the modules
it drives — the direction of the dependency is the whole distinction, and a
reader should not have to grep for it.

It is not a development-only tool either, which is why it still ships in the
image. Two of its three modes belong on the capture host: `--live` accounts
for real interfaces alongside the deployed tap, and `--ledger` files a stored
capture as the same `capture_window` rows a live run would have written. Only
the plain replay is a desk tool.

Offline counterpart to the live accounting hook: the same `FrameAccountant`
that runs inside the packet-ring walk, driven from replayed frames instead.
Having both share one implementation is the point — a rule developed against
a stored window is then the rule enforced on the wire, rather than a
description of one.

Exit status is the reviewable claim, not decoration: 0 only when the account
balanced, nothing was dropped or truncated, and no finding was raised. Any
other outcome is a window that cannot support a statement about what crossed
the link, and says which of those it was.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

# This file drives `app/`, unlike its neighbour here, which stands entirely
# apart from it. Run as a script, sys.path[0] is `tools/`, so the package root
# has to go on the path before the imports below resolve. Redundant inside the
# image, where PYTHONPATH already covers it, and harmless there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.accounting import AccountingSnapshot, FrameAccountant  # noqa: E402
from app.capture import pcap_frames  # noqa: E402
from app.policy import (  # noqa: E402
    DIRECTION_IN,
    DIRECTION_OUT,
    DIRECTION_UNKNOWN,
    LinkPolicy,
)
from app.windows import WindowMerger, WindowReport  # noqa: E402

log = logging.getLogger("frameprocessor.account")


def _unpinned_classes() -> frozenset[str]:
    """The default exceptions, less anything this link pins anyway.

    A replay has to make the same choice the live tap does, or the digests it
    prints are not the ones a deployment would compare against.
    """
    from app import config
    from app.accounting import UNPINNED_CLASSES

    return UNPINNED_CLASSES - config.FRAME_PROCESSOR_PIN_CLASSES


def _config():
    from app import config

    return config


def _whole_frame_classes() -> frozenset[str]:
    """Classes this link pins from byte zero. Same reason as above: a replay
    that pinned a different region would print digests no deployment uses."""
    from app import config

    return config.FRAME_PROCESSOR_PIN_WHOLE_FRAME


def account_files(
    paths: Iterable[str],
    *,
    quiet_findings: bool = True,
    policy: LinkPolicy | None = None,
) -> AccountingSnapshot:
    """Replay every frame of every file through one accountant.

    A replayed frame carries no kernel direction bit, so with a policy the
    direction comes from the source MAC — which is what `unknown` is for.
    """
    if quiet_findings:
        # The report lists findings itself; the per-finding warning is noise
        # when replaying hours of capture in one go.
        logging.getLogger("frameprocessor.accounting").setLevel(logging.ERROR)
    snapshot, _windows = account_files_windowed(paths, policy=policy, window_seconds=None)
    return snapshot


def account_files_windowed(
    paths: Iterable[str],
    *,
    policy: LinkPolicy | None = None,
    window_seconds: float | None,
    iface: str = "pcap",
) -> tuple[AccountingSnapshot, list[WindowReport]]:
    """Replay files through one accountant, cutting windows as the live tap
    does. The windows are what ``--ledger`` posts, so a stored capture can be
    filed retrospectively with the same rows a live run would have written."""
    accountant = FrameAccountant(
        policy=policy,
        iface=iface,
        unpinned_classes=_unpinned_classes(),
        whole_frame_classes=_whole_frame_classes(),
        allow_ip_options=_config().FRAME_PROCESSOR_ALLOW_IP_OPTIONS,
        allow_fragments=_config().FRAME_PROCESSOR_ALLOW_FRAGMENTS,
        window_seconds=window_seconds,
        # A replay is not cut short by a process starting mid-window: the
        # first frame's own window is where the record begins.
        process_epoch=0.0,
    )
    last_ts: float | None = None
    for path in paths:
        for frame in pcap_frames(path):
            accountant.observe(frame.data, frame.ts)
            last_ts = frame.ts
    windows = accountant.take_closed()
    if window_seconds and last_ts is not None:
        windows.extend(accountant.finish(last_ts))
    return accountant.snapshot(), windows


def account_live(
    ifaces: list[str],
    *,
    seconds: float,
    quiet_findings: bool = True,
    policy: LinkPolicy | None = None,
) -> AccountingSnapshot:
    """Account for live traffic without emitting anything anywhere.

    Exists so the ring-plus-observer path can be exercised on the real capture
    host without running the pipeline: this reads frames, counts them, and
    stops. It never builds a tap message, never contacts the message writer,
    and never writes to the ledger, so it can run alongside the deployed
    frame-processor rather than in place of it.

    That does mean two AF_PACKET readers on the same interfaces. Fine here —
    the kernel copies to each independently, so neither starves the other, and
    a spare reader costs only its own drop counter. It is *not* how the
    production account should be taken: two readers give two sets of drop
    figures that cannot be reconciled into one statement about the link, which
    is why the pipeline attaches the accountant to the reader it already has.
    """
    if quiet_findings:
        logging.getLogger("frameprocessor.accounting").setLevel(logging.ERROR)

    # Lazy: RingCapture needs AF_PACKET, so importing it must not be a
    # condition of using the replay path on a development machine.
    import select
    import time

    from app.capture import DEFAULT_RING_READ_BLOCKS, RingCapture

    from app import config
    from app.policy import FixedDirection

    # One accountant across every interface here, unlike the pipeline's one
    # per receiver — so no single fixed direction applies and every declared
    # beat is armed. That is right for a report covering all the ports at once.
    accountant = FrameAccountant(
        policy=policy,
        unpinned_classes=_unpinned_classes(),
        whole_frame_classes=_whole_frame_classes(),
        allow_ip_options=_config().FRAME_PROCESSOR_ALLOW_IP_OPTIONS,
        allow_fragments=_config().FRAME_PROCESSOR_ALLOW_FRAGMENTS,
    )
    sources = []
    for iface in ifaces:
        # Same rule as the live pipeline: a declared monitor-port direction
        # beats the kernel bit, which on a monitor port is always "received".
        fixed = config.FRAME_PROCESSOR_IFACE_DIRECTIONS.get(iface)
        observer = FixedDirection(accountant, fixed) if fixed else accountant
        log.info("%s direction=%s", iface, f"fixed {fixed}" if fixed else "kernel packet type")
        sources.append(RingCapture(iface, observer=observer))
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            for source in sources:
                # Segments are discarded on purpose. The walk is what fires the
                # observer, and reconstructing exchanges is not this tool's job.
                source.read_segments(max_blocks=DEFAULT_RING_READ_BLOCKS)
            select.select(sources, [], [], min(1.0, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        log.info("interrupted; reporting on what was seen so far")
    finally:
        dropped = 0
        for source in sources:
            try:
                dropped += source.poll_stats().dropped
            except Exception:  # noqa: BLE001 -- shutdown diagnostics are best effort
                log.exception("could not read final ring stats for %s", source.iface)
            source.close()
        # Applied once, cumulatively: poll_stats self-resets.
        accountant.note_kernel_dropped(dropped)
    return accountant.snapshot()


def _account_files_to_ledger(
    paths: list[str], policy: LinkPolicy | None, ledger_url: str
) -> AccountingSnapshot:
    """Replay, cut into windows, post every window to the ledger, and wait."""
    from app import config
    from app.ledger_reporter import LedgerReporter

    logging.getLogger("frameprocessor.accounting").setLevel(logging.ERROR)
    window_seconds = float(config.FRAME_PROCESSOR_WINDOW_SECONDS or 300)
    snapshot, windows = account_files_windowed(paths, policy=policy, window_seconds=window_seconds)
    merger = WindowMerger(
        ifaces=["pcap"],
        window_seconds=window_seconds,
        flush_delay=0.0,
        capture_host=config.FRAME_PROCESSOR_CAPTURE_HOST,
        tapped_hostname=config.FRAME_PROCESSOR_TAPPED_HOSTNAME,
        tap_version=config.FRAME_PROCESSOR_VERSION,
        max_groups=config.FRAME_PROCESSOR_WINDOW_MAX_GROUPS,
    )
    for report in windows:
        merger.add_report(report)
    rows = merger.finish()
    reporter = LedgerReporter(
        ledger_url,
        timeout=config.FRAME_PROCESSOR_LEDGER_TIMEOUT,
        queue_size=max(64, len(rows)),
        drain_seconds=max(30.0, 2.0 * len(rows)),
    )
    for row in rows:
        reporter.submit(row)
    stats = reporter.close()
    log.info(
        "posted %d capture window(s) to %s: delivered=%d duplicates=%d failed=%d",
        len(rows), ledger_url, stats.delivered, stats.duplicates, stats.failed,
    )
    return snapshot


def _render(
    snapshot: AccountingSnapshot,
    *,
    top_findings: int = 20,
    policy: LinkPolicy | None = None,
) -> str:
    lines: list[str] = []
    total_bytes = sum(t.frame_bytes for t in snapshot.classes.values())
    directions = snapshot.by_direction()

    lines.append("── link whitelist ──")
    lines.append(f"  {policy.describe() if policy else 'off (FRAME_PROCESSOR_PEER_MAC unset)'}")
    lines.append("")
    lines.append("── classes ──")
    lines.append(
        f"  {'class':22} {'frames':>9} {'bytes':>12}   share"
        f"  {'out':>8} {'in':>8} {'unknown':>8}"
    )
    for name, totals in sorted(
        snapshot.classes.items(), key=lambda kv: -kv[1].frames
    ):
        share = 100 * totals.frames / snapshot.observed if snapshot.observed else 0
        lines.append(
            f"  {name:22} {totals.frames:9d} {totals.frame_bytes:12d}   {share:5.1f}%"
            f"  {totals.frames_out:8d} {totals.frames_in:8d} {totals.frames_unknown:8d}"
        )
    lines.append(
        f"  {'TOTAL':22} {snapshot.classified:9d} {total_bytes:12d}         "
        f"  {directions[DIRECTION_OUT].frames:8d} {directions[DIRECTION_IN].frames:8d}"
        f" {directions[DIRECTION_UNKNOWN].frames:8d}"
    )

    if snapshot.pins:
        lines.append("")
        lines.append("── pins (source / class → distinct payloads) ──")
        lines.append(
            f"  {'source / class':44} {'frames':>8} {'region':>8} {'bytes':>6}"
            f" {'distinct':>9}  first digest      declared"
        )
        for key, pin in sorted(snapshot.pins.items(), key=lambda kv: -kv[1].frames):
            distinct = f"{pin.distinct}{'+' if pin.overflowed else ''}"
            flag = "" if pin.constant else "  ← varies"
            if pin.declared is None:
                declared = "—" if policy is None else "NO"
            else:
                declared = "match" if pin.declared == pin.digest and pin.constant else "MISMATCH"
            region = "frame" if pin.whole_frame else "payload"
            lines.append(
                f"  {key:44} {pin.frames:8d} {region:>8} {pin.payload_bytes:6d}"
                f" {distinct:>9}  {pin.digest}  {declared}{flag}"
            )

    if snapshot.beats:
        lines.append("")
        lines.append("── beats (source / class → declared cadence) ──")
        lines.append(
            f"  {'source / class':44} {'declared':>9} {'seen':>6} {'missed':>7}"
            f" {'extra':>6}  {'mean':>7} {'min':>7} {'max':>7}"
        )
        for key, beat in sorted(snapshot.beats.items()):
            lines.append(
                f"  {key:44} {beat.interval:8.0f}s {beat.beats:6d} {beat.missed:7d}"
                f" {beat.unscheduled:6d}  {beat.mean_gap:6.1f}s {beat.min_gap:6.1f}s"
                f" {beat.max_gap:6.1f}s"
            )

    lines.append("")
    lines.append("── account ──")
    lines.append(f"  observed        {snapshot.observed}")
    lines.append(f"  classified      {snapshot.classified}")
    lines.append(f"  balanced        {snapshot.balanced}")
    lines.append(f"  kernel dropped  {snapshot.kernel_dropped}")
    lines.append(f"  truncated       {snapshot.truncated}")
    lines.append(f"  errors          {snapshot.errors}")
    lines.append(f"  findings        {snapshot.findings} "
                 f"in {len(snapshot.finding_groups)} group(s)")
    lines.append(f"  COMPLETE        {snapshot.complete}")

    if snapshot.finding_groups:
        lines.append("")
        lines.append("── findings, by kind and class ──")
        lines.append(f"  {'kind':18} {'class':16} {'count':>8}  sources")
        groups = sorted(snapshot.finding_groups.values(), key=lambda g: -g.count)
        for group in groups:
            sources = ", ".join(sorted(group.sources)[:3]) or "?"
            if len(group.sources) > 3:
                sources += f" (+{len(group.sources) - 3} more)"
            lines.append(
                f"  {group.kind:18} {group.frame_class:16} {group.count:8d}  {sources}"
            )
        shown = 0
        lines.append("")
        lines.append("── finding samples ──")
        for group in groups:
            for finding in group.samples:
                if shown >= top_findings:
                    break
                lines.append(
                    f"  [{finding.kind}] {finding.frame_class} "
                    f"from {finding.source or '?'}: {finding.detail}"
                )
                shown += 1

    return "\n".join(lines)


def _as_json(snapshot: AccountingSnapshot) -> str:
    return json.dumps(
        {
            "observed": snapshot.observed,
            "classified": snapshot.classified,
            "balanced": snapshot.balanced,
            "complete": snapshot.complete,
            "kernel_dropped": snapshot.kernel_dropped,
            "truncated": snapshot.truncated,
            "errors": snapshot.errors,
            "classes": {
                name: {
                    "frames": t.frames,
                    "bytes": t.frame_bytes,
                    "out": t.frames_out,
                    "in": t.frames_in,
                    "unknown": t.frames_unknown,
                }
                for name, t in snapshot.classes.items()
            },
            "directions": {
                name: t.frames for name, t in snapshot.by_direction().items()
            },
            "pins": {
                key: {
                    "frames": pin.frames,
                    "payload_bytes": pin.payload_bytes,
                    "distinct": pin.distinct,
                    "overflowed": pin.overflowed,
                    "constant": pin.constant,
                    "digest": pin.digest,
                    "declared": pin.declared,
                    "first_seen": pin.first_seen,
                }
                for key, pin in snapshot.pins.items()
            },
            "findings": snapshot.findings,
            "finding_groups": [
                {
                    "kind": group.kind,
                    "class": group.frame_class,
                    "count": group.count,
                    "sources": sorted(group.sources),
                    "samples": [f.detail for f in group.samples],
                }
                for group in sorted(
                    snapshot.finding_groups.values(), key=lambda g: -g.count
                )
            ],
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools/account.py",
        description="Account for every frame in one or more capture files.",
    )
    parser.add_argument("paths", nargs="*", metavar="PCAP")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument(
        "--findings", type=int, default=20, metavar="N",
        help="how many findings to list (default 20)",
    )
    parser.add_argument(
        "--live", metavar="IFACE",
        help="account for live traffic on these interfaces (comma-separated) "
             "instead of replaying files. Emits nothing; needs NET_RAW.",
    )
    parser.add_argument(
        "--seconds", type=float, default=300.0, metavar="S",
        help="how long to capture in --live mode (default 300)",
    )
    parser.add_argument(
        "--ledger", metavar="URL",
        help="also cut the replay into capture windows and POST them to this "
             "ledger-api, exactly as the live tap would (PCAP mode only). "
             "Window length, capture host and tapped hostname come from the "
             "FRAME_PROCESSOR_* environment.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.live:
        if args.paths:
            parser.error("--live accounts for interfaces, so it takes no PCAP paths")
        ifaces = [name.strip() for name in args.live.split(",") if name.strip()]
        if not ifaces:
            parser.error("--live needs at least one interface")
        # One policy for all interfaces; the local MAC is read from the first.
        policy = LinkPolicy.from_config(iface=ifaces[0])
        log.info("accounting live on %s for %.0fs", ", ".join(ifaces), args.seconds)
        snapshot = account_live(ifaces, seconds=args.seconds, policy=policy)
    elif not args.paths:
        parser.error("give one or more PCAP paths, or --live IFACE[,IFACE]")
    elif args.ledger:
        policy = LinkPolicy.from_config()
        snapshot = _account_files_to_ledger(args.paths, policy, args.ledger)
    else:
        # The whitelist comes from the same FRAME_PROCESSOR_* environment the live
        # tap reads, so a rule tried here is the rule that will be enforced.
        policy = LinkPolicy.from_config()
        snapshot = account_files(args.paths, policy=policy)
    print(
        _as_json(snapshot)
        if args.json
        else _render(snapshot, top_findings=args.findings, policy=policy)
    )
    return 0 if snapshot.complete else 1


if __name__ == "__main__":
    sys.exit(main())
