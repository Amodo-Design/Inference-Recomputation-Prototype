// Was the tapped link fully accounted for while this inference crossed it.
//
// A second axis beside the verdict, styled as a different family from the
// pass/fail pills (outlined, not filled) so it is never read as one. The
// replay verdict stays true when the link was noisy; this says whether the
// link was quiet.
export type CaptureStatus = "complete" | "tainted" | "partial" | "uncovered";

// frame-processor writes a window up to a window length plus its flush delay after
// the frames it covers. An inference newer than this that reads "uncovered"
// is simply not written yet.
const PENDING_WINDOW_MS = 6 * 60 * 1000;

export const CAPTURE_STATUS_LABEL: Record<CaptureStatus, string> = {
  complete: "complete",
  tainted: "tainted",
  partial: "partial",
  uncovered: "no tap"
};

export const CAPTURE_STATUS_TITLE: Record<CaptureStatus, string> = {
  complete: "Every frame on the link during this inference was accounted for.",
  tainted: "The link was covered, but a window had drops, findings or a missing interface.",
  partial: "Capture windows cover only part of this inference's span.",
  uncovered: "No capture window overlaps this inference: the tap was not watching."
};

export function isPending(status: CaptureStatus | null | undefined, ts: string): boolean {
  return status === "uncovered" && Date.now() - new Date(ts).getTime() < PENDING_WINDOW_MS;
}

export default function CapturePill({
  status,
  findings,
  ts,
  href
}: {
  status: CaptureStatus | null | undefined;
  findings: number;
  ts: string;
  href?: string;
}) {
  if (!status) {
    return <span className="capture capture-unknown">-</span>;
  }
  const pending = isPending(status, ts);
  const label = pending ? "pending" : CAPTURE_STATUS_LABEL[status];
  const title = pending
    ? "The capture window for this inference has not been written yet."
    : CAPTURE_STATUS_TITLE[status];
  const className = `capture capture-${pending ? "pending" : status}`;
  const body = (
    <>
      {label}
      {findings > 0 && !pending ? <span className="capture-count">{findings}</span> : null}
    </>
  );
  return href ? (
    <a className={className} href={href} title={title}>
      {body}
    </a>
  ) : (
    <span className={className} title={title}>
      {body}
    </span>
  );
}
