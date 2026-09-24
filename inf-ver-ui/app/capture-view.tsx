import {
  DEFAULT_PAGE_SIZE,
  LEDGER_BASE,
  SearchParams,
  firstValue,
  formatDate,
  getJson,
  parsePage,
  parsePageSize
} from "./lib";
import PageSizeSelect from "./page-size-select";

// The Capture tab: frame-processor's account of the tapped link, one row per
// five-minute window, drilling down window → findings → sampled frames.
//
// Three levels, one URL each, all server-rendered from the ledger:
//   /?tab=capture                      the windows
//   /?tab=capture&event=<inference id> the windows that decided one
//                                      inference's capture status
//   /?tab=capture&window=<id>          one window and its findings
//   /?tab=capture&window=<id>&finding=<id>   one finding and its samples
//
// The thing this page is best at showing is ABSENCE: a window that is not
// there means the tap was blind for it, so gaps in the sequence are drawn as
// rows rather than left as a shorter list.

type CaptureWindowRead = {
  window_id: string;
  capture_host: string;
  ifaces: string;
  hardware_id: string | null;
  window_start: string;
  window_end: string;
  tap_version: string;
  process_epoch: string | null;
  observed: number;
  classified: number;
  kernel_dropped: number;
  truncated: number;
  errors: number;
  finding_count: number;
  finding_groups: number;
  finding_groups_overflow: number;
  complete: boolean;
  classes: Record<string, { frames: number; bytes: number; out: number; in: number; unknown: number }>;
  pins: Record<string, { digest: string; declared: string | null; frames: number; distinct: number; payload_b64: string | null }>;
};

type CaptureWindowPage = {
  items: CaptureWindowRead[];
  total: number;
  limit: number;
  offset: number;
};

type CaptureSample = { ts: string; detail: string; frame_b64: string | null };

type CaptureFindingRead = {
  finding_id: string;
  window_id: string;
  kind: string;
  frame_class: string;
  direction: "in" | "out" | "unknown";
  source_mac: string | null;
  count: number;
  first_ts: string;
  last_ts: string;
  samples: CaptureSample[];
};

type CaptureWindowDetail = CaptureWindowRead & { findings: CaptureFindingRead[] };

type InferenceEventRead = {
  id: string;
  session_id: string;
  ts: string;
  started_at: string | null;
  hardware_id: string;
};

type InferenceEventCapture = {
  inference_event_id: string;
  capture_status: "complete" | "tainted" | "partial" | "uncovered";
  windows: number;
  span_seconds: number;
  covered_seconds: number;
  kernel_dropped: number;
  findings: number;
};

// The inference whose windows are being shown, when the page was reached
// from a Capture pill.
type EventFocus = {
  event: InferenceEventRead;
  capture: InferenceEventCapture;
};

function href(parts: Record<string, string | number | undefined>): string {
  const params = new URLSearchParams({ tab: "capture" });
  for (const [key, value] of Object.entries(parts)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  return `/?${params.toString()}`;
}

function windowSeconds(window: CaptureWindowRead): number {
  return (new Date(window.window_end).getTime() - new Date(window.window_start).getTime()) / 1000;
}

function ageLabel(ms: number): string {
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return "under a minute ago";
  if (minutes < 120) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.round(hours / 24)} d ago`;
}

function classSummary(classes: CaptureWindowRead["classes"]): string {
  const entries = Object.entries(classes).sort((a, b) => b[1].frames - a[1].frames);
  if (entries.length === 0) return "no frames";
  return entries.map(([name, totals]) => `${name} ${totals.frames.toLocaleString()}`).join(" · ");
}

function hexDump(b64: string): string {
  const bytes = Buffer.from(b64, "base64");
  const lines: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 16) {
    const chunk = bytes.subarray(offset, offset + 16);
    const hex = Array.from(chunk, (b) => b.toString(16).padStart(2, "0"));
    const left = hex.slice(0, 8).join(" ");
    const right = hex.slice(8).join(" ");
    const ascii = Array.from(chunk, (b) => (b >= 0x20 && b < 0x7f ? String.fromCharCode(b) : ".")).join("");
    lines.push(
      `${offset.toString(16).padStart(4, "0")}  ${left.padEnd(23)}  ${right.padEnd(23)}  |${ascii}|`
    );
  }
  return lines.join("\n");
}

// Rows for the windows table, with a placeholder row wherever the sequence
// skips one or more window slots. Items arrive newest first.
type WindowRow =
  | { kind: "window"; window: CaptureWindowRead }
  | { kind: "missing"; from: string; to: string; count: number };

function withGaps(items: CaptureWindowRead[]): WindowRow[] {
  const rows: WindowRow[] = [];
  for (let i = 0; i < items.length; i += 1) {
    const window = items[i];
    rows.push({ kind: "window", window });
    const older = items[i + 1];
    if (!older) continue;
    const length = windowSeconds(window) * 1000;
    const gap = new Date(window.window_start).getTime() - new Date(older.window_end).getTime();
    if (length > 0 && gap >= length) {
      rows.push({
        kind: "missing",
        from: older.window_end,
        to: window.window_start,
        count: Math.round(gap / length)
      });
    }
  }
  return rows;
}

function DirectionLabel({ direction }: { direction: CaptureFindingRead["direction"] }) {
  const label = direction === "out" ? "out of node" : direction === "in" ? "into node" : "unknown";
  return <span className={`direction direction-${direction}`}>{label}</span>;
}

export default async function CaptureView({ params }: { params: SearchParams }) {
  const windowId = firstValue(params.window);
  const findingId = firstValue(params.finding);
  const page = parsePage(params.cpage);
  const limit = parsePageSize(params.climit);
  const offset = (page - 1) * limit;
  const incompleteOnly = firstValue(params.incomplete) === "1";
  const eventId = firstValue(params.event);
  let since = firstValue(params.since);
  let until: string | undefined;

  let latest: CaptureWindowRead | null = null;
  let windows: CaptureWindowPage | null = null;
  let detail: CaptureWindowDetail | null = null;
  let focus: EventFocus | null = null;
  try {
    if (eventId && !windowId) {
      // Reached from a Capture pill: show exactly the windows that overlap
      // the inference's span on the link, which are the ones that decided
      // its status. The span is the inference's, not the verification's.
      const [event, capture] = await Promise.all([
        getJson<InferenceEventRead>(LEDGER_BASE, `/inference-events/${encodeURIComponent(eventId)}`),
        getJson<InferenceEventCapture>(LEDGER_BASE, `/inference-events/${encodeURIComponent(eventId)}/capture`)
      ]);
      focus = { event, capture };
      since = event.started_at ?? event.ts;
      until = event.ts;
    }
    const latestPromise = getJson<CaptureWindowRead>(LEDGER_BASE, "/capture-windows/latest").catch(
      (error: unknown) => {
        // 404 = no windows yet, which the page says in words below.
        if (error instanceof Error && /404/.test(error.message)) return null;
        throw error;
      }
    );
    if (windowId) {
      [latest, detail] = await Promise.all([
        latestPromise,
        getJson<CaptureWindowDetail>(LEDGER_BASE, `/capture-windows/${encodeURIComponent(windowId)}`)
      ]);
    } else {
      const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });
      if (incompleteOnly) query.set("complete", "false");
      if (since) query.set("since", since);
      if (until) query.set("until", until);
      [latest, windows] = await Promise.all([
        latestPromise,
        getJson<CaptureWindowPage>(LEDGER_BASE, `/capture-windows?${query.toString()}`)
      ]);
    }
  } catch (error) {
    return (
      <div className="error">
        Could not load capture windows from {LEDGER_BASE}:{" "}
        {error instanceof Error ? error.message : "unknown error"}
      </div>
    );
  }

  const latestAgeMs = latest ? Date.now() - new Date(latest.window_end).getTime() : null;
  const latestLength = latest ? windowSeconds(latest) * 1000 : 0;
  const tapStale = latestAgeMs !== null && latestAgeMs > 2 * latestLength + 60_000;

  return (
    <>
      <section className="stats capture-stats" aria-label="Capture summary">
        <div className="stat">
          <span>Last window</span>
          <strong className={tapStale ? "accent-fail" : undefined}>
            {latestAgeMs === null ? "none" : ageLabel(latestAgeMs)}
          </strong>
          <div className="muted">
            {latest
              ? tapStale
                ? "The tap has stopped reporting."
                : `${latest.capture_host} · ${latest.ifaces}`
              : "No capture window has been recorded yet."}
          </div>
        </div>
        <div className="stat">
          <span>Last window state</span>
          <strong className={latest ? (latest.complete ? "accent-pass" : "accent-fail") : undefined}>
            {latest ? (latest.complete ? "complete" : "incomplete") : "-"}
          </strong>
          <div className="muted">
            {latest ? `${latest.observed.toLocaleString()} frames · ${latest.finding_count.toLocaleString()} findings` : ""}
          </div>
        </div>
        <div className="stat">
          <span>Tap version</span>
          <strong>{latest?.tap_version ?? "-"}</strong>
          <div className="muted">{latest?.process_epoch ? `capture up since ${formatDate(latest.process_epoch)}` : ""}</div>
        </div>
      </section>

      {detail ? (
        <WindowDetail detail={detail} findingId={findingId} />
      ) : windows ? (
        <>
          {focus ? <EventFocusBanner focus={focus} /> : null}
          <WindowsTable
            windows={windows}
            page={page}
            limit={limit}
            incompleteOnly={incompleteOnly}
            since={focus ? undefined : since}
            eventId={focus ? focus.event.id : undefined}
          />
        </>
      ) : null}
    </>
  );
}

function EventFocusBanner({ focus }: { focus: EventFocus }) {
  const { event, capture } = focus;
  const start = event.started_at ?? event.ts;
  return (
    <div className="notice capture-focus">
      <div>
        <strong>Inference {event.id}</strong>
        <span className={`capture capture-${capture.capture_status}`}>{capture.capture_status}</span>
      </div>
      <div className="muted">
        Ran {formatDate(start)}
        {event.started_at ? ` – ${formatDate(event.ts)} (${Math.round(capture.span_seconds)}s)` : ""}, session{" "}
        <span className="mono">{event.session_id}</span>. The windows below are the ones overlapping that span;
        they decided its status. {capture.findings.toLocaleString()} finding
        {capture.findings === 1 ? "" : "s"}, {capture.kernel_dropped.toLocaleString()} dropped frame
        {capture.kernel_dropped === 1 ? "" : "s"}.
      </div>
      <div>
        <a href={href({})}>All windows</a>
      </div>
    </div>
  );
}

function WindowsTable({
  windows,
  page,
  limit,
  incompleteOnly,
  since,
  eventId
}: {
  windows: CaptureWindowPage;
  page: number;
  limit: number;
  incompleteOnly: boolean;
  since: string | undefined;
  eventId: string | undefined;
}) {
  const first = windows.total === 0 ? 0 : windows.offset + 1;
  const last = Math.min(windows.offset + windows.items.length, windows.total);
  const hasPrevious = page > 1;
  const hasNext = windows.offset + windows.limit < windows.total;
  const keep = { incomplete: incompleteOnly ? 1 : undefined, since, event: eventId };
  const pageHref = (target: number) =>
    href({ ...keep, cpage: target, climit: limit === DEFAULT_PAGE_SIZE ? undefined : limit });
  const rows = withGaps(windows.items);

  return (
    <>
      <h2 className="section-title">{eventId ? "Windows covering this inference" : "Capture Windows"}</h2>
      <p className="muted">
        One row per window of the tapped link. A window is complete when every frame in it was
        classified, none were dropped, nothing departed from the whitelist and every capture
        interface reported. A window that is missing from the sequence means the tap was not
        watching.
      </p>
      <div className="table-actions">
        <div className="filters-inline">
          {incompleteOnly ? (
            <a href={href({ since, event: eventId })}>Show all windows</a>
          ) : (
            <a href={href({ incomplete: 1, since, event: eventId })}>Show incomplete only</a>
          )}
          {since ? <a href={href({ incomplete: incompleteOnly ? 1 : undefined })}>Clear time filter</a> : null}
        </div>
        <form method="get" className="page-size-form">
          <input type="hidden" name="tab" value="capture" />
          {incompleteOnly ? <input type="hidden" name="incomplete" value="1" /> : null}
          {since ? <input type="hidden" name="since" value={since} /> : null}
          {eventId ? <input type="hidden" name="event" value={eventId} /> : null}
          <PageSizeSelect id="capture-limit" name="climit" value={limit} />
          <div className="filter-actions">
            <button type="submit">Apply</button>
          </div>
        </form>
      </div>
      {windows.items.length === 0 ? (
        <div className="notice">No capture windows match.</div>
      ) : (
        <div className="table-wrap">
          <table className="table-narrow">
            <thead>
              <tr>
                <th>State</th>
                <th>Window start</th>
                <th>Frames</th>
                <th>Dropped</th>
                <th>Findings</th>
                <th>Classes</th>
                <th>Host · interfaces</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) =>
                row.kind === "missing" ? (
                  <tr key={`missing-${row.from}`} className="capture-missing-row">
                    <td>
                      <span className="capture capture-uncovered">no tap</span>
                    </td>
                    <td colSpan={7}>
                      {row.count === 1 ? "1 window" : `${row.count} windows`} missing between{" "}
                      {formatDate(row.from)} and {formatDate(row.to)}: nothing was recorded for this time.
                    </td>
                  </tr>
                ) : (
                  <tr key={row.window.window_id}>
                    <td>
                      <span className={`capture capture-${row.window.complete ? "complete" : "tainted"}`}>
                        {row.window.complete ? "complete" : "incomplete"}
                      </span>
                    </td>
                    <td>{formatDate(row.window.window_start)}</td>
                    <td className="mono">{row.window.observed.toLocaleString()}</td>
                    <td className={`mono${row.window.kernel_dropped ? " accent-fail" : ""}`}>
                      {row.window.kernel_dropped.toLocaleString()}
                    </td>
                    <td className={`mono${row.window.finding_count ? " accent-fail" : ""}`}>
                      {row.window.finding_count.toLocaleString()}
                      {row.window.finding_groups_overflow ? (
                        <div className="muted">+{row.window.finding_groups_overflow} beyond the group cap</div>
                      ) : null}
                    </td>
                    <td className="mono capture-classes">{classSummary(row.window.classes)}</td>
                    <td className="mono">
                      {row.window.capture_host}
                      <div className="muted">{row.window.ifaces}</div>
                    </td>
                    <td>
                      <a href={href({ window: row.window.window_id })}>Open</a>
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      )}
      <nav className="pager" aria-label="Capture windows pagination">
        <div>
          Showing {first.toLocaleString()}-{last.toLocaleString()} of {windows.total.toLocaleString()}
        </div>
        <div className="pager-links">
          {hasPrevious ? <a href={pageHref(page - 1)}>Previous</a> : <span className="disabled">Previous</span>}
          {hasNext ? <a href={pageHref(page + 1)}>Next</a> : <span className="disabled">Next</span>}
        </div>
      </nav>
    </>
  );
}

function WindowDetail({ detail, findingId }: { detail: CaptureWindowDetail; findingId: string | undefined }) {
  const finding = findingId ? detail.findings.find((f) => f.finding_id === findingId) : undefined;
  const classes = Object.entries(detail.classes).sort((a, b) => b[1].frames - a[1].frames);
  const pins = Object.entries(detail.pins);

  return (
    <>
      <div className="breadcrumbs">
        <a href={href({})}>Capture windows</a>
        <span> / </span>
        {finding ? (
          <>
            <a href={href({ window: detail.window_id })}>{formatDate(detail.window_start)}</a>
            <span> / </span>
            <span>
              {finding.kind} · {finding.frame_class}
            </span>
          </>
        ) : (
          <span>{formatDate(detail.window_start)}</span>
        )}
      </div>

      <h2 className="section-title">
        Window {formatDate(detail.window_start)} – {formatDate(detail.window_end)}{" "}
        <span className={`capture capture-${detail.complete ? "complete" : "tainted"}`}>
          {detail.complete ? "complete" : "incomplete"}
        </span>
      </h2>

      <section className="stats capture-stats" aria-label="Window counters">
        <div className="stat">
          <span>Frames</span>
          <strong>{detail.observed.toLocaleString()}</strong>
          <div className="muted">{detail.classified === detail.observed ? "all classified" : `${detail.classified.toLocaleString()} classified`}</div>
        </div>
        <div className="stat">
          <span>Dropped by kernel</span>
          <strong className={detail.kernel_dropped ? "accent-fail" : undefined}>{detail.kernel_dropped.toLocaleString()}</strong>
          <div className="muted">{detail.truncated ? `${detail.truncated} truncated` : "none truncated"}</div>
        </div>
        <div className="stat">
          <span>Findings</span>
          <strong className={detail.finding_count ? "accent-fail" : undefined}>{detail.finding_count.toLocaleString()}</strong>
          <div className="muted">
            {detail.finding_groups} group{detail.finding_groups === 1 ? "" : "s"}
            {detail.finding_groups_overflow ? `, ${detail.finding_groups_overflow} beyond the cap` : ""}
          </div>
        </div>
        <div className="stat">
          <span>Captured on</span>
          <strong className="stat-small">{detail.capture_host}</strong>
          <div className="muted">{detail.ifaces} · frame-processor {detail.tap_version}</div>
        </div>
      </section>

      {finding ? (
        <FindingDetail finding={finding} />
      ) : (
        <>
          <h3 className="section-title">Findings</h3>
          {detail.findings.length === 0 ? (
            <div className="notice">Nothing departed from the whitelist in this window.</div>
          ) : (
            <div className="table-wrap">
              <table className="table-narrow">
                <thead>
                  <tr>
                    <th>Kind</th>
                    <th>Class</th>
                    <th>Direction</th>
                    <th>Sender</th>
                    <th>Count</th>
                    <th>First seen</th>
                    <th>Last seen</th>
                    <th>Sample</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.findings.map((f) => (
                    <tr key={f.finding_id}>
                      <td className="mono">{f.kind}</td>
                      <td className="mono">{f.frame_class}</td>
                      <td>
                        <DirectionLabel direction={f.direction} />
                      </td>
                      <td className="mono">{f.source_mac ?? "-"}</td>
                      <td className="mono">{f.count.toLocaleString()}</td>
                      <td>{formatDate(f.first_ts)}</td>
                      <td>{formatDate(f.last_ts)}</td>
                      <td className="reason">{f.samples[0]?.detail ?? "-"}</td>
                      <td>
                        <a href={href({ window: detail.window_id, finding: f.finding_id })}>
                          {f.samples.length} sample{f.samples.length === 1 ? "" : "s"}
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <h3 className="section-title">Frame classes</h3>
          <div className="table-wrap">
            <table className="table-narrow">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Frames</th>
                  <th>Bytes</th>
                  <th>Out of node</th>
                  <th>Into node</th>
                  <th>Unknown direction</th>
                </tr>
              </thead>
              <tbody>
                {classes.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="muted">
                      No frames crossed the link in this window.
                    </td>
                  </tr>
                ) : (
                  classes.map(([name, totals]) => (
                    <tr key={name}>
                      <td className="mono">{name}</td>
                      <td className="mono">{totals.frames.toLocaleString()}</td>
                      <td className="mono">{totals.bytes.toLocaleString()}</td>
                      <td className="mono">{totals.out.toLocaleString()}</td>
                      <td className="mono">{totals.in.toLocaleString()}</td>
                      <td className="mono">{totals.unknown.toLocaleString()}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {pins.length > 0 ? (
            <>
              <h3 className="section-title">Pinned flows</h3>
              <p className="muted">
                Constant flows held to one payload. A declared pin was checked against the whitelist;
                an undeclared one was learned from the first frame seen.
              </p>
              <div className="table-wrap">
                <table className="table-narrow">
                  <thead>
                    <tr>
                      <th>Sender / class</th>
                      <th>Frames</th>
                      <th>Distinct payloads</th>
                      <th>Digest</th>
                      <th>Declared</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pins.map(([key, pin]) => (
                      <tr key={key}>
                        <td className="mono">{key}</td>
                        <td className="mono">{pin.frames.toLocaleString()}</td>
                        <td className={`mono${pin.distinct !== 1 ? " accent-fail" : ""}`}>{pin.distinct}</td>
                        <td className="mono">{pin.digest}</td>
                        <td className="mono">
                          {pin.declared === null
                            ? "learned"
                            : pin.declared === pin.digest
                              ? "match"
                              : `MISMATCH (${pin.declared})`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
        </>
      )}
    </>
  );
}

function FindingDetail({ finding }: { finding: CaptureFindingRead }) {
  return (
    <>
      <h3 className="section-title">
        <span className="mono">{finding.kind}</span> · <span className="mono">{finding.frame_class}</span>{" "}
        <DirectionLabel direction={finding.direction} />
      </h3>
      <p className="muted">
        {finding.count.toLocaleString()} occurrence{finding.count === 1 ? "" : "s"}
        {finding.source_mac ? ` from ${finding.source_mac}` : ""}, first {formatDate(finding.first_ts)}, last{" "}
        {formatDate(finding.last_ts)}. The first {finding.samples.length} are kept in full below; the frame
        bytes are the first part of the offending frame as it crossed the link.
      </p>
      {finding.samples.map((sample, index) => (
        <section key={`${sample.ts}-${index}`} className="capture-sample">
          <div className="capture-sample-header">
            <strong>Sample {index + 1}</strong>
            <span className="muted">{formatDate(sample.ts)}</span>
          </div>
          <div className="reason">{sample.detail}</div>
          {sample.frame_b64 ? (
            <pre className="hexdump">{hexDump(sample.frame_b64)}</pre>
          ) : (
            <div className="muted">No frame bytes: this finding is about a reassembled exchange, not one frame.</div>
          )}
        </section>
      ))}
    </>
  );
}
