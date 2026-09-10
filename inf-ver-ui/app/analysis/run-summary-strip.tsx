import { runStats } from "./run-stats";
import { allocationLabel, AllocationRow } from "../run-label";

export type RunDetail = {
  id: string;
  state: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  total_requests: number;
  completed: number;
  failed: number;
  error: string | null;
  settings: { models?: AllocationRow[] } & Record<string, unknown>;
  results: { elapsed_s: number; status_code: number | null }[];
};

function fmt(ts: string | null): string {
  return ts ? new Date(ts).toLocaleString() : "-";
}

/** Run-scoped context above the analysis charts: state, window, per-model
 * allocation and latency stats from the durable result rows. GPU economics
 * deliberately absent — that needs pod-window integrals (separation spec). */
export default function RunSummaryStrip({ run }: { run: RunDetail }) {
  const stats = runStats(run.results.map((r) => r.elapsed_s));
  return (
    <section className="chart-card run-summary-strip" aria-label="Selected run summary">
      <h3>Run {run.id}</h3>
      <dl className="run-summary-grid">
        <div><dt>State</dt><dd>{run.state}{run.error ? ` — ${run.error}` : ""}</dd></div>
        <div><dt>Window</dt><dd>{fmt(run.started_at)} → {fmt(run.finished_at)}</dd></div>
        <div>
          <dt>Allocation</dt>
          <dd>{allocationLabel(run.settings)}</dd>
        </div>
        <div><dt>Requests</dt><dd>{run.completed}/{run.total_requests} ({run.failed} failed)</dd></div>
        <div>
          <dt>Latency (s)</dt>
          <dd>
            {stats
              ? `mean ${stats.mean.toFixed(1)} · p50 ${stats.p50.toFixed(1)} · p95 ${stats.p95.toFixed(1)}`
              : "no results"}
          </dd>
        </div>
      </dl>
    </section>
  );
}
