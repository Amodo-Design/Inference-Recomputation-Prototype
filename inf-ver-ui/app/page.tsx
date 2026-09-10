import AnalysisView, { AnalysisFilters } from "./analysis-view";
import BulkActionButtons from "./bulk-action-buttons";
import DeleteEventButton from "./delete-event-button";
import ModelSettingsView from "./model-settings-view";
import ModelsView from "./models-view";
import TestRunnerView from "./test-runner-view";
import OutputDiffDetails from "./output-diff-details";
import RefreshButton from "./refresh-button";
import ReplayEventButton from "./replay-event-button";
import RowDetailModal from "./row-detail-modal";
import {
  ANALYTICS_BASE,
  DEFAULT_PAGE_SIZE,
  LEDGER_BASE,
  ModelRead,
  SearchParams,
  firstValue,
  formatDate,
  formatNumber,
  formatPercent,
  getJson,
  parsePage,
  parsePageSize
} from "./lib";
import PageSizeSelect from "./page-size-select";
import { AnalysisRow } from "./analysis/lib";
import RunSummaryStrip, { RunDetail } from "./analysis/run-summary-strip";
import RunEconomicsPanel from "./analysis/run-economics-panel";

const PROMPT_RUNNER_BASE =
  process.env.PROMPT_RUNNER_URL ?? "http://prompt-runner:8200";

type VerificationStats = {
  total: number;
  pass_count: number;
  fail_count: number;
  unverifiable_count: number;
  average_match_level: number | null;
};

type OutputTokenComparison = {
  index: number;
  prover_token_id: number;
  verifier_token_id: number;
  prover_text: string;
  verifier_text: string;
  exact_match: boolean;
  margin: number | null;
};

type VerifierDetail = {
  prompt_token_count: number | null;
  output_token_count: number | null;
  latency_ms: number | null;
  prover_output_text: string | null;
  verifier_output_text: string | null;
  prover_output_token_ids: number[];
  verifier_output_token_ids: number[];
  output_token_comparison: OutputTokenComparison[];
  metrics: Record<string, unknown>;
};

type VerificationEventView = {
  id: string;
  inference_event_id: string;
  ts: string;
  result: "pass" | "fail" | "unverifiable" | null;
  result_detail: string | null;
  error_code: string | null;
  verification_threshold: number | null;
  verifier_model_id: string;
  verifier_model_name: string | null;
  model_name: string;
  sampling_config: Record<string, unknown>;
  exact_match_level_pct: number | null;
  mean_logit_difference: number | null;
  std_dev_logit_difference: number | null;
  difr_margins: number[];
  session_id: string;
  input_text_representation: string | null;
  output_text_representation: string | null;
  verifier_detail: VerifierDetail | null;
};

type VerificationEventPage = {
  items: VerificationEventView[];
  total: number;
  limit: number;
  offset: number;
};

type UnverifiedEventView = {
  id: string;
  session_id: string;
  ts: string;
  model_name: string;
  sampling_config: Record<string, unknown>;
  input_text_representation: string | null;
  output_text_representation: string | null;
  hardware_id: string;
};

type UnverifiedEventPage = {
  items: UnverifiedEventView[];
  total: number;
  limit: number;
  offset: number;
};

type AnalysisPage = {
  items: AnalysisRow[];
  total: number;
  truncated: boolean;
};

type Filters = {
  result: string;
  model: string;
  q: string;
  // "session" groups the verified events table by session id (UI-only).
  group: string;
};

function filterQuery(filters: Filters): string {
  const params = new URLSearchParams();
  if (filters.result) params.set("result", filters.result);
  if (filters.model) params.set("model", filters.model);
  if (filters.q) params.set("search", filters.q);
  const query = params.toString();
  return query ? `&${query}` : "";
}

function statsQuery(filters: Filters): string {
  const params = new URLSearchParams();
  if (filters.model) params.set("model", filters.model);
  if (filters.q) params.set("search", filters.q);
  const query = params.toString();
  return query ? `?${query}` : "";
}

type PageState = {
  filters: Filters;
  page: number;
  limit: number;
  pendingPage: number;
  pendingLimit: number;
  pendingModel: string;
  pendingGroup: boolean;
};

function pageHref(state: PageState): string {
  const params = new URLSearchParams();
  if (state.page > 1) params.set("page", String(state.page));
  if (state.limit !== DEFAULT_PAGE_SIZE) params.set("limit", String(state.limit));
  if (state.pendingPage > 1) params.set("pending_page", String(state.pendingPage));
  if (state.pendingLimit !== DEFAULT_PAGE_SIZE) params.set("pending_limit", String(state.pendingLimit));
  if (state.filters.result) params.set("result", state.filters.result);
  if (state.filters.model) params.set("model", state.filters.model);
  if (state.filters.q) params.set("q", state.filters.q);
  if (state.filters.group) params.set("group", state.filters.group);
  if (state.pendingModel) params.set("pending_model", state.pendingModel);
  if (state.pendingGroup) params.set("pending_group", "session");
  const query = params.toString();
  return query ? `/?${query}` : "/";
}

function statusClass(result: VerificationEventView["result"]): string {
  return `status status-${result ?? "unverifiable"}`;
}

function groupItemsBySession<T extends { session_id: string }>(items: T[]): [string, T[]][] {
  // Group the current page's rows by session, in order of first appearance
  // (rows arrive newest-first).
  return Array.from(
    items.reduce((groups, item) => {
      const rows = groups.get(item.session_id) ?? [];
      rows.push(item);
      return groups.set(item.session_id, rows);
    }, new Map<string, T[]>())
  );
}

function SessionHeaderRow({
  sessionId,
  count,
  colSpan
}: {
  sessionId: string;
  count: number;
  colSpan: number;
}) {
  return (
    <tr className="session-row">
      <td colSpan={colSpan}>
        Session <span className="mono">{sessionId}</span> · {count.toLocaleString()} event
        {count === 1 ? "" : "s"} on this page
      </td>
    </tr>
  );
}

function buildModelNames(models: ModelRead[], current: string): string[] {
  // Keep a previously-typed (or stale) filter selectable so it stays visible.
  const names = Array.from(new Set(models.map((m) => m.model_name))).sort();
  if (current && !names.includes(current)) {
    names.unshift(current);
  }
  return names;
}

function TextDetails({
  summary,
  value
}: {
  summary: string;
  value: string | null;
}) {
  if (!value) {
    return <span className="muted">-</span>;
  }

  return (
    <details>
      <summary>{summary}</summary>
      <pre className="wrapped-text">{value}</pre>
    </details>
  );
}

function JsonDetails({
  summary,
  value
}: {
  summary: string;
  value: unknown;
}) {
  return (
    <details>
      <summary>{summary}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

function MarginsTable({ margins }: { margins: number[] }) {
  if (margins.length === 0) {
    return <span className="muted">-</span>;
  }
  return (
    <div className="margins-scroll">
      <table className="margins-table">
        <thead>
          <tr>
            <th>#</th>
            <th>margin</th>
          </tr>
        </thead>
        <tbody>
          {margins.map((margin, index) => (
            <tr key={index}>
              <td>{index}</td>
              <td>{margin.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EventRow({ event }: { event: VerificationEventView }) {
  return (
    <tr>
      <td>
        <span className={statusClass(event.result)}>{event.result ?? "-"}</span>
        {event.error_code ? <div className="muted">{event.error_code}</div> : null}
      </td>
      <td>{formatDate(event.ts)}</td>
      <td className="mono">{event.inference_event_id}</td>
      <td className="mono">{event.session_id}</td>
      <td>
        <div>{event.model_name}</div>
      </td>
      <td>
        <div className="mono">{event.verifier_model_name ?? "-"}</div>
      </td>
      <td className="mono verification-result-cell">
        <div>match {formatPercent(event.exact_match_level_pct)}</div>
        <div>threshold {formatNumber(event.verification_threshold, 4)}</div>
        <div>mean Δ {formatNumber(event.mean_logit_difference, 4)}</div>
      </td>
      <td className="margins-cell">
        <MarginsTable margins={event.difr_margins} />
      </td>
      <td>
        <div>Prompt {formatNumber(event.verifier_detail?.prompt_token_count ?? null)}</div>
        <div>Output {formatNumber(event.verifier_detail?.output_token_count ?? null)}</div>
      </td>
      <td className="reason">{event.result_detail ?? "-"}</td>
      <td>
        <TextDetails summary="View prompt" value={event.input_text_representation} />
      </td>
      <td>
        <OutputDiffDetails comparison={event.verifier_detail?.output_token_comparison ?? []} />
      </td>
      <td>
        <TextDetails
          summary="View output"
          value={event.verifier_detail?.prover_output_text ?? event.output_text_representation}
        />
      </td>
      <td>
        <TextDetails summary="View output" value={event.verifier_detail?.verifier_output_text ?? null} />
      </td>
      <td>
        <RowDetailModal value={event} />
      </td>
      <td>
        <span className="row-actions">
          <ReplayEventButton id={event.id} />
          <DeleteEventButton id={event.id} />
        </span>
      </td>
    </tr>
  );
}

function PendingRow({ event }: { event: UnverifiedEventView }) {
  return (
    <tr>
      <td>{formatDate(event.ts)}</td>
      <td className="mono">{event.id}</td>
      <td>{event.model_name}</td>
      <td className="mono">{event.session_id}</td>
      <td>
        <TextDetails summary="View prompt" value={event.input_text_representation} />
      </td>
      <td>
        <TextDetails summary="View output" value={event.output_text_representation} />
      </td>
      <td>
        <JsonDetails summary="All fields" value={event} />
      </td>
    </tr>
  );
}

function TabNav({
  tab
}: {
  tab: "verification" | "instances" | "settings" | "test" | "analysis";
}) {
  return (
    <nav className="tabs" aria-label="Dashboard sections">
      <a className={tab === "verification" ? "tab tab-active" : "tab"} href="/">
        Verification
      </a>
      <a className={tab === "instances" ? "tab tab-active" : "tab"} href="/?tab=instances">
        Model Instances
      </a>
      <a className={tab === "settings" ? "tab tab-active" : "tab"} href="/?tab=settings">
        Model Settings
      </a>
      <a className={tab === "test" ? "tab tab-active" : "tab"} href="/?tab=test">
        Test Runner
      </a>
      <a className={tab === "analysis" ? "tab tab-active" : "tab"} href="/?tab=analysis">
        Analysis
      </a>
    </nav>
  );
}

export default async function Page({
  searchParams
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;

  const tab = firstValue(params.tab);
  // "models" is the pre-split name for the instances tab; keep old links working.
  if (tab === "instances" || tab === "models") {
    return (
      <main className="page">
        <div className="shell">
          <header className="topbar">
            <div className="title">
              <span className="eyebrow">Prototype</span>
              <h1>Inference Verification</h1>
              <p>Executing models and deployment history</p>
            </div>
            <div className="topbar-side">
              <RefreshButton />
              <div className="api-label">Ledger {LEDGER_BASE}</div>
            </div>
          </header>
          <TabNav tab="instances" />
          <ModelsView params={params} />
        </div>
      </main>
    );
  }

  if (tab === "test") {
    return (
      <main className="page">
        <div className="shell">
          <header className="topbar">
            <div className="title">
              <span className="eyebrow">Prototype</span>
              <h1>Inference Verification</h1>
              <p>Deterministic prompt suite test runs</p>
            </div>
            <div className="topbar-side">
              <RefreshButton />
              <div className="api-label">Ledger {LEDGER_BASE}</div>
            </div>
          </header>
          <TabNav tab="test" />
          <TestRunnerView params={params} />
        </div>
      </main>
    );
  }

  if (tab === "analysis") {
    const analysisFilters: AnalysisFilters = {
      result: firstValue(params.a_result) ?? "",
      prover: firstValue(params.a_prover) ?? "",
      verifier: firstValue(params.a_verifier) ?? "",
      q: firstValue(params.a_q) ?? "",
      from: firstValue(params.a_from) ?? "",
      to: firstValue(params.a_to) ?? ""
    };
    const runId = firstValue(params.a_run) ?? "";
    const hasAnalysisFilters = Boolean(
      analysisFilters.result ||
        analysisFilters.prover ||
        analysisFilters.verifier ||
        analysisFilters.q ||
        analysisFilters.from ||
        analysisFilters.to ||
        runId
    );

    const analysisQuery = new URLSearchParams();
    if (analysisFilters.result) analysisQuery.set("result", analysisFilters.result);
    if (analysisFilters.prover) analysisQuery.set("prover_model", analysisFilters.prover);
    if (analysisFilters.verifier) analysisQuery.set("verifier_model", analysisFilters.verifier);
    if (analysisFilters.q) analysisQuery.set("search", analysisFilters.q);
    if (analysisFilters.from) analysisQuery.set("from_ts", `${analysisFilters.from}T00:00:00Z`);
    if (analysisFilters.to) analysisQuery.set("to_ts", `${analysisFilters.to}T23:59:59Z`);

    let recentRuns: { id: string; state: string }[] = [];
    try {
      const response = await fetch(`${PROMPT_RUNNER_BASE}/runs?limit=50`, { cache: "no-store" });
      if (response.ok) recentRuns = ((await response.json()) as { items: { id: string; state: string }[] }).items;
    } catch {
      // Picker degrades to empty when the prompt-runner is unreachable;
      // manual filters still work.
    }

    let selectedRun: RunDetail | null = null;
    if (runId) {
      try {
        const response = await fetch(
          `${PROMPT_RUNNER_BASE}/runs/${encodeURIComponent(runId)}`,
          { cache: "no-store" }
        );
        if (response.ok) selectedRun = await response.json();
      } catch {
        // Selected run stays null; manual filters remain in effect.
      }
    }

    // A selected run's window OVERRIDES the manual from/to filters — placed
    // AFTER the sets above so the run wins; clearing the picker (runId "")
    // restores manual filters since this block is skipped entirely. The run
    // window filters on the INFERENCE timestamp (the cohort definition):
    // verifications drain minutes after the run, so the verification-ts
    // from_ts/to_ts filter would never match a run's own events.
    if (selectedRun?.started_at) {
      analysisQuery.delete("from_ts");
      analysisQuery.delete("to_ts");
      analysisQuery.set("inference_from_ts", selectedRun.started_at);
      analysisQuery.set(
        "inference_to_ts",
        selectedRun.finished_at ?? new Date().toISOString()
      );
      const models = selectedRun.settings.models ?? [];
      if (models.length === 1) analysisQuery.set("prover_model", models[0].model);
    }

    let analysis: AnalysisPage;
    let analysisModels: ModelRead[];
    try {
      [analysis, analysisModels] = await Promise.all([
        getJson<AnalysisPage>(
          ANALYTICS_BASE,
          `/analysis?${analysisQuery.toString()}`
        ),
        getJson<ModelRead[]>(LEDGER_BASE, "/models?limit=1000")
      ]);
    } catch (error) {
      return (
        <main className="page">
          <div className="shell">
            <header className="topbar">
              <div className="title">
                <span className="eyebrow">Prototype</span>
                <h1>Inference Verification</h1>
                <p>Logit-difference analysis</p>
              </div>
              <div className="topbar-side">
                <RefreshButton />
                <div className="api-label">Ledger {LEDGER_BASE}</div>
              </div>
            </header>
            <TabNav tab="analysis" />
            <div className="error">
              Could not load analysis data from {ANALYTICS_BASE}:{" "}
              {error instanceof Error ? error.message : "unknown error"}
            </div>
          </div>
        </main>
      );
    }

    const proverNames = buildModelNames(analysisModels, analysisFilters.prover);
    const verifierNames = buildModelNames(analysisModels, analysisFilters.verifier);
    const allModelNames = buildModelNames(analysisModels, "");

    return (
      <main className="page">
        <div className="shell">
          <header className="topbar">
            <div className="title">
              <span className="eyebrow">Prototype</span>
              <h1>Inference Verification</h1>
              <p>Logit-difference analysis</p>
            </div>
            <div className="topbar-side">
              <RefreshButton />
              <div className="api-label">Ledger {LEDGER_BASE}</div>
            </div>
          </header>

          <TabNav tab="analysis" />

          <form className="filters" method="get" action="/" aria-label="Filter analysis data">
            <input type="hidden" name="tab" value="analysis" />
            <div className="filter-field">
              <label htmlFor="a-filter-result">Result</label>
              <select id="a-filter-result" name="a_result" defaultValue={analysisFilters.result}>
                <option value="">All</option>
                <option value="pass">Pass</option>
                <option value="fail">Fail</option>
                <option value="unverifiable">Unverifiable</option>
              </select>
            </div>
            <div className="filter-field">
              <label htmlFor="a-filter-prover">Prover Model</label>
              <select id="a-filter-prover" name="a_prover" defaultValue={analysisFilters.prover}>
                <option value="">All</option>
                {proverNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="filter-field">
              <label htmlFor="a-filter-verifier">Verifier Model</label>
              <select id="a-filter-verifier" name="a_verifier" defaultValue={analysisFilters.verifier}>
                <option value="">All</option>
                {verifierNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="filter-field filter-grow">
              <label htmlFor="a-filter-q">Search</label>
              <input
                id="a-filter-q"
                name="a_q"
                type="text"
                defaultValue={analysisFilters.q}
                placeholder="Inference event ID, reason or prompt text"
              />
            </div>
            <div className="filter-field">
              <label htmlFor="a-filter-from">From</label>
              <input id="a-filter-from" name="a_from" type="date" defaultValue={analysisFilters.from} />
            </div>
            <div className="filter-field">
              <label htmlFor="a-filter-to">To</label>
              <input id="a-filter-to" name="a_to" type="date" defaultValue={analysisFilters.to} />
            </div>
            <div className="filter-field">
              <label htmlFor="a-filter-run">Run</label>
              <select id="a-filter-run" name="a_run" defaultValue={runId}>
                <option value="">Manual window</option>
                {recentRuns.map((run) => (
                  <option key={run.id} value={run.id}>
                    {run.id} ({run.state})
                  </option>
                ))}
              </select>
            </div>
            <div className="filter-actions">
              <button type="submit">Filter</button>
              {hasAnalysisFilters ? <a href="/?tab=analysis">Clear</a> : null}
            </div>
          </form>

          {selectedRun ? <RunSummaryStrip run={selectedRun} /> : null}
          {selectedRun ? <RunEconomicsPanel runId={selectedRun.id} /> : null}

          <div className="count-line">
            {analysis.total.toLocaleString()} events analysed
          </div>
          {analysis.truncated ? (
            <div className="error">
              Result set truncated to {analysis.items.length.toLocaleString()} of{" "}
              {analysis.total.toLocaleString()} events; narrow the filters for a complete analysis.
            </div>
          ) : null}

          <AnalysisView
            rows={analysis.items}
            total={analysis.total}
            truncated={analysis.truncated}
            modelNames={allModelNames}
            filters={analysisFilters}
          />
        </div>
      </main>
    );
  }

  if (tab === "settings") {
    return (
      <main className="page">
        <div className="shell">
          <header className="topbar">
            <div className="title">
              <span className="eyebrow">Prototype</span>
              <h1>Inference Verification</h1>
              <p>Per-model verification settings</p>
            </div>
            <div className="topbar-side">
              <RefreshButton />
              <div className="api-label">Ledger {LEDGER_BASE}</div>
            </div>
          </header>
          <TabNav tab="settings" />
          <ModelSettingsView />
        </div>
      </main>
    );
  }

  const page = parsePage(params.page);
  const limit = parsePageSize(params.limit);
  const offset = (page - 1) * limit;
  const pendingPage = parsePage(params.pending_page);
  const pendingLimit = parsePageSize(params.pending_limit);
  const pendingOffset = (pendingPage - 1) * pendingLimit;
  const filters: Filters = {
    result: firstValue(params.result) ?? "",
    model: firstValue(params.model) ?? "",
    q: firstValue(params.q) ?? "",
    group: firstValue(params.group) === "session" ? "session" : ""
  };
  const hasFilters = Boolean(filters.result || filters.model || filters.q || filters.group);
  const groupBySession = filters.group === "session";
  const pendingModel = firstValue(params.pending_model) ?? "";
  const pendingGroup = firstValue(params.pending_group) === "session";
  const state: PageState = {
    filters,
    page,
    limit,
    pendingPage,
    pendingLimit,
    pendingModel,
    pendingGroup
  };

  let events: VerificationEventPage;
  let stats: VerificationStats;
  let pending: UnverifiedEventPage;
  let knownModels: ModelRead[];
  try {
    [events, stats, pending, knownModels] = await Promise.all([
      getJson<VerificationEventPage>(
        LEDGER_BASE,
        `/verification-events/view?limit=${limit}&offset=${offset}${filterQuery(filters)}`
      ),
      getJson<VerificationStats>(
        LEDGER_BASE,
        `/verification-events/stats${statsQuery(filters)}`
      ),
      getJson<UnverifiedEventPage>(
        LEDGER_BASE,
        `/inference-events/unverified/view?limit=${pendingLimit}&offset=${pendingOffset}${
          pendingModel ? `&model=${encodeURIComponent(pendingModel)}` : ""
        }`
      ),
      // Feeds the model filter dropdown with every registered model config
      // (provers and verifiers land in the same ledger table).
      getJson<ModelRead[]>(LEDGER_BASE, "/models?limit=1000")
    ]);
  } catch (error) {
    return (
      <main className="page">
        <div className="shell">
          <header className="topbar">
            <div className="title">
              <span className="eyebrow">Prototype</span>
              <h1>Inference Verification</h1>
              <p>Verification ledger results</p>
            </div>
          </header>
          <TabNav tab="verification" />
          <div className="error">
            Could not load verification data from {LEDGER_BASE}:{" "}
            {error instanceof Error ? error.message : "unknown error"}
          </div>
        </div>
      </main>
    );
  }

  const first = events.total === 0 ? 0 : events.offset + 1;
  const last = Math.min(events.offset + events.items.length, events.total);
  const hasPrevious = page > 1;
  const hasNext = events.offset + events.limit < events.total;

  const modelNames = buildModelNames(knownModels, filters.model);
  const pendingModelNames = buildModelNames(knownModels, pendingModel);

  const pendingFirst = pending.total === 0 ? 0 : pending.offset + 1;
  const pendingLast = Math.min(pending.offset + pending.items.length, pending.total);
  const pendingHasPrevious = pendingPage > 1;
  const pendingHasNext = pending.offset + pending.limit < pending.total;

  const exportFilterQuery = filterQuery(filters);

  return (
    <main className="page">
      <div className="shell">
        <header className="topbar">
          <div className="title">
            <span className="eyebrow">Prototype</span>
            <h1>Inference Verification</h1>
            <p>Verification ledger results</p>
          </div>
          <div className="topbar-side">
            <RefreshButton />
            <div className="api-label">Ledger {LEDGER_BASE}</div>
          </div>
        </header>

        <TabNav tab="verification" />

        <section className="stats" aria-label="Verification summary">
          <div className="stat">
            <span>Total</span>
            <strong>{stats.total.toLocaleString()}</strong>
          </div>
          <div className="stat">
            <span>Passed</span>
            <strong className="accent-pass">{stats.pass_count.toLocaleString()}</strong>
          </div>
          <div className="stat">
            <span>Failed</span>
            <strong className="accent-fail">{stats.fail_count.toLocaleString()}</strong>
          </div>
          <div className="stat">
            <span>Unverifiable</span>
            <strong className="accent-unknown">
              {stats.unverifiable_count.toLocaleString()}
            </strong>
          </div>
          <div className="stat">
            <span>Awaiting</span>
            <strong className="accent-unknown">{pending.total.toLocaleString()}</strong>
          </div>
          <div className="stat">
            <span>Avg Exact Match</span>
            <strong>{formatPercent(stats.average_match_level)}</strong>
          </div>
          {/* Latency hidden for now; restore alongside the table column.
          <div className="stat">
            <span>Avg Latency</span>
            <strong>{formatNumber(stats.average_latency_ms, 0)} ms</strong>
          </div>
          */}
        </section>

        <h2 className="section-title">Verified Events</h2>

        <form className="filters" method="get" action="/" aria-label="Filter events">
          {/* Preserve the awaiting-verification section's state: GET forms
              replace the whole query string with only their own fields. */}
          {pendingPage > 1 ? <input type="hidden" name="pending_page" value={pendingPage} /> : null}
          {pendingLimit !== DEFAULT_PAGE_SIZE ? (
            <input type="hidden" name="pending_limit" value={pendingLimit} />
          ) : null}
          {pendingModel ? <input type="hidden" name="pending_model" value={pendingModel} /> : null}
          {pendingGroup ? <input type="hidden" name="pending_group" value="session" /> : null}
          <div className="filter-field">
            <label htmlFor="filter-result">Result</label>
            <select id="filter-result" name="result" defaultValue={filters.result}>
              <option value="">All</option>
              <option value="pass">Pass</option>
              <option value="fail">Fail</option>
              <option value="unverifiable">Unverifiable</option>
            </select>
          </div>
          <div className="filter-field">
            <label htmlFor="filter-model">Model</label>
            <select id="filter-model" name="model" defaultValue={filters.model}>
              <option value="">All</option>
              {modelNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div className="filter-field filter-grow">
            <label htmlFor="filter-q">Search</label>
            <input
              id="filter-q"
              name="q"
              type="text"
              defaultValue={filters.q}
              placeholder="Inference event ID, reason or prompt text"
            />
          </div>
          <div className="filter-field">
            <label htmlFor="filter-group">Grouping</label>
            <label className="filter-check">
              <input
                id="filter-group"
                name="group"
                type="checkbox"
                value="session"
                defaultChecked={groupBySession}
              />
              By session
            </label>
          </div>
          <PageSizeSelect id="filter-limit" name="limit" value={limit} />
          <div className="filter-actions">
            <button type="submit">Filter</button>
            {hasFilters ? <a href="/">Clear</a> : null}
          </div>
        </form>

        <div className="table-actions">
          <a
            className="toolbar-link"
            href={`/api/verification-events/export${exportFilterQuery ? `?${exportFilterQuery.slice(1)}` : ""}`}
            download
          >
            Export CSV
          </a>
          <BulkActionButtons
            total={events.total}
            filters={{ result: filters.result, model: filters.model, q: filters.q }}
          />
        </div>

        {events.items.length === 0 ? (
          <div className="empty">
            {hasFilters
              ? "No verification events match the current filters."
              : "No verification events have been recorded."}
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Result</th>
                  <th>Created</th>
                  <th>Inference Event</th>
                  <th>Session</th>
                  <th>Prover Model</th>
                  <th>Verifier Model</th>
                  <th>Verification Result</th>
                  <th>DiFR Margins</th>
                  <th>Tokens</th>
                  {/* <th>Latency</th> */}
                  <th>Detail</th>
                  <th>User Prompt</th>
                  <th>Output Diff</th>
                  <th>Prover Output</th>
                  <th>Verifier Output</th>
                  <th>Row Detail</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {groupBySession
                  ? groupItemsBySession(events.items).flatMap(([sessionId, rows]) => [
                      <SessionHeaderRow
                        key={`session-${sessionId}`}
                        sessionId={sessionId}
                        count={rows.length}
                        colSpan={16}
                      />,
                      ...rows.map((event) => <EventRow key={event.id} event={event} />)
                    ])
                  : events.items.map((event) => <EventRow key={event.id} event={event} />)}
              </tbody>
            </table>
          </div>
        )}

        <nav className="pager" aria-label="Verified events pagination">
          <div>
            Showing {first.toLocaleString()}-{last.toLocaleString()} of{" "}
            {events.total.toLocaleString()}
          </div>
          <div className="pager-links">
            {hasPrevious ? (
              <a href={pageHref({ ...state, page: page - 1 })}>Previous</a>
            ) : (
              <span className="disabled">Previous</span>
            )}
            {hasNext ? (
              <a href={pageHref({ ...state, page: page + 1 })}>Next</a>
            ) : (
              <span className="disabled">Next</span>
            )}
          </div>
        </nav>

        <h2 className="section-title">
          Awaiting Verification ({pending.total.toLocaleString()})
        </h2>

        <form className="filters" method="get" action="/" aria-label="Filter awaiting events">
          {/* Preserve the Verified Events section's state the same way. */}
          {page > 1 ? <input type="hidden" name="page" value={page} /> : null}
          {limit !== DEFAULT_PAGE_SIZE ? <input type="hidden" name="limit" value={limit} /> : null}
          {filters.result ? <input type="hidden" name="result" value={filters.result} /> : null}
          {filters.model ? <input type="hidden" name="model" value={filters.model} /> : null}
          {filters.q ? <input type="hidden" name="q" value={filters.q} /> : null}
          {filters.group ? <input type="hidden" name="group" value={filters.group} /> : null}
          <div className="filter-field">
            <label htmlFor="filter-pending-model">Model</label>
            <select id="filter-pending-model" name="pending_model" defaultValue={pendingModel}>
              <option value="">All</option>
              {pendingModelNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div className="filter-field">
            <label htmlFor="filter-pending-group">Grouping</label>
            <label className="filter-check">
              <input
                id="filter-pending-group"
                name="pending_group"
                type="checkbox"
                value="session"
                defaultChecked={pendingGroup}
              />
              By session
            </label>
          </div>
          <PageSizeSelect id="filter-pending-limit" name="pending_limit" value={pendingLimit} />
          <div className="filter-actions">
            <button type="submit">Filter</button>
            {pendingModel || pendingGroup ? (
              <a href={pageHref({ ...state, pendingModel: "", pendingGroup: false, pendingPage: 1 })}>
                Clear
              </a>
            ) : null}
          </div>
        </form>

        {pending.items.length === 0 ? (
          <div className="empty">No events awaiting verification.</div>
        ) : (
          <div className="table-wrap">
            <table className="table-narrow">
              <thead>
                <tr>
                  <th>Created</th>
                  <th>Inference Event</th>
                  <th>Model</th>
                  <th>Session</th>
                  <th>User Prompt</th>
                  <th>Output</th>
                  <th>Database Row</th>
                </tr>
              </thead>
              <tbody>
                {pendingGroup
                  ? groupItemsBySession(pending.items).flatMap(([sessionId, rows]) => [
                      <SessionHeaderRow
                        key={`session-${sessionId}`}
                        sessionId={sessionId}
                        count={rows.length}
                        colSpan={7}
                      />,
                      ...rows.map((event) => <PendingRow key={event.id} event={event} />)
                    ])
                  : pending.items.map((event) => <PendingRow key={event.id} event={event} />)}
              </tbody>
            </table>
          </div>
        )}

        <nav className="pager" aria-label="Awaiting verification pagination">
          <div>
            Showing {pendingFirst.toLocaleString()}-{pendingLast.toLocaleString()} of{" "}
            {pending.total.toLocaleString()}
          </div>
          <div className="pager-links">
            {pendingHasPrevious ? (
              <a href={pageHref({ ...state, pendingPage: pendingPage - 1 })}>Previous</a>
            ) : (
              <span className="disabled">Previous</span>
            )}
            {pendingHasNext ? (
              <a href={pageHref({ ...state, pendingPage: pendingPage + 1 })}>Next</a>
            ) : (
              <span className="disabled">Next</span>
            )}
          </div>
        </nav>
      </div>
    </main>
  );
}
