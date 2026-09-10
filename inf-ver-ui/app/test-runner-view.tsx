import {
  DEFAULT_PAGE_SIZE,
  LEDGER_BASE,
  ModelRead,
  SearchParams,
  firstValue,
  formatDate,
  getJson,
  parsePage,
  parsePageSize
} from "./lib";
import TestRunnerForm from "./test-runner-form";
import TestRunnerProgress from "./test-runner-progress";
import CancelRunButton from "./cancel-run-button";
import DeleteAllRunsButton from "./delete-all-runs-button";
import DeleteRunButton from "./delete-run-button";
import { allocationLabel, AllocationRow } from "./run-label";

const PROMPT_RUNNER_BASE =
  process.env.PROMPT_RUNNER_URL ?? "http://prompt-runner:8200";
const LIST_LIMIT = 1000;

type ModelDeploymentRead = {
  deployment_id: string;
  model_id: string;
  hardware_id: string;
  started_at: string;
  ended_at: string | null;
};

type HardwareRead = {
  hardware_id: string;
  hostname: string;
  owner_id: string | null;
};

type HardwareOwnerRead = {
  owner_id: string;
  organisation_name: string;
};

type RunSummary = {
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
};

type RunPage = {
  items: RunSummary[];
  total: number;
  limit: number;
  offset: number;
};

const TERMINAL_STATES = "completed,error,cancelled,interrupted";
const LIVE_STATES = "queued,running";

function formatOptionalDate(value: string | null | undefined): string {
  return value ? formatDate(value) : "-";
}

function historyHref(page: number, size: number, model: string): string {
  const params = new URLSearchParams({ tab: "test" });
  if (page > 1) params.set("h_page", String(page));
  if (size !== DEFAULT_PAGE_SIZE) params.set("h_size", String(size));
  if (model) params.set("h_model", model);
  return `/?${params.toString()}`;
}

function stateBadgeClass(state: string): string {
  if (state === "completed") return "status status-pass";
  if (state === "queued" || state === "running") return "status status-unverifiable";
  return "status status-fail";
}

/** Runs the deterministic prompt suite (vendored from
 * the earlier standalone openwebui_prompt_runner script) through Open WebUI
 * against one or more RUNNING prover models, split by allocation percentage.
 * Every response is tapped into the ledger and verified by the
 * orchestrator's runners. Runs queue and execute serially; run history is
 * durable in Postgres. */
export default async function TestRunnerView({ params }: { params: SearchParams }) {
  const historyPage = parsePage(params.h_page);
  const historySize = parsePageSize(params.h_size);
  const historyModel = firstValue(params.h_model) ?? "";
  let activeProverModels: string[] = [];
  let allModelNames: string[] = [];
  let modelsError: string | null = null;
  try {
    const [deployments, models, hardware, owners] = await Promise.all([
      getJson<ModelDeploymentRead[]>(LEDGER_BASE, `/model-deployments?limit=${LIST_LIMIT}`),
      getJson<ModelRead[]>(LEDGER_BASE, `/models?limit=${LIST_LIMIT}`),
      getJson<HardwareRead[]>(LEDGER_BASE, `/hardware?limit=${LIST_LIMIT}`),
      getJson<HardwareOwnerRead[]>(LEDGER_BASE, `/hardware-owners?limit=${LIST_LIMIT}`)
    ]);
    const modelsById = new Map(models.map((m) => [m.model_id, m]));
    const hardwareById = new Map(hardware.map((h) => [h.hardware_id, h]));
    const ownersById = new Map(owners.map((o) => [o.owner_id, o]));
    // Target = models with an ACTIVE deployment on prover-owned hardware
    // (i.e. what the taps are serving right now). Falls back to any active
    // deployment when owners have not been declared yet.
    const activeNames = new Set<string>();
    const anyActiveNames = new Set<string>();
    for (const deployment of deployments) {
      if (deployment.ended_at !== null) continue;
      const name = modelsById.get(deployment.model_id)?.model_name;
      if (!name) continue;
      anyActiveNames.add(name);
      const host = hardwareById.get(deployment.hardware_id);
      const owner = host?.owner_id ? ownersById.get(host.owner_id) : null;
      if (owner?.organisation_name === "prover") {
        activeNames.add(name);
      }
    }
    activeProverModels = Array.from(
      activeNames.size > 0 ? activeNames : anyActiveNames
    ).sort();
    // History can reference models that are no longer deployed — the filter
    // offers every model the ledger has ever seen.
    allModelNames = Array.from(new Set(models.map((m) => m.model_name))).sort();
  } catch (error) {
    modelsError = error instanceof Error ? error.message : "unknown error";
  }

  // Two fetches: the live sections must see every queued/running run
  // regardless of how the (paged, filtered) history view is scoped.
  let live: RunSummary[] = [];
  let history: RunPage = { items: [], total: 0, limit: historySize, offset: 0 };
  let runnerError: string | null = null;
  try {
    const historyQuery = new URLSearchParams({
      states: TERMINAL_STATES,
      limit: String(historySize),
      offset: String((historyPage - 1) * historySize)
    });
    if (historyModel) historyQuery.set("model", historyModel);
    const [liveResponse, historyResponse] = await Promise.all([
      fetch(`${PROMPT_RUNNER_BASE}/runs?states=${LIVE_STATES}&limit=100`, { cache: "no-store" }),
      fetch(`${PROMPT_RUNNER_BASE}/runs?${historyQuery.toString()}`, { cache: "no-store" })
    ]);
    if (!liveResponse.ok) throw new Error(`prompt-runner returned ${liveResponse.status}`);
    if (!historyResponse.ok) throw new Error(`prompt-runner returned ${historyResponse.status}`);
    live = ((await liveResponse.json()) as RunPage).items;
    history = (await historyResponse.json()) as RunPage;
  } catch (error) {
    runnerError = error instanceof Error ? error.message : "unknown error";
  }

  const queued = live
    .filter((run) => run.state === "queued")
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
  const activeRun = live.find((run) => run.state === "running");
  const historyFirst = history.total === 0 ? 0 : history.offset + 1;
  const historyLast = Math.min(history.offset + history.items.length, history.total);
  const historyHasPrevious = historyPage > 1;
  const historyHasNext = history.offset + history.limit < history.total;

  return (
    <>
      <h2 className="section-title">Launch Test Run</h2>
      <p className="muted">
        Sends the deterministic PromptBench suite through Open WebUI, split
        across one or more prover models by percentage; every response is
        tapped into the ledger and verified by the orchestrator&apos;s
        runners. Runs queue and execute one at a time, serially. Sampling is
        not configurable here: the verify-taps pin each model&apos;s sampling
        config (see Model Instances), so every run always executes under the
        pinned values.
      </p>
      {modelsError ? (
        <div className="error">Could not load models from {LEDGER_BASE}: {modelsError}</div>
      ) : null}
      <TestRunnerForm models={activeProverModels} />

      {runnerError ? (
        <div className="error">
          Could not reach the prompt-runner service at {PROMPT_RUNNER_BASE}: {runnerError}
        </div>
      ) : null}

      <h2 className="section-title">Queue</h2>
      {queued.length === 0 ? (
        <div className="empty">No queued runs.</div>
      ) : (
        <div className="table-wrap">
          <table className="table-narrow">
            <thead>
              <tr>
                <th>Run</th>
                <th>Allocation</th>
                <th>Queued</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {queued.map((run) => (
                <tr key={run.id}>
                  <td className="mono">{run.id}</td>
                  <td>{allocationLabel(run.settings)}</td>
                  <td>{formatOptionalDate(run.created_at)}</td>
                  <td>
                    <CancelRunButton runId={run.id} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activeRun ? (
        <>
          <h2 className="section-title">Active Run</h2>
          <TestRunnerProgress runId={activeRun.id} />
        </>
      ) : null}

      <h2 className="section-title">
        Run History <DeleteAllRunsButton count={history.total} />
      </h2>
      <form className="filters" method="get" action="/" aria-label="Filter run history">
        <input type="hidden" name="tab" value="test" />
        {historySize !== DEFAULT_PAGE_SIZE ? (
          <input type="hidden" name="h_size" value={historySize} />
        ) : null}
        <div className="filter-field">
          <label htmlFor="h-filter-model">Model</label>
          <select id="h-filter-model" name="h_model" defaultValue={historyModel}>
            <option value="">All</option>
            {allModelNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-actions">
          <button type="submit">Filter</button>
          {historyModel ? <a href={historyHref(1, historySize, "")}>Clear</a> : null}
        </div>
      </form>
      {history.items.length === 0 ? (
        <div className="empty">
          {runnerError
            ? "Prompt-runner unreachable."
            : historyModel
              ? `No runs for ${historyModel}.`
              : "No completed runs yet."}
        </div>
      ) : (
        <div className="table-wrap">
          <table className="table-narrow">
            <thead>
              <tr>
                <th>Run</th>
                <th>State</th>
                <th>Model</th>
                <th>Requests</th>
                <th>Failed</th>
                <th>Queued</th>
                <th>Started</th>
                <th>Finished</th>
                <th>Settings</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {history.items.map((run) => (
                <tr key={run.id}>
                  <td className="mono">{run.id}</td>
                  <td>
                    <span className={stateBadgeClass(run.state)}>{run.state}</span>
                    {run.error ? <div className="muted">{run.error}</div> : null}
                  </td>
                  <td>{allocationLabel(run.settings)}</td>
                  <td className="mono">
                    {run.completed}/{run.total_requests}
                  </td>
                  <td className="mono">{run.failed}</td>
                  <td>{formatOptionalDate(run.created_at)}</td>
                  <td>{formatOptionalDate(run.started_at)}</td>
                  <td>{formatOptionalDate(run.finished_at)}</td>
                  <td>
                    <details>
                      <summary>Settings</summary>
                      <pre>{JSON.stringify(run.settings, null, 2)}</pre>
                    </details>
                  </td>
                  <td>
                    <DeleteRunButton runId={run.id} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <nav className="pager" aria-label="Run history pagination">
        <div>
          Showing {historyFirst.toLocaleString()}-{historyLast.toLocaleString()} of{" "}
          {history.total.toLocaleString()}
        </div>
        <div className="pager-links">
          {historyHasPrevious ? (
            <a href={historyHref(historyPage - 1, historySize, historyModel)}>Previous</a>
          ) : (
            <span className="disabled">Previous</span>
          )}
          {historyHasNext ? (
            <a href={historyHref(historyPage + 1, historySize, historyModel)}>Next</a>
          ) : (
            <span className="disabled">Next</span>
          )}
        </div>
      </nav>
    </>
  );
}
