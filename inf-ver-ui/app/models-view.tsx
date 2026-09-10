import {
  DEFAULT_PAGE_SIZE,
  LEDGER_BASE,
  ModelRead,
  SearchParams,
  firstValue,
  formatDate,
  formatNumber,
  getJson,
  parsePage,
  parsePageSize
} from "./lib";
import PageSizeSelect from "./page-size-select";

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
  gpu_product_id: string | null;
  cpu_product_id: string | null;
  owner_id: string | null;
  gpu_firmware_version: string | null;
};

type HardwareOwnerRead = {
  owner_id: string;
  organisation_name: string;
  is_trusted: boolean;
};

type DeploymentRow = {
  deployment: ModelDeploymentRead;
  model: ModelRead | null;
  hardware: HardwareRead | null;
  owner: HardwareOwnerRead | null;
};

// The ledger's list endpoints cap at 1000 rows; plenty for the prototype.
const LIST_LIMIT = 1000;

type SectionFilters = {
  model: string;
  host: string;
  status: string;
};

const EMPTY_FILTERS: SectionFilters = { model: "", host: "", status: "" };

type ModelsState = {
  active: SectionFilters;
  history: SectionFilters;
  activePage: number;
  activeLimit: number;
  historyPage: number;
  historyLimit: number;
};

function modelsHref(state: ModelsState): string {
  const params = new URLSearchParams();
  params.set("tab", "instances");
  if (state.active.model) params.set("m_model", state.active.model);
  if (state.active.host) params.set("m_host", state.active.host);
  if (state.active.status) params.set("m_status", state.active.status);
  if (state.history.model) params.set("m_hist_model", state.history.model);
  if (state.history.host) params.set("m_hist_host", state.history.host);
  if (state.history.status) params.set("m_hist_status", state.history.status);
  if (state.activePage > 1) params.set("m_page", String(state.activePage));
  if (state.activeLimit !== DEFAULT_PAGE_SIZE) params.set("m_limit", String(state.activeLimit));
  if (state.historyPage > 1) params.set("m_hist_page", String(state.historyPage));
  if (state.historyLimit !== DEFAULT_PAGE_SIZE) {
    params.set("m_hist_limit", String(state.historyLimit));
  }
  return `/?${params.toString()}`;
}

function applySectionFilters(rows: DeploymentRow[], filters: SectionFilters): DeploymentRow[] {
  return rows.filter(({ model, hardware }) => {
    if (filters.model && model?.model_name !== filters.model) return false;
    if (filters.host && hardware?.hostname !== filters.host) return false;
    return true;
  });
}

function Pager({
  label,
  page,
  pageSize,
  total,
  shown,
  previousHref,
  nextHref
}: {
  label: string;
  page: number;
  pageSize: number;
  total: number;
  shown: number;
  previousHref: string;
  nextHref: string;
}) {
  const offset = (page - 1) * pageSize;
  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + shown, total);
  const hasPrevious = page > 1;
  const hasNext = offset + pageSize < total;

  return (
    <nav className="pager" aria-label={label}>
      <div>
        Showing {first.toLocaleString()}-{last.toLocaleString()} of {total.toLocaleString()}
      </div>
      <div className="pager-links">
        {hasPrevious ? <a href={previousHref}>Previous</a> : <span className="disabled">Previous</span>}
        {hasNext ? <a href={nextHref}>Next</a> : <span className="disabled">Next</span>}
      </div>
    </nav>
  );
}

function SamplingConfigCell({ model }: { model: ModelRead | null }) {
  if (!model) {
    return <span className="muted">-</span>;
  }
  const { decoding_algorithm, temperature, top_k, top_p, seed } = model;
  const params: { label: string; value: number | null }[] = [
    { label: "temp", value: temperature },
    { label: "top_k", value: top_k },
    { label: "top_p", value: top_p },
    { label: "seed", value: seed }
  ];

  return (
    <div className="sampling-tags">
      {decoding_algorithm ? (
        <span className="sampling-tag sampling-tag-algo">{decoding_algorithm}</span>
      ) : (
        <span className="sampling-tag sampling-tag-empty">no algorithm</span>
      )}
      {params.map(({ label, value }) => (
        <span key={label} className={`sampling-tag${value === null ? " sampling-tag-empty" : ""}`}>
          <span className="sampling-tag-label">{label}</span>
          <span className="sampling-tag-value">{value === null ? "-" : value}</span>
        </span>
      ))}
    </div>
  );
}

function OwnerCell({ owner }: { owner: HardwareOwnerRead | null }) {
  if (!owner) {
    return <span className="muted">-</span>;
  }
  return (
    <>
      <div>{owner.organisation_name}</div>
      <span className={owner.is_trusted ? "status status-pass" : "status status-fail"}>
        {owner.is_trusted ? "trusted" : "untrusted"}
      </span>
    </>
  );
}

function DeploymentTable({
  rows,
  emptyMessage,
  showEnded
}: {
  rows: DeploymentRow[];
  emptyMessage: string;
  showEnded: boolean;
}) {
  if (rows.length === 0) {
    return <div className="empty">{emptyMessage}</div>;
  }

  return (
    <div className="table-wrap">
      <table className="table-narrow">
        <thead>
          <tr>
            <th>Model</th>
            <th>Sampling Config</th>
            <th>Host</th>
            <th>GPU</th>
            <th>Owner</th>
            <th>Started</th>
            {showEnded ? <th>Ended</th> : null}
            <th>Deployment</th>
            <th>Database Row</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ deployment, model, hardware, owner }) => (
            <tr key={deployment.deployment_id}>
              <td>{model?.model_name ?? deployment.model_id}</td>
              <td className="mono">
                <SamplingConfigCell model={model} />
              </td>
              <td className="mono">{hardware?.hostname ?? deployment.hardware_id}</td>
              <td>
                <div>{hardware?.gpu_product_id ?? "-"}</div>
                {hardware?.gpu_firmware_version ? (
                  <div className="muted">fw {hardware.gpu_firmware_version}</div>
                ) : null}
              </td>
              <td>
                <OwnerCell owner={owner} />
              </td>
              <td>{formatDate(deployment.started_at)}</td>
              {showEnded ? (
                <td>{deployment.ended_at ? formatDate(deployment.ended_at) : "-"}</td>
              ) : null}
              <td className="mono">{deployment.deployment_id}</td>
              <td>
                <details>
                  <summary>All fields</summary>
                  <pre>{JSON.stringify({ deployment, model, hardware, owner }, null, 2)}</pre>
                </details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectionFilterForm({
  idPrefix,
  fieldPrefix,
  filters,
  limit,
  otherState,
  modelNames,
  hostnames,
  statusOptions
}: {
  idPrefix: string;
  fieldPrefix: string;
  filters: SectionFilters;
  limit: number;
  otherState: ModelsState;
  modelNames: string[];
  hostnames: string[];
  statusOptions: { value: string; label: string }[] | null;
}) {
  const hasFilters = Boolean(filters.model || filters.host || filters.status);
  const isActive = idPrefix === "m";

  return (
    <form className="filters" method="get" action="/" aria-label={`Filter ${idPrefix}`}>
      <input type="hidden" name="tab" value="instances" />
      {/* Preserve the other section's filters/page/limit: GET forms replace
          the whole query string. */}
      {otherState.active.model ? <input type="hidden" name="m_model" value={otherState.active.model} /> : null}
      {otherState.active.host ? <input type="hidden" name="m_host" value={otherState.active.host} /> : null}
      {otherState.active.status ? <input type="hidden" name="m_status" value={otherState.active.status} /> : null}
      {otherState.history.model ? (
        <input type="hidden" name="m_hist_model" value={otherState.history.model} />
      ) : null}
      {otherState.history.host ? (
        <input type="hidden" name="m_hist_host" value={otherState.history.host} />
      ) : null}
      {otherState.history.status ? (
        <input type="hidden" name="m_hist_status" value={otherState.history.status} />
      ) : null}
      {otherState.activePage > 1 ? (
        <input type="hidden" name="m_page" value={otherState.activePage} />
      ) : null}
      {otherState.activeLimit !== DEFAULT_PAGE_SIZE ? (
        <input type="hidden" name="m_limit" value={otherState.activeLimit} />
      ) : null}
      {otherState.historyPage > 1 ? (
        <input type="hidden" name="m_hist_page" value={otherState.historyPage} />
      ) : null}
      {otherState.historyLimit !== DEFAULT_PAGE_SIZE ? (
        <input type="hidden" name="m_hist_limit" value={otherState.historyLimit} />
      ) : null}

      <div className="filter-field">
        <label htmlFor={`filter-${idPrefix}-model`}>Model</label>
        <select id={`filter-${idPrefix}-model`} name={`${fieldPrefix}model`} defaultValue={filters.model}>
          <option value="">All</option>
          {modelNames.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>
      <div className="filter-field">
        <label htmlFor={`filter-${idPrefix}-host`}>Host</label>
        <select id={`filter-${idPrefix}-host`} name={`${fieldPrefix}host`} defaultValue={filters.host}>
          <option value="">All</option>
          {hostnames.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>
      {statusOptions ? (
        <div className="filter-field">
          <label htmlFor={`filter-${idPrefix}-status`}>Status</label>
          <select id={`filter-${idPrefix}-status`} name={`${fieldPrefix}status`} defaultValue={filters.status}>
            <option value="">All</option>
            {statusOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      <PageSizeSelect id={`filter-${idPrefix}-limit`} name={`${fieldPrefix}limit`} value={limit} />
      <div className="filter-actions">
        <button type="submit">Filter</button>
        {hasFilters ? (
          <a
            href={modelsHref(
              isActive
                ? { ...otherState, active: EMPTY_FILTERS, activePage: 1 }
                : { ...otherState, history: EMPTY_FILTERS, historyPage: 1 }
            )}
          >
            Clear
          </a>
        ) : null}
      </div>
    </form>
  );
}

export default async function ModelsView({ params }: { params: SearchParams }) {
  const activeFilters: SectionFilters = {
    model: firstValue(params.m_model) ?? "",
    host: firstValue(params.m_host) ?? "",
    status: firstValue(params.m_status) ?? ""
  };
  const historyFilters: SectionFilters = {
    model: firstValue(params.m_hist_model) ?? "",
    host: firstValue(params.m_hist_host) ?? "",
    status: firstValue(params.m_hist_status) ?? ""
  };
  const activePage = parsePage(params.m_page);
  const activeLimit = parsePageSize(params.m_limit);
  const historyPage = parsePage(params.m_hist_page);
  const historyLimit = parsePageSize(params.m_hist_limit);
  const state: ModelsState = {
    active: activeFilters,
    history: historyFilters,
    activePage,
    activeLimit,
    historyPage,
    historyLimit
  };

  let deployments: ModelDeploymentRead[];
  let models: ModelRead[];
  let hardware: HardwareRead[];
  let owners: HardwareOwnerRead[];
  try {
    [deployments, models, hardware, owners] = await Promise.all([
      getJson<ModelDeploymentRead[]>(LEDGER_BASE, `/model-deployments?limit=${LIST_LIMIT}`),
      getJson<ModelRead[]>(LEDGER_BASE, `/models?limit=${LIST_LIMIT}`),
      getJson<HardwareRead[]>(LEDGER_BASE, `/hardware?limit=${LIST_LIMIT}`),
      getJson<HardwareOwnerRead[]>(LEDGER_BASE, `/hardware-owners?limit=${LIST_LIMIT}`)
    ]);
  } catch (error) {
    return (
      <div className="error">
        Could not load model data from {LEDGER_BASE}:{" "}
        {error instanceof Error ? error.message : "unknown error"}
      </div>
    );
  }

  const modelsById = new Map(models.map((m) => [m.model_id, m]));
  const hardwareById = new Map(hardware.map((h) => [h.hardware_id, h]));
  const ownersById = new Map(owners.map((o) => [o.owner_id, o]));

  const rows: DeploymentRow[] = deployments.map((deployment) => {
    const host = hardwareById.get(deployment.hardware_id) ?? null;
    return {
      deployment,
      model: modelsById.get(deployment.model_id) ?? null,
      hardware: host,
      owner: host?.owner_id ? (ownersById.get(host.owner_id) ?? null) : null
    };
  });

  const modelNames = Array.from(new Set(models.map((m) => m.model_name))).sort();
  const hostnames = Array.from(new Set(hardware.map((h) => h.hostname))).sort();

  const allActive = rows.filter((row) => row.deployment.ended_at === null);
  const allEnded = rows.filter((row) => row.deployment.ended_at !== null);
  const active = applySectionFilters(allActive, activeFilters);
  const ended = applySectionFilters(allEnded, historyFilters);
  const activeSlice = active.slice((activePage - 1) * activeLimit, activePage * activeLimit);
  const endedSlice = ended.slice((historyPage - 1) * historyLimit, historyPage * historyLimit);

  const trustedHosts = new Set(
    hardware
      .filter((h) => h.owner_id && ownersById.get(h.owner_id)?.is_trusted)
      .map((h) => h.hardware_id)
  );
  const activeOnTrusted = allActive.filter((row) =>
    trustedHosts.has(row.deployment.hardware_id)
  );

  return (
    <>
      <section className="stats" aria-label="Model summary">
        <div className="stat">
          <span>Executing Models</span>
          <strong className="accent-pass">{allActive.length.toLocaleString()}</strong>
        </div>
        <div className="stat">
          <span>On Trusted Hardware</span>
          <strong>{activeOnTrusted.length.toLocaleString()}</strong>
        </div>
        <div className="stat">
          <span>Model Configs</span>
          <strong>{models.length.toLocaleString()}</strong>
        </div>
        <div className="stat">
          <span>Hosts</span>
          <strong>{hardware.length.toLocaleString()}</strong>
        </div>
        <div className="stat">
          <span>Deployments</span>
          <strong>{formatNumber(deployments.length)}</strong>
        </div>
      </section>

      <h2 className="section-title">Executing Models</h2>

      <SectionFilterForm
        idPrefix="m"
        fieldPrefix="m_"
        filters={activeFilters}
        limit={activeLimit}
        otherState={state}
        modelNames={modelNames}
        hostnames={hostnames}
        statusOptions={[
          { value: "active", label: "Executing" },
          { value: "ended", label: "Ended" }
        ]}
      />

      <DeploymentTable
        rows={activeSlice}
        emptyMessage={
          activeFilters.model || activeFilters.host || activeFilters.status
            ? "No executing models match the current filters."
            : "No models are currently executing."
        }
        showEnded={false}
      />

      <Pager
        label="Executing models pagination"
        page={activePage}
        pageSize={activeLimit}
        total={active.length}
        shown={activeSlice.length}
        previousHref={modelsHref({ ...state, activePage: activePage - 1 })}
        nextHref={modelsHref({ ...state, activePage: activePage + 1 })}
      />

      <h2 className="section-title">
        Deployment History ({ended.length.toLocaleString()})
      </h2>

      <SectionFilterForm
        idPrefix="m-hist"
        fieldPrefix="m_hist_"
        filters={historyFilters}
        limit={historyLimit}
        otherState={state}
        modelNames={modelNames}
        hostnames={hostnames}
        statusOptions={null}
      />

      <DeploymentTable
        rows={endedSlice}
        emptyMessage={
          historyFilters.model || historyFilters.host
            ? "No ended deployments match the current filters."
            : "No deployments have ended yet."
        }
        showEnded={true}
      />

      <Pager
        label="Deployment history pagination"
        page={historyPage}
        pageSize={historyLimit}
        total={ended.length}
        shown={endedSlice.length}
        previousHref={modelsHref({ ...state, historyPage: historyPage - 1 })}
        nextHref={modelsHref({ ...state, historyPage: historyPage + 1 })}
      />
    </>
  );
}
