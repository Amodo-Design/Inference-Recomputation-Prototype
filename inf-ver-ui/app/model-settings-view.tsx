import { LEDGER_BASE, ModelRead, getJson } from "./lib";
import ModelSettingField from "./model-setting-field";

// The ledger's list endpoints cap at 1000 rows; plenty for the prototype.
const LIST_LIMIT = 1000;

function SamplingTags({ model }: { model: ModelRead }) {
  const params: { label: string; value: number | null }[] = [
    { label: "temp", value: model.temperature },
    { label: "top_k", value: model.top_k },
    { label: "top_p", value: model.top_p },
    { label: "seed", value: model.seed }
  ];
  return (
    <div className="sampling-tags">
      {model.decoding_algorithm ? (
        <span className="sampling-tag sampling-tag-algo">{model.decoding_algorithm}</span>
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

/** Per-model verification settings, one row per model config seen by the
 * ledger (NOT per deployment instance — the Model Instances tab covers
 * those). Both settings live on the ledger model row, mutable and outside
 * the identity hash:
 * - Logit difference threshold: pass mark; blank/NULL = verification PAUSED
 *   (the orchestrator spawns no runners; pending events wait).
 * - Delta max: per-token margin cap; also the margin recorded when the
 *   prover's token falls outside the verifier's candidate set. Blank/NULL =
 *   the runner's built-in default (10.0).
 */
export default async function ModelSettingsView() {
  let models: ModelRead[];
  try {
    models = await getJson<ModelRead[]>(LEDGER_BASE, `/models?limit=${LIST_LIMIT}`);
  } catch (error) {
    return (
      <div className="error">
        Could not load models from {LEDGER_BASE}:{" "}
        {error instanceof Error ? error.message : "unknown error"}
      </div>
    );
  }

  // Prover configs carry the sampling the taps declared; rows without any
  // sampling are verifier self-declarations (runners) — settings on those
  // are meaningless, so they are listed last and flagged.
  const isVerifierRow = (m: ModelRead) =>
    m.temperature === null && m.top_k === null && m.top_p === null && m.seed === null;
  const sorted = [...models].sort((a, b) => {
    const verifierOrder = Number(isVerifierRow(a)) - Number(isVerifierRow(b));
    if (verifierOrder !== 0) return verifierOrder;
    return (
      a.model_name.localeCompare(b.model_name) || (a.top_k ?? 0) - (b.top_k ?? 0)
    );
  });

  const paused = models.filter(
    (m) => !isVerifierRow(m) && m.verification_threshold === null
  ).length;

  return (
    <>
      <section className="stats" aria-label="Model settings summary">
        <div className="stat">
          <span>Model Configs</span>
          <strong>{models.length.toLocaleString()}</strong>
        </div>
        <div className="stat">
          <span>Verification Paused</span>
          <strong className={paused > 0 ? "accent-unknown" : ""}>
            {paused.toLocaleString()}
          </strong>
        </div>
      </section>

      <h2 className="section-title">Model Settings</h2>
      <p className="muted">
        One row per model configuration the ledger has seen. Settings apply to
        the configuration, not to individual deployments. A blank threshold
        pauses verification for that model (events stay pending); a blank
        delta max uses the runner default (10.0).
      </p>

      {sorted.length === 0 ? (
        <div className="empty">No models have been declared yet.</div>
      ) : (
        <div className="table-wrap">
          <table className="table-narrow">
            <thead>
              <tr>
                <th>Model</th>
                <th>Sampling Config</th>
                <th>Logit Difference Threshold</th>
                <th>Delta Max</th>
                <th>Database Row</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((model) => (
                <tr key={model.model_id}>
                  <td>
                    <div>{model.model_name}</div>
                    {isVerifierRow(model) ? (
                      <span className="muted">verifier self-declaration</span>
                    ) : null}
                  </td>
                  <td className="mono">
                    <SamplingTags model={model} />
                  </td>
                  <td>
                    {isVerifierRow(model) ? (
                      <span className="muted">-</span>
                    ) : (
                      <ModelSettingField
                        model={model}
                        field="verification_threshold"
                        endpoint="threshold"
                        placeholder="paused"
                        nullNote="Cleared — verification paused."
                        setNote="Set to"
                      />
                    )}
                  </td>
                  <td>
                    {isVerifierRow(model) ? (
                      <span className="muted">-</span>
                    ) : (
                      <ModelSettingField
                        model={model}
                        field="delta_max"
                        endpoint="delta-max"
                        placeholder="default (10.0)"
                        nullNote="Cleared — runner default (10.0)."
                        setNote="Set to"
                      />
                    )}
                  </td>
                  <td>
                    <details>
                      <summary>All fields</summary>
                      <pre>{JSON.stringify(model, null, 2)}</pre>
                    </details>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
