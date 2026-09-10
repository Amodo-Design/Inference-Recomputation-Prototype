import { ANALYTICS_BASE } from "../lib";
import { formatPerToken, formatRatio, formatSeconds } from "./economics-format";

type EconomicsSide = {
  busy_seconds: number | null;
  wall_clock_s: number | null;
  busy_per_token: number | null;
  wall_clock_per_token: number | null;
  method: string;
};

type Concurrency = {
  configured: number | null;
  observed_mean: number | null;
  observed_max: number | null;
};

type ModelEconomics = {
  model: string;
  prover: EconomicsSide & { output_tokens: number | null; sample_count: number | null; pods: string[] };
  verify: EconomicsSide & {
    prompt_plus_output_tokens: number | null;
    event_count: number;
    sample_count: number | null;
    pods: string[];
    concurrency: Concurrency | null;
    window: { from: string | null; to: string | null };
  };
};

type RunEconomics = {
  run_id: string;
  window: { from: string | null; to: string | null };
  models: ModelEconomics[];
};

// Verification advantage: GPU busy-seconds per token spent proving divided by
// per token spent verifying. Only defined when both sides measured a
// positive per-token cost — a null or zero verifier per-token reading would
// otherwise produce an infinite or meaningless multiplier.
function advantageRatio(proverBusyPerToken: number | null, verifyBusyPerToken: number | null): number | null {
  if (
    proverBusyPerToken === null ||
    verifyBusyPerToken === null ||
    !(proverBusyPerToken > 0) ||
    !(verifyBusyPerToken > 0)
  ) {
    return null;
  }
  return proverBusyPerToken / verifyBusyPerToken;
}

// "verifier conc 3 (global) · overlap μ2.5 max 3" — configured is the
// PLATFORM-WIDE verifier drain setting (ORCH_RUNNER_MODEL_CONCURRENCY in the
// kube ConfigMap), deliberately independent of the run form's per-model
// prover concurrency sliders — which is why it reads the same for every run.
// observed_mean/observed_max count OTHER simultaneously running
// verifications alongside each event, so at saturation the overlap reads
// ≈ configured − 1, not ≈ configured.
function concurrencyLabel(c: Concurrency): string {
  const conf = c.configured !== null ? `verifier conc ${c.configured} (global)` : "verifier conc —";
  if (c.observed_max === null || c.observed_mean === null) return conf;
  return `${conf} · overlap μ${c.observed_mean.toFixed(1)} max ${c.observed_max}`;
}

// Names the side(s) that kept the ratio undefined. A side is unusable when
// its per-token value is null (window unmeasured) OR non-positive (measured
// but zero activity) — blaming only nulls would misdirect the reader when a
// zero-valued but successfully-measured side is the culprit.
function describeMissingRatio(prover: number | null, verify: number | null): string {
  const proverBad = prover === null || !(prover > 0);
  const verifyBad = verify === null || !(verify > 0);
  const side =
    proverBad && verifyBad ? "prover and verifier GPU" : proverBad ? "prover GPU" : "verifier GPU";
  return `${side} per-token cost unavailable — see measurement detail`;
}

const CONCURRENCY_TITLE =
  "The platform-wide verifier drain concurrency (ORCH_RUNNER_MODEL_CONCURRENCY) in force when this run's " +
  "events were verified — NOT this run's prover concurrency setting. Overlap counts OTHER " +
  "simultaneously-running verifications, so it reads ≈ conc−1 at saturation.";

/** Run-scoped GPU economics: per-model prover/verify busy- and wall-clock
 * time per token, sourced from analytics-api's cohort-based /economics
 * route. Each side's method is labelled explicitly (both sides pod-window
 * integrals; methods rendered from the payload) so the two numbers are
 * never read as directly comparable measurement methodologies — only the
 * advantage stat claims that. */
export default async function RunEconomicsPanel({ runId }: { runId: string }) {
  let economics: RunEconomics;
  try {
    const response = await fetch(`${ANALYTICS_BASE}/runs/${encodeURIComponent(runId)}/economics`, {
      cache: "no-store"
    });
    if (!response.ok) {
      const reason =
        response.status === 409 ? "run has not started" : `analytics-api returned ${response.status}`;
      return (
        <section className="chart-card run-economics-panel" aria-label="Run economics">
          <p className="muted">economics unavailable: {reason}</p>
        </section>
      );
    }
    economics = await response.json();
  } catch (error) {
    return (
      <section className="chart-card run-economics-panel" aria-label="Run economics">
        <p className="muted">
          economics unavailable: {error instanceof Error ? error.message : "unreachable"}
        </p>
      </section>
    );
  }

  // Guard against an unexpected 200 shape (e.g. `models` missing or not an
  // array) degrading gracefully instead of throwing inside this server
  // component.
  const models = economics.models ?? [];

  if (models.length === 0) {
    return (
      <section className="chart-card run-economics-panel" aria-label="Run economics">
        <p className="muted">economics unavailable: no models in this run</p>
      </section>
    );
  }

  return (
    <section className="chart-card run-economics-panel" aria-label="Run economics">
      <h3>Run Economics</h3>
      {models.map((model) => {
        const ratio = advantageRatio(model.prover.busy_per_token, model.verify.busy_per_token);
        return (
          <div className="econ-model" key={model.model}>
            <div className="econ-model-header">
              <span className="econ-model-name">{model.model}</span>
              <span className="econ-concurrency" title={CONCURRENCY_TITLE}>
                {concurrencyLabel(
                  model.verify.concurrency ?? { configured: null, observed_mean: null, observed_max: null }
                )}
              </span>
            </div>

            <div className="econ-stats">
              <div className="econ-stat econ-stat-primary">
                <span className="control-label">Verification advantage</span>
                <span className="econ-stat-value econ-stat-value-primary">{formatRatio(ratio)}</span>
                <span className="econ-stat-desc muted">
                  {ratio !== null
                    ? "× less GPU time per token to verify than to generate"
                    : describeMissingRatio(model.prover.busy_per_token, model.verify.busy_per_token)}
                </span>
              </div>
              <div className="econ-stat">
                <span className="control-label">Prover GPU per token</span>
                <span className="econ-stat-value">{formatPerToken(model.prover.busy_per_token)}</span>
                <span className="econ-stat-desc muted">tensor-core busy-seconds per generated token</span>
              </div>
              <div className="econ-stat">
                <span className="control-label">Verifier GPU per token</span>
                <span className="econ-stat-value">{formatPerToken(model.verify.busy_per_token)}</span>
                <span className="econ-stat-desc muted">busy-seconds per checked token (prompt + output)</span>
              </div>
            </div>

            <details className="econ-detail">
              <summary>Measurement detail</summary>
              <div className="econ-sides">
                <div className="econ-side">
                  <h4>Prover</h4>
                  <dl>
                    <dt>GPU busy</dt>
                    <dd>{formatSeconds(model.prover.busy_seconds)}</dd>
                    <dt>Wall clock</dt>
                    <dd>{formatSeconds(model.prover.wall_clock_s)}</dd>
                    <dt>Tokens generated</dt>
                    <dd>{model.prover.output_tokens ?? "—"}</dd>
                    <dt>Wall clock per token</dt>
                    <dd>{formatPerToken(model.prover.wall_clock_per_token)}</dd>
                    <dt>DCGM samples</dt>
                    <dd>{model.prover.sample_count ?? "—"}</dd>
                  </dl>
                  <p className="econ-side-caption muted">
                    Measured as one GPU-activity integral over the run window across the run&apos;s
                    prover pods ({model.prover.method}): {model.prover.pods.join(", ") || "—"}.
                  </p>
                </div>
                <div className="econ-side">
                  <h4>Verifier</h4>
                  <dl>
                    <dt>GPU busy</dt>
                    <dd>{formatSeconds(model.verify.busy_seconds)}</dd>
                    <dt>Wall clock</dt>
                    <dd>{formatSeconds(model.verify.wall_clock_s)}</dd>
                    <dt>Tokens checked</dt>
                    <dd>{model.verify.prompt_plus_output_tokens ?? "—"}</dd>
                    <dt>Wall clock per token</dt>
                    <dd>{formatPerToken(model.verify.wall_clock_per_token)}</dd>
                    <dt>DCGM samples</dt>
                    <dd>{model.verify.sample_count ?? "—"}</dd>
                    <dt>Drain window</dt>
                    <dd>
                      {model.verify.window?.from && model.verify.window?.to
                        ? `${new Date(model.verify.window.from).toLocaleTimeString()} → ${new Date(model.verify.window.to).toLocaleTimeString()}`
                        : "—"}
                    </dd>
                  </dl>
                  <p className="econ-side-caption muted">
                    Measured as one integral over the drain window across the verifier pods (
                    {model.verify.method}): {model.verify.pods.join(", ") || "—"}.
                  </p>
                </div>
              </div>
            </details>
          </div>
        );
      })}
    </section>
  );
}
