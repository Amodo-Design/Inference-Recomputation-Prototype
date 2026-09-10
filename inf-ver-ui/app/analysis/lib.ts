// Pure math/data-shaping helpers for the Analysis tab. No React, no DOM.
// Later chart tasks import these exact names.

export type GpuActivity = {
  window_s: number;
  sample_count: number;
  tensor_active_time_s: number | null;
  sm_occupancy_mean: number | null;
  pipe_activity_s: Record<string, number> | null;
  concurrent_events: number | null;
};

// Display-time data-quality gate (spec: MIN_GPU_SAMPLES). At n=1-2 every
// Prometheus sample straddles the event-window boundary, so the integral
// mostly measures the ~400ms AROUND the event. Tunable here without
// re-enriching anything.
export const MIN_GPU_SAMPLES = 3;

export type AnalysisRow = {
  id: string;
  ts: string;
  result: string | null;
  mean_logit_difference: number | null;
  exact_match_level_pct: number | null;
  verification_threshold: number | null;
  prover_model: string;
  verifier_model: string;
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
  latency_ms: number | null;
  prover_gpu: GpuActivity | null;
  verify_gpu: GpuActivity | null;
};

export type DerivedRow = AnalysisRow & {
  pair: string; // `${prover_model} → ${verifier_model}`
  honest: boolean; // prover === verifier
  thresholdPass: boolean | null; // mean <= threshold (null if either missing)
  margin: number | null; // threshold − mean
  configKey: string; // JSON of [prover,verifier,temperature,top_k,top_p,threshold]
  proverBusyPerToken: number | null; // tensor-busy-seconds per generated token
  verifyBusyPerToken: number | null; // per prefilled token (prompt + output)
  efficiencyRatio: number | null; // prover per-token / verify per-token
  gpuEligible: boolean; // both sides sampled well enough & isolated
};

export const PALETTE: string[] = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
  "#4a3aa7",
  "#e34948",
];
export const OTHER_COLOR = "#5f5f5f";
export const PASS_COLOR = "#4f7e6b";
export const FAIL_COLOR = "#ca4f3e";

export function deriveRows(rows: AnalysisRow[]): DerivedRow[] {
  return rows.map((row) => {
    const pair = `${row.prover_model} → ${row.verifier_model}`;
    const honest = row.prover_model === row.verifier_model;
    const hasMean = row.mean_logit_difference !== null;
    const hasThreshold = row.verification_threshold !== null;
    const thresholdPass =
      hasMean && hasThreshold
        ? row.mean_logit_difference! <= row.verification_threshold!
        : null;
    const margin =
      hasMean && hasThreshold
        ? row.verification_threshold! - row.mean_logit_difference!
        : null;
    const configKey = JSON.stringify([
      row.prover_model,
      row.verifier_model,
      row.temperature,
      row.top_k,
      row.top_p,
      row.verification_threshold,
    ]);
    // Prover cost is per generated token (decode); verify cost is per
    // prefilled token (one pass over prompt + output).
    const proverBusyPerToken =
      row.prover_gpu?.tensor_active_time_s != null && (row.output_tokens ?? 0) > 0
        ? row.prover_gpu.tensor_active_time_s / row.output_tokens!
        : null;
    const verifyTokens = (row.prompt_tokens ?? 0) + (row.output_tokens ?? 0);
    const verifyBusyPerToken =
      row.verify_gpu?.tensor_active_time_s != null && verifyTokens > 0
        ? row.verify_gpu.tensor_active_time_s / verifyTokens
        : null;
    const efficiencyRatio =
      proverBusyPerToken !== null && verifyBusyPerToken !== null && verifyBusyPerToken > 0
        ? proverBusyPerToken / verifyBusyPerToken
        : null;
    const sideOk = (g: GpuActivity | null) =>
      g !== null && g.sample_count >= MIN_GPU_SAMPLES && (g.concurrent_events ?? 0) === 0;
    const gpuEligible = sideOk(row.prover_gpu) && sideOk(row.verify_gpu);
    return {
      ...row,
      pair,
      honest,
      thresholdPass,
      margin,
      configKey,
      proverBusyPerToken,
      verifyBusyPerToken,
      efficiencyRatio,
      gpuEligible,
    };
  });
}

// Quality gate for the GPU-efficiency charts. `excludedCount` counts rows
// that HAVE GPU data on both sides but fail the sample-count/concurrency
// gates (surfaced as "n excluded" in chart notes); rows without GPU data
// are pre-enrichment and not counted as exclusions.
export function gpuEligibleRows(rows: DerivedRow[]): {
  kept: DerivedRow[];
  excludedCount: number;
} {
  const withData = rows.filter((r) => r.prover_gpu !== null && r.verify_gpu !== null);
  const kept = withData.filter((r) => r.gpuEligible);
  return { kept, excludedCount: withData.length - kept.length };
}

// Model names arrive namespaced ("Qwen/Qwen2.5-1.5B-Instruct"); axis and
// in-plot labels only have room for the part after the last slash. Tooltips
// keep the full name.
export function shortModelName(model: string): string {
  const slash = model.lastIndexOf("/");
  return slash === -1 ? model : model.slice(slash + 1);
}

export function shortPairLabel(pair: string): string {
  return pair
    .split(" → ")
    .map(shortModelName)
    .join(" → ");
}

export function pairColorMap(rows: DerivedRow[]): Map<string, string> {
  const uniquePairs = Array.from(new Set(rows.map((r) => r.pair))).sort();
  const map = new Map<string, string>();
  uniquePairs.forEach((pair, i) => {
    map.set(pair, i < PALETTE.length ? PALETTE[i] : OTHER_COLOR);
  });
  return map;
}

// Loop-based max/min over a values array. Math.max(...arr)/Math.min(...arr)
// spreads the array as call arguments, which throws a RangeError ("Maximum
// call stack size exceeded") in Safari (and eventually elsewhere) once the
// array reaches roughly 60k-100k elements — a real risk for per-point chart
// data. These have no such ceiling.
export function maxOf(values: number[]): number {
  let max = -Infinity;
  for (const v of values) {
    if (v > max) max = v;
  }
  return max;
}

export function minOf(values: number[]): number {
  let min = Infinity;
  for (const v of values) {
    if (v < min) min = v;
  }
  return min;
}

export function weightedMean(
  values: number[],
  weights: number[]
): number | null {
  if (values.length === 0 || weights.length === 0) return null;
  let sumWV = 0;
  let sumW = 0;
  for (let i = 0; i < values.length; i++) {
    sumWV += values[i] * weights[i];
    sumW += weights[i];
  }
  if (sumW === 0) return null;
  return sumWV / sumW;
}

// numpy-default ("linear") quantile interpolation over an already-sorted array.
function quantile(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const h = (n - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

export function quantileBins(
  rows: { x: number; y: number }[],
  binCount: number
): {
  x0: number;
  x1: number;
  xMid: number;
  mean: number;
  p25: number;
  p75: number;
  count: number;
}[] {
  const xs = rows.map((r) => r.x);
  const uniqueX = new Set(xs);
  if (uniqueX.size < 2) return [];

  const sortedX = [...xs].sort((a, b) => a - b);
  const edges: number[] = [];
  for (let i = 0; i <= binCount; i++) {
    edges.push(quantile(sortedX, i / binCount));
  }
  // pd.qcut drops duplicate (degenerate) edges.
  const dedupedEdges: number[] = [edges[0]];
  for (let i = 1; i < edges.length; i++) {
    if (edges[i] !== dedupedEdges[dedupedEdges.length - 1]) {
      dedupedEdges.push(edges[i]);
    }
  }
  if (dedupedEdges.length < 2) return [];

  const bins: {
    x0: number;
    x1: number;
    xMid: number;
    mean: number;
    p25: number;
    p75: number;
    count: number;
  }[] = [];

  for (let i = 0; i < dedupedEdges.length - 1; i++) {
    const x0 = dedupedEdges[i];
    const x1 = dedupedEdges[i + 1];
    // First bin is inclusive of the minimum (pandas cut semantics for qcut).
    const inBin = rows.filter((r) =>
      i === 0 ? r.x >= x0 && r.x <= x1 : r.x > x0 && r.x <= x1
    );
    if (inBin.length === 0) continue;
    const ys = inBin.map((r) => r.y).sort((a, b) => a - b);
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
    bins.push({
      x0,
      x1,
      xMid: (x0 + x1) / 2,
      mean,
      p25: quantile(ys, 0.25),
      p75: quantile(ys, 0.75),
      count: inBin.length,
    });
  }
  return bins;
}

export function iqrUpperCut(values: number[], k: number): number {
  if (values.length < 4) return Infinity;
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = quantile(sorted, 0.25);
  const q3 = quantile(sorted, 0.75);
  return q3 + k * (q3 - q1);
}

export function clipUpper(
  rows: { x: number }[],
  k: number
): { kept: { x: number }[]; hiddenCount: number; cutoff: number } {
  const cutoff = iqrUpperCut(
    rows.map((r) => r.x),
    k
  );
  const kept = rows.filter((r) => r.x <= cutoff);
  return { kept, hiddenCount: rows.length - kept.length, cutoff };
}

export function thresholdSweep(
  rows: DerivedRow[],
  grid: number[]
): Map<string, { threshold: number; passRate: number }[]> {
  const byConfig = new Map<string, DerivedRow[]>();
  for (const row of rows) {
    const list = byConfig.get(row.configKey) ?? [];
    list.push(row);
    byConfig.set(row.configKey, list);
  }

  const result = new Map<string, { threshold: number; passRate: number }[]>();
  for (const [configKey, configRows] of byConfig) {
    const withMean = configRows.filter(
      (r) => r.mean_logit_difference !== null
    );
    const sweep = grid.map((threshold) => {
      const passRate =
        withMean.length === 0
          ? 0
          : withMean.filter((r) => r.mean_logit_difference! <= threshold)
              .length / withMean.length;
      return { threshold, passRate };
    });
    result.set(configKey, sweep);
  }
  return result;
}

export function makeGrid(start: number, end: number, step: number): number[] {
  const decimalsOf = (n: number) => {
    const s = n.toString();
    const dot = s.indexOf(".");
    return dot === -1 ? 0 : s.length - dot - 1;
  };
  const decimals = Math.max(decimalsOf(start), decimalsOf(end), decimalsOf(step));
  const n = Math.round((end - start) / step);
  const grid: number[] = [];
  for (let i = 0; i <= n; i++) {
    const raw = start + i * step;
    grid.push(Number(raw.toFixed(decimals)));
  }
  return grid;
}

export function groupWeightedMeanDiff(rows: DerivedRow[]): {
  verifier: string;
  prover: string;
  temperature: number | null;
  value: number;
  count: number;
}[] {
  const groups = new Map<
    string,
    {
      verifier: string;
      prover: string;
      temperature: number | null;
      values: number[];
    }
  >();
  for (const row of rows) {
    if (row.mean_logit_difference === null) continue;
    const key = JSON.stringify([
      row.verifier_model,
      row.prover_model,
      row.temperature,
    ]);
    const entry = groups.get(key) ?? {
      verifier: row.verifier_model,
      prover: row.prover_model,
      temperature: row.temperature,
      values: [],
    };
    entry.values.push(row.mean_logit_difference);
    groups.set(key, entry);
  }
  return Array.from(groups.values()).map((g) => ({
    verifier: g.verifier,
    prover: g.prover,
    temperature: g.temperature,
    value: weightedMean(
      g.values,
      g.values.map(() => 1)
    )!,
    count: g.values.length,
  }));
}

function niceFloorPow10(v: number): number {
  return Math.pow(10, Math.floor(Math.log10(v)));
}
function niceCeilPow10(v: number): number {
  return Math.pow(10, Math.ceil(Math.log10(v)));
}

export function heatmapNorm(
  values: number[]
): { kind: "log" | "symlog"; min: number; max: number } {
  const allPositive = values.every((v) => v > 0);
  if (allPositive) {
    const min = minOf(values);
    const max = maxOf(values);
    return { kind: "log", min: niceFloorPow10(min), max: niceCeilPow10(max) };
  }
  const maxAbs = maxOf(values.map((v) => Math.abs(v)));
  const bound = niceCeilPow10(maxAbs);
  return { kind: "symlog", min: -bound, max: bound };
}

const SYMLOG_LINTHRESH = 0.1;

function symlogTransform(v: number): number {
  const sign = Math.sign(v);
  const av = Math.abs(v);
  return av <= SYMLOG_LINTHRESH
    ? v / SYMLOG_LINTHRESH
    : sign * (1 + Math.log10(av / SYMLOG_LINTHRESH));
}

// Maps a value into [0, 1] position along a heatmapNorm scale (as produced
// by `heatmapNorm`), for use as a color-ramp lookup position. "log" applies
// a plain log10 interpolation between the norm's min/max; "symlog" applies
// the same symlog transform (linthresh 0.1) to value/min/max before
// interpolating, matching the sign-preserving log used for the ±-valued
// case. Result is clamped to [0, 1] since a single cell's value can fall
// slightly outside the shared min/max in edge cases.
export function normPosition(
  norm: { kind: "log" | "symlog"; min: number; max: number },
  value: number
): number {
  let lo: number;
  let hi: number;
  let v: number;
  if (norm.kind === "log") {
    lo = Math.log10(norm.min);
    hi = Math.log10(norm.max);
    v = Math.log10(value);
  } else {
    lo = symlogTransform(norm.min);
    hi = symlogTransform(norm.max);
    v = symlogTransform(value);
  }
  if (hi === lo) return 0.5;
  const pos = (v - lo) / (hi - lo);
  return Math.min(1, Math.max(0, pos));
}

export function verificationMargins(rows: DerivedRow[]): {
  verifier: string;
  incorrectProver: string;
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  value: number;
}[] {
  type Bucket = {
    verifier: string;
    temperature: number | null;
    top_k: number | null;
    top_p: number | null;
    honestValues: number[];
    byProver: Map<string, number[]>;
  };
  const buckets = new Map<string, Bucket>();

  for (const row of rows) {
    if (row.mean_logit_difference === null) continue;
    const key = JSON.stringify([
      row.verifier_model,
      row.temperature,
      row.top_k,
      row.top_p,
    ]);
    const bucket = buckets.get(key) ?? {
      verifier: row.verifier_model,
      temperature: row.temperature,
      top_k: row.top_k,
      top_p: row.top_p,
      honestValues: [],
      byProver: new Map<string, number[]>(),
    };
    if (row.honest) {
      bucket.honestValues.push(row.mean_logit_difference);
    } else {
      const list = bucket.byProver.get(row.prover_model) ?? [];
      list.push(row.mean_logit_difference);
      bucket.byProver.set(row.prover_model, list);
    }
    buckets.set(key, bucket);
  }

  const result: {
    verifier: string;
    incorrectProver: string;
    temperature: number | null;
    top_k: number | null;
    top_p: number | null;
    value: number;
  }[] = [];

  for (const bucket of buckets.values()) {
    const honestMean = weightedMean(
      bucket.honestValues,
      bucket.honestValues.map(() => 1)
    );
    if (honestMean === null) continue;
    for (const [prover, values] of bucket.byProver) {
      const incorrectMean = weightedMean(
        values,
        values.map(() => 1)
      )!;
      result.push({
        verifier: bucket.verifier,
        incorrectProver: prover,
        temperature: bucket.temperature,
        top_k: bucket.top_k,
        top_p: bucket.top_p,
        value: incorrectMean - honestMean,
      });
    }
  }
  return result;
}
