import { describe, it, expect } from "vitest";
import {
  AnalysisRow,
  DerivedRow,
  deriveRows,
  pairColorMap,
  PALETTE,
  OTHER_COLOR,
  PASS_COLOR,
  FAIL_COLOR,
  weightedMean,
  quantileBins,
  iqrUpperCut,
  clipUpper,
  thresholdSweep,
  makeGrid,
  groupWeightedMeanDiff,
  heatmapNorm,
  normPosition,
  shortModelName,
  shortPairLabel,
  verificationMargins,
  maxOf,
  minOf,
  gpuEligibleRows,
  MIN_GPU_SAMPLES,
  GpuActivity,
} from "./lib";

function mkRow(overrides: Partial<AnalysisRow> = {}): AnalysisRow {
  return {
    id: "id",
    ts: "2026-01-01T00:00:00Z",
    result: "pass",
    mean_logit_difference: null,
    exact_match_level_pct: null,
    verification_threshold: null,
    prover_model: "P",
    verifier_model: "V",
    temperature: null,
    top_k: null,
    top_p: null,
    prompt_tokens: null,
    output_tokens: null,
    latency_ms: null,
    prover_gpu: null,
    verify_gpu: null,
    ...overrides,
  };
}

describe("deriveRows", () => {
  it("sets honest, thresholdPass, margin for a matching prover/verifier under threshold", () => {
    const rows = [
      mkRow({
        id: "1",
        prover_model: "Qwen",
        verifier_model: "Qwen",
        mean_logit_difference: 0.1,
        verification_threshold: 0.3,
      }),
    ];
    const [d] = deriveRows(rows);
    expect(d.pair).toBe("Qwen → Qwen");
    expect(d.honest).toBe(true);
    expect(d.thresholdPass).toBe(true); // 0.1 <= 0.3
    expect(d.margin).toBeCloseTo(0.2); // 0.3 - 0.1
  });

  it("sets thresholdPass false and negative margin when mean exceeds threshold, honest false for differing models", () => {
    const rows = [
      mkRow({
        id: "2",
        prover_model: "Qwen",
        verifier_model: "Llama",
        mean_logit_difference: 0.5,
        verification_threshold: 0.3,
      }),
    ];
    const [d] = deriveRows(rows);
    expect(d.pair).toBe("Qwen → Llama");
    expect(d.honest).toBe(false);
    expect(d.thresholdPass).toBe(false); // 0.5 <= 0.3 is false
    expect(d.margin).toBeCloseTo(-0.2); // 0.3 - 0.5
  });

  it("sets thresholdPass and margin to null when threshold is missing", () => {
    const rows = [
      mkRow({
        id: "3",
        mean_logit_difference: 0.1,
        verification_threshold: null,
      }),
    ];
    const [d] = deriveRows(rows);
    expect(d.thresholdPass).toBeNull();
    expect(d.margin).toBeNull();
  });

  it("sets thresholdPass and margin to null when mean is missing", () => {
    const rows = [
      mkRow({
        id: "4",
        mean_logit_difference: null,
        verification_threshold: 0.3,
      }),
    ];
    const [d] = deriveRows(rows);
    expect(d.thresholdPass).toBeNull();
    expect(d.margin).toBeNull();
  });

  it("builds a configKey from prover/verifier/temperature/top_k/top_p/threshold", () => {
    const rows = [
      mkRow({
        id: "5",
        prover_model: "A",
        verifier_model: "B",
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        verification_threshold: 0.3,
      }),
    ];
    const [d] = deriveRows(rows);
    expect(d.configKey).toBe(JSON.stringify(["A", "B", 0.5, 10, 0.9, 0.3]));
  });
});

describe("pairColorMap", () => {
  it("assigns the first 3 palette hexes in alphabetical pair order for 3 pairs", () => {
    const rows = deriveRows([
      mkRow({ id: "1", prover_model: "C", verifier_model: "D" }), // "C → D"
      mkRow({ id: "2", prover_model: "A", verifier_model: "B" }), // "A → B"
      mkRow({ id: "3", prover_model: "B", verifier_model: "C" }), // "B → C"
    ]);
    const map = pairColorMap(rows);
    // alphabetical order: "A → B", "B → C", "C → D"
    expect(map.get("A → B")).toBe(PALETTE[0]);
    expect(map.get("B → C")).toBe(PALETTE[1]);
    expect(map.get("C → D")).toBe(PALETTE[2]);
  });

  it("assigns OTHER_COLOR to the 9th and 10th alphabetical pair", () => {
    const rows = deriveRows(
      Array.from({ length: 10 }, (_, i) =>
        mkRow({
          id: String(i),
          prover_model: `P${String(i).padStart(2, "0")}`,
          verifier_model: `V${String(i).padStart(2, "0")}`,
        })
      )
    );
    const map = pairColorMap(rows);
    const sortedPairs = rows.map((r) => r.pair).sort();
    expect(sortedPairs.length).toBe(10);
    for (let i = 0; i < 8; i++) {
      expect(map.get(sortedPairs[i])).toBe(PALETTE[i]);
    }
    expect(map.get(sortedPairs[8])).toBe(OTHER_COLOR);
    expect(map.get(sortedPairs[9])).toBe(OTHER_COLOR);
  });

  it("produces the same assignment when called on a filtered subset that still contains the full pair list", () => {
    const full = deriveRows([
      mkRow({ id: "1a", prover_model: "A", verifier_model: "B" }),
      mkRow({ id: "1b", prover_model: "A", verifier_model: "B" }),
      mkRow({ id: "2a", prover_model: "B", verifier_model: "C" }),
      mkRow({ id: "2b", prover_model: "B", verifier_model: "C" }),
      mkRow({ id: "3a", prover_model: "C", verifier_model: "D" }),
      mkRow({ id: "3b", prover_model: "C", verifier_model: "D" }),
    ]);
    const fullMap = pairColorMap(full);
    // Subset keeping only one row per pair -- the *set* of unique pairs is
    // unchanged, so callers must pass the full dataset for a stable mapping.
    const subset = full.filter((r) => r.id.endsWith("a"));
    const subsetMap = pairColorMap(subset);
    expect(Object.fromEntries(subsetMap)).toEqual(Object.fromEntries(fullMap));
  });
});

describe("weightedMean", () => {
  it("computes the weighted average", () => {
    expect(weightedMean([1, 3], [1, 3])).toBe(2.5);
  });

  it("returns null for empty input", () => {
    expect(weightedMean([], [])).toBeNull();
  });
});

describe("quantileBins", () => {
  it("bins 8 points into 4 quantile bins of 2 each with correct stats", () => {
    const rows = [1, 2, 3, 4, 5, 6, 7, 8].map((x) => ({ x, y: x * 10 }));
    const bins = quantileBins(rows, 4);
    expect(bins).toHaveLength(4);

    // quantile edges of [1..8] at 0/25/50/75/100%: 1, 2.75, 4.5, 6.25, 8
    expect(bins[0].x0).toBeCloseTo(1);
    expect(bins[0].x1).toBeCloseTo(2.75);
    expect(bins[0].xMid).toBeCloseTo(1.875);
    expect(bins[0].count).toBe(2);
    expect(bins[0].mean).toBeCloseTo(15); // mean([10,20])
    expect(bins[0].p25).toBeCloseTo(12.5);
    expect(bins[0].p75).toBeCloseTo(17.5);

    expect(bins[1].x0).toBeCloseTo(2.75);
    expect(bins[1].x1).toBeCloseTo(4.5);
    expect(bins[1].count).toBe(2);
    expect(bins[1].mean).toBeCloseTo(35); // mean([30,40])
    expect(bins[1].p25).toBeCloseTo(32.5);
    expect(bins[1].p75).toBeCloseTo(37.5);

    expect(bins[2].x0).toBeCloseTo(4.5);
    expect(bins[2].x1).toBeCloseTo(6.25);
    expect(bins[2].count).toBe(2);
    expect(bins[2].mean).toBeCloseTo(55); // mean([50,60])
    expect(bins[2].p25).toBeCloseTo(52.5);
    expect(bins[2].p75).toBeCloseTo(57.5);

    expect(bins[3].x0).toBeCloseTo(6.25);
    expect(bins[3].x1).toBeCloseTo(8);
    expect(bins[3].count).toBe(2);
    expect(bins[3].mean).toBeCloseTo(75); // mean([70,80])
    expect(bins[3].p25).toBeCloseTo(72.5);
    expect(bins[3].p75).toBeCloseTo(77.5);
  });

  it("returns [] when all x values are identical", () => {
    const rows = [1, 2, 3].map(() => ({ x: 5, y: 1 }));
    expect(quantileBins(rows, 4)).toEqual([]);
  });
});

describe("iqrUpperCut", () => {
  it("computes q3 + k*(q3-q1) by linear interpolation for [1..8], k=3", () => {
    // sorted [1..8], n=8
    // q1: h=(8-1)*0.25=1.75 -> sorted[1]=2, sorted[2]=3 -> 2 + 0.75*(3-2) = 2.75
    // q3: h=(8-1)*0.75=5.25 -> sorted[5]=6, sorted[6]=7 -> 6 + 0.25*(7-6) = 6.25
    // iqr = 3.5; cutoff = 6.25 + 3*3.5 = 16.75
    expect(iqrUpperCut([1, 2, 3, 4, 5, 6, 7, 8], 3)).toBeCloseTo(16.75);
  });

  it("returns Infinity for fewer than 4 values", () => {
    expect(iqrUpperCut([1, 2, 3], 3)).toBe(Infinity);
  });
});

describe("clipUpper", () => {
  it("hides rows above the IQR cutoff and reports the cutoff", () => {
    // values [1..8, 50], n=9
    // q1: h=(9-1)*0.25=2.0 -> sorted[2]=3
    // q3: h=(9-1)*0.75=6.0 -> sorted[6]=7
    // iqr=4; cutoff = 7 + 3*4 = 19; only 50 exceeds it
    const rows = [1, 2, 3, 4, 5, 6, 7, 8, 50].map((x) => ({ x }));
    const { kept, hiddenCount, cutoff } = clipUpper(rows, 3);
    expect(cutoff).toBeCloseTo(19);
    expect(hiddenCount).toBe(1);
    expect(kept).toHaveLength(8);
    expect(kept.map((r) => r.x)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });
});

describe("thresholdSweep", () => {
  it("computes pass rates across a grid for a single config", () => {
    const rows = deriveRows([
      mkRow({
        id: "1",
        prover_model: "A",
        verifier_model: "B",
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        verification_threshold: 0.3,
        mean_logit_difference: 0.1,
      }),
      mkRow({
        id: "2",
        prover_model: "A",
        verifier_model: "B",
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        verification_threshold: 0.3,
        mean_logit_difference: 0.5,
      }),
    ]);
    const result = thresholdSweep(rows, [0, 0.2, 1]);
    const key = rows[0].configKey;
    expect(result.get(key)).toEqual([
      { threshold: 0, passRate: 0 },
      { threshold: 0.2, passRate: 0.5 },
      { threshold: 1, passRate: 1 },
    ]);
  });
});

describe("makeGrid", () => {
  it("produces a float-safe inclusive grid", () => {
    expect(makeGrid(0, 0.2, 0.05)).toEqual([0, 0.05, 0.1, 0.15, 0.2]);
  });
});

describe("groupWeightedMeanDiff", () => {
  it("computes the weighted mean of mean_logit_difference per verifier/prover/temperature group", () => {
    const rows = deriveRows([
      mkRow({ id: "1", prover_model: "P1", verifier_model: "V1", temperature: 0.5, mean_logit_difference: 0.2 }),
      mkRow({ id: "2", prover_model: "P1", verifier_model: "V1", temperature: 0.5, mean_logit_difference: 0.4 }),
      mkRow({ id: "3", prover_model: "P2", verifier_model: "V2", temperature: 0.7, mean_logit_difference: 1.0 }),
      mkRow({ id: "4", prover_model: "P2", verifier_model: "V2", temperature: 0.7, mean_logit_difference: 2.0 }),
      mkRow({ id: "5", prover_model: "P2", verifier_model: "V2", temperature: 0.7, mean_logit_difference: 3.0 }),
    ]);
    const groups = groupWeightedMeanDiff(rows);
    expect(groups).toHaveLength(2);
    const g1 = groups.find((g) => g.verifier === "V1")!;
    expect(g1.prover).toBe("P1");
    expect(g1.temperature).toBe(0.5);
    expect(g1.count).toBe(2);
    expect(g1.value).toBeCloseTo(0.3); // mean(0.2, 0.4)

    const g2 = groups.find((g) => g.verifier === "V2")!;
    expect(g2.prover).toBe("P2");
    expect(g2.temperature).toBe(0.7);
    expect(g2.count).toBe(3);
    expect(g2.value).toBeCloseTo(2.0); // mean(1.0, 2.0, 3.0)
  });
});

describe("heatmapNorm", () => {
  it("uses log scale with power-of-ten padded bounds when all values are positive", () => {
    const result = heatmapNorm([0.01, 5]);
    expect(result.kind).toBe("log");
    expect(result.min).toBeCloseTo(0.01); // 10^floor(log10(0.01)) = 10^-2
    expect(result.max).toBeCloseTo(10); // 10^ceil(log10(5)) = 10^1
  });

  it("uses symlog scale when negative values are present", () => {
    const result = heatmapNorm([-1, 2]);
    expect(result.kind).toBe("symlog");
    // maxAbs = 2 -> 10^ceil(log10(2)) = 10^1 = 10, symmetric bounds
    expect(result.min).toBeCloseTo(-10);
    expect(result.max).toBeCloseTo(10);
  });
});

describe("normPosition", () => {
  it("interpolates log10 position between a log norm's min and max", () => {
    const norm = heatmapNorm([0.01, 5]); // -> { kind: "log", min: 0.01, max: 10 }
    expect(norm.kind).toBe("log");
    // (log10(1) - log10(0.01)) / (log10(10) - log10(0.01)) = (0 - -2) / (1 - -2) = 2/3
    expect(normPosition(norm, 1)).toBeCloseTo(2 / 3);
    expect(normPosition(norm, norm.min)).toBeCloseTo(0);
    expect(normPosition(norm, norm.max)).toBeCloseTo(1);
  });

  it("interpolates a symlog (linthresh 0.1) position between a symlog norm's min and max", () => {
    const norm = heatmapNorm([-1, 2]); // -> { kind: "symlog", min: -10, max: 10 }
    expect(norm.kind).toBe("symlog");
    // symlog(0) = 0 exactly halfway between symlog(-10) and symlog(10)
    expect(normPosition(norm, 0)).toBeCloseTo(0.5);
    expect(normPosition(norm, norm.min)).toBeCloseTo(0);
    expect(normPosition(norm, norm.max)).toBeCloseTo(1);
    // symlog(1) = 1 + log10(1/0.1) = 1 + 1 = 2; symlog(-10) = -(1+log10(100)) = -3; symlog(10) = 3
    // position = (2 - -3) / (3 - -3) = 5/6
    expect(normPosition(norm, 1)).toBeCloseTo(5 / 6);
  });

  it("clamps to [0, 1] for values outside the norm's range", () => {
    const norm = { kind: "log" as const, min: 1, max: 10 };
    expect(normPosition(norm, 0.1)).toBe(0);
    expect(normPosition(norm, 100)).toBe(1);
  });
});

describe("verificationMargins", () => {
  it("computes incorrect-minus-honest weighted mean margin when an honest baseline exists", () => {
    const rows = deriveRows([
      // honest baseline for verifier V at temp 0.5 / top_k 10 / top_p 0.9
      mkRow({
        id: "1",
        prover_model: "V",
        verifier_model: "V",
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        mean_logit_difference: 0.1,
      }),
      // cheater pair P -> V, same config
      mkRow({
        id: "2",
        prover_model: "P",
        verifier_model: "V",
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        mean_logit_difference: 0.6,
      }),
      // verifier W has only a cheater row, no honest W -> W baseline
      mkRow({
        id: "3",
        prover_model: "Q",
        verifier_model: "W",
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        mean_logit_difference: 0.9,
      }),
    ]);
    const margins = verificationMargins(rows);
    expect(margins).toHaveLength(1);
    expect(margins[0]).toMatchObject({
      verifier: "V",
      incorrectProver: "P",
      temperature: 0.5,
      top_k: 10,
      top_p: 0.9,
    });
    expect(margins[0].value).toBeCloseTo(0.5); // 0.6 - 0.1
    expect(margins.some((m) => m.verifier === "W")).toBe(false);
  });
});

describe("maxOf / minOf", () => {
  it("matches Math.max/Math.min for a normal array", () => {
    const values = [3, -7, 42, 0, 19, -19.5];
    expect(maxOf(values)).toBe(Math.max(...values));
    expect(minOf(values)).toBe(Math.min(...values));
  });

  it("does not throw a RangeError on arrays too large to spread as call args", () => {
    const huge = Array.from({ length: 200_000 }, (_, i) => i - 100_000);
    expect(maxOf(huge)).toBe(99_999);
    expect(minOf(huge)).toBe(-100_000);
  });
});

describe("color constants", () => {
  it("exposes the fixed PALETTE, OTHER_COLOR, PASS_COLOR, FAIL_COLOR", () => {
    expect(PALETTE).toEqual([
      "#2a78d6",
      "#eb6834",
      "#1baf7a",
      "#eda100",
      "#e87ba4",
      "#008300",
      "#4a3aa7",
      "#e34948",
    ]);
    expect(OTHER_COLOR).toBe("#5f5f5f");
    expect(PASS_COLOR).toBe("#4f7e6b");
    expect(FAIL_COLOR).toBe("#ca4f3e");
  });
});

const gpu = (over: Partial<GpuActivity> = {}): GpuActivity => ({
  window_s: 2.0,
  sample_count: 10,
  tensor_active_time_s: 1.2,
  sm_occupancy_mean: 0.4,
  pipe_activity_s: null,
  concurrent_events: 0,
  ...over,
});

const row = (over: Partial<AnalysisRow> = {}): AnalysisRow => ({
  id: "1",
  ts: "2026-07-30T00:00:00Z",
  result: "pass",
  mean_logit_difference: 0.1,
  exact_match_level_pct: 1,
  verification_threshold: 0.35,
  prover_model: "p",
  verifier_model: "p",
  temperature: 1,
  top_k: 200,
  top_p: 1,
  prompt_tokens: 100,
  output_tokens: 50,
  latency_ms: 500,
  prover_gpu: gpu(),
  verify_gpu: gpu({ tensor_active_time_s: 0.3 }),
  ...over,
});

describe("GPU derived fields", () => {
  it("computes busy-seconds per token for both sides", () => {
    const [d] = deriveRows([row()]);
    // Prover decodes 50 output tokens; verify prefills prompt+output = 150.
    expect(d.proverBusyPerToken).toBeCloseTo(1.2 / 50);
    expect(d.verifyBusyPerToken).toBeCloseTo(0.3 / 150);
    expect(d.efficiencyRatio).toBeCloseTo(1.2 / 50 / (0.3 / 150));
    expect(d.gpuEligible).toBe(true);
  });

  it("is null-safe when a side is missing", () => {
    const [d] = deriveRows([row({ verify_gpu: null })]);
    expect(d.verifyBusyPerToken).toBeNull();
    expect(d.efficiencyRatio).toBeNull();
    expect(d.gpuEligible).toBe(false);
  });

  it("gates on MIN_GPU_SAMPLES and concurrency", () => {
    const low = deriveRows([
      row({ verify_gpu: gpu({ sample_count: MIN_GPU_SAMPLES - 1 }) }),
    ])[0];
    expect(low.gpuEligible).toBe(false);
    const busy = deriveRows([
      row({ prover_gpu: gpu({ concurrent_events: 2 }) }),
    ])[0];
    expect(busy.gpuEligible).toBe(false);
  });

  it("gpuEligibleRows counts quality-gated exclusions only", () => {
    const rows = deriveRows([
      row(), // eligible
      row({ id: "2", prover_gpu: gpu({ concurrent_events: 1 }) }), // excluded
      row({ id: "3", prover_gpu: null }), // pre-enrichment: not counted
    ]);
    const { kept, excludedCount } = gpuEligibleRows(rows);
    expect(kept.map((r) => r.id)).toEqual(["1"]);
    expect(excludedCount).toBe(1);
  });
});

describe("shortModelName / shortPairLabel", () => {
  it("strips the namespace prefix", () => {
    expect(shortModelName("Qwen/Qwen2.5-1.5B-Instruct")).toBe("Qwen2.5-1.5B-Instruct");
    expect(shortModelName("openai/gpt-oss-20b")).toBe("gpt-oss-20b");
  });

  it("leaves un-namespaced names alone", () => {
    expect(shortModelName("gpt-oss-20b")).toBe("gpt-oss-20b");
  });

  it("shortens both sides of a pair label", () => {
    expect(shortPairLabel("Qwen/Qwen2.5-7B-Instruct → openai/gpt-oss-20b")).toBe(
      "Qwen2.5-7B-Instruct → gpt-oss-20b"
    );
  });
});
