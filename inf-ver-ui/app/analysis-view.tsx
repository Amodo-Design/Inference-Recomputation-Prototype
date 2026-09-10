"use client";

import { useCallback, useMemo, useState } from "react";
import { AnalysisRow, deriveRows, pairColorMap } from "./analysis/lib";
import LogitVsMatchLevel from "./analysis/charts/logit-vs-match-level";
import LatencyVsLogit from "./analysis/charts/latency-vs-logit";
import VerificationMargin from "./analysis/charts/verification-margin";
import DiffByTokenLength from "./analysis/charts/diff-by-token-length";
import LatencyByOutputLength from "./analysis/charts/latency-by-output-length";
import PassFailByOutputTokens from "./analysis/charts/pass-fail-by-output-tokens";
import ThresholdSensitivity from "./analysis/charts/threshold-sensitivity";
import VerifierProverHeatmap from "./analysis/charts/verifier-prover-heatmap";
import DiffByParam from "./analysis/charts/diff-by-param";
import GpuBusyPerToken from "./analysis/charts/gpu-busy-per-token";
import GpuEfficiencyVsOutput from "./analysis/charts/gpu-efficiency-vs-output";
import GpuOccupancyBySide from "./analysis/charts/gpu-occupancy-by-side";
import GpuPrecisionSplit from "./analysis/charts/gpu-precision-split";

// Chart registration point: chart tasks (A5-A8) replace each placeholder
// body in place, keyed by slug. Order here is the render order.
const CHART_CARDS: { slug: string; title: string }[] = [
  { slug: "logit-vs-match-level", title: "Mean Logit Difference vs Match Level" },
  { slug: "latency-vs-logit", title: "Mean Logit Difference vs Verification Latency" },
  { slug: "verification-margin", title: "Verification Logit Difference Gap vs Top-k" },
  { slug: "diff-by-prompt-length", title: "Mean Logit Difference vs Prompt Length" },
  { slug: "diff-by-output-length", title: "Mean Logit Difference vs Output Length" },
  { slug: "latency-by-output-length", title: "Verification Latency vs Output Length" },
  { slug: "pass-fail-by-output-tokens", title: "Output Token Pass/Fail Rate" },
  { slug: "threshold-sensitivity-runs", title: "Threshold Sensitivity by Temperature" },
  { slug: "threshold-sensitivity-band", title: "Threshold Sensitivity, Std Band" },
  { slug: "heatmap-faceted", title: "Verifier × Prover Mean Logit Difference" },
  { slug: "heatmap-split", title: "Verifier × Prover, Temperature Split" },
  { slug: "diff-by-temperature", title: "Mean Logit Difference vs Temperature" },
  { slug: "diff-by-top-k", title: "Mean Logit Difference vs Top-k" },
  { slug: "diff-by-top-p", title: "Mean Logit Difference vs Top-p" },
  { slug: "gpu-busy-per-token", title: "GPU Busy-Seconds per Token: Prover vs Verifier" },
  { slug: "gpu-efficiency-vs-output", title: "GPU Efficiency Ratio vs Output Length" },
  { slug: "gpu-occupancy-by-side", title: "SM Occupancy by Side" },
  { slug: "gpu-precision-split", title: "Tensor-Core Precision Split by Side" }
];

export type AnalysisFilters = {
  result: string;
  prover: string;
  verifier: string;
  q: string;
  from: string;
  to: string;
};

export default function AnalysisView({
  rows,
  total,
  truncated,
  modelNames,
  filters
}: {
  rows: AnalysisRow[];
  total: number;
  truncated: boolean;
  modelNames: string[];
  filters: AnalysisFilters;
}) {
  // Computed once per rows change; chart tasks read derivedRows/colorByPair
  // instead of recomputing per-card. total/truncated/modelNames/filters are
  // accepted now so the signature is locked for chart tasks that need them
  // (e.g. shared legends, axis domains) without another prop-drilling pass.
  const derivedRows = useMemo(() => deriveRows(rows), [rows]);
  const colorByPair = useMemo(() => pairColorMap(derivedRows), [derivedRows]);

  // Charts render collapsed by default (large visx trees are expensive to
  // mount and most are off-screen on load); this Set tracks which slugs the
  // user has opted to expand. Not mounting the chart component at all while
  // collapsed is the actual render-time saving — CSS-hiding it would still
  // pay the mount cost.
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  const toggleCard = useCallback((slug: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) {
        next.delete(slug);
      } else {
        next.add(slug);
      }
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpanded(new Set(CHART_CARDS.map((card) => card.slug)));
  }, []);

  const collapseAll = useCallback(() => {
    setExpanded(new Set());
  }, []);

  const renderChart = (slug: string) => {
    switch (slug) {
      case "logit-vs-match-level":
        return <LogitVsMatchLevel rows={derivedRows} colors={colorByPair} />;
      case "latency-vs-logit":
        return <LatencyVsLogit rows={derivedRows} colors={colorByPair} />;
      case "verification-margin":
        return <VerificationMargin rows={derivedRows} colors={colorByPair} />;
      case "diff-by-prompt-length":
        return <DiffByTokenLength rows={derivedRows} colors={colorByPair} axis="prompt" />;
      case "diff-by-output-length":
        return <DiffByTokenLength rows={derivedRows} colors={colorByPair} axis="output" />;
      case "latency-by-output-length":
        return <LatencyByOutputLength rows={derivedRows} colors={colorByPair} />;
      case "pass-fail-by-output-tokens":
        return <PassFailByOutputTokens rows={derivedRows} colors={colorByPair} />;
      case "threshold-sensitivity-runs":
        return <ThresholdSensitivity rows={derivedRows} colors={colorByPair} variant="runs" />;
      case "threshold-sensitivity-band":
        return <ThresholdSensitivity rows={derivedRows} colors={colorByPair} variant="band" />;
      case "heatmap-faceted":
        return <VerifierProverHeatmap rows={derivedRows} variant="faceted" />;
      case "heatmap-split":
        return <VerifierProverHeatmap rows={derivedRows} variant="split" />;
      case "diff-by-temperature":
        return <DiffByParam rows={derivedRows} colors={colorByPair} param="temperature" />;
      case "diff-by-top-k":
        return <DiffByParam rows={derivedRows} colors={colorByPair} param="top_k" />;
      case "diff-by-top-p":
        return <DiffByParam rows={derivedRows} colors={colorByPair} param="top_p" />;
      case "gpu-busy-per-token":
        return <GpuBusyPerToken rows={derivedRows} />;
      case "gpu-efficiency-vs-output":
        return <GpuEfficiencyVsOutput rows={derivedRows} colors={colorByPair} />;
      case "gpu-occupancy-by-side":
        return <GpuOccupancyBySide rows={derivedRows} />;
      case "gpu-precision-split":
        return <GpuPrecisionSplit rows={derivedRows} />;
      default:
        return <p className="muted">Chart coming in a later task.</p>;
    }
  };

  return (
    <div>
      <div className="analysis-charts-toolbar">
        <button type="button" className="delete-button action-button-neutral" onClick={expandAll}>
          Expand all
        </button>
        <button type="button" className="delete-button action-button-neutral" onClick={collapseAll}>
          Collapse all
        </button>
      </div>
      <div className="analysis-grid">
        {CHART_CARDS.map((card) => {
          const isOpen = expanded.has(card.slug);
          const panelId = `chart-panel-${card.slug}`;
          return (
            <section key={card.slug} className="chart-card" aria-label={card.title} data-chart-slug={card.slug}>
              <button
                type="button"
                className={isOpen ? "chart-card-toggle chart-card-toggle-open" : "chart-card-toggle"}
                aria-expanded={isOpen}
                aria-controls={panelId}
                onClick={() => toggleCard(card.slug)}
              >
                {/* When open, the mounted chart's own ChartFrame renders this
                    same title in its header, so the toggle bar drops the
                    title text here to avoid showing it twice — it becomes a
                    slim "Hide chart" strip instead. */}
                {isOpen ? null : <span className="chart-card-toggle-title">{card.title}</span>}
                <span className="chart-card-toggle-affordance">
                  {isOpen ? "Hide chart" : "Show chart"}
                  <span className="chart-card-toggle-chevron" aria-hidden="true">
                    {isOpen ? "▾" : "▸"}
                  </span>
                </span>
              </button>
              {isOpen ? (
                <div id={panelId} className="chart-card-panel">
                  {renderChart(card.slug)}
                </div>
              ) : null}
            </section>
          );
        })}
      </div>
    </div>
  );
}

// Re-exported so this module documents the shared shape without every
// consumer having to know it lives in ./analysis/lib.
export type { AnalysisRow };
