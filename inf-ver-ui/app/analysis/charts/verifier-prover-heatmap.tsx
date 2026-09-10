"use client";

// Verifier x Prover mean-logit-difference heatmap, built from
// `groupWeightedMeanDiff`. Two variants share this component (registered
// twice in analysis-view.tsx):
//   - "faceted": one small-multiple panel per distinct temperature, all
//     panels sharing a single x=prover / y=verifier grid and ONE color
//     scale (heatmapNorm computed once across every panel's values).
//   - "split": a single matrix whose y rows are composite
//     "verifier · temperature" labels, so every (verifier, temperature)
//     combination gets its own row against the same x=prover columns.
// Cell fill is a sequential blue ramp positioned by `normPosition` against
// the shared `heatmapNorm`; each cell is annotated with its value (3-4 sig
// figs) in mono text, colored ink or cream depending on the cell's ramp
// position (dark cells get cream text).

import { useMemo, useRef } from "react";
import { DerivedRow, groupWeightedMeanDiff, heatmapNorm, normPosition, shortModelName } from "../lib";
import { ChartFrame, useChartTooltip, useHiddenKeys } from "../chart-frame";

const RAMP_STOPS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
const CELL_W = 88;
const CELL_H = 40;
const MARGIN_TOP = 10;
const MARGIN_RIGHT = 16;
const PANEL_GAP = 28;
// Axis labels are 11px mono — roughly this many px per character. Both
// margins are computed from the actual (shortened) label text so long model
// names neither clip on the left nor overlap along the bottom.
const LABEL_PX_PER_CHAR = 6.7;

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rampColor(t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const segments = RAMP_STOPS.length - 1;
  const pos = clamped * segments;
  const i = Math.min(segments - 1, Math.floor(pos));
  const frac = pos - i;
  const [r0, g0, b0] = hexToRgb(RAMP_STOPS[i]);
  const [r1, g1, b1] = hexToRgb(RAMP_STOPS[i + 1]);
  const r = Math.round(r0 + (r1 - r0) * frac);
  const g = Math.round(g0 + (g1 - g0) * frac);
  const b = Math.round(b0 + (b1 - b0) * frac);
  return `rgb(${r}, ${g}, ${b})`;
}

function fmtValue(v: number): string {
  // 3-4 significant figures, trailing zeros trimmed.
  if (v === 0) return "0";
  const digits = Math.abs(v) >= 1 ? 3 : 4;
  return String(parseFloat(v.toPrecision(digits)));
}

type Cell = {
  verifier: string;
  prover: string;
  temperature: number | null;
  value: number;
  count: number;
};

function Grid({
  provers,
  yLabels,
  yLabelFor,
  cellFor,
  norm,
  showTooltip,
  hideTooltip,
}: {
  provers: string[];
  yLabels: string[];
  yLabelFor?: (yLabel: string) => string;
  cellFor: (yLabel: string, prover: string) => Cell | undefined;
  norm: { kind: "log" | "symlog"; min: number; max: number };
  showTooltip: ReturnType<typeof useChartTooltip>["showTooltip"];
  hideTooltip: () => void;
}) {
  const xTexts = provers.map(shortModelName);
  const yTexts = yLabels.map((l) => (yLabelFor ? yLabelFor(l) : l));
  const marginLeft =
    Math.ceil(Math.max(0, ...yTexts.map((t) => t.length)) * LABEL_PX_PER_CHAR) + 14;
  const maxXWidth = Math.max(0, ...xTexts.map((t) => t.length)) * LABEL_PX_PER_CHAR;
  const angleX = maxXWidth > CELL_W - 8;
  const marginBottom = angleX ? Math.ceil(maxXWidth * 0.5) + 28 : 40;

  const innerWidth = provers.length * CELL_W;
  const innerHeight = yLabels.length * CELL_H;
  const width = innerWidth + marginLeft + MARGIN_RIGHT;
  const height = innerHeight + MARGIN_TOP + marginBottom;

  return (
    <svg width={width} height={height} role="img" aria-label="Verifier x prover heatmap">
      <g transform={`translate(${marginLeft}, ${MARGIN_TOP})`}>
        {yLabels.map((yLabel, row) =>
          provers.map((prover, col) => {
            const cell = cellFor(yLabel, prover);
            const x = col * CELL_W;
            const y = row * CELL_H;
            if (!cell) {
              return (
                <rect
                  key={`${yLabel}-${prover}`}
                  x={x}
                  y={y}
                  width={CELL_W}
                  height={CELL_H}
                  fill="none"
                  stroke="var(--para)"
                  strokeOpacity={0.15}
                />
              );
            }
            const pos = normPosition(norm, cell.value);
            const fill = rampColor(pos);
            const textColor = pos > 0.55 ? "var(--cream)" : "var(--ink)";
            return (
              <g key={`${yLabel}-${prover}`}>
                <rect
                  x={x}
                  y={y}
                  width={CELL_W}
                  height={CELL_H}
                  fill={fill}
                  stroke="var(--cream)"
                  strokeWidth={1}
                  onMouseMove={(evt) =>
                    showTooltip(
                      evt,
                      <span>
                        {cell.prover} → {cell.verifier}
                        <br />
                        temp: {cell.temperature ?? "—"}
                        <br />
                        value: {cell.value.toFixed(4)}
                        <br />
                        count: {cell.count}
                      </span>
                    )
                  }
                  onMouseLeave={hideTooltip}
                />
                <text
                  x={x + CELL_W / 2}
                  y={y + CELL_H / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: textColor, pointerEvents: "none" }}
                >
                  {fmtValue(cell.value)}
                </text>
              </g>
            );
          })
        )}

        {yLabels.map((yLabel, row) => (
          <text
            key={`y-${yLabel}`}
            x={-8}
            y={row * CELL_H + CELL_H / 2}
            textAnchor="end"
            dominantBaseline="middle"
            style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
          >
            {yTexts[row]}
          </text>
        ))}

        {provers.map((prover, col) => {
          const x = col * CELL_W + CELL_W / 2;
          const y = innerHeight + 16;
          return (
            <text
              key={`x-${prover}`}
              x={x}
              y={y}
              textAnchor={angleX ? "end" : "middle"}
              transform={angleX ? `rotate(-30, ${x}, ${y})` : undefined}
              style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
            >
              {xTexts[col]}
            </text>
          );
        })}
      </g>
    </svg>
  );
}

export default function VerifierProverHeatmap({
  rows,
  variant,
}: {
  rows: DerivedRow[];
  variant: "faceted" | "split";
}) {
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const exportRef = useRef<HTMLDivElement | null>(null);

  const title =
    variant === "faceted"
      ? "Verifier × Prover Mean Logit Difference"
      : "Verifier × Prover, Temperature Split";
  const slug = variant === "faceted" ? "heatmap-faceted" : "heatmap-split";

  const cells = useMemo(() => groupWeightedMeanDiff(rows), [rows]);

  const provers = useMemo(
    () => Array.from(new Set(cells.map((c) => c.prover))).sort(),
    [cells]
  );
  const verifiers = useMemo(
    () => Array.from(new Set(cells.map((c) => c.verifier))).sort(),
    [cells]
  );
  const temperatures = useMemo(
    () =>
      Array.from(new Set(cells.map((c) => c.temperature))).sort((a, b) => {
        if (a === null) return -1;
        if (b === null) return 1;
        return a - b;
      }),
    [cells]
  );

  const norm = useMemo(() => heatmapNorm(cells.map((c) => c.value)), [cells]);

  const note =
    "Cell color = mean logit difference; " +
    (norm.kind === "log" ? "log scale." : "symlog scale.");

  if (cells.length === 0) {
    return (
      <ChartFrame title={title} slug={slug} legend={[]} hiddenKeys={new Set()} onToggleKey={() => {}} exportRef={exportRef}>
        <p className="muted">No data for the current filters.</p>
      </ChartFrame>
    );
  }

  return (
    <ChartFrame
      title={title}
      slug={slug}
      legend={[]}
      hiddenKeys={new Set()}
      onToggleKey={() => {}}
      note={note}
      exportRef={exportRef}
    >
      <div ref={containerRef} className="chart-tooltip-container">
        <div style={{ overflowX: "auto" }}>
          {variant === "faceted" ? (
            <div style={{ display: "flex", flexDirection: "column", gap: PANEL_GAP }}>
              {temperatures.map((temp) => {
                const panelCells = cells.filter((c) => c.temperature === temp);
                return (
                  <div key={String(temp)}>
                    <p
                      className="control-label"
                      style={{ marginBottom: 4 }}
                    >{`temperature = ${temp ?? "—"}`}</p>
                    <Grid
                      provers={provers}
                      yLabels={verifiers}
                      yLabelFor={shortModelName}
                      cellFor={(verifier, prover) =>
                        panelCells.find((c) => c.verifier === verifier && c.prover === prover)
                      }
                      norm={norm}
                      showTooltip={showTooltip}
                      hideTooltip={hideTooltip}
                    />
                  </div>
                );
              })}
            </div>
          ) : (
            (() => {
              const rowKeys = Array.from(
                new Map(
                  cells.map((c) => [`${c.verifier} · ${c.temperature ?? "—"}`, c])
                ).values()
              ).sort((a, b) => {
                const byVerifier = a.verifier.localeCompare(b.verifier);
                if (byVerifier !== 0) return byVerifier;
                if (a.temperature === null) return -1;
                if (b.temperature === null) return 1;
                return a.temperature - b.temperature;
              });
              const yLabels = rowKeys.map((c) => `${c.verifier} · ${c.temperature ?? "—"}`);
              const byLabel = (label: string): { verifier: string; temperature: number | null } => {
                const c = cells.find((x) => `${x.verifier} · ${x.temperature ?? "—"}` === label)!;
                return { verifier: c.verifier, temperature: c.temperature };
              };
              return (
                <Grid
                  provers={provers}
                  yLabels={yLabels}
                  yLabelFor={(label) => {
                    const { verifier, temperature } = byLabel(label);
                    return `${shortModelName(verifier)} · ${temperature ?? "—"}`;
                  }}
                  cellFor={(label, prover) => {
                    const { verifier, temperature } = byLabel(label);
                    return cells.find(
                      (c) => c.verifier === verifier && c.temperature === temperature && c.prover === prover
                    );
                  }}
                  norm={norm}
                  showTooltip={showTooltip}
                  hideTooltip={hideTooltip}
                />
              );
            })()
          )}
        </div>
        {tooltipElement}
      </div>
    </ChartFrame>
  );
}
