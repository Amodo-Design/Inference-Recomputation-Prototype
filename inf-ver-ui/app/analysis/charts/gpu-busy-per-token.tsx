"use client";

// Overlaid histograms (as fraction-of-events, not raw counts, so the two
// series are comparable regardless of how many prover vs verifier events
// passed the GPU-eligibility gate) of log10(tensor-busy seconds / token) for
// the prover (decode, per generated token) and verifier (prefill, per
// prompt+output token) sides of the same events. This is the chart the rest
// of the GPU-efficiency work exists to support: does the prover spend more
// or less GPU time per token than the verifier that checks it?
//
// Both histograms are built from the SAME paired subset of events (both the
// prover and verify busy-per-token values positive) so the two series
// describe the same underlying events rather than independently-filtered,
// potentially mismatched ones.

import { useMemo, useRef } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { DerivedRow, PALETTE, gpuEligibleRows, maxOf, minOf } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";

const HEIGHT = 380;
const MARGIN = { top: 20, right: 24, bottom: 44, left: 68 };
const BIN_COUNT = 24;

const PROVER_COLOR = PALETTE[0];
const VERIFY_COLOR = PALETTE[1];

type SeriesKey = "prover" | "verify";
type Series = { key: SeriesKey; label: string; color: string; values: number[] };

type Bin = { x0: number; x1: number; xMid: number; counts: Record<SeriesKey, number> };

// The histogram is computed in log10 space (bins uniform in log), and the x
// axis renders as a true log scale: tick marks at the 1-2-5 positions of
// each decade, labelled in physical units (µs, ms) rather than raw log10
// exponents. When the data spans many decades, only the decade (×1) ticks
// are kept so labels don't crowd.
function logAxisTicks(lo: number, hi: number): number[] {
  const mantissas = hi - lo > 3 ? [1] : [1, 2, 5];
  const ticks: number[] = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
    for (const m of mantissas) {
      const v = e + Math.log10(m);
      if (v >= lo - 1e-9 && v <= hi + 1e-9) ticks.push(v);
    }
  }
  return ticks;
}

const TIME_UNITS = [
  { limit: 1, div: 1, suffix: "s" },
  { limit: 1e-3, div: 1e-3, suffix: "ms" },
  { limit: 1e-6, div: 1e-6, suffix: "µs" },
  { limit: 0, div: 1e-9, suffix: "ns" },
];

function fmtSeconds(s: number): string {
  const unit = TIME_UNITS.find((u) => s >= u.limit) ?? TIME_UNITS[TIME_UNITS.length - 1];
  return `${parseFloat((s / unit.div).toPrecision(3))} ${unit.suffix}`;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return NaN;
  const mid = Math.floor(n / 2);
  return n % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function buildBins(series: Series[], binCount: number): { bins: Bin[]; domain: [number, number] } {
  const allLogs = series.flatMap((s) => s.values);
  if (allLogs.length === 0) return { bins: [], domain: [0, 1] };
  const min = minOf(allLogs);
  const max = maxOf(allLogs);
  const span = max - min;
  if (!(span > 0)) {
    // Degenerate (single distinct value) domain: pad it out so a single bin
    // still has non-zero width to render.
    const bins: Bin[] = [
      {
        x0: min - 0.5,
        x1: max + 0.5,
        xMid: min,
        counts: { prover: 0, verify: 0 },
      },
    ];
    for (const s of series) {
      for (const v of s.values) {
        if (v >= bins[0].x0 && v <= bins[0].x1) bins[0].counts[s.key] += 1;
      }
    }
    return { bins, domain: [bins[0].x0, bins[0].x1] };
  }

  const width = span / binCount;
  const bins: Bin[] = Array.from({ length: binCount }, (_, i) => {
    const x0 = min + i * width;
    const x1 = min + (i + 1) * width;
    return { x0, x1, xMid: (x0 + x1) / 2, counts: { prover: 0, verify: 0 } };
  });

  for (const s of series) {
    for (const v of s.values) {
      let idx = Math.floor((v - min) / width);
      if (idx < 0) idx = 0;
      if (idx > binCount - 1) idx = binCount - 1;
      bins[idx].counts[s.key] += 1;
    }
  }

  return { bins, domain: [min, max] };
}

export default function GpuBusyPerToken({ rows }: { rows: DerivedRow[] }) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const exportRef = useRef<HTMLDivElement | null>(null);

  const { kept, excludedCount } = useMemo(() => gpuEligibleRows(rows), [rows]);

  // gpuEligibleRows only guarantees enough GPU samples were seen; a busy
  // value can still be null (missing token counts) or non-positive (zero
  // tensor-integral). Both histograms must be drawn from the SAME subset of
  // events — filtering each series independently would let a row with a
  // zero-valued integral on one side feed only the other side's histogram,
  // so the two series would silently describe different event sets.
  const usable = useMemo(
    () =>
      kept.filter(
        (r) =>
          r.proverBusyPerToken !== null &&
          r.proverBusyPerToken > 0 &&
          r.verifyBusyPerToken !== null &&
          r.verifyBusyPerToken > 0
      ),
    [kept]
  );

  const series = useMemo<Series[]>(() => {
    const proverValues = usable.map((r) => Math.log10(r.proverBusyPerToken as number));
    const verifyValues = usable.map((r) => Math.log10(r.verifyBusyPerToken as number));
    return [
      { key: "prover", label: "prover", color: PROVER_COLOR, values: proverValues },
      { key: "verify", label: "verify", color: VERIFY_COLOR, values: verifyValues },
    ];
  }, [usable]);

  const { bins, domain } = useMemo(() => buildBins(series, BIN_COUNT), [series]);

  const totals: Record<SeriesKey, number> = {
    prover: series[0]?.values.length ?? 0,
    verify: series[1]?.values.length ?? 0,
  };

  const maxFraction = useMemo(() => {
    let max = 0;
    for (const bin of bins) {
      for (const s of series) {
        if (hiddenKeys.has(s.key)) continue;
        const total = totals[s.key];
        if (total === 0) continue;
        const fraction = bin.counts[s.key] / total;
        if (fraction > max) max = fraction;
      }
    }
    return max;
  }, [bins, series, totals, hiddenKeys]);

  const legend: LegendItem[] = series.map((s) => ({ key: s.key, label: s.label, color: s.color }));

  // Median of each side's log-values, for the in-plot ratio annotation: on a
  // log axis the horizontal distance between the two medians IS the ratio,
  // so a connector line at each side's median makes the chart's point —
  // "how much more GPU time per token does the prover spend?" — visually
  // explicit.
  const medianLogs: Record<SeriesKey, number> = {
    prover: median(series[0]?.values ?? []),
    verify: median(series[1]?.values ?? []),
  };

  // The ratio label itself is computed directly from each paired event's
  // efficiencyRatio (prover/verify), rather than from the ratio of the two
  // series' medians, so it reflects the median of the actual per-event
  // ratios. `usable` already guarantees both sides are positive, so every
  // efficiencyRatio here is finite — the isFinite filter is belt-and-braces.
  const efficiencyRatios = usable
    .map((r) => r.efficiencyRatio)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const medianRatio = efficiencyRatios.length > 0 ? median(efficiencyRatios) : null;

  const excludedNote = `${excludedCount} event${excludedCount === 1 ? "" : "s"} excluded (fewer than 3 GPU samples or concurrent activity).`;
  const omittedCount = kept.length - usable.length;
  const omittedSuffix =
    omittedCount > 0
      ? ` ${omittedCount} eligible event${omittedCount === 1 ? "" : "s"} omitted (zero-valued busy integral on one side).`
      : "";
  const ratioSuffix =
    medianRatio !== null
      ? ` Prover spends ${medianRatio.toFixed(1)}× the verifier's GPU time per token (ratio of medians).`
      : "";
  const note = `${excludedNote}${omittedSuffix}${ratioSuffix}`;

  return (
    <ChartFrame
      title="GPU Busy-Seconds per Token: Prover vs Verifier"
      slug="gpu-busy-per-token"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      note={note}
      exportRef={exportRef}
    >
      {kept.length === 0 ? (
        <p className="muted">No enriched GPU data for the current filters.</p>
      ) : (
        <div ref={containerRef} className="chart-tooltip-container">
          <div style={{ width: "100%", height: HEIGHT }}>
            <ParentSize>
              {({ width }) => {
                const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
                const innerHeight = Math.max(0, HEIGHT - MARGIN.top - MARGIN.bottom);

                // Pad the domain so edge bins render inside the plot instead
                // of flush against (and visually clipped by) the axes.
                const domainPad = (domain[1] - domain[0]) * 0.04 || 0.5;
                const xScale = scaleLinear<number>({
                  domain: [domain[0] - domainPad, domain[1] + domainPad],
                  range: [0, innerWidth],
                });
                const yScale = scaleLinear<number>({
                  domain: [0, maxFraction * 1.08 || 1],
                  range: [innerHeight, 0],
                });

                return (
                  <svg
                    width={width}
                    height={HEIGHT}
                    role="img"
                    aria-label="GPU busy-seconds per token: prover vs verifier"
                  >
                    <Group left={MARGIN.left} top={MARGIN.top}>
                      <AxisLeft
                        scale={yScale}
                        numTicks={5}
                        stroke="var(--para)"
                        tickStroke="var(--para)"
                        axisLineClassName="axis-line-faded"
                        tickClassName="axis-tick-faded"
                        tickLabelProps={LEFT_AXIS_TICK_PROPS}
                      />
                      <AxisBottom
                        top={innerHeight}
                        scale={xScale}
                        tickValues={logAxisTicks(domain[0] - domainPad, domain[1] + domainPad)}
                        tickFormat={(v) => fmtSeconds(Math.pow(10, Number(v)))}
                        stroke="var(--para)"
                        tickStroke="var(--para)"
                        axisLineClassName="axis-line-faded"
                        tickClassName="axis-tick-faded"
                        tickLabelProps={BOTTOM_AXIS_TICK_PROPS}
                      />
                      <text
                        x={innerWidth / 2}
                        y={innerHeight + 36}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Tensor-busy time per token (log scale)
                      </text>
                      <text
                        transform={`translate(${-56}, ${innerHeight / 2}) rotate(-90)`}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Fraction of events
                      </text>

                      {series.map((s) => {
                        if (hiddenKeys.has(s.key)) return null;
                        const total = totals[s.key];
                        return (
                          <Group key={s.key}>
                            {bins.map((bin, bi) => {
                              const count = bin.counts[s.key];
                              if (count === 0 || total === 0) return null;
                              const fraction = count / total;
                              const x0 = xScale(bin.x0);
                              const x1 = xScale(bin.x1);
                              const y = yScale(fraction);
                              const yBase = yScale(0);
                              if (!Number.isFinite(x0) || !Number.isFinite(x1) || !Number.isFinite(y)) {
                                return null;
                              }
                              return (
                                <rect
                                  key={bi}
                                  x={Math.min(x0, x1)}
                                  y={y}
                                  width={Math.max(0, Math.abs(x1 - x0))}
                                  height={Math.max(0, yBase - y)}
                                  fill={s.color}
                                  fillOpacity={0.45}
                                  onMouseMove={(evt) =>
                                    showTooltip(
                                      evt,
                                      <span>
                                        {s.label}: {count} events ≈{" "}
                                        {fmtSeconds(Math.pow(10, bin.xMid))}/token
                                      </span>
                                    )
                                  }
                                  onMouseLeave={hideTooltip}
                                />
                              );
                            })}
                          </Group>
                        );
                      })}

                      {/* Ratio annotation: dashed line at each side's median,
                          joined by a connector whose length — on this log
                          axis — is the prover/verify ratio. */}
                      {medianRatio !== null &&
                        !hiddenKeys.has("prover") &&
                        !hiddenKeys.has("verify") &&
                        (() => {
                          const xProver = xScale(medianLogs.prover);
                          const xVerify = xScale(medianLogs.verify);
                          if (!Number.isFinite(xProver) || !Number.isFinite(xVerify)) return null;
                          const connectorY = 12;
                          const xMid = (xProver + xVerify) / 2;
                          return (
                            <Group>
                              {(
                                [
                                  { key: "prover" as const, x: xProver },
                                  { key: "verify" as const, x: xVerify },
                                ]
                              ).map(({ key, x }, i) => (
                                <line
                                  key={key}
                                  x1={x}
                                  x2={x}
                                  y1={connectorY}
                                  y2={innerHeight}
                                  stroke={series[i === 0 ? 0 : 1].color}
                                  strokeWidth={1.5}
                                  strokeDasharray="5,3"
                                  strokeOpacity={0.9}
                                />
                              ))}
                              <line
                                x1={xVerify}
                                x2={xProver}
                                y1={connectorY}
                                y2={connectorY}
                                style={{ stroke: "var(--ink)" }}
                                strokeWidth={1}
                              />
                              <text
                                x={xMid}
                                y={connectorY - 5}
                                textAnchor="middle"
                                style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--ink)" }}
                              >
                                {`prover ×${medianRatio.toFixed(1)} verify`}
                              </text>
                            </Group>
                          );
                        })()}
                    </Group>
                  </svg>
                );
              }}
            </ParentSize>
          </div>
          {tooltipElement}
        </div>
      )}
    </ChartFrame>
  );
}
