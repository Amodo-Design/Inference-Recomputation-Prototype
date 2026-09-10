"use client";

// Faceted stacked-area histogram: pass/fail rate vs output token length, one
// small-multiple panel per prover→verifier pair (honest panels first, each
// annotated honest/cheater in its title). Bins are FIXED-WIDTH (not
// quantile) so they span the same observed min–max range and line up across
// every panel. Pass rate stacks under fail rate, summing to 100%; bins with
// no rows are gaps — the stack is not interpolated across them. The
// pass/fail legend is fixed (not toggleable); the per-pair legend toggles
// which panels render.

import { useMemo, useRef, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { Area } from "@visx/shape";
import { DerivedRow, FAIL_COLOR, PASS_COLOR, maxOf, minOf, shortPairLabel } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";
import { NumberControl } from "../controls";

const PANEL_HEIGHT = 172;
const MARGIN = { top: 30, right: 24, bottom: 30, left: 60 };
const DEFAULT_BIN_COUNT = 50;
const MIN_BIN_COUNT = 2;
const MAX_BIN_COUNT = 200;
const Y_TICKS = [0, 25, 50, 75, 100];

type Row = { x: number; pair: string; pass: boolean };

type Bin = {
  index: number;
  x0: number;
  x1: number;
  xMid: number;
  passRate: number;
  failRate: number;
  passCount: number;
  failCount: number;
  total: number;
};

// Fixed-width bins spanning [min, max], shared across every panel — distinct
// from `quantileBins` in lib.ts, which produces variable-width quantile
// bins. Bins with zero rows are omitted (callers treat gaps as breaks, not
// interpolated segments).
function fixedWidthBins(rows: Row[], min: number, max: number, binCount: number): Bin[] {
  const span = max - min;
  if (!(span > 0) || binCount < 1) return [];
  const width = span / binCount;

  const counts = Array.from({ length: binCount }, () => ({ pass: 0, fail: 0 }));
  for (const r of rows) {
    let idx = Math.floor((r.x - min) / width);
    if (idx < 0) idx = 0;
    if (idx > binCount - 1) idx = binCount - 1;
    if (r.pass) counts[idx].pass += 1;
    else counts[idx].fail += 1;
  }

  const bins: Bin[] = [];
  for (let i = 0; i < binCount; i++) {
    const { pass, fail } = counts[i];
    const total = pass + fail;
    if (total === 0) continue;
    const x0 = min + i * width;
    const x1 = min + (i + 1) * width;
    const passRate = (pass / total) * 100;
    bins.push({
      index: i,
      x0,
      x1,
      xMid: (x0 + x1) / 2,
      passRate,
      failRate: 100 - passRate,
      passCount: pass,
      failCount: fail,
      total,
    });
  }
  return bins;
}

// Group bins into contiguous runs (by bin index) so the stacked area is only
// drawn across runs of adjacent, non-empty bins — an empty bin is a gap,
// never bridged.
function contiguousRuns(bins: Bin[]): Bin[][] {
  const runs: Bin[][] = [];
  let current: Bin[] = [];
  for (const bin of bins) {
    if (current.length > 0 && bin.index !== current[current.length - 1].index + 1) {
      runs.push(current);
      current = [];
    }
    current.push(bin);
  }
  if (current.length > 0) runs.push(current);
  return runs;
}

export default function PassFailByOutputTokens({
  rows,
  colors,
}: {
  rows: DerivedRow[];
  colors: Map<string, string>;
}) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const [binCount, setBinCount] = useState(DEFAULT_BIN_COUNT);
  const exportRef = useRef<HTMLDivElement | null>(null);

  const allRows = useMemo<Row[]>(
    () =>
      rows
        .filter((r) => r.output_tokens !== null && r.thresholdPass !== null)
        .map((r) => ({
          x: r.output_tokens as number,
          pair: r.pair,
          pass: r.thresholdPass as boolean,
        })),
    [rows]
  );

  const uniqueXCount = new Set(allRows.map((r) => r.x)).size;
  const eligible = allRows.length > 0 && uniqueXCount > 1;

  const clampedBinCount = Math.min(MAX_BIN_COUNT, Math.max(MIN_BIN_COUNT, Math.round(binCount)));

  const minX = eligible ? minOf(allRows.map((r) => r.x)) : 0;
  const maxX = eligible ? maxOf(allRows.map((r) => r.x)) : 0;

  const honestPairs = useMemo(() => {
    const honestSet = new Set(rows.filter((r) => r.honest).map((r) => r.pair));
    return honestSet;
  }, [rows]);

  const pairs = useMemo(() => {
    const unique = Array.from(new Set(allRows.map((r) => r.pair)));
    unique.sort((a, b) => {
      const aHonest = honestPairs.has(a);
      const bHonest = honestPairs.has(b);
      if (aHonest !== bHonest) return aHonest ? -1 : 1;
      return a.localeCompare(b);
    });
    return unique;
  }, [allRows, honestPairs]);

  const legend: LegendItem[] = pairs.map((pair) => ({
    key: pair,
    label: pair,
    color: colors.get(pair) ?? "#5f5f5f",
  }));

  const binsByPair = useMemo(() => {
    const map = new Map<string, Bin[]>();
    if (!eligible) return map;
    for (const pair of pairs) {
      const pairRows = allRows.filter((r) => r.pair === pair);
      map.set(pair, fixedWidthBins(pairRows, minX, maxX, clampedBinCount));
    }
    return map;
  }, [allRows, pairs, minX, maxX, clampedBinCount, eligible]);

  const note = eligible
    ? `${clampedBinCount} fixed-width bins spanning ${minX}–${maxX} output tokens, shared across panels. Empty bins are gaps.`
    : null;

  return (
    <ChartFrame
      title="Output Token Pass/Fail Rate"
      slug="pass-fail-by-output-tokens"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      controls={
        <NumberControl
          label="Bins"
          value={binCount}
          onChange={setBinCount}
          min={MIN_BIN_COUNT}
          max={MAX_BIN_COUNT}
          step={1}
        />
      }
      note={note}
      exportRef={exportRef}
    >
      {!eligible ? (
        <p className="muted">
          Needs more than one distinct output-token value and rows with a known pass/fail
          result (mean logit difference and verification threshold both present) in the
          filtered data.
        </p>
      ) : (
        <>
          <div className="chart-legend" aria-hidden={false} style={{ marginBottom: 8 }}>
            <span className="chart-legend-item" style={{ cursor: "default" }}>
              <span className="chart-legend-swatch" style={{ background: PASS_COLOR }} />
              <span className="chart-legend-label">pass</span>
            </span>
            <span className="chart-legend-item" style={{ cursor: "default" }}>
              <span className="chart-legend-swatch" style={{ background: FAIL_COLOR }} />
              <span className="chart-legend-label">fail</span>
            </span>
          </div>
          <div ref={containerRef} className="chart-tooltip-container">
            <div>
              {pairs.map((pair) => {
                if (hiddenKeys.has(pair)) return null;
                const bins = binsByPair.get(pair) ?? [];
                const runs = contiguousRuns(bins);
                const kind = honestPairs.has(pair) ? "honest" : "cheater";
                return (
                  <div key={pair} style={{ width: "100%", height: PANEL_HEIGHT }}>
                    <ParentSize>
                      {({ width }) => {
                        const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
                        const innerHeight = Math.max(0, PANEL_HEIGHT - MARGIN.top - MARGIN.bottom);

                        const xScale = scaleLinear<number>({
                          domain: [minX, maxX],
                          range: [0, innerWidth],
                        });
                        const yScale = scaleLinear<number>({
                          domain: [0, 100],
                          range: [innerHeight, 0],
                        });

                        return (
                          <svg
                            width={width}
                            height={PANEL_HEIGHT}
                            role="img"
                            aria-label={`Pass/fail rate vs output tokens, ${pair} (${kind})`}
                          >
                            <Group left={MARGIN.left} top={MARGIN.top}>
                              <text
                                x={0}
                                y={-10}
                                style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--ink)" }}
                              >
                                {`${shortPairLabel(pair)} (${kind})`}
                              </text>
                              <AxisLeft
                                scale={yScale}
                                tickValues={Y_TICKS}
                                stroke="var(--para)"
                                tickStroke="var(--para)"
                                axisLineClassName="axis-line-faded"
                                tickClassName="axis-tick-faded"
                                tickLabelProps={LEFT_AXIS_TICK_PROPS}
                              />
                              <AxisBottom
                                top={innerHeight}
                                scale={xScale}
                                numTicks={5}
                                stroke="var(--para)"
                                tickStroke="var(--para)"
                                axisLineClassName="axis-line-faded"
                                tickClassName="axis-tick-faded"
                                tickLabelProps={BOTTOM_AXIS_TICK_PROPS}
                              />

                              {runs.map((run, ri) => {
                                // A run of length 1 collapses to a zero-width
                                // Area path (no visible fill), so isolated
                                // non-empty bins are rendered as a rect pair
                                // spanning the bin's own width instead.
                                if (run.length >= 2) {
                                  return (
                                    <Group key={`fail-${ri}`}>
                                      <Area
                                        data={run}
                                        x={(d) => xScale(d.xMid)}
                                        y0={(d) => yScale(d.passRate)}
                                        y1={(d) => yScale(100)}
                                        fill={FAIL_COLOR}
                                        fillOpacity={0.85}
                                        stroke="none"
                                      />
                                    </Group>
                                  );
                                }
                                const bin = run[0];
                                const cx = xScale(bin.xMid);
                                if (!Number.isFinite(cx)) return null;
                                const rectWidth = Math.max(2, innerWidth / clampedBinCount);
                                const yTop = yScale(100);
                                const yBottom = yScale(bin.passRate);
                                return (
                                  <rect
                                    key={`fail-${ri}`}
                                    x={cx - rectWidth / 2}
                                    y={yTop}
                                    width={rectWidth}
                                    height={Math.max(0, yBottom - yTop)}
                                    fill={FAIL_COLOR}
                                    fillOpacity={0.85}
                                  />
                                );
                              })}
                              {runs.map((run, ri) => {
                                if (run.length >= 2) {
                                  return (
                                    <Group key={`pass-${ri}`}>
                                      <Area
                                        data={run}
                                        x={(d) => xScale(d.xMid)}
                                        y0={(d) => yScale(0)}
                                        y1={(d) => yScale(d.passRate)}
                                        fill={PASS_COLOR}
                                        fillOpacity={0.85}
                                        stroke="none"
                                      />
                                    </Group>
                                  );
                                }
                                const bin = run[0];
                                const cx = xScale(bin.xMid);
                                if (!Number.isFinite(cx)) return null;
                                const rectWidth = Math.max(2, innerWidth / clampedBinCount);
                                const yTop = yScale(bin.passRate);
                                const yBottom = yScale(0);
                                return (
                                  <rect
                                    key={`pass-${ri}`}
                                    x={cx - rectWidth / 2}
                                    y={yTop}
                                    width={rectWidth}
                                    height={Math.max(0, yBottom - yTop)}
                                    fill={PASS_COLOR}
                                    fillOpacity={0.85}
                                  />
                                );
                              })}
                              {bins.map((bin, bi) => {
                                const cx = xScale(bin.xMid);
                                if (!Number.isFinite(cx)) return null;
                                return (
                                  <rect
                                    key={bi}
                                    x={cx - Math.max(1, innerWidth / clampedBinCount / 2)}
                                    y={0}
                                    width={Math.max(2, innerWidth / clampedBinCount)}
                                    height={innerHeight}
                                    fill="transparent"
                                    onMouseMove={(evt) =>
                                      showTooltip(
                                        evt,
                                        <span>
                                          {pair} ({kind})
                                          <br />
                                          output tokens: {bin.x0.toFixed(0)}–{bin.x1.toFixed(0)}
                                          <br />
                                          pass: {bin.passCount} ({bin.passRate.toFixed(1)}%)
                                          <br />
                                          fail: {bin.failCount} ({bin.failRate.toFixed(1)}%)
                                        </span>
                                      )
                                    }
                                    onMouseLeave={hideTooltip}
                                  />
                                );
                              })}
                            </Group>
                          </svg>
                        );
                      }}
                    </ParentSize>
                  </div>
                );
              })}
            </div>
            {tooltipElement}
          </div>
        </>
      )}
    </ChartFrame>
  );
}
