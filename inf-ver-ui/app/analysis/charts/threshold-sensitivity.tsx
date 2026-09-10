"use client";

// Threshold sensitivity sweep: pass rate vs candidate verification threshold,
// one series per prover→verifier pair, built from `thresholdSweep`. Two
// variants share this component (registered twice in analysis-view.tsx):
//   - "runs": thin per-config curves + a bold pair-mean curve.
//   - "band": pair-mean curve + a mean±std shaded band across the pair's
//     configs (std clipped so the band never leaves [0, 1]).
// X is symlog (small thresholds get room without a giant tail); Y is a
// 0-1.06 pass-rate domain rendered as a percentage. Pair means get direct
// end-of-line labels (honest/cheater) instead of relying on the legend.

import { useMemo, useRef, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { scaleSymlog } from "d3-scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { Area, LinePath } from "@visx/shape";
import { DerivedRow, makeGrid, shortModelName, shortPairLabel, thresholdSweep } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useHiddenKeys } from "../chart-frame";
import { GridControl, GridTriple } from "../controls";

const HEIGHT = 380;
const MARGIN = { top: 20, bottom: 44, left: 60 };
// End-of-line labels live in the right margin, so it must fit the longest
// label. ~6.1px/char at 10px mono, plus the 6px gap the labels are inset.
const LABEL_PX_PER_CHAR = 6.1;
const MIN_RIGHT_MARGIN = 90;
const MAX_RIGHT_MARGIN = 280;
const DEFAULT_GRID: GridTriple = { start: 0, end: 5, step: 0.01 };
const Y_DOMAIN: [number, number] = [0, 1.06];
const LABEL_MIN_GAP = 14;

// Standard log-scale tick progression — 1, 2, 5 per decade (0.1, 0.2, 0.5,
// 1, 2, 5, 10, ...) — plus 0 for the symlog's linear origin. Powers of ten
// are computed from integer exponents so the values stay exact.
function symlogTicks(max: number): number[] {
  const ticks = [0];
  for (let e = -1; Math.pow(10, e) <= max * (1 + 1e-9); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v <= max * (1 + 1e-9)) ticks.push(Number(v.toPrecision(12)));
    }
  }
  return ticks;
}

type MeanPoint = { threshold: number; mean: number; std: number };

function computeMeanStd(seriesList: { threshold: number; passRate: number }[][], grid: number[]): MeanPoint[] {
  return grid.map((threshold, i) => {
    const vals = seriesList
      .map((s) => s[i]?.passRate)
      .filter((v): v is number => v !== undefined);
    if (vals.length === 0) return { threshold, mean: 0, std: 0 };
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
    return { threshold, mean, std: Math.sqrt(variance) };
  });
}

export default function ThresholdSensitivity({
  rows,
  colors,
  variant,
}: {
  rows: DerivedRow[];
  colors: Map<string, string>;
  variant: "runs" | "band";
}) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const exportRef = useRef<HTMLDivElement | null>(null);
  const [gridTriple, setGridTriple] = useState<GridTriple>(DEFAULT_GRID);

  const title =
    variant === "runs"
      ? "Threshold Sensitivity by Temperature"
      : "Threshold Sensitivity, Std Band";
  const slug = variant === "runs" ? "threshold-sensitivity-runs" : "threshold-sensitivity-band";

  const grid = useMemo(
    () => makeGrid(gridTriple.start, gridTriple.end, gridTriple.step),
    [gridTriple]
  );

  // Memoized on [rows, grid] only — the sweep recomputes a pass-rate curve
  // per distinct config, so this is the one expensive step worth guarding
  // against re-running on every render (toggling a legend key, resizing).
  const sweep = useMemo(() => thresholdSweep(rows, grid), [rows, grid]);

  const pairMeta = useMemo(() => {
    const map = new Map<string, { honest: boolean; configKeys: Set<string> }>();
    for (const r of rows) {
      const entry = map.get(r.pair) ?? { honest: r.honest, configKeys: new Set<string>() };
      entry.configKeys.add(r.configKey);
      map.set(r.pair, entry);
    }
    return map;
  }, [rows]);

  const pairs = useMemo(() => Array.from(pairMeta.keys()).sort(), [pairMeta]);

  const configPair = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of rows) map.set(r.configKey, r.pair);
    return map;
  }, [rows]);

  const meanStdByPair = useMemo(() => {
    const result = new Map<string, MeanPoint[]>();
    for (const pair of pairs) {
      const configKeys = Array.from(pairMeta.get(pair)!.configKeys);
      const seriesList = configKeys.map((ck) => sweep.get(ck) ?? []);
      result.set(pair, computeMeanStd(seriesList, grid));
    }
    return result;
  }, [pairs, pairMeta, sweep, grid]);

  const legend: LegendItem[] = pairs.map((pair) => ({
    key: pair,
    label: pair,
    color: colors.get(pair) ?? "#5f5f5f",
  }));

  const end = grid.length > 0 ? grid[grid.length - 1] : 1;
  const xDomain: [number, number] = [0, end > 0 ? end : 1];
  const tickValues = symlogTicks(xDomain[1]);

  // Direct end-of-line label text: honest pairs are "model → model", so a
  // single short model name reads better and takes half the space.
  const labelTextFor = (pair: string) => {
    const honest = pairMeta.get(pair)?.honest ?? false;
    return honest
      ? `${shortModelName(pair.split(" → ")[0])} (honest)`
      : `${shortPairLabel(pair)} (cheater)`;
  };
  const maxLabelChars = pairs.reduce((max, pair) => Math.max(max, labelTextFor(pair).length), 0);
  const marginRight = Math.min(
    MAX_RIGHT_MARGIN,
    Math.max(MIN_RIGHT_MARGIN, Math.ceil(maxLabelChars * LABEL_PX_PER_CHAR) + 12)
  );

  if (rows.length === 0) {
    return (
      <ChartFrame
        title={title}
        slug={slug}
        legend={[]}
        hiddenKeys={hiddenKeys}
        onToggleKey={toggleKey}
        exportRef={exportRef}
      >
        <p className="muted">No data for the current filters.</p>
      </ChartFrame>
    );
  }

  return (
    <ChartFrame
      title={title}
      slug={slug}
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      controls={
        <GridControl
          start={gridTriple.start}
          end={gridTriple.end}
          step={gridTriple.step}
          onChange={setGridTriple}
        />
      }
      note={
        variant === "runs"
          ? "Thin lines = each config's own sweep; bold = pair mean."
          : "Shaded band = pair mean ± std across that pair's configs."
      }
      exportRef={exportRef}
    >
      {grid.length === 0 ? (
        <p className="muted">Grid range is empty — widen start/end.</p>
      ) : (
        <div style={{ width: "100%", height: HEIGHT }}>
          <ParentSize>
            {({ width }) => {
              const innerWidth = Math.max(0, width - MARGIN.left - marginRight);
              const innerHeight = Math.max(0, HEIGHT - MARGIN.top - MARGIN.bottom);

              const xScale = scaleSymlog<number, number>()
                .constant(0.1)
                .domain(xDomain)
                .range([0, innerWidth]);

              const yScale = scaleLinear<number>({
                domain: Y_DOMAIN,
                range: [innerHeight, 0],
              });

              // Direct end-of-line labels for each visible pair mean, nudged
              // apart vertically so overlapping means don't collide.
              const rawLabels = pairs
                .filter((pair) => !hiddenKeys.has(pair))
                .map((pair) => {
                  const series = meanStdByPair.get(pair) ?? [];
                  const last = series[series.length - 1];
                  if (!last) return null;
                  return {
                    key: pair,
                    text: labelTextFor(pair),
                    color: colors.get(pair) ?? "#5f5f5f",
                    y: yScale(last.mean),
                  };
                })
                .filter((l): l is { key: string; text: string; color: string; y: number } => l !== null)
                .sort((a, b) => a.y - b.y);
              for (let i = 1; i < rawLabels.length; i++) {
                if (rawLabels[i].y - rawLabels[i - 1].y < LABEL_MIN_GAP) {
                  rawLabels[i].y = rawLabels[i - 1].y + LABEL_MIN_GAP;
                }
              }

              return (
                <svg width={width} height={HEIGHT} role="img" aria-label={title}>
                  <Group left={MARGIN.left} top={MARGIN.top}>
                    <AxisLeft
                      scale={yScale}
                      numTicks={5}
                      tickFormat={(v) => `${Math.round((v as number) * 100)}%`}
                      stroke="var(--para)"
                      tickStroke="var(--para)"
                      axisLineClassName="axis-line-faded"
                      tickClassName="axis-tick-faded"
                      tickLabelProps={LEFT_AXIS_TICK_PROPS}
                    />
                    <AxisBottom
                      top={innerHeight}
                      scale={xScale}
                      tickValues={tickValues}
                      tickFormat={(v) => String(v)}
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
                      Candidate threshold
                    </text>
                    <text
                      transform={`translate(${-48}, ${innerHeight / 2}) rotate(-90)`}
                      textAnchor="middle"
                      style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                    >
                      Pass rate
                    </text>

                    {variant === "band" &&
                      pairs.map((pair) => {
                        if (hiddenKeys.has(pair)) return null;
                        const series = meanStdByPair.get(pair) ?? [];
                        if (series.length < 2) return null;
                        const color = colors.get(pair) ?? "#5f5f5f";
                        return (
                          <Area
                            key={`band-${pair}`}
                            data={series}
                            x={(d) => xScale(d.threshold)}
                            y0={(d) => yScale(Math.max(0, Math.min(1, d.mean - d.std)))}
                            y1={(d) => yScale(Math.max(0, Math.min(1, d.mean + d.std)))}
                            fill={color}
                            fillOpacity={0.18}
                            stroke="none"
                          />
                        );
                      })}

                    {variant === "runs" &&
                      Array.from(sweep.entries()).map(([configKey, series]) => {
                        const pair = configPair.get(configKey);
                        if (!pair || hiddenKeys.has(pair)) return null;
                        if (series.length < 2) return null;
                        const color = colors.get(pair) ?? "#5f5f5f";
                        return (
                          <LinePath
                            key={`run-${configKey}`}
                            data={series}
                            x={(d) => xScale(d.threshold)}
                            y={(d) => yScale(d.passRate)}
                            stroke={color}
                            strokeWidth={1}
                            strokeOpacity={0.35}
                            fill="none"
                          />
                        );
                      })}

                    {pairs.map((pair) => {
                      if (hiddenKeys.has(pair)) return null;
                      const series = meanStdByPair.get(pair) ?? [];
                      if (series.length < 2) return null;
                      const color = colors.get(pair) ?? "#5f5f5f";
                      return (
                        <LinePath
                          key={`mean-${pair}`}
                          data={series}
                          x={(d) => xScale(d.threshold)}
                          y={(d) => yScale(d.mean)}
                          stroke={color}
                          strokeWidth={2.5}
                          fill="none"
                        />
                      );
                    })}

                    {rawLabels.map((l) => (
                      <text
                        key={`label-${l.key}`}
                        x={innerWidth + 6}
                        y={l.y}
                        dominantBaseline="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--ink)" }}
                      >
                        {l.text}
                      </text>
                    ))}
                  </Group>
                </svg>
              );
            }}
          </ParentSize>
        </div>
      )}
    </ChartFrame>
  );
}
