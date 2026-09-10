"use client";

// Sampling-sweep line chart: mean logit difference vs one sampling
// parameter (`param`), one line per "family" — rows grouped by configKey
// minus the swept param, i.e. the same prover→verifier pair plus every
// OTHER fixed sampling param (and threshold) held constant. Registered
// three times in analysis-view.tsx (param="temperature" | "top_k" |
// "top_p"). A pair with more than one family (e.g. swept at two different
// fixed top_p values) gets a dash pattern per family so its lines stay
// distinguishable while sharing the pair's color.
//
// Sweep-aware empty state: if no family has >=2 distinct values of `param`
// in the filtered data, there is nothing to plot a trend against, so we
// show an explanatory card instead of a flat/degenerate chart.

import { useMemo, useRef } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { DerivedRow, weightedMean, maxOf } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";

const HEIGHT = 380;
const MARGIN = { top: 20, right: 24, bottom: 44, left: 68 };
// SVG stroke-dasharray patterns cycled across a pair's families beyond the
// first (which stays solid).
const DASH_PATTERNS = ["none", "6,3", "2,2", "8,2,2,2"];

type ParamName = "temperature" | "top_k" | "top_p";

const PARAM_META: Record<ParamName, { title: string; slug: string; xLabel: string }> = {
  temperature: {
    title: "Mean Logit Difference vs Temperature",
    slug: "diff-by-temperature",
    xLabel: "Temperature",
  },
  top_k: {
    title: "Mean Logit Difference vs Top-k",
    slug: "diff-by-top-k",
    xLabel: "Top-k",
  },
  top_p: {
    title: "Mean Logit Difference vs Top-p",
    slug: "diff-by-top-p",
    xLabel: "Top-p",
  },
};

function paramValueOf(row: DerivedRow, param: ParamName): number | null {
  return row[param];
}

// Family key = configKey with the swept param's slot blanked out, so rows
// that differ only in the swept param land in the same family.
function familyKeyFor(row: DerivedRow, param: ParamName): string {
  return JSON.stringify([
    row.prover_model,
    row.verifier_model,
    param === "temperature" ? null : row.temperature,
    param === "top_k" ? null : row.top_k,
    param === "top_p" ? null : row.top_p,
    row.verification_threshold,
  ]);
}

function familyLabel(row: DerivedRow, param: ParamName): string {
  const parts: string[] = [];
  if (param !== "temperature") parts.push(`t=${row.temperature ?? "—"}`);
  if (param !== "top_k") parts.push(`k=${row.top_k ?? "—"}`);
  if (param !== "top_p") parts.push(`p=${row.top_p ?? "—"}`);
  return `${row.pair} · ${parts.join(" ")}`;
}

type Family = {
  key: string;
  label: string;
  pair: string;
  points: Map<number, number[]>;
};

export default function DiffByParam({
  rows,
  colors,
  param,
}: {
  rows: DerivedRow[];
  colors: Map<string, string>;
  param: ParamName;
}) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const exportRef = useRef<HTMLDivElement | null>(null);

  const meta = PARAM_META[param];

  const families = useMemo(() => {
    const map = new Map<string, Family>();
    for (const row of rows) {
      const paramValue = paramValueOf(row, param);
      if (paramValue === null || row.mean_logit_difference === null) continue;
      const key = familyKeyFor(row, param);
      const entry =
        map.get(key) ??
        ({ key, label: familyLabel(row, param), pair: row.pair, points: new Map() } as Family);
      const list = entry.points.get(paramValue) ?? [];
      list.push(row.mean_logit_difference);
      entry.points.set(paramValue, list);
      map.set(key, entry);
    }
    return map;
  }, [rows, param]);

  const familyList = useMemo(
    () => Array.from(families.values()).sort((a, b) => a.label.localeCompare(b.label)),
    [families]
  );

  const hasSweep = familyList.some((f) => f.points.size >= 2);

  // Dash pattern per family: families sharing a pair get index 0 = solid,
  // subsequent families for that pair cycle through DASH_PATTERNS[1..].
  const dashByFamily = useMemo(() => {
    const map = new Map<string, string>();
    const seenPerPair = new Map<string, number>();
    for (const family of familyList) {
      const i = seenPerPair.get(family.pair) ?? 0;
      map.set(family.key, DASH_PATTERNS[i % DASH_PATTERNS.length]);
      seenPerPair.set(family.pair, i + 1);
    }
    return map;
  }, [familyList]);

  const series = useMemo(
    () =>
      familyList.map((family) => {
        const points = Array.from(family.points.entries())
          .map(([x, values]) => ({
            x,
            y: weightedMean(values, values.map(() => 1))!,
            count: values.length,
          }))
          .sort((a, b) => a.x - b.x);
        return { ...family, points };
      }),
    [familyList]
  );

  const legend: LegendItem[] = familyList.map((f) => ({
    key: f.key,
    label: f.label,
    color: colors.get(f.pair) ?? "#5f5f5f",
    dashed: dashByFamily.get(f.key) !== "none",
  }));

  const allValues = series.flatMap((s) => s.points.map((p) => p.x));
  const maxX = allValues.length > 0 ? maxOf(allValues) : 0;
  const tickValues = Array.from(new Set(allValues)).sort((a, b) => a - b);

  const allY = series.flatMap((s) => s.points.map((p) => p.y));
  const maxY = allY.length > 0 ? maxOf(allY) : 0;
  const yDomain: [number, number] = [0, maxY * 1.08 || 1];

  if (rows.length === 0 || familyList.length === 0) {
    return (
      <ChartFrame
        title={meta.title}
        slug={meta.slug}
        legend={[]}
        hiddenKeys={hiddenKeys}
        onToggleKey={toggleKey}
        exportRef={exportRef}
      >
        <p className="muted">No data for the current filters.</p>
      </ChartFrame>
    );
  }

  if (!hasSweep) {
    return (
      <ChartFrame
        title={meta.title}
        slug={meta.slug}
        legend={[]}
        hiddenKeys={hiddenKeys}
        onToggleKey={toggleKey}
        exportRef={exportRef}
      >
        <p className="muted">
          Only one {meta.xLabel.toLowerCase()} value in the filtered data — this chart needs
          runs across multiple sampling configs.
        </p>
      </ChartFrame>
    );
  }

  return (
    <ChartFrame
      title={meta.title}
      slug={meta.slug}
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      note="Line = family (pair + fixed sampling params); dash pattern distinguishes a pair's multiple families."
      exportRef={exportRef}
    >
      <div ref={containerRef} className="chart-tooltip-container">
        <div style={{ width: "100%", height: HEIGHT }}>
          <ParentSize>
            {({ width }) => {
              const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
              const innerHeight = Math.max(0, HEIGHT - MARGIN.top - MARGIN.bottom);

              const xScale = scaleLinear<number>({
                domain: [0, maxX * 1.08 || 1],
                range: [0, innerWidth],
              });
              const yScale = scaleLinear<number>({
                domain: yDomain,
                range: [innerHeight, 0],
              });

              return (
                <svg width={width} height={HEIGHT} role="img" aria-label={meta.title}>
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
                      tickValues={tickValues}
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
                      {meta.xLabel}
                    </text>
                    <text
                      transform={`translate(${-56}, ${innerHeight / 2}) rotate(-90)`}
                      textAnchor="middle"
                      style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                    >
                      Mean logit difference
                    </text>

                    {series.map((s) => {
                      if (hiddenKeys.has(s.key)) return null;
                      if (s.points.length < 2) return null;
                      const color = colors.get(s.pair) ?? "#5f5f5f";
                      const dash = dashByFamily.get(s.key) ?? "none";
                      return (
                        <LinePath
                          key={`line-${s.key}`}
                          data={s.points}
                          x={(d) => xScale(d.x)}
                          y={(d) => yScale(d.y)}
                          stroke={color}
                          strokeWidth={2}
                          strokeDasharray={dash === "none" ? undefined : dash}
                          fill="none"
                        />
                      );
                    })}

                    {series.map((s) => {
                      if (hiddenKeys.has(s.key)) return null;
                      const color = colors.get(s.pair) ?? "#5f5f5f";
                      return s.points.map((p, i) => {
                        const cx = xScale(p.x);
                        const cy = yScale(p.y);
                        if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
                        return (
                          <g key={`${s.key}-${i}`}>
                            <circle
                              cx={cx}
                              cy={cy}
                              r={8}
                              fill="transparent"
                              onMouseMove={(evt) =>
                                showTooltip(
                                  evt,
                                  <span>
                                    {s.label}
                                    <br />
                                    {meta.xLabel.toLowerCase()}: {p.x}
                                    <br />
                                    logit diff: {p.y.toFixed(4)}
                                    <br />
                                    count: {p.count}
                                  </span>
                                )
                              }
                              onMouseLeave={hideTooltip}
                            />
                            <circle cx={cx} cy={cy} r={3} fill={color} />
                          </g>
                        );
                      });
                    })}
                  </Group>
                </svg>
              );
            }}
          </ParentSize>
        </div>
        {tooltipElement}
      </div>
    </ChartFrame>
  );
}
