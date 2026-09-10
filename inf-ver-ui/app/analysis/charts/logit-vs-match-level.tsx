"use client";

// Scatter: mean logit difference vs exact-match level, one series per
// prover→verifier pair, with dashed reference lines at each distinct
// verification threshold in play.

import { useMemo, useRef, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear, scaleLog } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { DerivedRow, maxOf, minOf } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";
import { SelectControl } from "../controls";

const HEIGHT = 380;
const MARGIN = { top: 20, right: 110, bottom: 44, left: 68 };

type YScale = "linear" | "log";

export default function LogitVsMatchLevel({
  rows,
  colors,
}: {
  rows: DerivedRow[];
  colors: Map<string, string>;
}) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const [yScaleKind, setYScaleKind] = useState<YScale>("linear");
  const exportRef = useRef<HTMLDivElement | null>(null);

  const points = useMemo(
    () =>
      rows
        .filter(
          (r) => r.exact_match_level_pct !== null && r.mean_logit_difference !== null
        )
        .map((r) => ({
          x: r.exact_match_level_pct as number,
          y: r.mean_logit_difference as number,
          pair: r.pair,
          threshold: r.verification_threshold,
        })),
    [rows]
  );

  const allYPositive = points.length > 0 && points.every((p) => p.y > 0);
  const canLog = allYPositive;

  const yKind: YScale = canLog ? yScaleKind : "linear";

  const thresholds = useMemo(() => {
    const distinct = Array.from(
      new Set(
        points
          .map((p) => p.threshold)
          .filter((t): t is number => t !== null)
      )
    ).sort((a, b) => a - b);
    return distinct.slice(0, 4);
  }, [points]);

  const pairs = useMemo(
    () => Array.from(new Set(points.map((p) => p.pair))).sort(),
    [points]
  );

  const legend: LegendItem[] = pairs.map((pair) => ({
    key: pair,
    label: pair,
    color: colors.get(pair) ?? "#5f5f5f",
  }));

  // The threshold reference lines must land inside the plot, so the y domain
  // covers the thresholds as well as the data.
  const maxY =
    points.length > 0 ? maxOf([...points.map((p) => p.y), ...thresholds]) : 0;
  const minY =
    points.length > 0 ? minOf([...points.map((p) => p.y), ...thresholds]) : 0;

  const note =
    points.length === 0
      ? null
      : !canLog
        ? "Log y-scale disabled: not all mean-logit-difference values are positive."
        : null;

  return (
    <ChartFrame
      title="Mean Logit Difference vs Match Level"
      slug="logit-vs-match-level"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      controls={
        <span title={canLog ? undefined : "Log scale needs all y-values > 0"}>
          <SelectControl
            label="Y scale"
            value={yKind}
            onChange={(v) => setYScaleKind(v as YScale)}
            options={[
              { value: "linear", label: "Linear" },
              { value: "log", label: canLog ? "Log" : "Log (disabled)" },
            ]}
          />
        </span>
      }
      note={note}
      exportRef={exportRef}
    >
      {points.length === 0 ? (
        <p className="muted">No data for the current filters.</p>
      ) : (
        <div ref={containerRef} className="chart-tooltip-container">
          <div style={{ width: "100%", height: HEIGHT }}>
            <ParentSize>
              {({ width }) => {
                const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
                const innerHeight = Math.max(0, HEIGHT - MARGIN.top - MARGIN.bottom);

                const xScale = scaleLinear<number>({
                  domain: [0, 1],
                  range: [0, innerWidth],
                });

                const yDomain: [number, number] =
                  yKind === "log"
                    ? [Math.max(minY, Number.EPSILON), maxY * 1.08]
                    : [0, maxY * 1.08 || 1];

                const yScale =
                  yKind === "log"
                    ? scaleLog<number>({ domain: yDomain, range: [innerHeight, 0] })
                    : scaleLinear<number>({ domain: yDomain, range: [innerHeight, 0] });

                return (
                  <svg width={width} height={HEIGHT} role="img" aria-label="Mean logit difference vs match level">
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
                        numTicks={5}
                        stroke="var(--para)"
                        tickStroke="var(--para)"
                        axisLineClassName="axis-line-faded"
                        tickClassName="axis-tick-faded"
                        tickFormat={(v) => `${Math.round(Number(v) * 100)}%`}
                        tickLabelProps={BOTTOM_AXIS_TICK_PROPS}
                      />
                      <text
                        x={innerWidth / 2}
                        y={innerHeight + 36}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Match level
                      </text>
                      <text
                        transform={`translate(${-56}, ${innerHeight / 2}) rotate(-90)`}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Mean logit difference
                      </text>

                      {thresholds.map((t) => {
                        const y = yScale(t);
                        if (!Number.isFinite(y)) return null;
                        return (
                          <Group key={`threshold-${t}`}>
                            <line
                              x1={0}
                              x2={innerWidth}
                              y1={y}
                              y2={y}
                              style={{ stroke: "var(--ink)", strokeOpacity: 0.4 }}
                              strokeDasharray="4,3"
                              strokeWidth={1}
                            />
                            <text
                              x={innerWidth + 6}
                              y={y}
                              dy="0.32em"
                              textAnchor="start"
                              style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--ink)", opacity: 0.7 }}
                            >
                              {`threshold ${t.toFixed(4)}`}
                            </text>
                          </Group>
                        );
                      })}

                      {points.map((p, i) => {
                        if (hiddenKeys.has(p.pair)) return null;
                        const cx = xScale(p.x);
                        const cy = yScale(Math.max(p.y, yDomain[0]));
                        if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
                        const color = colors.get(p.pair) ?? "#5f5f5f";
                        return (
                          <g key={i}>
                            <circle
                              cx={cx}
                              cy={cy}
                              r={8}
                              fill="transparent"
                              onMouseMove={(evt) =>
                                showTooltip(
                                  evt,
                                  <span>
                                    {p.pair}
                                    <br />
                                    match: {(p.x * 100).toFixed(1)}%
                                    <br />
                                    logit diff: {p.y.toFixed(4)}
                                  </span>
                                )
                              }
                              onMouseLeave={hideTooltip}
                            />
                            <circle cx={cx} cy={cy} r={3} fill={color} fillOpacity={0.55} />
                          </g>
                        );
                      })}
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
