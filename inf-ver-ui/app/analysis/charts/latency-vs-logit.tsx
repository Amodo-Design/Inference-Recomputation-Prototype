"use client";

// Scatter: mean logit difference vs verification latency, one series per
// prover→verifier pair, with an optional IQR×6 upper clip on latency.

import { useMemo, useRef, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { clipUpper, maxOf, DerivedRow } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";
import { CheckboxControl } from "../controls";

const HEIGHT = 380;
const MARGIN = { top: 20, right: 24, bottom: 44, left: 68 };
const IQR_K = 6;

type Point = { x: number; y: number; pair: string };

export default function LatencyVsLogit({
  rows,
  colors,
}: {
  rows: DerivedRow[];
  colors: Map<string, string>;
}) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const [clipEnabled, setClipEnabled] = useState(true);
  const exportRef = useRef<HTMLDivElement | null>(null);

  const allPoints = useMemo<Point[]>(
    () =>
      rows
        .filter(
          (r) =>
            r.latency_ms !== null &&
            r.output_tokens !== null &&
            r.mean_logit_difference !== null
        )
        .map((r) => ({
          x: r.latency_ms as number,
          y: r.mean_logit_difference as number,
          pair: r.pair,
        })),
    [rows]
  );

  const clipResult = useMemo(() => clipUpper(allPoints, IQR_K), [allPoints]);
  const kept = clipResult.kept as Point[];

  const points = clipEnabled ? kept : allPoints;

  const pairs = useMemo(
    () => Array.from(new Set(allPoints.map((p) => p.pair))).sort(),
    [allPoints]
  );

  const legend: LegendItem[] = pairs.map((pair) => ({
    key: pair,
    label: pair,
    color: colors.get(pair) ?? "#5f5f5f",
  }));

  const maxY = points.length > 0 ? maxOf(points.map((p) => p.y)) : 0;
  const maxX = points.length > 0 ? maxOf(points.map((p) => p.x)) : 0;

  const note =
    allPoints.length === 0
      ? null
      : clipEnabled
        ? `${clipResult.hiddenCount} point${clipResult.hiddenCount === 1 ? "" : "s"} hidden above cutoff ${clipResult.cutoff.toFixed(1)} ms (IQR×${IQR_K}).`
        : `Clip off — showing all ${allPoints.length} points (IQR×${IQR_K} cutoff would be ${clipResult.cutoff.toFixed(1)} ms).`;

  return (
    <ChartFrame
      title="Mean Logit Difference vs Verification Latency"
      slug="latency-vs-logit"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      controls={
        <CheckboxControl
          label="Clip outliers (IQR×6)"
          checked={clipEnabled}
          onChange={setClipEnabled}
        />
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
                  domain: [0, maxX * 1.08 || 1],
                  range: [0, innerWidth],
                });
                const yScale = scaleLinear<number>({
                  domain: [0, maxY * 1.08 || 1],
                  range: [innerHeight, 0],
                });

                return (
                  <svg width={width} height={HEIGHT} role="img" aria-label="Mean logit difference vs verification latency">
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
                        tickLabelProps={BOTTOM_AXIS_TICK_PROPS}
                      />
                      <text
                        x={innerWidth / 2}
                        y={innerHeight + 36}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Latency (ms)
                      </text>
                      <text
                        transform={`translate(${-56}, ${innerHeight / 2}) rotate(-90)`}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Mean logit difference
                      </text>

                      {points.map((p, i) => {
                        if (hiddenKeys.has(p.pair)) return null;
                        const cx = xScale(p.x);
                        const cy = yScale(p.y);
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
                                    latency: {p.x.toFixed(1)} ms
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
