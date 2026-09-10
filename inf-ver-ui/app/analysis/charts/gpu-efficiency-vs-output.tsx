"use client";

// Scatter: GPU efficiency ratio (prover busy-s/token ÷ verify busy-s/token)
// vs output length, one series per prover→verifier pair. A dashed reference
// line at y=1 separates "prover costs more per token than verify" (above)
// from "verify costs more per token than prover" (below) — i.e. above the
// line, verification is cheaper (more efficient) per token than proving.

import { useMemo, useRef, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { DerivedRow, clipUpper, gpuEligibleRows, maxOf } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";
import { CheckboxControl } from "../controls";

const HEIGHT = 380;
const MARGIN = { top: 20, right: 24, bottom: 44, left: 68 };
const IQR_K = 6;

type Point = { x: number; y: number; pair: string };
// Wrapper so clipUpper (which clips on a generic `x` field) can be reused to
// clip on the ratio (our chart's y value) instead of the chart's x value.
type ClipWrapper = { x: number; original: Point };

export default function GpuEfficiencyVsOutput({
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

  const { kept: eligibleRows, excludedCount } = useMemo(() => gpuEligibleRows(rows), [rows]);

  const allPoints = useMemo<Point[]>(
    () =>
      eligibleRows
        .filter((r) => r.output_tokens !== null && r.efficiencyRatio !== null)
        .map((r) => ({
          x: r.output_tokens as number,
          y: r.efficiencyRatio as number,
          pair: r.pair,
        })),
    [eligibleRows]
  );

  const clipResult = useMemo(() => {
    const wrapped: ClipWrapper[] = allPoints.map((p) => ({ x: p.y, original: p }));
    const result = clipUpper(wrapped, IQR_K);
    const kept = (result.kept as ClipWrapper[]).map((w) => w.original);
    return { kept, hiddenCount: result.hiddenCount, cutoff: result.cutoff };
  }, [allPoints]);

  const points = clipEnabled ? clipResult.kept : allPoints;

  const pairs = useMemo(
    () => Array.from(new Set(allPoints.map((p) => p.pair))).sort(),
    [allPoints]
  );

  const legend: LegendItem[] = pairs.map((pair) => ({
    key: pair,
    label: pair,
    color: colors.get(pair) ?? "#5f5f5f",
  }));

  const maxX = points.length > 0 ? maxOf(points.map((p) => p.x)) : 0;
  const maxY = points.length > 0 ? Math.max(maxOf(points.map((p) => p.y)), 1) : 1;

  const excludedNote = `${excludedCount} event${excludedCount === 1 ? "" : "s"} excluded (fewer than 3 GPU samples or concurrent activity).`;
  // iqrUpperCut returns Infinity below 4 points — no clipping happens, so
  // don't print a meaningless "cutoff Infinity" note.
  const clipNote =
    allPoints.length === 0
      ? ""
      : !Number.isFinite(clipResult.cutoff)
        ? ` No outlier cutoff (needs at least 4 points for IQR×${IQR_K}).`
        : clipEnabled
          ? ` ${clipResult.hiddenCount} point${clipResult.hiddenCount === 1 ? "" : "s"} hidden above cutoff ${clipResult.cutoff.toFixed(2)} (IQR×${IQR_K}).`
          : ` Clip off — showing all ${allPoints.length} points (IQR×${IQR_K} cutoff would be ${clipResult.cutoff.toFixed(2)}).`;
  const note = `${excludedNote}${clipNote}`;

  return (
    <ChartFrame
      title="GPU Efficiency Ratio vs Output Length"
      slug="gpu-efficiency-vs-output"
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
        <p className="muted">No enriched GPU data for the current filters.</p>
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
                  domain: [0, maxY * 1.08],
                  range: [innerHeight, 0],
                });
                const refLineY = yScale(1);

                return (
                  <svg
                    width={width}
                    height={HEIGHT}
                    role="img"
                    aria-label="GPU efficiency ratio vs output length"
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
                        Output tokens
                      </text>
                      <text
                        transform={`translate(${-56}, ${innerHeight / 2}) rotate(-90)`}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        prover busy-s/token ÷ verify busy-s/token
                      </text>
                      <line
                        x1={0}
                        x2={innerWidth}
                        y1={refLineY}
                        y2={refLineY}
                        strokeDasharray="4 4"
                        stroke="var(--para)"
                        strokeWidth={1}
                      />

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
                                    output tokens: {p.x.toFixed(0)}
                                    <br />
                                    efficiency ratio: {p.y.toFixed(3)}
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
