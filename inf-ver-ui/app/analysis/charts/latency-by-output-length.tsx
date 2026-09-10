"use client";

// Scatter + binned band: verification latency vs output token length, one
// series per prover→verifier pair. Each pair gets raw scatter points plus a
// quantile-binned mean line with a p25-p75 band, built via `quantileBins`.
// Outlier clipping applies IQR×3 on the x axis (output tokens) and IQR×6 on
// the y axis (latency) — both covered by a single toggle, per spec.

import { useMemo, useRef, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { Area, LinePath } from "@visx/shape";
import { clipUpper, quantileBins, maxOf, DerivedRow } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";
import { CheckboxControl, NumberControl } from "../controls";

const HEIGHT = 380;
const MARGIN = { top: 20, right: 24, bottom: 44, left: 68 };
const IQR_K_X = 3;
const IQR_K_Y = 6;
const DEFAULT_BIN_COUNT = 10;
const MIN_BIN_COUNT = 2;
const MAX_BIN_COUNT = 50;

type Point = { x: number; y: number; pair: string };

export default function LatencyByOutputLength({
  rows,
  colors,
}: {
  rows: DerivedRow[];
  colors: Map<string, string>;
}) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const [clipEnabled, setClipEnabled] = useState(true);
  const [binCount, setBinCount] = useState(DEFAULT_BIN_COUNT);
  const exportRef = useRef<HTMLDivElement | null>(null);

  const allPoints = useMemo<Point[]>(
    () =>
      rows
        .filter((r) => r.output_tokens !== null && r.latency_ms !== null)
        .map((r) => ({
          x: r.output_tokens as number,
          y: r.latency_ms as number,
          pair: r.pair,
        })),
    [rows]
  );

  // Clip x (output tokens, IQR×3) and y (latency, IQR×6) independently, then
  // keep only points inside both cutoffs.
  const xClip = useMemo(() => clipUpper(allPoints, IQR_K_X), [allPoints]);
  const yClip = useMemo(
    () => clipUpper(allPoints.map((p) => ({ x: p.y })), IQR_K_Y),
    [allPoints]
  );

  const kept = useMemo(
    () => allPoints.filter((p) => p.x <= xClip.cutoff && p.y <= yClip.cutoff),
    [allPoints, xClip.cutoff, yClip.cutoff]
  );

  const points = clipEnabled ? kept : allPoints;

  const clampedBinCount = Math.min(MAX_BIN_COUNT, Math.max(MIN_BIN_COUNT, Math.round(binCount)));

  const pairs = useMemo(
    () => Array.from(new Set(allPoints.map((p) => p.pair))).sort(),
    [allPoints]
  );

  const legend: LegendItem[] = pairs.map((pair) => ({
    key: pair,
    label: pair,
    color: colors.get(pair) ?? "#5f5f5f",
  }));

  const binsByPair = useMemo(() => {
    const map = new Map<string, ReturnType<typeof quantileBins>>();
    for (const pair of pairs) {
      const pairPoints = points
        .filter((p) => p.pair === pair)
        .map((p) => ({ x: p.x, y: p.y }));
      map.set(pair, quantileBins(pairPoints, clampedBinCount));
    }
    return map;
  }, [points, pairs, clampedBinCount]);

  const uniqueXCount = new Set(points.map((p) => p.x)).size;

  const maxX = points.length > 0 ? maxOf(points.map((p) => p.x)) : 0;
  const maxY = points.length > 0 ? maxOf(points.map((p) => p.y)) : 0;

  const xHiddenCount = allPoints.filter((p) => p.x > xClip.cutoff).length;
  const yHiddenCount = allPoints.filter((p) => p.y > yClip.cutoff).length;
  const hiddenCount = allPoints.length - kept.length;
  const note =
    allPoints.length === 0
      ? null
      : clipEnabled
        ? `${hiddenCount} point${hiddenCount === 1 ? "" : "s"} hidden above cutoff — x: ${xHiddenCount} above ${xClip.cutoff.toFixed(1)} tokens (IQR×${IQR_K_X}), y: ${yHiddenCount} above ${yClip.cutoff.toFixed(1)} ms (IQR×${IQR_K_Y}).`
        : `Clip off — showing all ${allPoints.length} points (x cutoff would be ${xClip.cutoff.toFixed(1)} tokens hiding ${xHiddenCount}; y cutoff would be ${yClip.cutoff.toFixed(1)} ms hiding ${yHiddenCount}).`;

  return (
    <ChartFrame
      title="Verification Latency vs Output Length"
      slug="latency-by-output-length"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      controls={
        <>
          <CheckboxControl
            label={`Clip outliers (x IQR×${IQR_K_X}, y IQR×${IQR_K_Y})`}
            checked={clipEnabled}
            onChange={setClipEnabled}
          />
          <NumberControl
            label="Bins"
            value={binCount}
            onChange={setBinCount}
            min={MIN_BIN_COUNT}
            max={MAX_BIN_COUNT}
            step={1}
          />
        </>
      }
      note={note}
      exportRef={exportRef}
    >
      {allPoints.length === 0 ? (
        <p className="muted">No data for the current filters.</p>
      ) : uniqueXCount < 2 ? (
        <p className="muted">Need at least 2 distinct output token values to bin.</p>
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
                  <svg width={width} height={HEIGHT} role="img" aria-label="Verification latency vs output length">
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
                        Latency (ms)
                      </text>

                      {pairs.map((pair) => {
                        if (hiddenKeys.has(pair)) return null;
                        const bins = binsByPair.get(pair) ?? [];
                        if (bins.length < 2) return null;
                        const color = colors.get(pair) ?? "#5f5f5f";
                        return (
                          <Group key={`band-${pair}`}>
                            <Area
                              data={bins}
                              x={(d) => xScale(d.xMid)}
                              y0={(d) => yScale(d.p25)}
                              y1={(d) => yScale(d.p75)}
                              fill={color}
                              fillOpacity={0.15}
                              stroke="none"
                            />
                          </Group>
                        );
                      })}

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
                                    output tokens: {p.x}
                                    <br />
                                    latency: {p.y.toFixed(1)} ms
                                  </span>
                                )
                              }
                              onMouseLeave={hideTooltip}
                            />
                            <circle cx={cx} cy={cy} r={2.5} fill={color} fillOpacity={0.25} />
                          </g>
                        );
                      })}

                      {pairs.map((pair) => {
                        if (hiddenKeys.has(pair)) return null;
                        const bins = binsByPair.get(pair) ?? [];
                        if (bins.length < 2) return null;
                        const color = colors.get(pair) ?? "#5f5f5f";
                        return (
                          <LinePath
                            key={`line-${pair}`}
                            data={bins}
                            x={(d) => xScale(d.xMid)}
                            y={(d) => yScale(d.mean)}
                            stroke={color}
                            strokeWidth={2}
                            fill="none"
                          />
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
