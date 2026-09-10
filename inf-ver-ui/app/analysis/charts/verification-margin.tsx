"use client";

// Faceted scatter: verification logit-difference gap (incorrect prover mean
// minus honest mean) vs top_k, one small-multiple panel per distinct top_p.
// Point color encodes temperature on a continuous blue ramp; marker shape
// encodes the incorrect prover (color is reserved for temperature).

import { useMemo, useRef, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { DerivedRow, verificationMargins, maxOf, minOf } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";

const PANEL_HEIGHT = 160;
const MARGIN = { top: 14, right: 24, bottom: 30, left: 60 };

const RAMP_STOPS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
const MARKER_SHAPES = ["circle", "square", "triangle", "diamond", "cross"] as const;
type MarkerShape = (typeof MARKER_SHAPES)[number];

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

function Marker({
  shape,
  cx,
  cy,
  color,
  size = 4,
}: {
  shape: MarkerShape;
  cx: number;
  cy: number;
  color: string;
  size?: number;
}) {
  const common = { fill: color, fillOpacity: 0.65 };
  switch (shape) {
    case "square":
      return <rect x={cx - size} y={cy - size} width={size * 2} height={size * 2} {...common} />;
    case "triangle": {
      const points = `${cx},${cy - size * 1.15} ${cx - size} ,${cy + size * 0.85} ${cx + size},${cy + size * 0.85}`;
      return <polygon points={points} {...common} />;
    }
    case "diamond": {
      const points = `${cx},${cy - size * 1.2} ${cx + size * 1.2},${cy} ${cx},${cy + size * 1.2} ${cx - size * 1.2},${cy}`;
      return <polygon points={points} {...common} />;
    }
    case "cross": {
      const w = size * 0.6;
      return (
        <g {...common}>
          <rect x={cx - size} y={cy - w} width={size * 2} height={w * 2} />
          <rect x={cx - w} y={cy - size} width={w * 2} height={size * 2} />
        </g>
      );
    }
    case "circle":
    default:
      return <circle cx={cx} cy={cy} r={size} {...common} />;
  }
}

export default function VerificationMargin({
  rows,
  colors,
}: {
  rows: DerivedRow[];
  colors: Map<string, string>;
}) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const exportRef = useRef<HTMLDivElement | null>(null);
  void colors;

  const margins = useMemo(() => verificationMargins(rows), [rows]);

  const points = useMemo(
    () => margins.filter((m) => m.top_k !== null && m.top_p !== null),
    [margins]
  );

  const provers = useMemo(
    () => Array.from(new Set(points.map((p) => p.incorrectProver))).sort(),
    [points]
  );
  const shapeByProver = useMemo(() => {
    const map = new Map<string, MarkerShape>();
    provers.forEach((prover, i) => {
      map.set(prover, MARKER_SHAPES[i % MARKER_SHAPES.length]);
    });
    return map;
  }, [provers]);

  const topPValues = useMemo(
    () =>
      Array.from(new Set(points.map((p) => p.top_p as number))).sort((a, b) => a - b),
    [points]
  );

  const temperatures = points
    .map((p) => p.temperature)
    .filter((t): t is number => t !== null);
  const tempMin = temperatures.length > 0 ? minOf(temperatures) : 0;
  const tempMax = temperatures.length > 0 ? maxOf(temperatures) : 0;

  const colorForTemp = (temp: number | null) => {
    if (temp === null) return "var(--para)";
    if (tempMax === tempMin) return RAMP_STOPS[2];
    return rampColor((temp - tempMin) / (tempMax - tempMin));
  };

  const maxTopK = points.length > 0 ? maxOf(points.map((p) => p.top_k as number)) : 0;
  const topKTicks = Array.from(new Set(points.map((p) => p.top_k as number))).sort(
    (a, b) => a - b
  );

  const values = points.map((p) => p.value);
  const rawMin = values.length > 0 ? minOf(values) : 0;
  const rawMax = values.length > 0 ? maxOf(values) : 0;
  const yMin = Math.min(0, rawMin);
  const yMax = Math.max(0, rawMax);
  const pad = (yMax - yMin) * 0.08 || 1;
  const yDomain: [number, number] = [yMin - pad, yMax + pad];

  const legend: LegendItem[] = provers.map((prover) => ({
    key: prover,
    label: `incorrect prover: ${prover}`,
    color: "var(--ink)",
  }));

  if (margins.length === 0) {
    return (
      <ChartFrame
        title="Verification Logit Difference Gap vs Top-k"
        slug="verification-margin"
        legend={[]}
        hiddenKeys={hiddenKeys}
        onToggleKey={toggleKey}
        exportRef={exportRef}
      >
        <p className="muted">
          Needs both honest (prover = verifier) and cheating (prover ≠ verifier) runs
          sharing the same verifier and sampling config in the filtered data — the gap is
          the cheater&apos;s mean logit difference minus the honest mean.
        </p>
      </ChartFrame>
    );
  }

  if (points.length === 0) {
    return (
      <ChartFrame
        title="Verification Logit Difference Gap vs Top-k"
        slug="verification-margin"
        legend={[]}
        hiddenKeys={hiddenKeys}
        onToggleKey={toggleKey}
        exportRef={exportRef}
      >
        <p className="muted">
          No margin points with top-k and top-p available in the filtered data.
        </p>
      </ChartFrame>
    );
  }

  return (
    <ChartFrame
      title="Verification Logit Difference Gap vs Top-k"
      slug="verification-margin"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      note={
        temperatures.length > 0
          ? `Point color = temperature (${tempMin.toFixed(2)} → ${tempMax.toFixed(2)}). One panel per top_p; shared top_k axis.`
          : "One panel per top_p; shared top_k axis."
      }
      exportRef={exportRef}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span className="control-label">temp {tempMin.toFixed(2)}</span>
        <span
          style={{
            display: "inline-block",
            width: 96,
            height: 8,
            borderRadius: 4,
            background: `linear-gradient(to right, ${RAMP_STOPS.join(", ")})`,
          }}
        />
        <span className="control-label">{tempMax.toFixed(2)}</span>
      </div>
      <div ref={containerRef} className="chart-tooltip-container">
        <div>
          {topPValues.map((topP) => {
            const facetPoints = points.filter((p) => p.top_p === topP);
            return (
              <div key={topP} style={{ width: "100%", height: PANEL_HEIGHT }}>
                <ParentSize>
                  {({ width }) => {
                    const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
                    const innerHeight = Math.max(
                      0,
                      PANEL_HEIGHT - MARGIN.top - MARGIN.bottom
                    );

                    const xScale = scaleLinear<number>({
                      domain: [0, maxTopK * 1.08 || 1],
                      range: [0, innerWidth],
                    });
                    const yScale = scaleLinear<number>({
                      domain: yDomain,
                      range: [innerHeight, 0],
                    });
                    const zeroY = yScale(0);

                    return (
                      <svg
                        width={width}
                        height={PANEL_HEIGHT}
                        role="img"
                        aria-label={`Verification margin vs top_k, top_p ${topP}`}
                      >
                        <Group left={MARGIN.left} top={MARGIN.top}>
                          <text
                            x={0}
                            y={-2}
                            style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--ink)" }}
                          >
                            {`top_p = ${topP}`}
                          </text>
                          <AxisLeft
                            scale={yScale}
                            numTicks={3}
                            stroke="var(--para)"
                            tickStroke="var(--para)"
                            axisLineClassName="axis-line-faded"
                            tickClassName="axis-tick-faded"
                            tickLabelProps={LEFT_AXIS_TICK_PROPS}
                          />
                          <AxisBottom
                            top={innerHeight}
                            scale={xScale}
                            tickValues={topKTicks}
                            stroke="var(--para)"
                            tickStroke="var(--para)"
                            axisLineClassName="axis-line-faded"
                            tickClassName="axis-tick-faded"
                            tickLabelProps={BOTTOM_AXIS_TICK_PROPS}
                          />
                          <line
                            x1={0}
                            x2={innerWidth}
                            y1={zeroY}
                            y2={zeroY}
                            style={{ stroke: "var(--para)" }}
                            strokeWidth={1}
                          />

                          {facetPoints.map((p, i) => {
                            if (hiddenKeys.has(p.incorrectProver)) return null;
                            const cx = xScale(p.top_k as number);
                            const cy = yScale(p.value);
                            if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
                            const shape = shapeByProver.get(p.incorrectProver) ?? "circle";
                            const color = colorForTemp(p.temperature);
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
                                        {p.incorrectProver} → {p.verifier}
                                        <br />
                                        top_k: {p.top_k} · top_p: {p.top_p}
                                        <br />
                                        temp: {p.temperature ?? "—"}
                                        <br />
                                        margin: {p.value.toFixed(4)}
                                      </span>
                                    )
                                  }
                                  onMouseLeave={hideTooltip}
                                />
                                <Marker shape={shape} cx={cx} cy={cy} color={color} />
                              </g>
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
    </ChartFrame>
  );
}
