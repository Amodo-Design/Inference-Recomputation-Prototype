"use client";

// Grouped bar chart: mean SM occupancy per prover→verifier pair, one bar for
// the prover side and one for the verifier side. A pair with no valid
// occupancy samples on a side simply omits that bar rather than drawing a
// zero-height one.

import { useMemo, useRef } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { DerivedRow, PALETTE, gpuEligibleRows, shortPairLabel } from "../lib";
import { LEFT_AXIS_TICK_PROPS, BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";

const HEIGHT = 380;
const MARGIN = { top: 20, right: 24, bottom: 56, left: 68 };

const PROVER_COLOR = PALETTE[0];
const VERIFY_COLOR = PALETTE[1];

type SideKey = "prover" | "verify";

type PairStats = {
  pair: string;
  prover: { mean: number; count: number } | null;
  verify: { mean: number; count: number } | null;
};

function meanOf(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export default function GpuOccupancyBySide({ rows }: { rows: DerivedRow[] }) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const exportRef = useRef<HTMLDivElement | null>(null);

  const { kept, excludedCount } = useMemo(() => gpuEligibleRows(rows), [rows]);

  const stats = useMemo<PairStats[]>(() => {
    const pairs = Array.from(new Set(kept.map((r) => r.pair))).sort();
    return pairs.map((pair) => {
      const pairRows = kept.filter((r) => r.pair === pair);
      const proverValues = pairRows
        .map((r) => r.prover_gpu?.sm_occupancy_mean ?? null)
        .filter((v): v is number => v !== null);
      const verifyValues = pairRows
        .map((r) => r.verify_gpu?.sm_occupancy_mean ?? null)
        .filter((v): v is number => v !== null);
      return {
        pair,
        prover: proverValues.length > 0 ? { mean: meanOf(proverValues), count: proverValues.length } : null,
        verify: verifyValues.length > 0 ? { mean: meanOf(verifyValues), count: verifyValues.length } : null,
      };
    });
  }, [kept]);

  const legend: LegendItem[] = [
    { key: "prover", label: "prover", color: PROVER_COLOR },
    { key: "verify", label: "verify", color: VERIFY_COLOR },
  ];

  const excludedNote = `${excludedCount} event${excludedCount === 1 ? "" : "s"} excluded (fewer than 3 GPU samples or concurrent activity).`;

  const pairsWithData = stats.filter((s) => s.prover !== null || s.verify !== null);

  // Occupancy means are often far below 1 (mostly-idle sampling windows), so
  // a fixed [0, 1] domain renders sub-pixel, invisible bars. Scale to the
  // data instead, capped at the metric's natural [0, 1] bound.
  const maxMean = pairsWithData.reduce(
    (max, s) => Math.max(max, s.prover?.mean ?? 0, s.verify?.mean ?? 0),
    0
  );
  const yMax = maxMean > 0 ? Math.min(1, maxMean * 1.15) : 1;

  const fmtMean = (v: number) => (v >= 0.01 || v === 0 ? v.toFixed(2) : v.toPrecision(2));

  return (
    <ChartFrame
      title="SM Occupancy by Side"
      slug="gpu-occupancy-by-side"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      note={excludedNote}
      exportRef={exportRef}
    >
      {kept.length === 0 || pairsWithData.length === 0 ? (
        <p className="muted">No enriched GPU data for the current filters.</p>
      ) : (
        <div ref={containerRef} className="chart-tooltip-container">
          <div style={{ width: "100%", height: HEIGHT }}>
            <ParentSize>
              {({ width }) => {
                const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
                const innerHeight = Math.max(0, HEIGHT - MARGIN.top - MARGIN.bottom);

                const xScale = scaleBand<string>({
                  domain: pairsWithData.map((s) => s.pair),
                  range: [0, innerWidth],
                  padding: 0.3,
                });
                const yScale = scaleLinear<number>({
                  domain: [0, yMax],
                  range: [innerHeight, 0],
                });
                const bandwidth = xScale.bandwidth();
                const barWidth = bandwidth / 2;

                return (
                  <svg width={width} height={HEIGHT} role="img" aria-label="Mean SM occupancy by side">
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
                        tickFormat={(pair) => shortPairLabel(pair as string)}
                        stroke="var(--para)"
                        tickStroke="var(--para)"
                        axisLineClassName="axis-line-faded"
                        tickClassName="axis-tick-faded"
                        tickLabelProps={BOTTOM_AXIS_TICK_PROPS}
                      />
                      <text
                        transform={`translate(${-56}, ${innerHeight / 2}) rotate(-90)`}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Mean SM occupancy
                      </text>

                      {pairsWithData.map((s) => {
                        const bandX = xScale(s.pair) ?? 0;
                        const sides: { key: SideKey; color: string; data: { mean: number; count: number } | null }[] = [
                          { key: "prover", color: PROVER_COLOR, data: s.prover },
                          { key: "verify", color: VERIFY_COLOR, data: s.verify },
                        ];
                        return (
                          <Group key={s.pair}>
                            {sides.map((side, si) => {
                              if (side.data === null || hiddenKeys.has(side.key)) return null;
                              const x = bandX + si * barWidth;
                              const y = yScale(side.data.mean);
                              const yBase = yScale(0);
                              return (
                                <Group key={side.key}>
                                  <rect
                                    x={x}
                                    y={y}
                                    width={barWidth}
                                    height={Math.max(0, yBase - y)}
                                    fill={side.color}
                                    fillOpacity={0.85}
                                    onMouseMove={(evt) =>
                                      showTooltip(
                                        evt,
                                        <span>
                                          {s.pair} · {side.key}: {fmtMean(side.data!.mean)} (n={side.data!.count})
                                        </span>
                                      )
                                    }
                                    onMouseLeave={hideTooltip}
                                  />
                                  <text
                                    x={x + barWidth / 2}
                                    y={y - 4}
                                    textAnchor="middle"
                                    style={{
                                      fontFamily: "var(--font-mono)",
                                      fontSize: 10,
                                      fill: "var(--para)",
                                      pointerEvents: "none",
                                    }}
                                  >
                                    {fmtMean(side.data.mean)}
                                  </text>
                                </Group>
                              );
                            })}
                          </Group>
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
