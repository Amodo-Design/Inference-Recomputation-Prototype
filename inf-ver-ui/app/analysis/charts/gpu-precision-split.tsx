"use client";

// Two stacked horizontal bars — one for the prover side, one for the
// verifier side — showing what fraction of each side's total tensor-active
// time ran on HMMA (FP16-class) tensor pipes vs IMMA (INT8-class) pipes vs
// other tensor pipes (e.g. DFMA), normalised to that side's own total.
//
// HMMA/IMMA are sampled independently of TENSOR_ACTIVE by DCGM, so in short
// or noisy windows hmma + imma can exceed the reported total, which would
// otherwise make the stack overshoot 100%. Each side is normalised by
// denom = max(total, hmma + imma) instead of total alone, so the stack
// always sums to exactly 1 regardless of which regime the sampling landed in.

import { useMemo, useRef } from "react";
import type { ReactNode } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear } from "@visx/scale";
import { AxisBottom } from "@visx/axis";
import { Group } from "@visx/group";
import { DerivedRow, OTHER_COLOR, PALETTE, gpuEligibleRows } from "../lib";
import { BOTTOM_AXIS_TICK_PROPS, ChartFrame, LegendItem, useChartTooltip, useHiddenKeys } from "../chart-frame";

const HEIGHT = 220;
const MARGIN = { top: 20, right: 24, bottom: 44, left: 90 };

const HMMA_KEY = "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE";
const IMMA_KEY = "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE";

const HMMA_COLOR = PALETTE[0];
const IMMA_COLOR = PALETTE[2];

type SegmentKey = "hmma" | "imma" | "other";
type Side = { label: string; total: number; denom: number; hmma: number; imma: number; other: number };

function sumSide(rows: DerivedRow[], gpuKey: "prover_gpu" | "verify_gpu"): Omit<Side, "label"> {
  let total = 0;
  let hmma = 0;
  let imma = 0;
  for (const r of rows) {
    const gpu = r[gpuKey];
    if (!gpu) continue;
    total += gpu.tensor_active_time_s ?? 0;
    hmma += gpu.pipe_activity_s?.[HMMA_KEY] ?? 0;
    imma += gpu.pipe_activity_s?.[IMMA_KEY] ?? 0;
  }
  const other = Math.max(0, total - hmma - imma);
  const denom = Math.max(total, hmma + imma);
  return { total, denom, hmma, imma, other };
}

export default function GpuPrecisionSplit({ rows }: { rows: DerivedRow[] }) {
  const { hiddenKeys, toggleKey } = useHiddenKeys();
  const { containerRef, tooltipElement, showTooltip, hideTooltip } = useChartTooltip();
  const exportRef = useRef<HTMLDivElement | null>(null);

  const { kept, excludedCount } = useMemo(() => gpuEligibleRows(rows), [rows]);

  const sides = useMemo<Side[]>(() => {
    const proverSums = sumSide(kept, "prover_gpu");
    const verifySums = sumSide(kept, "verify_gpu");
    return [
      { label: "Prover", ...proverSums },
      { label: "Verifier", ...verifySums },
    ];
  }, [kept]);

  const legend: LegendItem[] = [
    { key: "hmma", label: "HMMA (FP16-class)", color: HMMA_COLOR },
    { key: "imma", label: "IMMA (INT8-class)", color: IMMA_COLOR },
    { key: "other", label: "other tensor pipes", color: OTHER_COLOR },
  ];

  const excludedNote = `${excludedCount} event${excludedCount === 1 ? "" : "s"} excluded (fewer than 3 GPU samples or concurrent activity).`;

  const anyData = sides.some((s) => s.denom > 0);

  return (
    <ChartFrame
      title="Tensor-Core Precision Split by Side"
      slug="gpu-precision-split"
      legend={legend}
      hiddenKeys={hiddenKeys}
      onToggleKey={toggleKey}
      note={excludedNote}
      exportRef={exportRef}
    >
      {kept.length === 0 || !anyData ? (
        <p className="muted">No enriched GPU data for the current filters.</p>
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
                const xScalePercent = scaleLinear<number>({
                  domain: [0, 100],
                  range: [0, innerWidth],
                });
                const yScale = scaleBand<string>({
                  domain: sides.map((s) => s.label),
                  range: [0, innerHeight],
                  padding: 0.4,
                });
                const barHeight = yScale.bandwidth();

                return (
                  <svg
                    width={width}
                    height={HEIGHT}
                    role="img"
                    aria-label="Tensor-core precision split by side"
                  >
                    <Group left={MARGIN.left} top={MARGIN.top}>
                      <AxisBottom
                        top={innerHeight}
                        scale={xScalePercent}
                        numTicks={5}
                        stroke="var(--para)"
                        tickStroke="var(--para)"
                        axisLineClassName="axis-line-faded"
                        tickClassName="axis-tick-faded"
                        tickLabelProps={BOTTOM_AXIS_TICK_PROPS}
                        tickFormat={(v) => `${Math.round(Number(v))}%`}
                      />
                      <text
                        x={innerWidth / 2}
                        y={innerHeight + 36}
                        textAnchor="middle"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--para)" }}
                      >
                        Share of tensor-active time
                      </text>

                      {sides.map((s) => {
                        const bandY = yScale(s.label) ?? 0;
                        const labelY = bandY + barHeight / 2;
                        return (
                          <Group key={s.label}>
                            <text
                              x={-10}
                              y={labelY}
                              textAnchor="end"
                              dominantBaseline="middle"
                              style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--ink)" }}
                            >
                              {s.label}
                            </text>
                            {s.denom === 0 ? (
                              <text
                                x={4}
                                y={labelY}
                                dominantBaseline="middle"
                                className="muted"
                                style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
                              >
                                No enriched GPU data for the current filters.
                              </text>
                            ) : (
                              (
                                [
                                  { key: "hmma" as SegmentKey, seconds: s.hmma, color: HMMA_COLOR },
                                  { key: "imma" as SegmentKey, seconds: s.imma, color: IMMA_COLOR },
                                  { key: "other" as SegmentKey, seconds: s.other, color: OTHER_COLOR },
                                ] satisfies { key: SegmentKey; seconds: number; color: string }[]
                              ).reduce<{ nodes: ReactNode[]; cursor: number }>(
                                (acc, seg) => {
                                  if (hiddenKeys.has(seg.key)) return acc;
                                  const fraction = seg.seconds / s.denom;
                                  const x0 = xScale(acc.cursor);
                                  const segWidth = xScale(acc.cursor + fraction) - x0;
                                  acc.nodes.push(
                                    <rect
                                      key={seg.key}
                                      x={x0}
                                      y={bandY}
                                      width={Math.max(0, segWidth)}
                                      height={barHeight}
                                      fill={seg.color}
                                      fillOpacity={0.85}
                                      onMouseMove={(evt) =>
                                        showTooltip(
                                          evt,
                                          <span>
                                            {seg.key}: {(fraction * 100).toFixed(1)}% ({seg.seconds.toFixed(1)} s)
                                          </span>
                                        )
                                      }
                                      onMouseLeave={hideTooltip}
                                    />
                                  );
                                  acc.cursor += fraction;
                                  return acc;
                                },
                                { nodes: [], cursor: 0 }
                              ).nodes
                            )}
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
