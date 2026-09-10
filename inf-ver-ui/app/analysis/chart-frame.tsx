"use client";

// Shared chart chrome for the Analysis tab: frame (title, Save PNG, controls
// row, legend, note) plus two small hooks (hidden-series state, a
// dependency-free tooltip) that every chart task (A5-A8) reuses.

import { useCallback, useRef, useState } from "react";
import type { ReactNode, RefObject, MouseEvent as ReactMouseEvent } from "react";
import { exportPng } from "./export-png";

// visx drops its default tick-label positioning entirely when a custom
// tickLabelProps is passed, so every chart that only wanted to set the font
// also lost textAnchor/dx/dy — leaving y labels sitting on the axis line and
// x labels start-anchored at their ticks. These re-supply the anchoring the
// visx defaults provide, plus the app's mono styling.
const TICK_LABEL_FONT = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  fill: "var(--para)",
} as const;

export const LEFT_AXIS_TICK_PROPS = () => ({
  ...TICK_LABEL_FONT,
  textAnchor: "end" as const,
  dx: "-0.25em",
  dy: "0.25em",
});

export const BOTTOM_AXIS_TICK_PROPS = () => ({
  ...TICK_LABEL_FONT,
  textAnchor: "middle" as const,
  dy: "0.25em",
});

export type LegendItem = {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
};

export function useHiddenKeys() {
  const [hiddenKeys, setHiddenKeys] = useState<Set<string>>(() => new Set());

  const toggleKey = useCallback((key: string) => {
    setHiddenKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  return { hiddenKeys, toggleKey };
}

type TooltipState = { x: number; y: number; content: ReactNode } | null;

export function useChartTooltip() {
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const showTooltip = useCallback(
    (evt: ReactMouseEvent, content: ReactNode) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const rawX = evt.clientX - rect.left + 12;
      const rawY = evt.clientY - rect.top + 12;
      const x = Math.min(Math.max(rawX, 4), Math.max(rect.width - 140, 4));
      const y = Math.min(Math.max(rawY, 4), Math.max(rect.height - 32, 4));
      setTooltip({ x, y, content });
    },
    []
  );

  const hideTooltip = useCallback(() => setTooltip(null), []);

  const tooltipElement = tooltip ? (
    <div className="chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
      {tooltip.content}
    </div>
  ) : null;

  return { containerRef, tooltipElement, showTooltip, hideTooltip };
}

export function ChartFrame({
  title,
  slug,
  legend,
  hiddenKeys,
  onToggleKey,
  controls,
  note,
  children,
  exportRef,
}: {
  title: string;
  slug: string;
  legend: LegendItem[];
  hiddenKeys: Set<string>;
  onToggleKey: (key: string) => void;
  controls?: ReactNode;
  note?: string | null;
  children: ReactNode;
  exportRef: RefObject<HTMLDivElement | null>;
}) {
  const [exporting, setExporting] = useState(false);

  const handleExport = useCallback(async () => {
    const container = exportRef.current;
    if (!container || exporting) return;
    setExporting(true);
    try {
      await exportPng(container, slug);
    } finally {
      setExporting(false);
    }
  }, [exportRef, slug, exporting]);

  return (
    <div className="chart-frame">
      <div className="chart-frame-header">
        <h4 className="chart-frame-title">{title}</h4>
        <button
          type="button"
          className="delete-button action-button-neutral"
          onClick={handleExport}
          disabled={exporting}
        >
          {exporting ? "Saving…" : "Save PNG"}
        </button>
      </div>
      {controls ? <div className="chart-controls">{controls}</div> : null}
      <div className="chart-area" ref={exportRef}>
        {children}
      </div>
      {legend.length >= 2 ? (
        <div className="chart-legend">
          {legend.map((item) => {
            const hidden = hiddenKeys.has(item.key);
            return (
              <button
                key={item.key}
                type="button"
                className="chart-legend-item"
                style={{ opacity: hidden ? 0.35 : 1 }}
                onClick={() => onToggleKey(item.key)}
                aria-pressed={!hidden}
              >
                <span
                  className={
                    item.dashed
                      ? "chart-legend-swatch chart-legend-swatch-dashed"
                      : "chart-legend-swatch"
                  }
                  style={
                    item.dashed
                      ? { borderTopColor: item.color }
                      : { background: item.color }
                  }
                />
                <span className="chart-legend-label">{item.label}</span>
              </button>
            );
          })}
        </div>
      ) : null}
      {note ? <p className="chart-note muted">{note}</p> : null}
    </div>
  );
}
