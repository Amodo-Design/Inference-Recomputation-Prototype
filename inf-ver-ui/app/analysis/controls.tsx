"use client";

// Small labeled form controls shared by chart-specific control rows
// (`controls` prop on ChartFrame). Mono, uppercase, compact — matches the
// app's chrome rather than default browser widget styling.

import { useEffect, type ReactNode } from "react";

function ControlLabel({ children }: { children: ReactNode }) {
  return <span className="control-label">{children}</span>;
}

export function NumberControl({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="control number-control">
      <ControlLabel>{label}</ControlLabel>
      <input
        type="number"
        className="control-input"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const next = e.target.valueAsNumber;
          if (!Number.isNaN(next)) onChange(next);
        }}
      />
    </label>
  );
}

export function CheckboxControl({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="control checkbox-control">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <ControlLabel>{label}</ControlLabel>
    </label>
  );
}

export function SelectControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="control select-control">
      <ControlLabel>{label}</ControlLabel>
      <select
        className="control-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export type GridTriple = { start: number; end: number; step: number };

const MAX_GRID_POINTS = 500;

export function GridControl({
  start,
  end,
  step,
  onChange,
}: {
  start: number;
  end: number;
  step: number;
  onChange: (grid: GridTriple) => void;
}) {
  const emit = (next: GridTriple) => {
    const span = next.end - next.start;
    // A non-positive span (end <= start) is nonsensical to clamp a step
    // against — pass the values through unchanged and let validation
    // elsewhere flag the range instead of emitting a clamped step here.
    if (span <= 0) {
      onChange(next);
      return;
    }
    let clampedStep = next.step;
    if (clampedStep > 0) {
      const points = span / clampedStep;
      if (points > MAX_GRID_POINTS) {
        clampedStep = span / MAX_GRID_POINTS;
      }
    }
    onChange({ start: next.start, end: next.end, step: clampedStep });
  };

  // Apply the same clamp to the initial props once on mount: if the caller
  // supplies a start/end/step that already implies more than
  // MAX_GRID_POINTS points, notify with the clamped step. Otherwise leave
  // the initial props alone (no need to re-emit an unchanged value).
  useEffect(() => {
    const span = end - start;
    if (span <= 0 || step <= 0) return;
    const points = span / step;
    if (points > MAX_GRID_POINTS) {
      onChange({ start, end, step: span / MAX_GRID_POINTS });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="control grid-control">
      <ControlLabel>Grid</ControlLabel>
      <span className="grid-control-fields">
        <label className="grid-field">
          <span className="control-label">start</span>
          <input
            type="number"
            className="control-input control-input-grid"
            value={start}
            onChange={(e) => {
              const next = e.target.valueAsNumber;
              if (!Number.isNaN(next)) emit({ start: next, end, step });
            }}
          />
        </label>
        <label className="grid-field">
          <span className="control-label">end</span>
          <input
            type="number"
            className="control-input control-input-grid"
            value={end}
            onChange={(e) => {
              const next = e.target.valueAsNumber;
              if (!Number.isNaN(next)) emit({ start, end: next, step });
            }}
          />
        </label>
        <label className="grid-field">
          <span className="control-label">step</span>
          <input
            type="number"
            className="control-input control-input-grid"
            value={step}
            onChange={(e) => {
              const next = e.target.valueAsNumber;
              if (!Number.isNaN(next) && next > 0) emit({ start, end, step: next });
            }}
          />
        </label>
      </span>
    </div>
  );
}
