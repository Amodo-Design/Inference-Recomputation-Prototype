// Shared number formatting for the run-economics panel: per-token GPU
// busy/wall-clock values are tiny fractions of a second, so a fixed-point
// format either collapses to "0.00" or drowns in zeros — compact scientific
// notation keeps the magnitude readable.
export function formatPerToken(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return `${value.toExponential(1).replace("e+", "e")} s/tok`;
}

export function formatSeconds(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)} s`;
}

// Verification-advantage multiplier (prover busy/token ÷ verifier busy/token).
export function formatRatio(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(1)}×`;
}
