// Latency stats for the run summary strip. Nearest-rank percentiles
// (ceil(p·n)th of the sorted values): simple, exact for small n, and never
// interpolates values that were not observed.
export function runStats(
  elapsed: number[]
): { mean: number; p50: number; p95: number } | null {
  if (elapsed.length === 0) return null;
  const sorted = [...elapsed].sort((a, b) => a - b);
  const rank = (p: number) => sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)];
  const mean = sorted.reduce((sum, v) => sum + v, 0) / sorted.length;
  return { mean, p50: rank(0.5), p95: rank(0.95) };
}
