import { describe, expect, it } from "vitest";
import { runStats } from "./run-stats";

describe("runStats", () => {
  it("returns null on empty input", () => {
    expect(runStats([])).toBeNull();
  });

  it("computes mean, p50, p95 by nearest-rank", () => {
    const elapsed = Array.from({ length: 100 }, (_, i) => i + 1); // 1..100
    const stats = runStats(elapsed);
    expect(stats?.mean).toBeCloseTo(50.5);
    expect(stats?.p50).toBe(50);
    expect(stats?.p95).toBe(95);
  });

  it("handles single element", () => {
    expect(runStats([2.5])).toEqual({ mean: 2.5, p50: 2.5, p95: 2.5 });
  });

  it("does not mutate its input", () => {
    const input = [3, 1, 2];
    runStats(input);
    expect(input).toEqual([3, 1, 2]);
  });
});
