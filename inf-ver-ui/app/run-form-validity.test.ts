import { describe, it, expect } from "vitest";
import { AllocationRow, rowsAreValid } from "./run-label";

function rows(...partials: Partial<AllocationRow>[]): AllocationRow[] {
  return partials.map((p) => ({ model: "m", percent: 100, concurrency: 1, ...p }));
}

describe("rowsAreValid", () => {
  it("rejects a percent-0 row (e.g. a freshly added row before seeding)", () => {
    expect(rowsAreValid(rows({ percent: 0 }))).toBe(false);
  });

  it("rejects concurrency 0", () => {
    expect(rowsAreValid(rows({ concurrency: 0 }))).toBe(false);
  });

  it("rejects concurrency 33", () => {
    expect(rowsAreValid(rows({ concurrency: 33 }))).toBe(false);
  });

  it("accepts concurrency at the bounds (1 and 32)", () => {
    expect(rowsAreValid(rows({ concurrency: 1 }))).toBe(true);
    expect(rowsAreValid(rows({ concurrency: 32 }))).toBe(true);
  });

  it("rejects a missing model", () => {
    expect(rowsAreValid(rows({ model: "" }))).toBe(false);
  });

  it("does not itself check the total===100 rule (caller's job)", () => {
    // Two valid rows that individually satisfy bounds but sum to 60; this
    // predicate only checks per-row bounds, not the cross-row total.
    expect(
      rowsAreValid([
        { model: "a", percent: 30, concurrency: 1 },
        { model: "b", percent: 30, concurrency: 1 },
      ])
    ).toBe(true);
  });

  it("does not itself check for duplicate models (caller's job)", () => {
    expect(
      rowsAreValid([
        { model: "a", percent: 50, concurrency: 1 },
        { model: "a", percent: 50, concurrency: 1 },
      ])
    ).toBe(true);
  });

  it("happy path: single valid row", () => {
    expect(rowsAreValid(rows({ percent: 100, concurrency: 1 }))).toBe(true);
  });

  it("happy path: multiple valid rows summing to 100 with no duplicates", () => {
    const validRows: AllocationRow[] = [
      { model: "a", percent: 60, concurrency: 2 },
      { model: "b", percent: 40, concurrency: 1 },
    ];
    expect(rowsAreValid(validRows)).toBe(true);
    const total = validRows.reduce((sum, r) => sum + r.percent, 0);
    const duplicates = new Set(validRows.map((r) => r.model)).size !== validRows.length;
    expect(total === 100 && !duplicates).toBe(true);
  });
});
