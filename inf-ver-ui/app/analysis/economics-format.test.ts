import { describe, expect, it } from "vitest";
import { formatPerToken, formatRatio, formatSeconds } from "./economics-format";

describe("formatPerToken", () => {
  it("formats busy-per-token in compact scientific form", () => {
    expect(formatPerToken(1.87e-4)).toBe("1.9e-4 s/tok");
    expect(formatPerToken(0.5)).toBe("5.0e-1 s/tok");
  });
  it("dashes on null", () => {
    expect(formatPerToken(null)).toBe("—");
  });
});

describe("formatSeconds", () => {
  it("rounds to 2dp with unit", () => {
    expect(formatSeconds(0.754)).toBe("0.75 s");
    expect(formatSeconds(null)).toBe("—");
  });
});

describe("formatRatio", () => {
  it("renders one-decimal multiplier", () => {
    expect(formatRatio(17.84)).toBe("17.8×");
  });
  it("dashes on null or non-finite", () => {
    expect(formatRatio(null)).toBe("—");
    expect(formatRatio(Infinity)).toBe("—");
  });
});
