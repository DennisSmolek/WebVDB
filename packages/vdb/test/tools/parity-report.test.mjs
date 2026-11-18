import { describe, expect, it } from "vitest";
import { overallPassed, summarizeParity } from "../../../../scripts/lib/parity-report.mjs";

const equal = {
  coordinate: [0, 0, 0],
  native: { value: 1, active: true },
  typescript: { value: 1, active: true },
  numericBound: 0,
};

describe("native parity report", () => {
  it("requires exact FLOAT decoded parity", () => {
    const report = summarizeParity(
      "sphere",
      "float",
      [{ ...equal, typescript: { value: 1 + Number.EPSILON, active: true } }],
      [],
    );
    expect(report.passed).toBe(false);
    expect(report.numericFailures).toBe(1);
    expect(report.invalidNumericBounds).toBe(0);
  });

  it("accepts FP8 error within the encoded leaf quantum", () => {
    const report = summarizeParity(
      "sphere",
      "fp8",
      [{ ...equal, typescript: { value: 1.01, active: true }, numericBound: 0.02 }],
      [],
    );
    expect(report.passed).toBe(true);
    expect(report.numericFailures).toBe(0);
  });

  it("rejects non-finite FP8 bounds", () => {
    const report = summarizeParity(
      "sphere",
      "fp8",
      [{ ...equal, typescript: { value: 100, active: true }, numericBound: Number.POSITIVE_INFINITY }],
      [],
    );
    expect(report.passed).toBe(false);
    expect(report.numericFailures).toBe(1);
    expect(report.invalidNumericBounds).toBe(1);
  });

  it("fails unclassified byte differences", () => {
    const report = summarizeParity("sphere", "float", [equal], [
      { classification: "unclassified-structural-byte-difference" },
    ]);
    expect(report.passed).toBe(false);
    expect(report.unclassifiedByteDifferences).toHaveLength(1);
  });

  it("reports classified semantic differences without silently failing the value gate", () => {
    const report = summarizeParity(
      "sphere",
      "float",
      [
        {
          ...equal,
          native: { value: 3, active: false },
          typescript: { value: 0, active: false },
          classification: "near-field/root-background",
        },
      ],
      ["active-tile-compression", "disabled-crc", "trailing-padding"],
    );
    expect(report.passed).toBe(true);
    expect(report.classifiedSemanticDifferences).toHaveLength(1);
    expect(report.byteDifferences).toContain("active-tile-compression");
  });

  it("fails the aggregate when any sample/mode fails", () => {
    const passing = summarizeParity("sphere", "float", [equal], []);
    const failing = summarizeParity(
      "cube",
      "float",
      [{ ...equal, native: { value: 1, active: false } }],
      [],
    );
    expect(overallPassed([passing])).toBe(true);
    expect(overallPassed([passing, failing])).toBe(false);
  });
});
