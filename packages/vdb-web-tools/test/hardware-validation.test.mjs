import { describe, expect, it } from "vitest";
import {
  exactStateValidator,
  validateExactState,
} from "../../../scripts/lib/hardware-validation.mjs";

describe("hardware report state validation", () => {
  const expected = {
    fixture: "sphere_fog_fp8.nvdb",
    adapter: "apple metal-3",
    presentation: "direct",
    resolution: "640x480",
    stepSize: 0.45,
    shadowSteps: 16,
    sampleBudgetCap: 8192,
    swizzleShim: false,
    keepAliveShim: false,
  };

  it("accepts an exact state", () => {
    expect(validateExactState(expected, { ready: true, ...expected })).toEqual([]);
  });

  it("reports every mismatched or missing field", () => {
    const errors = exactStateValidator(expected)({
      fixture: "wrong.nvdb",
      adapter: "apple metal-3",
      presentation: "readback",
    });
    expect(errors).toContain('fixture: expected "sphere_fog_fp8.nvdb", got "wrong.nvdb"');
    expect(errors).toContain('presentation: expected "direct", got "readback"');
    expect(errors.some((error) => error.startsWith("resolution:"))).toBe(true);
    expect(errors).toHaveLength(8);
  });
});
