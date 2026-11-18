import { describe, expect, it } from "vitest";
import {
  vdbBuildFromDense,
  buildFromVdb,
  vdbInspect,
  vdbQuantize,
  vdbTransform,
  writeNvdb,
} from "../../src/tools/index.js";

// The v1 API is real now (buildFromVdb, quantize, transform, inspect) — each has
// its own suite (build-from-vdb / quantize / inspect-transform tests). This file
// just guards the public barrel: everything is exported and callable.
describe("@pmndrs/vdb/tools public barrel", () => {
  it("exports the v1 API as functions", () => {
    for (const fn of [vdbBuildFromDense, writeNvdb, buildFromVdb, vdbQuantize, vdbTransform, vdbInspect]) {
      expect(typeof fn).toBe("function");
    }
  });

  it("the v1 API validates its inputs instead of silently no-op'ing", () => {
    // Not a NanoVDB image -> a clear throw (not a Phase-5 stub message).
    expect(() => vdbInspect(new Uint32Array(8))).toThrow();
    expect(() => vdbQuantize(new Uint32Array(8), "fp8")).toThrow();
    // buildFromVdb needs a grid with a transform.
    expect(() => buildFromVdb({} as never)).toThrow();
  });
});
