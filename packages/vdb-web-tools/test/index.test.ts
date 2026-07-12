import { describe, expect, it } from "vitest";
import {
  buildFromDense,
  buildFromVdb,
  inspect,
  quantize,
  readNvdb,
  transform,
  writeNvdb,
} from "../src/index.js";

// The v1 API is real now (buildFromVdb, quantize, transform, inspect) — each has
// its own suite (build-from-vdb / quantize / inspect-transform tests). This file
// just guards the public barrel: everything is exported and callable, and the one
// remaining stub (readNvdb) still throws with its Phase-5 pointer.
describe("vdb-web-tools public barrel", () => {
  it("exports the v1 API as functions", () => {
    for (const fn of [buildFromDense, writeNvdb, buildFromVdb, quantize, transform, inspect]) {
      expect(typeof fn).toBe("function");
    }
  });

  it("readNvdb is the only remaining stub", () => {
    expect(() => readNvdb(new ArrayBuffer(0))).toThrowError(/Phase 5/);
  });

  it("the v1 API validates its inputs instead of silently no-op'ing", () => {
    // Not a NanoVDB image -> a clear throw (not a Phase-5 stub message).
    expect(() => inspect(new Uint32Array(8))).toThrow();
    expect(() => quantize(new Uint32Array(8), "fp8")).toThrow();
    // buildFromVdb needs a grid with a transform.
    expect(() => buildFromVdb({} as never)).toThrow();
  });
});
