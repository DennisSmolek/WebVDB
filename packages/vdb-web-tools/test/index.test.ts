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

// `parseVdb` (5a), `buildFromDense`/`writeNvdb` (5b) are real now — see
// test/parse-vdb.test.ts and test/build-from-dense.test.ts. The remaining
// v1 API (buildFromVdb, quantize, transform, inspect, readNvdb) is next.
describe("vdb-web-tools stubs (post-5a/5b gate)", () => {
  it("implemented APIs are exported as functions", () => {
    expect(typeof buildFromDense).toBe("function");
    expect(typeof writeNvdb).toBe("function");
    // Real behavior is covered by their own suites; this guards the barrel.
    expect(buildFromDense.length).toBeGreaterThanOrEqual(2);
  });

  it("every not-yet-implemented v1 API throws until its wave lands", () => {
    const buf = new ArrayBuffer(0);
    expect(() => buildFromVdb({})).toThrowError(/Phase 5/);
    expect(() => quantize({}, "fp8")).toThrowError(/Phase 5/);
    expect(() => transform({}, new Float32Array(16))).toThrowError(/Phase 5/);
    expect(() => inspect({})).toThrowError(/Phase 5/);
    expect(() => readNvdb(buf)).toThrowError(/Phase 5/);
  });
});
