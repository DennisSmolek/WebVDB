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

// `parseVdb` graduated from stub to a real implementation in Phase 5 — see
// test/parse-vdb.test.ts. The rest of the v1 API (NanoVDB serializer,
// quantization, affine transform, inspect, .nvdb I/O) is still Phase 5b+.
describe("vdb-web-tools stubs (post-parser gate)", () => {
  it("every not-yet-implemented v1 API throws until its wave lands", () => {
    const buf = new ArrayBuffer(0);
    expect(() => buildFromVdb({})).toThrowError(/Phase 5/);
    expect(() => buildFromDense(new Float32Array(0), [0, 0, 0])).toThrowError(/Phase 5/);
    expect(() => quantize({}, "fp8")).toThrowError(/Phase 5/);
    expect(() => transform({}, new Float32Array(16))).toThrowError(/Phase 5/);
    expect(() => inspect({})).toThrowError(/Phase 5/);
    expect(() => readNvdb(buf)).toThrowError(/Phase 5/);
    expect(() => writeNvdb({})).toThrowError(/Phase 5/);
  });
});
