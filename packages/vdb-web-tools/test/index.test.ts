import { describe, expect, it } from "vitest";
import {
  buildFromDense,
  buildFromVdb,
  inspect,
  parseVdb,
  quantize,
  readNvdb,
  transform,
  writeNvdb,
} from "../src/index.js";

describe("vdb-web-tools stubs (Phase 0 gate)", () => {
  it("every v1 API throws until Phase 5 lands", () => {
    const buf = new ArrayBuffer(0);
    expect(() => parseVdb(buf)).toThrowError(/Phase 5/);
    expect(() => buildFromVdb({})).toThrowError(/Phase 5/);
    expect(() => buildFromDense(new Float32Array(0), [0, 0, 0])).toThrowError(/Phase 5/);
    expect(() => quantize({}, "fp8")).toThrowError(/Phase 5/);
    expect(() => transform({}, new Float32Array(16))).toThrowError(/Phase 5/);
    expect(() => inspect({})).toThrowError(/Phase 5/);
    expect(() => readNvdb(buf)).toThrowError(/Phase 5/);
    expect(() => writeNvdb({})).toThrowError(/Phase 5/);
  });
});
