import { describe, expect, it } from "vitest";
import * as three from "../../src/three/index.js";
import { NanoVDBGrid, computeRequiredLimits, createVolumeRenderer, nextPow2 } from "../../src/three/index.js";

/**
 * Phase 3 gate: `NanoVDBGrid` and `createVolumeRenderer` are real now (SPEC
 * §3.1/§3.4) — see test/grid.test.ts and test/renderer.test.ts for the
 * substantive coverage. This file just guards the public barrel export
 * surface and the couple of behaviors that don't need a real grid or a GPU.
 */

describe("@pmndrs/vdb/three public API surface", () => {
  it("exports the K2 adapter and drops the v1 binding trick (K2.3)", () => {
    for (const name of ["getDevice", "getGPUTexture", "VDBGridNode", "vdbSample", "vdbReadValue", "VDBVolumeMaterial", "vdbProxyGeometry", "VDBSequence"]) {
      expect(typeof (three as Record<string, unknown>)[name], name).toBe("function");
    }
    for (const name of ["NanoVDBSequence", "assembleVolumeWgsl", "rewriteBufferGlobal", "samplerForGridType", "assertHasBufferGlobal", "DEFAULT_BUFFER_NAME"]) {
      expect(name in three, name).toBe(false);
    }
  });

  it("exports NanoVDBGrid, createVolumeRenderer, and the limits helpers", () => {
    expect(typeof NanoVDBGrid).toBe("function");
    expect(typeof createVolumeRenderer).toBe("function");
    expect(typeof computeRequiredLimits).toBe("function");
    expect(typeof nextPow2).toBe("function");
  });

  it("NanoVDBGrid rejects an unsupported grid type instead of guessing", () => {
    const metadata = {
      name: "bogus",
      gridType: "Vec3f",
      gridClass: "Unknown",
      worldBBox: { min: [0, 0, 0], max: [1, 1, 1] },
      indexBBox: { min: [0, 0, 0], max: [1, 1, 1] },
      voxelSize: [1, 1, 1],
      voxelCount: 0,
      gridByteSize: 0,
    } as const;
    expect(() => new NanoVDBGrid({ image: new Uint32Array(0), metadata })).toThrowError(/unsupported grid type/);
  });

  it("createVolumeRenderer rejects with a clear error when WebGPU is unavailable (Node)", async () => {
    await expect(createVolumeRenderer()).rejects.toThrowError(/WebGPU is unavailable/);
  });
});
