import { describe, expect, it } from "vitest";
import * as three from "../../src/three/index.js";
import { computeRequiredLimits, createVolumeRenderer, nextPow2 } from "../../src/three/index.js";

/**
 * Guards the public barrel export surface and the couple of behaviors that
 * don't need a real grid or a GPU (see test/renderer.test.ts for
 * `createVolumeRenderer`).
 */

describe("@pmndrs/vdb/three public API surface", () => {
  it("exports the K2 adapter and drops the v1 binding trick (K2.3)", () => {
    for (const name of ["getDevice", "getGPUTexture", "VDBGridNode", "vdbSample", "vdbReadValue", "VDBVolumeMaterial", "vdbProxyGeometry", "VDBSequence"]) {
      expect(typeof (three as Record<string, unknown>)[name], name).toBe("function");
    }
    for (const name of ["NanoVDBGrid", "NanoVDBVolumeMaterial", "NanoVDBSequence", "assembleVolumeWgsl", "rewriteBufferGlobal", "samplerForGridType", "assertHasBufferGlobal", "DEFAULT_BUFFER_NAME"]) {
      expect(name in three, name).toBe(false);
    }
  });

  it("exports createVolumeRenderer, and the limits helpers", () => {
    expect(typeof createVolumeRenderer).toBe("function");
    expect(typeof computeRequiredLimits).toBe("function");
    expect(typeof nextPow2).toBe("function");
  });

  it("createVolumeRenderer rejects with a clear error when WebGPU is unavailable (Node)", async () => {
    await expect(createVolumeRenderer()).rejects.toThrowError(/WebGPU is unavailable/);
  });
});
