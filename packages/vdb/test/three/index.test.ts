import { describe, expect, it } from "vitest";
import * as three from "../../src/three/index.js";

/** Guards the public barrel export surface (D17: classes `VDB*`, functions `vdb*`). */

describe("@pmndrs/vdb/three public API surface", () => {
  it("exports the K2 adapter and drops the v1 binding trick (K2.3)", () => {
    for (const name of ["getDevice", "getGPUTexture", "VDBGridNode", "vdbSample", "vdbReadValue", "VDBVolumeMaterial", "vdbProxyGeometry", "VDBSequence", "vdbGridStats", "vdbDecodeToAtlas"]) {
      expect(typeof (three as Record<string, unknown>)[name], name).toBe("function");
    }
    for (const name of ["NanoVDBGrid", "NanoVDBVolumeMaterial", "NanoVDBSequence", "assembleVolumeWgsl", "rewriteBufferGlobal", "samplerForGridType", "assertHasBufferGlobal", "DEFAULT_BUFFER_NAME"]) {
      expect(name in three, name).toBe(false);
    }
  });

  it("has no v1 holdovers: no renderer bootstrap, limits helpers, or unprefixed compute wrappers (K2 review)", () => {
    for (const name of ["createVolumeRenderer", "computeRequiredLimits", "nextPow2", "gridStats", "decodeToAtlas", "valueTransform"]) {
      expect(name in three, name).toBe(false);
    }
  });

  it("every runtime export follows D17 (VDB* classes and constants, vdb* functions) or the pmndrs device helpers", () => {
    const pmndrsHelpers = new Set(["getDevice", "getGPUTexture"]);
    for (const name of Object.keys(three)) {
      expect(/^(VDB|vdb)/.test(name) || pmndrsHelpers.has(name), name).toBe(true);
    }
  });
});
