/**
 * K3.7 `VDBLevelSetMaterial` and `VDBPoints` in Node: construction, option
 * validation and the trace entry's text (public `vdb*` API only). The shaders
 * run on a GPU in e2e/k3-levelset.spec.ts.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { MeshStandardNodeMaterial, PointsNodeMaterial } from "three/webgpu";
import { VDB_GPU_LAYOUT_VERSION, VDBError } from "../../src/core/index.js";
import { VDB_LEVEL_SET_SAMPLES_CAP, VDBGridNode, VDBLevelSetMaterial, VDBPoints, vdbAttenuatedPointSize, vdbPointDiscTexture } from "../../src/three/internal.js";
import { fakeRenderer } from "./fake-three.js";

const shared = fakeRenderer().renderer;

function pool(info?: (id: number) => { voxelSize: number[] }) {
  return {
    device: shared.backend.device as unknown as GPUDevice,
    buffer: { label: "pool" } as unknown as GPUBuffer,
    version: 1,
    layoutVersion: VDB_GPU_LAYOUT_VERSION,
    types: new Set(["float"]) as ReadonlySet<never>,
    ...(info ? { info } : {}),
  };
}

const code = (fn: () => unknown): string | undefined => {
  try {
    fn();
  } catch (e) {
    return e instanceof VDBError ? e.code : `not a VDBError: ${String(e)}`;
  }
  return undefined;
};

describe("VDBLevelSetMaterial (K3.7)", () => {
  it("is a BackSide MeshStandardNodeMaterial with a mask, depth, normal and the trace as nodes", () => {
    const grids = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["levelSet"] });
    const m = new VDBLevelSetMaterial(shared, grids, 2, { color: 0xff0000, roughness: 0.25 });
    expect(m).toBeInstanceOf(MeshStandardNodeMaterial);
    expect(m.isVDBLevelSetMaterial).toBe(true);
    expect(m.side).toBe(THREE.BackSide);
    expect(m.shadowSide).toBe(THREE.BackSide);
    expect(m.color.getHex()).toBe(0xff0000);
    expect(m.roughness).toBe(0.25);
    expect(m.gridId.value).toBe(2);
    expect(m.maskNode).toBe(m.hitMaskNode);
    expect(m.normalNode).toBe(m.normalViewNode);
    expect(m.depthNode).not.toBeNull();
    expect(m.maxSamples.value).toBe(512);
    expect(m.normalStep.value).toBe(1);
    // No pool info: one index voxel per SDF unit.
    expect(m.sdfToIndex.value).toBe(1);
    // The entry only calls the public API and compiles the sample cap in.
    expect(m.traceSource).toContain("vdbLevelSetHit(grid_id, o, d, &span, sdf_to_index, n)");
    expect(m.traceSource).toContain(`clamp(max_samples, 1.0, ${VDB_LEVEL_SET_SAMPLES_CAP}.0)`);
    expect(m.traceSource).not.toMatch(/pnanovdb|vdb_[a-z]/);
  });

  it("takes sdfToIndex from the pool's voxel size, or the parameter", () => {
    const grids = new VDBGridNode(shared, pool(() => ({ voxelSize: [0.05, 0.05, 0.05] })), { profile: "march", helpers: ["levelSet"] });
    expect(new VDBLevelSetMaterial(shared, grids, 0).sdfToIndex.value).toBeCloseTo(20, 12);
    expect(new VDBLevelSetMaterial(shared, grids, 0, { sdfToIndex: 4 }).sdfToIndex.value).toBe(4);
    expect(new VDBLevelSetMaterial(shared, grids, 0, { maxSamplesCap: 64 }).traceSource).toContain("clamp(max_samples, 1.0, 64.0)");
  });

  it("rejects a VDBGridNode without march + levelSet, bad ids and bad options", () => {
    const sample = new VDBGridNode(shared, pool(), { profile: "sample" });
    const march = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["hddaCellEnd"] });
    const ok = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["levelSet"] });
    expect(code(() => new VDBLevelSetMaterial(shared, sample, 0))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, march, 0))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, {} as never, 0))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, ok, -1))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, ok, 0, { sdfToIndex: 0 }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, ok, 0, { maxSamplesCap: 0 }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(fakeRenderer().renderer, ok, 0))).toBe("VDB_BAD_OPTION");
  });
});

describe("VDBPoints (K3.7)", () => {
  const positions = Float32Array.from([0, 0, 0, 1, 2, 3, -1, 0, 4]);

  it("draws count instanced sprites with the disc, Cd times the colour, half the spacing as diameter", () => {
    const colors = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const p = new VDBPoints({ positions, colors, spacing: 0.4 }, { color: 0x808080 });
    expect(p).toBeInstanceOf(THREE.Object3D);
    expect(p.isVDBPoints).toBe(true);
    expect(p.material).toBeInstanceOf(PointsNodeMaterial);
    expect(p.material.alphaMap).toBe(vdbPointDiscTexture());
    expect(p.material.alphaTest).toBe(0.5);
    expect((p as unknown as { count: number }).count).toBe(3);
    expect(p.pointCount).toBe(3);
    expect(p.positionAttribute.array).toBe(positions);
    expect(p.colorAttribute!.array).toBe(colors);
    expect(p.diameter).toBeCloseTo(0.2, 12);
    expect(p.frustumCulled).toBe(false);
    expect(new VDBPoints({ positions }, { size: 3 }).diameter).toBe(3);
    expect(new VDBPoints({ positions, count: 2 }).pointCount).toBe(2);
    expect(new VDBPoints({ positions }).colorAttribute).toBeNull();
  });

  it("sizes in pixels as his getAttenuatedPointSize, from the render target height when one is set", () => {
    const p = new VDBPoints({ positions, spacing: 2 });
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
    p.updateMatrixWorld(true);
    const perUnit = vdbAttenuatedPointSize(p, camera, 1);
    expect(perUnit).toBeCloseTo(1 / Math.tan(Math.PI / 6), 12);
    p.updatePixelScale({ getSize: (v) => v.set(640, 480), getRenderTarget: () => ({ height: 200 }) }, camera);
    expect(p.pixelScale.value).toBeCloseTo(perUnit * 1 * 200 * 0.5, 9);
    p.updatePixelScale({ getSize: (v) => v.set(640, 480), getRenderTarget: () => null }, camera);
    expect(p.pixelScale.value).toBeCloseTo(perUnit * 1 * 480 * 0.5, 9);
  });

  it("rejects malformed data", () => {
    expect(code(() => new VDBPoints({ positions: new Float32Array(4) }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBPoints({ positions, count: 4 }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBPoints({ positions, colors: new Float32Array(3) }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBPoints({} as never))).toBe("VDB_BAD_OPTION");
  });
});
