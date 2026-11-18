/**
 * `VDBVolumeMaterial` and `vdbProxyGeometry` (K2.3) in Node: construction on a
 * `VDBGridNode` over a stand-in renderer, the march entry's text (public `vdb*`
 * API only, no v1 binding trick), option validation, and the proxy box. The
 * shader itself runs on a GPU in e2e/demo-02, demo-05 and demo-07.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import { VDB_GPU_LAYOUT_VERSION, VDBError, VDBGridPool } from "../../src/core/index.js";
import {
  VDBGridNode,
  VDBVolumeMaterial,
  VDB_VOLUME_MAX_STEPS_CAP,
  VDB_VOLUME_SAMPLE_BUDGET_CAP,
  VDB_VOLUME_SHADOW_STEPS_CAP,
  vdbProxyGeometry,
} from "../../src/three/index.js";
import { FakeDevice } from "../core/fake-gpu.js";
import { makeImage } from "../core/synthetic-image.js";
import { fakeRenderer } from "./fake-three.js";

function fakePool(device: object, types: string[] = ["float"]) {
  return {
    device: device as GPUDevice,
    buffer: { label: "pool" } as unknown as GPUBuffer,
    version: 1,
    layoutVersion: VDB_GPU_LAYOUT_VERSION,
    types: new Set(types) as ReadonlySet<never>,
  };
}

const shared = fakeRenderer().renderer;

/** A node on the shared stand-in renderer (one device for every node). */
function setup(options: object = { helpers: ["rayClipSupport"] }) {
  const grids = new VDBGridNode(shared, fakePool(shared.backend.device), options as never);
  return { renderer: shared, grids };
}

const code = (fn: () => unknown): string | undefined => {
  try {
    fn();
  } catch (e) {
    return e instanceof VDBError ? e.code : `not a VDBError: ${String(e)}`;
  }
  return undefined;
};

describe("VDBVolumeMaterial (K2.3)", () => {
  it("constructs a transparent, depth-write-off, premultiplied BackSide NodeMaterial", () => {
    const { renderer, grids } = setup();
    const mat = new VDBVolumeMaterial(renderer, grids, 3);
    expect(mat).toBeInstanceOf(NodeMaterial);
    expect(mat.isVDBVolumeMaterial).toBe(true);
    expect(mat.transparent).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.depthTest).toBe(true);
    expect(mat.side).toBe(THREE.BackSide);
    expect(mat.premultipliedAlpha).toBe(true);
    expect(mat.toneMapped).toBe(false);
    expect(mat.fragmentNode).toBeTruthy();
    expect(mat.grids).toBe(grids);
    // A number id becomes a live uint uniform.
    expect(mat.gridId.value).toBe(3);
    expect(mat.gridId.nodeType).toBe("uint");
  });

  it("exposes the v1 uniform surface with v1's defaults; overrides flow through", () => {
    const { renderer, grids } = setup();
    const mat = new VDBVolumeMaterial(renderer, grids, 0);
    expect(mat.densityScale.value).toBe(40);
    expect(mat.stepSize.value).toBe(0.75);
    expect(mat.maxSteps.value).toBe(512);
    expect(mat.anisotropy.value).toBe(0.3);
    expect(mat.shadowSteps.value).toBe(12);
    expect(mat.shadowDensity.value).toBe(1);
    expect(mat.ambient.value).toBe(0.15);
    expect(mat.sunIntensity.value).toBe(3);
    expect(mat.jitter.value).toBe(1);
    expect(mat.sunDirection.value.length()).toBeCloseTo(1, 12);
    const mat2 = new VDBVolumeMaterial(renderer, grids, 0, { densityScale: 10, jitter: false, sunDirection: new THREE.Vector3(0, 2, 0) });
    expect(mat2.densityScale.value).toBe(10);
    expect(mat2.jitter.value).toBe(0);
    expect(mat2.sunDirection.value.toArray()).toEqual([0, 1, 0]);
  });

  it("marches with the public vdb* API only: no storage pointer, arrayLength or vendored names", () => {
    const { renderer, grids } = setup();
    const src = new VDBVolumeMaterial(renderer, grids, 0).entrySource;
    expect(src.match(/\bfn\s+\w+\s*\(/g)).toEqual(["fn vdbThreeVolumeMarch("]); // wgslFn parses the first fn
    for (const call of ["vdbWorldToIndex(", "vdbWorldToIndexDir(", "vdbRayClipSupport(", "vdbRayClip(", "vdbAccessorInit(", "vdbSample(&acc,", "vdbSample(&sacc,"]) {
      expect(src).toContain(call);
    }
    expect(src).not.toMatch(/ptr<storage|arrayLength|pnanovdb_|nanovdb_buffer|vdb_/);
    expect(src).toContain(`clamp(max_steps, 1.0, ${VDB_VOLUME_MAX_STEPS_CAP}.0)`);
    expect(src).toContain(`clamp(shadow_steps, 0.0, ${VDB_VOLUME_SHADOW_STEPS_CAP}.0)`);
    expect(src).toContain(`let budget = ${VDB_VOLUME_SAMPLE_BUDGET_CAP};`);
    // The first sample is never before t0 = max(entry, 0) (behind the eye), and stays on the lattice.
    expect(src).toContain("let t0 = max(span.x, 0.0);");
    expect(src).toMatch(/t \+= step_size \* jitter \* vdbThreeHash12\(pix\);\s*\/\/[^]*?if t < t0 \{\s*t \+= ceil\(\(t0 - t\) \/ step_size\) \* step_size;\s*\}/);
    // Every sample counts against the budget, and both loops stop on it.
    expect(src.match(/samples\+\+/g)).toHaveLength(2);
    expect(src.match(/samples >= budget/g)).toHaveLength(2);
  });

  it("compiles custom caps in", () => {
    const { renderer, grids } = setup();
    const src = new VDBVolumeMaterial(renderer, grids, 0, { maxStepsCap: 256, shadowStepsCap: 32, sampleBudgetCap: 4096 }).entrySource;
    expect(src).toContain("clamp(max_steps, 1.0, 256.0)");
    expect(src).toContain("clamp(shadow_steps, 0.0, 32.0)");
    expect(src).toContain("let budget = 4096;");
  });

  it("validates the node, the id and the caps", () => {
    const { renderer, grids } = setup();
    expect(code(() => new VDBVolumeMaterial(renderer, {} as never, 0))).toBe("VDB_BAD_OPTION");
    // The node must compile vdbSample and the support clip.
    const noHelper = setup({}).grids;
    expect(() => new VDBVolumeMaterial(renderer, noHelper, 0)).toThrow(/helpers: \['rayClipSupport'\]/);
    expect(code(() => new VDBVolumeMaterial(renderer, setup({ profile: "read", helpers: [] }).grids, 0))).toBe("VDB_BAD_OPTION");
    expect(() => new VDBVolumeMaterial(renderer, setup({ profile: "read" }).grids, 0)).toThrow(/profile 'sample' or 'march'/);
    expect(code(() => new VDBVolumeMaterial(renderer, setup({ profile: "march", helpers: ["rayClipSupport", "hddaCellEnd"] }).grids, 0))).toBeUndefined();
    // Another renderer, on another device or on the same one: one VDBGridNode per renderer.
    expect(code(() => new VDBVolumeMaterial(fakeRenderer().renderer, grids, 0))).toBe("VDB_BAD_OPTION");
    const sameDevice = fakeRenderer(shared.backend.device).renderer;
    expect(() => new VDBVolumeMaterial(sameDevice, grids, 0)).toThrow(/another renderer.*one VDBGridNode per renderer/);
    expect(code(() => new VDBVolumeMaterial(sameDevice, new VDBGridNode(sameDevice, fakePool(shared.backend.device), { helpers: ["rayClipSupport"] }), 0))).toBeUndefined();
    expect(code(() => new VDBVolumeMaterial({ backend: { isWebGPUBackend: true, device: null } } as never, grids, 0))).toBe("VDB_UNSUPPORTED_DEVICE");
    for (const id of [-1, 1.5, Number.NaN, "0", null]) {
      expect(code(() => new VDBVolumeMaterial(renderer, grids, id as never))).toBe("VDB_BAD_OPTION");
    }
    for (const caps of [{ maxStepsCap: 0 }, { shadowStepsCap: 1.5 }, { sampleBudgetCap: -1 }]) {
      expect(code(() => new VDBVolumeMaterial(renderer, grids, 0, caps))).toBe("VDB_BAD_OPTION");
    }
  });

  it("takes a uint node as the id, unchanged", () => {
    const { renderer, grids } = setup();
    const node = { isNode: true, tag: "my id" };
    expect(new VDBVolumeMaterial(renderer, grids, node).gridId).toBe(node);
  });
});

describe("VDBGridNode helpers option (K2.3)", () => {
  it("passes helpers to the module, normalized as a sorted set", () => {
    const { grids } = setup({ profile: "march", helpers: ["rayClipSupport", "cellBoundTrilinear", "rayClipSupport"] });
    expect(grids.options.helpers).toEqual(["cellBoundTrilinear", "rayClipSupport"]);
    const mod = grids.module as unknown as { moduleFor(ref: string): string };
    const text = mod.moduleFor("vdbPool.value");
    expect(text).toContain("fn vdbRayClipSupport(");
    expect(text).toContain("fn vdbCellBoundTrilinear(");
    expect(text).not.toContain("fn vdbHddaCellEnd(");
    // Without helpers the option is absent (and the module has none).
    expect("helpers" in setup({}).grids.options).toBe(false);
    // One-shot iterables survive validation; bad helpers are the generator's VDB_BAD_OPTION.
    expect(setup({ helpers: new Set(["rayClipSupport"]).values() }).grids.options.helpers).toEqual(["rayClipSupport"]);
    expect(code(() => setup({ helpers: ["nope"] }))).toBe("VDB_BAD_OPTION");
    expect(code(() => setup({ helpers: "rayClipSupport" }))).toBe("VDB_BAD_OPTION");
    expect(code(() => setup({ profile: "sample", helpers: ["hddaCellEnd"] }))).toBe("VDB_BAD_OPTION");
  });
});

describe("vdbProxyGeometry (K2.3)", () => {
  const boxOf = (g: THREE.BufferGeometry) => {
    g.computeBoundingBox();
    return g.boundingBox!;
  };

  it("pad 0 is pool.info(id).worldBBox; pad 1 adds the trilinear support shell", async () => {
    const device = new FakeDevice();
    const pool = await VDBGridPool.create(device.asGPU());
    const id = pool.add(makeImage({ bytes: 4096, voxelSize: 0.5, translation: [1, 2, 3], bboxMin: [-4, 0, 2], bboxMax: [7, 3, 9] }));
    const info = pool.info(id);
    const box = boxOf(vdbProxyGeometry(info));
    for (let a = 0; a < 3; a++) {
      expect(box.min.getComponent(a)).toBeCloseTo(info.worldBBox.min[a]!, 6);
      expect(box.max.getComponent(a)).toBeCloseTo(info.worldBBox.max[a]!, 6);
    }
    const padded = boxOf(vdbProxyGeometry(info, { pad: 1 }));
    for (let a = 0; a < 3; a++) {
      expect(padded.min.getComponent(a)).toBeCloseTo(info.worldBBox.min[a]! - 0.5, 6);
      expect(padded.max.getComponent(a)).toBeCloseTo(info.worldBBox.max[a]! + 0.5, 6);
    }
    expect(code(() => vdbProxyGeometry(info, { pad: -1 }))).toBe("VDB_BAD_OPTION");
  });

  it("bounds a rotated instance by its transformed corners", () => {
    // 90° about z: index (x, y, z) -> world (-y, x, z).
    const info = { indexBBox: { min: [0, 0, 0], max: [3, 1, 0] }, indexToWorld: [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0] };
    const box = boxOf(vdbProxyGeometry(info));
    expect(box.min.toArray()).toEqual([-2, 0, 0]);
    expect(box.max.toArray()).toEqual([0, 4, 1]);
  });
});
