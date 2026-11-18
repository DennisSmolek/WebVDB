import { describe, expect, it } from "vitest";
import { Data3DTexture, FloatType, LinearFilter, RedFormat, UnsignedByteType } from "three";
import { VDBGridPool } from "../../src/core/index.js";
import { decodeToAtlas, gridStats, valueTransform } from "../../src/three/compute.js";
import * as three from "../../src/three/index.js";
import { computeFake } from "../core/fake-compute.js";
import { makeImage } from "../core/synthetic-image.js";

// K2.3: the three wrappers over the core compute tools. They take the renderer
// (getDevice) and keep v1's promise shape; the core tools' own contract is in
// test/core/compute.test.ts and the GPU parity in e2e/compute.spec.ts.

function fakeRenderer(device: GPUDevice | null) {
  return { backend: { isWebGPUBackend: true, device, get: () => ({}) } };
}

async function setup() {
  const { device } = computeFake();
  const pool = await VDBGridPool.create(device);
  const id = pool.add(makeImage({ gridType: 14, bboxMin: [0, 0, 0], bboxMax: [5, 3, 1], rootMin: 0, rootMax: 1 }));
  pool.flush();
  return { device, pool, id, renderer: fakeRenderer(device) };
}

describe("@pmndrs/vdb/three compute wrappers", () => {
  it("are exported from the barrel; the v1 shader-assembly helpers are gone", () => {
    expect(three.gridStats).toBe(gridStats);
    expect(three.decodeToAtlas).toBe(decodeToAtlas);
    expect(three.valueTransform).toBe(valueTransform);
    for (const gone of ["buildComputeShaderSource", "buildValueTransformShaderSource", "resolveTransformBody", "VALUE_TRANSFORM_PRESETS"]) {
      expect(gone in three).toBe(false);
    }
  });

  it("gridStats resolves through the renderer's device, and rejects (not throws) for an uninitialized renderer", async () => {
    const { pool, id, renderer } = await setup();
    const r = await gridStats(renderer, pool, id, { histogramBins: 8 });
    expect(r.histogram).toHaveLength(8);
    expect(r.range).toEqual({ min: 0, max: 1 });
    const pending = gridStats(fakeRenderer(null), pool, id);
    await expect(pending).rejects.toMatchObject({ code: "VDB_UNSUPPORTED_DEVICE" });
  });

  it("decodeToAtlas returns v1's result plus a ready Data3DTexture", async () => {
    const { pool, id, renderer } = await setup();
    const a = await decodeToAtlas(renderer, pool, id, { maxDim: 4 });
    expect(a.dims).toEqual([4, 4, 2]);
    expect(a.data).toBeInstanceOf(Uint8Array);
    expect(a.texture).toBeInstanceOf(Data3DTexture);
    expect(a.texture.image).toMatchObject({ data: a.data, width: 4, height: 4, depth: 2 });
    expect([a.texture.format, a.texture.type, a.texture.minFilter, a.texture.magFilter, a.texture.unpackAlignment]).toEqual([
      RedFormat,
      UnsignedByteType,
      LinearFilter,
      LinearFilter,
      1,
    ]);
    const f = await decodeToAtlas(renderer, pool, id, { format: "float32" });
    expect(f.data).toBeInstanceOf(Float32Array);
    expect(f.texture.type).toBe(FloatType);
  });

  it("valueTransform is the CPU path (no renderer): Float grids only", async () => {
    const { pool, id } = await setup();
    expect(() => valueTransform(pool, id, new Uint32Array(256), "double")).toThrow(/Float/);
  });
});
