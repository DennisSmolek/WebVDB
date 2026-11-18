import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  NvdbFile,
  atlasDims,
  atlasFromRaw,
  bboxSize,
  vdbBuildFromDense,
  vdbDecodeToAtlasCpu,
  getDim,
  vdbGridStatsCpu,
  indexBBox,
  isActive,
  vdbReadValueCpu,
  vdbReadValueF32,
  rootStats,
  vdbSampleTrilinear,
  transformLeafValues,
} from "../../src/tools/internal.js";

// K2.3: CPU twins of the core compute tools. The GPU side is checked against
// these in e2e/compute.spec.ts (bit-exact on SwiftShader); here they are
// checked against independent references.

const dir = new URL("../../../../fixtures/primitives/", import.meta.url);
const has = (n: string) => existsSync(new URL(`${n}.nvdb`, dir));
const load = (n: string) => NvdbFile.fromArrayBuffer(new Uint8Array(readFileSync(new URL(`${n}.nvdb`, dir))).buffer).gridImage(0);
const sidecar = (n: string) => JSON.parse(readFileSync(new URL(`${n}.sidecar.json`, dir), "utf8")) as { grid: { activeVoxelCount: number } };
const FIXTURES = ["box_fog_float", "sphere_fog_fp8", "torus_fog_fpn"].filter(has);

function signedGrid(): Uint32Array {
  const dims: [number, number, number] = [12, 9, 7];
  const values = new Float32Array(dims[0] * dims[1] * dims[2]).map((_, i) => (i % 5 === 0 ? 0 : Math.sin(i * 0.37)));
  return vdbBuildFromDense(values, dims, { origin: [-3, 2, -5], gridClass: "Unknown" });
}

describe("bboxSize / atlasDims", () => {
  it("counts inclusive voxels and clamps each axis to maxDim (no proportional downscale)", () => {
    expect(bboxSize({ min: [-40, -20, -30], max: [40, 20, 30] })).toEqual([81, 41, 61]);
    expect(bboxSize({ min: [0, 0, 0], max: [0, 0, 0] })).toEqual([1, 1, 1]);
    expect(atlasDims({ min: [-40, -20, -30], max: [40, 20, 30] }, 256)).toEqual([81, 41, 61]);
    expect(atlasDims({ min: [-500, -20, -30], max: [500, 20, 30] }, 256)).toEqual([256, 41, 61]);
    expect(atlasDims({ min: [0, 0, 0], max: [1000, 1000, 1000] }, 128)).toEqual([128, 128, 128]);
  });
});

describe("atlasFromRaw (v1's post-readback step)", () => {
  const raw = new Float32Array([-1, 0, 0.25, 0.5, 1, 2, 300]);
  it("uint8 normalized maps [min, max] to [0, 255] with rounding and clamping", () => {
    expect(Array.from(atlasFromRaw(raw, { format: "uint8", normalize: true, range: { min: 0, max: 1 } }))).toEqual([0, 0, 64, 128, 255, 255, 255]);
  });
  it("uint8 raw rounds and clamps to [0, 255]", () => {
    expect(Array.from(atlasFromRaw(raw, { format: "uint8", normalize: false, range: { min: 0, max: 1 } }))).toEqual([0, 0, 0, 1, 1, 2, 255]);
  });
  it("float32 normalized maps to [0, 1]; raw returns the input itself", () => {
    expect(Array.from(atlasFromRaw(raw, { format: "float32", normalize: true, range: { min: 0, max: 2 } }))).toEqual([0, 0, 0.125, 0.25, 0.5, 1, 1]);
    expect(atlasFromRaw(raw, { format: "float32", normalize: false, range: { min: 0, max: 1 } })).toBe(raw);
  });
});

describe.skipIf(FIXTURES.length === 0)("vdbGridStatsCpu", () => {
  it.each(FIXTURES)("%s: count/min/max equal a plain f64 scan, mean close, histogram sums to the count", (name) => {
    const image = load(name);
    const s = vdbGridStatsCpu(image);
    const { min: lo, max: hi } = indexBBox(image);
    let count = 0;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (let x = lo[0]; x <= hi[0]; x++) {
      for (let y = lo[1]; y <= hi[1]; y++) {
        for (let z = lo[2]; z <= hi[2]; z++) {
          const r = vdbReadValueCpu(image, [x, y, z]);
          if (!r.active) continue;
          count++;
          sum += r.value;
          min = Math.min(min, r.value);
          max = Math.max(max, r.value);
        }
      }
    }
    expect(s.usedAtomicFastPath).toBe(true);
    expect(s.activeVoxelCount).toBe(count);
    expect(s.activeVoxelCount).toBe(sidecar(name).grid.activeVoxelCount);
    expect(s.min).toBe(Math.fround(min));
    expect(s.max).toBe(Math.fround(max));
    // Fixed-point mean: within 2^-20 of the range per voxel on average.
    expect(Math.abs(s.mean - sum / count)).toBeLessThan((s.range.max - s.range.min) * 2 ** -19);
    expect(s.histogram.reduce((a, b) => a + b, 0)).toBe(count);
    expect(s.range).toEqual(rootStats(image));
    expect(s.range.min).toBeLessThanOrEqual(s.min);
    expect(s.range.max).toBeGreaterThanOrEqual(s.max);
  });

  it("a grid with negative values takes the dump path (f64 mean) and honours histogramBins", () => {
    const image = signedGrid();
    const s = vdbGridStatsCpu(image, { histogramBins: 16 });
    expect(rootStats(image).min).toBeLessThan(0);
    expect(s.usedAtomicFastPath).toBe(false);
    expect(s.histogram).toHaveLength(16);
    expect(s.histogram.reduce((a, b) => a + b, 0)).toBe(s.activeVoxelCount);
    expect(s.min).toBeLessThan(0);
    expect(s.max).toBeGreaterThan(0);
  });
});

describe.skipIf(FIXTURES.length === 0)("vdbDecodeToAtlasCpu", () => {
  it.each(FIXTURES)("%s: nearest reads voxel floor(min + i + 1) at full resolution (v1's cell-centre convention)", (name) => {
    const image = load(name);
    const { raw, dims } = vdbDecodeToAtlasCpu(image, { filter: "nearest", maxDim: 1 << 20 });
    const { min } = indexBBox(image);
    expect(dims).toEqual(bboxSize(indexBBox(image)));
    // pos = min + (i + 0.5) / n * n = min + i + 0.5, so floor(pos + 0.5) = min + i + 1.
    for (const i of [0, 7, 1000, raw.length - 1]) {
      const x = i % dims[0];
      const y = Math.floor(i / dims[0]) % dims[1];
      const z = Math.floor(i / (dims[0] * dims[1]));
      expect(raw[i]).toBe(vdbReadValueF32(image, [min[0] + x + 1, min[1] + y + 1, min[2] + z + 1]));
    }
  });

  it.each(FIXTURES)("%s: trilinear agrees with the f64 vdbSampleTrilinear twin to f32 rounding", (name) => {
    const image = load(name);
    const { raw, dims } = vdbDecodeToAtlasCpu(image, { maxDim: 24 });
    const bbox = indexBBox(image);
    const size = bboxSize(bbox);
    let maxAbs = 0;
    for (let i = 0; i < raw.length; i += 7) {
      const g = [i % dims[0], Math.floor(i / dims[0]) % dims[1], Math.floor(i / (dims[0] * dims[1]))];
      const p = [0, 1, 2].map((a) => bbox.min[a]! + ((g[a]! + 0.5) / dims[a]!) * size[a]!) as [number, number, number];
      maxAbs = Math.max(maxAbs, Math.abs(raw[i]! - vdbSampleTrilinear(image, p)));
    }
    expect(maxAbs).toBeLessThan(1e-4);
  });
});

describe.skipIf(!has("box_fog_float"))("transformLeafValues", () => {
  it("rewrites leaf values inside the bbox only; tiles, stats and the source stay as they were", () => {
    const image = load("box_fog_float");
    const before = image.slice();
    const out = transformLeafValues(image, (v) => v * 2 + 1);
    expect(image).toEqual(before);
    const { min, max } = indexBBox(image);
    let leaves = 0;
    let tiles = 0;
    for (let n = 0; n < 5000; n++) {
      const ijk = [0, 1, 2].map((a) => min[a]! + ((n * [7919, 104729, 1299709][a]!) % (max[a]! - min[a]! + 1))) as [number, number, number];
      if (getDim(image, ijk) === 1) {
        leaves++;
        expect(vdbReadValueF32(out, ijk)).toBe(Math.fround(vdbReadValueF32(image, ijk) * 2 + 1));
      } else {
        tiles++;
        expect(vdbReadValueF32(out, ijk)).toBe(vdbReadValueF32(image, ijk));
      }
      expect(isActive(out, ijk)).toBe(isActive(image, ijk));
    }
    expect(leaves).toBeGreaterThan(0);
    expect(tiles).toBeGreaterThan(0);
    expect(rootStats(out)).toEqual(rootStats(image)); // stats are not recomputed (documented)
    // Only leaf value words changed: everything before the first leaf is byte-identical.
    let firstDiff = 0;
    while (firstDiff < out.length && out[firstDiff] === image[firstDiff]) firstDiff++;
    expect(firstDiff).toBeGreaterThan(200);
  });

  it("refuses quantized grids and non-functions", () => {
    if (has("sphere_fog_fp8")) expect(() => transformLeafValues(load("sphere_fog_fp8"), (v) => v)).toThrow(/not Float/);
    expect(() => transformLeafValues(load("box_fog_float"), "double" as never)).toThrow(/function/);
  });
});
