/**
 * K4.4 resample-to-resolution bake: `vdbResampleGrid` (point / box / trilinear)
 * and the comparison report's error metric `vdbResampleError`.
 */
import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  vdbBuildFromDense,
  buildFromVdb,
  leafVoxels,
  parseVdb,
  vdbQuantize,
  vdbReadValueCpu,
  vdbResampleError,
  vdbResampleGrid,
  sampleWvdbDense,
  treeNodes,
  writeNvdb,
  readWvdb,
  vdbToWvdb,
  wvdbDenseToNvdb,
  wvdbDenseVolume,
  wvdbSourceFromVdbGrid,
  type VdbGrid,
} from "../../src/tools/internal.js";
import {
  GRID_OFF_GRID_TYPE,
  GRID_OFF_MAP,
  GRID_OFF_VOXEL_SIZE,
  GRID_OFF_WORLD_BBOX,
  GRID_TYPE_FP8,
  GRID_TYPE_FPN,
  MAP_OFF_MATD,
  MAP_OFF_VECD,
} from "../../src/tools/nanovdb/bytes.js";
import { vdbSample } from "./wvdb-helpers.js";

type Vec3 = [number, number, number];

function f64(image: Uint32Array, off: number): number {
  return new DataView(image.buffer, image.byteOffset, image.byteLength).getFloat64(off, true);
}
function mapOf(image: Uint32Array): { scale: Vec3; vec: Vec3 } {
  return {
    scale: [0, 4, 8].map((i) => f64(image, GRID_OFF_MAP + MAP_OFF_MATD + i * 8)) as Vec3,
    vec: [0, 1, 2].map((i) => f64(image, GRID_OFF_MAP + MAP_OFF_VECD + i * 8)) as Vec3,
  };
}
function leafCount(image: Uint32Array): number {
  return treeNodes(image).filter((n) => n.level === 0).length;
}

/** Deterministic integer field (dyadic averages stay exact). */
function denseGrid(dims: Vec3, value: (x: number, y: number, z: number) => number): Float32Array {
  const v = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let x = 0; x < dims[0]; x++)
    for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) v[(x * dims[1] + y) * dims[2] + z] = value(x, y, z);
  return v;
}

const hash = (x: number, y: number, z: number): number => ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) >>> 0;

/** A minimal float `VdbGrid` over a dense block (exercises the `.vdb` source path without a file). */
function fakeVdbGrid(values: Float32Array, dims: Vec3, voxelSize: number, translation: Vec3): VdbGrid {
  const at = (x: number, y: number, z: number): number =>
    x < 0 || y < 0 || z < 0 || x >= dims[0] || y >= dims[1] || z >= dims[2] ? 0 : values[(x * dims[1] + y) * dims[2] + z]!;
  const grid = {
    name: "fake",
    gridType: "Tree_float_5_4_3",
    valueType: "float",
    background: 0,
    transform: {
      type: "UniformScaleTranslateMap",
      matrix: [voxelSize, 0, 0, translation[0], 0, voxelSize, 0, translation[1], 0, 0, voxelSize, translation[2], 0, 0, 0, 1],
      voxelSize: [voxelSize, voxelSize, voxelSize],
    },
    metadata: { class: "fog volume" },
    indexBBox: null,
    readValue: (ijk: Vec3) => ({ value: at(...ijk), active: at(...ijk) !== 0 }),
    *iterLeaves() {
      for (let ox = 0; ox < dims[0]; ox += 8)
        for (let oy = 0; oy < dims[1]; oy += 8)
          for (let oz = 0; oz < dims[2]; oz += 8) {
            const v = new Float32Array(512);
            const mask = new Uint32Array(16);
            for (let n = 0; n < 512; n++) {
              v[n] = at(ox + (n >> 6), oy + ((n >> 3) & 7), oz + (n & 7));
              if (v[n] !== 0) mask[n >>> 5]! |= 1 << (n & 31);
            }
            if (mask.some((w) => w !== 0)) yield { origin: [ox, oy, oz] as Vec3, values: v, valueMask: mask };
          }
    },
    *iterTiles() {},
  };
  return grid as unknown as VdbGrid;
}

describe("vdbResampleGrid: identity at the source resolution", () => {
  const dims: Vec3 = [21, 13, 10];
  // Sparse: a third of the voxels are background (inactive).
  const values = denseGrid(dims, (x, y, z) => (hash(x, y, z) % 3 === 0 ? 0 : 1 + (hash(x, y, z) % 97) / 8));
  const image = vdbBuildFromDense(values, dims, { origin: [5, -3, 2], voxelSize: 0.25, worldOrigin: [10, 20, -7] });
  const activeCount = values.filter((v) => v !== 0).length;

  for (const filter of ["point", "box", "trilinear"] as const) {
    it(`${filter}: reproduces every value and the active set exactly`, () => {
      const r = vdbResampleGrid(image, { filter });
      expect(r.voxelSizeXYZ).toEqual([0.25, 0.25, 0.25]);
      expect(r.voxelCount).toBe(activeCount);
      const { scale, vec } = mapOf(r.image);
      for (let x = 0; x < dims[0]; x++)
        for (let y = 0; y < dims[1]; y++)
          for (let z = 0; z < dims[2]; z++) {
            const src = vdbReadValueCpu(image, [5 + x, -3 + y, 2 + z]);
            // source world centre -> target index
            const w = [10 + x * 0.25, 20 + y * 0.25, -7 + z * 0.25];
            const ijk = [0, 1, 2].map((a) => Math.round((w[a]! - vec[a]!) / scale[a]!)) as Vec3;
            const dst = vdbReadValueCpu(r.image, ijk);
            expect(dst.active).toBe(src.active);
            if (src.active) expect(dst.value).toBe(src.value);
          }
    });
  }

  it("keeps the world box: target voxel 0 sits on the first source voxel", () => {
    const r = vdbResampleGrid(image, { filter: "box" });
    expect(r.dimensions).toEqual(dims);
    const { vec } = mapOf(r.image);
    // first active source voxel along each axis is index offset 0 here (hash leaves corners active or not; use box)
    expect(r.bbox[0]).toEqual([10 - 0.125, 20 - 0.125, -7 - 0.125].map((v, a) => v + firstActive(a) * 0.25));
    for (let a = 0; a < 3; a++) expect(vec[a]).toBeCloseTo(r.bbox[0][a]! + 0.125, 12);
  });

  function firstActive(axis: number): number {
    let m = Infinity;
    for (let x = 0; x < dims[0]; x++)
      for (let y = 0; y < dims[1]; y++)
        for (let z = 0; z < dims[2]; z++) if (values[(x * dims[1] + y) * dims[2] + z] !== 0) m = Math.min(m, [x, y, z][axis]!);
    return m;
  }
});

describe("vdbResampleGrid: downsample by 2", () => {
  const dims: Vec3 = [16, 16, 16];
  // Integer values; an empty (background) octant so sparsity is exercised.
  const values = denseGrid(dims, (x, y, z) => (x >= 8 && y >= 8 && z >= 8 ? 0 : 1 + (hash(x, y, z) % 50)));
  const image = vdbBuildFromDense(values, dims, { voxelSize: 1 });
  const src = (x: number, y: number, z: number): number => values[(x * 16 + y) * 16 + z]!;

  it("box: each target voxel is the exact mean of its 2x2x2 block; the empty octant stays empty", () => {
    const r = vdbResampleGrid(image, { resolution: 8, filter: "box" });
    expect(r.dimensions).toEqual([8, 8, 8]);
    expect(r.voxelSizeXYZ).toEqual([2, 2, 2]);
    let active = 0;
    for (let x = 0; x < 8; x++)
      for (let y = 0; y < 8; y++)
        for (let z = 0; z < 8; z++) {
          let sum = 0;
          for (let d = 0; d < 8; d++) sum += src(2 * x + (d & 1), 2 * y + ((d >> 1) & 1), 2 * z + (d >> 2));
          const got = vdbReadValueCpu(r.image, [x, y, z]);
          const empty = x >= 4 && y >= 4 && z >= 4;
          expect(got.active).toBe(!empty);
          expect(got.value).toBe(Math.fround(sum / 8));
          if (got.active) active++;
        }
    expect(active).toBe(512 - 64);
  });

  it("point: picks one voxel of each block (no prefilter); trilinear: blends at the block centre", () => {
    const p = vdbResampleGrid(image, { resolution: 8, filter: "point" });
    const t = vdbResampleGrid(image, { resolution: 8, filter: "trilinear" });
    for (let x = 0; x < 4; x++)
      for (let y = 0; y < 4; y++)
        for (let z = 0; z < 4; z++) {
          // centre of target voxel j is source index 2j + 0.5: Math.round -> 2j + 1, trilinear = 2x2x2 mean
          expect(vdbReadValueCpu(p.image, [x, y, z]).value).toBe(src(2 * x + 1, 2 * y + 1, 2 * z + 1));
          let sum = 0;
          for (let d = 0; d < 8; d++) sum += src(2 * x + (d & 1), 2 * y + ((d >> 1) & 1), 2 * z + (d >> 2));
          expect(vdbReadValueCpu(t.image, [x, y, z]).value).toBe(Math.fround(sum / 8));
        }
  });
});

describe("vdbResampleGrid: transform, filters and sparsity", () => {
  it("trilinear reproduces a linear field at any ratio (interior voxels)", () => {
    const dims: Vec3 = [24, 24, 24];
    const lin = (x: number, y: number, z: number): number => 1 + x + 2 * y + 3 * z;
    const image = vdbBuildFromDense(denseGrid(dims, lin), dims, { voxelSize: 0.5, worldOrigin: [1, 2, 3] });
    const r = vdbResampleGrid(image, { voxelSize: 0.85, filter: "trilinear" });
    const { scale, vec } = mapOf(r.image);
    for (let j = 1; j < r.dimensions[0] - 2; j++) {
      const ijk: Vec3 = [j, j + 1 < r.dimensions[1] - 1 ? j + 1 : j, j];
      const w = [0, 1, 2].map((a) => vec[a]! + scale[a]! * ijk[a]!);
      const s = [0, 1, 2].map((a) => (w[a]! - [1, 2, 3][a]!) / 0.5);
      expect(vdbReadValueCpu(r.image, ijk).value).toBeCloseTo(lin(s[0]!, s[1]!, s[2]!), 3);
    }
  });

  it("box conserves mass at any ratio, also at negative index coordinates", () => {
    const dims: Vec3 = [23, 17, 19];
    const values = denseGrid(dims, (x, y, z) => (hash(x, y, z) % 5 === 0 ? 0 : (hash(x, y, z) % 64) / 16));
    const image = vdbBuildFromDense(values, dims, { origin: [-40, -9, 3], voxelSize: 0.5, worldOrigin: [-20, -4.5, 1.5] });
    const total = (img: Uint32Array): number => {
      let s = 0;
      for (const n of treeNodes(img)) if (n.level === 0) for (const v of leafVoxels(img, n.address).values) s += v;
      return s;
    };
    const source = total(image) * 0.125;
    for (const voxelSize of [0.7, 1, 1.9, 4.3]) {
      const r = vdbResampleGrid(image, { voxelSize, filter: "box" });
      expect(total(r.image) * voxelSize ** 3).toBeCloseTo(source, 2);
    }
  });

  it("dimensions keep the box exactly with anisotropic voxels; the Map, GridData and quantize agree", () => {
    const dims: Vec3 = [30, 12, 20];
    const image = vdbBuildFromDense(denseGrid(dims, (x) => 1 + (x % 5)), dims, { voxelSize: 1 });
    const bbox: [Vec3, Vec3] = [[-0.5, -0.5, -0.5], [29.5, 11.5, 19.5]];
    const r = vdbResampleGrid(image, { dimensions: [10, 6, 8], bbox, quantize: "fp8" });
    expect(r.bbox).toEqual(bbox);
    const vs: Vec3 = [3, 2, 2.5];
    expect(r.voxelSizeXYZ).toEqual(vs);
    const { scale, vec } = mapOf(r.image);
    expect(scale).toEqual(vs);
    expect(vec).toEqual([1, 0.5, 0.75]); // box min + vs / 2
    expect([0, 1, 2].map((a) => f64(r.image, GRID_OFF_VOXEL_SIZE + a * 8))).toEqual(vs);
    // NanoVDB worldBBox convention: [imin * vs + t, (imax + 1) * vs + t]
    expect([0, 1, 2].map((a) => f64(r.image, GRID_OFF_WORLD_BBOX + a * 8))).toEqual(vec);
    expect([0, 1, 2].map((a) => f64(r.image, GRID_OFF_WORLD_BBOX + 24 + a * 8))).toEqual([31, 12.5, 20.75]);
    expect(new Uint32Array(r.image.buffer, r.image.byteOffset + GRID_OFF_GRID_TYPE, 1)[0]).toBe(GRID_TYPE_FP8);
    // x: 3 source voxels per target voxel, values 1 + (x % 5) -> box mean of x = 3j..3j+2
    for (let j = 0; j < 10; j++) {
      const mean = ((1 + ((3 * j) % 5)) + (1 + ((3 * j + 1) % 5)) + (1 + ((3 * j + 2) % 5))) / 3;
      expect(vdbReadValueCpu(r.image, [j, 2, 3]).value).toBeCloseTo(mean, 1); // fp8 code step = (max - min) / 255
    }
  });

  it("only touches leaves near data, and cutoff drops faint voxels", () => {
    // Two 8^3 blobs 4000 voxels apart: the target keeps two islands, nothing in between.
    const leaf = (origin: Vec3, v: number) => ({ origin, v });
    const blobs = [leaf([0, 0, 0], 2), leaf([4000, 8, 16], 0.01)];
    const dims: Vec3 = [4008, 24, 24];
    // Build through the fake .vdb path: dense would be 2.3M voxels, still fine but iterLeaves is the point.
    const values = new Float32Array(dims[0] * dims[1] * dims[2]);
    for (const b of blobs)
      for (let n = 0; n < 512; n++)
        values[((b.origin[0] + (n >> 6)) * dims[1] + b.origin[1] + ((n >> 3) & 7)) * dims[2] + b.origin[2] + (n & 7)] = b.v;
    const grid = fakeVdbGrid(values, dims, 1, [0, 0, 0]);
    const r = vdbResampleGrid(grid, { voxelSize: 2, filter: "box" });
    expect(leafCount(r.image)).toBeLessThanOrEqual(16); // each 4^3 target island straddles at most 2^3 leaves
    expect(r.voxelCount).toBe(2 * 64);
    const cut = vdbResampleGrid(grid, { voxelSize: 2, filter: "box", cutoff: 0.1 });
    expect(cut.voxelCount).toBe(64);
  });

  it("the .vdb source path equals the NanoVDB source path", () => {
    const dims: Vec3 = [19, 11, 14];
    const values = denseGrid(dims, (x, y, z) => (hash(x, y, z) % 4 === 0 ? 0 : (hash(x, y, z) % 255) / 16));
    const grid = fakeVdbGrid(values, dims, 0.3, [1, -2, 0.5]);
    const image = buildFromVdb(grid);
    for (const filter of ["point", "box", "trilinear"] as const) {
      const a = vdbResampleGrid(grid, { resolution: 7, filter });
      const b = vdbResampleGrid(image, { resolution: 7, filter });
      expect(a.image).toEqual(b.image);
    }
  });

  it("rejects conflicting resolution options and unknown filters", () => {
    const image = vdbBuildFromDense(new Float32Array([1]), [1, 1, 1]);
    expect(() => vdbResampleGrid(image, { resolution: 4, voxelSize: 1 })).toThrow(/at most one/);
    expect(() => vdbResampleGrid(image, { filter: "lanczos" as never })).toThrow(/unknown filter/);
  });
});

// smoke.vdb: our pinned openvdb.org sample (pnpm fixtures:vdb-samples).
const smokeVdb = vdbSample("smoke.vdb");

describe.skipIf(!smokeVdb)("smoke.vdb", () => {
  const load = () => {
    const bytes = readFileSync(smokeVdb!);
    const file = parseVdb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    return file.grids.find((g) => g.valueType === "float")!;
  };

  it("'point' on his 128^3 lattice equals the .wvdb bake's samples", () => {
    const grid = load();
    const src = wvdbSourceFromVdbGrid(grid);
    const dims: Vec3 = [128, 128, 128];
    const his = sampleWvdbDense(src, src.bbox, dims, 8);
    const r = vdbResampleGrid(grid, { filter: "point", dimensions: dims, bbox: src.bbox! });
    // Every voxel of every emitted leaf equals his sample; every non-zero sample of his lies in an emitted leaf.
    const covered = new Uint8Array(128 * 128 * 128);
    let mismatches = 0;
    for (const node of treeNodes(r.image)) {
      if (node.level !== 0) continue;
      const { values } = leafVoxels(r.image, node.address);
      for (let n = 0; n < 512; n++) {
        const [x, y, z] = [node.origin[0] + (n >> 6), node.origin[1] + ((n >> 3) & 7), node.origin[2] + (n & 7)];
        if (x >= 128 || y >= 128 || z >= 128) continue;
        const t = x + 128 * (y + 128 * z);
        covered[t] = 1;
        if (values[n] !== his[t]) mismatches++;
      }
    }
    expect(mismatches).toBe(0);
    let lost = 0;
    for (let t = 0; t < his.length; t++) if (his[t] !== 0 && !covered[t]) lost++;
    expect(lost).toBe(0);
  });

  it("his 128^3 with FpN: ~474 KiB gz and ~18% error, as in the comparison report", () => {
    const grid = load();
    const src = wvdbSourceFromVdbGrid(grid);
    const sizes: Record<string, number> = {};
    for (const filter of ["point", "box"] as const) {
      const r = vdbResampleGrid(grid, { filter, dimensions: [128, 128, 128], bbox: src.bbox!, quantize: "fpn" });
      expect(new Uint32Array(r.image.buffer, r.image.byteOffset + GRID_OFF_GRID_TYPE, 1)[0]).toBe(GRID_TYPE_FPN);
      const kib = gzipSync(new Uint8Array(writeNvdb([r.image])), { level: 9 }).length / 1024;
      const err = vdbResampleError(grid, r.image);
      sizes[filter] = kib;
      expect(kib).toBeGreaterThan(400);
      expect(kib).toBeLessThan(550);
      expect(err.percent).toBeGreaterThan(15);
      expect(err.percent).toBeLessThan(20);
      expect(err.count).toBeGreaterThan(140_000);
    }
    // The metric itself: full-resolution FpN is ~1.2% (comparison report).
    const full = vdbQuantize(buildFromVdb(grid), "fpn");
    expect(vdbResampleError(grid, full).percent).toBeCloseTo(1.2, 1);
  });

  it("his default 128 .wvdb bake (non-cubic voxels), read back by wvdbDenseToNvdb: 42.3% (G15a; 45.4% with one voxel size)", () => {
    const bytes = readFileSync(smokeVdb!);
    const file = parseVdb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    const dense = wvdbDenseVolume(readWvdb(vdbToWvdb(file, { resolution: 128, lods: 1 }).bytes), { mipmaps: false });
    expect(dense.dimensions).toEqual([64, 128, 64]);
    const grid = wvdbDenseToNvdb(dense);
    expect(new Set(grid.voxelSizeXYZ).size).toBe(3);
    const err = vdbResampleError(file, grid.image, { stride: 7 });
    expect(err.count).toBeGreaterThan(140_000);
    expect(err.percent).toBeCloseTo(42.3, 1);
  });
});
