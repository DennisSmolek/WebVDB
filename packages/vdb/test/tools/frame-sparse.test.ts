import { describe, expect, it } from "vitest";
import {
  vdbBuildFromDenseDetailed,
  indexBBox,
  nodeAverage,
  vdbReadValueF32,
  readWvdb,
  vdbSampleTrilinear,
  vdbBoxMipAtlasF32,
  vdbBoxPyramid,
  vdbFrameNodeAverageSample,
  vdbFrameSkipEmpty,
  vdbBoxPyramidLayout,
  vdbPyramidBaseLevel,
  wvdbDenseToNvdb,
  wvdbDenseVolume,
  wvdbMipAtlas,
  nvdbToWvdb,
  vdbResampleError,
  vdbResampleGrid,
  vdbTransform,
} from "../../src/tools/internal.js";
import { GRID_OFF_MAP, MAP_OFF_INVMATD, MAP_OFF_MATD, MAP_OFF_VECD } from "../../src/tools/nanovdb/bytes.js";
import { fixture } from "./wvdb-helpers.js";

// K3.2: the CPU half of the sparse volume source (LOD strategies) and the twins of its WGSL.

type V3 = [number, number, number];

/** A fog blob: a soft ball plus a separated small cube, background 0, inside [0, n)³. */
function blob(n: number): { values: Float32Array; dims: V3 } {
  const values = new Float32Array(n * n * n);
  const c = n * 0.4;
  for (let z = 0; z < n; z++)
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const r = Math.hypot(x - c, y - c, z - c) / (n * 0.25);
        let v = r < 1 ? 1 - r * r : 0;
        if (x >= n - 10 && x < n - 4 && y >= 3 && y < 9 && z >= 3 && z < 9) v = 0.5;
        values[(x * n + y) * n + z] = v; // vdbBuildFromDense is x-major
      }
  return { values, dims: [n, n, n] };
}

const lcg = (seed: number) => () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);

describe("wvdbDenseToNvdb: the dense texels as a NanoVDB FLOAT grid", () => {
  it("is voxel-exact: voxel (x, y, z) reads texel / 255 as f32, zero texels are the 0 background", () => {
    const dense = wvdbDenseVolume(readWvdb(fixture("cloud_04_variant_0000.wvdb")), { mipmaps: false });
    const grid = wvdbDenseToNvdb(dense);
    const [w, h, d] = dense.dimensions;
    expect(grid.textureIndexBox).toEqual({ min: [0, 0, 0], max: [w - 1, h - 1, d - 1] });
    const tex = dense.density.data;
    let nonZero = 0;
    for (let i = 0; i < tex.length; i += 7) {
      const x = i % w;
      const y = Math.floor(i / w) % h;
      const z = Math.floor(i / (w * h));
      expect(vdbReadValueF32(grid.image, [x, y, z])).toBe(Math.fround(tex[i]! / 255));
      if (tex[i]) nonZero++;
    }
    expect(nonZero).toBeGreaterThan(1000);
    expect(grid.voxelCount).toBe(dense.activeTexels);
  });

  // G15a: a default bake rounds each side's texel count on its own, so its voxels are not cubic. The grid must give
  // each axis its own voxel size, or it reads back shifted and stretched in world space.
  describe("a non-cubic bake reads back in place (per-axis voxel size, G15a)", () => {
    // A smooth fog source in [0, 1] (no clamp in the 8-bit bake) on a 40 x 24 x 18 box, voxel 0.5, off the origin.
    const src = (() => {
      const dims: V3 = [80, 48, 36];
      const values = new Float32Array(dims[0] * dims[1] * dims[2]);
      for (let x = 0; x < dims[0]; x++)
        for (let y = 0; y < dims[1]; y++)
          for (let z = 0; z < dims[2]; z++) {
            const r = Math.hypot((x - 38) / 34, (y - 23) / 20, (z - 17) / 15);
            values[(x * dims[1] + y) * dims[2] + z] = r < 1 ? 0.9 * (1 - r * r) * (0.75 + 0.25 * Math.sin(x * 0.3 + y * 0.2)) : 0;
          }
      return vdbBuildFromDenseDetailed(values, dims, { voxelSize: 0.5, worldOrigin: [-7, 3, 11] });
    })();
    const bake = nvdbToWvdb(src.image, { resolution: 32, lods: 1 });
    const dense = wvdbDenseVolume(readWvdb(bake.bytes), { mipmaps: false });
    const grid = wvdbDenseToNvdb(dense);
    const [w, h, d] = dense.dimensions;
    const box = dense.bbox!;
    const size = [0, 1, 2].map((a) => box[1][a]! - box[0][a]!);
    const voxel = [0, 1, 2].map((a) => size[a]! / dense.dimensions[a]!);
    /** The image's Map (double precision): index → world is `mat · ijk + vec`. */
    const mapOf = (image: Uint32Array) => {
      const v = new DataView(image.buffer, image.byteOffset, image.byteLength);
      const f64 = (off: number, n: number) => Array.from({ length: n }, (_, i) => v.getFloat64(GRID_OFF_MAP + off + i * 8, true));
      return { mat: f64(MAP_OFF_MATD, 9), inv: f64(MAP_OFF_INVMATD, 9), vec: f64(MAP_OFF_VECD, 3) };
    };

    it("the bake is non-cubic, and each axis gets its own voxel size", () => {
      expect(Math.max(w, h, d)).toBe(32);
      expect(Math.max(...voxel) - Math.min(...voxel)).toBeGreaterThan(0.01);
      for (let a = 0; a < 3; a++) expect(grid.voxelSizeXYZ[a]).toBeCloseTo(voxel[a]!, 12);
      expect(grid.voxelSize).toBe(grid.voxelSizeXYZ[0]);
      const { mat, vec } = mapOf(grid.image);
      expect(mat).toEqual([voxel[0], 0, 0, 0, voxel[1], 0, 0, 0, voxel[2]].map((x) => expect.closeTo(x!, 12)));
      // Voxel i's centre is the texel centre: box.min + (i + ½) · voxel.
      for (let a = 0; a < 3; a++) expect(vec[a]).toBeCloseTo(box[0][a]! + voxel[a]! / 2, 12);
      // The world box runs from the first texel's centre to one voxel past the last's (NanoVDB's convention).
      for (let a = 0; a < 3; a++) {
        expect(grid.worldBBox.min[a]).toBeCloseTo(vec[a]! + grid.indexBBox.min[a]! * voxel[a]!, 9);
        expect(grid.worldBBox.max[a]).toBeCloseTo(vec[a]! + (grid.indexBBox.max[a]! + 1) * voxel[a]!, 9);
      }
    });

    it("lands on the same lattice as vdbResampleGrid's 'point' resample of that box", () => {
      const r = vdbResampleGrid(src.image, { filter: "point", dimensions: dense.dimensions, bbox: box });
      const ours = mapOf(grid.image);
      const theirs = mapOf(r.image);
      for (let i = 0; i < 9; i++) expect(ours.mat[i]).toBeCloseTo(theirs.mat[i]!, 12);
      for (let i = 0; i < 3; i++) expect(ours.vec[i]).toBeCloseTo(theirs.vec[i]!, 12);
    });

    it("a world-space sample through the Map equals the texture's trilinear sample at that point", () => {
      const tex = dense.density.data;
      const texel = (x: number, y: number, z: number) => tex[x + (y + z * h) * w]! / 255;
      const { inv, vec } = mapOf(grid.image);
      const rnd = lcg(15);
      let nonZero = 0;
      for (let n = 0; n < 2000; n++) {
        // Points inside the box, away from the clamped outer half texel.
        const p = [0, 1, 2].map((a) => box[0][a]! + voxel[a]! * (0.5 + rnd() * (dense.dimensions[a]! - 1)));
        // Independent of the grid: texel coordinates from the .wvdb box alone (texel centres at i + ½).
        const u = [0, 1, 2].map((a) => ((p[a]! - box[0][a]!) / size[a]!) * dense.dimensions[a]! - 0.5);
        const i0 = u.map((c, a) => Math.min(Math.floor(c), dense.dimensions[a]! - 2));
        const f = u.map((c, a) => c - i0[a]!);
        let expected = 0;
        for (let k = 0; k < 8; k++) {
          const dx = k & 1, dy = (k >> 1) & 1, dz = k >> 2;
          const wgt = (dx ? f[0]! : 1 - f[0]!) * (dy ? f[1]! : 1 - f[1]!) * (dz ? f[2]! : 1 - f[2]!);
          expected += wgt * texel(i0[0]! + dx, i0[1]! + dy, i0[2]! + dz);
        }
        const ijk = [0, 1, 2].map((a) => (p[a]! - vec[a]!) * inv[a * 4]!) as V3;
        const got = vdbSampleTrilinear(grid.image, ijk);
        expect(got).toBeCloseTo(expected, 5);
        if (expected > 0.05) nonZero++;
      }
      expect(nonZero).toBeGreaterThan(300);
    });

    it("vdbResampleError against the source is the bake's own error, not a misplacement", () => {
      const placed = vdbResampleError(src.image, grid.image, { stride: 3 }).percent;
      // The same voxels with one voxel size (the x side's, the pre-G15a Map) for all three axes.
      const xOnly = vdbTransform(grid.image, { voxelSize: voxel[0]!, worldOrigin: [box[0][0]! + voxel[0]! / 2, box[0][1]! + voxel[0]! / 2, box[0][2]! + voxel[0]! / 2] });
      const misplaced = vdbResampleError(src.image, xOnly, { stride: 3 }).percent;
      expect(placed).toBeLessThan(misplaced / 2);
    });
  });
});

describe("nodeAverage (twin of the vdbNodeAverage helper)", () => {
  const { values, dims } = blob(40);
  const built = vdbBuildFromDenseDetailed(values, dims);
  const img = built.image;
  const mean = (lo: V3, size: number) => {
    let s = 0;
    for (let z = lo[2]; z < lo[2] + size; z++)
      for (let y = lo[1]; y < lo[1] + size; y++)
        for (let x = lo[0]; x < lo[0] + size; x++) s += x < dims[0] && y < dims[1] && z < dims[2] && x >= 0 && y >= 0 && z >= 0 ? values[(x * dims[1] + y) * dims[2] + z]! : 0;
    return s / size ** 3;
  };

  it("is the exact 8³ box mean of a fog leaf (ave × active fraction)", () => {
    for (const lo of [[8, 8, 8], [16, 8, 16], [0, 0, 0], [24, 16, 8], [32, 0, 0]] as V3[]) {
      expect(nodeAverage(img, lo, 8)).toBeCloseTo(mean(lo, 8), 5);
      expect(nodeAverage(img, [lo[0] + 3, lo[1] + 5, lo[2] + 7], 8)).toBe(nodeAverage(img, lo, 8));
    }
  });

  it("is the background outside every leaf and the active mean of the lower node at dim 128", () => {
    expect(nodeAverage(img, [200, 200, 200], 8)).toBe(0);
    expect(nodeAverage(img, [-300, 0, 0], 128)).toBe(0);
    let s = 0;
    let n = 0;
    for (const v of values) if (v !== 0) (s += v), n++;
    expect(nodeAverage(img, [5, 5, 5], 128)).toBeCloseTo(s / n, 5);
  });

  it("the node-average tap interpolates leaf averages on the leaf-centre lattice", () => {
    // At a leaf centre (8j + 3.5) the tap is that leaf's average; between two centres it is linear.
    const a = nodeAverage(img, [8, 8, 8], 8);
    const b = nodeAverage(img, [16, 8, 8], 8);
    expect(vdbFrameNodeAverageSample(img, [11.5, 11.5, 11.5], 8)).toBeCloseTo(a, 6);
    expect(vdbFrameNodeAverageSample(img, [15.5, 11.5, 11.5], 8)).toBeCloseTo((a + b) / 2, 6);
  });
});

describe("vdbBoxPyramid ('coarse-pyramid' LOD)", () => {
  it("L₀ = max(1, ceil(log2(maxTexels / pyramidMax)))", () => {
    expect(vdbPyramidBaseLevel([84, 128, 43], 128)).toBe(1);
    expect(vdbPyramidBaseLevel([500, 340, 600], 128)).toBe(3);
    expect(vdbPyramidBaseLevel([1024, 10, 10], 128)).toBe(3);
    expect(vdbPyramidBaseLevel([1025, 10, 10], 128)).toBe(4);
  });

  it("the base level is the box mean of 2^L₀ blocks (aligned box) and the atlas is his layout in float", () => {
    const { values, dims } = blob(32);
    const built = vdbBuildFromDenseDetailed(values, dims);
    const box = { min: [0, 0, 0] as V3, max: [31, 31, 31] as V3 };
    const p = vdbBoxPyramid(built.image, { indexBox: box, level: 2 });
    expect(p.level).toBe(2);
    expect(p.dimensions).toEqual([8, 8, 8]);
    for (const [jx, jy, jz] of [[2, 2, 2], [3, 1, 4], [7, 0, 0], [5, 6, 7]] as V3[]) {
      let s = 0;
      for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) s += values[((jx * 4 + x) * 32 + jy * 4 + y) * 32 + jz * 4 + z]!;
      expect(p.data[jx + (jy + jz * 8) * 8]).toBeCloseTo(s / 64, 5);
    }
    expect(p.atlas!.levels).toBe(3);
    expect(p.lodLevels).toBe(5);
    // Same layout and (up to his 8-bit rounding) the same values as wvdbMipAtlas on the bytes.
    const bytes = new Uint8Array(p.data.length);
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.round(p.data[i]! * 255);
    const his = wvdbMipAtlas(bytes, p.dimensions, 1)!;
    expect(p.atlas!.dimensions).toEqual([...his.dimensions]);
    let worst = 0;
    for (let i = 0; i < his.data.length; i++) worst = Math.max(worst, Math.abs(his.data[i]! / 255 - p.atlas!.data[i]!));
    expect(worst).toBeLessThan(3 / 255);
  });

  it("vdbBoxPyramidLayout (shared with the core GPU build, K3.2a) = the CPU pyramid's sizes and his atlas offsets", () => {
    const grid = wvdbDenseToNvdb(wvdbDenseVolume(readWvdb(fixture("cloud_04_variant_0000.wvdb")), { mipmaps: false }));
    const box = grid.textureIndexBox;
    for (const pyramidMax of [128, 64, 32, 7, 1]) {
      const p = vdbBoxPyramid(grid.image, { indexBox: box, pyramidMax });
      const l = vdbBoxPyramidLayout([box.max[0] - box.min[0] + 1, box.max[1] - box.min[1] + 1, box.max[2] - box.min[2] + 1], { pyramidMax });
      expect([l.level, ...l.dimensions, l.lodLevels]).toEqual([p.level, ...p.dimensions, p.lodLevels]);
      expect(l.atlas?.dimensions ?? null).toEqual(p.atlas?.dimensions ?? null);
      expect(l.atlas?.levels.length ?? 0).toBe(p.atlas?.levels ?? 0);
      if (!l.atlas) continue;
      // Each level sits at x = W − W / 2^(l−1) and is the ceil half of the one below (frameMipmaps.js packAtlas).
      const W = l.atlas.dimensions[0];
      let below = l.dimensions;
      l.atlas.levels.forEach((level, i) => {
        expect(level.offset).toBe(W - W / 2 ** i);
        expect(level.dimensions).toEqual(below.map((v) => Math.max(1, Math.ceil(v / 2))));
        expect(level.offset + level.dimensions[0]).toBeLessThanOrEqual(W);
        below = level.dimensions;
      });
    }
    expect(vdbBoxPyramidLayout([1, 1, 1], { level: 0 }).atlas).toBeNull();
    expect(() => vdbBoxPyramidLayout([0, 4, 4])).toThrow(/positive integers/);
  });

  it("vdbBoxMipAtlasF32 averages 2x2x2 with clamped edge taps", () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]); // 3x3x1
    const a = vdbBoxMipAtlasF32(data, [3, 3, 1])!;
    // level 1 (2x2x1): texel (1,1) reads x 2,2 and y 2,2 (clamped): (9*8)/8
    expect(a.levels).toBe(2);
    expect(a.data[0]).toBeCloseTo((1 + 2 + 4 + 5) * 2 / 8, 6);
  });
});

describe("vdbFrameSkipEmpty (twin of the shadow march's HDDA query)", () => {
  const { values, dims } = blob(48);
  const built = vdbBuildFromDenseDetailed(values, dims);
  const img = built.image;
  const bb = indexBBox(img);

  it("never skips a sample that is not exactly 0 (level 0, 2x2x2 bound) and stops in non-empty cells", () => {
    const rnd = lcg(7);
    let skipped = 0;
    let rays = 0;
    for (let r = 0; r < 300; r++) {
      const o: V3 = [rnd() * 60 - 6, rnd() * 60 - 6, rnd() * 60 - 6];
      const d: V3 = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
      const t1 = 120;
      let t = 0;
      while (t < t1) {
        const [ts, te] = vdbFrameSkipEmpty(img, o, d, t, t1, 0, false);
        expect(ts).toBeGreaterThanOrEqual(t);
        // Every point in [t, ts) samples to exactly 0.
        for (let k = 0; k < 24; k++) {
          const s = t + ((ts - t) * (k + 0.5)) / 24;
          if (s >= ts) continue;
          const p: V3 = [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
          expect(vdbSampleTrilinear(img, p)).toBe(0);
        }
        skipped += ts - t;
        if (ts >= t1) break;
        t = Math.max(te, ts + 1e-3);
      }
      rays++;
    }
    expect(rays).toBe(300);
    expect(skipped).toBeGreaterThan(1000);
  }, 30_000);

  it("the dilated bound covers taps that reach 8 voxels past the cell (coarse levels)", () => {
    const rnd = lcg(11);
    for (let r = 0; r < 200; r++) {
      const o: V3 = [rnd() * 60 - 6, rnd() * 60 - 6, rnd() * 60 - 6];
      const d: V3 = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
      const [ts] = vdbFrameSkipEmpty(img, o, d, 0, 120, 0, true);
      for (let k = 0; k < 16; k++) {
        const s = (ts * (k + 0.5)) / 16;
        if (s >= ts) continue;
        const p: V3 = [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
        // The level-0 tap and the level-3 node-average tap are 0, and (first sample of each ray, brute force)
        // every voxel a level-2 pyramid tap reaches, floor(p) − 6 … floor(p) + 7, is 0.
        expect(vdbSampleTrilinear(img, p)).toBe(0);
        expect(vdbFrameNodeAverageSample(img, p, 8)).toBe(0);
        if (k === 0 && r % 4 === 0) {
          const b: V3 = [Math.floor(p[0]), Math.floor(p[1]), Math.floor(p[2])];
          for (let z = -6; z <= 7; z++) for (let y = -6; y <= 7; y++) for (let x = -6; x <= 7; x++) expect(vdbReadValueF32(img, [b[0] + x, b[1] + y, b[2] + z])).toBe(0);
        }
      }
    }
    expect(bb.min[0]).toBeGreaterThanOrEqual(0);
  }, 30_000);
});
