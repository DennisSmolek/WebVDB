/**
 * K4.2 / K4.7 on the openvdb.org samples (skipped when an asset is absent;
 * these files are outside the pinned `pnpm fixtures:vdb-samples` set).
 *
 * - `explosion.vdb` (v223, Blosc): density, temperature and the `vec3s`
 *   velocity grid `v`. The vector grid builds a NanoVDB Vec3f grid that the
 *   CPU oracle reads back voxel for voxel.
 * - `bunny.vdb`, `icosahedron.vdb`, `fire.vdb` (OpenVDB 12.1, v223, half
 *   float): the K4.7 files.
 * - `sphere_points.vdb`, `waterfall_points.vdb`: point data grids.
 *
 * Independent checks, all written by OpenVDB or by mjurczyk's reader, not by
 * us: the grids' `file_voxel_count` and `file_bbox_min/max` metadata; the
 * level-set narrow band (|value| <= background); his `sphere_points.wvdb`
 * header (point count and bbox, decoded by his reader from the same source)
 * and, with `MJURCZYK_OPENVDB_DIR`, its decoded positions point for point.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  buildFromVdbDetailed,
  type BuiltGrid,
  readValue,
  readValueVec3,
  readWvdb,
  sampleTrilinearVec3,
  writeNvdb,
  type VdbFile,
  type VdbGrid,
  type VdbPointDataGrid,
  type VdbVec3Grid,
  type WvdbPointsGrid,
} from "../../src/tools/index.js";
import { GRID_SIZE, TREE_OFF_NODE_OFFSET_ROOT, VEC3F_LAYOUT } from "../../src/tools/nanovdb/bytes.js";
import { lengthSqrF32 } from "../../src/tools/nanovdb/stats.js";
import { parseVdb } from "../../src/tools/vdb/index.js";
import { bytesEqual, hisAsset } from "./wvdb-helpers.js";

type Coord = [number, number, number];

const here = path.dirname(fileURLToPath(import.meta.url));
const sample = (name: string): string => path.resolve(here, "../../../../fixtures/vdb-samples", name);
const has = (name: string): boolean => existsSync(sample(name));

function parseFile(p: string): VdbFile {
  const buf = readFileSync(p);
  return parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

const isOn = (mask: Uint32Array, n: number): boolean => ((mask[n >>> 5]! >>> (n & 31)) & 1) !== 0;
const leafCoord = (o: Coord, n: number): Coord => [o[0] + ((n >> 6) & 7), o[1] + ((n >> 3) & 7), o[2] + (n & 7)];

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Active-voxel count and tight index bbox from the parsed leaves (no tiles in these files). */
function leafStats(grid: VdbGrid): { count: number; min: Coord; max: Coord } {
  let count = 0;
  const min: Coord = [Infinity, Infinity, Infinity];
  const max: Coord = [-Infinity, -Infinity, -Infinity];
  for (const leaf of grid.iterLeaves()) {
    for (let n = 0; n < 512; n++) {
      if (!isOn(leaf.valueMask, n)) continue;
      count++;
      const c = leafCoord(leaf.origin, n);
      for (let a = 0; a < 3; a++) {
        if (c[a]! < min[a]!) min[a] = c[a]!;
        if (c[a]! > max[a]!) max[a] = c[a]!;
      }
    }
  }
  return { count, min, max };
}

/** OpenVDB's own metadata: the active count and active bbox it wrote. */
function expectMatchesFileMetadata(grid: VdbGrid): void {
  const s = leafStats(grid);
  expect(grid.activeVoxelCount).toBe(grid.metadata["file_voxel_count"]);
  expect(BigInt(s.count) + [...grid.iterTiles()].reduce((a, t) => a + BigInt(t.dim) ** 3n, 0n)).toBe(grid.activeVoxelCount);
  expect(s.min).toEqual(grid.metadata["file_bbox_min"]);
  expect(s.max).toEqual(grid.metadata["file_bbox_max"]);
}

// The assets are large (explosion 76 MB, waterfall 157 MB): load them in
// beforeAll and drop them in afterAll so suites do not hold each other's data.

describe.skipIf(!has("explosion.vdb"))("explosion.vdb — every grid, incl. the vec3s velocity", () => {
  let file: VdbFile;
  let v: VdbVec3Grid;
  beforeAll(() => {
    file = parseFile(sample("explosion.vdb"));
    v = file.grids.find((g) => g.name === "v") as VdbVec3Grid;
  });
  afterAll(() => {
    file = undefined as unknown as VdbFile;
    v = undefined as unknown as VdbVec3Grid;
  });

  it("parses density, temperature and v; counts and bboxes match the file metadata", () => {
    expect(file.fileVersion).toBe(223);
    expect(file.grids.map((g) => [g.name, g.valueType])).toEqual([
      ["density", "float"],
      ["temperature", "float"],
      ["v", "vec3s"],
    ]);
    for (const g of file.grids) expectMatchesFileMetadata(g);
    expect(v.background).toEqual([0, 0, 0]);
    expect(v.metadata["file_compression"]).toBe("blosc + active values");
  });

  it("the float grids build FLOAT images that read back exactly", () => {
    for (const g of file.grids.filter((x) => x.valueType === "float")) {
      const image = buildFromVdbDetailed(g).image;
      let checked = 0;
      for (const leaf of g.iterLeaves()) {
        for (let n = 0; n < 512; n += 37) {
          const ijk = leafCoord(leaf.origin, n);
          expect(readValue(image, ijk)).toEqual(g.readValue(ijk));
          checked++;
        }
      }
      expect(checked).toBeGreaterThan(1000);
    }
  });

  describe("v -> NanoVDB Vec3f", () => {
    let built: BuiltGrid;
    let image: Uint32Array;
    beforeAll(() => {
      built = buildFromVdbDetailed(v);
      image = built.image;
    });
    afterAll(() => {
      built = undefined as unknown as BuiltGrid;
      image = undefined as unknown as Uint32Array;
    });

    it("readValueVec3 equals the parsed tree at every active voxel and 4000 random bbox coords", () => {
      expect(built.voxelCount).toBe(Number(v.activeVoxelCount));
      expect(built.indexBBox).toEqual({ min: v.metadata["file_bbox_min"], max: v.metadata["file_bbox_max"] });
      let mismatches = 0;
      for (const leaf of v.iterLeaves()) {
        for (let n = 0; n < 512; n++) {
          if (!isOn(leaf.valueMask, n)) continue;
          const ijk = leafCoord(leaf.origin, n);
          const got = readValueVec3(image, ijk);
          const want = v.readValue(ijk);
          if (!got.active || got.value[0] !== want.value[0] || got.value[1] !== want.value[1] || got.value[2] !== want.value[2]) {
            mismatches++;
          }
        }
      }
      expect(mismatches).toBe(0);
      const r = rng(42);
      const lo = v.metadata["file_bbox_min"] as Coord;
      const hi = v.metadata["file_bbox_max"] as Coord;
      for (let i = 0; i < 4000; i++) {
        const ijk = [0, 1, 2].map((a) => lo[a]! - 4 + Math.floor(r() * (hi[a]! - lo[a]! + 9))) as Coord;
        expect(readValueVec3(image, ijk)).toEqual(v.readValue(ijk));
      }
    });

    it("sampleTrilinearVec3 is componentwise trilinear of the parsed tree (500 points)", () => {
      const r = rng(7);
      const lo = v.metadata["file_bbox_min"] as Coord;
      const hi = v.metadata["file_bbox_max"] as Coord;
      const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
      for (let i = 0; i < 500; i++) {
        const p = [0, 1, 2].map((a) => lo[a]! + r() * (hi[a]! - lo[a]!)) as Coord;
        const b = p.map(Math.floor) as Coord;
        const f = p.map((c, a) => c - b[a]!);
        const tap = (dx: number, dy: number, dz: number): Coord => v.readValue([b[0] + dx, b[1] + dy, b[2] + dz]).value;
        const want = [0, 1, 2].map((c) => {
          const c00 = lerp(tap(0, 0, 0)[c]!, tap(1, 0, 0)[c]!, f[0]!);
          const c10 = lerp(tap(0, 1, 0)[c]!, tap(1, 1, 0)[c]!, f[0]!);
          const c01 = lerp(tap(0, 0, 1)[c]!, tap(1, 0, 1)[c]!, f[0]!);
          const c11 = lerp(tap(0, 1, 1)[c]!, tap(1, 1, 1)[c]!, f[0]!);
          return lerp(lerp(c00, c10, f[1]!), lerp(c01, c11, f[1]!), f[2]!);
        });
        expect(sampleTrilinearVec3(image, p)).toEqual(want);
      }
    });

    it("root min/max are the active vectors of least/greatest lengthSqr", () => {
      let minS = Infinity;
      let maxS = -Infinity;
      let minV: number[] = [];
      let maxV: number[] = [];
      for (const leaf of v.iterLeaves()) {
        for (let n = 0; n < 512; n++) {
          if (!isOn(leaf.valueMask, n)) continue;
          const [x, y, z] = [leaf.values[n * 3]!, leaf.values[n * 3 + 1]!, leaf.values[n * 3 + 2]!];
          const s = lengthSqrF32(x, y, z);
          if (s < minS) [minS, minV] = [s, [x, y, z]];
          if (s > maxS) [maxS, maxV] = [s, [x, y, z]];
        }
      }
      const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);
      const root = GRID_SIZE + Number(dv.getBigUint64(GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT, true));
      const vec = (off: number): number[] => [0, 1, 2].map((i) => dv.getFloat32(root + off + i * 4, true));
      expect(lengthSqrF32(...(vec(VEC3F_LAYOUT.rootOffMin) as Coord))).toBe(minS);
      expect(vec(VEC3F_LAYOUT.rootOffMax)).toEqual(maxV);
      expect(minV).toHaveLength(3);
    });

    it("writeNvdb carries the image unchanged", () => {
      const out = new Uint8Array(writeNvdb([image]));
      const bytes = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
      expect(bytesEqual(out.subarray(out.length - bytes.length), bytes)).toBe(true); // toEqual on 66 MB is too slow
    });
  });
});

describe.each(["bunny.vdb", "icosahedron.vdb", "fire.vdb"])("K4.7 — %s (OpenVDB 12.1 half float)", (name) => {
  it.skipIf(!has(name))("parses every grid; counts, bboxes and value ranges are OpenVDB's", () => {
    const file = parseFile(sample(name));
    expect(file.fileVersion).toBe(223);
    expect(file.grids.some((g) => g.metadata["is_saved_as_half_float"] === true)).toBe(true);
    for (const g of file.grids) {
      expect(g.valueType).toBe("float");
      expectMatchesFileMetadata(g);
      const isLevelSet = g.metadata["class"] === "level set";
      const bg = g.background as number;
      // Level sets: active values lie in the narrow band, |v| <= background.
      // Fire's density/temperature: active voxels are non-empty, v > 0.
      let outOfRange = 0;
      for (const leaf of g.iterLeaves()) {
        for (let n = 0; n < 512; n++) {
          if (!isOn(leaf.valueMask, n)) continue;
          const value = leaf.values[n]!;
          const ok = Number.isFinite(value) && (isLevelSet ? Math.abs(value) <= bg : value > 0);
          if (!ok) outOfRange++;
        }
      }
      expect(outOfRange, `${g.name}: active values outside the expected range`).toBe(0);
      const image = buildFromVdbDetailed(g).image;
      let mismatches = 0;
      for (const leaf of [...g.iterLeaves()].slice(0, 400)) {
        for (let n = 0; n < 512; n += 3) {
          const ijk = leafCoord(leaf.origin, n);
          const a = readValue(image, ijk);
          const b = g.readValue(ijk);
          if (a.value !== b.value || a.active !== b.active) mismatches++;
        }
      }
      expect(mismatches).toBe(0);
    }
  }, 60_000);
});

/** His `sphere_points.wvdb` header (decoded by his reader from his copy of the source). */
const HIS_SPHERE_POINTS = {
  count: 9802,
  bbox: [
    [-0.9998753070831299, -1.000001072883606, -0.9998753070831299],
    [0.9998735189437866, 0.9999992251396179, 0.9998735189437866],
  ],
};

function pointBBox(positions: Float32Array): [number[], number[]] {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i++) {
    lo[i % 3] = Math.min(lo[i % 3]!, positions[i]!);
    hi[i % 3] = Math.max(hi[i % 3]!, positions[i]!);
  }
  return [lo, hi];
}

/** Voxels that hold at least one point, from the parsed end offsets. */
function voxelsWithPoints(grid: VdbPointDataGrid): number {
  let n = 0;
  for (const leaf of grid.iterLeaves()) {
    let prev = 0;
    for (let i = 0; i < 512; i++) {
      if (leaf.values[i]! > prev) n++;
      prev = leaf.values[i]!;
    }
  }
  return n;
}

/** Every point lies in the world box of the file's active index bbox, padded by half a voxel. */
function expectInsideFileBBox(grid: VdbPointDataGrid, positions: Float32Array): void {
  const m = grid.transform.matrix;
  const lo = grid.metadata["file_bbox_min"] as Coord;
  const hi = grid.metadata["file_bbox_max"] as Coord;
  const [plo, phi] = pointBBox(positions);
  for (let a = 0; a < 3; a++) {
    const s = m[a * 5]!;
    const t = m[a * 4 + 3]!;
    expect(plo[a]!).toBeGreaterThanOrEqual(Math.fround((lo[a]! - 0.5) * s + t) - 1e-6);
    expect(phi[a]!).toBeLessThanOrEqual(Math.fround((hi[a]! + 0.5) * s + t) + 1e-6);
  }
}

describe.skipIf(!has("sphere_points.vdb"))("sphere_points.vdb — point data grid", () => {
  let grid: VdbPointDataGrid;
  beforeAll(() => {
    grid = parseFile(sample("sphere_points.vdb")).grids[0] as VdbPointDataGrid;
  });

  it("tree: active voxels = voxels with points = file_voxel_count; P is fxpt16", () => {
    expect(grid.valueType).toBe("ptdataidx32");
    expect(grid.activeVoxelCount).toBe(grid.metadata["file_voxel_count"]);
    expect(BigInt(voxelsWithPoints(grid))).toBe(grid.activeVoxelCount);
    expect(grid.attributes).toEqual([{ name: "P", type: "vec3s", codec: "fxpt16", index: 0 }]);
  });

  it("decodes 9,802 points on the unit sphere; count and bbox equal his .wvdb header", () => {
    const pts = grid.points();
    expect(pts.count).toBe(HIS_SPHERE_POINTS.count);
    expect(grid.pointCount).toBe(HIS_SPHERE_POINTS.count);
    const [lo, hi] = pointBBox(pts.positions);
    expect(lo).toEqual(HIS_SPHERE_POINTS.bbox[0]);
    expect(hi).toEqual(HIS_SPHERE_POINTS.bbox[1]);
    for (let i = 0; i < pts.count; i++) {
      expect(Math.hypot(pts.positions[i * 3]!, pts.positions[i * 3 + 1]!, pts.positions[i * 3 + 2]!)).toBeCloseTo(1, 4);
    }
    expectInsideFileBBox(grid, pts.positions);
  });

  const hisVdb = hisAsset("sphere_points.vdb");
  const hisWvdb = hisAsset("sphere_points.wvdb");
  it.skipIf(!hisVdb || !hisWvdb)("matches his .wvdb positions point for point (MJURCZYK_OPENVDB_DIR)", () => {
    // His .wvdb was baked from his copy of the .vdb (4 bytes larger, same points).
    const his = (parseFile(hisVdb!).grids[0] as VdbPointDataGrid).points();
    expect(his.positions).toEqual(grid.points().positions);
    const g = readWvdb(new Uint8Array(readFileSync(hisWvdb!))).grids[0] as WvdbPointsGrid;
    // His LOD k holds the points i that are multiples of 8^(lods-1-k), first LOD first.
    const lods = g.levels.length;
    const buckets: number[][] = Array.from({ length: lods }, () => []);
    for (let i = 0; i < his.count; i++) {
      let k = 0;
      while (k < lods - 1 && i % 8 ** (lods - 1 - k) !== 0) k++;
      buckets[k]!.push(i);
    }
    expect(buckets.map((b) => b.length)).toEqual(g.levels.map((l) => l.count));
    // 16-bit codes over a ~2-unit box: half a step is ~1.5e-5.
    let maxErr = 0;
    g.levels.forEach((l, k) => {
      buckets[k]!.forEach((i, j) => {
        for (let a = 0; a < 3; a++) maxErr = Math.max(maxErr, Math.abs(l.positions[j * 3 + a]! - his.positions[i * 3 + a]!));
      });
    });
    expect(maxErr).toBeLessThan(2 / 65535);
  });
});

describe.skipIf(!has("waterfall_points.vdb"))("waterfall_points.vdb — 9.5M points, four attributes", () => {
  it("decodes P, Cd, id and v; ids are unique; points stay inside the file bbox", () => {
    const grid = parseFile(sample("waterfall_points.vdb")).grids[0] as VdbPointDataGrid;
    expect(grid.activeVoxelCount).toBe(grid.metadata["file_voxel_count"]);
    expect(BigInt(voxelsWithPoints(grid))).toBe(grid.activeVoxelCount);
    expect(grid.attributes.map((a) => `${a.name}:${a.type}:${a.codec}`)).toEqual([
      "P:vec3s:fxpt16",
      "Cd:vec3s:ufxpt8",
      "id:int32:null",
      "v:vec3s:trnc",
    ]);
    const pts = grid.points();
    expect(pts.count).toBe(9_525_253);
    expect(pts.skipped).toEqual([]);
    expectInsideFileBBox(grid, pts.positions);
    const cd = pts.attributes["Cd"] as Float32Array;
    expect(cd).toHaveLength(pts.count * 3);
    for (let i = 0; i < cd.length; i += 101) {
      expect(cd[i]).toBeGreaterThanOrEqual(0);
      expect(cd[i]).toBeLessThanOrEqual(1);
      expect(Math.round(cd[i]! * 255)).toBeCloseTo(cd[i]! * 255, 3); // 8-bit codes
    }
    const ids = Int32Array.from(pts.attributes["id"] as Int32Array).sort();
    let duplicates = 0;
    for (let i = 1; i < ids.length; i++) if (ids[i] === ids[i - 1]) duplicates++;
    expect(duplicates).toBe(0);
  }, 60_000);
});
