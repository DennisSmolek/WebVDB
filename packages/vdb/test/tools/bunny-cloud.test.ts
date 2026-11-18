/**
 * K4 exit gate, first item: `bunny_cloud.vdb` parses and round-trips through
 * the CPU oracle.
 *
 * The openvdb.org `bunny_cloud.vdb` (OpenVDB 12.1, file version 223) is a
 * Blosc/LZ4-compressed fog volume with ~19.2M active voxels and two active
 * lower-node tiles. Oracles:
 *  - the file's own `file_voxel_count` metadata, written by OpenVDB, for the
 *    parsed active-voxel count (leaf masks plus 512 per lower tile);
 *  - the parsed `.vdb` tree's `readValue` for the NanoVDB image read through
 *    the proven CPU reader (`readValue`, `sampleTrilinear`), at every tile
 *    voxel, thousands of random active leaf voxels and random bbox coords.
 *
 * Skipped when the asset is absent (`pnpm fixtures:vdb-samples` does not fetch
 * it; the Stanford bunny derivatives are excluded from the pinned set).
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

import { buildFromVdbDetailed, type BuiltGrid } from "../../src/tools/index.js";
import { readValue } from "../../src/tools/cpu/read-value.js";
import { sampleTrilinear } from "../../src/tools/cpu/sample-trilinear.js";
import { parseVdb } from "../../src/tools/vdb/index.js";
import type { VdbGrid, VdbLeaf, VdbTile } from "../../src/tools/vdb/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BUNNY = path.resolve(__dirname, "../../../../fixtures/vdb-samples/bunny_cloud.vdb");

type Coord = [number, number, number];

/** Deterministic PRNG (mulberry32) so failures reproduce. */
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

const isOn = (mask: Uint32Array, n: number): boolean => ((mask[n >>> 5]! >>> (n & 31)) & 1) !== 0;
const leafCoord = (o: Coord, n: number): Coord => [o[0] + ((n >> 6) & 7), o[1] + ((n >> 3) & 7), o[2] + (n & 7)];

describe.skipIf(!existsSync(BUNNY))("bunny_cloud.vdb — parse, build, CPU-oracle round trip (K4 gate)", () => {
  let grid: VdbGrid;
  let leaves: VdbLeaf[];
  let tiles: VdbTile[];
  let built: BuiltGrid;

  beforeAll(() => {
    const buf = readFileSync(BUNNY);
    const file = parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(file.grids).toHaveLength(1);
    grid = file.grids[0]!;
    leaves = [...grid.iterLeaves()];
    tiles = [...grid.iterTiles()];
    built = buildFromVdbDetailed(grid);
  }, 120_000);

  it("parses: counts match the file's own metadata", () => {
    expect(grid.name).toBe("density");
    expect(grid.metadata["class"]).toBe("fog volume");
    expect(grid.activeVoxelCount).toBe(19_210_271n);
    expect(grid.metadata["file_voxel_count"]).toBe(grid.activeVoxelCount);
    expect(leaves).toHaveLength(66_212);
    expect(tiles.map((t) => t.dim)).toEqual([8, 8]);

    let leafActive = 0;
    let negative = 0;
    for (const leaf of leaves) {
      for (let n = 0; n < 512; n++) {
        if (!isOn(leaf.valueMask, n)) continue;
        leafActive++;
        const v = leaf.values[n]!;
        if (!Number.isFinite(v) || v < 0) negative++;
      }
    }
    expect(leafActive + tiles.length * 512).toBe(Number(grid.activeVoxelCount));
    expect(negative).toBe(0); // a fog volume: finite, non-negative densities
  });

  it("builds to NanoVDB with the tiles kept as lower-node tiles", () => {
    expect(built.voxelCount).toBe(Number(grid.activeVoxelCount));
    expect(built.tileCounts).toEqual({ lower: 2, upper: 0, root: 0 });
    expect(built.nodeCounts.leaf).toBe(66_212);
    expect(built.indexBBox).toEqual(grid.indexBBox);
  });

  it("every tile voxel reads back as the tile value, active", () => {
    let checked = 0;
    for (const t of tiles) {
      for (let x = 0; x < t.dim; x++)
        for (let y = 0; y < t.dim; y++)
          for (let z = 0; z < t.dim; z++) {
            const ijk: Coord = [t.origin[0] + x, t.origin[1] + y, t.origin[2] + z];
            expect(grid.readValue(ijk)).toEqual({ value: t.value, active: true });
            expect(readValue(built.image, ijk)).toEqual({ value: t.value, active: true });
            checked++;
          }
    }
    expect(checked).toBe(1024);
  });

  it("4000 random active leaf voxels and 4000 random bbox coords match the parsed tree", () => {
    const rand = rng(0x4b41);
    let mismatches = 0;
    for (let i = 0; i < 4000; i++) {
      const leaf = leaves[Math.floor(rand() * leaves.length)]!;
      let n = Math.floor(rand() * 512);
      while (!isOn(leaf.valueMask, n)) n = (n + 1) & 511; // every stored leaf has an active voxel
      const ijk = leafCoord(leaf.origin, n);
      const want = grid.readValue(ijk);
      const got = readValue(built.image, ijk);
      if (!want.active || got.value !== want.value || got.active !== want.active) mismatches++;
    }
    const { min, max } = grid.indexBBox!;
    for (let i = 0; i < 4000; i++) {
      const ijk = [0, 1, 2].map((a) => min[a]! + Math.floor(rand() * (max[a]! - min[a]! + 1))) as Coord;
      const want = grid.readValue(ijk);
      const got = readValue(built.image, ijk);
      if (got.value !== want.value || got.active !== want.active) mismatches++;
    }
    expect(mismatches).toBe(0);
  });

  it("sampleTrilinear matches trilinear interpolation of the parsed tree", () => {
    const rand = rng(0xb0b);
    const { min, max } = grid.indexBBox!;
    const tap = (x: number, y: number, z: number): number => grid.readValue([x, y, z]).value;
    let maxErr = 0;
    // Half the points near tiles (so taps straddle tile edges), half anywhere.
    for (let i = 0; i < 1000; i++) {
      let p: Coord;
      if (i % 2 === 0) {
        const t = tiles[i % tiles.length]!;
        p = [0, 1, 2].map((a) => t.origin[a]! - 2 + rand() * (t.dim + 4)) as Coord;
      } else {
        p = [0, 1, 2].map((a) => min[a]! + rand() * (max[a]! - min[a]!)) as Coord;
      }
      const [bx, by, bz] = p.map(Math.floor) as Coord;
      const [fx, fy, fz] = [p[0] - bx, p[1] - by, p[2] - bz];
      const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
      const want = lerp(
        lerp(lerp(tap(bx, by, bz), tap(bx + 1, by, bz), fx), lerp(tap(bx, by + 1, bz), tap(bx + 1, by + 1, bz), fx), fy),
        lerp(
          lerp(tap(bx, by, bz + 1), tap(bx + 1, by, bz + 1), fx),
          lerp(tap(bx, by + 1, bz + 1), tap(bx + 1, by + 1, bz + 1), fx),
          fy,
        ),
        fz,
      );
      maxErr = Math.max(maxErr, Math.abs(sampleTrilinear(built.image, p) - want));
    }
    expect(maxErr).toBe(0);
  });
});
