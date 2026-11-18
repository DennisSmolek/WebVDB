/**
 * Active tiles above leaf level (K4.1), end to end without downloaded assets.
 *
 * A synthetic `.vdb` (test/tools/synthetic-vdb.ts) carries an active tile at
 * every level — root (4096^3), upper (128^3), lower (8^3, two of them, one at
 * negative coords) — plus inactive tiles that must be dropped, and a few
 * leaves. It is parsed, built to NanoVDB, quantized and written, and every
 * stage is checked against the values the spec put in.
 *
 * Oracles: the spec itself (parser), the proven CPU reader `readValue` /
 * `vdbSampleTrilinear` (NanoVDB), and NanoVDB `GridStats` for stats weighting.
 */

import { describe, expect, it } from "vitest";

import { buildFromVdbDetailed, vdbInspect, vdbQuantizeDetailed, writeNvdb } from "../../src/tools/index.js";
import { vdbReadValueCpu } from "../../src/tools/cpu/read-value.js";
import { vdbSampleTrilinear } from "../../src/tools/cpu/sample-trilinear.js";
import { vdbBuildFromLeavesDetailed } from "../../src/tools/nanovdb/serialize.js";
import { VDB_FLOAT_LEAF_CODEC } from "../../src/tools/nanovdb/leaf-codec.js";
import { parseVdb } from "../../src/tools/vdb/index.js";
import type { VdbGrid, VdbTile } from "../../src/tools/vdb/types.js";
import { writeSyntheticVdb, type SynthLeaf, type SynthSpec } from "./synthetic-vdb.js";

type Coord = [number, number, number];

function makeLeaf(origin: Coord, fill: (n: number) => number, active: (n: number) => boolean): SynthLeaf {
  const values = new Float32Array(512);
  const valueMask = new Uint32Array(16);
  for (let n = 0; n < 512; n++) {
    values[n] = fill(n);
    if (active(n)) valueMask[n >>> 5]! |= 1 << (n & 31);
  }
  return { origin, values, valueMask };
}

const ROOT_TILE = { origin: [-4096, 0, 0] as Coord, value: 0.25 };
const UPPER_TILE = { origin: [128, 0, 0] as Coord, value: 0.5 };
const LOWER_TILE = { origin: [8, 0, 0] as Coord, value: 0.75 };
const LOWER_TILE_NEG = { origin: [8, -8, 0] as Coord, value: -1.5 };

const LEAF_A = makeLeaf([0, 0, 0], (n) => n / 512, (n) => n % 3 === 0);
const LEAF_B = makeLeaf([0, -8, 0], (n) => -n / 256, (n) => n % 5 === 0);

function spec(fileVersion: 221 | 222): SynthSpec {
  return {
    fileVersion,
    background: 0,
    rootTiles: [
      { ...ROOT_TILE, active: true },
      { origin: [8192, 0, 0], value: 9, active: false }, // inactive: kept (K4.9)
    ],
    uppers: [
      {
        origin: [0, 0, 0],
        tiles: [
          { ...UPPER_TILE, active: true },
          { origin: [256, 0, 0], value: 7, active: false }, // inactive: kept (K4.9)
        ],
        lowers: [
          {
            origin: [0, 0, 0],
            tiles: [
              { ...LOWER_TILE, active: true },
              { origin: [16, 0, 0], value: 3, active: false }, // inactive: kept (K4.9)
            ],
            leaves: [LEAF_A],
          },
        ],
      },
      {
        origin: [0, -4096, 0],
        tiles: [],
        lowers: [{ origin: [0, -128, 0], tiles: [{ ...LOWER_TILE_NEG, active: true }], leaves: [LEAF_B] }],
      },
    ],
  };
}

function popcount(words: Uint32Array): number {
  let c = 0;
  for (const w of words) for (let b = 0; b < 32; b++) c += (w >>> b) & 1;
  return c;
}

const LEAF_ACTIVE = popcount(LEAF_A.valueMask) + popcount(LEAF_B.valueMask);
const TILE_VOXELS = 4096 ** 3 + 128 ** 3 + 2 * 512;

function parseSynthetic(fileVersion: 221 | 222): VdbGrid {
  const file = parseVdb(writeSyntheticVdb(spec(fileVersion)));
  expect(file.grids).toHaveLength(1);
  return file.grids[0]!;
}

const sortTiles = (tiles: VdbTile[]): VdbTile[] =>
  [...tiles].sort((a, b) => a.dim - b.dim || a.origin[0] - b.origin[0] || a.origin[1] - b.origin[1]);

/** A deterministic spread of coordinates inside a tile, including its corners. */
function coordsInTile(origin: Coord, dim: number, count = 64): Coord[] {
  const out: Coord[] = [origin, [origin[0] + dim - 1, origin[1] + dim - 1, origin[2] + dim - 1]];
  let s = 12345 + origin[0] * 7 + origin[1] * 13;
  const rnd = (): number => {
    s = (s * 1103515245 + 12345) >>> 0;
    return s % dim;
  };
  while (out.length < count) out.push([origin[0] + rnd(), origin[1] + rnd(), origin[2] + rnd()]);
  return out;
}

const ALL_TILES = [
  { ...ROOT_TILE, dim: 4096 },
  { ...UPPER_TILE, dim: 128 },
  { ...LOWER_TILE, dim: 8 },
  { ...LOWER_TILE_NEG, dim: 8 },
];

describe.each([222, 221] as const)("parseVdb — active tiles (file version %i)", (fileVersion) => {
  it("returns every active tile at its level and drops inactive ones", () => {
    const grid = parseSynthetic(fileVersion);
    expect(sortTiles([...grid.iterTiles()])).toEqual(
      sortTiles([
        { origin: [8, -8, 0], dim: 8, value: -1.5 },
        { origin: [8, 0, 0], dim: 8, value: 0.75 },
        { origin: [128, 0, 0], dim: 128, value: 0.5 },
        { origin: [-4096, 0, 0], dim: 4096, value: 0.25 },
      ]),
    );
    expect([...grid.iterLeaves()]).toHaveLength(2);
    expect(grid.activeVoxelCount).toBe(BigInt(LEAF_ACTIVE + TILE_VOXELS));
  });

  it("readValue resolves leaf, then active tile, then background", () => {
    const grid = parseSynthetic(fileVersion);
    for (const t of ALL_TILES) {
      for (const ijk of coordsInTile(t.origin, t.dim)) {
        expect(grid.readValue(ijk)).toEqual({ value: t.value, active: true });
      }
    }
    // Leaf voxels, active and inactive, read their stored values.
    for (const n of [0, 1, 3, 100, 511]) {
      const ijk: Coord = [(n >> 6) & 7, (n >> 3) & 7, n & 7];
      expect(grid.readValue(ijk)).toEqual({ value: LEAF_A.values[n], active: n % 3 === 0 });
    }
    // Inactive tiles keep their stored value (K4.9); empty space reads as background.
    for (const [ijk, value] of [[[8192, 5, 5], 9], [[256 + 3, 1, 1], 7], [[16 + 2, 0, 0], 3]] as [Coord, number][]) {
      expect(grid.readValue(ijk)).toEqual({ value, active: false });
    }
    expect(grid.readValue([0, 4096 * 2, 0])).toEqual({ value: 0, active: false });
    expect([...grid.iterInactiveTiles()].map((t) => [t.dim, t.value]).sort()).toEqual([[128, 7], [4096, 9], [8, 3]].sort());
  });
});

describe("buildFromVdb — active tiles become NanoVDB tiles", () => {
  const grid = parseSynthetic(222);
  const built = buildFromVdbDetailed(grid);
  const image = built.image;
  const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const TREE = 672;

  it("records tile counts and the tile-inclusive voxel count in TreeData", () => {
    expect(built.tileCounts).toEqual({ lower: 2, upper: 1, root: 1 });
    expect(dv.getUint32(TREE + 44, true)).toBe(2); // mTileCount[0] (lower)
    expect(dv.getUint32(TREE + 48, true)).toBe(1); // mTileCount[1] (upper)
    expect(dv.getUint32(TREE + 52, true)).toBe(1); // mTileCount[2] (root)
    expect(dv.getBigUint64(TREE + 56, true)).toBe(BigInt(LEAF_ACTIVE + TILE_VOXELS));
    expect(built.voxelCount).toBe(LEAF_ACTIVE + TILE_VOXELS);
    // Tiles are not expanded: only the two real leaves exist.
    expect(built.nodeCounts).toEqual({ leaf: 2, lower: 2, upper: 2 });
    expect(vdbInspect(image).voxelCount).toBe(LEAF_ACTIVE + TILE_VOXELS);
  });

  it("writes the root tile as state-on, child-less, holding its value, in origin order", () => {
    const rootOff = TREE + Number(dv.getBigUint64(TREE + 24, true));
    expect(dv.getUint32(rootOff + 24, true)).toBe(4); // table size: 2 tiles (one inactive) + 2 children
    const entries = [0, 1, 2, 3].map((i) => {
      const t = rootOff + 64 + i * 32;
      return {
        key: dv.getBigUint64(t, true),
        child: dv.getBigInt64(t + 8, true),
        state: dv.getUint32(t + 16, true),
        value: dv.getFloat32(t + 20, true),
      };
    });
    // Sorted by unsigned NanoVDB key (K4.12): [0,0,0] < [0,-4096,0] < [-4096,0,0] < [8192,0,0].
    // Keys are z | y<<21 | x<<42 over the coordinate >>> 12 (as u32).
    const byKey = new Map(entries.map((e) => [e.key, e]));
    expect(entries.map((e) => e.key)).toEqual([...entries.map((e) => e.key)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)));
    expect(byKey.size).toBe(4);
    const tiles = entries.filter((e) => e.child === 0n);
    expect(tiles).toContainEqual(expect.objectContaining({ state: 1, value: 0.25 }));
    expect(tiles).toContainEqual(expect.objectContaining({ state: 0, value: 9 })); // inactive root tile
    expect(entries.filter((e) => e.child !== 0n)).toHaveLength(2);
    // Root bbox covers the root tile's full extent, the upper tile and the negative leaf/tile.
    expect([0, 4, 8].map((o) => dv.getInt32(rootOff + o, true))).toEqual([-4096, -8, 0]);
    expect([12, 16, 20].map((o) => dv.getInt32(rootOff + o, true))).toEqual([255, 4095, 4095]);
    expect(built.indexBBox).toEqual({ min: [-4096, -8, 0], max: [255, 4095, 4095] });
  });

  it("the CPU oracle returns tile values, active, everywhere inside each tile", () => {
    for (const t of ALL_TILES) {
      for (const ijk of coordsInTile(t.origin, t.dim)) {
        expect(vdbReadValueCpu(image, ijk)).toEqual({ value: t.value, active: true });
      }
    }
  });

  it("leaf voxels and empty space are unchanged", () => {
    for (const leaf of grid.iterLeaves()) {
      for (let n = 0; n < 512; n++) {
        const ijk: Coord = [leaf.origin[0] + ((n >> 6) & 7), leaf.origin[1] + ((n >> 3) & 7), leaf.origin[2] + (n & 7)];
        const active = ((leaf.valueMask[n >>> 5]! >>> (n & 31)) & 1) !== 0;
        expect(vdbReadValueCpu(image, ijk)).toEqual({ value: leaf.values[n], active });
      }
    }
    for (const [ijk, value] of [[[8192, 5, 5], 9], [[259, 1, 1], 7], [[18, 0, 0], 3], [[0, 8192, 0], 0]] as [Coord, number][]) {
      expect(vdbReadValueCpu(image, ijk)).toEqual({ value, active: false });
    }
  });

  it("vdbSampleTrilinear inside a tile is the tile value, and blends across a tile edge", () => {
    expect(vdbSampleTrilinear(image, [190.3, 60.7, 3.2])).toBeCloseTo(0.5, 6);
    expect(vdbSampleTrilinear(image, [-2000.5, 100.25, 7.75])).toBeCloseTo(0.25, 6);
    expect(vdbSampleTrilinear(image, [11.5, 3.5, 3.5])).toBeCloseTo(0.75, 6);
    // Halfway between the upper tile (x=128) and empty space (x=127, outside every leaf/tile).
    expect(vdbSampleTrilinear(image, [127.5, 64, 64])).toBeCloseTo(0.25, 6);
  });

  it("node stats follow NanoVDB GridStats tile weighting", () => {
    // Root stats: the root tile weighs 32768, the upper tile 4096, each lower tile 512.
    let n = 32768 + 4096 + 2 * 512;
    let sum = 0.25 * 32768 + 0.5 * 4096 + 0.75 * 512 + -1.5 * 512;
    let sumSq = 0.25 ** 2 * 32768 + 0.5 ** 2 * 4096 + 0.75 ** 2 * 512 + 1.5 ** 2 * 512;
    let min = -1.5;
    let max = 0.75;
    for (const leaf of grid.iterLeaves()) {
      for (let i = 0; i < 512; i++) {
        if (((leaf.valueMask[i >>> 5]! >>> (i & 31)) & 1) === 0) continue;
        const v = leaf.values[i]!;
        n++;
        sum += v;
        sumSq += v * v;
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
    }
    const ave = sum / n;
    const std = Math.sqrt(sumSq / n - ave * ave);
    const rootOff = TREE + Number(dv.getBigUint64(TREE + 24, true));
    // FLOAT RootData: background@28, min@32, max@36, ave@40, stddev@44.
    expect(dv.getFloat32(rootOff + 32, true)).toBe(Math.fround(min));
    expect(dv.getFloat32(rootOff + 36, true)).toBe(Math.fround(max));
    expect(dv.getFloat32(rootOff + 40, true)).toBeCloseTo(ave, 6);
    expect(dv.getFloat32(rootOff + 44, true)).toBeCloseTo(std, 6);
  });

  it("quantize keeps the tiles as tiles (fp8 and fpn), including the root tile", () => {
    for (const mode of ["fp8", "fpn"] as const) {
      const q = vdbQuantizeDetailed(image, mode);
      expect(q.tileCounts).toEqual({ lower: 2, upper: 1, root: 1 });
      expect(q.nodeCounts).toEqual(built.nodeCounts);
      expect(q.voxelCount).toBe(built.voxelCount);
      for (const t of ALL_TILES) {
        for (const ijk of coordsInTile(t.origin, t.dim, 8)) {
          expect(vdbReadValueCpu(q.image, ijk)).toEqual({ value: t.value, active: true }); // tiles stay f32
        }
      }
    }
  });

  it("writeNvdb copies the tile counts into the file metadata", () => {
    const file = new DataView(writeNvdb([image]));
    const md = 16; // FileHeader size
    expect([0, 4, 8].map((o) => file.getUint32(md + 156 + o, true))).toEqual([2, 1, 1]);
  });
});

describe("buildFromLeaves — tile validation", () => {
  const leaf = { origin: [0, 0, 0] as Coord, values: new Float32Array(512), valueMask: new Uint32Array(16).fill(1) };
  const build = (tiles: { origin: Coord; dim: number; value: number }[]) =>
    vdbBuildFromLeavesDetailed([leaf], VDB_FLOAT_LEAF_CODEC, { tiles });

  it("rejects a tile that shares a slot with a leaf", () => {
    expect(() => build([{ origin: [0, 0, 0], dim: 8, value: 1 }])).toThrow(/overlaps/);
  });
  it("rejects an upper tile over an existing lower node", () => {
    expect(() => build([{ origin: [0, 0, 0], dim: 128, value: 1 }])).toThrow(/overlaps/);
  });
  it("rejects a root tile over an existing upper node", () => {
    expect(() => build([{ origin: [0, 0, 0], dim: 4096, value: 1 }])).toThrow(/overlaps/);
  });
  it("rejects misaligned tiles, unknown sizes and duplicates", () => {
    expect(() => build([{ origin: [12, 0, 0], dim: 8, value: 1 }])).toThrow(/aligned/);
    expect(() => build([{ origin: [0, 0, 0], dim: 16, value: 1 }])).toThrow(/8, 128 or 4096/);
    const t = { origin: [64, 0, 0] as Coord, dim: 8, value: 1 };
    expect(() => build([t, t])).toThrow(/duplicate/);
  });
});
