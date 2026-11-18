/**
 * K4.2 vector grids end to end without downloaded assets, plus the K4.7
 * half-float empty-array rule.
 *
 * Synthetic `.vdb` files (test/tools/synthetic-vdb.ts) carry a vector grid
 * with leaves (one at negative coords), an active tile at every level and
 * inactive tiles, written in every storage the parser handles: raw, zlib and
 * Blosc (LZ4, byte shuffle, split streams), full width and half, with and
 * without active-mask compression, at file versions 221, 222 and 223. The
 * grid is parsed, built to a NanoVDB Vec3f grid and written to `.nvdb`.
 *
 * Oracles: the spec values (parser); the parsed tree through
 * `readValueVec3` / `sampleTrilinearVec3` (NanoVDB image); NanoVDB
 * `tools/GridStats.h` `Stats<Vec3f, 1>` recomputed by brute force (stats);
 * the vendored PNanoVDB stride tables (layout).
 */

import { describe, expect, it } from "vitest";

import {
  buildFromVdbDetailed,
  FILE_HEADER_SIZE,
  FILE_METADATA_SIZE,
  inspect,
  readValue,
  readValueVec3,
  sampleTrilinearVec3,
  writeNvdb,
  type VdbGrid,
  type VdbVec3Grid,
} from "../../src/tools/index.js";
import { gridTypeConstantsFor } from "../../src/tools/cpu/stride-tables.js";
import { GRID_OFF_GRID_TYPE, GRID_SIZE, TREE_OFF_NODE_OFFSET_ROOT, VEC3F_LAYOUT } from "../../src/tools/nanovdb/bytes.js";
import { lengthSqrF32 } from "../../src/tools/nanovdb/stats.js";
import { parseVdb } from "../../src/tools/vdb/index.js";
import { floatToHalf, writeSyntheticVdb, type SynthLeaf, type SynthSpec, type SynthValueType } from "./synthetic-vdb.js";

type Coord = [number, number, number];
type V3 = [number, number, number];

const isOn = (mask: Uint32Array, n: number): boolean => ((mask[n >>> 5]! >>> (n & 31)) & 1) !== 0;
const leafCoord = (o: Coord, n: number): Coord => [o[0] + ((n >> 6) & 7), o[1] + ((n >> 3) & 7), o[2] + (n & 7)];

/** Voxel value: exact in half, f32 and i32 (multiples of 1/8, |v| < 64). */
const field = ([x, y, z]: Coord): V3 => [x * 0.25 + 1, y * 0.5 - 2, z * 0.125 + 0.5];
const fieldInt = ([x, y, z]: Coord): V3 => [x + 1, 2 * y - 3, 3 - z];

function makeLeaf(origin: Coord, f: (c: Coord) => V3): SynthLeaf {
  const values = new Float32Array(512 * 3);
  const valueMask = new Uint32Array(16);
  for (let n = 0; n < 512; n++) {
    if (n % 3 === 0) continue; // inactive voxels hold 0 (active-mask compression drops them)
    valueMask[n >>> 5]! |= 1 << (n & 31);
    values.set(f(leafCoord(origin, n)), n * 3);
  }
  return { origin, values, valueMask };
}

const ROOT_TILE: { origin: Coord; value: V3 } = { origin: [4096, 0, 0], value: [1, -2, 3] };
const UPPER_TILE: { origin: Coord; value: V3 } = { origin: [128, 0, 0], value: [5, 7, -4] };
const LOWER_TILE: { origin: Coord; value: V3 } = { origin: [16, 0, 0], value: [-6, 9, 2] };

interface Variant {
  label: string;
  fileVersion: 221 | 222 | 223;
  valueType?: SynthValueType;
  half?: boolean;
  compression?: SynthSpec["compression"];
  background?: V3;
}

function spec(v: Variant): SynthSpec {
  const f = v.valueType === "vec3i" ? fieldInt : field;
  const bg = v.background ?? [0, 0, 0];
  return {
    fileVersion: v.fileVersion,
    valueType: v.valueType ?? "vec3s",
    half: v.half ?? false,
    ...(v.compression ? { compression: v.compression } : {}),
    background: bg,
    voxelSize: 0.5,
    rootTiles: [
      { ...ROOT_TILE, active: true },
      { origin: [8192, 0, 0], value: bg, active: false }, // inactive: dropped
    ],
    uppers: [
      {
        origin: [0, 0, 0],
        tiles: [
          { ...UPPER_TILE, active: true },
          { origin: [256, 0, 0], value: bg, active: false },
        ],
        lowers: [
          {
            origin: [0, 0, 0],
            tiles: [
              { ...LOWER_TILE, active: true },
              { origin: [24, 0, 0], value: bg, active: false },
            ],
            leaves: [makeLeaf([0, 0, 0], f), makeLeaf([8, 0, 0], f)],
          },
        ],
      },
      {
        // No tiles anywhere in this branch: with active-mask compression its
        // internal nodes store zero values (the K4.7 case for half floats).
        origin: [-4096, -4096, -4096],
        tiles: [],
        lowers: [{ origin: [-128, -128, -128], tiles: [], leaves: [makeLeaf([-8, -8, -8], f)] }],
      },
    ],
  };
}

const VARIANTS: Variant[] = [
  { label: "v222 raw, non-zero background", fileVersion: 222, background: [0.5, -1, 2] },
  { label: "v221 raw (packed internal values)", fileVersion: 221, background: [0.5, -1, 2] },
  { label: "v222 zip + active mask", fileVersion: 222, compression: { zip: true, activeMask: true } },
  { label: "v222 zip + active mask, half", fileVersion: 222, half: true, compression: { zip: true, activeMask: true } },
  { label: "v223 blosc + active mask", fileVersion: 223, compression: { blosc: true, activeMask: true } },
  { label: "v223 blosc + active mask, half", fileVersion: 223, half: true, compression: { blosc: true, activeMask: true } },
  { label: "v223 blosc, all values", fileVersion: 223, compression: { blosc: true } },
  { label: "vec3d v222 zip", fileVersion: 222, valueType: "vec3d", compression: { zip: true } },
  { label: "vec3i v223 blosc + active mask, half flag ignored", fileVersion: 223, valueType: "vec3i", half: true, compression: { blosc: true, activeMask: true } },
];

function parseOne(s: SynthSpec): VdbVec3Grid {
  const file = parseVdb(writeSyntheticVdb(s));
  expect(file.grids).toHaveLength(1);
  const grid = file.grids[0]!;
  if (grid.valueType === "float" || grid.valueType === "ptdataidx32") throw new Error("expected a vector grid");
  return grid;
}

/** Expected value at `ijk` straight from the spec. */
function expected(s: SynthSpec, ijk: Coord): { value: V3; active: boolean } {
  for (const upper of s.uppers) {
    for (const lower of upper.lowers) {
      for (const leaf of lower.leaves) {
        const d = [0, 1, 2].map((a) => ijk[a]! - leaf.origin[a]!);
        if (d.every((c) => c >= 0 && c < 8)) {
          const n = (d[0]! << 6) | (d[1]! << 3) | d[2]!;
          return { value: Array.from(leaf.values.subarray(n * 3, n * 3 + 3)) as V3, active: isOn(leaf.valueMask, n) };
        }
      }
    }
  }
  for (const [t, dim] of [
    [LOWER_TILE, 8],
    [UPPER_TILE, 128],
    [ROOT_TILE, 4096],
  ] as const) {
    if ([0, 1, 2].every((a) => ijk[a]! >= t.origin[a]! && ijk[a]! < t.origin[a]! + dim)) return { value: t.value, active: true };
  }
  return { value: s.background as V3, active: false };
}

const PROBES: Coord[] = [
  [0, 0, 0],
  [3, 5, 7],
  [9, 2, 6],
  [15, 7, 7],
  [-1, -1, -1],
  [-8, -8, -8],
  [-5, -3, -2],
  [16, 0, 0],
  [23, 7, 7],
  [24, 0, 0], // inactive lower tile: background
  [130, 64, 100], // upper tile
  [256, 0, 0], // inactive upper tile
  [5000, 4000, 100], // root tile
  [8192, 0, 0], // inactive root tile
  [-9, 0, 0],
];

describe.each(VARIANTS)("parseVdb — vector grid ($label)", (variant) => {
  const s = spec(variant);

  it("exposes value type, components, background and active count", () => {
    const grid = parseOne(s);
    expect(grid.valueType).toBe(variant.valueType ?? "vec3s");
    expect(grid.components).toBe(3);
    expect(grid.background).toEqual(s.background);
    // 3 leaves x 341 active voxels + the three active tiles.
    expect(grid.activeVoxelCount).toBe(BigInt(3 * 341 + 8 ** 3 + 128 ** 3 + 4096 ** 3));
    const tiles = [...grid.iterTiles()].sort((a, b) => a.dim - b.dim);
    expect(tiles).toEqual([
      { origin: LOWER_TILE.origin, dim: 8, value: LOWER_TILE.value },
      { origin: UPPER_TILE.origin, dim: 128, value: UPPER_TILE.value },
      { origin: ROOT_TILE.origin, dim: 4096, value: ROOT_TILE.value },
    ]);
    const leaves = [...grid.iterLeaves()];
    expect(leaves).toHaveLength(3);
    for (const leaf of leaves) expect(leaf.values).toHaveLength(1536);
  });

  it("readValue returns [x, y, z] tuples: leaf voxels, tiles, background", () => {
    const grid = parseOne(s);
    for (const ijk of PROBES) expect(grid.readValue(ijk), `at ${ijk}`).toEqual(expected(s, ijk));
    for (const leaf of grid.iterLeaves()) {
      for (let n = 0; n < 512; n += 5) {
        const ijk = leafCoord(leaf.origin, n);
        expect(grid.readValue(ijk)).toEqual(expected(s, ijk));
      }
    }
  });
});

describe("K4.7 — half-float arrays with no values store nothing", () => {
  const halfFloatSpec = (compression: SynthSpec["compression"]): SynthSpec => ({
    fileVersion: 223,
    half: true,
    compression,
    background: 0,
    rootTiles: [],
    uppers: [
      {
        origin: [0, 0, 0],
        tiles: [], // no active tiles: upper and lower store zero values
        lowers: [
          {
            origin: [0, 0, 0],
            tiles: [],
            leaves: [
              {
                origin: [0, 0, 0],
                values: Float32Array.from({ length: 512 }, (_, n) => (n % 2 ? n / 64 : 0)),
                valueMask: new Uint32Array(16).fill(0xaaaaaaaa),
              },
            ],
          },
        ],
      },
    ],
  });

  it.each([
    ["blosc", { blosc: true, activeMask: true }],
    ["zip", { zip: true, activeMask: true }],
  ] as const)("a %s half-float float grid with tile-free internal nodes parses", (_, compression) => {
    const grid = parseVdb(writeSyntheticVdb(halfFloatSpec(compression))).grids[0]!;
    expect(grid.valueType).toBe("float");
    expect(grid.activeVoxelCount).toBe(256n);
    expect(grid.readValue([0, 0, 1])).toEqual({ value: 1 / 64, active: true });
    expect(grid.readValue([7, 7, 7])).toEqual({ value: 511 / 64, active: true });
    expect(grid.readValue([0, 0, 2])).toEqual({ value: 0, active: false });
  });

  it("the synthetic writer stores halves (sanity of the oracle)", () => {
    expect(floatToHalf(1)).toBe(0x3c00);
    expect(floatToHalf(-2)).toBe(0xc000);
    expect(floatToHalf(0.5)).toBe(0x3800);
    expect(floatToHalf(65504)).toBe(0x7bff);
    expect(floatToHalf(2 ** -24)).toBe(0x0001);
  });
});

describe("buildFromVdb — Vec3f NanoVDB grid", () => {
  const s = spec({ label: "", fileVersion: 223, compression: { blosc: true, activeMask: true } });
  const grid = parseOne(s);
  const built = buildFromVdbDetailed(grid);
  const image = built.image;
  const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);

  it("is a GridType Vec3f image with Vec3f node sizes", () => {
    expect(dv.getUint32(GRID_OFF_GRID_TYPE, true)).toBe(6);
    expect(built.nodeCounts).toEqual({ leaf: 3, lower: 2, upper: 2 });
    expect(built.tileCounts).toEqual({ lower: 1, upper: 1, root: 1 });
    expect(built.voxelCount).toBe(3 * 341 + 8 ** 3 + 128 ** 3 + 4096 ** 3);
    const L = VEC3F_LAYOUT;
    const rootTiles = 3; // two uppers + one tile
    expect(image.byteLength).toBe(GRID_SIZE + 64 + L.rootSize + rootTiles * L.rootTileSize + 2 * L.upperSize + 2 * L.lowerSize + 3 * L.leafSize);
    expect(L.leafSize).toBe(128 + 512 * 12);
    const report = inspect(image);
    expect(report.gridType).toBe("Vec3f");
    const { total, ...sections } = report.memoryBreakdown;
    expect(Object.values(sections).reduce((a, b) => a + b, 0)).toBe(total);
    expect(sections["upper"]).toBe(2 * L.upperSize);
  });

  it("VEC3F_LAYOUT equals the vendored PNanoVDB stride tables", () => {
    const g = gridTypeConstantsFor("VEC3F");
    const L = VEC3F_LAYOUT;
    expect([L.rootSize, L.rootTileSize, L.rootTileOffValue, L.rootOffBackground]).toEqual([
      g.root_size,
      g.root_tile_size,
      g.root_tile_off_value,
      g.root_off_background,
    ]);
    expect([L.rootOffMin, L.rootOffMax, L.rootOffAve, L.rootOffStdDev]).toEqual([g.root_off_min, g.root_off_max, g.root_off_ave, g.root_off_stddev]);
    expect([L.tableStride, L.valueStrideBits]).toEqual([g.table_stride, g.value_stride_bits]);
    expect([L.upperSize, L.upperOffTable, L.upperOffMin, L.upperOffMax, L.upperOffAve, L.upperOffStdDev]).toEqual([
      g.upper_size, g.upper_off_table, g.upper_off_min, g.upper_off_max, g.upper_off_ave, g.upper_off_stddev,
    ]);
    expect([L.lowerSize, L.lowerOffTable, L.lowerOffMin, L.lowerOffMax, L.lowerOffAve, L.lowerOffStdDev]).toEqual([
      g.lower_size, g.lower_off_table, g.lower_off_min, g.lower_off_max, g.lower_off_ave, g.lower_off_stddev,
    ]);
    expect([L.leafSize, L.leafOffTable, L.leafOffMin, L.leafOffMax, L.leafOffAve, L.leafOffStdDev]).toEqual([
      g.leaf_size, g.leaf_off_table, g.leaf_off_min, g.leaf_off_max, g.leaf_off_ave, g.leaf_off_stddev,
    ]);
  });

  it("readValueVec3 matches the parsed tree everywhere we probe", () => {
    for (const ijk of PROBES) expect(readValueVec3(image, ijk), `at ${ijk}`).toEqual(grid.readValue(ijk));
    for (const leaf of grid.iterLeaves()) {
      for (let n = 0; n < 512; n++) {
        const ijk = leafCoord(leaf.origin, n);
        expect(readValueVec3(image, ijk)).toEqual(grid.readValue(ijk));
      }
    }
  });

  it("sampleTrilinearVec3 is componentwise trilinear of the parsed tree", () => {
    const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
    const points: V3[] = [
      [3.25, 4.5, 6.75],
      [7.5, 7.5, 7.5], // leaf 0 / leaf 1 seam
      [15.5, 3.2, 7.9], // leaf -> lower tile
      [23.6, 7.4, 7.7], // lower tile -> background
      [-0.5, -0.5, -0.5], // negative leaf -> background
      [127.5, 2.25, 9.75], // background -> upper tile
    ];
    for (const p of points) {
      const b = p.map(Math.floor) as Coord;
      const f = p.map((c, a) => c - b[a]!);
      const v = (dx: number, dy: number, dz: number): V3 => grid.readValue([b[0] + dx, b[1] + dy, b[2] + dz]).value;
      const want = [0, 1, 2].map((c) => {
        const c00 = lerp(v(0, 0, 0)[c]!, v(1, 0, 0)[c]!, f[0]!);
        const c10 = lerp(v(0, 1, 0)[c]!, v(1, 1, 0)[c]!, f[0]!);
        const c01 = lerp(v(0, 0, 1)[c]!, v(1, 0, 1)[c]!, f[0]!);
        const c11 = lerp(v(0, 1, 1)[c]!, v(1, 1, 1)[c]!, f[0]!);
        return lerp(lerp(c00, c10, f[1]!), lerp(c01, c11, f[1]!), f[2]!);
      });
      expect(sampleTrilinearVec3(image, p)).toEqual(want);
    }
    // outside: 'zero' — taps beyond the root bbox read 0.
    expect(sampleTrilinearVec3(image, [-5000, -5000, -5000], { outside: "zero" })).toEqual([0, 0, 0]);
  });

  it("node stats follow GridStats Stats<Vec3f>: extremes by lengthSqr, ave/std of lengthSqr", () => {
    // Brute force over the root: every active leaf voxel (weight 1) and every
    // tile with its child-slot weight (lower 512, upper 4096, root 32768).
    const samples: { v: V3; w: number }[] = [];
    for (const leaf of grid.iterLeaves()) {
      for (let n = 0; n < 512; n++) if (isOn(leaf.valueMask, n)) samples.push({ v: Array.from(leaf.values.subarray(n * 3, n * 3 + 3)) as V3, w: 1 });
    }
    samples.push({ v: LOWER_TILE.value, w: 512 }, { v: UPPER_TILE.value, w: 4096 }, { v: ROOT_TILE.value, w: 32768 });
    const s2 = (v: V3): number => lengthSqrF32(v[0], v[1], v[2]);
    const minV = samples.reduce((a, b) => (s2(b.v) < s2(a.v) ? b : a)).v;
    const maxV = samples.reduce((a, b) => (s2(b.v) > s2(a.v) ? b : a)).v;
    const W = samples.reduce((a, b) => a + b.w, 0);
    const ave = samples.reduce((a, b) => a + b.w * s2(b.v), 0) / W;
    const variance = samples.reduce((a, b) => a + b.w * (s2(b.v) - ave) ** 2, 0) / W;

    const root = GRID_SIZE + Number(dv.getBigUint64(GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT, true));
    const vec = (off: number): number[] => [0, 1, 2].map((i) => dv.getFloat32(off + i * 4, true));
    const L = VEC3F_LAYOUT;
    expect(vec(root + L.rootOffMin)).toEqual(minV);
    expect(vec(root + L.rootOffMax)).toEqual(maxV);
    expect(dv.getFloat32(root + L.rootOffAve, true)).toBeCloseTo(ave, 2);
    expect(dv.getFloat32(root + L.rootOffStdDev, true) / Math.sqrt(variance)).toBeCloseTo(1, 5);
    expect(vec(root + L.rootOffBackground)).toEqual([0, 0, 0]);
  });

  it("writeNvdb round-trips the image (GridType 6 in the file metadata)", () => {
    const file = new Uint8Array(writeNvdb([image]));
    const fdv = new DataView(file.buffer);
    expect(fdv.getUint32(FILE_HEADER_SIZE + 32, true)).toBe(6); // FileMetaData.gridType
    expect(Number(fdv.getBigUint64(FILE_HEADER_SIZE + 24, true))).toBe(built.voxelCount); // voxelCount
    const bytes = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
    const start = file.length - bytes.length;
    expect(start).toBe(FILE_HEADER_SIZE + FILE_METADATA_SIZE + "density".length + 1);
    expect(file.subarray(start)).toEqual(bytes);
  });

  it("the scalar and vector CPU readers refuse each other's grids", () => {
    expect(() => readValue(image, [0, 0, 1])).toThrow(/readValueVec3/);
    const floatGrid = parseVdb(
      writeSyntheticVdb({ background: 0, rootTiles: [{ origin: [0, 0, 0], value: 1, active: true }], uppers: [] }),
    ).grids[0]! as VdbGrid;
    const floatImage = buildFromVdbDetailed(floatGrid).image;
    expect(() => readValueVec3(floatImage, [0, 0, 0])).toThrow(/not VEC3F/);
  });
});
