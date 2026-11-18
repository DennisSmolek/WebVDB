/**
 * K4.2 point data grids (`ptdataidx32`) without downloaded assets.
 *
 * Synthetic files (test/tools/synthetic-points.ts) hold two leaves with ten
 * points and attributes in every codec the decoder handles — `P` as fxpt16,
 * colour as ufxpt8, int32 ids, a half-truncated float, unit vectors, a
 * uniform array, a strided array, a group array — plus one attribute it does
 * not handle. They are written plain (no compression) and as Blosc files with
 * raw or Blosc pages small enough to split the arrays over several pages.
 *
 * Oracles: the values the test stored, decoded with OpenVDB's codec formulas
 * (`FixedPointCodec`, `TruncateCodec`, `QuantizedUnitVec`) written out here
 * independently; world position = index-to-world(ijk + P).
 */

import { describe, expect, it } from "vitest";

import { buildFromVdb, parseVdb, type VdbPointDataGrid } from "../../src/tools/index.js";
import { unpackUnitVec } from "../../src/tools/vdb/points.js";
import { writeSyntheticPoints, type SynthPointLeaf, type SynthPointSpec } from "./synthetic-points.js";
import { floatToHalf } from "./synthetic-vdb.js";

type Coord = [number, number, number];

const fround = Math.fround;
const half = (x: number): number => {
  const h = floatToHalf(x);
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  const s = h & 0x8000 ? -1 : 1;
  return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
};

/** Points per voxel: leaf 0 has 2 + 1 + 3 points, leaf 1 has 4. */
const LEAF_POINTS: { origin: Coord; voxels: [number, number][] }[] = [
  { origin: [0, 0, 0], voxels: [[0, 2], [9, 1], [511, 3]] },
  { origin: [8, 16, 0], voxels: [[100, 4]] },
];

function offsetsFor(voxels: [number, number][]): Uint32Array {
  const counts = new Uint32Array(512);
  for (const [n, c] of voxels) counts[n] = c;
  const offsets = new Uint32Array(512);
  let total = 0;
  for (let n = 0; n < 512; n++) offsets[n] = total += counts[n]!;
  return offsets;
}

const ATTRS = [
  { name: "P", type: "vec3s", codec: "fxpt16" },
  { name: "Cd", type: "vec3s", codec: "ufxpt8" },
  { name: "id", type: "int32", codec: "null" },
  { name: "density", type: "float", codec: "trnc" },
  { name: "N", type: "vec3s", codec: "uvec" },
  { name: "pscale", type: "float", codec: "null", uniform: true },
  { name: "w", type: "float", codec: "null", stride: 2 },
  { name: "__groups0", type: "uint8", codec: "grp" },
  { name: "odd", type: "vec3d", codec: "fxpt8" }, // not a real OpenVDB pair: skipped
];
const GROUPS = [
  { name: "red", offset: 0 },
  { name: "big", offset: 3 },
];

/** Stored code / value for global point index `p`. */
const pCode = (p: number, c: number): number => (p * 7919 + c * 104729) % 65536;
const cdCode = (p: number, c: number): number => (p * 37 + c * 91) % 256;
const unitCodes = [0, 255, 0x8000 | 255, 0x2000, 0x4000 | (5 << 7) | 9, 0x1fff, 1234, 4321, 0x7abc, 0xe001];

function leafArrays(first: number, count: number): Uint8Array[] {
  const u16 = (n: number, f: (i: number) => number): Uint8Array => {
    const b = new Uint8Array(n * 2);
    const v = new DataView(b.buffer);
    for (let i = 0; i < n; i++) v.setUint16(i * 2, f(i), true);
    return b;
  };
  const f32 = (n: number, f: (i: number) => number): Uint8Array => {
    const b = new Uint8Array(n * 4);
    const v = new DataView(b.buffer);
    for (let i = 0; i < n; i++) v.setFloat32(i * 4, f(i), true);
    return b;
  };
  const i32 = (n: number, f: (i: number) => number): Uint8Array => {
    const b = new Uint8Array(n * 4);
    const v = new DataView(b.buffer);
    for (let i = 0; i < n; i++) v.setInt32(i * 4, f(i), true);
    return b;
  };
  return [
    u16(count * 3, (i) => pCode(first + Math.floor(i / 3), i % 3)),
    Uint8Array.from({ length: count * 3 }, (_, i) => cdCode(first + Math.floor(i / 3), i % 3)),
    i32(count, (i) => 1000 + first + i),
    u16(count, (i) => floatToHalf((first + i) * 0.375 - 1)),
    u16(count, (i) => unitCodes[first + i]!),
    f32(1, () => 0.125 * (first + 1)),
    f32(count * 2, (i) => first * 10 + i * 0.5),
    Uint8Array.from({ length: count }, (_, i) => (first + i) % 2 | (((first + i) % 3 === 0 ? 1 : 0) << 3)),
    new Uint8Array(count * 3),
  ];
}

function leaves(): SynthPointLeaf[] {
  let first = 0;
  return LEAF_POINTS.map(({ origin, voxels }) => {
    const offsets = offsetsFor(voxels);
    const count = offsets[511]!;
    const leaf = { origin, offsets, arrays: leafArrays(first, count) };
    first += count;
    return leaf;
  });
}

const VOXEL_SIZE = 0.5;
const TRANSLATE: Coord = [1, 2, 3];

function spec(over: Partial<SynthPointSpec>): SynthPointSpec {
  return {
    voxelSize: VOXEL_SIZE,
    translate: TRANSLATE,
    attributes: ATTRS,
    groups: GROUPS,
    leaves: leaves(),
    mode: "plain",
    ...over,
  };
}

/** Each point's voxel ijk, in file order. */
function pointVoxels(): Coord[] {
  const out: Coord[] = [];
  for (const { origin, voxels } of LEAF_POINTS) {
    for (const [n, c] of voxels) {
      for (let k = 0; k < c; k++) out.push([origin[0] + (n >> 6), origin[1] + ((n >> 3) & 7), origin[2] + (n & 7)]);
    }
  }
  return out;
}

/** `QuantizedUnitVec::unpack`, written out from the header's formula. */
function unitVec(data: number): number[] {
  let x = (data & 0x1f80) >> 7;
  let y = data & 0x7f;
  if (x + y > 126) {
    x = 127 - x;
    y = 127 - y;
  }
  const z = 126 - x - y;
  const w = fround(1 / Math.sqrt(x * x + y * y + z * z));
  return [
    (data & 0x8000 ? -1 : 1) * fround(x * w),
    (data & 0x4000 ? -1 : 1) * fround(y * w),
    (data & 0x2000 ? -1 : 1) * fround(z * w),
  ];
}

const VARIANTS: [string, Partial<SynthPointSpec>][] = [
  ["plain v222", { mode: "plain", fileVersion: 222 }],
  ["plain, unshared descriptors", { mode: "plain", sharedDescriptor: false }],
  ["paged, raw pages, raw voxels", { mode: "paged", pageSize: 1 << 20 }],
  ["paged, small raw pages", { mode: "paged", pageSize: 24 }],
  ["paged, Blosc pages + voxels", { mode: "paged", pageSize: 40, pageBlosc: true, voxelBlosc: true }],
];

describe.each(VARIANTS)("point data grid (%s)", (_, over) => {
  const file = parseVdb(writeSyntheticPoints(spec(over)));
  const grid = file.grids[0]! as VdbPointDataGrid;

  it("parses the tree: voxel end offsets, masks, counts, descriptor", () => {
    expect(grid.valueType).toBe("ptdataidx32");
    expect(grid.pointCount).toBe(10);
    expect(grid.activeVoxelCount).toBe(4n);
    expect(grid.attributes.map((a) => a.name)).toEqual(ATTRS.map((a) => a.name));
    expect(grid.attributes[1]).toEqual({ name: "Cd", type: "vec3s", codec: "ufxpt8", index: 1 });
    expect(grid.groupNames.sort()).toEqual(["big", "red"]);
    expect(grid.readValue([0, 0, 0])).toEqual({ value: 2, active: true });
    expect(grid.readValue([0, 1, 1])).toEqual({ value: 3, active: true }); // voxel 9
    expect(grid.readValue([0, 1, 2])).toEqual({ value: 3, active: false });
    expect(grid.readValue([7, 7, 7])).toEqual({ value: 6, active: true }); // voxel 511
    expect(grid.readValue([9, 20, 4])).toEqual({ value: 4, active: true }); // leaf 1, voxel 100
    expect(grid.readValue([50, 50, 50])).toEqual({ value: 0, active: false });
    expect(() => buildFromVdb(grid)).toThrow(/point data grid/);
  });

  it("decodes world positions from fxpt16 P: transform(ijk + code/65535 - 0.5)", () => {
    const pts = grid.points();
    expect(pts.count).toBe(10);
    pointVoxels().forEach((ijk, p) => {
      for (let c = 0; c < 3; c++) {
        const offset = fround(fround(pCode(p, c) / 65535) - 0.5);
        expect(pts.positions[p * 3 + c]).toBe(fround(VOXEL_SIZE * (ijk[c]! + offset) + TRANSLATE[c]!));
      }
    });
  });

  it("decodes every supported attribute codec", () => {
    const pts = grid.points();
    expect(Object.keys(pts.attributes).sort()).toEqual(["Cd", "N", "density", "id", "pscale", "w"]);
    expect(pts.skipped.map((s) => s.name)).toEqual(["odd"]);
    expect(pts.attributeShapes["Cd"]).toEqual({ components: 3, stride: 1 });
    expect(pts.attributeShapes["w"]).toEqual({ components: 1, stride: 2 });
    const firsts = [0, 0, 0, 0, 0, 0, 6, 6, 6, 6];
    for (let p = 0; p < 10; p++) {
      for (let c = 0; c < 3; c++) expect(pts.attributes["Cd"]![p * 3 + c]).toBe(fround(cdCode(p, c) / 255));
      expect(pts.attributes["id"]![p]).toBe(1000 + p);
      expect(pts.attributes["density"]![p]).toBe(half(p * 0.375 - 1));
      expect(Array.from(pts.attributes["N"]!.subarray(p * 3, p * 3 + 3) as Float32Array)).toEqual(unitVec(unitCodes[p]!));
      expect(pts.attributes["pscale"]![p]).toBe(0.125 * (firsts[p]! + 1));
      const local = p - firsts[p]!;
      for (let s = 0; s < 2; s++) expect(pts.attributes["w"]![p * 2 + s]).toBe(firsts[p]! * 10 + (local * 2 + s) * 0.5);
      expect(pts.groups["red"]![p]).toBe(p % 2);
      expect(pts.groups["big"]![p]).toBe(p % 3 === 0 ? 1 : 0);
    }
    expect(pts.attributes["id"]).toBeInstanceOf(Int32Array);
    expect(grid.points()).toBe(pts); // cached
  });
});

describe("unpackUnitVec (QuantizedUnitVec::unpack)", () => {
  it("decodes the axes and signs OpenVDB's pack() produces", () => {
    const out = new Float32Array(3);
    // pack((1,0,0)): xbits 126 > 63 -> complement (1, 127) = 255
    unpackUnitVec(255, out, 0);
    expect(Array.from(out)).toEqual([1, 0, 0]);
    unpackUnitVec(0x8000 | 255, out, 0);
    expect(Array.from(out)).toEqual([-1, 0, 0]);
    // pack((0,1,0)): ybits 126 -> 126
    unpackUnitVec(126, out, 0);
    expect(Array.from(out)).toEqual([0, 1, 0]);
    // pack((0,0,-1)): z sign only
    unpackUnitVec(0x2000, out, 0);
    expect(Array.from(out)).toEqual([0, 0, -1]);
    // a diagonal stays (close to) unit length
    unpackUnitVec((42 << 7) | 42, out, 0);
    expect(Math.hypot(out[0]!, out[1]!, out[2]!)).toBeCloseTo(1, 6);
  });
});
