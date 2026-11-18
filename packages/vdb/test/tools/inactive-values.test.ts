/**
 * K4.8 / K4.9 / K4.12 / K4.14: inactive values, inactive tiles, root-table key
 * order, file-version and metadata hardening.
 *
 * - `readCompressedValues` restores inactive values exactly as OpenVDB's
 *   `io::readCompressedValues` (Compression.h): `inactiveVal1 = background`,
 *   `inactiveVal0 = (meta == 0) ? background : -background`, overridden by the
 *   one or two values stored for metadata 2, 4 and 5; a selection mask picks
 *   between them for 3, 4 and 5. Checked per mode for scalar and vec3 values.
 * - The level-set samples (skipped when absent) must hold only +/-background in
 *   their inactive leaf voxels, and the teapot's interior must read -background.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { buildFromVdbDetailed, readValue, type VdbFile } from "../../src/tools/index.js";
import { VdbUnsupportedError } from "../../src/tools/vdb/errors.js";
import { ByteReader } from "../../src/tools/vdb/byte-reader.js";
import { FLOAT_FORMAT, VEC3S_FORMAT, readCompressedValues, type ValueFormat } from "../../src/tools/vdb/compression.js";
import { readMetadataMap } from "../../src/tools/vdb/metadata.js";
import { parseVdb } from "../../src/tools/vdb/index.js";
import { writeSyntheticVdb, type SynthLeaf } from "./synthetic-vdb.js";

type Coord = [number, number, number];

// ---------------------------------------------------------------------------
// K4.8: per-mode inactive value restoration
// ---------------------------------------------------------------------------

const N = 512;
/** Active slots: every third voxel. */
const ACTIVE = (i: number): boolean => i % 3 === 0;
/** Selection mask: every fifth voxel selects inactiveVal1. */
const SELECT = (i: number): boolean => i % 5 === 0;

function maskBytes(pred: (i: number) => boolean): Uint8Array {
  const out = new Uint8Array(N / 8);
  for (let i = 0; i < N; i++) if (pred(i)) out[i >> 3]! |= 1 << (i & 7);
  return out;
}

function f32Bytes(values: readonly number[]): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  const dv = new DataView(out.buffer);
  values.forEach((v, i) => dv.setFloat32(i * 4, v, true));
  return out;
}

function concat(...parts: Uint8Array[]): ArrayBuffer {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out.buffer;
}

const valueMask = (() => {
  const words = new Uint32Array(16);
  for (let i = 0; i < N; i++) if (ACTIVE(i)) words[i >>> 5]! |= 1 << (i & 31);
  return words;
})();
const activeCount = Array.from({ length: N }, (_, i) => i).filter(ACTIVE).length;

interface Mode {
  meta: number;
  /** Stored full-width inactive values (one scalar each; vec3 spreads it to [s, s+1, s+2]). */
  stored: number[];
  hasMask: boolean;
  /** Expected inactiveVal0 / inactiveVal1 given the background and the stored values. */
  expect: (bg: number[], st: number[][]) => { v0: number[]; v1: number[] };
}

const neg = (v: number[]): number[] => v.map((c) => -c);
const MODES: Record<string, Mode> = {
  NoMaskOrInactiveVals: { meta: 0, stored: [], hasMask: false, expect: (bg) => ({ v0: bg, v1: bg }) },
  NoMaskAndMinusBg: { meta: 1, stored: [], hasMask: false, expect: (bg) => ({ v0: neg(bg), v1: bg }) },
  NoMaskAndOneInactiveVal: { meta: 2, stored: [7.5], hasMask: false, expect: (bg, st) => ({ v0: st[0]!, v1: bg }) },
  MaskAndNoInactiveVals: { meta: 3, stored: [], hasMask: true, expect: (bg) => ({ v0: neg(bg), v1: bg }) },
  MaskAndOneInactiveVal: { meta: 4, stored: [-3.25], hasMask: true, expect: (bg, st) => ({ v0: st[0]!, v1: bg }) },
  MaskAndTwoInactiveVals: { meta: 5, stored: [4.5, -6.5], hasMask: true, expect: (_bg, st) => ({ v0: st[0]!, v1: st[1]! }) },
};

function decode(mode: Mode, format: ValueFormat, bg: number[]): { values: Float32Array; v0: number[]; v1: number[] } {
  const comps = format.components;
  const stored = mode.stored.map((s) => (comps === 1 ? [s] : [s, s + 1, s + 2]));
  const { v0: e0, v1: e1 } = mode.expect(bg, stored);

  const active: number[] = [];
  for (let i = 0; i < N; i++) {
    if (!ACTIVE(i)) continue;
    for (let c = 0; c < comps; c++) active.push(100 + i + c * 0.5);
  }
  const parts: Uint8Array[] = [Uint8Array.of(mode.meta)];
  for (const s of stored) parts.push(f32Bytes(s));
  if (mode.hasMask) parts.push(maskBytes(SELECT));
  parts.push(f32Bytes(active));

  const values = readCompressedValues(new ByteReader(concat(...parts)), {
    fileVersion: 223,
    compression: { zip: false, activeMask: true, blosc: false },
    useHalf: false,
    background: bg,
    numValues: N,
    valueMask,
    format,
  });
  // Active values come back in slot order.
  let at = 0;
  for (let i = 0; i < N; i++) {
    if (ACTIVE(i)) {
      for (let c = 0; c < comps; c++) expect(values[i * comps + c]).toBe(100 + i + c * 0.5);
      at++;
    }
  }
  expect(at).toBe(activeCount);
  return { values, v0: e0, v1: e1 };
}

describe("K4.8: readCompressedValues restores inactive values like OpenVDB", () => {
  for (const [label, format, bg] of [
    ["scalar", FLOAT_FORMAT, [0.75]],
    ["vec3", VEC3S_FORMAT, [0.75, -1.5, 2]],
  ] as [string, ValueFormat, number[]][]) {
    for (const [name, mode] of Object.entries(MODES)) {
      it(`${label}, ${name}`, () => {
        const { values, v0, v1 } = decode(mode, format, bg);
        const comps = format.components;
        for (let i = 0; i < N; i++) {
          if (ACTIVE(i)) continue;
          const want = mode.hasMask && SELECT(i) ? v1 : v0;
          expect(Array.from(values.subarray(i * comps, (i + 1) * comps))).toEqual(want);
        }
      });
    }
  }

  it("no mask compression and NoMaskAndAllVals leave every value as written", () => {
    const all = Array.from({ length: N }, (_, i) => i * 0.5);
    const values = readCompressedValues(new ByteReader(concat(Uint8Array.of(6), f32Bytes(all))), {
      fileVersion: 223,
      compression: { zip: false, activeMask: true, blosc: false },
      useHalf: false,
      background: [9],
      numValues: N,
      valueMask,
    });
    expect(Array.from(values)).toEqual(all);
  });
});

// ---------------------------------------------------------------------------
// K4.8 / K4.9 on the level-set samples
// ---------------------------------------------------------------------------

const here = path.dirname(fileURLToPath(import.meta.url));
const sample = (name: string): string => path.resolve(here, "../../../../fixtures/vdb-samples", name);

function parseSample(name: string): VdbFile {
  const buf = readFileSync(sample(name));
  return parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
}

describe.each(["bunny.vdb", "utahteapot.vdb", "icosahedron.vdb"])("level-set sample %s", (name) => {
  const present = existsSync(sample(name));
  it.skipIf(!present)("inactive leaf voxels hold +/-background, never 0", () => {
    const file = parseSample(name);
    let inactive = 0;
    let negative = 0;
    let bad = 0;
    for (const grid of file.grids) {
      if (grid.valueType !== "float") continue;
      const bg = grid.background as number;
      for (const leaf of grid.iterLeaves()) {
        for (let n = 0; n < N; n++) {
          if ((leaf.valueMask[n >>> 5]! >>> (n & 31)) & 1) continue;
          inactive++;
          const v = leaf.values[n]!;
          if (Math.abs(v) !== Math.abs(bg)) bad++;
          if (v < 0) negative++;
        }
      }
    }
    expect(bad).toBe(0);
    expect(inactive).toBeGreaterThan(0);
    expect(negative).toBeGreaterThan(0);
  }, 60_000);
});

describe("utahteapot interior (K4.9)", () => {
  it.skipIf(!existsSync(sample("utahteapot.vdb")))("reads -background, inactive, inside the surface", () => {
    const file = parseSample("utahteapot.vdb");
    const grid = file.grids[0]!;
    if (grid.valueType !== "float") throw new Error("expected a float grid");
    const bg = grid.background;
    expect(bg).toBeGreaterThan(0);
    // Inactive tiles in the interior carry -background.
    const tiles = [...grid.iterInactiveTiles()];
    expect(tiles.length).toBeGreaterThan(0);
    expect(tiles.every((t) => t.value === -bg)).toBe(true);
    const t = tiles[0]!;
    const inside: Coord = [t.origin[0] + 1, t.origin[1] + 1, t.origin[2] + 1];
    expect(grid.readValue(inside)).toEqual({ value: -bg, active: false });

    // Through NanoVDB: the same point reads -background, inactive.
    const built = buildFromVdbDetailed(grid);
    expect(readValue(built.image, inside)).toEqual({ value: -bg, active: false });
    // Far away reads +background.
    expect(grid.readValue([100000, 0, 0])).toEqual({ value: bg, active: false });
    expect(readValue(built.image, [100000, 0, 0])).toEqual({ value: bg, active: false });
    // Active voxel count is unchanged by inactive tiles.
    expect(built.voxelCount).toBe(Number(grid.activeVoxelCount));
  });
});

// ---------------------------------------------------------------------------
// K4.9 synthetic: inactive tiles at every level, value + state through to NanoVDB
// ---------------------------------------------------------------------------

describe("K4.9 synthetic inactive tiles", () => {
  const leaf: SynthLeaf = {
    origin: [0, 0, 0],
    values: new Float32Array(512).fill(2),
    valueMask: new Uint32Array(16).fill(0xffffffff),
  };
  const buf = writeSyntheticVdb({
    fileVersion: 222,
    background: 2,
    rootTiles: [{ origin: [8192, 0, 0], value: -2, active: false }],
    uppers: [
      {
        origin: [0, 0, 0],
        tiles: [{ origin: [256, 0, 0], value: -2, active: false }],
        lowers: [
          {
            origin: [0, 0, 0],
            tiles: [
              { origin: [16, 0, 0], value: -2, active: false },
              // Inactive tile equal to the background: reads like no tile, not kept.
              { origin: [24, 0, 0], value: 2, active: false },
            ],
            leaves: [leaf],
          },
        ],
      },
    ],
  });
  const grid = parseVdb(buf).grids[0]!;
  if (grid.valueType !== "float") throw new Error("expected float");
  const built = buildFromVdbDetailed(grid);

  it("keeps value and inactive state through parse and readValue", () => {
    expect([...grid.iterInactiveTiles()].map((t) => [t.dim, t.value]).sort()).toEqual([[128, -2], [4096, -2], [8, -2]].sort());
    for (const ijk of [[8192, 7, 7], [259, 1, 1], [18, 0, 0]] as Coord[]) {
      expect(grid.readValue(ijk)).toEqual({ value: -2, active: false });
    }
    expect(grid.readValue([25, 0, 0])).toEqual({ value: 2, active: false });
    expect(grid.activeVoxelCount).toBe(512n);
    expect([...grid.iterTiles()]).toHaveLength(0);
  });

  it("carries them into NanoVDB with value, no active state and no stats or counts", () => {
    for (const ijk of [[8192, 7, 7], [259, 1, 1], [18, 0, 0]] as Coord[]) {
      expect(readValue(built.image, ijk)).toEqual({ value: -2, active: false });
    }
    expect(readValue(built.image, [25, 0, 0])).toEqual({ value: 2, active: false });
    expect(built.tileCounts).toEqual({ lower: 0, upper: 0, root: 0 });
    expect(built.voxelCount).toBe(512);
    expect(built.indexBBox).toEqual({ min: [0, 0, 0], max: [7, 7, 7] });
  });
});

// ---------------------------------------------------------------------------
// K4.12: root entries in unsigned key order
// ---------------------------------------------------------------------------

describe("K4.12 root table key order", () => {
  const mk = (origin: Coord): SynthLeaf => ({
    origin,
    values: new Float32Array(512).fill(1),
    valueMask: new Uint32Array(16).fill(1),
  });
  const origins: Coord[] = [
    [-4096, 0, 0],
    [0, 0, 0],
    [0, -4096, 0],
    [0, 0, -4096],
    [4096, 4096, 4096],
    [-8192, -4096, 0],
  ];
  const spec = {
    fileVersion: 222 as const,
    background: 0,
    rootTiles: [],
    uppers: origins.map((o) => ({
      origin: o,
      tiles: [],
      lowers: [{ origin: o, tiles: [], leaves: [mk(o)] }],
    })),
  };
  it("sorts by the u64 NanoVDB key, so negative coordinates come after non-negative ones", () => {
    const grid = parseVdb(writeSyntheticVdb(spec)).grids[0]!;
    const { image } = buildFromVdbDetailed(grid);
    const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);
    const TREE = 672;
    const rootOff = TREE + Number(dv.getBigUint64(TREE + 24, true));
    const n = dv.getUint32(rootOff + 24, true);
    expect(n).toBe(origins.length);
    const keys: bigint[] = [];
    for (let i = 0; i < n; i++) keys.push(dv.getBigUint64(rootOff + 64 + i * 32, true));
    expect(keys).toEqual([...keys].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)));
    expect(new Set(keys).size).toBe(origins.length);
    // The signed-origin order would put [-8192,...] first; the key order puts it after [4096,...].
    const key = (o: Coord): bigint => BigInt(o[2] >>> 12) | (BigInt(o[1] >>> 12) << 21n) | (BigInt(o[0] >>> 12) << 42n);
    expect(keys).toEqual(origins.map(key).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)));
    expect(keys[0]).toBe(key([0, 0, 0]));
  });
});

// ---------------------------------------------------------------------------
// K4.14: file version ceiling, prototype-safe metadata
// ---------------------------------------------------------------------------

describe("K4.14 parser hardening", () => {
  it("rejects file versions above 225 (as OpenVDB does) and accepts 225's header", () => {
    const buf = writeSyntheticVdb({ fileVersion: 222, background: 0, rootTiles: [], uppers: [] });
    const patch = (version: number): ArrayBuffer => {
      const copy = buf.slice(0);
      new DataView(copy).setUint32(8, version, true);
      return copy;
    };
    expect(() => parseVdb(patch(226))).toThrow(VdbUnsupportedError);
    expect(() => parseVdb(patch(226))).toThrow(/226.*maximum supported version 225/);
    expect(() => parseVdb(patch(0xffffffff))).toThrow(VdbUnsupportedError);
    // 225 passes the version gate (it may fail later on the synthetic payload, but not as unsupported-version).
    try {
      parseVdb(patch(225));
    } catch (e) {
      expect(String((e as Error).message)).not.toMatch(/maximum supported version/);
    }
  });

  it("readMetadataMap keeps a __proto__ name as an own key and does not pollute prototypes", () => {
    const enc = new TextEncoder();
    const str = (s: string): Uint8Array => {
      const b = enc.encode(s);
      const out = new Uint8Array(4 + b.length);
      new DataView(out.buffer).setUint32(0, b.length, true);
      out.set(b, 4);
      return out;
    };
    const u32 = (n: number): Uint8Array => {
      const out = new Uint8Array(4);
      new DataView(out.buffer).setUint32(0, n, true);
      return out;
    };
    const value = enc.encode("evil");
    const bytes = concat(u32(2), str("__proto__"), str("string"), u32(value.length), value, str("class"), str("string"), u32(3), enc.encode("fog"));
    const map = readMetadataMap(new ByteReader(bytes));
    expect(Object.getPrototypeOf(map)).toBeNull();
    expect(Object.keys(map)).toEqual(["__proto__", "class"]);
    expect(Object.hasOwn(map, "__proto__")).toBe(true);
    expect(map["__proto__"]).toBe("evil");
    expect(map["class"]).toBe("fog");
    expect(({} as Record<string, unknown>)["class"]).toBeUndefined();
    expect(Object.getPrototypeOf({})).toBe(Object.prototype);
    expect(({} as { evil?: unknown }).evil).toBeUndefined();
  });
});
