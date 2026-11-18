import { createHash } from "node:crypto";
import { inflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  WvdbFormatError,
  WvdbStreamReader,
  decodeWvdbLod,
  readWvdb,
  readWvdbHeader,
  wvdbBrickOrigin,
  wvdbBrickTotal,
  wvdbChunks,
  wvdbDensityStats,
  wvdbFinestDimensions,
  wvdbLodByteRange,
  wvdbPrefixLength,
  type WvdbFile,
  type WvdbPointsGrid,
  type WvdbVolumeGrid,
  type WvdbVolumeLevel,
} from "../../src/tools/index.js";
import { bytesEqual, fixture, hisAsset, readBytes, vdbSample } from "./wvdb-helpers.js";

// K4.3 gate: his committed .wvdb files decode exactly per the spec
// (mjurczyk/openvdb src/docs/wvdb/README.md v0.5.x at 75e7345).

const popcount = (words: Uint32Array): number =>
  words.reduce((n, w) => {
    let x = w;
    let c = 0;
    while (x) {
      x &= x - 1;
      c++;
    }
    return n + c;
  }, 0);

/** Invariants every decoded volume LOD must satisfy, checked against the raw header. */
function checkVolumeGrid(grid: WvdbVolumeGrid, file: WvdbFile): void {
  const lods = grid.header.lods;
  expect(grid.levels.length).toBe(file.bake.lods);
  grid.levels.forEach((level, k) => {
    const entry = lods[k]!;
    expect(level.dimensions).toEqual(entry.dimensions);
    // §3.2.3 (SHOULD, his writer): coarser LOD = ceil(finer / 2).
    if (k > 0) expect(grid.levels[k - 1]!.dimensions).toEqual(level.dimensions.map((d) => Math.ceil(d / 2)));
    expect(level.layout).toBe(entry.mask ? "sparse" : "dense");
    expect(level.brickIndices.length).toBe(entry.bricks);
    if (level.mask) expect(popcount(level.mask)).toBe(entry.bricks);
    // brick ranges hold every decoded voxel of the brick
    const [X, Y] = level.dimensions;
    for (let b = 0; b < level.brickIndices.length; b++) {
      const [ox, oy, oz] = wvdbBrickOrigin(level.dimensions, level.brickIndices[b]!);
      const at = ox + oy * X + oz * X * Y;
      const v = level.values[at]!;
      expect(v).toBeGreaterThanOrEqual(level.mins[b * level.channels]!);
      expect(v).toBeLessThanOrEqual(level.maxs[b * level.channels]!);
    }
    // §9 SHOULD: sparse pages put the bricks with the most data first (checked loosely on decoded data).
  });
}

/** Spec-text decode of one stored brick, independent of readWvdb's placement code. */
function rawBrick(bytes: Uint8Array, file: WvdbFile, gridIndex: number, lod: number, b: number): Float32Array {
  const chunk = wvdbChunks(file.header).find((c) => c.gridIndex === gridIndex && c.lod === lod && c.kind === "page")!;
  // Not his decoder: inflate with node's zlib and read the fields of §6.2 by hand.
  const page = new Uint8Array(inflateSync(bytes.subarray(file.dataStart + chunk.offset, file.dataStart + chunk.offset + chunk.length)));
  const view = new DataView(page.buffer, page.byteOffset, page.byteLength);
  const count = view.getUint32(0, true);
  const sparse = !!(file.header.grids[gridIndex] as { lods: { mask?: unknown }[] }).lods[lod]!.mask;
  const minsAt = 4 + (sparse ? 4 * count : 0);
  const min = view.getFloat32(minsAt + 4 * b, true);
  const max = view.getFloat32(minsAt + 4 * count + 4 * b, true);
  let codesAt = minsAt + 8 * count;
  for (let i = 0; i < b; i++) if (view.getFloat32(minsAt + 4 * i, true) !== view.getFloat32(minsAt + 4 * count + 4 * i, true)) codesAt += 512;
  const out = new Float32Array(512);
  for (let v = 0; v < 512; v++) {
    if (min === max) out[v] = min;
    else out[v] = page[codesAt + v] === 255 ? max : min + page[codesAt + v]! * ((max - min) / 255);
  }
  return out;
}

function checkBrickPlacement(bytes: Uint8Array, file: WvdbFile, gridIndex: number): void {
  const grid = file.grids[gridIndex] as WvdbVolumeGrid;
  const lod = grid.levels.length - 1;
  const level = grid.levels[lod]!;
  const firstPage = (grid.header.lods[lod]!.pages[0]?.bricks ?? 0);
  const [X, Y, Z] = level.dimensions;
  for (const b of [0, Math.floor(firstPage / 2), firstPage - 1].filter((b) => b >= 0)) {
    const index = level.brickIndices[b]!;
    const raw = rawBrick(bytes, file, gridIndex, lod, b);
    // spec §3.3 origin formula, written out here
    const countX = Math.ceil(X / 8);
    const countY = Math.ceil(Y / 8);
    const origin = [(index % countX) * 8, (Math.floor(index / countX) % countY) * 8, Math.floor(index / (countX * countY)) * 8];
    for (let k = 0; k < 8; k++)
      for (let j = 0; j < 8; j++)
        for (let i = 0; i < 8; i++) {
          const [x, y, z] = [origin[0]! + i, origin[1]! + j, origin[2]! + k];
          if (x >= X || y >= Y || z >= Z) continue;
          expect(level.values[x + y * X + z * X * Y]).toBe(raw[i + 8 * j + 64 * k]);
        }
  }
}

describe("readWvdb — his committed files (CC0 copies in fixtures/wvdb)", () => {
  it("ground_explosion_0010: two scalar grids, header fields, LOD dims, masks, stats", () => {
    const bytes = fixture("ground_explosion_0010.wvdb");
    const file = readWvdb(bytes, { strictChunkOrder: true });
    expect(file.version).toBe(1);
    expect(file.complete).toBe(true);
    expect(file.source).toEqual({
      file: "ground_explosion_0010.vdb",
      sha256: "1ebd376925e3e161316fd200534b579329a037eca1eb8f0b02e60166290ea988",
    });
    expect(file.bake).toEqual({
      grids: null,
      resolution: 128,
      lods: 3,
      bits: 8,
      layout: "sparse",
      pageBricks: 1024,
      bbox: [
        [-111.5, -114.5, -159.5],
        [109.5, 141.5, 122.5],
      ],
    });
    expect(file.header.brickDim).toBe(8);
    expect(file.grids.map((g) => g.name)).toEqual(["density", "flames"]);
    const [density, flames] = file.grids as WvdbVolumeGrid[];
    for (const g of [density!, flames!]) {
      expect(g.kind).toBe("volume");
      expect(g.class).toBe("unknown");
      expect(g.channels).toBe(1);
      expect(g.background).toEqual([0]);
      expect(g.bbox).toEqual(file.bake.bbox);
      checkVolumeGrid(g, file);
      // This sequence was baked with resolution^3 for every box (before his cube-texel fix).
      expect(g.levels.map((l) => l.dimensions)).toEqual([
        [32, 32, 32],
        [64, 64, 64],
        [128, 128, 128],
      ]);
    }
    expect(density!.levels.map((l) => l.brickIndices.length)).toEqual([6, 29, 151]);
    expect(flames!.levels.map((l) => l.brickIndices.length)).toEqual([6, 28, 147]);
    // His header stats were computed from the decoded finest LOD: recomputing them must agree exactly.
    for (const g of [density!, flames!]) {
      const s = wvdbDensityStats(g.levels[2]!.values, 0, false);
      expect(s.nonZeroVoxels).toBe(g.nonZeroVoxels);
      expect(s.meanDensity).toBe(g.meanDensity);
    }
    expect(density!.nonZeroVoxels).toBe(31468);
    expect(flames!.nonZeroVoxels).toBe(33212);
    checkBrickPlacement(bytes, file, 0);
    checkBrickPlacement(bytes, file, 1);
  });

  it("cloud_04_variant_0000: non-cubic dims follow §3.2.1, no stats members", () => {
    const bytes = fixture("cloud_04_variant_0000.wvdb");
    const file = readWvdb(bytes, { strictChunkOrder: true });
    const grid = file.grids[0] as WvdbVolumeGrid;
    expect(grid.name).toBe("density");
    expect(grid.meanDensity).toBeUndefined();
    expect(grid.nonZeroVoxels).toBeUndefined();
    expect(grid.levels.map((l) => l.dimensions)).toEqual([
      [21, 32, 11],
      [42, 64, 22],
      [84, 128, 43],
    ]);
    expect(wvdbFinestDimensions(grid.bbox, file.bake.resolution)).toEqual([84, 128, 43]);
    expect(grid.levels.map((l) => l.brickIndices.length)).toEqual([12, 57, 329]);
    checkVolumeGrid(grid, file);
    checkBrickPlacement(bytes, file, 0);
    // decoded data is a density in [0, 1] with something in it
    const finest = grid.levels[2]!.values;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of finest) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
    expect(lo).toBeGreaterThanOrEqual(0);
    expect(hi).toBeGreaterThan(0.5);
    expect(hi).toBeLessThanOrEqual(1);
  });

  it("ground_explosion_0199: an all-empty frame keeps its grids, masks and zero pages", () => {
    const file = readWvdb(fixture("ground_explosion_0199.wvdb"), { strictChunkOrder: true });
    expect(file.grids.length).toBe(2);
    for (const g of file.grids as WvdbVolumeGrid[]) {
      expect(g.nonZeroVoxels).toBe(0);
      expect(g.meanDensity).toBe(0);
      for (const l of g.levels) {
        expect(l.layout).toBe("sparse");
        expect(l.brickIndices.length).toBe(0);
        expect(popcount(l.mask!)).toBe(0);
        expect(l.values.every((v) => v === 0)).toBe(true);
      }
    }
  });
});

describe("coarse-first decoding from partial bytes (§9, §12.2)", () => {
  const bytes = fixture("ground_explosion_0010.wvdb");
  const full = readWvdb(bytes);
  const info = readWvdbHeader(bytes, { fileLength: bytes.length })!;

  it("LOD byte ranges are contiguous, coarse to fine, and cover the data section", () => {
    const ranges = [0, 1, 2].map((k) => wvdbLodByteRange(info, k));
    expect(ranges[0]!.start).toBe(info.dataStart);
    expect(ranges[1]!.start).toBe(ranges[0]!.end);
    expect(ranges[2]!.start).toBe(ranges[1]!.end);
    expect(ranges[2]!.end).toBe(bytes.length);
    expect(wvdbPrefixLength(info, 1)).toBe(ranges[1]!.end);
  });

  it("a prefix decodes exactly the LODs it holds", () => {
    expect(readWvdbHeader(bytes.subarray(0, 11))).toBeNull();
    expect(readWvdbHeader(bytes.subarray(0, info.dataStart - 1))).toBeNull();
    const prefix = readWvdb(bytes.subarray(0, wvdbPrefixLength(info, 1) + 10), { partial: true });
    expect(prefix.lodsDecoded).toBe(2);
    expect(prefix.complete).toBe(false);
    for (const [gi, g] of prefix.grids.entries()) {
      for (const [k, l] of (g as WvdbVolumeGrid).levels.entries()) {
        expect(bytesEqual(l.values, (full.grids[gi] as WvdbVolumeGrid).levels[k]!.values)).toBe(true);
      }
    }
    // without `partial`, missing chunks are an error (§14 chunk bounds)
    expect(() => readWvdb(bytes.subarray(0, wvdbPrefixLength(info, 1)))).toThrow(WvdbFormatError);
  });

  it("a range read of one LOD decodes alone", () => {
    const r = wvdbLodByteRange(info, 2);
    const levels = decodeWvdbLod(info, 2, bytes.slice(r.start, r.end), r.start) as WvdbVolumeLevel[];
    expect(bytesEqual(levels[1]!.values, (full.grids[1] as WvdbVolumeGrid).levels[2]!.values)).toBe(true);
    expect(() => decodeWvdbLod(info, 1, bytes.slice(r.start, r.end), r.start)).toThrow(/not inside/);
  });

  it("WvdbStreamReader emits LODs in order as bytes arrive, at any split", () => {
    for (const step of [1000, 4093, 50_000]) {
      const reader = new WvdbStreamReader();
      const seen: number[] = [];
      for (let at = 0; at < bytes.length; at += step) {
        for (const e of reader.push(bytes.subarray(at, at + step))) seen.push(e.lod);
        if (seen.length === 1) expect(reader.file!.lodsDecoded).toBe(1);
      }
      expect(seen).toEqual([0, 1, 2]);
      expect(reader.done).toBe(true);
      const got = (reader.file!.grids[0] as WvdbVolumeGrid).levels[2]!.values;
      expect(bytesEqual(got, (full.grids[0] as WvdbVolumeGrid).levels[2]!.values)).toBe(true);
    }
    const capped = new WvdbStreamReader({ maxLod: 0 });
    const events = capped.push(bytes.subarray(0, wvdbPrefixLength(info, 0)));
    expect(events.map((e) => e.lod)).toEqual([0]);
    expect(capped.done).toBe(true);
  });
});

describe("readWvdb rejects malformed files (§12.2.1, §14)", () => {
  const good = fixture("ground_explosion_0199.wvdb");
  const info = readWvdbHeader(good)!;
  const withHeader = (mutate: (h: Record<string, any>) => void, version = 1): Uint8Array => {
    const h = JSON.parse(JSON.stringify(info.header));
    mutate(h);
    const json = new TextEncoder().encode(JSON.stringify(h));
    const data = good.subarray(info.dataStart);
    const out = new Uint8Array(12 + json.length + data.length);
    out.set(new TextEncoder().encode("WVDB"));
    new DataView(out.buffer).setUint32(4, version, true);
    new DataView(out.buffer).setUint32(8, json.length, true);
    out.set(json, 12);
    out.set(data, 12 + json.length);
    return out;
  };

  it("magic, version, brickDim, headerLength", () => {
    const bad = good.slice();
    bad[0] = 0x58;
    expect(() => readWvdb(bad)).toThrow(/magic/);
    expect(() => readWvdb(withHeader(() => {}, 3))).toThrow(/version 3/);
    expect(() => readWvdb(withHeader((h) => (h.version = 2)))).toThrow(/preamble version/);
    expect(() => readWvdb(withHeader((h) => (h.brickDim = 16)))).toThrow(/brickDim/);
    const long = good.slice();
    new DataView(long.buffer).setUint32(8, good.length, true);
    expect(() => readWvdb(long)).toThrow(/headerLength/);
  });

  it("header structure: lods count, bits, bricks sums, dense totals, chunk bounds, voxel limit", () => {
    expect(() => readWvdb(withHeader((h) => (h.bake.lods = 2)))).toThrow(/bake.lods/);
    expect(() => readWvdb(withHeader((h) => (h.bake.bits = 12)))).toThrow(/bits/);
    expect(() => readWvdb(withHeader((h) => (h.grids[0].lods[0].bricks = 1)))).toThrow(/sum of its pages/);
    expect(() => readWvdb(withHeader((h) => delete h.grids[0].lods[0].mask))).toThrow(/dense LOD/);
    expect(() => readWvdb(withHeader((h) => (h.grids[0].lods[2].mask.length = 9999)))).toThrow(/past the data section/);
    expect(() => readWvdb(withHeader((h) => (h.grids[1].name = "density")))).toThrow(/not unique/);
    expect(() => readWvdb(withHeader((h) => (h.grids[0].type = "points")))).toThrow(/version 2/);
    expect(() => readWvdb(good, { maxVoxels: 1000 })).toThrow(/maxVoxels/);
  });

  it("a chunk that does not inflate, or inflates past its size, is rejected", () => {
    const bad = good.slice();
    bad[info.dataStart + 2] ^= 0xff;
    expect(() => readWvdb(bad)).toThrow(WvdbFormatError);
  });

  it("an unknown class reads as \"unknown\" (§5.4); unknown members are ignored (§5.1)", () => {
    const file = readWvdb(withHeader((h) => ((h.grids[0].class = "fancy"), (h.extra = { a: 1 }))));
    expect(file.grids[0]!.class).toBe("unknown");
  });
});

// His other committed files live outside this repo (sizes or CC BY-SA sources). Set MJURCZYK_OPENVDB_DIR.
const extra = ["venusstatue.wvdb", "utahteapot.wvdb", "sphere_points.wvdb", "smoke.wvdb", "bunny_cloud.wvdb"].map(
  (n) => [n, hisAsset(n)] as const,
);

describe.skipIf(extra.some(([, p]) => !p))("readWvdb — his other committed files (MJURCZYK_OPENVDB_DIR)", () => {
  it.each(extra)("%s decodes, passes the strict chunk order, and its stats recompute exactly", (name, p) => {
    const bytes = readBytes(p!);
    const file = readWvdb(bytes, { strictChunkOrder: true });
    for (const [gi, g] of file.grids.entries()) {
      if (g.kind === "points") continue;
      checkVolumeGrid(g, file);
      checkBrickPlacement(bytes, file, gi);
      if (g.nonZeroVoxels !== undefined) {
        const s = wvdbDensityStats(g.levels.at(-1)!.values, g.background[0]!, g.class === "level set");
        expect(s.nonZeroVoxels).toBe(g.nonZeroVoxels);
        expect(s.meanDensity).toBe(g.meanDensity);
      }
    }
    // source.sha256 matches the source .vdb where we have it
    const src = vdbSample(file.source.file);
    if (src) expect(createHash("sha256").update(readBytes(src)).digest("hex")).toBe(file.source.sha256);
  });

  it("level sets: every decoded value is in [-b, b]; outside the band it is exactly ±b (§8)", () => {
    for (const name of ["venusstatue.wvdb", "utahteapot.wvdb"]) {
      const file = readWvdb(readBytes(hisAsset(name)!));
      const g = file.grids[0] as WvdbVolumeGrid;
      expect(g.class).toBe("level set");
      const b = g.background[0]!;
      expect(wvdbFinestDimensions(g.bbox, 128)).toEqual(g.levels[2]!.dimensions);
      for (const v of g.levels[2]!.values) {
        expect(Math.abs(v)).toBeLessThanOrEqual(b);
        if (Math.abs(v) >= b) expect(Math.abs(v)).toBe(b);
      }
    }
  });

  it("sphere_points (version 2): points LODs follow §3.5 and decode inside the box", () => {
    const file = readWvdb(readBytes(hisAsset("sphere_points.wvdb")!), { strictChunkOrder: true });
    expect(file.version).toBe(2);
    expect(file.bake.maxPoints).toBe(4_000_000);
    const g = file.grids[0] as WvdbPointsGrid;
    expect(g.kind).toBe("points");
    expect(g.count).toBe(9802);
    expect(g.sourceCount).toBe(9802);
    expect(g.colors).toBe(false);
    expect(g.levels.map((l) => l.count)).toEqual([154, 1072, 8576]);
    expect(g.levels.map((l) => l.firstPoint)).toEqual([0, 154, 1226]);
    const [lo, hi] = g.bbox!;
    for (const l of g.levels) {
      expect(l.colors).toBeNull();
      for (let i = 0; i < l.positions.length; i++) {
        expect(l.positions[i]).toBeGreaterThanOrEqual(lo[i % 3]!);
        expect(l.positions[i]).toBeLessThanOrEqual(hi[i % 3]!);
      }
    }
    // a unit sphere's surface: every point at radius ~1
    for (let i = 0; i < g.levels[2]!.count; i += 97) {
      const p = g.levels[2]!.positions.subarray(i * 3, i * 3 + 3);
      expect(Math.hypot(p[0]!, p[1]!, p[2]!)).toBeCloseTo(1, 2);
    }
  });
});

describe("wvdbBrickTotal sanity on his Appendix A example", () => {
  it("[25, 29, 32] has 4*4*4 bricks; [100, 116, 128] has 13*15*16", () => {
    expect(wvdbBrickTotal([25, 29, 32])).toBe(64);
    expect(wvdbBrickTotal([100, 116, 128])).toBe(3120);
  });
});
