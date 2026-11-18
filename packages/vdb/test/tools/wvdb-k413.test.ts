/**
 * K4.13 `.wvdb` follow-ups:
 *  (a) `levelSetSampler: 'mjurczyk'` (his `sampleLevelSet`) reproduces his
 *      committed level-set files (utahteapot from a digest + our pinned
 *      sample; venusstatue from his checkout via MJURCZYK_OPENVDB_DIR);
 *  (b) the stream reader decodes page by page with bounded memory and can emit
 *      partial LODs; hostile streams fail with `WvdbFormatError`;
 *  (c) `sampling: 'box'` prefilters with the K4.4 box filter.
 */

import { createHash } from "node:crypto";
import { unzlibSync as inflate } from "fflate";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  WvdbFormatError,
  WvdbStreamReader,
  bakeWvdb,
  buildFromLeavesDetailed,
  buildFromVdb,
  nanoVdbToWvdb,
  parseVdb,
  readValue,
  readWvdb,
  readWvdbHeader,
  resampleDense,
  sampleWvdbDense,
  sampleWvdbDenseBox,
  sampleWvdbLevelSetMjurczyk,
  vdbToWvdb,
  wvdbChunkByteLimit,
  wvdbChunks,
  wvdbSourceFromNanoVdb,
  wvdbSourceFromVdbGrid,
  writeWvdb,
  type WvdbLodEvent,
  type WvdbPointsGrid,
  type WvdbVolumeGrid,
  type WvdbVolumeLevel,
} from "../../src/tools/index.js";
import { FLOAT_LEAF_CODEC } from "../../src/tools/nanovdb/leaf-codec.js";
import type { LeafInput } from "../../src/tools/nanovdb/tree.js";
import { WVDB_FIXTURES, bytesEqual, fixture, headerWithoutOffsets, hisAsset, readBytes, vdbSample } from "./wvdb-helpers.js";

const identity = (b: Uint8Array): Uint8Array => b;
type Vec3 = [number, number, number];

// ---------------------------------------------------------------------------
// (a) level-set sampler
// ---------------------------------------------------------------------------

/** A sphere level set (radius r, background b): active band |sdf| < b, inactive ±b outside it. */
function sphereLevelSet(r: number, b: number, n: number): Uint32Array {
  const leaves: LeafInput[] = [];
  const c = n / 2;
  for (let ox = 0; ox < n; ox += 8) {
    for (let oy = 0; oy < n; oy += 8) {
      for (let oz = 0; oz < n; oz += 8) {
        const values = new Float32Array(512);
        const valueMask = new Uint32Array(16);
        let any = false;
        for (let i = 0; i < 512; i++) {
          const d = Math.hypot(ox + (i >> 6) - c, oy + ((i >> 3) & 7) - c, oz + (i & 7) - c) - r;
          values[i] = Math.abs(d) < b ? d : d < 0 ? -b : b;
          if (Math.abs(d) < b) {
            valueMask[i >>> 5]! |= 1 << (i & 31);
            any = true;
          }
        }
        if (any || Math.hypot(ox + 4 - c, oy + 4 - c, oz + 4 - c) < r) leaves.push({ origin: [ox, oy, oz], values, valueMask });
      }
    }
  }
  return buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC, { background: b, gridClass: "LevelSet", gridName: "ls", voxelSize: 0.5 }).image;
}

describe("levelSetSampler 'mjurczyk' (his sampleLevelSet)", () => {
  const b = 1.5; // 3 voxels of 0.5
  const image = sphereLevelSet(10, b, 32);
  const source = wvdbSourceFromNanoVdb(image);
  const box = source.bbox!;

  it("stays in [-b, b], keeps |v| < b one step inside the band, and is ±b far from the surface", () => {
    for (const dims of [[16, 16, 16], [64, 64, 64]] as Vec3[]) {
      const v = sampleWvdbLevelSetMjurczyk(source, box, dims, 8);
      const band = b - (2 * b) / 255;
      let inside = 0;
      let outside = 0;
      for (const x of v) {
        expect(Math.abs(x)).toBeLessThanOrEqual(b);
        if (Math.abs(x) < b) expect(Math.abs(x)).toBeLessThanOrEqual(Math.fround(band));
        if (x === -b) inside++;
        if (x === b) outside++;
      }
      expect(inside).toBeGreaterThan(0);
      expect(outside).toBeGreaterThan(0);
    }
  });

  it("texels finer than a voxel are a trilinear sample of the clamped distance at the texel centre", () => {
    const dims: Vec3 = [64, 64, 64]; // texel 0.5 voxel: no 4x4x4 filter
    const v = sampleWvdbLevelSetMjurczyk(source, box, dims, 16);
    const min = box[0];
    const size = [0, 1, 2].map((a) => box[1][a]! - min[a]!);
    const dist = (ijk: Vec3): number => {
      const r = readValue(image, ijk.map(Math.round) as Vec3);
      return r.active ? Math.min(Math.max(r.value, -b), b) : r.value < 0 ? -b : b;
    };
    let checked = 0;
    for (let k = 0; k < 4000; k++) {
      const x = (k * 7) % 64;
      const y = (k * 13) % 64;
      const z = (k * 29) % 64;
      const p = [x, y, z].map((t, a) => (min[a]! + ((t + 0.5) / dims[a]!) * size[a]!) / 0.5) as Vec3; // world -> index (voxel 0.5, no offset)
      let want = dist(p);
      if (Math.abs(want) < b) {
        const f = p.map(Math.floor) as Vec3;
        const w = p.map((t, a) => t - f[a]!);
        want = 0;
        for (let i = 0; i < 8; i++) {
          const d = [i & 1, (i >> 1) & 1, (i >> 2) & 1];
          const weight = d.reduce((acc, on, a) => acc * (on ? w[a]! : 1 - w[a]!), 1);
          if (weight > 0) want += weight * dist([f[0] + d[0]!, f[1] + d[1]!, f[2] + d[2]!]);
        }
        checked++;
      }
      const got = v[x + y * 64 + z * 64 * 64]!;
      const band = b - (2 * b) / 65535;
      const expected = Math.abs(want) < b ? Math.min(Math.max(want, -band), band) : want;
      expect(got).toBeCloseTo(expected, 5);
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("texels coarser than a voxel that cross the surface are the mean of 4x4x4 sub-samples", () => {
    const dims: Vec3 = [8, 8, 8]; // ~2.5 voxels per texel
    const v = sampleWvdbLevelSetMjurczyk(source, box, dims, 16);
    const spec = sampleWvdbDense(source, box, dims, 16);
    // Crossing texels are smooth means; the spec point sample only sees in/out there.
    let differ = 0;
    for (let i = 0; i < v.length; i++) if (v[i] !== spec[i]) differ++;
    expect(differ).toBeGreaterThan(0);
    // The centre texel is deep inside: -b by both.
    expect(v[4 + 4 * 8 + 4 * 64]).toBe(-b);
  });

  it("bakeWvdb: 'spec' stays the default; 'mjurczyk' only changes level sets", () => {
    const a = nanoVdbToWvdb(image, { compress: identity, resolution: 32 });
    const s = nanoVdbToWvdb(image, { compress: identity, resolution: 32, levelSetSampler: "spec" });
    const m = nanoVdbToWvdb(image, { compress: identity, resolution: 32, levelSetSampler: "mjurczyk" });
    expect(bytesEqual(a.bytes, s.bytes)).toBe(true);
    expect(bytesEqual(a.bytes, m.bytes)).toBe(false);
    expect(() => nanoVdbToWvdb(image, { levelSetSampler: "x" as "spec" })).toThrow(/levelSetSampler/);
    const noIndex = { ...source, index: undefined };
    expect(() => bakeWvdb([noIndex], { levelSetSampler: "mjurczyk", resolution: 8 })).toThrow(/index access/);
  });
});

/** His level-set files: inflated chunks and header (offsets zeroed) of our 'mjurczyk' bake vs his. */
function expectReproduces(vdbBytes: Uint8Array, fileName: string, his: { header: unknown; chunks: { inflatedBytes: number; inflatedSha256: string }[] }): void {
  const ours = vdbToWvdb(vdbBytes, { fileName, levelSetSampler: "mjurczyk", compress: identity });
  expect(headerWithoutOffsets(ours.header)).toEqual(his.header);
  const dataStart = 12 + new DataView(ours.bytes.buffer, ours.bytes.byteOffset).getUint32(8, true);
  const chunks = wvdbChunks(ours.header);
  expect(chunks.length).toBe(his.chunks.length);
  chunks.forEach((c, i) => {
    const raw = ours.bytes.subarray(dataStart + c.offset, dataStart + c.offset + c.length);
    expect(raw.length).toBe(his.chunks[i]!.inflatedBytes);
    expect(createHash("sha256").update(raw).digest("hex")).toBe(his.chunks[i]!.inflatedSha256);
  });
}

function digestOfHisFile(bytes: Uint8Array): { header: unknown; chunks: { inflatedBytes: number; inflatedSha256: string }[] } {
  const file = readWvdb(bytes);
  const { header } = file;
  const info = readWvdbHeader(bytes, { fileLength: bytes.length })!;
  const raw = (c: ReturnType<typeof wvdbChunks>[number]): Uint8Array =>
    inflate(bytes.subarray(info.dataStart + c.offset, info.dataStart + c.offset + c.length));
  return {
    header: headerWithoutOffsets(header),
    chunks: wvdbChunks(header).map((c) => {
      const r = raw(c);
      return { inflatedBytes: r.length, inflatedSha256: createHash("sha256").update(r).digest("hex") };
    }),
  };
}


const teapotVdb = vdbSample("utahteapot.vdb");
describe.skipIf(!teapotVdb)("utahteapot.vdb → .wvdb with levelSetSampler 'mjurczyk' reproduces his utahteapot.wvdb", () => {
  it("same header (offsets zeroed) and every inflated chunk (digest)", () => {
    const digest = JSON.parse(readFileSync(path.join(WVDB_FIXTURES, "utahteapot.wvdb.digest.json"), "utf8"));
    expectReproduces(readBytes(teapotVdb!), "utahteapot.vdb", { header: digest.headerWithoutOffsets, chunks: digest.chunks });
  }, 30_000);

  it("the NanoVDB path gives the same bytes as the .vdb path", () => {
    const bytes = readBytes(teapotVdb!);
    const file = parseVdb(bytes.slice().buffer);
    const a = vdbToWvdb(file, { levelSetSampler: "mjurczyk", compress: identity });
    const b = nanoVdbToWvdb(buildFromVdb(file.grids[0]!), { levelSetSampler: "mjurczyk", compress: identity });
    expect(bytesEqual(a.bytes, b.bytes)).toBe(true);
  }, 30_000);
});

const hisVenus = hisAsset("venusstatue.vdb");
const hisVenusWvdb = hisAsset("venusstatue.wvdb");
describe.skipIf(!hisVenus || !hisVenusWvdb)("venusstatue.vdb → .wvdb reproduces his venusstatue.wvdb (MJURCZYK_OPENVDB_DIR)", () => {
  it("same header and every inflated chunk; the spec sampler does not", () => {
    const his = digestOfHisFile(readBytes(hisVenusWvdb!));
    const vdb = readBytes(hisVenus!);
    expectReproduces(vdb, "venusstatue.vdb", his);
    const spec = vdbToWvdb(vdb, { fileName: "venusstatue.vdb" });
    const a = (readWvdb(spec.bytes).grids[0] as WvdbVolumeGrid).levels[2]!.values;
    const b = (readWvdb(readBytes(hisVenusWvdb!)).grids[0] as WvdbVolumeGrid).levels[2]!.values;
    expect(bytesEqual(a, b)).toBe(false);
  }, 30_000);
});

const hisTeapotWvdb = hisAsset("utahteapot.wvdb");
describe.skipIf(!hisTeapotWvdb)("the committed utahteapot digest is his file's (MJURCZYK_OPENVDB_DIR)", () => {
  it("digest matches", () => {
    const digest = JSON.parse(readFileSync(path.join(WVDB_FIXTURES, "utahteapot.wvdb.digest.json"), "utf8"));
    const bytes = readBytes(hisTeapotWvdb!);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(digest.fileSha256);
    const his = digestOfHisFile(bytes);
    expect(his.header).toEqual(digest.headerWithoutOffsets);
    expect(his.chunks).toEqual(digest.chunks.map((c: { inflatedBytes: number; inflatedSha256: string }) => ({ inflatedBytes: c.inflatedBytes, inflatedSha256: c.inflatedSha256 })));
  });
});

// ---------------------------------------------------------------------------
// (c) box sampling
// ---------------------------------------------------------------------------

describe("sampling 'box' (K4.4 box filter on his lattice)", () => {
  // 16^3 voxels with a smooth field, voxel size 1, origin 0.
  const leaves: LeafInput[] = [];
  const field = (x: number, y: number, z: number): number => Math.fround(1 + 0.1 * x + 0.01 * y * y + 0.03 * ((x * y + z) % 5));
  for (let ox = 0; ox < 16; ox += 8) for (let oy = 0; oy < 16; oy += 8) for (let oz = 0; oz < 16; oz += 8) {
    const values = new Float32Array(512);
    for (let i = 0; i < 512; i++) values[i] = field(ox + (i >> 6), oy + ((i >> 3) & 7), oz + (i & 7));
    leaves.push({ origin: [ox, oy, oz], values, valueMask: new Uint32Array(16).fill(0xffffffff) });
  }
  const image = buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC, { gridName: "density" }).image;
  const source = wvdbSourceFromNanoVdb(image);

  it("a downsample by 2 on an aligned box is the exact 2x2x2 mean", () => {
    const box: [Vec3, Vec3] = [[-0.5, -0.5, -0.5], [15.5, 15.5, 15.5]];
    const v = sampleWvdbDenseBox(source, box, [8, 8, 8]);
    for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      let sum = 0;
      for (let i = 0; i < 8; i++) sum += field(2 * x + (i & 1), 2 * y + ((i >> 1) & 1), 2 * z + (i >> 2));
      expect(v[x + y * 8 + z * 64]).toBeCloseTo(sum / 8, 5);
    }
  });

  it("conserves mass on his default lattice where point sampling does not, and bakes", () => {
    const dims: Vec3 = [5, 5, 5]; // texels of 3 voxels: point sampling skips 2 of 3
    const box = source.bbox!;
    const texel = (15 / 5) ** 3;
    let mass = 0;
    for (let x = 0; x < 16; x++) for (let y = 0; y < 16; y++) for (let z = 0; z < 16; z++) mass += field(x, y, z);
    // His box is the voxel-centre box [0, 15]: it covers 15^3 of the 16^3 cell volume, so compare to that part.
    const boxV = sampleWvdbDenseBox(source, box, dims);
    const pointV = sampleWvdbDense(source, box, dims);
    const sum = (a: Float32Array): number => a.reduce((s, x) => s + x, 0) * texel;
    let inner = 0;
    for (let x = 0; x < 16; x++) for (let y = 0; y < 16; y++) for (let z = 0; z < 16; z++) {
      const w = (t: number): number => (t === 0 || t === 15 ? 0.5 : 1);
      inner += field(x, y, z) * w(x) * w(y) * w(z);
    }
    expect(sum(boxV) / inner).toBeCloseTo(1, 4);
    expect(Math.abs(sum(pointV) / inner - 1)).toBeGreaterThan(Math.abs(sum(boxV) / inner - 1));
    expect(mass).toBeGreaterThan(inner);
    const baked = nanoVdbToWvdb(image, { sampling: "box", resolution: 5 });
    const decoded = (readWvdb(baked.bytes).grids[0] as WvdbVolumeGrid).levels[2]!.values;
    for (let i = 0; i < decoded.length; i++) expect(Math.abs(decoded[i]! - boxV[i]!)).toBeLessThan(0.02);
  });

  it("a one-voxel-thick box axis samples the nearest voxel there; errors are clear", () => {
    const flat: [Vec3, Vec3] = [[0, 0, 5], [15, 15, 5]];
    const v = resampleDense(image, { bbox: flat, dimensions: [16, 16, 1], filter: "box" });
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      // x/y texels are 15/16 voxel wide: a blend of neighbours, but z is exactly voxel 5.
      expect(v.active[x + y * 16]).toBe(1);
    }
    const mid = resampleDense(image, { bbox: [[3, 3, 5], [3, 3, 5]], dimensions: [1, 1, 1] });
    expect(mid.values[0]).toBe(field(3, 3, 5));
    expect(() => bakeWvdb([{ ...source, grid: undefined }], { sampling: "box", resolution: 4 })).toThrow(/needs the source grid/);
    expect(() => bakeWvdb([source], { sampling: "linear" as "box" })).toThrow(/unknown sampling/);
    expect(() => resampleDense(image, { bbox: [[0, 0, 0], [-1, 0, 0]], dimensions: [1, 1, 1] })).toThrow(/invalid bbox/);
  });
});

const smokeVdb = vdbSample("smoke.vdb");
describe.skipIf(!smokeVdb)("smoke.vdb: box vs nearest bake", () => {
  it("box keeps the mass of the source within 1%; nearest is the committed point bake", () => {
    const file = parseVdb(readBytes(smokeVdb!).slice().buffer);
    const grid = file.grids[0]!;
    const source = wvdbSourceFromVdbGrid(grid);
    const box = source.bbox!;
    const dims: Vec3 = [64, 64, 64].map((d, a) => Math.max(1, Math.round(((box[1][a]! - box[0][a]!) * 64) / Math.max(...[0, 1, 2].map((i) => box[1][i]! - box[0][i]!))))) as Vec3;
    const vs = grid.transform.matrix[0]!;
    let mass = 0;
    for (const leaf of grid.iterLeaves()) {
      for (let n = 0; n < 512; n++) if ((leaf.valueMask[n >>> 5]! >>> (n & 31)) & 1) mass += leaf.values[n]!;
    }
    mass *= vs ** 3;
    const texelVolume = [0, 1, 2].reduce((p, a) => p * ((box[1][a]! - box[0][a]!) / dims[a]!), 1);
    const total = (v: Float32Array): number => v.reduce((s, x) => s + x, 0) * texelVolume;
    const boxRatio = total(sampleWvdbDenseBox(source, box, dims)) / mass;
    const pointRatio = total(sampleWvdbDense(source, box, dims)) / mass;
    // His box is the voxel-centre box: half a voxel of the shell is outside it.
    expect(Math.abs(boxRatio - 1)).toBeLessThan(0.01);
    expect(Math.abs(pointRatio - 1)).toBeGreaterThan(Math.abs(boxRatio - 1));
  }, 30_000);
});

// ---------------------------------------------------------------------------
// (b) page-by-page stream decode
// ---------------------------------------------------------------------------

/** Re-writes a fixture with small pages, so each LOD has several page chunks. */
function manyPages(name: string, pageBricks = 3): Uint8Array {
  const file = readWvdb(fixture(name));
  const grids = file.grids.map((g) => {
    const v = g as WvdbVolumeGrid;
    return {
      name: v.name,
      class: v.class,
      channels: v.channels,
      background: v.channels === 1 ? v.background[0]! : v.background,
      bbox: v.bbox,
      levels: v.levels.map((l) => ({ dimensions: l.dimensions, values: l.values })),
      stats: false as const,
    };
  });
  return writeWvdb(grids, { lods: file.bake.lods, pageBricks, source: file.source }).bytes;
}

function streamAll(bytes: Uint8Array, step: number, reader = new WvdbStreamReader({ partialLods: true })): WvdbLodEvent[] {
  const events: WvdbLodEvent[] = [];
  for (let at = 0; at < bytes.length; at += step) events.push(...reader.push(bytes.subarray(at, at + step)));
  reader.end();
  return events;
}

describe("WvdbStreamReader: page by page within a LOD (§12.2), bounded memory", () => {
  const bytes = manyPages("ground_explosion_0010.wvdb");
  const full = readWvdb(bytes);
  const info = readWvdbHeader(bytes, { fileLength: bytes.length })!;
  const chunks = wvdbChunks(info.header);
  const maxChunk = Math.max(...chunks.map((c) => c.length));

  it("has several pages per LOD", () => {
    const pages = chunks.filter((c) => c.kind === "page" && c.lod === 2 && c.gridIndex === 0);
    expect(pages.length).toBeGreaterThan(3);
  });

  it("emits a partial event per chunk, then the LOD; partial levels fill in and end equal to readWvdb", () => {
    for (const step of [1, 97, 1000, 65_536]) {
      const reader = new WvdbStreamReader({ partialLods: true });
      const events = streamAll(bytes, step, reader);
      const partial = events.filter((e) => !e.complete);
      const complete = events.filter((e) => e.complete);
      expect(partial.length).toBe(chunks.length);
      expect(complete.map((e) => e.lod)).toEqual([0, 1, 2]);
      expect(partial.map((e) => e.chunk!.offset)).toEqual(chunks.map((c) => c.offset));
      for (const [gi, g] of reader.file!.grids.entries()) {
        for (const [k, l] of (g as WvdbVolumeGrid).levels.entries()) {
          const want = (full.grids[gi] as WvdbVolumeGrid).levels[k]!;
          expect(bytesEqual(l.values, want.values)).toBe(true);
          expect(bytesEqual(l.brickIndices, want.brickIndices)).toBe(true);
          expect(l.bricksDecoded).toBe(want.brickIndices.length);
        }
      }
      // Memory: at most one chunk (or the header before it parses) plus one push.
      expect(reader.peakBufferedBytes).toBeLessThanOrEqual(Math.max(maxChunk, info.dataStart) + step);
      expect(reader.bufferedBytes).toBe(0);
    }
  });

  it("a partial LOD holds exactly the bricks decoded so far", () => {
    const reader = new WvdbStreamReader({ partialLods: true });
    let checks = 0;
    let mismatches = 0;
    for (let at = 0; at < bytes.length; at += 2048) {
      const events = reader.push(bytes.subarray(at, at + 2048));
      // Levels are live (later pages of the same push are in too): check the last page event of the push.
      const e = events.filter((x) => !x.complete && x.chunk!.kind === "page").at(-1);
      if (!e) continue;
      const level = e.levels[e.chunk!.gridIndex] as WvdbVolumeLevel;
      if (level.bricksDecoded === level.brickIndices.length) continue; // only partial LODs
      const want = (full.grids[e.chunk!.gridIndex] as WvdbVolumeGrid).levels[e.lod]!;
      const [X, Y] = level.dimensions;
      const [cx, cy] = level.brickCounts;
      // Pages decode in file order, so the decoded bricks are the first `bricksDecoded` of the file order.
      expect(level.bricksDecoded).toBe(e.chunk!.first + e.chunk!.count);
      const decoded = new Set(Array.from(level.brickIndices.subarray(0, level.bricksDecoded)));
      const bg = (full.grids[e.chunk!.gridIndex] as WvdbVolumeGrid).background[0]!;
      for (let i = 0; i < level.values.length; i++) {
        const x = i % X;
        const y = Math.floor(i / X) % Y;
        const z = Math.floor(i / (X * Y));
        const brick = (x >> 3) + (y >> 3) * cx + (z >> 3) * cx * cy;
        if (level.values[i] !== (decoded.has(brick) ? want.values[i] : bg)) mismatches++;
      }
      checks++;
    }
    expect(mismatches).toBe(0);
    expect(checks).toBeGreaterThan(3);
  });

  it("without partialLods: one event per LOD, as before", () => {
    const reader = new WvdbStreamReader();
    const events = streamAll(bytes, 777, reader);
    expect(events.map((e) => [e.lod, e.complete])).toEqual([[0, true], [1, true], [2, true]]);
  });

  it("points grids stream too", () => {
    const n = 5000;
    const positions = new Float32Array(n * 3).map((_, i) => Math.sin(i * 0.37));
    const colors = new Float32Array(n * 3).map((_, i) => (i % 11) / 10);
    const file = writeWvdb([{ type: "points", name: "pts", positions, colors }], { lods: 3 }).bytes;
    const whole = readWvdb(file);
    const reader = new WvdbStreamReader({ partialLods: true });
    streamAll(file, 333, reader);
    const g = reader.file!.grids[0] as WvdbPointsGrid;
    g.levels.forEach((l, k) => {
      const w = (whole.grids[0] as WvdbPointsGrid).levels[k]!;
      expect(bytesEqual(l.positions, w.positions)).toBe(true);
      expect(bytesEqual(l.colors!, w.colors!)).toBe(true);
      expect(l.pointsDecoded).toBe(l.count);
      expect(l.firstPoint).toBe(w.firstPoint);
    });
  });

  it("chunks out of §9 order still decode, and LODs still complete coarse first", () => {
    // Put LOD 2's chunks before LOD 0's and LOD 1's in the data section.
    const h = JSON.parse(JSON.stringify(info.header));
    const refs: { lod: number; ref: { offset: number; length: number } }[] = [];
    for (let lod = 0; lod < h.bake.lods; lod++) {
      for (const g of h.grids) {
        const e = g.lods[lod];
        if (e.mask) refs.push({ lod, ref: e.mask });
        for (const p of e.pages) refs.push({ lod, ref: p });
      }
    }
    const data = bytes.subarray(info.dataStart);
    const order = [...refs.filter((r) => r.lod === 2), ...refs.filter((r) => r.lod !== 2)];
    const parts: Uint8Array[] = [];
    let at = 0;
    for (const r of order) {
      parts.push(data.slice(r.ref.offset, r.ref.offset + r.ref.length));
      r.ref.offset = at;
      at += r.ref.length;
    }
    const json = new TextEncoder().encode(JSON.stringify(h));
    const out = new Uint8Array(12 + json.length + at);
    out.set(bytes.subarray(0, 8));
    new DataView(out.buffer).setUint32(8, json.length, true);
    out.set(json, 12);
    let o = 12 + json.length;
    for (const p of parts) {
      out.set(p, o);
      o += p.length;
    }
    const reader = new WvdbStreamReader({ partialLods: true });
    const events = streamAll(out, 4096, reader);
    expect(events.filter((e) => e.complete).map((e) => e.lod)).toEqual([0, 1, 2]);
    expect(events.find((e) => !e.complete)!.lod).toBe(2);
    const g = reader.file!.grids[0] as WvdbVolumeGrid;
    expect(bytesEqual(g.levels[2]!.values, (full.grids[0] as WvdbVolumeGrid).levels[2]!.values)).toBe(true);
    expect(bytesEqual(g.levels[2]!.brickIndices, (full.grids[0] as WvdbVolumeGrid).levels[2]!.brickIndices)).toBe(true);
  });
});

describe("WvdbStreamReader: hostile streams", () => {
  const good = manyPages("ground_explosion_0010.wvdb");
  const info = readWvdbHeader(good, { fileLength: good.length })!;
  const withHeader = (mutate: (h: Record<string, any>) => void): Uint8Array => {
    const h = JSON.parse(JSON.stringify(info.header));
    mutate(h);
    const json = new TextEncoder().encode(JSON.stringify(h));
    const data = good.subarray(info.dataStart);
    const out = new Uint8Array(12 + json.length + data.length);
    out.set(good.subarray(0, 8));
    new DataView(out.buffer).setUint32(8, json.length, true);
    out.set(json, 12);
    out.set(data, 12 + json.length);
    return out;
  };

  it("a header longer than maxHeaderBytes fails as soon as the preamble is in", () => {
    const big = good.slice(0, 12);
    new DataView(big.buffer).setUint32(8, 0x7fffffff, true);
    const reader = new WvdbStreamReader();
    expect(() => reader.push(big)).toThrow(/maxHeaderBytes/);
    expect(() => new WvdbStreamReader({ maxHeaderBytes: 10 }).push(good)).toThrow(/maxHeaderBytes/);
    // The preamble split over pushes is still read whole.
    const split = new WvdbStreamReader();
    expect(split.push(good.subarray(0, 5))).toEqual([]);
    expect(split.push(good.subarray(5, 11))).toEqual([]);
    expect(split.push(good.subarray(11))).toHaveLength(3);
  });

  it("a chunk longer than any deflate of its content is refused at the header, before buffering it", () => {
    const bad = withHeader((h) => (h.grids[0].lods[0].pages[0].length = 1 << 30));
    const reader = new WvdbStreamReader();
    expect(() => reader.push(bad.subarray(0, 12 + new DataView(bad.buffer).getUint32(8, true)))).toThrow(/more than a deflate/);
    expect(wvdbChunkByteLimit(1000)).toBe(1000 + 1 + 1024);
  });

  it("a corrupt chunk is a WvdbFormatError, and the reader rethrows it afterwards", () => {
    const bad = good.slice();
    const c = wvdbChunks(info.header).find((x) => x.lod === 1 && x.kind === "page")!;
    bad[info.dataStart + c.offset + 2] ^= 0xff;
    bad[info.dataStart + c.offset + 3] ^= 0xff;
    const reader = new WvdbStreamReader({ partialLods: true });
    let err: unknown = null;
    try {
      streamAll(bad, 100, reader);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(WvdbFormatError);
    expect(() => reader.push(new Uint8Array(1))).toThrow(err as Error);
    expect(() => reader.end()).toThrow(err as Error);
    expect(reader.file!.lodsDecoded).toBe(1);
  });

  it("truncation: end() names the missing LOD; a mask that disagrees with the pages fails at the LOD's end", () => {
    const reader = new WvdbStreamReader();
    reader.push(good.subarray(0, good.length - 10));
    expect(() => reader.end()).toThrow(/before LOD 2/);
    const early = new WvdbStreamReader();
    early.push(good.subarray(0, 8));
    expect(() => early.end()).toThrow(/header/);

    // Swap LOD 0's mask chunk with grid 1's (same size, different bits).
    const h = JSON.parse(JSON.stringify(info.header));
    const [m0, m1] = [h.grids[0].lods[2].mask, h.grids[1].lods[2].mask];
    if (m0.length === m1.length) {
      [h.grids[0].lods[2].mask, h.grids[1].lods[2].mask] = [m1, m0];
      const json = new TextEncoder().encode(JSON.stringify(h));
      const out = new Uint8Array(12 + json.length + good.length - info.dataStart);
      out.set(good.subarray(0, 8));
      new DataView(out.buffer).setUint32(8, json.length, true);
      out.set(json, 12);
      out.set(good.subarray(info.dataStart), 12 + json.length);
      expect(() => streamAll(out, 1000)).toThrow(WvdbFormatError);
    }
  });

  it("200 random single-byte corruptions: a WvdbFormatError or a clean decode, never another error", () => {
    let seed = 1234;
    const rand = (): number => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
    let failures = 0;
    for (let k = 0; k < 200; k++) {
      const bad = good.slice();
      const at = Math.floor(rand() * bad.length);
      bad[at] = (bad[at]! + 1 + Math.floor(rand() * 255)) & 0xff;
      try {
        streamAll(bad, 1 + Math.floor(rand() * 5000), new WvdbStreamReader({ partialLods: k % 2 === 0, maxVoxels: 1 << 22 }));
      } catch (e) {
        if (!(e instanceof WvdbFormatError)) throw new Error(`corruption at ${at}: ${String(e)}`);
        failures++;
      }
    }
    expect(failures).toBeGreaterThan(50);
  });
});
