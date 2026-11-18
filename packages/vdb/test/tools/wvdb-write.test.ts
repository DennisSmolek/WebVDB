import { constants, deflateSync, inflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  downsampleWvdbLevel,
  readWvdb,
  wvdbBrickOrigin,
  wvdbChunks,
  writeWvdb,
  type WvdbFile,
  type WvdbGridInput,
  type WvdbPointsGrid,
  type WvdbVec3,
  type WvdbVolumeGrid,
  type WriteWvdbOptions,
} from "../../src/tools/internal.js";
import { bytesEqual, fixture, headerWithoutOffsets, hisAsset, readBytes } from "./wvdb-helpers.js";

const nodeZlib9 = (b: Uint8Array): Uint8Array => new Uint8Array(deflateSync(b, { level: constants.Z_BEST_COMPRESSION }));

function field(dims: WvdbVec3, fn: (x: number, y: number, z: number) => number, channels = 1): Float32Array {
  const [X, Y, Z] = dims;
  const out = new Float32Array(X * Y * Z * channels);
  for (let c = 0; c < channels; c++)
    for (let z = 0; z < Z; z++)
      for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) out[c * X * Y * Z + x + y * X + z * X * Y] = fn(x + c * 100, y, z);
  return out;
}

/** A soft blob in [0, 1] with exact zeros outside radius r. */
const blob = (dims: WvdbVec3, r: number) =>
  field(dims, (x, y, z) => {
    const d = Math.hypot(x - dims[0] / 2, y - dims[1] / 2, z - dims[2] / 2) / r;
    return d < 1 ? Math.fround(1 - d) : 0;
  });

/** Re-encode a decoded file with the same settings (decoded values, optional original brick order). */
function reencode(file: WvdbFile, keepOrder: boolean, compress?: (b: Uint8Array) => Uint8Array) {
  const h = file.header;
  const grids: WvdbGridInput[] = file.grids.map((g) => {
    if (g.kind === "points") {
      // Undo the §3.5 LOD split to recover the stored order.
      const out = { positions: new Float32Array(g.count * 3), colors: g.colors ? new Float32Array(g.count * 3) : null };
      const n = g.levels.length;
      g.levels.forEach((l, lod) => {
        const stride = 8 ** (n - 1 - lod);
        let k = 0;
        for (let p = 0; p < g.count; p += stride) {
          if (lod !== 0 && p % (stride * 8) === 0) continue;
          out.positions.set(l.positions.subarray(k * 3, k * 3 + 3), p * 3);
          out.colors?.set(l.colors!.subarray(k * 3, k * 3 + 3), p * 3);
          k++;
        }
      });
      return { type: "points", name: g.name, class: g.header.class, ...out, sourceCount: g.sourceCount, spacing: g.spacing };
    }
    return {
      name: g.name,
      class: g.header.class,
      channels: g.channels,
      background: g.header.background,
      bbox: g.bbox,
      levels: g.levels.map((l) => ({ dimensions: l.dimensions, values: l.values, ...(keepOrder ? { brickOrder: l.brickIndices } : {}) })),
      stats: g.meanDensity === undefined ? false : { meanDensity: g.meanDensity, nonZeroVoxels: g.nonZeroVoxels! },
    };
  });
  return writeWvdb(grids, {
    lods: h.bake.lods,
    bits: h.bake.bits,
    layout: h.bake.layout,
    pageBricks: h.bake.pageBricks,
    resolution: h.bake.resolution,
    gridNames: h.bake.grids,
    bakeBBox: h.bake.bbox ?? null,
    ...(h.bake.maxPoints !== undefined ? { maxPoints: h.bake.maxPoints } : {}),
    source: h.source,
    ...(compress ? { compress } : {}),
  });
}

function inflatedChunks(bytes: Uint8Array, file: WvdbFile): Uint8Array[] {
  return wvdbChunks(file.header).map(
    (c) => new Uint8Array(inflateSync(bytes.subarray(file.dataStart + c.offset, file.dataStart + c.offset + c.length))),
  );
}

function expectSameDecoded(a: WvdbFile, b: WvdbFile): void {
  expect(a.grids.length).toBe(b.grids.length);
  a.grids.forEach((g, gi) => {
    const h = b.grids[gi]!;
    g.levels.forEach((l, k) => {
      const m = h.levels[k]!;
      if (l.kind === "volume" && m.kind === "volume") {
        expect(m.dimensions).toEqual(l.dimensions);
        expect(bytesEqual(m.values, l.values)).toBe(true);
      } else if (l.kind === "points" && m.kind === "points") {
        expect(bytesEqual(m.positions, l.positions)).toBe(true);
        if (l.colors) expect(bytesEqual(m.colors!, l.colors)).toBe(true);
      } else {
        throw new Error("kind mismatch");
      }
    });
  });
}

describe("writeWvdb → readWvdb round trip", () => {
  const cases: { name: string; grids: WvdbGridInput[]; opts: WriteWvdbOptions }[] = [
    {
      name: "scalar, sparse, 8-bit, odd dims, several pages",
      grids: [{ name: "density", class: "fog volume", background: 0, bbox: [[0, 0, 0], [1, 2, 1]], finest: { dimensions: [21, 37, 19], values: blob([21, 37, 19], 9) } }],
      opts: { pageBricks: 3 },
    },
    {
      name: "scalar, dense, 16-bit",
      grids: [{ name: "temperature", background: 0.5, bbox: null, finest: { dimensions: [17, 9, 12], values: field([17, 9, 12], (x, y, z) => Math.sin(x * 0.3) * y + z) } }],
      opts: { layout: "dense", bits: 16, lods: 2 },
    },
    {
      name: "vector grid (3 channels) + scalar grid",
      grids: [
        { name: "Cd", channels: 3, background: [0, 0.25, 1], bbox: [[-1, -1, -1], [1, 1, 1]], finest: { dimensions: [16, 16, 16], values: field([16, 16, 16], (x, y, z) => ((x * 7 + y * 3 + z) % 11) / 10, 3) } },
        { name: "density", background: 0, bbox: [[-1, -1, -1], [1, 1, 1]], finest: { dimensions: [16, 16, 16], values: blob([16, 16, 16], 6) } },
      ],
      opts: { lods: 4 },
    },
    {
      name: "level set class",
      grids: [{ name: "ls", class: "level set", background: 0.3, bbox: [[0, 0, 0], [3, 3, 3]], finest: { dimensions: [24, 24, 24], values: field([24, 24, 24], (x, y, z) => Math.max(-0.3, Math.min(0.3, (Math.hypot(x - 12, y - 12, z - 12) - 8) * 0.05))) } }],
      opts: {},
    },
    {
      name: "points with colours, thinned by maxPoints, + a volume (version 2)",
      grids: [
        { type: "points", name: "pts", positions: Float32Array.from({ length: 3 * 1000 }, (_, i) => Math.sin(i * 1.7) * 5), colors: Float32Array.from({ length: 3 * 1000 }, (_, i) => (i % 7) / 6) },
        { name: "density", background: 0, bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: [8, 8, 8], values: blob([8, 8, 8], 3) } },
      ],
      opts: { maxPoints: 400 },
    },
    {
      name: "all-background grid with no box (sparse LODs with no pages)",
      grids: [{ name: "empty", background: 0, bbox: null, finest: { dimensions: [128, 128, 128], values: new Float32Array(128 ** 3) } }],
      opts: {},
    },
  ];

  it.each(cases)("$name: decoded values re-encode to identical decoded values", ({ grids, opts }) => {
    const a = writeWvdb(grids, opts);
    const fa = readWvdb(a.bytes, { strictChunkOrder: true });
    expect(fa.header).toEqual(a.header);
    const b = reencode(fa, false);
    expectSameDecoded(fa, readWvdb(b.bytes));
    // deterministic (§12.1 SHOULD)
    expect(bytesEqual(writeWvdb(grids, opts).bytes, a.bytes)).toBe(true);
  });

  it("quantization error is at most half a code step; constant and background voxels are exact", () => {
    const dims: WvdbVec3 = [21, 37, 19];
    const values = blob(dims, 9);
    for (const bits of [8, 16] as const) {
      const file = readWvdb(writeWvdb([{ name: "d", background: 0, bbox: null, finest: { dimensions: dims, values } }], { bits, lods: 1 }).bytes);
      const level = (file.grids[0] as WvdbVolumeGrid).levels[0]!;
      const maxCode = bits === 8 ? 255 : 65535;
      for (let b = 0; b < level.brickIndices.length; b++) {
        const step = (level.maxs[b]! - level.mins[b]!) / maxCode;
        const [ox, oy, oz] = wvdbBrickOrigin(dims, level.brickIndices[b]!);
        for (let k = 0; k < 8; k++)
          for (let j = 0; j < 8; j++)
            for (let i = 0; i < 8; i++) {
              if (ox + i >= dims[0] || oy + j >= dims[1] || oz + k >= dims[2]) continue;
              const at = ox + i + (oy + j) * dims[0] + (oz + k) * dims[0] * dims[1];
              expect(Math.abs(level.values[at]! - values[at]!)).toBeLessThanOrEqual(step / 2 + 1e-7);
            }
      }
      // background voxels (unstored bricks, and brick minimums) decode exactly
      for (let i = 0; i < values.length; i++) if (values[i] === 0) expect(level.values[i]).toBe(0);
    }
  });

  it("sparse stores exactly the bricks that are not all background, most data first (§3.4, §9)", () => {
    const dims: WvdbVec3 = [40, 40, 40];
    const values = new Float32Array(40 ** 3).fill(2);
    const set = (x: number, y: number, z: number, v: number) => (values[x + y * 40 + z * 1600] = v);
    set(0, 0, 0, 2.5); // brick 0, weight 0.5 (+ no padding: brick fully inside)
    set(39, 39, 39, 5); // last brick (124): weight 3
    set(20, 3, 3, 2.000001); // brick 2: tiny weight
    const file = readWvdb(writeWvdb([{ name: "d", background: 2, bbox: null, finest: { dimensions: dims, values } }], { lods: 1 }).bytes);
    const level = (file.grids[0] as WvdbVolumeGrid).levels[0]!;
    expect(Array.from(level.brickIndices)).toEqual([124, 0, 2]);
    const dense = readWvdb(writeWvdb([{ name: "d", background: 2, bbox: null, finest: { dimensions: dims, values } }], { lods: 1, layout: "dense" }).bytes);
    const dl = (dense.grids[0] as WvdbVolumeGrid).levels[0]!;
    expect(dl.mask).toBeNull();
    expect(Array.from(dl.brickIndices)).toEqual(Array.from({ length: 125 }, (_, i) => i));
    expect(bytesEqual(dl.values, level.values)).toBe(true);
  });

  it("voxels outside the LOD copy the nearest edge voxel, so they do not widen a brick's range (§3.3)", () => {
    // 9 wide: brick 1 holds only x = 8; padding must replicate it, giving a constant brick.
    const dims: WvdbVec3 = [9, 8, 8];
    const values = field(dims, (x) => (x === 8 ? 7 : x / 10));
    const file = readWvdb(writeWvdb([{ name: "d", background: 0, bbox: null, finest: { dimensions: dims, values } }], { lods: 1 }).bytes);
    const level = (file.grids[0] as WvdbVolumeGrid).levels[0]!;
    const b = Array.from(level.brickIndices).indexOf(1);
    expect(level.mins[b]).toBe(7);
    expect(level.maxs[b]).toBe(7);
  });

  it("coarse LODs are the 2x mean over the voxels that exist (§3.2.4)", () => {
    const l = downsampleWvdbLevel({ dimensions: [3, 1, 1], values: new Float32Array([1, 3, 10]) }, 1);
    expect(l.dimensions).toEqual([2, 1, 1]);
    expect(Array.from(l.values)).toEqual([2, 10]);
  });

  it("header: version 1 without points, 2 with; bake members; points LOD split", () => {
    const vol: WvdbGridInput = { name: "d", background: 0, bbox: null, finest: { dimensions: [8, 8, 8], values: new Float32Array(512) } };
    const v1 = writeWvdb([vol]).header;
    expect(v1.version).toBe(1);
    expect(v1.bake).toEqual({ grids: null, resolution: 8, lods: 3, bits: 8, layout: "sparse", pageBricks: 1024, bbox: null });
    expect(v1.brickDim).toBe(8);
    const pts = Float32Array.from({ length: 30 }, (_, i) => i);
    const v2 = writeWvdb([vol, { type: "points", name: "p", positions: pts }]);
    expect(v2.header.version).toBe(2);
    expect(v2.header.bake.maxPoints).toBe(4_000_000);
    const file = readWvdb(v2.bytes);
    const p = file.grids[1] as WvdbPointsGrid;
    expect(p.levels.map((l) => l.count)).toEqual([1, 1, 8]); // 10 points, 3 LODs: {0}, {8}, {1..7, 9}
    expect(p.bbox).toEqual([[0, 1, 2], [27, 28, 29]]);
    expect(p.levels[0]!.positions[0]).toBe(0);
  });

  it("rejects bad input", () => {
    const vol = { name: "d", background: 0, bbox: null, finest: { dimensions: [8, 8, 8] as WvdbVec3, values: new Float32Array(511) } };
    expect(() => writeWvdb([vol])).toThrow(/X\*Y\*Z/);
    expect(() => writeWvdb([], { bits: 12 as 8 })).toThrow(/bits/);
    const ok = { ...vol, finest: { dimensions: [8, 8, 8] as WvdbVec3, values: new Float32Array(512) } };
    expect(() => writeWvdb([ok, ok])).toThrow(/unique/);
    const values = new Float32Array(512);
    values[3] = 1;
    expect(() => writeWvdb([{ ...vol, levels: [{ dimensions: [8, 8, 8], values, brickOrder: [] }] }], { lods: 1 })).toThrow(/omits brick 0/);
  });
});

describe("re-encoding his files with matching settings", () => {
  const committed = ["ground_explosion_0010.wvdb", "ground_explosion_0199.wvdb", "cloud_04_variant_0000.wvdb"].map(
    (n) => [n, fixture(n)] as const,
  );
  const extra = ["venusstatue.wvdb", "utahteapot.wvdb", "sphere_points.wvdb", "smoke.wvdb", "bunny_cloud.wvdb"]
    .map((n) => [n, hisAsset(n)] as const)
    .filter(([, p]) => p)
    .map(([n, p]) => [n, readBytes(p!)] as const);

  it.each([...committed, ...extra])(
    "%s: with his brick order, header and every inflated chunk are byte-identical",
    (_name, bytes) => {
      const his = readWvdb(bytes);
      const ours = reencode(his, true, (b) => b); // identity "compressor": compare inflated bytes directly
      expect(JSON.stringify(headerWithoutOffsets(ours.header))).toBe(JSON.stringify(headerWithoutOffsets(his.header)));
      const hisChunks = inflatedChunks(bytes, his);
      const dataStart = 12 + new DataView(ours.bytes.buffer).getUint32(8, true);
      const ourChunks = wvdbChunks(ours.header).map((c) => ours.bytes.subarray(dataStart + c.offset, dataStart + c.offset + c.length));
      expect(ourChunks.length).toBe(hisChunks.length);
      hisChunks.forEach((chunk, i) => expect(bytesEqual(ourChunks[i]!, chunk)).toBe(true));
    },
  );

  it.each([...committed, ...extra])("%s: with our default order and node zlib 9, decoded content is identical", (_name, bytes) => {
    const his = readWvdb(bytes);
    const ours = readWvdb(reencode(his, false, nodeZlib9).bytes, { strictChunkOrder: true });
    expectSameDecoded(his, ours);
  });
});
