/**
 * Half grids (K7.H, NanoVDB `GridType::Half` = 9, NanoVDB 13.2): the binary16
 * conversions, `vdbQuantize(…, 'half')`, the CPU readers (`vdbReadValueCpu`,
 * the `march` twins) and the `.nvdb` / pool round trips. The native cross-check
 * (NanoVDB v14.0.0 output) is `half-native.test.ts`.
 */
import { describe, expect, it } from "vitest";

import { VDBGridPool } from "../../src/core/index.js";
import { halfBitsToNumber } from "../../src/core/half.js";
import {
  NvdbFile,
  VDB_HALF_LEAF_CODEC,
  vdbBuildFromDense,
  vdbBuildFromLeavesDetailed,
  vdbFloatToHalf,
  vdbHalfToFloat,
  vdbInspect,
  vdbQuantize,
  vdbReadValueCpu,
  vdbReadValueF32,
  vdbRoundToHalf,
  vdbSampleTrilinear,
  writeNvdb,
} from "../../src/tools/index.js";
import { defaultValueMax, nodeMax, nodeStats, treeNodes, leafVoxels } from "../../src/tools/cpu/march.js";
import { rootStats } from "../../src/tools/cpu/compute.js";
import { gridTypeConstants, gridTypeIds } from "../../src/tools/cpu/stride-tables.js";
import { GRID_TYPE_HALF, HALF_LAYOUT } from "../../src/tools/nanovdb/bytes.js";
import { halfToFloat as vdbParserHalfToFloat } from "../../src/tools/vdb/half-float.js";
import { FakeDevice } from "../core/fake-gpu.js";

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const bitsOf = (v: number): number => {
  f32[0] = v;
  return u32[0]!;
};
const fromBits = (b: number): number => {
  u32[0] = b >>> 0;
  return f32[0]!;
};
/** Same value and the same sign of zero. */
const same = (a: number, b: number): boolean => Object.is(a, b);

/** Every finite half value in increasing order, with its bits (-0 dropped). */
const FINITE: { v: number; h: number }[] = (() => {
  const out: { v: number; h: number }[] = [];
  for (let h = 0; h < 0x10000; h++) {
    const v = vdbHalfToFloat(h);
    if (Number.isFinite(v) && h !== 0x8000) out.push({ v, h });
  }
  return out.sort((a, b) => a.v - b.v);
})();

/** Brute-force nearest half (ties to the even mantissa), +-Infinity past 65520. */
function nearestHalf(x: number): number {
  if (x >= 65520) return 0x7c00;
  if (x <= -65520) return 0xfc00;
  let lo = 0;
  let hi = FINITE.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (FINITE[mid]!.v <= x) lo = mid;
    else hi = mid;
  }
  const a = FINITE[lo]!;
  const b = FINITE[hi]!;
  const da = Math.abs(x - a.v);
  const db = Math.abs(x - b.v);
  const pick = da < db ? a : db < da ? b : (a.h & 1) === 0 ? a : b;
  // Zero keeps the input's sign.
  if (pick.v === 0) return x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  return pick.h;
}

describe("binary16 conversions (NanoVDB math/HalfLite.h)", () => {
  it("vdbHalfToFloat decodes all 65536 patterns exactly, as the core and the .vdb parser do", () => {
    for (let h = 0; h < 0x10000; h++) {
      const v = vdbHalfToFloat(h);
      const e = (h >>> 10) & 31;
      const m = h & 1023;
      const s = h & 0x8000 ? -1 : 1;
      const want = e === 31 ? (m ? NaN : s * Infinity) : e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
      expect(same(v, want), `0x${h.toString(16)}`).toBe(true);
      expect(Math.fround(v) === v || Number.isNaN(v), "every half is an f32").toBe(true);
      expect(same(halfBitsToNumber(h), v), `core 0x${h.toString(16)}`).toBe(true);
      expect(same(vdbParserHalfToFloat(h), v), `vdb parser 0x${h.toString(16)}`).toBe(true);
    }
  });

  it("vdbFloatToHalf round-trips every half, incl. -0, subnormals and infinities", () => {
    for (let h = 0; h < 0x10000; h++) {
      const v = vdbHalfToFloat(h);
      if (Number.isNaN(v)) continue;
      expect(vdbFloatToHalf(v), `0x${h.toString(16)}`).toBe(h);
    }
  });

  it("rounds to nearest, ties to even, and overflows to infinity past 65519.99…", () => {
    // Midpoints between consecutive finite halves go to the even neighbour.
    for (let k = 0; k + 1 < FINITE.length; k++) {
      const a = FINITE[k]!;
      const b = FINITE[k + 1]!;
      const mid = (a.v + b.v) / 2;
      if (Math.fround(mid) !== mid) continue; // not representable in f32 (never happens for halves)
      const want = (a.h & 1) === 0 ? a.h : b.h;
      const got = vdbFloatToHalf(mid);
      expect(vdbHalfToFloat(got) === vdbHalfToFloat(want), `mid ${mid}`).toBe(true);
    }
    expect(vdbFloatToHalf(65504)).toBe(0x7bff);
    expect(vdbFloatToHalf(65519.99)).toBe(0x7bff);
    expect(vdbFloatToHalf(65520)).toBe(0x7c00);
    expect(vdbFloatToHalf(-1e9)).toBe(0xfc00);
    expect(vdbFloatToHalf(2 ** -25)).toBe(0); // tie between 0 and the smallest subnormal: even (0)
    expect(vdbFloatToHalf(fromBits(bitsOf(2 ** -25) + 1))).toBe(1);
    expect(vdbFloatToHalf(3 * 2 ** -25)).toBe(2); // tie between 1 and 2: even
    expect(vdbFloatToHalf(-0)).toBe(0x8000);
    expect(vdbFloatToHalf(1e-30)).toBe(0);
    expect(vdbFloatToHalf(-1e-30)).toBe(0x8000);
  });

  it("matches a brute-force nearest-even search on 200k random f32", () => {
    let seed = 12345;
    const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
    for (let k = 0; k < 200_000; k++) {
      // Spread over every exponent the conversion branches on (2^-30 .. 2^17).
      const x = Math.fround((rnd() < 0.5 ? -1 : 1) * 2 ** (rnd() * 47 - 30));
      const got = vdbFloatToHalf(x);
      const want = nearestHalf(x);
      if (got !== want) expect(`0x${got.toString(16)}`, `x=${x}`).toBe(`0x${want.toString(16)}`);
    }
  });

  it("keeps NaN a NaN with its sign", () => {
    expect(Number.isNaN(vdbHalfToFloat(vdbFloatToHalf(NaN)))).toBe(true);
    const neg = fromBits(0xffc00000);
    expect(vdbFloatToHalf(neg) & 0xfc00).toBe(0xfc00);
    // (JS numbers don't keep NaN payloads, so HalfLite's "payload below the kept bits -> 1" can't be driven from here.)
    expect(vdbRoundToHalf(0.1)).toBe(vdbHalfToFloat(0x2e66));
  });
});

describe("Half layout (PNanoVDB.h HALF row)", () => {
  it("HALF_LAYOUT is the vendored stride-table row 9", () => {
    const c = gridTypeConstants.HALF!;
    expect(gridTypeIds.HALF).toBe(GRID_TYPE_HALF);
    const L = HALF_LAYOUT;
    expect([L.rootOffBackground, L.rootOffMin, L.rootOffMax, L.rootOffAve, L.rootOffStdDev, L.rootSize]).toEqual([
      c.root_off_background, c.root_off_min, c.root_off_max, c.root_off_ave, c.root_off_stddev, c.root_size,
    ]);
    expect([L.valueStrideBits, L.tableStride, L.rootTileOffValue, L.rootTileSize]).toEqual([
      c.value_stride_bits, c.table_stride, c.root_tile_off_value, c.root_tile_size,
    ]);
    expect([L.upperOffMin, L.upperOffMax, L.upperOffAve, L.upperOffStdDev, L.upperOffTable, L.upperSize]).toEqual([
      c.upper_off_min, c.upper_off_max, c.upper_off_ave, c.upper_off_stddev, c.upper_off_table, c.upper_size,
    ]);
    expect([L.lowerOffMin, L.lowerOffMax, L.lowerOffAve, L.lowerOffStdDev, L.lowerOffTable, L.lowerSize]).toEqual([
      c.lower_off_min, c.lower_off_max, c.lower_off_ave, c.lower_off_stddev, c.lower_off_table, c.lower_size,
    ]);
    expect([L.leafOffMin, L.leafOffMax, L.leafOffAve, L.leafOffStdDev, L.leafOffTable, L.leafSize]).toEqual([
      c.leaf_off_min, c.leaf_off_max, c.leaf_off_ave, c.leaf_off_stddev, c.leaf_off_table, c.leaf_size,
    ]);
  });
});

/** A fog ramp with values outside [0, 1] and a few that only a half can't hold exactly. */
function field(dim: number): { values: Float32Array; dims: [number, number, number] } {
  const values = new Float32Array(dim * dim * dim);
  const c = (dim - 1) / 2;
  const r = dim * 0.45;
  for (let x = 0; x < dim; x++)
    for (let y = 0; y < dim; y++)
      for (let z = 0; z < dim; z++) {
        const d = Math.hypot(x - c, y - c, z - c);
        // 3.7 * ramp + tiny offsets: most values are not halves; x < 4 gives a uniform block (tiles after collapse).
        values[(x * dim + y) * dim + z] = x < 8 && y < 8 && z < 8 ? 2.5 : d < r ? 3.7 * (1 - d / r) + 1e-4 * (x - y) : 0;
      }
  return { values, dims: [dim, dim, dim] };
}

describe("vdbQuantize(…, 'half')", () => {
  const { values, dims } = field(28);
  const floatImg = vdbBuildFromDense(values, dims, { background: 0, gridName: "ramp" });
  const half = vdbQuantize(floatImg, "half");

  it("writes a NanoVDB Half grid with the source topology", () => {
    const report = vdbInspect(half);
    expect(report.gridType).toBe("Half");
    expect(new DataView(half.buffer).getUint32(636, true)).toBe(9);
    expect(report.voxelCount).toBe(vdbInspect(floatImg).voxelCount);
    expect(report.nodeCounts).toEqual(vdbInspect(floatImg).nodeCounts);
    expect(report.memoryBreakdown.leaf).toBe(report.nodeCounts.leaf * 1120);
    expect(report.memoryBreakdown.total).toBe(half.byteLength);
  });

  it("every voxel reads back as the nearest half of the source (both CPU readers, every level)", () => {
    let tiles = 0;
    for (let x = -2; x < dims[0] + 2; x++)
      for (let y = -2; y < dims[1] + 2; y++)
        for (let z = -2; z < dims[2] + 2; z++) {
          const src = vdbReadValueCpu(floatImg, [x, y, z]);
          const got = vdbReadValueCpu(half, [x, y, z]);
          expect(got.active).toBe(src.active);
          const want = vdbRoundToHalf(src.value);
          if (got.value !== want) expect(got.value, `${x},${y},${z}`).toBe(want);
          const g32 = vdbReadValueF32(half, [x, y, z]);
          if (g32 !== want) expect(g32, `f32 twin ${x},${y},${z}`).toBe(want);
          if (src.active && src.value === 2.5) tiles++;
        }
    expect(tiles).toBe(512);
    // The uniform block is a lower-node tile, not a leaf.
    expect(treeNodes(half).some((n) => n.activeTiles.some((t) => t.dim === 8 && t.value === 2.5))).toBe(true);
  });

  it("stats: binary16 min/max over the stored halves, f32 ave/stddev, at every level", () => {
    const nodes = treeNodes(half);
    let leaves = 0;
    for (const n of nodes) {
      if (n.level !== 0) continue;
      const { values: v, active } = leafVoxels(half, n.address);
      let mn = Infinity;
      let mx = -Infinity;
      let sum = 0;
      let cnt = 0;
      for (let i = 0; i < 512; i++) {
        if (!active[i]) continue;
        mn = Math.min(mn, v[i]!);
        mx = Math.max(mx, v[i]!);
        sum += v[i]!;
        cnt++;
      }
      if (cnt === 0) continue;
      const s = nodeStats(half, 0, n.address);
      expect(s.min).toBe(mn);
      expect(s.max).toBe(mx);
      expect(s.ave).toBeCloseTo(sum / cnt, 5);
      leaves++;
    }
    expect(leaves).toBeGreaterThan(10);
    const root = nodeStats(half, 3, nodes[0]!.address);
    let srcMax = -Infinity;
    let srcMin = Infinity;
    for (let k = 0; k < values.length; k++) {
      if (values[k] === 0) continue; // inactive (the background)
      srcMax = Math.max(srcMax, vdbRoundToHalf(values[k]!));
      srcMin = Math.min(srcMin, vdbRoundToHalf(values[k]!));
    }
    expect(root.max).toBe(srcMax);
    expect(root.min).toBe(srcMin);
    expect(vdbFloatToHalf(root.max)).toBeLessThan(0x7c00);
    expect(rootStats(half)).toEqual({ min: root.min, max: root.max });
    expect(defaultValueMax(half)).toBe(Math.max(root.max, 0));
    // vdbNodeMax at a leaf voxel = the leaf's max (a half), at a tile = the tile's value.
    expect(nodeMax(half, [2, 2, 2], 8)).toBe(2.5);
  });

  it("samples trilinearly between halves", () => {
    const p: [number, number, number] = [13.25, 12.5, 14.75];
    const s = vdbSampleTrilinear(half, p);
    const ref = vdbSampleTrilinear(floatImg, p);
    expect(Math.abs(s - ref)).toBeLessThan(4 * 2 ** -11);
  });

  it("round-trips through writeNvdb / NvdbFile as a Half grid", () => {
    const file = NvdbFile.fromArrayBuffer(writeNvdb([half]));
    expect(file.grids[0]!.gridType).toBe("Half");
    expect(Array.from(file.gridImage(0))).toEqual(Array.from(half));
  });

  it("builds Half grids from leaves directly (codec rounds values and stats)", () => {
    const leaf = new Float32Array(512).fill(0.1);
    const mask = new Uint32Array(16).fill(0xffffffff);
    const built = vdbBuildFromLeavesDetailed([{ origin: [0, 0, 0], values: leaf, valueMask: mask }], VDB_HALF_LEAF_CODEC, {
      background: 1 / 3,
      tiles: [{ origin: [8, 0, 0], dim: 8, value: 0.7 }],
    });
    expect(vdbReadValueCpu(built.image, [1, 2, 3]).value).toBe(vdbRoundToHalf(0.1));
    expect(vdbReadValueCpu(built.image, [9, 0, 0]).value).toBe(vdbRoundToHalf(0.7));
    expect(vdbReadValueCpu(built.image, [-100, 0, 0]).value).toBe(vdbRoundToHalf(1 / 3));
    const leafNode = treeNodes(built.image).find((n) => n.level === 0)!;
    expect(nodeStats(built.image, 0, leafNode.address).min).toBe(vdbRoundToHalf(0.1));
    expect(nodeStats(built.image, 0, leafNode.address).stddev).toBe(0);
  });

  it("a VDBGridPool holds it: type 'half', binary16 background and root stats decoded", async () => {
    const device = new FakeDevice();
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 1 << 16 });
    const id = pool.add(half);
    const info = pool.info(id);
    expect(info.type).toBe("half");
    expect([...pool.types]).toEqual(["half"]);
    expect(info.rootMax).toBe(nodeStats(half, 3, treeNodes(half)[0]!.address).max);
    expect(info.background).toBe(0);
  });
});
