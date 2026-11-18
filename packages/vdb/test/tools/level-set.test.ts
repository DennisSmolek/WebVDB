/**
 * K3.7: the CPU twins of the `'levelSet'` WGSL helper (`levelSetHit`,
 * `levelSetGradient`, `levelSetNormal`) and `vdbPointRenderData`.
 *
 * Oracles, independent of the twin's walk:
 * - `levelSetHitReference`: a fixed 1/8-voxel march over the whole support
 *   interval with no skipping and no sphere steps (f64), then 30 bisections;
 * - analytic narrow-band level sets built here (a sphere and a torus, values
 *   `clamp(sdf, -bg, bg)`, interior 8^3 cells as inactive `-bg` lower tiles,
 *   the outside as the `+bg` background, NanoVDB class LevelSet): hits on the
 *   sphere lie at radius R, normals are radial;
 * - the chain rule for the world normal under an anisotropic, rotated map:
 *   `normalize(W^T (q - c))` for the sphere, which a `W` (not transposed) gets
 *   wrong.
 * The GPU half runs in e2e/k3-levelset.spec.ts against these twins.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  buildFromLeavesDetailed,
  buildFromVdb,
  FLOAT_LEAF_CODEC,
  levelSetGradient,
  levelSetHit,
  levelSetHitReference,
  levelSetNormal,
  parseVdb,
  vdbPointRenderData,
  type VdbPointDataGrid,
  type VdbPointList,
} from "../../src/tools/index.js";

type V3 = [number, number, number];

/** A narrow-band level set of `sdf` (index units in, index units out; stored as world = index * voxelSize). */
function buildLevelSet(sdf: (p: V3) => number, lo: V3, hi: V3, voxelSize = 0.05, band = 3): Uint32Array {
  const bg = band * voxelSize;
  const leaves = [];
  const tiles = [];
  const a = lo.map((v) => Math.floor(v / 8) * 8) as V3;
  const b = hi.map((v) => Math.ceil(v / 8) * 8) as V3;
  for (let x = a[0]; x < b[0]; x += 8) {
    for (let y = a[1]; y < b[1]; y += 8) {
      for (let z = a[2]; z < b[2]; z += 8) {
        const values = new Float32Array(512);
        const valueMask = new Uint32Array(16);
        let active = 0;
        let inside = 0;
        for (let n = 0; n < 512; n++) {
          const d = sdf([x + (n >> 6), y + ((n >> 3) & 7), z + (n & 7)]) * voxelSize;
          values[n] = Math.min(bg, Math.max(-bg, d));
          if (Math.abs(d) < bg) {
            valueMask[n >> 5]! |= 1 << (n & 31);
            active++;
          } else if (d < 0) inside++;
        }
        if (active > 0) leaves.push({ origin: [x, y, z] as V3, values, valueMask });
        else if (inside === 512) tiles.push({ origin: [x, y, z] as V3, dim: 8, value: -bg, active: false });
      }
    }
  }
  return buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC, { voxelSize, background: bg, gridClass: "LevelSet", tiles }).image;
}

const C: V3 = [3.3, -2.6, 1.7];
const R = 21.5;
const sphereSdf = (p: V3) => Math.hypot(p[0] - C[0], p[1] - C[1], p[2] - C[2]) - R;
const SPHERE = buildLevelSet(sphereSdf, [-24, -30, -24], [30, 24, 30]);
const torusSdf = (p: V3) => Math.hypot(Math.hypot(p[0], p[2]) - 16, p[1]) - 5.5;
const TORUS = buildLevelSet(torusSdf, [-28, -12, -28], [28, 12, 28]);
const VOXEL = 0.05;

/** Deterministic rays from a sphere of radius `far` around `centre` toward points near it. */
function rays(count: number, centre: V3, far: number, spread: number, seed = 7): { o: V3; d: V3 }[] {
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  const out = [];
  for (let i = 0; i < count; i++) {
    const th = rnd() * Math.PI * 2;
    const ph = Math.acos(2 * rnd() - 1);
    const o: V3 = [centre[0] + far * Math.sin(ph) * Math.cos(th), centre[1] + far * Math.sin(ph) * Math.sin(th), centre[2] + far * Math.cos(ph)];
    const t = [0, 1, 2].map((k) => centre[k]! + (rnd() - 0.5) * spread);
    const d = t.map((v, k) => v - o[k]!);
    const l = Math.hypot(d[0]!, d[1]!, d[2]!);
    out.push({ o, d: d.map((v) => v / l) as V3 });
  }
  return out;
}

const angleDeg = (a: readonly number[], b: readonly number[]) => {
  const c = (a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!) / (Math.hypot(a[0]!, a[1]!, a[2]!) * Math.hypot(b[0]!, b[1]!, b[2]!));
  return (Math.acos(Math.min(1, Math.max(-1, c))) * 180) / Math.PI;
};

describe("levelSetHit (twin of vdbLevelSetHit)", () => {
  for (const [name, image, centre] of [
    ["sphere", SPHERE, C],
    ["torus", TORUS, [0, 0, 0] as V3],
  ] as const) {
    it(`${name}: finds the brute-force oracle's first crossing on 1,500 rays`, () => {
      let hits = 0;
      let maxDt = 0;
      let skipped = 0;
      for (const { o, d } of rays(1500, centre, 60, 50)) {
        const twin = levelSetHit(image, o, d, [0, 1e30], { sdfToIndex: 1 / VOXEL });
        const ref = levelSetHitReference(image, o, d, [0, 1e30]);
        expect(twin.hit, JSON.stringify({ o, d, twin, ref })).toBe(ref.hit);
        skipped += twin.skipped;
        if (!twin.hit) continue;
        hits++;
        maxDt = Math.max(maxDt, Math.abs(twin.t - ref.t));
      }
      expect(hits).toBeGreaterThan(400);
      expect(maxDt).toBeLessThan(1e-3);
      // The HDDA skipped tile and background cells (the reason for the walk).
      expect(skipped).toBeGreaterThan(200);
    });
  }

  it("sphere: hits lie on radius R, normals are radial (analytic oracle)", () => {
    let maxR = 0;
    let maxA = 0;
    for (const { o, d } of rays(500, C, 60, 30, 11)) {
      const h = levelSetHit(SPHERE, o, d, [0, 1e30], { sdfToIndex: 1 / VOXEL });
      if (!h.hit) continue;
      const p: V3 = [o[0] + d[0] * h.t, o[1] + d[1] * h.t, o[2] + d[2] * h.t];
      maxR = Math.max(maxR, Math.abs(Math.hypot(p[0] - C[0], p[1] - C[1], p[2] - C[2]) - R));
      const n = levelSetNormal(SPHERE, p, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
      maxA = Math.max(maxA, angleDeg(n, [p[0] - C[0], p[1] - C[1], p[2] - C[2]]));
    }
    expect(maxR).toBeLessThan(0.02);
    expect(maxA).toBeLessThan(0.5);
  });

  it("starts inside: the first sample is the hit; the sample cap misses; a ray past the box misses", () => {
    const inside = levelSetHit(SPHERE, C, [1, 0, 0], [0, 1e30], { sdfToIndex: 1 / VOXEL });
    expect(inside).toMatchObject({ hit: true, t: 0, samples: 1 });
    expect(levelSetHit(SPHERE, [3.3, -2.6, -80], [0, 0, 1], [0, 1e30], { sdfToIndex: 1 / VOXEL, maxSamples: 2 }).hit).toBe(false);
    expect(levelSetHit(SPHERE, [3.3, -2.6, -80], [0, 0, 1], [0, 1e30], { sdfToIndex: 1 / VOXEL }).hit).toBe(true);
    expect(levelSetHit(SPHERE, [100, 100, 100], [1, 0, 0], [0, 1e30])).toMatchObject({ hit: false, samples: 0 });
    // A t interval that ends before the surface misses.
    expect(levelSetHit(SPHERE, [3.3, -2.6, -80], [0, 0, 1], [0, 40], { sdfToIndex: 1 / VOXEL }).hit).toBe(false);
  });

  it("deep inside (all-inside tiles around the start): hits at the first sample, never skips inside tiles", () => {
    const big = buildLevelSet((p) => Math.hypot(p[0], p[1], p[2]) - 45, [-50, -50, -50], [50, 50, 50]);
    for (const d of [[1, 0, 0], [0.6, 0.8, 0], [-0.48, 0.6, 0.64]] as V3[]) {
      expect(levelSetHit(big, [1.5, -2.5, 0.5], d, [0, 1e30], { sdfToIndex: 1 / VOXEL })).toMatchObject({ hit: true, t: 0, samples: 1, skipped: 0 });
    }
  });

  it("the step scale only changes the cost: sdfToIndex 1 (a minimum-step march) finds the same crossings", () => {
    for (const { o, d } of rays(200, C, 60, 30, 3)) {
      const fast = levelSetHit(SPHERE, o, d, [0, 1e30], { sdfToIndex: 1 / VOXEL });
      const slow = levelSetHit(SPHERE, o, d, [0, 1e30], { sdfToIndex: 1, maxSamples: 4096 });
      expect(slow.hit).toBe(fast.hit);
      if (fast.hit) {
        expect(Math.abs(slow.t - fast.t)).toBeLessThan(1e-3);
        expect(slow.samples).toBeGreaterThanOrEqual(fast.samples);
      }
    }
  });
});

describe("levelSetNormal (twin of vdbLevelSetNormal)", () => {
  it("is W^T times the index gradient: the sphere seen through a rotated, anisotropic map", () => {
    // world -> index: W = rotation * diag(1, 2, 0.5), plus a translation.
    const c = Math.cos(0.7);
    const s = Math.sin(0.7);
    const rot = [c, -s, 0, s, c, 0, 0, 0, 1];
    const diag = [1, 2, 0.5];
    const W = [0, 1, 2].map((r) => [0, 1, 2].map((k) => rot[r * 3 + k]! * diag[k]!));
    const worldToIndex = [...W[0]!, 4, ...W[1]!, -3, ...W[2]!, 2];
    let maxA = 0;
    let checked = 0;
    for (const { o, d } of rays(300, C, 60, 30, 5)) {
      const h = levelSetHit(SPHERE, o, d, [0, 1e30], { sdfToIndex: 1 / VOXEL });
      if (!h.hit) continue;
      const q: V3 = [o[0] + d[0] * h.t, o[1] + d[1] * h.t, o[2] + d[2] * h.t];
      const n = levelSetNormal(SPHERE, q, worldToIndex);
      const g = [q[0] - C[0], q[1] - C[1], q[2] - C[2]];
      const expected = [0, 1, 2].map((k) => W[0]![k]! * g[0]! + W[1]![k]! * g[1]! + W[2]![k]! * g[2]!);
      maxA = Math.max(maxA, angleDeg(n, expected));
      checked++;
    }
    expect(checked).toBeGreaterThan(100);
    expect(maxA).toBeLessThan(0.5);
  });

  it("the gradient is in SDF units per voxel and the normal falls back to +y for a zero gradient", () => {
    const g = levelSetGradient(SPHERE, [C[0] + R, C[1], C[2]]);
    expect(g[0]).toBeCloseTo(VOXEL, 3);
    expect(Math.abs(g[1]) + Math.abs(g[2])).toBeLessThan(1e-3);
    // Far outside: every tap is the background, the gradient is 0.
    expect(levelSetNormal(SPHERE, [200, 200, 200], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])).toEqual([0, 1, 0]);
  });
});

const TEAPOT = new URL("../../../../fixtures/vdb-samples/utahteapot.vdb", import.meta.url);
const SPHERE_POINTS = new URL("../../../../fixtures/vdb-samples/sphere_points.vdb", import.meta.url);
const read = (u: URL) => {
  const b = readFileSync(u);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

describe.skipIf(!existsSync(TEAPOT))("utahteapot.vdb (asset)", () => {
  it("the twin matches the brute-force oracle on 300 rays", () => {
    const grid = parseVdb(read(TEAPOT)).grids.find((g) => g.valueType === "float")!;
    const image = buildFromVdb(grid);
    let hits = 0;
    for (const { o, d } of rays(300, [-10, 0, 0], 900, 600)) {
      const twin = levelSetHit(image, o, d, [0, 1e30], { sdfToIndex: 1 / grid.transform.voxelSize[0] });
      const ref = levelSetHitReference(image, o, d, [0, 1e30]);
      expect(twin.hit).toBe(ref.hit);
      if (twin.hit) {
        hits++;
        expect(Math.abs(twin.t - ref.t)).toBeLessThan(2e-3);
      }
    }
    expect(hits).toBeGreaterThan(100);
  }, 60_000);
});

describe("vdbPointRenderData (his createPointData)", () => {
  /** A stand-in point grid: n points, Cd = (i, 2i, 3i) / n, 4 points per active voxel, voxel size 0.5. */
  function fakeGrid(n: number, withColor = true): VdbPointDataGrid {
    const positions = Float32Array.from({ length: n * 3 }, (_, k) => k);
    const list: VdbPointList = {
      count: n,
      positions,
      attributes: withColor ? { Cd: Float32Array.from({ length: n * 3 }, (_, k) => (Math.floor(k / 3) * ((k % 3) + 1)) / n) } : {},
      attributeShapes: withColor ? { Cd: { components: 3, stride: 1 } } : {},
      groups: {},
      skipped: [],
    };
    return {
      valueType: "ptdataidx32",
      pointCount: n,
      activeVoxelCount: BigInt(n / 4),
      transform: { type: "UniformScaleMap", matrix: [], voxelSize: [0.5, 0.5, 0.5] },
      points: () => list,
    } as unknown as VdbPointDataGrid;
  }

  it("keeps every point below maxPoints; spacing = voxelSize / cbrt(points per voxel)", () => {
    const d = vdbPointRenderData(fakeGrid(1000));
    expect(d).toMatchObject({ count: 1000, totalCount: 1000, stride: 1 });
    expect(d.positions).toEqual(fakeGrid(1000).points().positions);
    expect(d.colors![3 * 7 + 1]).toBeCloseTo((7 * 2) / 1000, 6);
    expect(d.spacing).toBeCloseTo(0.5 / Math.cbrt(4), 12);
  });

  it("above maxPoints keeps every stride-th point and widens the spacing by cbrt(stride)", () => {
    const d = vdbPointRenderData(fakeGrid(1000), { maxPoints: 300 });
    expect(d.stride).toBe(4);
    expect(d.count).toBe(250);
    expect([...d.positions.subarray(3, 6)]).toEqual([12, 13, 14]); // point 4
    expect(d.spacing).toBeCloseTo((0.5 / Math.cbrt(4)) * Math.cbrt(4), 12);
    expect(vdbPointRenderData(fakeGrid(1000, false)).colors).toBeNull();
    expect(() => vdbPointRenderData({ valueType: "float" } as never)).toThrow(/point data grid/);
  });

  it.skipIf(!existsSync(SPHERE_POINTS))("sphere_points.vdb: 9,802 points, the parser's positions", () => {
    const grid = parseVdb(read(SPHERE_POINTS)).grids.find((g) => g.valueType === "ptdataidx32") as VdbPointDataGrid;
    const d = vdbPointRenderData(grid);
    expect(d.count).toBe(9802);
    expect(d.positions).toEqual(grid.points().positions);
    expect(d.spacing).toBeGreaterThan(0);
  });
});
