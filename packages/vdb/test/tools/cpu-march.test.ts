/**
 * K1.4b CPU twins of the `march` profile (`src/tools/cpu/march.ts`) and the
 * `outside` sampler twins.
 *
 * What is checked, and why it is the right claim (design note §5):
 * 1. Fp leaf stats: the fork's fixed readers (twin: `nodeStats`) decode the
 *    uint16 codes as NanoVDB `LeafFnBase` does. On the native Fp fixtures the
 *    stats are of the decoded active values (`GridStats.h` iterates
 *    `cbeginValueOn`), so min/max equal `StatsAccumulator` over the decoded
 *    active voxels and ave/stddev are within half a quantum (code rounding).
 * 2. Node stats bound ACTIVE values: every active voxel and active tile in a
 *    node's subtree is <= its max. Nothing more.
 * 3. The trilinear caveat: a sample's taps reach one voxel past its cell, so
 *    the cell's own bound is not a bound for the sample. The dilated bound
 *    (max over the 2x2x2 support of each tap's own bound) is, provided every
 *    inactive leaf voxel is <= its leaf max, which is checked per fixture.
 * 4. The walk: cells tile the clipped ray; every active voxel the ray crosses
 *    (value >= threshold) lies in a non-skipped cell whose bound holds it.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { vdbBuildFromDense, vdbQuantize } from "../../src/tools/index.js";
import {
  cellBoundTrilinear,
  defaultValueMax,
  getDim,
  gridHasStats,
  hddaCellEnd,
  hddaInit,
  hddaStep,
  hddaWalk,
  indexBBox,
  isActive,
  leafVoxels,
  nodeMax,
  nodeStats,
  rayClip,
  rayClipSupport,
  vdbReadValueF32,
  treeNodes,
  type HddaSegment,
  type Vec3,
} from "../../src/tools/cpu/march.js";
import { vdbReadValueCpu } from "../../src/tools/cpu/read-value.js";
import { vdbSampleNearest, vdbSampleTrilinear } from "../../src/tools/cpu/sample-trilinear.js";
import { StatsAccumulator } from "../../src/tools/nanovdb/stats.js";
import { NvdbFile } from "../../src/tools/nvdb-file.js";

const PRIMITIVES = ["box", "sphere", "torus"].flatMap((shape) => ["float", "fp8", "fpn"].map((type) => `${shape}_fog_${type}`));
const fixtureUrl = (name: string) => new URL(`../../../../fixtures/primitives/${name}.nvdb`, import.meta.url);
const present = PRIMITIVES.every((n) => existsSync(fixtureUrl(n)));
const images = new Map<string, Uint32Array>();
function load(name: string): Uint32Array {
  let img = images.get(name);
  if (!img) {
    const buf = readFileSync(fileURLToPath(fixtureUrl(name)));
    img = NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).gridImage(0);
    images.set(name, img);
  }
  return img;
}

/** Deterministic PRNG (mulberry32). */
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

const f32 = (img: Uint32Array) => new Float32Array(img.buffer, img.byteOffset, img.length);
/** Leaf mQuantum (byte 84 of an Fp leaf). */
const leafQuantum = (img: Uint32Array, leaf: number) => f32(img)[(leaf + 84) >>> 2]!;

function leafAccumulator(img: Uint32Array, leaf: number): StatsAccumulator {
  const { values, active } = leafVoxels(img, leaf);
  const acc = new StatsAccumulator();
  for (let n = 0; n < 512; n++) if (active[n]) acc.addVoxel(values[n]!, n >> 6, (n >> 3) & 7, n & 7);
  return acc;
}

/** Tolerance for comparing an f32 stat against an f32 value (two roundings of code * quantum + minimum). */
const ulps = (v: number) => 4 * 2 ** -23 * Math.max(1, Math.abs(v));

// ---------------------------------------------------------------------------
// 1. Fp leaf stats

describe("Fp leaf stats: fixed readers' twin vs tools stats", () => {
  it.skipIf(!present).each(PRIMITIVES.filter((n) => !n.endsWith("float")))("%s: every leaf", (name) => {
    const img = load(name);
    expect(gridHasStats(img)).toBe(true);
    let leaves = 0;
    let coded = 0;
    let misreads = 0;
    for (const node of treeNodes(img)) {
      if (node.level !== 0) continue;
      const acc = leafAccumulator(img, node.address);
      if (acc.isEmpty) continue;
      leaves++;
      const s = nodeStats(img, 0, node.address);
      const q = leafQuantum(img, node.address);
      expect(Math.abs(s.min - acc.min), `${name} leaf ${node.origin} min`).toBeLessThanOrEqual(ulps(acc.min));
      expect(Math.abs(s.max - acc.max), `${name} leaf ${node.origin} max`).toBeLessThanOrEqual(ulps(acc.max));
      expect(Math.abs(s.ave - acc.average)).toBeLessThanOrEqual(q / 2 + ulps(acc.average));
      expect(Math.abs(s.stddev - acc.stdDev)).toBeLessThanOrEqual(q / 2 + ulps(acc.stdDev));
      // The pre-fix reader: f32 at leaf + 90 reads the word at 88 (min | max << 16).
      // It is right by accident only when both codes are 0 (an all-zero leaf).
      if (img[(node.address + 88) >>> 2] === 0) continue;
      coded++;
      const old = f32(img)[(node.address + 90) >>> 2]!;
      if (!(Math.abs(old - acc.max) <= ulps(acc.max))) misreads++;
    }
    expect(leaves).toBeGreaterThan(100);
    expect(coded).toBeGreaterThan(100);
    expect(misreads).toBe(coded); // the old f32 read was wrong for every leaf with a non-zero code
  });

  it("decodes the codes our own quantizer writes (code * quantum + minimum; stddev code * quantum)", () => {
    const n = 24;
    const values = new Float32Array(n * n * n);
    const r = rng(7);
    for (let i = 0; i < values.length; i++) values[i] = r() < 0.6 ? r() * 2 : 0;
    const float = vdbBuildFromDense(values, [n, n, n], { origin: [-5, 3, 9] });
    const floatLeaves = new Map(treeNodes(float).filter((t) => t.level === 0).map((t) => [t.origin.join(), t.address]));
    for (const mode of ["fp8", "fpn"] as const) {
      const img = vdbQuantize(float, mode);
      let checked = 0;
      for (const node of treeNodes(img)) {
        if (node.level !== 0) continue;
        // Our quantizer encodes stats of the ORIGINAL float values (round to nearest code).
        const acc = leafAccumulator(float, floatLeaves.get(node.origin.join())!);
        if (acc.isEmpty) continue;
        const s = nodeStats(img, 0, node.address);
        const q = leafQuantum(img, node.address);
        const words = new Uint16Array(img.buffer, img.byteOffset + node.address + 88, 4);
        const minimum = f32(img)[(node.address + 80) >>> 2]!;
        expect(s.min).toBe(Math.fround(Math.fround(words[0]! * q) + minimum));
        expect(s.max).toBe(Math.fround(Math.fround(words[1]! * q) + minimum));
        expect(s.ave).toBe(Math.fround(Math.fround(words[2]! * q) + minimum));
        expect(s.stddev).toBe(Math.fround(words[3]! * q));
        for (const [got, want] of [[s.min, acc.min], [s.max, acc.max], [s.ave, acc.average], [s.stddev, acc.stdDev]] as const) {
          expect(Math.abs(got - want)).toBeLessThanOrEqual(q / 2 + 1e-6);
        }
        checked++;
      }
      expect(checked).toBeGreaterThan(10);
    }
  });
});

// ---------------------------------------------------------------------------
// 2. Node stats bound active values; nodeMax / getDim

/** For every node, the max over the ACTIVE voxels and tiles of its subtree (and min). */
function subtreeExtrema(img: Uint32Array) {
  const nodes = treeNodes(img);
  const max = new Float64Array(nodes.length).fill(-Infinity);
  const min = new Float64Array(nodes.length).fill(Infinity);
  let inactiveAboveLeafMax = 0;
  for (let i = nodes.length - 1; i >= 0; i--) {
    const node = nodes[i]!;
    if (node.level === 0) {
      const { values, active } = leafVoxels(img, node.address);
      const leafMax = nodeStats(img, 0, node.address).max;
      for (let n = 0; n < 512; n++) {
        if (active[n]) {
          max[i] = Math.max(max[i]!, values[n]!);
          min[i] = Math.min(min[i]!, values[n]!);
        } else if (values[n]! > leafMax) inactiveAboveLeafMax++;
      }
    }
    for (const t of node.activeTiles) {
      max[i] = Math.max(max[i]!, t.value);
      min[i] = Math.min(min[i]!, t.value);
    }
    if (node.parent >= 0) {
      max[node.parent] = Math.max(max[node.parent]!, max[i]!);
      min[node.parent] = Math.min(min[node.parent]!, min[i]!);
    }
  }
  return { nodes, max, min, inactiveAboveLeafMax };
}

describe("node stats bound the active values of their subtree", () => {
  it.skipIf(!present).each(PRIMITIVES)("%s", (name) => {
    const img = load(name);
    const { nodes, max, min } = subtreeExtrema(img);
    const counts = [0, 0, 0, 0];
    nodes.forEach((node, i) => {
      if (max[i] === -Infinity) return; // no active values: stats say nothing
      const s = nodeStats(img, node.level, node.address);
      expect(s.max, `${name} level ${node.level} at ${node.origin}`).toBeGreaterThanOrEqual(max[i]! - ulps(max[i]!));
      expect(s.min).toBeLessThanOrEqual(min[i]! + ulps(min[i]!));
      counts[node.level]!++;
    });
    expect(counts[0]).toBeGreaterThan(0);
    expect(counts[3]).toBe(1);
  });
});

describe("getDim and nodeMax", () => {
  it.skipIf(!present).each(PRIMITIVES)("%s: dims, tile values and coarse bounds at random voxels", (name) => {
    const img = load(name);
    const { min, max } = indexBBox(img);
    const r = rng(11);
    const rootMax = nodeStats(img, 3, treeNodes(img)[0]!.address).max;
    for (let k = 0; k < 2000; k++) {
      const ijk: Vec3 = [0, 1, 2].map((a) => Math.floor(min[a]! - 16 + r() * (max[a]! - min[a]! + 33))) as Vec3;
      const a = getDim(img, ijk);
      expect([1, 8, 128, 4096]).toContain(a);
      const v = vdbReadValueF32(img, ijk);
      expect(Math.abs(v - vdbReadValueCpu(img, ijk).value)).toBeLessThanOrEqual(1e-6);
      if (a > 1) {
        for (const dim of [1, 8, 128, 4096].filter((d) => d <= a)) expect(nodeMax(img, ijk, dim)).toBe(v); // uniform cell: its value
      }
      if (isActive(img, ijk)) {
        for (const dim of [1, 8, 64, 128, 4096, 1 << 20]) expect(nodeMax(img, ijk, dim)).toBeGreaterThanOrEqual(v - ulps(v));
        expect(nodeMax(img, ijk, 1 << 20)).toBeGreaterThanOrEqual(a > 1 && 1 << 20 <= a ? v : Math.min(rootMax, v));
      }
    }
  });

  it("without HAS_MIN_MAX the stats branches return valueMax (+Infinity by default), tiles keep their value", () => {
    const n = 20;
    const values = new Float32Array(n * n * n).map((_, i) => (i % 3 === 0 ? 0.5 + (i % 7) / 10 : 0));
    const withStats = vdbBuildFromDense(values, [n, n, n]);
    expect(gridHasStats(withStats)).toBe(true);
    const img = withStats.slice(); // the twins cache per array: edit a copy before first use
    img[20 >>> 2]! &= ~4; // clear HAS_MIN_MAX in GridData.mFlags (byte 20)
    expect(gridHasStats(img)).toBe(false);
    expect(defaultValueMax(img)).toBe(Infinity);
    expect(nodeMax(img, [1, 1, 1], 8)).toBe(Infinity);
    expect(nodeMax(img, [1, 1, 1], 8, { valueMax: 2 })).toBe(2);
    expect(nodeMax(img, [5000, 0, 0], 8)).toBe(0); // background: uniform, its value
    const segs = hddaWalk(img, [-10, 3.3, 4.7], [1, 0.01, 0.02]);
    expect(segs.some((s) => !s.skip && s.max === Infinity)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 3. The trilinear caveat

describe("trilinear samples need the dilated bound", () => {
  it.skipIf(!present).each(PRIMITIVES)("%s", (name) => {
    const img = load(name);
    const { inactiveAboveLeafMax } = subtreeExtrema(img);
    // Precondition of the dilated bound (true for these native fog grids).
    expect(inactiveAboveLeafMax).toBe(0);
    const { min, max } = indexBBox(img);
    const r = rng(23);
    let undilatedMisses = 0;
    for (let k = 0; k < 3000; k++) {
      const p: Vec3 = [0, 1, 2].map((a) => min[a]! - 2 + r() * (max[a]! - min[a]! + 4)) as Vec3;
      const s = vdbSampleTrilinear(img, p);
      const b: Vec3 = [Math.floor(p[0]), Math.floor(p[1]), Math.floor(p[2])];
      let dilated = -Infinity;
      for (let t = 0; t < 8; t++) {
        const tap: Vec3 = [b[0] + (t & 1), b[1] + ((t >> 1) & 1), b[2] + ((t >> 2) & 1)];
        dilated = Math.max(dilated, nodeMax(img, tap, getDim(img, tap)));
      }
      expect(s, `${name} at ${p}`).toBeLessThanOrEqual(dilated + 1e-5);
      if (s > nodeMax(img, b, Math.max(getDim(img, b), 8)) + 1e-5) undilatedMisses++;
    }
    // The caveat is real: the sample's own cell bound fails near leaf borders on the
    // curved fixtures (measured 4/3000 on sphere, 9/3000 on torus, up to 0.29 over; 0 on box).
    if (!name.startsWith("box")) expect(undilatedMisses).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 4. The walk

/** f64 voxel crossings of the ray on [t0, t1]: each voxel's [ta, tb]. */
function voxelIntervals(o: Vec3, d: Vec3, t0: number, t1: number): { ijk: Vec3; ta: number; tb: number }[] {
  const ts = [t0, t1];
  for (let a = 0; a < 3; a++) {
    if (d[a] === 0) continue;
    const p0 = o[a]! + d[a]! * t0;
    const p1 = o[a]! + d[a]! * t1;
    for (let c = Math.ceil(Math.min(p0, p1)); c <= Math.floor(Math.max(p0, p1)); c++) ts.push((c - o[a]!) / d[a]!);
  }
  ts.sort((x, y) => x - y);
  const out: { ijk: Vec3; ta: number; tb: number }[] = [];
  for (let i = 0; i + 1 < ts.length; i++) {
    const ta = Math.max(ts[i]!, t0);
    const tb = Math.min(ts[i + 1]!, t1);
    if (tb <= ta) continue;
    const tm = (ta + tb) / 2;
    out.push({ ijk: [Math.floor(o[0] + d[0] * tm), Math.floor(o[1] + d[1] * tm), Math.floor(o[2] + d[2] * tm)], ta, tb });
  }
  return out;
}

function randomRay(img: Uint32Array, r: () => number): { o: Vec3; d: Vec3 } {
  const { min, max } = indexBBox(img);
  const c = [0, 1, 2].map((a) => (min[a]! + max[a]! + 1) / 2);
  const ext = Math.max(...[0, 1, 2].map((a) => max[a]! - min[a]! + 1));
  const inside = r() < 0.25;
  const o = [0, 1, 2].map((a) => (inside ? min[a]! + r() * (max[a]! - min[a]! + 1) : c[a]! + (r() - 0.5) * 2 * ext)).map(Math.fround) as Vec3;
  const target = [0, 1, 2].map((a) => min[a]! + r() * (max[a]! - min[a]! + 1));
  let d = [0, 1, 2].map((a) => target[a]! - o[a]! + (r() - 0.5) * 1e-3);
  const len = Math.hypot(...d);
  d = d.map((x) => x / len);
  return { o, d: d.map(Math.fround) as Vec3 };
}

function checkWalk(img: Uint32Array, o: Vec3, d: Vec3, segs: HddaSegment[], threshold: number, label: string): number {
  const clip = rayClip(img, o, d, [0, 3.4e38]);
  if (!clip.hit) {
    expect(segs).toEqual([]);
    return 0;
  }
  const eps = (t: number) => 1e-4 * Math.max(1, Math.abs(t));
  // Cells tile the clipped interval.
  expect(segs.length, label).toBeGreaterThan(0);
  expect(segs[0]!.t0).toBe(clip.t0);
  expect(Math.abs(segs[segs.length - 1]!.t1 - clip.t1), label).toBeLessThanOrEqual(eps(clip.t1));
  for (let k = 1; k < segs.length; k++) {
    const gap = segs[k]!.t0 - segs[k - 1]!.t1;
    expect(Math.abs(gap), `${label} gap at ${k}`).toBeLessThanOrEqual(eps(segs[k]!.t0));
  }
  for (const s of segs) expect(s.t1).toBeGreaterThanOrEqual(s.t0 - eps(s.t0));
  // Every active voxel the ray crosses (value >= threshold) is in a kept cell that bounds it.
  let checked = 0;
  for (const v of voxelIntervals(o, d, clip.t0, clip.t1)) {
    if (v.tb - v.ta < 0.1 || !isActive(img, v.ijk)) continue;
    const value = vdbReadValueF32(img, v.ijk);
    if (value < threshold) continue;
    const tm = (v.ta + v.tb) / 2;
    const kept = segs.filter((s) => !s.skip && s.t0 - eps(tm) <= tm && tm <= s.t1 + eps(tm));
    expect(kept.length, `${label}: active voxel ${v.ijk} (t ${tm}) not in a kept cell`).toBeGreaterThan(0);
    const exact = kept.find((s) => s.t0 <= tm && tm <= s.t1);
    if (exact) expect(value, `${label}: voxel ${v.ijk} above its cell bound`).toBeLessThanOrEqual(exact.max + ulps(value));
    checked++;
  }
  return checked;
}

describe("hddaWalk properties", () => {
  it.skipIf(!present).each(PRIMITIVES)("%s: random rays, leafDim 8 and 1, with and without a threshold", (name) => {
    const img = load(name);
    const r = rng(name.length * 977);
    const rootMax = nodeStats(img, 3, treeNodes(img)[0]!.address).max;
    let active = 0;
    let skipped = 0;
    for (let k = 0; k < 60; k++) {
      const { o, d } = randomRay(img, r);
      const leafDim = k % 4 === 3 ? 1 : 8;
      const threshold = k % 3 === 2 ? rootMax * 0.5 : -Infinity;
      const segs = hddaWalk(img, o, d, { leafDim, threshold });
      for (const s of segs) {
        expect([1, 8, 128, 4096]).toContain(s.dim);
        expect(s.dim).toBeGreaterThanOrEqual(leafDim);
        if (s.skip) skipped++;
      }
      active += checkWalk(img, o, d, segs, threshold, `${name} ray ${k}`);
    }
    expect(active).toBeGreaterThan(50);
    expect(skipped).toBeGreaterThan(0);
  });

  it("a ray that misses the bbox walks nothing", () => {
    const n = 16;
    const img = vdbBuildFromDense(new Float32Array(n * n * n).fill(1), [n, n, n]);
    expect(hddaWalk(img, [-10, -10, -10], [-1, 0.5, 0.25])).toEqual([]);
    const segs = hddaWalk(img, [-10, 4.5, 3.25], [1, 0.001, 0.002]);
    expect(segs.length).toBeGreaterThan(0);
    // A constant block: leaves or (our builder's) active tiles, all kept, at the leaf dim.
    // The vendored step continues while tmin <= tmax, so a zero-length cell at the exit face can follow.
    const cells = segs.filter((s) => s.t1 > s.t0);
    expect(cells.map((s) => s.voxel[0])).toEqual([0, 8]);
    expect(cells.every((s) => s.dim === 8 && !s.skip && s.active && s.max === 1)).toBe(true);
    expect(segs.slice(cells.length).every((s) => s.t0 === s.t1)).toBe(true);
    expect(segs[0]!.t0).toBe(10);
  });
});

// ---------------------------------------------------------------------------
// 5. outside: 'zero' samplers

describe("outside: 'zero' samplers (voxel centres at integers)", () => {
  // Value 1 on voxels [0, 7]^3, background 3 (as on the sphere fixture's root).
  const img = vdbBuildFromDense(new Float32Array(512).fill(1), [8, 8, 8], { background: 3 });

  it("trilinear: zero outside [min - 1, max + 1), taps outside the box read 0, symmetric at both ends", () => {
    const zero = { outside: "zero" } as const;
    expect(vdbSampleTrilinear(img, [3, 3, 3], zero)).toBe(1);
    expect(vdbSampleTrilinear(img, [-0.5, 3, 3], zero)).toBeCloseTo(0.5, 6);
    expect(vdbSampleTrilinear(img, [7.5, 3, 3], zero)).toBeCloseTo(0.5, 6);
    expect(vdbSampleTrilinear(img, [-1, 3, 3], zero)).toBe(0);
    expect(vdbSampleTrilinear(img, [8, 3, 3], zero)).toBe(0);
    expect(vdbSampleTrilinear(img, [-1.2, 3, 3], zero)).toBe(0);
    // Background mode blends the background in.
    expect(vdbSampleTrilinear(img, [7.5, 3, 3])).toBeCloseTo(2, 6);
  });

  it("nearest: zero outside [min - 0.5, max + 0.5)", () => {
    const zero = { outside: "zero" } as const;
    expect(vdbSampleNearest(img, [-0.4, 3, 3], zero)).toBe(1);
    expect(vdbSampleNearest(img, [-0.5, 3, 3], zero)).toBe(1);
    expect(vdbSampleNearest(img, [-0.6, 3, 3], zero)).toBe(0);
    expect(vdbSampleNearest(img, [7.4, 3, 3], zero)).toBe(1);
    expect(vdbSampleNearest(img, [7.5, 3, 3], zero)).toBe(0);
    expect(vdbSampleNearest(img, [7.6, 3, 3], zero)).toBe(0);
    expect(vdbSampleNearest(img, [7.6, 3, 3])).toBe(3); // background mode reads voxel 8
  });
});

// ---------------------------------------------------------------------------
// 6. K1.7a opt-in helper twins

describe("cellBoundTrilinear (vdbCellBoundTrilinear twin)", () => {
  it.skipIf(!present).each(PRIMITIVES)("%s: the 2x2x2 nodeMax block bounds every trilinear sample whose base lies in the cell", (name) => {
    const img = load(name);
    // Precondition (as for the dilated bound above): inactive leaf voxels are <= their leaf max.
    expect(subtreeExtrema(img).inactiveAboveLeafMax).toBe(0);
    const { min, max } = indexBBox(img);
    const r = rng(31 + name.length);
    let cellOnlyMisses = 0;
    for (let k = 0; k < 3000; k++) {
      const p: Vec3 = [0, 1, 2].map((a) => min[a]! - 2 + r() * (max[a]! - min[a]! + 4)) as Vec3;
      const s = vdbSampleTrilinear(img, p);
      const b: Vec3 = [Math.floor(p[0]), Math.floor(p[1]), Math.floor(p[2])];
      for (const dim of [8, 1]) {
        const cell: Vec3 = [b[0] & ~(dim - 1), b[1] & ~(dim - 1), b[2] & ~(dim - 1)];
        // Any voxel of the cell names it: the corner and the base voxel give the same bound.
        const bound = cellBoundTrilinear(img, cell, dim);
        expect(cellBoundTrilinear(img, b, dim)).toBe(bound);
        expect(s, `${name} at ${p} dim ${dim}`).toBeLessThanOrEqual(bound + 1e-5);
        // Definition: the max of vdbNodeMax over the cell and its +x/+y/+z neighbours.
        let want = -Infinity;
        for (let t = 0; t < 8; t++) {
          want = Math.max(want, nodeMax(img, [cell[0] + (t & 1) * dim, cell[1] + ((t >> 1) & 1) * dim, cell[2] + ((t >> 2) & 1) * dim], dim));
        }
        expect(bound).toBe(want);
        if (dim === 8 && s > nodeMax(img, cell, 8) + 1e-5) cellOnlyMisses++;
      }
    }
    // The cell's own bound is not enough on the curved fixtures (the reason for the helper).
    if (!name.startsWith("box")) expect(cellOnlyMisses).toBeGreaterThan(0);
  });

  it("reaches into the +x neighbour cell: an empty cell before a block of 1", () => {
    // Voxels [8, 15] x [0, 7] x [0, 7] are 1: cell x=0 holds nothing, but its samples at x in (7, 8) read voxel 8.
    const dense = new Float32Array(16 * 8 * 8);
    for (let x = 8; x < 16; x++) for (let y = 0; y < 8; y++) for (let z = 0; z < 8; z++) dense[x * 64 + y * 8 + z] = 1;
    const img = vdbBuildFromDense(dense, [16, 8, 8]);
    expect(nodeMax(img, [0, 0, 0], 8)).toBeLessThanOrEqual(0);
    expect(vdbSampleTrilinear(img, [7.5, 3, 3])).toBeCloseTo(0.5, 6);
    expect(cellBoundTrilinear(img, [0, 0, 0], 8)).toBe(1);
  });
});

describe("hddaCellEnd (vdbHddaCellEnd twin)", () => {
  it.skipIf(!present).each(PRIMITIVES)("%s: is min(next, tmax), and the next cell starts where it says the current one ends", (name) => {
    const img = load(name);
    const r = rng(53 + name.length);
    let steps = 0;
    for (let k = 0; k < 20; k++) {
      const { o, d } = randomRay(img, r);
      const clip = rayClip(img, o, d, [0, 3.4e38]);
      if (!clip.hit) continue;
      const h = hddaInit(o, clip.t0, d, clip.t1, 8);
      for (let i = 0; i < 4096; i++) {
        const end = hddaCellEnd(h);
        expect(end).toBe(Math.min(h.next[0], h.next[1], h.next[2], h.tmax));
        expect(end).toBeGreaterThanOrEqual(h.tmin);
        if (!hddaStep(h)) break;
        // Forward stepping may nudge a zero-length cell; otherwise the cells tile exactly.
        expect(Math.abs(h.tmin - end)).toBeLessThanOrEqual(1e-5 * Math.max(1, Math.abs(end)));
        steps++;
      }
    }
    expect(steps).toBeGreaterThan(20);
  });
});

describe("rayClipSupport (vdbRayClipSupport twin)", () => {
  // Value 1 on voxels [0, 7]^3.
  const img = vdbBuildFromDense(new Float32Array(512).fill(1), [8, 8, 8]);
  const o: Vec3 = [-10, 3.5, 3.5];
  const d: Vec3 = [1, 0, 0];

  it("trilinear: [min - 1, max + 1] keeps the min-side voxel of support that vdbRayClip cuts", () => {
    expect(rayClip(img, o, d, [0, 100])).toEqual({ hit: true, t0: 10, t1: 18 });
    expect(rayClipSupport(img, o, d, [0, 100])).toEqual({ hit: true, t0: 9, t1: 18 });
    // What vdbRayClip cuts: x in (-1, 0) samples non-zero, also with outside: 'zero'.
    expect(vdbSampleTrilinear(img, [-0.5, 3.5, 3.5], { outside: "zero" })).toBeCloseTo(0.5, 6);
    // At and past the support box the zero-outside field is 0.
    expect(vdbSampleTrilinear(img, [-1, 3.5, 3.5], { outside: "zero" })).toBe(0);
    expect(vdbSampleTrilinear(img, [8, 3.5, 3.5], { outside: "zero" })).toBe(0);
  });

  it("nearest: [min - 0.5, max + 0.5]", () => {
    expect(rayClipSupport(img, o, d, [0, 100], { sampler: "nearest" })).toEqual({ hit: true, t0: 9.5, t1: 17.5 });
    expect(vdbSampleNearest(img, [-0.5, 3.5, 3.5], { outside: "zero" })).toBe(1);
    expect(vdbSampleNearest(img, [-0.6, 3.5, 3.5], { outside: "zero" })).toBe(0);
  });

  it("hits a ray that passes only through the min-side support, which vdbRayClip misses", () => {
    const graze: Vec3 = [-0.5, -10, 3.5];
    expect(rayClip(img, graze, [0, 1, 0], [0, 100]).hit).toBe(false);
    expect(rayClipSupport(img, graze, [0, 1, 0], [0, 100])).toEqual({ hit: true, t0: 9, t1: 18 });
    expect(vdbSampleTrilinear(img, [-0.5, 3.5, 3.5], { outside: "zero" })).toBeGreaterThan(0);
  });

  it.skipIf(!present).each(PRIMITIVES)("%s: the support interval contains the voxel-cell interval", (name) => {
    const fixture = load(name);
    const r = rng(71 + name.length);
    let hits = 0;
    for (let k = 0; k < 200; k++) {
      const ray = randomRay(fixture, r);
      const a = rayClip(fixture, ray.o, ray.d, [0, 3.4e38]);
      const b = rayClipSupport(fixture, ray.o, ray.d, [0, 3.4e38]);
      if (!a.hit) continue;
      hits++;
      expect(b.hit).toBe(true);
      expect(b.t0).toBeLessThanOrEqual(a.t0);
      expect(b.t1).toBeGreaterThanOrEqual(a.t1);
    }
    expect(hits).toBeGreaterThan(50);
  });
});
