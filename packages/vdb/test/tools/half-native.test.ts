/**
 * Half grids against native NanoVDB (K7.H): OpenVDB v14.0.0
 * `tools::createNanoGrid<…, nanovdb::Half>` (NanoVDB 32.11.0), baked by
 * `docker/fixture-bake/bake_half.cpp` (`pnpm fixtures:bake-half`).
 *
 * Always (committed sidecars):
 *   - `fixtures/half/half_edge.sidecar.json`: our writer builds the same grid
 *     (`VDB_HALF_LEAF_CODEC`) from the same f32 inputs and must store the same
 *     bits for every voxel, inactive probe and the background, the same leaf
 *     bbox/flags, the same binary16 min/max at every node, and ave/stddev
 *     within f32 rounding (native accumulates in f64 Welford, we in f64 sums).
 * With the float primitives (`pnpm fixtures:bake`): `vdbQuantize(float, 'half')`
 *   matches the native Half sidecars' probe bits and leaf-value hash.
 * With the native Half grids (`pnpm fixtures:bake-half`): our readers decode
 *   them as native does, and our quantize output equals them node for node.
 */
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { NvdbFile, VDB_HALF_LEAF_CODEC, vdbBuildFromLeavesDetailed, vdbFloatToHalf, vdbHalfToFloat, vdbQuantize, vdbReadValueCpu } from "../../src/tools/index.js";
import { isActive, nodeStats, treeNodes, vdbReadValueF32, type TreeNode, type Vec3 } from "../../src/tools/cpu/march.js";

const FIXTURES = new URL("../../../../fixtures/", import.meta.url);
const read = (rel: string) => readFileSync(new URL(rel, FIXTURES));
const has = (rel: string) => existsSync(new URL(rel, FIXTURES));
function loadImage(rel: string): Uint32Array {
  const buf = read(rel);
  return NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).gridImage(0);
}

interface Stats {
  min: number;
  max: number;
  ave: number;
  stddev: number;
}
interface NativeSummary {
  nanovdb: string;
  gridFlags: number;
  background: number;
  root: Stats;
  leafBlocksFnv1a: number;
  leafValuesFnv1a: number;
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const fromBits = (b: number) => {
  u32[0] = b >>> 0;
  return f32[0]!;
};
const bitsOf = (v: number) => {
  f32[0] = v;
  return u32[0]!;
};
/** f32 ulps between two values given as f32 bits (NaN == NaN). */
function ulps(a: number, b: number): number {
  const x = fromBits(a);
  const y = fromBits(b);
  if (Number.isNaN(x) && Number.isNaN(y)) return 0;
  if (x === y) return 0;
  const ord = (bits: number) => (bits & 0x80000000 ? -(bits & 0x7fffffff) : bits);
  return Math.abs(ord(a >>> 0) - ord(b >>> 0));
}

/** Raw stats of the node at `address` (`level` 0 leaf, 1 lower, 2 upper, 3 root) of a Half image. */
function rawStats(image: Uint32Array, level: 0 | 1 | 2 | 3, address: number): Stats {
  const v = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const off = [80, 1056, 8224, 30][level]!;
  // Half: min at off, max at off + 2, ave at off + 4 (root: 36), stddev at off + 8 (root: 40).
  const ave = level === 3 ? address + 36 : address + off + 4;
  return {
    min: v.getUint16(address + off, true),
    max: v.getUint16(address + off + 2, true),
    ave: v.getUint32(ave, true),
    stddev: v.getUint32(ave + 4, true),
  };
}

function fnv1a(bytes: Uint8Array, h = 0x811c9dc5): number {
  for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i]!, 0x01000193) >>> 0;
  return h;
}

/** FNV-1a of every leaf's value table (bytes 96..1120), in memory (breadth-first) order: bake_half.cpp's `leafValuesFnv1a`. */
function leafValuesHash(image: Uint32Array): number {
  const v = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const firstLeaf = 672 + Number(v.getBigUint64(672, true));
  const count = v.getUint32(672 + 32, true);
  const bytes = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
  let h = 0x811c9dc5;
  for (let i = 0; i < count; i++) {
    const leaf = firstLeaf + i * 1120;
    h = fnv1a(bytes.subarray(leaf + 96, leaf + 1120), h);
  }
  return h;
}

const halfBits = (image: Uint32Array, ijk: Vec3) => vdbFloatToHalf(vdbReadValueCpu(image, ijk).value);

// ---------------------------------------------------------------------------

describe("half_edge: our writer vs native createNanoGrid<build::Grid<float>, Half>", () => {
  const sc = JSON.parse(read("half/half_edge.sidecar.json").toString("utf8")) as {
    grid: { gridByteSize: number; activeVoxelCount: number; nodeCounts: { leaf: number; lower: number; upper: number }; backgroundInput: number };
    native: NativeSummary;
    voxels: { ijk: Vec3; input: number; bits: number; active: boolean }[];
    inactive: { ijk: Vec3; bits: number; active: boolean }[];
    leaves: { origin: Vec3; flags: number; bboxMin: Vec3; bboxDif: Vec3; stats: Stats }[];
    lowers: { origin: Vec3; stats: Stats }[];
    uppers: { origin: Vec3; stats: Stats }[];
  };
  const background = fromBits(sc.grid.backgroundInput);

  // Rebuild it from leaves: inactive voxels hold the background, as build::Grid does.
  const leaves = new Map<string, { origin: [number, number, number]; values: Float32Array; valueMask: Uint32Array }>();
  for (const v of sc.voxels) {
    const origin: [number, number, number] = [v.ijk[0] & ~7, v.ijk[1] & ~7, v.ijk[2] & ~7];
    const key = origin.join(",");
    let leaf = leaves.get(key);
    if (!leaf) leaves.set(key, (leaf = { origin, values: new Float32Array(512).fill(background), valueMask: new Uint32Array(16) }));
    const n = ((v.ijk[0] & 7) << 6) | ((v.ijk[1] & 7) << 3) | (v.ijk[2] & 7);
    leaf.values[n] = fromBits(v.input);
    leaf.valueMask[n >>> 5]! |= 1 << (n & 31);
  }
  const built = vdbBuildFromLeavesDetailed(leaves.values(), VDB_HALF_LEAF_CODEC, { background, gridName: "half_edge", gridClass: "Unknown" });
  const image = built.image;
  const nodes = treeNodes(image);
  const byOrigin = (level: number) => new Map(nodes.filter((n) => n.level === level).map((n) => [n.origin.join(","), n] as [string, TreeNode]));

  it("the sidecar was baked by NanoVDB >= 13.2 (Half implemented)", () => {
    const [major, minor] = sc.native.nanovdb.split(".").map(Number) as [number, number];
    expect(major).toBe(32);
    expect(minor).toBeGreaterThanOrEqual(11);
  });

  it("vdbFloatToHalf gives native's bits for every input (ties, subnormals, overflow, -0)", () => {
    for (const v of sc.voxels) expect(vdbFloatToHalf(fromBits(v.input)), `input ${fromBits(v.input)}`).toBe(v.bits);
    expect(vdbFloatToHalf(background)).toBe(sc.native.background);
  });

  it("stores the same bits for every voxel, inactive probe and the background; same layout size", () => {
    for (const v of sc.voxels) {
      expect(halfBits(image, v.ijk), `${v.ijk}`).toBe(v.bits);
      expect(isActive(image, v.ijk), `${v.ijk}`).toBe(v.active);
    }
    for (const p of sc.inactive) {
      expect(halfBits(image, p.ijk), `${p.ijk}`).toBe(p.bits);
      expect(isActive(image, p.ijk)).toBe(p.active);
    }
    expect(built.voxelCount).toBe(sc.grid.activeVoxelCount);
    expect(built.nodeCounts).toEqual(sc.grid.nodeCounts);
    expect(image.byteLength).toBe(sc.grid.gridByteSize);
    expect(new DataView(image.buffer).getUint32(20, true)).toBe(sc.native.gridFlags);
    const root = nodes[0]!.address;
    expect(new DataView(image.buffer).getUint16(root + 28, true)).toBe(sc.native.background);
  });

  it("node stats: binary16 min/max equal, f32 ave/stddev within 1 ulp (NaN where native has NaN), leaf bbox and flags equal", () => {
    // Where an input overflowed to +-infinity, native's f64 Welford gives NaN and our f64 sums may give
    // +-infinity (DIVERGES: Half stats with infinite values): both are "not finite".
    const close = (got: number, want: number, what: string) => {
      if (!Number.isFinite(fromBits(want))) expect(Number.isFinite(fromBits(got)), `${what} ${fromBits(got)} vs ${fromBits(want)}`).toBe(false);
      else expect(ulps(got, want), `${what} ${fromBits(got)} vs ${fromBits(want)}`).toBeLessThanOrEqual(1);
    };
    const check = (got: Stats, want: Stats, what: string) => {
      expect(got.min, `${what} min`).toBe(want.min);
      expect(got.max, `${what} max`).toBe(want.max);
      close(got.ave, want.ave, `${what} ave`);
      close(got.stddev, want.stddev, `${what} stddev`);
    };
    const leafNodes = byOrigin(0);
    const v = new DataView(image.buffer);
    for (const l of sc.leaves) {
      const n = leafNodes.get(l.origin.join(","));
      expect(n, `leaf ${l.origin}`).toBeTruthy();
      check(rawStats(image, 0, n!.address), l.stats, `leaf ${l.origin}`);
      expect([v.getInt32(n!.address, true), v.getInt32(n!.address + 4, true), v.getInt32(n!.address + 8, true)]).toEqual(l.bboxMin);
      expect([v.getUint8(n!.address + 12), v.getUint8(n!.address + 13), v.getUint8(n!.address + 14)]).toEqual(l.bboxDif);
      expect(v.getUint8(n!.address + 15), `leaf flags ${l.origin}`).toBe(l.flags);
      // The f32 twin reads the binary16 min/max exactly.
      const s = nodeStats(image, 0, n!.address);
      expect(Object.is(s.min, vdbHalfToFloat(l.stats.min))).toBe(true);
    }
    const lowers = byOrigin(1);
    for (const l of sc.lowers) check(rawStats(image, 1, lowers.get(l.origin.join(","))!.address), l.stats, `lower ${l.origin}`);
    const uppers = byOrigin(2);
    for (const u of sc.uppers) check(rawStats(image, 2, uppers.get(u.origin.join(","))!.address), u.stats, `upper ${u.origin}`);
    check(rawStats(image, 3, nodes[0]!.address), sc.native.root, "root");
  });

  it.skipIf(!has("half/half_edge.nvdb"))("native .nvdb: our readers decode every voxel to native's bits; our leaf blocks equal native's", () => {
    const native = loadImage("half/half_edge.nvdb");
    for (const v of sc.voxels) {
      expect(halfBits(native, v.ijk)).toBe(v.bits);
      expect(vdbFloatToHalf(vdbReadValueF32(native, v.ijk))).toBe(v.bits);
    }
    for (const p of sc.inactive) expect(halfBits(native, p.ijk)).toBe(p.bits);
    expect(leafValuesHash(native)).toBe(sc.native.leafValuesFnv1a);
    expect(leafValuesHash(image)).toBe(sc.native.leafValuesFnv1a);
    // Whole leaf blocks: bbox, flags, mask, stats and values (ave/stddev bits may differ by an ulp; they agree here).
    const nNodes = treeNodes(native).filter((n) => n.level === 0);
    const ours = byOrigin(0);
    const nb = new Uint8Array(native.buffer, native.byteOffset, native.byteLength);
    const ob = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
    let identical = 0;
    for (const n of nNodes) {
      const o = ours.get(n.origin.join(","))!;
      const a = nb.subarray(n.address, n.address + 1120);
      const b = ob.subarray(o.address, o.address + 1120);
      // Bytes 84..92 (ave, stddev) are compared in ulps above; everything else must match.
      for (let k = 0; k < 1120; k++) if (k < 84 || k >= 92) expect(b[k], `leaf ${n.origin} byte ${k}`).toBe(a[k]);
      if (a.every((x, k) => x === b[k])) identical++;
    }
    expect(identical).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------

const PRIMITIVES = ["sphere_fog", "torus_fog", "box_fog"] as const;

describe.each(PRIMITIVES)("%s: vdbQuantize(float, 'half') vs native createNanoGrid<FloatGrid, Half>", (name) => {
  const sc = JSON.parse(read(`primitives/${name}_half.sidecar.json`).toString("utf8")) as {
    grid: { activeVoxelCount: number; nodeCounts: { leaf: number; lower: number; upper: number } };
    native: NativeSummary;
    samples: { ijk: Vec3; value: number; bits: number; active: boolean }[];
  };
  const haveFloat = has(`primitives/${name}_float.nvdb`);
  const haveHalf = has(`primitives/${name}_half.nvdb`);
  const ours = haveFloat ? vdbQuantize(loadImage(`primitives/${name}_float.nvdb`), "half") : null;

  it("sidecar probes decode to their bits (sidecar self-consistency)", () => {
    // `value` is printed with 9 significant digits: it names the f32 (every half is one) exactly.
    for (const s of sc.samples) expect(vdbHalfToFloat(s.bits)).toBe(Math.fround(s.value));
  });

  it.skipIf(!haveFloat)("same bits at every probe, same leaf value tables (hash), same root min/max", () => {
    for (const s of sc.samples) {
      expect(halfBits(ours!, s.ijk), `${s.ijk}`).toBe(s.bits);
      expect(isActive(ours!, s.ijk), `${s.ijk}`).toBe(s.active);
    }
    expect(leafValuesHash(ours!)).toBe(sc.native.leafValuesFnv1a);
    const root = rawStats(ours!, 3, treeNodes(ours!)[0]!.address);
    // Root min may be +0 or -0 (native keeps the first zero it meets); compare values.
    expect(vdbHalfToFloat(root.min)).toBe(vdbHalfToFloat(sc.native.root.min));
    expect(root.max).toBe(sc.native.root.max);
    expect(ulps(root.ave, sc.native.root.ave)).toBeLessThanOrEqual(2);
    expect(ulps(root.stddev, sc.native.root.stddev)).toBeLessThanOrEqual(2);
  });

  it.skipIf(!haveHalf)("our readers decode the native Half grid to the sidecar's bits", () => {
    const native = loadImage(`primitives/${name}_half.nvdb`);
    for (const s of sc.samples) {
      expect(halfBits(native, s.ijk), `${s.ijk}`).toBe(s.bits);
      expect(vdbReadValueF32(native, s.ijk)).toBe(Math.fround(s.value));
      expect(isActive(native, s.ijk)).toBe(s.active);
    }
    expect(leafValuesHash(native)).toBe(sc.native.leafValuesFnv1a);
  });

  it.skipIf(!haveHalf || !haveFloat)("node for node: same values, masks, tiles and binary16 min/max as native", () => {
    const native = loadImage(`primitives/${name}_half.nvdb`);
    const key = (n: TreeNode) => [n.level, ...n.origin].join(",");
    const a = treeNodes(native);
    // Same node set; root tiles may be stored in another order (treeNodes walks them in table order).
    const oursByKey = new Map(treeNodes(ours!).map((n) => [key(n), n]));
    expect([...oursByKey.keys()].sort()).toEqual(a.map(key).sort());
    const b = a.map((n) => oursByKey.get(key(n))!);
    const nb = new Uint8Array(native.buffer, native.byteOffset, native.byteLength);
    const ob = new Uint8Array(ours!.buffer, ours!.byteOffset, ours!.byteLength);
    let aveUlps = 0;
    for (let k = 1; k < a.length; k++) {
      const n = a[k]!;
      const o = b[k]!;
      const tileKey = (t: { origin: Vec3; dim: number }) => `${t.dim}:${t.origin.join(",")}`;
      expect([...o.activeTiles].sort((x, y) => (tileKey(x) < tileKey(y) ? -1 : 1))).toEqual([...n.activeTiles].sort((x, y) => (tileKey(x) < tileKey(y) ? -1 : 1)));
      const sa = rawStats(native, n.level as 0 | 1 | 2, n.address);
      const sb = rawStats(ours!, o.level as 0 | 1 | 2, o.address);
      expect(vdbHalfToFloat(sb.min), `${n.level} ${n.origin} min`).toBe(vdbHalfToFloat(sa.min));
      expect(sb.max, `${n.level} ${n.origin} max`).toBe(sa.max);
      aveUlps = Math.max(aveUlps, ulps(sb.ave, sa.ave), ulps(sb.stddev, sa.stddev));
      if (n.level === 0) {
        // Value mask (16..80) and value table (96..1120) byte for byte.
        for (let i = 16; i < 80; i++) if (ob[o.address + i] !== nb[n.address + i]) expect(`mask byte ${i}`).toBe("equal");
        for (let i = 96; i < 1120; i++) if (ob[o.address + i] !== nb[n.address + i]) expect(`${n.origin} value byte ${i}`).toBe("equal");
      }
    }
    expect(aveUlps).toBeLessThanOrEqual(4);
  });
});
