/**
 * Phase 5 serializer test suite (T3). The correctness oracle is the repo's own
 * proven readers — nothing native runs here:
 *   - `read-value.ts` (657/657-validated CPU reference) reads our images;
 *   - `NanoVDBFile` parses our `writeNvdb` files (incl. its mGridSize check);
 *   - `sphere_fog_float.nvdb` (native `nanovdb_convert` output) is the
 *     structural-conformance reference.
 *
 * Invariants (see the T3 brief):
 *   A. round-trip value fidelity (every voxel + background outside)
 *   B. loader acceptance + gridImage byte-equality + mGridSize cross-check
 *   C. structural conformance vs a native fixture
 *   D. emit a git-ignored fixture for the GPU harness
 * Plus a cross-check that bytes.ts constants match the extracted stride tables.
 */

import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  buildFromDense,
  buildFromDenseDetailed,
  writeNvdb,
} from "../src/nanovdb/index.js";
import { readValue } from "../../nanovdb-wgsl/src/cpu/read-value.js";
import { NanoVDBFile } from "../../nanovdb-wgsl/src/nvdb-file.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Deterministic splitmix32 PRNG for reproducible random inputs. */
function splitmix32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x9e3779b9) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const denseIndex = (dims: [number, number, number], x: number, y: number, z: number): number =>
  (x * dims[1] + y) * dims[2] + z;

/** Fast native byte comparison of two u32 images. */
function bytesEqual(a: Uint32Array, b: Uint32Array): boolean {
  const ba = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
  const bb = Buffer.from(b.buffer, b.byteOffset, b.byteLength);
  return ba.length === bb.length && Buffer.compare(ba, bb) === 0;
}

interface Case {
  name: string;
  dims: [number, number, number];
  values: Float32Array;
  origin?: [number, number, number];
  background?: number;
  activeThreshold?: number;
}

function singleLeafCase(): Case {
  // 8x8x8 -> exactly one leaf. Ramp values; a couple zeros for inactive slots.
  const dims: [number, number, number] = [8, 8, 8];
  const values = new Float32Array(8 * 8 * 8);
  for (let x = 0; x < 8; x++)
    for (let y = 0; y < 8; y++)
      for (let z = 0; z < 8; z++) {
        const i = denseIndex(dims, x, y, z);
        values[i] = (x + y + z) % 5 === 0 ? 0 : 0.1 + 0.01 * (x * 64 + y * 8 + z);
      }
  return { name: "single-leaf 8^3", dims, values };
}

function crossBoundaryCase(): Case {
  // 33^3 crosses leaf (8) and lower (128 within one) boundaries; offset origin
  // so it straddles a negative/positive index-space region.
  const dims: [number, number, number] = [33, 33, 33];
  const values = new Float32Array(33 * 33 * 33);
  for (let x = 0; x < 33; x++)
    for (let y = 0; y < 33; y++)
      for (let z = 0; z < 33; z++) {
        const i = denseIndex(dims, x, y, z);
        // sphere-ish blob centred in the block
        const dx = x - 16,
          dy = y - 16,
          dz = z - 16;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        values[i] = r < 14 ? 1 - r / 14 : 0;
      }
  return { name: "33^3 crossing node boundaries", dims, values, origin: [-4, -4, -4] };
}

function sparseBlobCase(): Case {
  // 100^3 with a small blob near one corner and lots of background -> exercises
  // background collapse and multi-upper-node topology (spans past 4096? no, but
  // crosses many lower/upper-internal slots and origin puts it across 0).
  const dims: [number, number, number] = [100, 100, 100];
  const values = new Float32Array(100 * 100 * 100);
  const centres: [number, number, number][] = [
    [20, 20, 20],
    [80, 75, 30],
    [50, 50, 90],
  ];
  for (const [cx, cy, cz] of centres) {
    for (let x = 0; x < 100; x++)
      for (let y = 0; y < 100; y++)
        for (let z = 0; z < 100; z++) {
          const dx = x - cx,
            dy = y - cy,
            dz = z - cz;
          const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (r < 10) {
            const i = denseIndex(dims, x, y, z);
            values[i] = Math.max(values[i]!, 1 - r / 10);
          }
        }
  }
  return { name: "100^3 sparse blob", dims, values, origin: [-50, -50, -50] };
}

function randomCase(): Case {
  const dims: [number, number, number] = [20, 20, 20];
  const values = new Float32Array(20 * 20 * 20);
  const rng = splitmix32(0xc0ffee);
  for (let i = 0; i < values.length; i++) {
    const r = rng();
    values[i] = r < 0.3 ? 0 : r; // ~30% background
  }
  return { name: "randomized 20^3 seeded", dims, values };
}

function nonZeroBackgroundCase(): Case {
  // Verifies background propagation into internal-node tiles + leaf slots when
  // background != 0.
  const dims: [number, number, number] = [24, 24, 24];
  const bg = 3;
  const values = new Float32Array(24 * 24 * 24).fill(bg);
  for (let x = 8; x < 16; x++)
    for (let y = 8; y < 16; y++)
      for (let z = 8; z < 16; z++) {
        values[denseIndex(dims, x, y, z)] = 0.5 + 0.01 * (x + y + z);
      }
  return { name: "24^3 nonzero background", dims, values, background: bg };
}

const CASES: Case[] = [
  singleLeafCase(),
  crossBoundaryCase(),
  sparseBlobCase(),
  randomCase(),
  nonZeroBackgroundCase(),
];

function isActive(v: number, background: number, threshold?: number): boolean {
  return threshold === undefined ? v !== background : Math.abs(v - background) > threshold;
}

// ---------------------------------------------------------------------------
// A. Round-trip value fidelity
// ---------------------------------------------------------------------------

describe("A. round-trip value fidelity via read-value.ts", () => {
  for (const c of CASES) {
    it(`${c.name}: every voxel + background outside`, () => {
      const origin = c.origin ?? [0, 0, 0];
      const background = c.background ?? 0;
      const image = buildFromDense(c.values, c.dims, {
        origin,
        background,
        ...(c.activeThreshold !== undefined ? { activeThreshold: c.activeThreshold } : {}),
      });

      let mismatches = 0;
      for (let x = 0; x < c.dims[0] && mismatches < 5; x++)
        for (let y = 0; y < c.dims[1] && mismatches < 5; y++)
          for (let z = 0; z < c.dims[2] && mismatches < 5; z++) {
            const src = c.values[denseIndex(c.dims, x, y, z)]!;
            const ijk: [number, number, number] = [origin[0] + x, origin[1] + y, origin[2] + z];
            const got = readValue(image, ijk);
            if (got.value !== src || got.active !== isActive(src, background)) {
              mismatches++;
              expect
                .soft(`${c.name} @${ijk}: value ${got.value}/${src} active ${got.active}`)
                .toBe(`${c.name} @${ijk}: value ${src}/${src} active ${isActive(src, background)}`);
            }
          }
      expect(mismatches).toBe(0);

      // Background outside the dense block (well past the active bbox).
      for (const far of [
        [origin[0] - 9999, origin[1], origin[2]],
        [origin[0], origin[1] + 9999, origin[2]],
        [origin[0] + c.dims[0] + 5000, origin[1], origin[2] + 5000],
      ] as [number, number, number][]) {
        const got = readValue(image, far);
        expect(got.active).toBe(false);
        expect(got.value).toBe(background);
      }
    });
  }

  it("activeThreshold controls the active flag (value still exact in leaves)", () => {
    const dims: [number, number, number] = [8, 8, 8];
    const values = new Float32Array(8 * 8 * 8);
    // Central voxel strongly active, a ring of small values.
    values[denseIndex(dims, 4, 4, 4)] = 1;
    values[denseIndex(dims, 4, 4, 5)] = 0.05;
    const image = buildFromDense(values, dims, { background: 0, activeThreshold: 0.1 });
    expect(readValue(image, [4, 4, 4])).toEqual({ value: 1, active: true });
    // 0.05 is below threshold -> inactive, but it shares a leaf with the active
    // voxel so its exact (f32-rounded) value is preserved.
    expect(readValue(image, [4, 4, 5])).toEqual({ value: Math.fround(0.05), active: false });
  });
});

// ---------------------------------------------------------------------------
// B. Loader acceptance + gridImage byte-equality + mGridSize cross-check
// ---------------------------------------------------------------------------

describe("B. writeNvdb parses via NanoVDBFile", () => {
  it("single-grid file: metadata + byte-equal image + mGridSize passes", () => {
    const c = crossBoundaryCase();
    const built = buildFromDenseDetailed(c.values, c.dims, {
      origin: c.origin!,
      gridName: "blob",
      gridClass: "FogVolume",
    });
    const file = writeNvdb([built.image]);
    const parsed = NanoVDBFile.fromArrayBuffer(file); // throws if mGridSize mismatches
    expect(parsed.grids).toHaveLength(1);
    const md = parsed.grids[0]!;
    expect(md.name).toBe("blob");
    expect(md.gridType).toBe("Float");
    expect(md.gridClass).toBe("FogVolume");
    expect(md.voxelCount).toBe(built.voxelCount);
    expect(md.indexBBox.min).toEqual(built.indexBBox.min);
    expect(md.indexBBox.max).toEqual(built.indexBBox.max);
    expect(md.worldBBox.min).toEqual(built.worldBBox.min);
    expect(md.worldBBox.max).toEqual(built.worldBBox.max);
    expect(md.voxelSize).toEqual([1, 1, 1]);
    // gridImage byte-equality with the input image (Buffer.compare is O(n)
    // native — avoid vitest's slow structural deep-equal on multi-MB buffers).
    const got = parsed.gridImage(0);
    expect(got.length).toBe(built.image.length);
    expect(bytesEqual(got, built.image)).toBe(true);
    // And it round-trips through readValue after the loader hands it back.
    expect(readValue(got, [built.indexBBox.min[0], built.indexBBox.min[1], built.indexBBox.min[2]]).value).toBe(
      readValue(built.image, [built.indexBBox.min[0], built.indexBBox.min[1], built.indexBBox.min[2]]).value,
    );
  });

  it("multi-grid file: two grids parse independently", () => {
    const a = buildFromDense(singleLeafCase().values, [8, 8, 8], { gridName: "a", voxelSize: 2 });
    const bCase = randomCase();
    const b = buildFromDense(bCase.values, bCase.dims, { gridName: "b" });
    const file = writeNvdb([a, b]);
    const parsed = NanoVDBFile.fromArrayBuffer(file);
    expect(parsed.grids.map((g) => g.name)).toEqual(["a", "b"]);
    expect(parsed.grids[0]!.voxelSize).toEqual([2, 2, 2]);
    expect(bytesEqual(parsed.gridImage(1), b)).toBe(true);
  });

  it("voxelSize + worldOrigin land in worldBBox correctly", () => {
    const c = singleLeafCase();
    const built = buildFromDenseDetailed(c.values, c.dims, {
      voxelSize: 0.5,
      worldOrigin: [10, 20, 30],
      origin: [0, 0, 0],
    });
    // world = index*0.5 + 10/20/30 (translation = worldOrigin - origin*vs = worldOrigin)
    expect(built.worldBBox.min[0]).toBeCloseTo(built.indexBBox.min[0] * 0.5 + 10, 5);
    expect(built.worldBBox.max[0]).toBeCloseTo((built.indexBBox.max[0] + 1) * 0.5 + 10, 5);
    const file = writeNvdb([built.image]);
    const parsed = NanoVDBFile.fromArrayBuffer(file);
    expect(parsed.grids[0]!.worldBBox.min[2]).toBeCloseTo(built.indexBBox.min[2] * 0.5 + 30, 5);
  });
});

// ---------------------------------------------------------------------------
// C. Structural conformance vs the native fixture
// ---------------------------------------------------------------------------

const FIXTURE = fileURLToPath(
  new URL("../../../fixtures/primitives/sphere_fog_float.nvdb", import.meta.url),
);

interface WalkResult {
  gridType: number;
  gridClass: number;
  versionMajor: number;
  nUpper: number;
  nLower: number;
  nLeaf: number;
  nRootTiles: number;
}

/**
 * Structural checker: walks a FLOAT grid image and asserts spec conformance —
 * 32-byte alignment of every block, node counts consistent with child masks,
 * tile/child offsets resolving in-bounds, and exact magic/version/ABI fields.
 */
function walkAndCheck(image: Uint32Array): WalkResult {
  const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const total = image.byteLength;
  const popcount = (x: number): number => {
    x = x >>> 0;
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  };
  const maskBits = (base: number, words: number): number => {
    let c = 0;
    for (let i = 0; i < words; i++) c += popcount(dv.getUint32(base + i * 4, true));
    return c;
  };
  const aligned = (off: number): boolean => off % 32 === 0;

  // GridData.
  expect(dv.getBigUint64(0, true)).toBe(0x314244566f6e614en); // NanoVDB1
  const version = dv.getUint32(16, true);
  const versionMajor = (version >>> 21) & 0x7ff;
  expect(versionMajor).toBe(32);
  const gridType = dv.getUint32(636, true);
  const gridClass = dv.getUint32(632, true);
  expect(gridType).toBe(1); // Float
  expect(Number(dv.getBigUint64(32, true))).toBe(total); // mGridSize

  // Tree block offsets.
  const treeOff = 672;
  expect(aligned(treeOff)).toBe(true);
  const leafOff = treeOff + Number(dv.getBigUint64(treeOff + 0, true));
  const lowerOff = treeOff + Number(dv.getBigUint64(treeOff + 8, true));
  const upperOff = treeOff + Number(dv.getBigUint64(treeOff + 16, true));
  const rootOff = treeOff + Number(dv.getBigUint64(treeOff + 24, true));
  const nLeaf = dv.getUint32(treeOff + 32, true);
  const nLower = dv.getUint32(treeOff + 36, true);
  const nUpper = dv.getUint32(treeOff + 40, true);
  for (const off of [leafOff, lowerOff, upperOff, rootOff]) expect(aligned(off)).toBe(true);

  // Sizes (FLOAT ABI 32.9.1).
  const UPPER = 270400,
    LOWER = 33856,
    LEAF = 2144;
  expect(leafOff + nLeaf * LEAF).toBeLessThanOrEqual(total);
  expect(upperOff + nUpper * UPPER).toBe(lowerOff);
  expect(lowerOff + nLower * LOWER).toBe(leafOff);

  // Root.
  const nRootTiles = dv.getUint32(rootOff + 24, true);
  let childUpperCount = 0;
  for (let i = 0; i < nRootTiles; i++) {
    const tileOff = rootOff + 64 + i * 32;
    const child = Number(dv.getBigInt64(tileOff + 8, true));
    if (child !== 0) {
      const upperAbs = rootOff + child;
      expect(aligned(upperAbs)).toBe(true);
      expect(upperAbs).toBeGreaterThanOrEqual(upperOff);
      expect(upperAbs).toBeLessThan(lowerOff);
      expect((upperAbs - upperOff) % UPPER).toBe(0);
      childUpperCount++;
    }
  }

  // Every upper node: childMask popcount == number of in-bounds lower children.
  for (let u = 0; u < nUpper; u++) {
    const nodeOff = upperOff + u * UPPER;
    const childMaskBits = maskBits(nodeOff + 4128, 1024); // 32768 bits
    for (let n = 0; n < 32768; n++) {
      const word = dv.getUint32(nodeOff + 4128 + (n >>> 5) * 4, true);
      if (((word >>> (n & 31)) & 1) === 0) continue;
      const rel = Number(dv.getBigInt64(nodeOff + 8256 + n * 8, true));
      const lowerAbs = nodeOff + rel;
      expect(aligned(lowerAbs)).toBe(true);
      expect(lowerAbs).toBeGreaterThanOrEqual(lowerOff);
      expect(lowerAbs).toBeLessThan(leafOff);
      expect((lowerAbs - lowerOff) % LOWER).toBe(0);
    }
    expect(childMaskBits).toBeGreaterThanOrEqual(0);
  }

  // Every lower node: child leaf offsets resolve into the leaf block.
  let totalLeafChildren = 0;
  for (let l = 0; l < nLower; l++) {
    const nodeOff = lowerOff + l * LOWER;
    for (let n = 0; n < 4096; n++) {
      const word = dv.getUint32(nodeOff + 544 + (n >>> 5) * 4, true);
      if (((word >>> (n & 31)) & 1) === 0) continue;
      const rel = Number(dv.getBigInt64(nodeOff + 1088 + n * 8, true));
      const leafAbs = nodeOff + rel;
      expect(aligned(leafAbs)).toBe(true);
      expect(leafAbs).toBeGreaterThanOrEqual(leafOff);
      expect(leafAbs + LEAF).toBeLessThanOrEqual(total);
      expect((leafAbs - leafOff) % LEAF).toBe(0);
      totalLeafChildren++;
    }
  }
  // Every leaf is the child of exactly one lower node.
  expect(totalLeafChildren).toBe(nLeaf);
  expect(childUpperCount).toBe(nRootTiles);

  return { gridType, gridClass, versionMajor, nUpper, nLower, nLeaf, nRootTiles };
}

describe("C. structural conformance", () => {
  it("the native sphere_fog_float fixture passes the structural checker", () => {
    const file = readFileSync(FIXTURE);
    const parsed = NanoVDBFile.fromArrayBuffer(
      file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer,
    );
    const image = parsed.gridImage(0);
    const r = walkAndCheck(image);
    expect(r.nLeaf).toBe(788);
    expect(r.nLower).toBe(8);
    expect(r.nUpper).toBe(8);
    expect(r.nRootTiles).toBe(8);
  });

  it("a same-topology grid we build passes the identical checker", () => {
    // Build a small multi-upper grid (origin straddling 0 -> 8 octants like the
    // sphere) and run the very same structural assertions.
    const dims: [number, number, number] = [6, 6, 6];
    const values = new Float32Array(6 * 6 * 6).fill(1);
    const image = buildFromDense(values, dims, { origin: [-3, -3, -3] });
    const r = walkAndCheck(image);
    expect(r.gridType).toBe(1);
    expect(r.nRootTiles).toBe(8); // one upper per octant around origin
    expect(r.nUpper).toBe(8);
  });

  it("a single-leaf grid passes structurally", () => {
    const c = singleLeafCase();
    const image = buildFromDense(c.values, c.dims);
    const r = walkAndCheck(image);
    expect(r.nLeaf).toBe(1);
    expect(r.nLower).toBe(1);
    expect(r.nUpper).toBe(1);
    expect(r.nRootTiles).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// D. Emit a git-ignored fixture for the GPU harness + verify stats
// ---------------------------------------------------------------------------

describe("D. generated fixture + stats", () => {
  it("writes fixtures/generated/dense_test.nvdb", () => {
    const c = crossBoundaryCase();
    const built = buildFromDenseDetailed(c.values, c.dims, {
      origin: c.origin!,
      gridName: "dense_test",
      gridClass: "FogVolume",
    });
    const file = writeNvdb([built.image]);
    const outDir = fileURLToPath(new URL("../../../fixtures/generated/", import.meta.url));
    mkdirSync(outDir, { recursive: true });
    writeFileSync(outDir + "dense_test.nvdb", Buffer.from(file));
    // Sanity: it re-parses.
    const parsed = NanoVDBFile.fromArrayBuffer(file);
    expect(parsed.grids[0]!.name).toBe("dense_test");
  });

  it("node stats match a direct recompute over active voxels (root level)", () => {
    const c = sparseBlobCase();
    const origin = c.origin!;
    const built = buildFromDenseDetailed(c.values, c.dims, { origin });
    const image = built.image;
    const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);

    // Recompute expected root stats over all active voxels.
    let n = 0,
      sum = 0,
      sumSq = 0,
      min = Infinity,
      max = -Infinity;
    for (let i = 0; i < c.values.length; i++) {
      const v = c.values[i]!;
      if (v === 0) continue;
      n++;
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const ave = sum / n;
    const std = Math.sqrt(Math.max(0, sumSq / n - ave * ave));

    const treeOff = 672;
    const rootOff = treeOff + Number(dv.getBigUint64(treeOff + 24, true));
    expect(Number(dv.getBigUint64(treeOff + 56, true))).toBe(n); // voxelCount
    expect(dv.getFloat32(rootOff + 32, true)).toBeCloseTo(min, 5); // root min
    expect(dv.getFloat32(rootOff + 36, true)).toBeCloseTo(max, 5); // root max
    expect(dv.getFloat32(rootOff + 40, true)).toBeCloseTo(ave, 4); // root ave
    expect(dv.getFloat32(rootOff + 44, true)).toBeCloseTo(std, 4); // root stddev
    // Root index bbox == tight active bbox.
    expect(dv.getInt32(rootOff + 0, true)).toBe(built.indexBBox.min[0]);
    expect(dv.getInt32(rootOff + 12, true)).toBe(built.indexBBox.max[0]);
  });
});

// ---------------------------------------------------------------------------
// bytes.ts constants cross-check against the extracted stride tables
// ---------------------------------------------------------------------------

describe("bytes.ts constants match vendor/stride-tables.json", () => {
  it("FLOAT layout + common offsets are in lockstep with the extracted ABI", async () => {
    const json = JSON.parse(
      readFileSync(
        fileURLToPath(
          new URL("../../nanovdb-wgsl/vendor/stride-tables.json", import.meta.url),
        ),
        "utf8",
      ),
    ) as {
      defines: Record<string, number | string>;
      gridTypeConstants: Record<string, Record<string, number>>;
    };
    const B = await import("../src/nanovdb/bytes.js");
    const d = json.defines;
    const f = json.gridTypeConstants.FLOAT!;

    expect(B.GRID_SIZE).toBe(d.PNANOVDB_GRID_SIZE);
    expect(B.TREE_SIZE).toBe(d.PNANOVDB_TREE_SIZE);
    expect(B.GRID_OFF_MAP).toBe(d.PNANOVDB_GRID_OFF_MAP);
    expect(B.GRID_OFF_WORLD_BBOX).toBe(d.PNANOVDB_GRID_OFF_WORLD_BBOX);
    expect(B.GRID_OFF_VOXEL_SIZE).toBe(d.PNANOVDB_GRID_OFF_VOXEL_SIZE);
    expect(B.GRID_OFF_GRID_CLASS).toBe(d.PNANOVDB_GRID_OFF_GRID_CLASS);
    expect(B.GRID_OFF_GRID_TYPE).toBe(d.PNANOVDB_GRID_OFF_GRID_TYPE);
    expect(B.GRID_OFF_BLIND_METADATA_OFFSET).toBe(d.PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET);
    expect(B.MAP_SIZE).toBe(d.PNANOVDB_MAP_SIZE);
    expect(B.MAP_OFF_MATD).toBe(d.PNANOVDB_MAP_OFF_MATD);
    expect(B.MAP_OFF_VECD).toBe(d.PNANOVDB_MAP_OFF_VECD);
    expect(B.MAP_OFF_TAPERD).toBe(d.PNANOVDB_MAP_OFF_TAPERD);

    expect(B.UPPER_OFF_CHILD_MASK).toBe(d.PNANOVDB_UPPER_OFF_CHILD_MASK);
    expect(B.LOWER_OFF_CHILD_MASK).toBe(d.PNANOVDB_LOWER_OFF_CHILD_MASK);
    expect(B.NODE_OFF_VALUE_MASK).toBe(d.PNANOVDB_UPPER_OFF_VALUE_MASK);
    expect(B.NODE_OFF_FLAGS).toBe(d.PNANOVDB_UPPER_OFF_FLAGS);
    expect(B.LEAF_OFF_VALUE_MASK).toBe(d.PNANOVDB_LEAF_OFF_VALUE_MASK);
    expect(B.LEAF_OFF_BBOX_DIF_AND_FLAGS).toBe(d.PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS);
    expect(B.UPPER_TABLE_COUNT).toBe(d.PNANOVDB_UPPER_TABLE_COUNT);
    expect(B.LOWER_TABLE_COUNT).toBe(d.PNANOVDB_LOWER_TABLE_COUNT);
    expect(B.LEAF_TABLE_COUNT).toBe(d.PNANOVDB_LEAF_TABLE_COUNT);

    const L = B.FLOAT_LAYOUT;
    expect(L.rootSize).toBe(f.root_size);
    expect(L.rootTileSize).toBe(f.root_tile_size);
    expect(L.rootTileOffValue).toBe(f.root_tile_off_value);
    expect(L.rootOffBackground).toBe(f.root_off_background);
    expect(L.rootOffMin).toBe(f.root_off_min);
    expect(L.rootOffStdDev).toBe(f.root_off_stddev);
    expect(L.tableStride).toBe(f.table_stride);
    expect(L.valueStrideBits).toBe(f.value_stride_bits);
    expect(L.upperSize).toBe(f.upper_size);
    expect(L.upperOffTable).toBe(f.upper_off_table);
    expect(L.upperOffMin).toBe(f.upper_off_min);
    expect(L.lowerSize).toBe(f.lower_size);
    expect(L.lowerOffTable).toBe(f.lower_off_table);
    expect(L.leafSize).toBe(f.leaf_size);
    expect(L.leafOffTable).toBe(f.leaf_off_table);
    expect(L.leafOffMin).toBe(f.leaf_off_min);
    expect(L.leafOffStdDev).toBe(f.leaf_off_stddev);

    expect(B.VERSION_PACKED).toBe((32 << 21) | (9 << 10) | 1);
    expect(B.MAGIC_GRID).toBe(BigInt(d.PNANOVDB_MAGIC_GRID as string));
    expect(B.MAGIC_FILE).toBe(BigInt(d.PNANOVDB_MAGIC_FILE as string));
  });
});
