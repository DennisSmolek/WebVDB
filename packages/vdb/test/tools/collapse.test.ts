/**
 * K4.6: uniform regions collapse into tiles in our own builds (OpenVDB
 * `tools::prune`, tolerance 0).
 *
 * Oracles: the CPU reader (`readValue`, `sampleTrilinear`) on the image built
 * without collapse; the K4.1 tile path (a collapsed build must be the same
 * bytes as the same tree given as explicit tiles); NanoVDB GridStats tile
 * weights computed by hand; and SHA-256 digests of re-encoded native fixtures
 * taken before K4.6 (collapse off must not move a byte).
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  NanoVDBFile,
  bakeCoarseFirst,
  buildFromDenseDetailed,
  buildFromVdb,
  collapseGrid,
  nodeStats,
  parseVdb,
  quantize,
  resampleGrid,
  treeNodes,
} from "../../src/tools/index.js";
import { readValue, readValueVec3 } from "../../src/tools/cpu/read-value.js";
import { sampleTrilinear } from "../../src/tools/cpu/sample-trilinear.js";
import { FLOAT_LEAF_CODEC, VEC3F_LEAF_CODEC } from "../../src/tools/nanovdb/leaf-codec.js";
import { buildFromLeavesDetailed } from "../../src/tools/nanovdb/serialize.js";
import type { LeafInput, TileInput } from "../../src/tools/nanovdb/tree.js";

type Coord = [number, number, number];
const here = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(here, "../../../../fixtures");

function leaf(origin: Coord, fill: (n: number) => number, active: (n: number) => boolean = () => true): LeafInput {
  const values = new Float32Array(512);
  const valueMask = new Uint32Array(16);
  for (let n = 0; n < 512; n++) {
    values[n] = fill(n);
    if (active(n)) valueMask[n >>> 5]! |= 1 << (n & 31);
  }
  return { origin, values, valueMask };
}

const bytesOf = (u32: Uint32Array): Uint8Array => new Uint8Array(u32.buffer, u32.byteOffset, u32.byteLength);
const sha = (u32: Uint32Array): string => createHash("sha256").update(bytesOf(u32)).digest("hex");
const sameBytes = (a: Uint32Array, b: Uint32Array): boolean =>
  a.length === b.length && a.every((w, i) => w === b[i]);

/** Every voxel of `box` reads the same value and state in both images. */
function expectSameValues(a: Uint32Array, b: Uint32Array, min: Coord, max: Coord, step = 1): number {
  let n = 0;
  for (let x = min[0]; x <= max[0]; x += step) {
    for (let y = min[1]; y <= max[1]; y += step) {
      for (let z = min[2]; z <= max[2]; z += step) {
        const ra = readValue(a, [x, y, z]);
        const rb = readValue(b, [x, y, z]);
        if (ra.value !== rb.value || ra.active !== rb.active) {
          throw new Error(`[${x}, ${y}, ${z}]: ${ra.value}/${ra.active} vs ${rb.value}/${rb.active}`);
        }
        n++;
      }
    }
  }
  return n;
}

function rootStats(image: Uint32Array): ReturnType<typeof nodeStats> {
  const root = [...treeNodes(image)].find((n) => n.level === 3)!;
  return nodeStats(image, 3, root.address);
}

/** A dense field with a uniform 128^3 lower-node block, a uniform leaf, a varying region and a lone voxel. */
function denseField(): { values: Float32Array; dims: Coord } {
  const dims: Coord = [144, 136, 128];
  const values = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let x = 0; x < dims[0]; x++) {
    for (let y = 0; y < dims[1]; y++) {
      for (let z = 0; z < dims[2]; z++) {
        let v = 0;
        if (x < 128 && y < 128) v = 2; // a whole lower node: collapses to an upper tile
        else if (x >= 136 && y < 8 && z < 8) v = 0.5; // one uniform leaf: a lower tile
        else if (x >= 128 && x < 136 && y < 16) v = Math.fround(0.01 * (1 + ((x * 7 + y * 3 + z) % 13))); // varying leaves
        values[(x * dims[1] + y) * dims[2] + z] = v;
      }
    }
  }
  values[(140 * dims[1] + 130) * dims[2] + 3] = 10; // a lone voxel in a leaf of its own
  return { values, dims };
}

describe("collapse: buildFromDense", () => {
  const { values, dims } = denseField();
  const off = buildFromDenseDetailed(values, dims, { collapse: false });
  const on = buildFromDenseDetailed(values, dims); // default on

  it("turns the uniform lower node into an upper tile and the uniform leaf into a lower tile", () => {
    expect(off.tileCounts).toEqual({ lower: 0, upper: 0, root: 0 });
    // 4096 leaves in the block, 1 uniform, 32 varying, 1 lone; lower nodes at x 0 and 128 (y 0 and 128).
    expect(off.nodeCounts).toEqual({ leaf: 4096 + 1 + 32 + 1, lower: 3, upper: 1 });
    expect(on.tileCounts).toEqual({ lower: 1, upper: 1, root: 0 });
    expect(on.nodeCounts).toEqual({ leaf: 33, lower: 2, upper: 1 });
    // 4097 leaves of 2144 bytes and one lower node of 33,856 bytes are gone.
    expect(off.image.byteLength - on.image.byteLength).toBe(4097 * 2144 + 33856);
  });

  it("every voxel reads the same value and state (readValue, sampleTrilinear)", () => {
    expect(expectSameValues(off.image, on.image, [-2, -2, -2], [145, 137, 129], 1)).toBeGreaterThan(2.5e6);
    for (let k = 0; k < 2000; k++) {
      const p: Coord = [((k * 37) % 1460) / 10 - 1, ((k * 53) % 1380) / 10 - 1, ((k * 71) % 1300) / 10 - 1];
      expect(sampleTrilinear(on.image, p)).toBe(sampleTrilinear(off.image, p));
    }
  });

  it("counts, bbox, min and max are unchanged; ave and stddev follow GridStats tile weights", () => {
    expect(on.voxelCount).toBe(off.voxelCount);
    expect(on.indexBBox).toEqual(off.indexBBox);
    expect(on.worldBBox).toEqual(off.worldBBox);
    const a = rootStats(off.image);
    const b = rootStats(on.image);
    expect(b.min).toBe(a.min);
    expect(b.max).toBe(a.max);
    // GridStats: an upper tile weighs 4096 (a lower node's slots), a lower tile 512, a voxel 1.
    const varying: number[] = [];
    for (let x = 128; x < 136; x++) {
      for (let y = 0; y < 16; y++) for (let z = 0; z < 128; z++) varying.push(Math.fround(0.01 * (1 + ((x * 7 + y * 3 + z) % 13))));
    }
    let sum = 4096 * 2 + 512 * 0.5 + 10;
    let sumSq = 4096 * 4 + 512 * 0.25 + 100;
    let count = 4096 + 512 + 1;
    for (const v of varying) {
      sum += v;
      sumSq += v * v;
      count++;
    }
    const ave = sum / count;
    expect(b.ave).toBeCloseTo(ave, 5);
    expect(b.stddev).toBeCloseTo(Math.sqrt(Math.max(0, sumSq / count - ave * ave)), 4);
    // Without collapse the block weighs its 2M voxels, so the averages differ, as they would natively.
    expect(Math.abs(a.ave - b.ave)).toBeGreaterThan(0.1);
  });

  it("is the same image as the same tree given as explicit tiles (the K4.1 path)", () => {
    const leaves: LeafInput[] = [];
    // Rebuild the remaining leaves from the uncollapsed image.
    for (const node of treeNodes(off.image)) {
      if (node.level !== 0) continue;
      const [ox, oy, oz] = node.origin;
      const uniformBlock = ox < 128 && oy < 128;
      const uniformLeaf = ox === 136 && oy === 0 && oz === 0;
      if (uniformBlock || uniformLeaf) continue;
      leaves.push(leaf([ox, oy, oz], (n) => readValue(off.image, [ox + (n >> 6), oy + ((n >> 3) & 7), oz + (n & 7)]).value,
        (n) => readValue(off.image, [ox + (n >> 6), oy + ((n >> 3) & 7), oz + (n & 7)]).active));
    }
    const tiles: TileInput[] = [
      { origin: [0, 0, 0], dim: 128, value: 2 },
      { origin: [136, 0, 0], dim: 8, value: 0.5 },
    ];
    const explicit = buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC, { tiles, gridName: "dense" });
    expect(sameBytes(explicit.image, on.image)).toBe(true);
  });
});

describe("collapse: rules (OpenVDB isConstant at tolerance 0)", () => {
  it("collapses all-inactive leaves to inactive tiles, drops background ones, keeps mixed-state leaves", () => {
    const leaves = [
      leaf([0, 0, 0], () => 1), // active uniform -> lower tile
      leaf([8, 0, 0], () => 5, () => false), // inactive uniform, not background -> inactive lower tile
      leaf([16, 0, 0], () => 0, () => false), // inactive background -> nothing
      leaf([24, 0, 0], () => 1, (n) => n !== 7), // mixed state -> stays a leaf
      leaf([32, 0, 0], (n) => (n === 3 ? 1.5 : 1)), // one value differs -> stays a leaf
      leaf([40, 0, 0], (n) => (n % 2 ? -0 : 0)), // -0 equals 0 -> lower tile (value of voxel 0)
      leaf([48, 0, 0], () => Number.NaN), // NaN never equals -> stays a leaf
    ];
    const off = buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC);
    const on = buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC, { collapse: true });
    expect(off.nodeCounts.leaf).toBe(7);
    expect(on.nodeCounts.leaf).toBe(3);
    expect(on.tileCounts.lower).toBe(2);
    expect(on.voxelCount).toBe(off.voxelCount);
    expect(readValue(on.image, [9, 1, 2])).toEqual({ value: 5, active: false });
    expect(readValue(on.image, [17, 1, 2])).toEqual({ value: 0, active: false });
    expect(Object.is(readValue(on.image, [41, 0, 0]).value, 0)).toBe(true);
    for (const x of [0, 8, 16, 24, 32, 40]) expectSameValues(off.image, on.image, [x, 0, 0], [x + 7, 7, 7]);
    expect(Number.isNaN(readValue(on.image, [50, 1, 1]).value)).toBe(true);
    // Default for buildFromLeaves is off.
    expect(sameBytes(buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC, { collapse: false }).image, off.image)).toBe(true);
  });

  it("a full upper node of equal tiles becomes a root tile; of inactive background, it disappears", () => {
    const upperTiles: TileInput[] = [];
    for (let i = 0; i < 32; i++) for (let j = 0; j < 32; j++) for (let k = 0; k < 32; k++) {
      upperTiles.push({ origin: [i * 128, j * 128, k * 128], dim: 128, value: 3 });
    }
    const lone = leaf([-8, 0, 0], (n) => n / 512);
    const off = buildFromLeavesDetailed([lone], FLOAT_LEAF_CODEC, { tiles: upperTiles });
    const on = buildFromLeavesDetailed([lone], FLOAT_LEAF_CODEC, { tiles: upperTiles, collapse: true });
    expect(off.tileCounts).toEqual({ lower: 0, upper: 32768, root: 0 });
    expect(on.tileCounts).toEqual({ lower: 0, upper: 0, root: 1 });
    expect(on.nodeCounts).toEqual({ leaf: 1, lower: 1, upper: 1 });
    expect(on.voxelCount).toBe(off.voxelCount);
    expect(off.voxelCount).toBe(4096 ** 3 + 512);
    for (const p of [[0, 0, 0], [4095, 4095, 4095], [1000, 2000, 3000], [-8, 0, 0], [-1, 7, 7], [4096, 0, 0]] as Coord[]) {
      expect(readValue(on.image, p)).toEqual(readValue(off.image, p));
    }
    // Root stats: a root tile weighs 32768 (an upper node's slots), the same as 32768 upper tiles of 4096 do not.
    const s = rootStats(on.image);
    const ave = (32768 * 3 + [...Array(512).keys()].reduce((acc, n) => acc + Math.fround(n / 512), 0)) / (32768 + 512);
    expect(s.ave).toBeCloseTo(ave, 5);

    // Inactive background tiles all over an upper node: erased, as RootNode::eraseBackgroundTiles does.
    const bgTiles = upperTiles.map((t) => ({ ...t, value: 0, active: false }));
    const erased = buildFromLeavesDetailed([lone], FLOAT_LEAF_CODEC, { tiles: bgTiles, collapse: true });
    expect(erased.nodeCounts).toEqual({ leaf: 1, lower: 1, upper: 1 });
    expect(erased.image.byteLength).toBe(buildFromLeavesDetailed([lone], FLOAT_LEAF_CODEC).image.byteLength);
  });

  it("Vec3f leaves collapse componentwise", () => {
    const make = (origin: Coord, v: (n: number) => [number, number, number]): LeafInput => {
      const values = new Float32Array(512 * 3);
      for (let n = 0; n < 512; n++) values.set(v(n), n * 3);
      return { origin, values, valueMask: new Uint32Array(16).fill(0xffffffff) };
    };
    const leaves = [make([0, 0, 0], () => [1, 2, 3]), make([8, 0, 0], (n) => [1, 2, n === 9 ? 4 : 3])];
    const on = buildFromLeavesDetailed(leaves, VEC3F_LEAF_CODEC, { collapse: true });
    expect(on.nodeCounts.leaf).toBe(1);
    expect(on.tileCounts.lower).toBe(1);
    expect(readValueVec3(on.image, [3, 4, 5]).value).toEqual([1, 2, 3]);
    expect(readValueVec3(on.image, [8, 1, 1]).value).toEqual([1, 2, 4]); // offset 9 = (0, 1, 1)
  });
});

describe("collapse: resample, quantize, coarse-first", () => {
  // A 128^3 uniform block at the origin plus a soft shell: resampling by 1 keeps whole uniform leaves.
  const { values, dims } = denseField();
  const src = buildFromDenseDetailed(values, dims, { collapse: false }).image;

  it("resampleGrid collapses by default and reads back the same values", () => {
    const on = resampleGrid(src, { filter: "point" });
    const off = resampleGrid(src, { filter: "point", collapse: false });
    expect(on.tileCounts.upper + on.tileCounts.lower).toBeGreaterThan(0);
    expect(off.tileCounts).toEqual({ lower: 0, upper: 0, root: 0 });
    expect(on.image.byteLength).toBeLessThan(off.image.byteLength);
    expect(on.voxelCount).toBe(off.voxelCount);
    expectSameValues(off.image, on.image, [-1, -1, -1], [dims[0], dims[1], dims[2]], 3);
    // Quantized output keeps the tiles (quantize keeps source tiles since K4.1).
    const q = resampleGrid(src, { filter: "point", quantize: "fp8" });
    expect(q.tileCounts).toEqual(on.tileCounts);
  });

  it("quantize keeps the source topology unless asked; collapseGrid then quantize is the same", () => {
    const keep = quantize(src, "fp8");
    expect(sameBytes(keep, quantize(src, "fp8", undefined, { collapse: false }))).toBe(true);
    const collapsed = quantize(src, "fp8", undefined, { collapse: true });
    expect(collapsed.byteLength).toBeLessThan(keep.byteLength);
    expect(sameBytes(collapsed, quantize(collapseGrid(src).image, "fp8"))).toBe(true);
    expect(sameBytes(collapseGrid(src).image, buildFromDenseDetailed(values, dims).image)).toBe(true);
  });

  it("bakeCoarseFirst: previews collapse by default, the full grid does not", () => {
    const file = NanoVDBFile.fromArrayBuffer(bakeCoarseFirst(src, { previewResolution: 144, quantize: "none", codec: "none" }).buffer as ArrayBuffer);
    expect(file.grids.length).toBe(2);
    const preview = file.gridImage(0);
    const full = file.gridImage(1);
    expect(sameBytes(full, src)).toBe(true);
    const tilesIn = (img: Uint32Array): number => [...treeNodes(img)].reduce((s, n) => s + n.activeTiles.length, 0);
    expect(tilesIn(preview)).toBeGreaterThan(0);
    const both = NanoVDBFile.fromArrayBuffer(bakeCoarseFirst(src, { previewResolution: 144, quantize: "none", codec: "none", collapse: true }).buffer as ArrayBuffer);
    expect(tilesIn(both.gridImage(1))).toBeGreaterThan(0);
    const neither = NanoVDBFile.fromArrayBuffer(bakeCoarseFirst(src, { previewResolution: 144, quantize: "none", codec: "none", collapse: false }).buffer as ArrayBuffer);
    expect(tilesIn(neither.gridImage(0))).toBe(0);
  });
});

// Digests of the re-encoded native fixtures, taken on v2 at 17fb52f (before K4.6).
const PRE_K46 = {
  "sphere_fog_float->fp8": "045b1f2efe09be7dfc1f11d4e33ec952daf44bfba74b28b70f09bf0890ce1172",
  "sphere_fog_float->fpn": "5ffb0c4145c509cbe3aed7e498bec8a43fd99764ecef8f7a26543bfa8809b5bb",
  "box_fog_float->fp8": "c44fb44520822a08430af92b171cb17206f09e20781dc096bee05ebb69601b82",
  "box_fog_float->fpn": "b366e3f359b61091a8a7190a9bd1a4f7542cb6c88c2d542f7d852f484beb5f81",
  "torus_fog_float->fp8": "6b512b363e4f5889176920b69eb261eb48f43e716cbbf3eb9b3d3030f7936003",
  "torus_fog_float->fpn": "48b7c9a931204c98b5d5ac7a02470e5b71f21012e816dbb357f617163ba2e1ed",
  "smoke.vdb->float": "0361ad203332473682831373667346243eac3e84e9a15d7a2b04dd2232a39612",
  "sphere.vdb->float": "a51c43ba43c01ed7851eea882de4268a97668b5bd00c7658ecaedfb69d931628",
  "utahteapot.vdb->float": "9f5f59556f3e97e3956993b3edd5690f5ddca1c8c8536a997a89acc863582330",
  "cube.vdb->float": "8f1190444762ce807db2c6967fd252d0af636ae1de467e9c5103dc20ea5d559a",
} as const;

const primitive = (n: string): string => path.join(FIXTURES, "primitives", `${n}.nvdb`);
const sample = (n: string): string => path.join(FIXTURES, "vdb-samples", `${n}.vdb`);
const loadNvdb = (p: string): Uint32Array => {
  const b = readFileSync(p);
  return NanoVDBFile.fromArrayBuffer(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer).gridImage(0);
};

describe("collapse off: re-encoded native inputs are byte-identical to before K4.6", () => {
  for (const s of ["sphere", "box", "torus"]) {
    it.skipIf(!existsSync(primitive(`${s}_fog_float`)))(`${s}_fog_float → fp8 / fpn`, () => {
      const f = loadNvdb(primitive(`${s}_fog_float`));
      for (const m of ["fp8", "fpn"] as const) {
        const key = `${s}_fog_float->${m}` as keyof typeof PRE_K46;
        expect(sha(quantize(f, m))).toBe(PRE_K46[key]);
        expect(sha(quantize(f, m, undefined, { collapse: false }))).toBe(PRE_K46[key]);
      }
      // Native primitives are already pruned: collapsing finds nothing more.
      const c = collapseGrid(f);
      expect(sameBytes(c.image, collapseGrid(c.image).image)).toBe(true);
      expect(c.nodeCounts.leaf).toBe([...treeNodes(f)].filter((n) => n.level === 0).length);
    });
  }
  for (const n of ["smoke", "sphere", "utahteapot", "cube"]) {
    it.skipIf(!existsSync(sample(n)))(`${n}.vdb → buildFromVdb`, () => {
      const b = readFileSync(sample(n));
      const grid = parseVdb(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer).grids.find(
        (g) => g.valueType === "float",
      )!;
      const key = `${n}.vdb->float` as keyof typeof PRE_K46;
      expect(sha(buildFromVdb(grid))).toBe(PRE_K46[key]);
      expect(sha(buildFromVdb(grid, { collapse: false }))).toBe(PRE_K46[key]);
    });
  }
});
