import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ByteReader } from "../src/vdb/byte-reader.js";
import { readCompressedValues } from "../src/vdb/compression.js";
import { VdbFormatError, VdbUnsupportedError } from "../src/vdb/errors.js";
import { parseVdb } from "../src/vdb/index.js";
import type { VdbGrid } from "../src/vdb/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = path.resolve(__dirname, "../../../fixtures/vdb-samples");

/**
 * Expected values below were discovered by parsing the real fixtures once
 * (`node scripts/fetch-vdb-samples.mjs`, then a throwaway probe script) and
 * cross-checked against each grid's own `file_voxel_count` metadata — a
 * ground-truth field OpenVDB itself writes at bake time, independent of
 * this parser's leaf traversal. All four samples' `file_voxel_count`
 * matched our computed `activeVoxelCount` exactly, which is strong
 * independent evidence the topology/mask/compression/half-float decoding
 * here is correct. See the phase handoff for the caveat this does *not*
 * replace: no byte/value parity check against `nanovdb_convert` yet (that
 * anchor needs the native toolchain).
 */
const EXPECTATIONS = {
  "sphere.vdb": {
    name: "ls_sphere",
    gridType: "Tree_float_5_4_3",
    gridClass: "level set",
    activeVoxelCount: 270638n,
    leafCount: 1451,
    background: 0.1500244140625,
    indexBBox: { min: [-62, -62, -62], max: [62, 62, 62] },
    transformType: "UniformScaleMap",
    voxelSize: 0.05000000074505806,
    savedAsHalfFloat: true,
  },
  "cube.vdb": {
    name: "ls_cube",
    gridType: "Tree_float_5_4_3",
    gridClass: "level set",
    activeVoxelCount: 1452218n,
    leafCount: 6812,
    background: 0.1500244140625,
    indexBBox: { min: [-112, -112, -112], max: [112, 112, 112] },
    transformType: "UniformScaleMap",
    voxelSize: 0.05000000074505806,
    savedAsHalfFloat: true,
  },
  "smoke.vdb": {
    name: "density",
    gridType: "Tree_float_5_4_3",
    gridClass: "fog volume",
    activeVoxelCount: 1049275n,
    leafCount: 3117,
    background: 0,
    indexBBox: { min: [1, 2, 1], max: [111, 223, 112] },
    transformType: "UniformScaleTranslateMap",
    voxelSize: 0.4761904761904745,
    savedAsHalfFloat: true,
  },
  "utahteapot.vdb": {
    name: "ls_utahteapot",
    gridType: "Tree_float_5_4_3",
    gridClass: "level set",
    activeVoxelCount: 6960047n,
    leafCount: 35099,
    background: 0.300048828125,
    indexBBox: { min: [-500, -231, -308], max: [480, 231, 308] },
    transformType: "UniformScaleMap",
    voxelSize: 0.10000000149011612,
    savedAsHalfFloat: true,
  },
} as const;

const SAMPLE_NAMES = Object.keys(EXPECTATIONS) as (keyof typeof EXPECTATIONS)[];
const fixturesAvailable = SAMPLE_NAMES.every((name) => existsSync(path.join(FIXTURES_DIR, name)));

function popcount32(x: number): number {
  let v = x >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

async function loadGrid(name: keyof typeof EXPECTATIONS): Promise<VdbGrid> {
  const buf = await readFile(path.join(FIXTURES_DIR, name));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const file = parseVdb(ab);
  expect(file.grids).toHaveLength(1);
  return file.grids[0]!;
}

describe.skipIf(!fixturesAvailable)("parseVdb — real fixture corpus", () => {
  it.each(SAMPLE_NAMES)("%s: grid name/type/class/bbox/background match discovery", async (name) => {
    const exp = EXPECTATIONS[name];
    const grid = await loadGrid(name);

    expect(grid.name).toBe(exp.name);
    expect(grid.gridType).toBe(exp.gridType);
    expect(grid.metadata["class"]).toBe(exp.gridClass);
    expect(grid.metadata["is_saved_as_half_float"]).toBe(exp.savedAsHalfFloat);
    expect(grid.background).toBe(exp.background);
    expect(grid.indexBBox).toEqual(exp.indexBBox);
    expect(grid.transform.type).toBe(exp.transformType);
    expect(grid.transform.voxelSize).toEqual([exp.voxelSize, exp.voxelSize, exp.voxelSize]);
    expect(grid.transform.matrix).toHaveLength(16);
  });

  it.each(SAMPLE_NAMES)("%s: activeVoxelCount matches file_voxel_count metadata and leaf mask sum", async (name) => {
    const exp = EXPECTATIONS[name];
    const grid = await loadGrid(name);

    expect(grid.activeVoxelCount).toBe(exp.activeVoxelCount);
    expect(grid.metadata["file_voxel_count"]).toBe(exp.activeVoxelCount);

    let leafCount = 0;
    let maskSum = 0n;
    for (const leaf of grid.iterLeaves()) {
      leafCount++;
      for (const word of leaf.valueMask) maskSum += BigInt(popcount32(word));
    }
    expect(leafCount).toBe(exp.leafCount);
    expect(maskSum).toBe(exp.activeVoxelCount);
  });

  it.each(SAMPLE_NAMES)("%s: every leaf origin lies within indexBBox", async (name) => {
    const grid = await loadGrid(name);
    const bbox = grid.indexBBox;
    expect(bbox).not.toBeNull();
    const { min, max } = bbox!;
    for (const leaf of grid.iterLeaves()) {
      for (let axis = 0; axis < 3; axis++) {
        const o = leaf.origin[axis]!;
        // A leaf spans 8 voxels from its origin, so the origin itself can
        // sit up to 7 voxels below the grid's tight active-voxel bbox min.
        expect(o).toBeGreaterThanOrEqual(min[axis]! - 7);
        expect(o).toBeLessThanOrEqual(max[axis]!);
      }
    }
  });

  it.each(SAMPLE_NAMES)("%s: readValue at leaf-interior coords matches the leaf's values array", async (name) => {
    const grid = await loadGrid(name);
    const leaves = [...grid.iterLeaves()];
    expect(leaves.length).toBeGreaterThan(0);

    for (let i = 0; i < 20; i++) {
      const leaf = leaves[i % leaves.length]!;
      const offset = (i * 37 + i * i) % 512; // spread across the 8^3 leaf, deterministic
      const x = offset >> 6;
      const rem = offset & 63;
      const y = rem >> 3;
      const z = rem & 7;
      const ijk: [number, number, number] = [
        leaf.origin[0] + x,
        leaf.origin[1] + y,
        leaf.origin[2] + z,
      ];
      const expectedActive = ((leaf.valueMask[offset >>> 5]! >>> (offset & 31)) & 1) !== 0;
      const result = grid.readValue(ijk);
      expect(result.value).toBe(leaf.values[offset]);
      expect(result.active).toBe(expectedActive);
    }
  });

  it("sphere.vdb: level set actives straddle zero (narrow band, not a fog volume)", async () => {
    const grid = await loadGrid("sphere.vdb");
    expect(grid.metadata["class"]).toBe("level set");

    let anyNegative = false;
    let anyPositive = false;
    for (const leaf of grid.iterLeaves()) {
      for (let i = 0; i < 512; i++) {
        if (((leaf.valueMask[i >>> 5]! >>> (i & 31)) & 1) === 0) continue;
        const v = leaf.values[i]!;
        if (v < 0) anyNegative = true;
        if (v > 0) anyPositive = true;
      }
    }
    // Documented finding: all four samples in this corpus parse as level
    // sets with both signs of active value near the narrow band (none of
    // them exercise the fog-volume "(0, +inf) only" branch as a *level
    // set*; smoke.vdb below is genuinely a fog volume and is asserted
    // separately with the other branch).
    expect(anyNegative).toBe(true);
    expect(anyPositive).toBe(true);
  });

  it("smoke.vdb: fog volume actives are strictly positive (no upper clamp to 1 — density can exceed 1)", async () => {
    const grid = await loadGrid("smoke.vdb");
    expect(grid.metadata["class"]).toBe("fog volume");

    let anyNegative = false;
    let min = Infinity;
    let max = -Infinity;
    for (const leaf of grid.iterLeaves()) {
      for (let i = 0; i < 512; i++) {
        if (((leaf.valueMask[i >>> 5]! >>> (i & 31)) & 1) === 0) continue;
        const v = leaf.values[i]!;
        if (v < 0) anyNegative = true;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    expect(anyNegative).toBe(false);
    expect(min).toBeGreaterThan(0);
    // Documented finding: density values go well above 1 (observed max
    // ~5.7) — the brief's "(0,1]" guess doesn't hold for this fixture.
    expect(max).toBeGreaterThan(1);
  });
});

if (!fixturesAvailable) {
  // eslint-disable-next-line no-console
  console.warn(
    `[parse-vdb.test] fixtures/vdb-samples/ is missing samples — run \`pnpm fixtures:vdb-samples\` ` +
      "to exercise the real-fixture suite (skipped for now).",
  );
}

describe("parseVdb — error paths (no fixtures needed)", () => {
  it("rejects a garbage magic number", () => {
    const buf = new ArrayBuffer(16);
    new DataView(buf).setUint32(0, 0xdeadbeef, true);
    expect(() => parseVdb(buf)).toThrow(VdbFormatError);
    expect(() => parseVdb(buf)).toThrow(/magic/i);
  });

  it("rejects a truncated buffer (valid magic, nothing else)", () => {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setUint32(0, 0x56444220, true);
    expect(() => parseVdb(buf)).toThrow(VdbFormatError);
  });

  it("rejects an empty buffer", () => {
    expect(() => parseVdb(new ArrayBuffer(0))).toThrow(VdbFormatError);
  });

  it("rejects blosc-compressed node data with a clear, actionable error", () => {
    // Exercise the compressed-value reader directly: a minimal node-value
    // payload flagged as blosc-compressed with a real (positive) byte
    // count, which this parser deliberately does not implement.
    const bytes = new Uint8Array(1 + 8 + 4);
    bytes[0] = 0; // NodeMetaData.NoMaskOrInactiveVals
    new DataView(bytes.buffer).setBigInt64(1, 4n, true); // numCompressedBytes = 4 (> 0)
    const reader = new ByteReader(bytes.buffer);

    expect(() =>
      readCompressedValues(reader, {
        fileVersion: 222,
        compression: { zip: false, activeMask: false, blosc: true },
        useHalf: false,
        background: 0,
        numValues: 8,
        valueMask: new Uint32Array(1),
      }),
    ).toThrow(VdbUnsupportedError);
    expect(() =>
      readCompressedValues(new ByteReader(bytes.buffer), {
        fileVersion: 222,
        compression: { zip: false, activeMask: false, blosc: true },
        useHalf: false,
        background: 0,
        numValues: 8,
        valueMask: new Uint32Array(1),
      }),
    ).toThrow(/blosc/i);
  });
});
