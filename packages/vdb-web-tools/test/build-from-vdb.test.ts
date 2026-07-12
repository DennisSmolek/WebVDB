/**
 * buildFromVdb (T3, Phase 5 wave 2) — parser leaves -> FLOAT NanoVDB image.
 *
 * Oracle: the repo's proven CPU reader (`read-value.ts`). For each of the four
 * real openvdb.org samples we assert every active voxel reported by the parser's
 * `iterLeaves()` reads back exactly on the built image (and is marked active),
 * spot-check that inactive coordinates fall through to background, and confirm
 * metadata (voxel count, index bbox, transform) is consistent with the parser.
 *
 * The Utah teapot (~7M active voxels in a large bbox) is deliberately streamed —
 * `buildFromVdb` feeds the leaf iterator straight into the serializer, never
 * materialising a dense array — and its readback is spot-checked (capped) so CI
 * stays fast.
 */

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { buildFromVdb, buildFromVdbDetailed } from "../src/index.js";
import { parseVdb } from "../src/vdb/index.js";
import type { VdbGrid } from "../src/vdb/types.js";
import { readValue } from "../../nanovdb-wgsl/src/cpu/read-value.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = path.resolve(__dirname, "../../../fixtures/vdb-samples");

const SAMPLES = ["sphere.vdb", "cube.vdb", "smoke.vdb", "utahteapot.vdb"] as const;
const fixturesAvailable = SAMPLES.every((n) => existsSync(path.join(FIXTURES_DIR, n)));

/** Cap the exhaustive readback per grid so the teapot (7M voxels) stays fast. */
const READBACK_CAP = 300_000;

async function loadGrid(name: string): Promise<VdbGrid> {
  const buf = await readFile(path.join(FIXTURES_DIR, name));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return parseVdb(ab).grids[0]!;
}

function leafVoxelCoord(origin: [number, number, number], n: number): [number, number, number] {
  return [origin[0] + ((n >> 6) & 7), origin[1] + ((n >> 3) & 7), origin[2] + (n & 7)];
}

describe.skipIf(!fixturesAvailable)("buildFromVdb — real .vdb corpus round-trips through the CPU reader", () => {
  it.each(SAMPLES)("%s: every active voxel reads back exactly (capped) + active flag matches", async (name) => {
    const grid = await loadGrid(name);
    const image = buildFromVdb(grid);

    let checked = 0;
    let maxErr = 0;
    let activeMismatch = 0;
    outer: for (const leaf of grid.iterLeaves()) {
      for (let n = 0; n < 512; n++) {
        if (((leaf.valueMask[n >>> 5]! >>> (n & 31)) & 1) === 0) continue;
        const ijk = leafVoxelCoord(leaf.origin, n);
        const got = readValue(image, ijk);
        maxErr = Math.max(maxErr, Math.abs(got.value - leaf.values[n]!));
        if (!got.active) activeMismatch++;
        if (++checked >= READBACK_CAP) break outer;
      }
    }

    expect(checked).toBeGreaterThan(0);
    expect(maxErr).toBe(0); // FLOAT is a lossless representation of the parser's f32 values
    expect(activeMismatch).toBe(0);
  });

  it.each(SAMPLES)("%s: metadata (voxel count, bbox, transform) is consistent with the parser", async (name) => {
    const grid = await loadGrid(name);
    const built = buildFromVdbDetailed(grid);

    expect(built.voxelCount).toBe(Number(grid.activeVoxelCount));
    // The built index bbox is the tight active bbox; it must sit inside the
    // parser's reported bbox (which is also the tight active bbox for these grids).
    expect(built.indexBBox).toEqual(grid.indexBBox);

    // The .vdb uniform scale+translate is carried into the NanoVDB Map: the
    // GridData voxelSize (f64 @608) and gridType (@636 == FLOAT/1) reflect it.
    const dv = new DataView(built.image.buffer, built.image.byteOffset, built.image.byteLength);
    const vs = grid.transform.voxelSize[0]!;
    expect(dv.getFloat64(608, true)).toBeCloseTo(vs, 9);
    expect(dv.getUint32(636, true)).toBe(1); // GRID_TYPE_FLOAT
    // Map's f32 diagonal scale + translation match the .vdb transform.
    expect(dv.getFloat32(296, true)).toBeCloseTo(vs, 6); // Map matF[0]
    expect(dv.getFloat32(296 + 72, true)).toBeCloseTo(grid.transform.matrix[3]!, 6); // vecF[0] = tx
  });

  it("sphere.vdb: inactive coordinates fall through to background / inactive", async () => {
    const grid = await loadGrid("sphere.vdb");
    const image = buildFromVdb(grid);
    // A coordinate far outside the index bbox has no leaf -> background, inactive.
    const far: [number, number, number] = [
      grid.indexBBox!.max[0] + 10_000,
      grid.indexBBox!.max[1] + 10_000,
      grid.indexBBox!.max[2] + 10_000,
    ];
    const got = readValue(image, far);
    expect(got.active).toBe(false);
    expect(got.value).toBeCloseTo(grid.background, 6);
  });
});

describe("buildFromVdb — transform validation", () => {
  function fakeGrid(matrix: number[]): VdbGrid {
    return {
      name: "fake",
      gridType: "Tree_float_5_4_3",
      transform: { type: "AffineMap", matrix, voxelSize: [1, 1, 1] },
      metadata: {},
      indexBBox: null,
      activeVoxelCount: 0n,
      background: 0,
      readValue: () => ({ value: 0, active: false }),
      iterLeaves: () => [],
    };
  }

  it("accepts a uniform scale + translate", () => {
    // prettier-ignore
    const m = [2, 0, 0, 5, 0, 2, 0, 6, 0, 0, 2, 7, 0, 0, 0, 1];
    expect(() => buildFromVdb(fakeGrid(m))).not.toThrow();
  });

  it("throws with a GPU-resample pointer on a non-uniform scale", () => {
    // prettier-ignore
    const m = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];
    expect(() => buildFromVdb(fakeGrid(m))).toThrowError(/non-uniform|GPU resampling/i);
  });

  it("throws on a rotation/shear", () => {
    // prettier-ignore
    const m = [2, 0.3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];
    expect(() => buildFromVdb(fakeGrid(m))).toThrowError(/rotated|sheared|GPU resampling/i);
  });
});
