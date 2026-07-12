/**
 * inspect + transform (T3, Phase 5 wave 2).
 *
 * `inspect` is checked against native `nanovdb_convert` fixtures: grid type/class,
 * active voxel count and per-level node counts come from each fixture's sidecar;
 * the memory breakdown must sum to the on-disk `mGridSize`.
 *
 * `transform` is a metadata-only Map edit — it rewrites GridData's transform on a
 * copy and leaves the tree/leaf region byte-for-byte identical. We assert the
 * loader reads back the new voxel size/translation, the voxel region is
 * unchanged, and rotation/shear throws.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { buildFromDense, inspect, transform } from "../src/index.js";
import { readValue } from "../../nanovdb-wgsl/src/cpu/read-value.js";
import { NanoVDBFile } from "../../nanovdb-wgsl/src/nvdb-file.js";

function loadPrimitive(name: string): Uint32Array {
  const p = fileURLToPath(new URL(`../../../fixtures/primitives/${name}`, import.meta.url));
  const buf = readFileSync(p);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return NanoVDBFile.fromArrayBuffer(ab).gridImage(0);
}

interface Sidecar {
  grid: {
    type: string;
    class: string;
    gridByteSize: number;
    activeVoxelCount: number;
    nodeCounts: { leaf: number; lower: number; upper: number };
  };
}
function loadSidecar(name: string): Sidecar {
  return JSON.parse(
    readFileSync(fileURLToPath(new URL(`../../../fixtures/primitives/${name}`, import.meta.url)), "utf8"),
  ) as Sidecar;
}

const FIXTURES = [
  { nvdb: "box_fog_float.nvdb", sidecar: "box_fog_float.sidecar.json", type: "Float" },
  { nvdb: "box_fog_fp8.nvdb", sidecar: "box_fog_fp8.sidecar.json", type: "Fp8" },
  { nvdb: "box_fog_fpn.nvdb", sidecar: "box_fog_fpn.sidecar.json", type: "FpN" },
  { nvdb: "sphere_fog_float.nvdb", sidecar: "sphere_fog_float.sidecar.json", type: "Float" },
  { nvdb: "sphere_fog_fp8.nvdb", sidecar: "sphere_fog_fp8.sidecar.json", type: "Fp8" },
] as const;

describe("inspect — against native fixtures", () => {
  it.each(FIXTURES)("$nvdb: type/class/voxelCount/nodeCounts + memory sums to mGridSize", (f) => {
    const image = loadPrimitive(f.nvdb);
    const sc = loadSidecar(f.sidecar);
    const rep = inspect(image);

    expect(rep.gridType).toBe(f.type);
    expect(rep.gridClass).toBe(sc.grid.class);
    expect(rep.voxelCount).toBe(sc.grid.activeVoxelCount);
    expect(rep.nodeCounts).toEqual({
      upper: sc.grid.nodeCounts.upper,
      lower: sc.grid.nodeCounts.lower,
      leaf: sc.grid.nodeCounts.leaf,
    });

    const mb = rep.memoryBreakdown;
    expect(mb.total).toBe(sc.grid.gridByteSize);
    const summed = mb.gridData! + mb.tree! + mb.root! + mb.upper! + mb.lower! + mb.leaf!;
    expect(summed).toBe(mb.total);
  });

  it("box_fog: concrete per-leaf memory anchor (float 336x2144, fp8 336x608)", () => {
    // A concrete anchor (the T3 brief's "788-leaf sphere numbers" analogue):
    // box_fog_float has 336 leaves at 2144 B each = 720384 B of leaf data.
    const rep = inspect(loadPrimitive("box_fog_float.nvdb"));
    expect(rep.nodeCounts.leaf).toBe(336);
    expect(rep.memoryBreakdown.leaf).toBe(336 * 2144);
    // Its fp8 counterpart: same 336 leaves at 608 B = 204288 B.
    const fp8 = inspect(loadPrimitive("box_fog_fp8.nvdb"));
    expect(fp8.nodeCounts.leaf).toBe(336);
    expect(fp8.memoryBreakdown.leaf).toBe(336 * 608);
  });
});

describe("transform — metadata-only affine Map edit", () => {
  function sampleGrid(): { image: Uint32Array; dims: [number, number, number]; values: Float32Array } {
    const dims: [number, number, number] = [16, 16, 16];
    const values = new Float32Array(16 * 16 * 16);
    for (let x = 0; x < 16; x++)
      for (let y = 0; y < 16; y++)
        for (let z = 0; z < 16; z++) {
          const d = Math.hypot(x - 8, y - 8, z - 8);
          values[(x * 16 + y) * 16 + z] = d < 7 ? 1 - d / 7 : 0;
        }
    return { image: buildFromDense(values, dims, { voxelSize: 1, background: 0 }), dims, values };
  }

  it("rewrites voxelSize + Map; the loader reflects the new transform", () => {
    const { image } = sampleGrid();
    const out = transform(image, { voxelSize: 0.25, worldOrigin: [3, 4, 5] });

    const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
    expect(dv.getFloat64(608, true)).toBeCloseTo(0.25, 9); // voxelSize
    expect(dv.getFloat32(296, true)).toBeCloseTo(0.25, 6); // Map matF[0]
    expect(dv.getFloat32(296 + 36, true)).toBeCloseTo(4, 6); // invMatF[0] = 1/0.25
    expect(dv.getFloat32(296 + 72, true)).toBeCloseTo(3, 6); // vecF[0]
    expect(dv.getFloat64(296 + 232, true)).toBeCloseTo(3, 9); // vecD[0]
  });

  it("leaves every voxel value untouched (tree/leaf region byte-identical)", () => {
    const { image, dims } = sampleGrid();
    const out = transform(image, { voxelSize: 10, worldOrigin: [100, 0, 0] });

    // Tree region (everything after GridData's 672 bytes) is byte-for-byte equal.
    const a = new Uint8Array(image.buffer, image.byteOffset + 672, image.byteLength - 672);
    const b = new Uint8Array(out.buffer, out.byteOffset + 672, out.byteLength - 672);
    expect(Buffer.compare(Buffer.from(a), Buffer.from(b))).toBe(0);

    // And values read back identically (index-space lookups are transform-invariant).
    for (let x = 0; x < dims[0]; x++)
      for (let y = 0; y < dims[1]; y++)
        for (let z = 0; z < dims[2]; z++) {
          expect(readValue(out, [x, y, z]).value).toBe(readValue(image, [x, y, z]).value);
        }
    // Source image is not mutated.
    expect(readValue(image, [8, 8, 8]).value).toBeGreaterThan(0);
  });

  it("accepts a 4x4 uniform-scale matrix and updates the world bbox", () => {
    const { image } = sampleGrid();
    // prettier-ignore
    const m = new Float32Array([2, 0, 0, 1, 0, 2, 0, 2, 0, 0, 2, 3, 0, 0, 0, 1]);
    const out = transform(image, m);
    const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
    expect(dv.getFloat64(608, true)).toBeCloseTo(2, 9);
    // world bbox min = indexMin*2 + t; read index bbox from root for the check.
    const rootOff = 672 + Number(dv.getBigUint64(672 + 24, true));
    const iminX = dv.getInt32(rootOff + 0, true);
    expect(dv.getFloat64(560, true)).toBeCloseTo(iminX * 2 + 1, 6);
  });

  it("throws with a GPU-resample pointer on rotation/shear or non-uniform scale", () => {
    const { image } = sampleGrid();
    // prettier-ignore
    const rot = new Float32Array([2, 0.5, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
    expect(() => transform(image, rot)).toThrowError(/rotated|sheared|GPU resampling/i);
    // prettier-ignore
    const aniso = new Float32Array([2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
    expect(() => transform(image, aniso)).toThrowError(/non-uniform|GPU resampling/i);
  });
});
