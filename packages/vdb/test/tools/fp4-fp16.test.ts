/**
 * Fp4 / Fp16 (K5.4b): quantize -> NvdbFile load -> CPU oracle decode.
 *
 * No native Fp4/Fp16 fixtures exist, so the oracle is the Float source: every
 * active voxel must decode within one leaf quantum, `readValue` (f64) and
 * `vdbReadValueF32` (GPU order) must agree to f32 rounding, and the leaf geometry
 * must be NanoVDB's `LeafData<Fp4|Fp16>` (96-byte header + 256 / 1024 code bytes).
 */
import { describe, expect, it } from "vitest";

import {
  vdbBuildFromDense,
  vdbInspect,
  NvdbFile,
  vdbQuantize,
  vdbReadValueCpu,
  vdbReadValueF32,
  vdbSampleNearest,
  vdbSampleTrilinear,
  writeNvdb,
} from "../../src/tools/index.js";

const MODES = [
  { mode: "fp4", type: "Fp4", levels: 15, leafBytes: 352 },
  { mode: "fp16", type: "Fp16", levels: 65535, leafBytes: 1120 },
] as const;

function ramp(dim: number): { values: Float32Array; dims: [number, number, number] } {
  const values = new Float32Array(dim * dim * dim);
  const c = (dim - 1) / 2;
  const r = dim * 0.45;
  for (let x = 0; x < dim; x++)
    for (let y = 0; y < dim; y++)
      for (let z = 0; z < dim; z++) {
        const d = Math.hypot(x - c, y - c, z - c);
        values[(x * dim + y) * dim + z] = d < r ? 1 - d / r : 0;
      }
  return { values, dims: [dim, dim, dim] };
}

describe.each(MODES)("$type grids", ({ mode, type, levels, leafBytes }) => {
  const { values, dims } = ramp(24);
  const floatImg = vdbBuildFromDense(values, dims, { background: 0 });
  const q = vdbQuantize(floatImg, mode);
  // Ramp range is <= 1, so a leaf quantum is <= 1 / levels.
  const bound = 1 / levels + 1e-6;

  it("decodes every active voxel within a quantum of the f32 source", () => {
    let maxErr = 0;
    for (let x = 0; x < dims[0]; x++)
      for (let y = 0; y < dims[1]; y++)
        for (let z = 0; z < dims[2]; z++) {
          const src = vdbReadValueCpu(floatImg, [x, y, z]);
          if (!src.active) continue;
          const got = vdbReadValueCpu(q, [x, y, z]);
          expect(got.active).toBe(true);
          const err = Math.abs(got.value - src.value);
          expect(err).toBeLessThanOrEqual(bound);
          maxErr = Math.max(maxErr, err);
          // GPU-order oracle: same code, f32 arithmetic, within f32 rounding.
          expect(Math.abs(vdbReadValueF32(q, [x, y, z]) - got.value)).toBeLessThan(1e-6);
        }
    if (mode === "fp4") expect(maxErr).toBeGreaterThan(1 / 255); // visibly lossy
  });

  it("uses the Fp leaf geometry", () => {
    const f = vdbInspect(floatImg);
    const r = vdbInspect(q);
    expect(r.leafCount).toBe(f.leafCount);
    const delta = f.totalBytes - r.totalBytes; // float leaf is 2144 bytes
    expect(delta).toBe(f.leafCount * (2144 - leafBytes));
  });

  it("round-trips through writeNvdb and NvdbFile", () => {
    const file = NvdbFile.fromArrayBuffer(writeNvdb([q]));
    expect(file.grids[0]!.gridType).toBe(type);
    const back = file.gridImage(0);
    expect(vdbReadValueCpu(back, [12, 12, 12]).value).toBe(vdbReadValueCpu(q, [12, 12, 12]).value);
  });

  it("samples nearest and trilinear against the source", () => {
    for (const p of [[12, 12, 12], [10.5, 11.25, 12.75], [6.2, 12, 9.9]] as [number, number, number][]) {
      expect(Math.abs(vdbSampleTrilinear(q, p) - vdbSampleTrilinear(floatImg, p))).toBeLessThanOrEqual(bound);
      expect(Math.abs(vdbSampleNearest(q, p) - vdbSampleNearest(floatImg, p))).toBeLessThanOrEqual(bound);
    }
  });
});
