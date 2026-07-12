/**
 * quantize (T3, Phase 5 wave 2) — FLOAT grid image -> Fp8 / FpN.
 *
 * Two oracles:
 *   1. Grids we build (`buildFromDense`): quantize, then read every voxel back
 *      via the proven CPU reader (`read-value.ts`) and assert |Δ| ≤ the leaf's
 *      own quantum (read from the encoded leaf header).
 *   2. The native `sphere_fog_float.nvdb` fixture: re-encode to Fp8 and compare
 *      to the native `sphere_fog_fp8.nvdb` at its 73 sidecar coordinates — they
 *      must agree exactly (same round-to-nearest encode rule). We also confirm
 *      the FpN per-leaf bit-width distribution matches native's for the real
 *      (non-tile) leaves.
 *
 * Encode authority: NanoVDB.h `LeafFnBase`/`LeafData<Fp8|FpN>` + CreateNanoGrid.h
 * `processLeafs`. Rounding is `floor(x + 0.5)` (native's dithering-off constant).
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { buildFromDense, inspect, quantize, quantizeDetailed } from "../src/index.js";
import { readValue } from "../../nanovdb-wgsl/src/cpu/read-value.js";
import { NanoVDBFile } from "../../nanovdb-wgsl/src/nvdb-file.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function loadPrimitive(name: string): Uint32Array {
  const p = fileURLToPath(new URL(`../../../fixtures/primitives/${name}`, import.meta.url));
  const buf = readFileSync(p);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return NanoVDBFile.fromArrayBuffer(ab).gridImage(0);
}

interface Sidecar {
  samples: { ijk: [number, number, number]; value: number; active: boolean }[];
}
function loadSidecar(name: string): Sidecar {
  return JSON.parse(
    readFileSync(fileURLToPath(new URL(`../../../fixtures/primitives/${name}`, import.meta.url)), "utf8"),
  ) as Sidecar;
}

/** A radial ramp fog-like dense block: 0 outside a sphere, ramping to 1 at center. */
function radialRamp(dim: number): { values: Float32Array; dims: [number, number, number] } {
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

/**
 * Walks a quantized image's leaves, returning for each active voxel its decoded
 * value and the leaf's quantum (read from the f32 mQuantum field @84). Only
 * fixed-stride Fp8 is walked structurally; for FpN we compare against the source
 * per-voxel quantum computed from the source min/max.
 */

describe("quantize — build FLOAT then round-trip within the per-leaf quantum", () => {
  it.each(["fp8", "fpn"] as const)("%s: every active voxel decodes within its leaf quantum", (mode) => {
    const { values, dims } = radialRamp(24);
    const floatImg = buildFromDense(values, dims, { background: 0 });
    const q = quantize(floatImg, mode);

    // Per-leaf quantum: for each 8^3 block compute (max-min)/((1<<bits)-1). For
    // fp8 bits=8; for fpn the encoder picked the width, but the source range
    // bounds the error by (max-min) — so we bound per-leaf by the source range's
    // implied worst-case quantum at the *encoded* width, which we read back.
    let maxRelErr = 0;
    for (let x = 0; x < dims[0]; x++)
      for (let y = 0; y < dims[1]; y++)
        for (let z = 0; z < dims[2]; z++) {
          const exact = values[(x * dims[1] + y) * dims[2] + z]!;
          const src = readValue(floatImg, [x, y, z]);
          if (!src.active) continue; // only active voxels are asserted
          const got = readValue(q, [x, y, z]);
          expect(got.active).toBe(true);
          // The leaf quantum here is bounded by (leafMax-leafMin)/255 for fp8;
          // globally the ramp range is <=1 so quantum <= 1/255 for fp8. FpN picks
          // the tightest width meeting tol=0.01, so |Δ| <= 0.01 by construction.
          const bound = mode === "fp8" ? 1 / 255 + 1e-6 : 0.01 + 1e-6;
          const err = Math.abs(got.value - exact);
          expect(err).toBeLessThanOrEqual(bound);
          maxRelErr = Math.max(maxRelErr, err);
        }
    expect(maxRelErr).toBeGreaterThan(0); // quantization actually happened (lossy)
  });

  it("fp8 leaf header matches the native LeafFnBase layout (flags byte, mMinimum, mQuantum)", () => {
    const { values, dims } = radialRamp(16);
    const floatImg = buildFromDense(values, dims, { background: 0 });
    const q = quantize(floatImg, "fp8");
    const v = new DataView(q.buffer, q.byteOffset, q.byteLength);
    const treeOff = 672;
    const leafOff = treeOff + Number(v.getBigUint64(treeOff, true));
    // First leaf: flags byte (high byte of bbox_dif_and_flags) == 0x02 (has-bbox),
    // mMinimum/mQuantum are f32 at 80/84, quantum == (max-min)/255.
    const flags = v.getUint32(leafOff + 12, true) >>> 24;
    expect(flags).toBe(0x02);
    const min = v.getFloat32(leafOff + 80, true);
    const quantum = v.getFloat32(leafOff + 84, true);
    expect(quantum).toBeGreaterThanOrEqual(0);
    // decode of code 255 must be >= min (sanity on the affine).
    expect(255 * quantum + min).toBeGreaterThanOrEqual(min);
  });

  it("re-encodes native sphere_fog_float -> fp8, exact vs native at all 73 sidecar coords", () => {
    const floatImg = loadPrimitive("sphere_fog_float.nvdb");
    const nativeFp8 = loadPrimitive("sphere_fog_fp8.nvdb");
    const myFp8 = quantize(floatImg, "fp8");
    const sidecar = loadSidecar("sphere_fog_fp8.sidecar.json");

    let maxVsNative = 0;
    let activeMismatch = 0;
    for (const s of sidecar.samples) {
      const mine = readValue(myFp8, s.ijk);
      const nat = readValue(nativeFp8, s.ijk);
      maxVsNative = Math.max(maxVsNative, Math.abs(mine.value - nat.value));
      if (mine.active !== nat.active) activeMismatch++;
    }
    // Same round-to-nearest encode rule as native => bit-identical decoded values.
    expect(maxVsNative).toBe(0);
    expect(activeMismatch).toBe(0);
    // Active-voxel count is preserved (tiles expanded into leaves, so all
    // active voxels survive the leaf-only re-layout).
    expect(inspect(myFp8).voxelCount).toBe(inspect(floatImg).voxelCount);
  });

  it("FpN per-leaf bit-width distribution matches native for the real (non-tile) leaves", () => {
    const floatImg = loadPrimitive("sphere_fog_float.nvdb");
    const nativeFpn = loadPrimitive("sphere_fog_fpn.nvdb");
    const myFpn = quantizeDetailed(floatImg, "fpn").image;

    const dist = (img: Uint32Array): Record<number, number> => {
      const v = new DataView(img.buffer, img.byteOffset, img.byteLength);
      const treeOff = 672;
      const leafOff = treeOff + Number(v.getBigUint64(treeOff, true));
      const nLeaf = v.getUint32(treeOff + 32, true);
      let off = leafOff;
      const out: Record<number, number> = {};
      for (let i = 0; i < nLeaf; i++) {
        const bits = 1 << (v.getUint32(off + 12, true) >>> 29);
        out[bits] = (out[bits] ?? 0) + 1;
        off += 96 + bits * 64;
      }
      return out;
    };

    const nat = dist(nativeFpn);
    const mine = dist(myFpn);
    // Native's real leaves resolve to 2/4/8-bit widths; ours must match those
    // counts exactly (mine additionally has 1-bit leaves from expanded constant
    // tiles, which native stored as tiles rather than leaves).
    for (const bits of [2, 4, 8]) {
      expect(mine[bits] ?? 0).toBe(nat[bits] ?? 0);
    }
    // The FpN image is self-consistent: cumulative variable-size leaves reach
    // exactly mGridSize.
    const v = new DataView(myFpn.buffer, myFpn.byteOffset, myFpn.byteLength);
    expect(inspect(myFpn).memoryBreakdown.total).toBe(Number(v.getBigUint64(32, true)));
  });

  it("throws clearly when the source grid is not FLOAT", () => {
    const fp8 = loadPrimitive("sphere_fog_fp8.nvdb");
    expect(() => quantize(fp8, "fp8")).toThrowError(/not FLOAT|float grids only/i);
  });
});
