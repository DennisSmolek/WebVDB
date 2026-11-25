/**
 * K6.R: the random numbers the reference renderer's GPU kernel and its CPU twin share. A 32-bit LCG with the
 * PCG-RXS-M-XS output permutation (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", `pcg`), in u32
 * arithmetic only, so the WGSL (`vdbRefRng*` in `core/reference/wgsl.ts`) and this file produce the same bits.
 * Floats take the top 24 bits (`x >>> 8` × 2⁻²⁴): exact in f32, in [0, 1).
 *
 * Seeding: one stream per (seed, sample index, pixel): `state = pcg(pixel + pcg(seed ^ pcg(sampleIndex)))`.
 */

/** The PCG hash of a u32 (`pcg` of Jarzynski & Olano 2020). */
export function vdbRefPcg(v: number): number {
  const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

/** A random stream; `next()` returns a float in [0, 1) with 24 random bits (exact in f32). */
export interface VDBRefRng {
  /** The LCG state (u32). */
  state: number;
  /** Draws so far (for tests). */
  draws: number;
  next(): number;
}

/** The stream of pixel `pixel` (row-major index), sample `sampleIndex` (batch × spp + s) and `seed`. */
export function vdbRefRng(seed: number, sampleIndex: number, pixel: number): VDBRefRng {
  const rng: VDBRefRng = {
    state: vdbRefPcg((pixel + vdbRefPcg((seed ^ vdbRefPcg(sampleIndex >>> 0)) >>> 0)) >>> 0),
    draws: 0,
    next() {
      this.state = (Math.imul(this.state, 747796405) + 2891336453) >>> 0;
      const s = this.state;
      const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
      this.draws++;
      return (((w >>> 22) ^ w) >>> 8) * 5.9604644775390625e-8;
    },
  };
  return rng;
}
