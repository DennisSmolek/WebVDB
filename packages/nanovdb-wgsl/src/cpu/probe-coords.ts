/**
 * Deterministic pseudo-random probe-coordinate generation for the Phase 2 GPU
 * test harness (docs/PLAN.md). The whole point of this module is
 * cross-language reproducibility: the same `seed` must produce the exact
 * same coordinate sequence here, in a future WGSL/compute-shader host, and
 * in any other language that implements the same generator — so both the
 * PRNG algorithm and the exact order operations are drawn in are pinned and
 * documented below.
 *
 * ---------------------------------------------------------------------------
 * PRNG: splitmix64
 * ---------------------------------------------------------------------------
 * Bit-for-bit the same algorithm as `Rng` in
 * `docker/fixture-bake/bake_primitives.cpp` (the generator used to bake the
 * sidecar ground-truth samples), so a port to any other host only has to
 * match one small, well-known mixing function:
 *
 *   state += 0x9E3779B97F4A7C15
 *   z = state
 *   z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
 *   z = (z ^ (z >> 27)) * 0x94D049BB133111EB
 *   z = z ^ (z >> 31)
 *   return z   // the 64-bit output; `state` persists across calls
 *
 * All arithmetic is unsigned 64-bit wraparound (masked to 64 bits after
 * every add/multiply/xor here, since JS `bigint` doesn't wrap on its own).
 * The initial `state` is exactly the `seed` passed in — there is no extra
 * mixing of the seed before the first `next()` call (matching
 * `Rng(uint64_t seed) : state(seed) {}`).
 *
 * `range(lo, hi)` (inclusive) matches `Rng::range`:
 *   lo + int32(next() % uint64(hi - lo + 1))
 *
 * ---------------------------------------------------------------------------
 * Generation order (new to this module — not baked anywhere upstream, so
 * pin it precisely here):
 * ---------------------------------------------------------------------------
 * `probeCoords`: for each of `count` points, draw exactly one `next()` per
 * axis, in x, y, z order, each via `range(axisMin - dilate, axisMax + dilate)`.
 * That's 3 `next()` calls per point, in axis order x, y, z.
 *
 * `probePoints`: for each of `count` points, per axis in x, y, z order draw
 * TWO `next()` calls back to back: one for the integer coord (`range`, same
 * bounds as `probeCoords`) and immediately after it one for the fractional
 * part (`next() % 4`, mapped to `{0, 0.25, 0.5, 0.75}`). So the per-point call
 * order is: ix, fx, iy, fy, iz, fz — 6 `next()` calls per point.
 */

const SPLITMIX64_INC = 0x9e3779b97f4a7c15n;
const SPLITMIX64_MUL1 = 0xbf58476d1ce4e5b9n;
const SPLITMIX64_MUL2 = 0x94d049bb133111ebn;
const MASK64 = (1n << 64n) - 1n;

class SplitMix64 {
  private state: bigint;

  constructor(seed: bigint) {
    this.state = seed & MASK64;
  }

  /** One splitmix64 step; returns the next 64-bit unsigned output. */
  next(): bigint {
    this.state = (this.state + SPLITMIX64_INC) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * SPLITMIX64_MUL1) & MASK64;
    z = ((z ^ (z >> 27n)) * SPLITMIX64_MUL2) & MASK64;
    z = z ^ (z >> 31n);
    return z;
  }

  /** Inclusive integer range, matching `Rng::range` in bake_primitives.cpp. */
  range(lo: number, hi: number): number {
    const span = BigInt(hi - lo + 1);
    const r = this.next() % span;
    return lo + Number(r);
  }
}

const DEFAULT_DILATE = 4;

export interface ProbeOpts {
  /** splitmix64 initial state. */
  seed: bigint;
  count: number;
  bboxMin: [number, number, number];
  bboxMax: [number, number, number];
  /** Voxels beyond bboxMin/bboxMax the probe range is dilated by. Default 4. */
  dilate?: number;
}

/**
 * `count` deterministic integer coords, uniformly drawn (inclusive) over
 * `[bboxMin - dilate, bboxMax + dilate]` per axis. See the module doc comment
 * for the exact splitmix64 draw order (one `next()` per axis, x then y then z).
 */
export function probeCoords(opts: ProbeOpts): Array<[number, number, number]> {
  const { seed, count, bboxMin, bboxMax, dilate = DEFAULT_DILATE } = opts;
  const rng = new SplitMix64(seed);
  const out: Array<[number, number, number]> = [];
  for (let i = 0; i < count; i++) {
    const x = rng.range(bboxMin[0] - dilate, bboxMax[0] + dilate);
    const y = rng.range(bboxMin[1] - dilate, bboxMax[1] + dilate);
    const z = rng.range(bboxMin[2] - dilate, bboxMax[2] + dilate);
    out.push([x, y, z]);
  }
  return out;
}

const FRACTIONS = [0, 0.25, 0.5, 0.75] as const;

/**
 * `count` deterministic continuous coords: an integer base coord (same
 * distribution as `probeCoords`) plus a per-axis fraction from
 * `{0, 0.25, 0.5, 0.75}`. See the module doc comment for the exact
 * splitmix64 draw order (per axis, in x, y, z order: one `next()` for the
 * integer part immediately followed by one `next()` for the fraction).
 */
export function probePoints(opts: ProbeOpts): Array<[number, number, number]> {
  const { seed, count, bboxMin, bboxMax, dilate = DEFAULT_DILATE } = opts;
  const rng = new SplitMix64(seed);
  const out: Array<[number, number, number]> = [];
  for (let i = 0; i < count; i++) {
    const axes: [number, number, number] = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
      const lo = bboxMin[axis]! - dilate;
      const hi = bboxMax[axis]! + dilate;
      const base = rng.range(lo, hi);
      const fractionIndex = Number(rng.next() % 4n);
      axes[axis] = base + FRACTIONS[fractionIndex]!;
    }
    out.push(axes);
  }
  return out;
}
