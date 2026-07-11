/**
 * Local copy of `packages/nanovdb-wgsl/src/cpu/probe-coords.ts` for the GPU
 * parity harness page. Duplicated (not imported) because the original
 * module lives inside a package whose public `exports` map only publishes
 * `.` and `./pnanovdb.wgsl` (see `packages/nanovdb-wgsl/package.json`) — this
 * harness's lane is CREATE-only under `examples/src/harness/`, so rather
 * than widen that package's exports (out of lane) this is a byte-for-byte
 * transliteration of the same splitmix64 generator and draw order. The
 * algorithm is copied verbatim; see the original for the full derivation
 * notes (bit-for-bit match with `docker/fixture-bake/bake_primitives.cpp`'s
 * `Rng`, generation order documented there).
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

  next(): bigint {
    this.state = (this.state + SPLITMIX64_INC) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * SPLITMIX64_MUL1) & MASK64;
    z = ((z ^ (z >> 27n)) * SPLITMIX64_MUL2) & MASK64;
    z = z ^ (z >> 31n);
    return z;
  }

  range(lo: number, hi: number): number {
    const span = BigInt(hi - lo + 1);
    const r = this.next() % span;
    return lo + Number(r);
  }
}

const DEFAULT_DILATE = 4;

export interface ProbeOpts {
  seed: bigint;
  count: number;
  bboxMin: readonly [number, number, number];
  bboxMax: readonly [number, number, number];
  dilate?: number;
}

/** `count` deterministic integer coords — one `next()` per axis, x then y then z. */
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

/** `count` deterministic continuous coords — per axis, integer draw then fraction draw. */
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
