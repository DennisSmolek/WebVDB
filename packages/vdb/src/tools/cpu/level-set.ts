/**
 * CPU twins (D11) of the opt-in `'levelSet'` WGSL helper (K3.7,
 * `core/wgsl/level-set.ts`): `vdbLevelSetHit`, `vdbLevelSetGradient`,
 * `vdbLevelSetNormal`. Same walk, same sample positions, f32 arithmetic in the
 * WGSL's order (`Math.fround`), so a GPU hit and this one agree to the f32
 * noise of the sample positions (GPUs may fuse multiply-adds; compare `t`
 * with a tolerance). Plus {@link levelSetHitReference}, a brute-force fixed-step
 * march with no skipping: the independent oracle the twin is tested against.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/levelset/webgpu/LevelSetSurfaceMaterial.js`): the sphere
 * trace with a minimum step, the secant step between the last outside and the
 * first inside sample, central-difference normals. Copyright (c)
 * mjurczyk/openvdb contributors, MIT (see THIRD_PARTY.md). Changes: sparse
 * NanoVDB tree at full resolution, HDDA skip, bisection, normals through the
 * grid affine (DIVERGES.md).
 */
import { getDim, hddaInit, hddaStep, hddaUpdate, indexBBox, readValueF32, type Hdda, type Vec3 } from "./march.js";

const f = Math.fround;

/** Options of {@link levelSetHit}. */
export interface LevelSetHitOptions {
  /** Index voxels per SDF unit (1 / voxel size for a level set in its own world units). Default 1. */
  sdfToIndex?: number;
  /** Sample cap (the WGSL `maxSamples`); a ray that runs out misses. Default 512. */
  maxSamples?: number;
}

/** Result of {@link levelSetHit}. */
export interface LevelSetHit {
  hit: boolean;
  /** The crossing `t` on a hit (else NaN). */
  t: number;
  /** Trilinear samples taken, refinement included. */
  samples: number;
  /** HDDA cells visited and cells skipped (diagnostics). */
  cells: number;
  skipped: number;
}

/**
 * f32 trilinear sample with the vendored tap order: taps through
 * `readValueF32`, `mix(a, b, t)` as `a + t * (b - a)`, each op rounded.
 */
export function sampleTrilinearF32(gridWords: Uint32Array, xyz: readonly number[]): number {
  const bx = Math.floor(xyz[0]!);
  const by = Math.floor(xyz[1]!);
  const bz = Math.floor(xyz[2]!);
  const u = f(xyz[0]! - bx);
  const v = f(xyz[1]! - by);
  const w = f(xyz[2]! - bz);
  const c = (dx: number, dy: number, dz: number) => readValueF32(gridWords, [bx + dx, by + dy, bz + dz]);
  const mix = (a: number, b: number, t: number) => f(a + f(t * f(b - a)));
  const x00 = mix(c(0, 0, 0), c(1, 0, 0), u);
  const x10 = mix(c(0, 1, 0), c(1, 1, 0), u);
  const x01 = mix(c(0, 0, 1), c(1, 0, 1), u);
  const x11 = mix(c(0, 1, 1), c(1, 1, 1), u);
  return mix(mix(x00, x10, v), mix(x01, x11, v), w);
}

const at = (o: Vec3, d: Vec3, t: number): Vec3 => [f(o[0] + f(d[0] * t)), f(o[1] + f(d[1] * t)), f(o[2] + f(d[2] * t))];

/** Twin of `vdb_ls_skip`: every cell of the 2x2x2 block of `dim`-cells from `c` is uniform and above 0. */
function skipCell(gridWords: Uint32Array, c: Vec3, dim: number): boolean {
  for (let k = 0; k < 8; k++) {
    const p: Vec3 = [c[0] + (k & 1) * dim, c[1] + ((k >> 1) & 1) * dim, c[2] + ((k >> 2) & 1) * dim];
    if (getDim(gridWords, p) < dim) return false;
    if (!(readValueF32(gridWords, p) > 0)) return false;
  }
  return true;
}

/** The vendored `hdda_ray_clip` slab test against `[lo, hi]` (f32). */
function clip(lo: Vec3, hi: Vec3, o: Vec3, d: Vec3, t0: number, t1: number): { hit: boolean; t0: number; t1: number } {
  const near: number[] = [];
  const far: number[] = [];
  for (let i = 0; i < 3; i++) {
    const inv = f(1 / d[i]!);
    const a = f(f(lo[i]! - o[i]!) * inv);
    const b = f(f(hi[i]! - o[i]!) * inv);
    near.push(Math.min(a, b));
    far.push(Math.max(a, b));
  }
  const tnear = Math.max(near[0]!, near[1]!, near[2]!);
  const tfar = Math.min(far[0]!, far[1]!, far[2]!);
  const r0 = Math.max(f(t0), tnear);
  const r1 = Math.min(f(t1), tfar);
  return { hit: tnear <= tfar && r0 <= r1, t0: r0, t1: r1 };
}

/**
 * Twin of `vdbLevelSetHit(g, o, d, &t, sdfToIndex, maxSamples)`: the first
 * zero crossing (from >= 0 to < 0) of the trilinear SDF along `o + d * t`,
 * `t` in `[t0, t1]` clipped to the support box `[min - 1, max + 1]`, in index
 * space. A ray whose first sample is inside hits there.
 */
export function levelSetHit(
  gridWords: Uint32Array,
  origin: readonly number[],
  direction: readonly number[],
  t: readonly [number, number],
  options: LevelSetHitOptions = {},
): LevelSetHit {
  const miss = (samples: number, cells: number, skipped: number): LevelSetHit => ({ hit: false, t: NaN, samples, cells, skipped });
  const o: Vec3 = [f(origin[0]!), f(origin[1]!), f(origin[2]!)];
  const d: Vec3 = [f(direction[0]!), f(direction[1]!), f(direction[2]!)];
  const box = indexBBox(gridWords);
  const c = clip(
    [box.min[0] - 1, box.min[1] - 1, box.min[2] - 1],
    [box.max[0] + 1, box.max[1] + 1, box.max[2] + 1],
    o,
    d,
    t[0],
    t[1],
  );
  if (!c.hit) return miss(0, 0, 0);
  const t0 = c.t0;
  const t1 = c.t1;
  const maxSamples = options.maxSamples ?? 512;
  const inv = f(1 / Math.max(f(Math.hypot(d[0], d[1], d[2])), 1e-30));
  const minStep = f(0.25 * inv);
  const k = f(f(f(0.95 * f(options.sdfToIndex ?? 1)) * inv));
  const sample = (tt: number) => sampleTrilinearF32(gridWords, at(o, d, tt));
  const p0 = at(o, d, t0);
  const h: Hdda = hddaInit(o, t0, d, t1, Math.max(getDim(gridWords, [Math.floor(p0[0]), Math.floor(p0[1]), Math.floor(p0[2])]), 8));
  let tc = t0;
  let tp = t0;
  let vp = 0;
  let have = false;
  let n = 0;
  let updates = 0;
  let fineUntil = -3.0e38;
  let crossed = false;
  let vc = 0;
  let cells = 0;
  let skipped = 0;
  for (let it = 0; it < 8192; it++) {
    let want = Math.max(getDim(gridWords, h.voxel), 8);
    if (h.tmin < fineUntil) want = 8;
    if (want !== h.dim && updates < 2) {
      hddaUpdate(h, o, d, want);
      updates++;
      continue;
    }
    updates = 0;
    const te = Math.min(Math.min(h.next[0], h.next[1]), Math.min(h.next[2], h.tmax));
    if (te >= tc) {
      const dim = h.dim;
      cells++;
      if (skipCell(gridWords, [h.voxel[0], h.voxel[1], h.voxel[2]], dim)) {
        skipped++;
        tc = te;
      } else if (dim > 8 && te > h.tmin) {
        fineUntil = te;
        hddaUpdate(h, o, d, 8);
        continue;
      } else {
        for (;;) {
          if (tc > te || n >= maxSamples) break;
          vc = sample(tc);
          n++;
          if (vc < 0) {
            crossed = true;
            break;
          }
          tp = tc;
          vp = vc;
          have = true;
          tc = f(tc + Math.max(f(vc * k), minStep));
        }
        if (crossed || n >= maxSamples) break;
      }
    }
    if (!hddaStep(h)) break;
  }
  if (!crossed && have && tp < t1 && n < maxSamples) {
    tc = t1;
    vc = sample(tc);
    n++;
    crossed = vc < 0;
  }
  if (!crossed) return miss(n, cells, skipped);
  if (!have) return { hit: true, t: tc, samples: n, cells, skipped };
  let lo = tp;
  let hi = tc;
  let vlo = vp;
  let vhi = vc;
  for (let i = 0; i < 8; i++) {
    const m = f(0.5 * f(lo + hi));
    const vm = sample(m);
    n++;
    if (vm < 0) {
      hi = m;
      vhi = vm;
    } else {
      lo = m;
      vlo = vm;
    }
  }
  return { hit: true, t: f(lo + f(f(hi - lo) * f(vlo / f(vlo - vhi)))), samples: n, cells, skipped };
}

/**
 * Brute-force oracle for {@link levelSetHit}: a fixed-step march (`step`
 * voxels, default 0.125) through the whole support-box interval with no
 * skipping and no sphere steps (f64 trilinear through `readValueF32` taps),
 * then the same bisection + secant refinement in f64. Slow; for tests.
 */
export function levelSetHitReference(
  gridWords: Uint32Array,
  origin: readonly number[],
  direction: readonly number[],
  t: readonly [number, number],
  options: { step?: number } = {},
): { hit: boolean; t: number } {
  const o: Vec3 = [origin[0]!, origin[1]!, origin[2]!];
  const d: Vec3 = [direction[0]!, direction[1]!, direction[2]!];
  const box = indexBBox(gridWords);
  const c = clip([box.min[0] - 1, box.min[1] - 1, box.min[2] - 1], [box.max[0] + 1, box.max[1] + 1, box.max[2] + 1], o, d, t[0], t[1]);
  if (!c.hit) return { hit: false, t: NaN };
  const len = Math.hypot(d[0], d[1], d[2]);
  const dt = (options.step ?? 0.125) / len;
  const sample = (tt: number): number => {
    const p = [o[0] + d[0] * tt, o[1] + d[1] * tt, o[2] + d[2] * tt];
    const b = p.map(Math.floor) as Vec3;
    const u = [p[0]! - b[0], p[1]! - b[1], p[2]! - b[2]];
    const tap = (dx: number, dy: number, dz: number) => readValueF32(gridWords, [b[0] + dx, b[1] + dy, b[2] + dz]);
    const l = (a: number, bb: number, s: number) => a + (bb - a) * s;
    return l(
      l(l(tap(0, 0, 0), tap(1, 0, 0), u[0]!), l(tap(0, 1, 0), tap(1, 1, 0), u[0]!), u[1]!),
      l(l(tap(0, 0, 1), tap(1, 0, 1), u[0]!), l(tap(0, 1, 1), tap(1, 1, 1), u[0]!), u[1]!),
      u[2]!,
    );
  };
  let tp = c.t0;
  let vp = sample(tp);
  if (vp < 0) return { hit: true, t: tp };
  for (let tc = c.t0 + dt; ; tc += dt) {
    const tt = Math.min(tc, c.t1);
    const v = sample(tt);
    if (v < 0) {
      let lo = tp;
      let hi = tt;
      let vlo = vp;
      let vhi = v;
      for (let i = 0; i < 30; i++) {
        const m = 0.5 * (lo + hi);
        const vm = sample(m);
        if (vm < 0) {
          hi = m;
          vhi = vm;
        } else {
          lo = m;
          vlo = vm;
        }
      }
      return { hit: true, t: lo + (hi - lo) * (vlo / (vlo - vhi)) };
    }
    if (tt >= c.t1) return { hit: false, t: NaN };
    tp = tt;
    vp = v;
  }
}

/**
 * Twin of `vdbLevelSetGradient(acc, xyz, h)`: central differences of the f32
 * trilinear sample at +/- `h` index voxels, in SDF units per voxel.
 */
export function levelSetGradient(gridWords: Uint32Array, xyz: readonly number[], h = 1): Vec3 {
  const p: Vec3 = [f(xyz[0]!), f(xyz[1]!), f(xyz[2]!)];
  const s = (dx: number, dy: number, dz: number) => sampleTrilinearF32(gridWords, [f(p[0] + dx), f(p[1] + dy), f(p[2] + dz)]);
  const hh = f(h);
  const den = f(2 * hh);
  return [f(f(s(hh, 0, 0) - s(-hh, 0, 0)) / den), f(f(s(0, hh, 0) - s(0, -hh, 0)) / den), f(f(s(0, 0, hh) - s(0, 0, -hh)) / den)];
}

/**
 * Twin of `vdbLevelSetNormal(g, xyz, h)`: the world-space unit normal
 * `normalize(W^T * gradient)`, with `worldToIndex` the grid's 3x4 row-major
 * world-to-index affine (`pool.info(id).worldToIndex`; only the linear part is
 * used). A zero gradient gives `[0, 1, 0]`.
 */
export function levelSetNormal(gridWords: Uint32Array, xyz: readonly number[], worldToIndex: ArrayLike<number>, h = 1): Vec3 {
  const g = levelSetGradient(gridWords, xyz, h);
  const m = worldToIndex;
  const n: Vec3 = [
    m[0]! * g[0] + m[4]! * g[1] + m[8]! * g[2],
    m[1]! * g[0] + m[5]! * g[1] + m[9]! * g[2],
    m[2]! * g[0] + m[6]! * g[1] + m[10]! * g[2],
  ];
  const l = Math.hypot(n[0], n[1], n[2]);
  if (!(l > 0)) return [0, 1, 0];
  return [n[0] / l, n[1] / l, n[2] / l];
}
