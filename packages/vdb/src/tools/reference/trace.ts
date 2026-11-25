/**
 * K6.R: the CPU twin of the reference renderer's GPU kernel (`core/reference/wgsl.ts` `vdbRefTraceCs`): a
 * brute-force volumetric path tracer (design K6-core-renderer.md §7.4) over the renderer's medium model
 * ({@link VDBReferenceScene}). Same algorithm, same random stream ({@link vdbRefRng}), same draw order, in f64:
 *
 * 1. The camera ray of the pixel (centre, or uniform in the pixel with `pixelFilter: 'box'`: 2 draws).
 * 2. The primary transmittance T by ratio tracking (Novák et al. 2014) to infinity.
 * 3. One path by delta tracking (Woodcock): at each real collision with volume i (picked by σᵢ / σ̄): emission
 *    (β · Le), then β ·= albedo, next-event estimation of every light (phase × ratio-tracked transmittance), then
 *    the bounce cap, Russian roulette on β from `rrStart`, and a new direction sampled from the phase (3 draws:
 *    lobe, cos θ, φ). A path that leaves the media after ≥ 1 collision gathers the environment.
 *
 * Tracking runs piecewise over the ray's intervals in each volume's box (or inscribed ball): the majorant on a
 * piece is Σ σᵢ · majorantᵢ of the volumes covering it, so empty gaps cost no steps. Each tentative step draws 1
 * number (free flight); delta tracking draws 1 more to accept or reject; ratio tracking draws 1 when it plays
 * Russian roulette (T < {@link VDB_REFERENCE_RATIO_RR}: survive ½, × 2).
 *
 * Output per sample: (L.r, L.g, L.b, T): L the radiance scattered or emitted toward the camera by the media (the
 * background behind is NOT in L: composite `L + T · background`).
 */

import { vdbRefRng, type VDBRefRng } from "./rng.js";
import {
  VDB_REFERENCE_MAX_STEPS,
  VDB_REFERENCE_RATIO_RR,
  vdbReferenceSettings,
  vdbRefInvert4,
  vdbRefTransformPoint,
  vdbRefTransformVector,
  type VDBRefVec3,
  type VDBReferenceEnvironment,
  type VDBReferenceLight,
  type VDBReferenceScene,
  type VDBReferenceSettings,
  type VDBReferenceVolume,
} from "./scene.js";

/** What the CPU twin reads from a volume's source at uvw (level 0; the source's own clamping). */
export interface VDBReferenceMediumCpu {
  /** density × mask (before the cutoff). */
  density(uvw: readonly number[]): number;
  /** The albedo channel in [0, 1] (default (1, 1, 1)). */
  albedo?(uvw: readonly number[]): VDBRefVec3;
  /** The emission channel value (default 0). */
  emission?(uvw: readonly number[]): number;
}

/**
 * Deliberate breaks for the fail-before checks (each must fail an analytic test); the GPU kernel has the same
 * names as WGSL patches (`core/reference/wgsl.ts` `VDB_REFERENCE_MUTATIONS`).
 */
export type VDBReferenceMutation = "ratio-weight" | "delta-accept" | "phase-norm" | "nee-transmittance" | "emission-albedo" | "escape-depth";

/** Options of {@link vdbReferenceRenderCpu}. */
export interface VDBReferenceCpuOptions {
  /** Batches to accumulate (batch b uses sample indices b·spp … b·spp + spp − 1). Default 1. */
  batches?: number;
  /** First batch index (to continue an accumulation). Default 0. */
  firstBatch?: number;
  /** Only these pixels ([x, y], y from the top); default every pixel. */
  pixels?: readonly (readonly [number, number])[];
  /** A deliberate break (tests). */
  mutate?: VDBReferenceMutation;
}

/** Accumulated per-pixel sums of batch means: `sum` and `sumSq` (RGBA = L.rgb, T), `batches` of them. */
export interface VDBReferenceAccum {
  width: number;
  height: number;
  batches: number;
  /** Σ over batches of the batch mean, RGBA per pixel (row 0 at the top). */
  sum: Float64Array | Float32Array;
  /** Σ over batches of the squared batch mean. */
  sumSq: Float64Array | Float32Array;
}

/** Statistics the CPU twin keeps (not on the GPU). */
export interface VDBReferenceCpuStats {
  samples: number;
  /** Tentative collisions where σ(x) exceeded the majorant (delta tracking is biased then). */
  majorantViolations: number;
  /** Paths cut by `maxBounces` while their throughput was > 0. */
  truncated: number;
  /** Tracks stopped by {@link VDB_REFERENCE_MAX_STEPS}. */
  stepLimited: number;
  /** Mean scattering events per path. */
  meanDepth: number;
  /** Paths with at least 2 scattering events (multiple scattering), and the longest path. */
  multiple: number;
  maxDepth: number;
}

interface PreparedVolume {
  src: VDBReferenceVolume;
  medium: VDBReferenceMediumCpu;
  worldToLocal: Float64Array;
  sigma: number;
  majorant: number;
  cutoff: number;
  albedo: VDBRefVec3;
  g: number;
  backG: number;
  backWeight: number;
  sphere: boolean;
  emits: boolean;
  emission: VDBRefVec3;
  t0: number;
  t1: number;
}

const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const normalize = (a: readonly number[]): VDBRefVec3 => {
  const l = Math.hypot(a[0]!, a[1]!, a[2]!);
  return [a[0]! / l, a[1]! / l, a[2]! / l];
};

/** smoothstep with e1 ≤ e0 read as a step at e0 (WGSL's smoothstep is undefined there). */
export function vdbRefSmoothstep(e0: number, e1: number, x: number): number {
  if (!(e1 > e0)) return x >= e0 ? 1 : 0;
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

/** Henyey-Greenstein phase (per steradian; integrates to 1 over the sphere). */
export function vdbRefHenyeyGreenstein(cosTheta: number, g: number): number {
  const d = 1 + g * g - 2 * g * cosTheta;
  return (1 - g * g) / (4 * Math.PI * d * Math.sqrt(d));
}

/** The dual-lobe phase (1 − w)·HG(g) + w·HG(backG) per steradian. */
export function vdbRefPhase(cosTheta: number, g: number, backG: number, backWeight: number): number {
  const a = vdbRefHenyeyGreenstein(cosTheta, g);
  return backWeight > 0 ? a + (vdbRefHenyeyGreenstein(cosTheta, backG) - a) * backWeight : a;
}

/** cos θ of an HG sample from u ∈ [0, 1) (inverse CDF; isotropic for |g| < 1e-3). */
export function vdbRefSampleHgCos(g: number, u: number): number {
  if (Math.abs(g) < 1e-3) return 1 - 2 * u;
  const s = (1 - g * g) / (1 - g + 2 * g * u);
  return Math.min(Math.max((1 + g * g - s * s) / (2 * g), -1), 1);
}

/** A direction at cos θ, azimuth 2π·u from `n` (Duff et al. 2017 basis). */
export function vdbRefDirectionAround(n: readonly number[], cosTheta: number, u: number): VDBRefVec3 {
  const sign = n[2]! >= 0 ? 1 : -1;
  const a = -1 / (sign + n[2]!);
  const b = n[0]! * n[1]! * a;
  const b1 = [1 + sign * n[0]! * n[0]! * a, sign * b, -sign * n[0]!];
  const b2 = [b, sign + n[1]! * n[1]! * a, -n[1]!];
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const phi = 2 * Math.PI * u;
  const c = Math.cos(phi) * sinTheta;
  const s = Math.sin(phi) * sinTheta;
  return [c * b1[0]! + s * b2[0]! + cosTheta * n[0]!, c * b1[1]! + s * b2[1]! + cosTheta * n[1]!, c * b1[2]! + s * b2[2]! + cosTheta * n[2]!];
}

/** The ray interval [t0, t1] inside a volume (local box or inscribed ball), clipped to [0, tMax]; t0 > t1 if none. */
export function vdbRefVolumeInterval(worldToLocal: ArrayLike<number>, sphere: boolean, o: readonly number[], d: readonly number[], tMax: number): [number, number] {
  const lo = vdbRefTransformPoint(worldToLocal, o);
  const ld = vdbRefTransformVector(worldToLocal, d);
  let t0: number;
  let t1: number;
  if (sphere) {
    const a = dot(ld, ld);
    const b = dot(lo, ld);
    const c = dot(lo, lo) - 0.25;
    const disc = b * b - a * c;
    if (disc <= 0 || a <= 0) return [1, 0];
    const q = Math.sqrt(disc);
    t0 = (-b - q) / a;
    t1 = (-b + q) / a;
  } else {
    t0 = -1e30;
    t1 = 1e30;
    for (let k = 0; k < 3; k++) {
      // A ray parallel to a slab (WGSL has no reliable ±∞): inside it or a miss.
      if (Math.abs(ld[k]!) < 1e-12) {
        if (Math.abs(lo[k]!) > 0.5) return [1, 0];
        continue;
      }
      const ta = (-0.5 - lo[k]!) / ld[k]!;
      const tb = (0.5 - lo[k]!) / ld[k]!;
      t0 = Math.max(t0, Math.min(ta, tb));
      t1 = Math.min(t1, Math.max(ta, tb));
    }
  }
  t0 = Math.max(t0, 0);
  t1 = Math.min(t1, tMax);
  return t0 < t1 ? [t0, t1] : [1, 0];
}

/** Environment radiance in direction `d` (unit). */
export function vdbRefEnvironmentRadiance(env: VDBReferenceEnvironment | undefined, d: readonly number[]): VDBRefVec3 {
  const out: VDBRefVec3 = [0, 0, 0];
  if (!env) return out;
  if (env.uniform) for (let c = 0; c < 3; c++) out[c] = out[c]! + env.uniform[c]!;
  for (const h of env.hemispheres ?? []) {
    const up = normalize(h.up);
    const src = dot(d, up) >= 0 ? h.sky : h.ground;
    for (let c = 0; c < 3; c++) out[c] = out[c]! + src[c]!;
  }
  const e = env.equirect;
  if (e) {
    const r = e.rotation;
    const m = r ? [r[0]! * d[0]! + r[1]! * d[1]! + r[2]! * d[2]!, r[3]! * d[0]! + r[4]! * d[1]! + r[5]! * d[2]!, r[6]! * d[0]! + r[7]! * d[1]! + r[8]! * d[2]!] : d;
    const u = Math.atan2(m[2]!, m[0]!) / (2 * Math.PI) + 0.5;
    const v = Math.asin(Math.min(Math.max(m[1]!, -1), 1)) / Math.PI + 0.5;
    const x = Math.min(Math.max(Math.floor(u * e.width), 0), e.width - 1);
    const y = Math.min(Math.max(Math.floor((1 - v) * e.height), 0), e.height - 1);
    const k = (y * e.width + x) * 4;
    const s = e.intensity ?? 1;
    for (let c = 0; c < 3; c++) out[c] = out[c]! + e.data[k + c]! * s;
  }
  return out;
}

/** The camera ray of pixel (px, py) (y from the top) at sub-pixel offset (jx, jy) ∈ [0, 1)². */
export function vdbRefCameraRay(camera: VDBReferenceScene["camera"], width: number, height: number, px: number, py: number, jx: number, jy: number): { o: VDBRefVec3; d: VDBRefVec3 } {
  const ndc = [((px + jx) / width) * 2 - 1, (1 - (py + jy) / height) * 2 - 1];
  const near = vdbRefTransformPoint(camera.projectionInverse, [ndc[0]!, ndc[1]!, 0]);
  const far = vdbRefTransformPoint(camera.projectionInverse, [ndc[0]!, ndc[1]!, 1]);
  const dv = normalize([far[0] - near[0], far[1] - near[1], far[2] - near[2]]);
  const ov = camera.orthographic ? vdbRefTransformPoint(camera.projectionInverse, [ndc[0]!, ndc[1]!, -1]) : [0, 0, 0];
  return { o: vdbRefTransformPoint(camera.cameraToWorld, ov), d: normalize(vdbRefTransformVector(camera.cameraToWorld, dv)) };
}

interface Tracker {
  vols: PreparedVolume[];
  stats: VDBReferenceCpuStats;
  mutate: VDBReferenceMutation | undefined;
}

/** Sample the medium of volume `v` at world `x`: σt (after the cutoff). */
function sigmaAt(v: PreparedVolume, x: readonly number[], stats: VDBReferenceCpuStats): number {
  const l = vdbRefTransformPoint(v.worldToLocal, x);
  const value = v.medium.density([l[0] + 0.5, l[1] + 0.5, l[2] + 0.5]);
  const s = value > v.cutoff ? value : 0;
  if (s * v.sigma > v.majorant * (1 + 1e-6)) stats.majorantViolations++;
  return s * v.sigma;
}

/**
 * Delta (`ratio` false: returns the collision) or ratio tracking (returns T) along o + t·d, t ∈ [0, tMax].
 * Intervals are computed first (no draws); see the file comment for the draw order.
 */
function track(tr: Tracker, rng: VDBRefRng, o: readonly number[], d: readonly number[], tMax: number, ratio: boolean): { escaped: boolean; t: number; vol: number; T: number } {
  const { vols } = tr;
  for (const v of vols) [v.t0, v.t1] = vdbRefVolumeInterval(v.worldToLocal, v.sphere, o, d, tMax);
  let t = 0;
  let T = 1;
  const sig = new Float64Array(vols.length);
  for (let step = 0; step < VDB_REFERENCE_MAX_STEPS; step++) {
    let sbar = 0;
    let tb = tMax;
    for (const v of vols) {
      if (v.t0 <= t && t < v.t1) {
        sbar += v.majorant;
        tb = Math.min(tb, v.t1);
      } else if (v.t0 > t && v.t0 <= v.t1) {
        tb = Math.min(tb, v.t0);
      }
    }
    if (!(sbar > 0)) {
      if (tb >= tMax) return { escaped: true, t: tMax, vol: -1, T };
      t = tb;
      continue;
    }
    const dt = -Math.log(1 - rng.next()) / sbar;
    if (t + dt >= tb) {
      t = tb;
      if (t >= tMax) return { escaped: true, t: tMax, vol: -1, T };
      continue;
    }
    t += dt;
    const x = [o[0]! + t * d[0]!, o[1]! + t * d[1]!, o[2]! + t * d[2]!];
    let sum = 0;
    for (let i = 0; i < vols.length; i++) {
      const v = vols[i]!;
      sig[i] = v.t0 <= t && t < v.t1 ? sigmaAt(v, x, tr.stats) : 0;
      sum += sig[i]!;
    }
    if (!ratio) {
      // A null collision unless u falls in a volume's share; the mutation accepts every tentative collision.
      const u = tr.mutate === "delta-accept" ? -1 : rng.next() * sbar;
      let acc = 0;
      for (let i = 0; i < vols.length; i++) {
        const v = vols[i]!;
        if (!(v.t0 <= t && t < v.t1)) continue;
        acc += sig[i]!;
        if (u < acc) return { escaped: false, t, vol: i, T };
      }
    } else {
      T *= 1 - (tr.mutate === "ratio-weight" ? 0.9 : 1) * (sum / sbar);
      if (T < VDB_REFERENCE_RATIO_RR) {
        if (rng.next() >= 0.5) return { escaped: false, t, vol: -1, T: 0 };
        T *= 2;
      }
    }
  }
  tr.stats.stepLimited++;
  return { escaped: true, t, vol: -1, T: ratio ? T : 0 };
}

function prepare(scene: VDBReferenceScene, media: readonly VDBReferenceMediumCpu[]): PreparedVolume[] {
  if (media.length !== scene.volumes.length) throw new RangeError(`vdbReferenceRenderCpu: ${scene.volumes.length} volumes but ${media.length} media`);
  return scene.volumes.map((src, i) => {
    const e = src.emission ?? {};
    const c = e.color ?? [1, 1, 1];
    const k = e.intensity ?? 1;
    return {
      src,
      medium: media[i]!,
      worldToLocal: vdbRefInvert4(src.localToWorld),
      sigma: src.sigma,
      majorant: src.sigma * (src.majorant ?? 1),
      cutoff: src.cutoff ?? 0,
      albedo: [...(src.albedo ?? [1, 1, 1])] as VDBRefVec3,
      g: src.phase?.g ?? 0,
      backG: src.phase?.backG ?? 0,
      backWeight: src.phase?.backWeight ?? 0,
      sphere: src.shape === "sphere",
      emits: !!src.emits,
      emission: [c[0] * k, c[1] * k, c[2] * k],
      t0: 1,
      t1: 0,
    };
  });
}

/** The emitted radiance Le of volume `v` at uvw (the renderer's temperature model). */
function emitted(v: PreparedVolume, uvw: readonly number[]): VDBRefVec3 {
  if (!v.emits || !v.medium.emission) return [0, 0, 0];
  const range = v.src.emission?.temperatureRange ?? [0, 6500];
  const temperature = range[0] + (range[1] - range[0]) * v.medium.emission(uvw);
  if (!(temperature > 0)) return [0, 0, 0];
  const power = Math.pow(temperature / Math.max(range[1], 1), 4);
  return [v.emission[0] * power, v.emission[1] * power, v.emission[2] * power];
}

/** One light's NEE setup at x: direction, distance and Li (already × spot attenuation); null when it is dark. */
export function vdbRefLightAt(l: VDBReferenceLight, x: readonly number[]): { dir: VDBRefVec3; dist: number; li: VDBRefVec3 } | null {
  if (l.kind === "directional") {
    return l.irradiance.some((c) => c > 0) ? { dir: normalize(l.direction), dist: 1e30, li: [...l.irradiance] as VDBRefVec3 } : null;
  }
  const to = [l.position[0] - x[0]!, l.position[1] - x[1]!, l.position[2] - x[2]!];
  const dist = Math.hypot(to[0]!, to[1]!, to[2]!);
  if (!(dist > 0)) return null;
  const dir: VDBRefVec3 = [to[0]! / dist, to[1]! / dist, to[2]! / dist];
  let k = l.inverseSquare ? 1 / (dist * dist) : 1;
  if (l.kind === "spot") k *= vdbRefSmoothstep(l.cosOuter, l.cosInner, -dot(dir, normalize(l.direction)));
  if (!(k > 0)) return null;
  const li: VDBRefVec3 = [l.intensity[0] * k, l.intensity[1] * k, l.intensity[2] * k];
  return li.some((c) => c > 0) ? { dir, dist, li } : null;
}

/**
 * One sample of pixel (px, py): (L.r, L.g, L.b, T). `sampleIndex` = batch · spp + s.
 */
export function vdbReferenceSampleCpu(
  scene: VDBReferenceScene,
  vols: PreparedVolume[],
  settings: Required<VDBReferenceSettings>,
  px: number,
  py: number,
  sampleIndex: number,
  stats: VDBReferenceCpuStats,
  mutate?: VDBReferenceMutation,
): [number, number, number, number] {
  const rng = vdbRefRng(settings.seed, sampleIndex, py * settings.width + px);
  const tr: Tracker = { vols, stats, mutate };
  let jx = 0.5;
  let jy = 0.5;
  if (settings.pixelFilter === "box") {
    jx = rng.next();
    jy = rng.next();
  }
  const ray = vdbRefCameraRay(scene.camera, settings.width, settings.height, px, py, jx, jy);
  const primary = track(tr, rng, ray.o, ray.d, 1e30, true);
  const L: VDBRefVec3 = [0, 0, 0];
  const beta: VDBRefVec3 = [1, 1, 1];
  let o: readonly number[] = ray.o;
  let d: VDBRefVec3 = ray.d;
  let depth = 0;
  const lights = scene.lights ?? [];
  stats.samples++;
  for (;;) {
    const hit = track(tr, rng, o, d, 1e30, false);
    if (hit.escaped) {
      if (depth > 0 && !(mutate === "escape-depth" && depth > 1)) {
        const e = vdbRefEnvironmentRadiance(scene.environment, d);
        for (let c = 0; c < 3; c++) L[c] = L[c]! + beta[c]! * e[c]!;
      }
      break;
    }
    depth++;
    const v = vols[hit.vol]!;
    const x: VDBRefVec3 = [o[0]! + hit.t * d[0], o[1]! + hit.t * d[1], o[2]! + hit.t * d[2]];
    const l = vdbRefTransformPoint(v.worldToLocal, x);
    const uvw = [l[0] + 0.5, l[1] + 0.5, l[2] + 0.5];
    const a = v.medium.albedo ? v.medium.albedo(uvw) : [1, 1, 1];
    const albedo = [v.albedo[0] * a[0]!, v.albedo[1] * a[1]!, v.albedo[2] * a[2]!];
    const le = emitted(v, uvw);
    for (let c = 0; c < 3; c++) L[c] = L[c]! + beta[c]! * le[c]! * (mutate === "emission-albedo" ? albedo[c]! : 1);
    for (let c = 0; c < 3; c++) beta[c] = beta[c]! * albedo[c]!;
    // A black collision ends the path (no draws: the GPU kernel ends it at the same point).
    if (!(Math.max(beta[0], beta[1], beta[2]) > 0)) break;
    for (const light of lights) {
      const nee = vdbRefLightAt(light, x);
      if (!nee) continue;
      const p = vdbRefPhase(dot(d, nee.dir), v.g, v.backG, v.backWeight) * (mutate === "phase-norm" ? 4 * Math.PI : 1);
      const tl = track(tr, rng, x, nee.dir, nee.dist, true).T;
      const t = mutate === "nee-transmittance" ? 1 : tl;
      for (let c = 0; c < 3; c++) L[c] = L[c]! + beta[c]! * nee.li[c]! * p * t;
    }
    if (depth >= settings.maxBounces) {
      if (beta.some((b) => b > 0)) stats.truncated++;
      break;
    }
    if (depth >= settings.rrStart) {
      const q = Math.min(Math.max(beta[0], beta[1], beta[2]), 1);
      if (q < 1) {
        if (rng.next() >= q) break;
        for (let c = 0; c < 3; c++) beta[c] = beta[c]! / q;
      }
    }
    const lobe = rng.next();
    const u1 = rng.next();
    const u2 = rng.next();
    const g = v.backWeight > 0 && lobe < v.backWeight ? v.backG : v.g;
    d = normalize(vdbRefDirectionAround(d, vdbRefSampleHgCos(g, u1), u2));
    o = x;
  }
  stats.meanDepth += depth;
  if (depth >= 2) stats.multiple++;
  stats.maxDepth = Math.max(stats.maxDepth, depth);
  return [L[0], L[1], L[2], primary.T];
}

/**
 * Render the reference on the CPU: `batches` batches of `spp` samples per pixel, accumulated as per-pixel sums of
 * batch means (the GPU accumulation's layout). Small images or sampled pixels only (it is f64 JavaScript).
 */
export function vdbReferenceRenderCpu(
  scene: VDBReferenceScene,
  media: readonly VDBReferenceMediumCpu[],
  settingsIn: VDBReferenceSettings,
  o: VDBReferenceCpuOptions = {},
): VDBReferenceAccum & { stats: VDBReferenceCpuStats } {
  const settings = vdbReferenceSettings(settingsIn);
  const vols = prepare(scene, media);
  const batches = o.batches ?? 1;
  const first = o.firstBatch ?? 0;
  const { width, height, spp } = settings;
  const sum = new Float64Array(width * height * 4);
  const sumSq = new Float64Array(width * height * 4);
  const stats: VDBReferenceCpuStats = { samples: 0, majorantViolations: 0, truncated: 0, stepLimited: 0, meanDepth: 0, multiple: 0, maxDepth: 0 };
  const pixels = o.pixels ?? Array.from({ length: width * height }, (_, i) => [i % width, Math.floor(i / width)] as const);
  for (const [px, py] of pixels) {
    const k = (py * width + px) * 4;
    for (let b = first; b < first + batches; b++) {
      const m = [0, 0, 0, 0];
      for (let s = 0; s < spp; s++) {
        const r = vdbReferenceSampleCpu(scene, vols, settings, px, py, b * spp + s, stats, o.mutate);
        for (let c = 0; c < 4; c++) m[c] = m[c]! + r[c]!;
      }
      for (let c = 0; c < 4; c++) {
        const mean = m[c]! / spp;
        sum[k + c] = sum[k + c]! + mean;
        sumSq[k + c] = sumSq[k + c]! + mean * mean;
      }
    }
  }
  stats.meanDepth = stats.samples ? stats.meanDepth / stats.samples : 0;
  return { width, height, batches, sum, sumSq, stats };
}

/**
 * Mean and standard error of the mean per pixel and channel from accumulated sums (n batches; the error needs
 * n ≥ 2: it is the spread of batch means / √n).
 */
export function vdbReferenceStatistics(a: VDBReferenceAccum): { mean: Float64Array; stderr: Float64Array } {
  const n = a.batches;
  const mean = new Float64Array(a.sum.length);
  const stderr = new Float64Array(a.sum.length);
  for (let i = 0; i < a.sum.length; i++) {
    const m = a.sum[i]! / n;
    mean[i] = m;
    stderr[i] = n > 1 ? Math.sqrt(Math.max(a.sumSq[i]! / n - m * m, 0) / (n - 1)) : Infinity;
  }
  return { mean, stderr };
}
