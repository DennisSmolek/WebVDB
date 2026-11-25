/**
 * K6.R: closed forms and high-precision quadratures the reference renderer is checked against (homogeneous media in
 * one box or ball).
 *
 * - Beer–Lambert: T = exp(−σ · chord).
 * - Emission only (albedo 0): L = Le · (1 − T) (the renderer's source term σt · Le).
 * - Single scattering of a directional light: L = ∫ σ · albedo · p(cos θ) · E · exp(−σ (t − t₀)) · exp(−σ ℓ(t)) dt
 *   over the camera ray's interval, ℓ(t) the distance from the point to the medium's exit toward the light
 *   (composite Simpson, n panels).
 * - White furnace: albedo 1 under a uniform environment Lₑ (and no other light): L + T · Lₑ = Lₑ.
 * - K6.R1: single scattering of a point light (three's falloff or none) and with a tabulated phase.
 */

import { vdbRefDistanceAttenuation, vdbRefInvert4, type VDBRefVec3, type VDBReferenceFalloff, type VDBReferencePhase } from "./scene.js";
import { vdbRefPhaseTableEval } from "./tables.js";
import { vdbRefPhase, vdbRefVolumeInterval } from "./trace.js";

/** A homogeneous medium in one box or ball. */
export interface VDBRefHomogeneous {
  localToWorld: ArrayLike<number>;
  sphere: boolean;
  /** σt per world unit. */
  sigma: number;
}

const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

/** The chord [t₀, t₁] of the ray o + t·d (|d| = 1) in the medium, or null. */
export function vdbRefChord(m: VDBRefHomogeneous, o: readonly number[], d: readonly number[]): [number, number] | null {
  const [t0, t1] = vdbRefVolumeInterval(vdbRefInvert4(m.localToWorld), m.sphere, o, d, 1e30);
  return t0 < t1 ? [t0, t1] : null;
}

/** Beer–Lambert transmittance of the ray through the medium. */
export function vdbRefBeerLambert(m: VDBRefHomogeneous, o: readonly number[], d: readonly number[]): number {
  const c = vdbRefChord(m, o, d);
  return c ? Math.exp(-m.sigma * (c[1] - c[0])) : 1;
}

/** Radiance of a purely emitting medium (albedo 0) of emitted radiance Le: Le · (1 − T). */
export function vdbRefEmissionOnly(m: VDBRefHomogeneous, o: readonly number[], d: readonly number[], le: VDBRefVec3): VDBRefVec3 {
  const a = 1 - vdbRefBeerLambert(m, o, d);
  return [le[0] * a, le[1] * a, le[2] * a];
}

/**
 * Single-scattered radiance of a directional light (irradiance `e`, `toLight` unit) along o + t·d (|d| = 1) in a
 * homogeneous medium of scattering albedo `albedo`, phase (g, backG, backWeight); composite Simpson, `n` panels.
 */
export function vdbRefSingleScatter(
  m: VDBRefHomogeneous,
  o: readonly number[],
  d: readonly number[],
  toLight: readonly number[],
  e: VDBRefVec3,
  albedo: VDBRefVec3,
  phase: VDBReferencePhase,
  n = 4096,
): VDBRefVec3 {
  const c = vdbRefChord(m, o, d);
  if (!c) return [0, 0, 0];
  const w2l = vdbRefInvert4(m.localToWorld);
  const pc = phaseOf(phase, dot(d, toLight));
  const f = (t: number) => {
    const x = [o[0]! + t * d[0]!, o[1]! + t * d[1]!, o[2]! + t * d[2]!];
    const [, exit] = vdbRefVolumeInterval(w2l, m.sphere, x, toLight, 1e30);
    const l = Math.max(exit, 0);
    return Math.exp(-m.sigma * (t - c[0])) * Math.exp(-m.sigma * l);
  };
  const steps = n % 2 === 0 ? n : n + 1;
  const h = (c[1] - c[0]) / steps;
  let s = f(c[0]) + f(c[1]);
  for (let i = 1; i < steps; i++) s += (i % 2 ? 4 : 2) * f(c[0] + i * h);
  const integral = (s * h) / 3;
  return [0, 1, 2].map((k) => m.sigma * albedo[k]! * pc[k]! * e[k]! * integral) as VDBRefVec3;
}

/** The phase per steradian per channel (HG / dual-lobe, or the table). */
function phaseOf(phase: VDBReferencePhase, cosTheta: number): VDBRefVec3 {
  if (phase.table) return vdbRefPhaseTableEval(phase.table, cosTheta);
  const p = vdbRefPhase(cosTheta, phase.g, phase.backG ?? 0, phase.backWeight ?? 0);
  return [p, p, p];
}

/**
 * K6.R1: single-scattered radiance of a point light (`intensity` at `position`, `falloff` three's attenuation or
 * none) along o + t·d (|d| = 1) in a homogeneous medium; composite Simpson, `n` panels. The light's transmittance
 * is exp(−σ · the part of the segment x → light inside the medium).
 */
export function vdbRefSingleScatterPoint(
  m: VDBRefHomogeneous,
  o: readonly number[],
  d: readonly number[],
  light: { position: VDBRefVec3; intensity: VDBRefVec3; falloff?: VDBReferenceFalloff },
  albedo: VDBRefVec3,
  phase: VDBReferencePhase,
  n = 4096,
): VDBRefVec3 {
  const c = vdbRefChord(m, o, d);
  if (!c) return [0, 0, 0];
  const w2l = vdbRefInvert4(m.localToWorld);
  const f = (t: number): VDBRefVec3 => {
    const x = [o[0]! + t * d[0]!, o[1]! + t * d[1]!, o[2]! + t * d[2]!];
    const to = [light.position[0] - x[0]!, light.position[1] - x[1]!, light.position[2] - x[2]!];
    const dist = Math.hypot(to[0]!, to[1]!, to[2]!);
    const dir = [to[0]! / dist, to[1]! / dist, to[2]! / dist];
    const [t0, t1] = vdbRefVolumeInterval(w2l, m.sphere, x, dir, dist);
    const inside = t0 < t1 ? t1 - t0 : 0;
    const k = (light.falloff ? vdbRefDistanceAttenuation(dist, light.falloff) : 1) * Math.exp(-m.sigma * (t - c[0])) * Math.exp(-m.sigma * inside);
    const p = phaseOf(phase, dot(d, dir));
    return [p[0] * k, p[1] * k, p[2] * k];
  };
  const steps = n % 2 === 0 ? n : n + 1;
  const h = (c[1] - c[0]) / steps;
  const s: VDBRefVec3 = [0, 0, 0];
  for (let i = 0; i <= steps; i++) {
    const w = i === 0 || i === steps ? 1 : i % 2 ? 4 : 2;
    const v = f(c[0] + i * h);
    for (let k = 0; k < 3; k++) s[k] = s[k]! + w * v[k]!;
  }
  return [0, 1, 2].map((k) => (m.sigma * albedo[k]! * light.intensity[k]! * s[k]! * h) / 3) as VDBRefVec3;
}
