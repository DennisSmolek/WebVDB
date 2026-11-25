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
 */

import { vdbRefInvert4, type VDBRefVec3 } from "./scene.js";
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
  phase: { g: number; backG?: number; backWeight?: number },
  n = 4096,
): VDBRefVec3 {
  const c = vdbRefChord(m, o, d);
  if (!c) return [0, 0, 0];
  const w2l = vdbRefInvert4(m.localToWorld);
  const p = vdbRefPhase(dot(d, toLight), phase.g, phase.backG ?? 0, phase.backWeight ?? 0);
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
  return [0, 1, 2].map((k) => m.sigma * albedo[k]! * p * e[k]! * integral) as VDBRefVec3;
}
