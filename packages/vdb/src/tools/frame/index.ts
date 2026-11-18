// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L94–107 (frameSliceDepth, frameGetViewPosition),
//   L129–142 (frameDistanceFade), L157–185 (getSampledLod, getBoxSteps, frameNoise), L187–194
//   (frameHenyeyGreenstein), L273–288 (framePhaseWeight), L330–373 (shadow step / depth / transmittance),
//   L464–481 (frameShadowedLight);
//   src/three-openvdb/frame/webgpu/FrameInjectMaterial.js L193–232 (slice pre-integration);
//   src/three-openvdb/frame/webgpu/FrameIntegrateMaterial.js L219–293 (slice march and composite);
//   src/three-openvdb/frame/core/frameView.js L258–271 (getFroxelSlice, getFroxelSliceRange).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
//   K3.4: frameShaderChunks.js L196–203 (frameHenyeyGreensteinColor), L273–328 (the Lorenz-Mie branches
//   of framePhaseWeight / framePhaseColor, framePhaseWeightScaled), L342–399 (octave march depth and
//   transmittance, frameShadowOpticalDepth, frameOctaveLight), L416–462 (frameTwoSidedDiffusion,
//   frameShadowTotal), L464–481 (frameShadowedLight with octaves); core/framePhase.js L110–129
//   (getFrameScatteringOctaves).
// Changes: CPU (f64) twins of the TSL shader maths, used as test oracles (grey; the spectral twins are in
// spectral.ts). Divergences: DIVERGES.md (K3.1 and K3.4 rows).

/**
 * CPU twins of the froxel pipeline maths (K3.1, R1 §6 Step 1; CLAUDE.md "every WGSL entry has a CPU
 * twin"). Each function mirrors one TSL chunk of `@pmndrs/vdb/three` `frame/chunks.ts`, in f64, so
 * the GPU passes can be checked against it (e2e `k3-frame` probe) and the maths can move into `core`
 * as WGSL later (K3.8) with the same oracle. Engine-free: plain numbers and tuples.
 */

import { vdbFrameMieAsymmetry, vdbFrameMiePhase, vdbFrameMieRgbMode, vdbFrameMieRgbPhase } from "./mie.js";

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];

const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const fract = (x: number) => x - Math.floor(x);

/** frameSliceDepth: view depth at the slice fraction `t` of the log range (near, far). */
export function vdbFrameSliceDepth(range: Vec2, t: number): number {
  return range[0] * Math.pow(range[1] / range[0], t);
}

/** frameView.getFroxelSlice: fractional slice of a view depth (inverse of {@link vdbFrameSliceDepth} × depth). */
export function vdbFrameFroxelSlice(viewDepth: number, range: Vec2, depth: number): number {
  return (depth * Math.log(Math.max(viewDepth, range[0]) / range[0])) / Math.log(range[1] / range[0]);
}

/** frameView.getFroxelSliceRange: first slice and slice count of a view depth range. */
export function vdbFrameSliceRange(viewNear: number, viewFar: number, range: Vec2, depth: number): { first: number; count: number } {
  const first = Math.max(0, Math.floor(vdbFrameFroxelSlice(viewNear, range, depth)));
  const last = Math.min(depth - 1, Math.ceil(vdbFrameFroxelSlice(viewFar, range, depth)));
  return { first, count: last - first + 1 };
}

/** Column-major 4x4 times (x, y, z, w). */
function mul4(m: ArrayLike<number>, v: Vec4): Vec4 {
  const o: Vec4 = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) o[r] = m[r]! * v[0] + m[4 + r]! * v[1] + m[8 + r]! * v[2] + m[12 + r]! * v[3];
  return o;
}

/**
 * frameGetViewPosition: the view-space point at view depth `viewDepth` on the ray of `uv`
 * (bottom-left [0, 1]²), from the inverse projection (column-major). NDC depth 0 and 1 are both in
 * the frustum (WebGPU and WebGL), so the ray is the line through the two unprojected points.
 */
export function vdbFrameViewPosition(projectionInverse: ArrayLike<number>, uv: Vec2, viewDepth: number): Vec3 {
  const ndc: Vec2 = [uv[0] * 2 - 1, uv[1] * 2 - 1];
  const n = mul4(projectionInverse, [ndc[0], ndc[1], 0, 1]);
  const f = mul4(projectionInverse, [ndc[0], ndc[1], 1, 1]);
  const near: Vec3 = [n[0] / n[3], n[1] / n[3], n[2] / n[3]];
  const far: Vec3 = [f[0] / f[3], f[1] / f[3], f[2] / f[3]];
  const t = (viewDepth + near[2]) / (near[2] - far[2]);
  return [near[0] + (far[0] - near[0]) * t, near[1] + (far[1] - near[1]) * t, near[2] + (far[2] - near[2]) * t];
}

/** frameGetFroxel: slice, uv and view positions of the froxel at an atlas pixel (bottom-left `fragCoord`). */
export function vdbFrameFroxel(
  grid: Vec3,
  columns: number,
  range: Vec2,
  projectionInverse: ArrayLike<number>,
  fragCoord: Vec2,
): { tile: Vec2; slice: number; uv: Vec2; sliceNear: number; sliceFar: number; viewNear: Vec3; viewFar: Vec3 } {
  const tile: Vec2 = [Math.floor(fragCoord[0] / grid[0]), Math.floor(fragCoord[1] / grid[1])];
  const slice = tile[1] * columns + tile[0];
  const uv: Vec2 = [(fragCoord[0] - tile[0] * grid[0]) / grid[0], (fragCoord[1] - tile[1] * grid[1]) / grid[1]];
  const sliceNear = vdbFrameSliceDepth(range, slice / grid[2]);
  const sliceFar = vdbFrameSliceDepth(range, (slice + 1) / grid[2]);
  return {
    tile,
    slice,
    uv,
    sliceNear,
    sliceFar,
    viewNear: vdbFrameViewPosition(projectionInverse, uv, sliceNear),
    viewFar: vdbFrameViewPosition(projectionInverse, uv, sliceFar),
  };
}

/** frameDistanceFade: density scale at a view position (radius 0 = no fade). */
export function vdbFrameDistanceFade(fadeRadius: number, viewPosition: Vec3): number {
  if (!(fadeRadius > 0)) return 1;
  const k = 4.0;
  const end = Math.exp(-k);
  const x = clamp(Math.hypot(...viewPosition) / fadeRadius, 0, 1);
  return (Math.exp(-k * x * x) - end) / (1 - end);
}

/** getBoxSteps: sub-steps across the in-box part of a slice (2 per voxel of the read level, ≤ subSteps). */
export function vdbFrameBoxSteps(boxLength: number, maxTexels: number, sampledLod: number, subSteps: number): number {
  const lod = Math.max(0, Math.log2(boxLength * maxTexels));
  return Math.trunc(clamp(Math.ceil(Math.pow(2, lod - sampledLod) * 2), 1, subSteps));
}

/** frameNoise: interleaved gradient noise (Jimenez 2014); 0.5 when jitter is off (negative). */
export function vdbFrameNoise(jitter: number, pixel: Vec2): number {
  const noise = fract(52.9829189 * fract(pixel[0] * 0.06711056 + pixel[1] * 0.00583715));
  return jitter < 0 ? 0.5 : fract(noise + jitter);
}

/** frameHenyeyGreenstein: HG phase divided by the isotropic phase 1/(4π); g = 0 gives 1. */
export function vdbFrameHenyeyGreenstein(cosTheta: number, g: number): number {
  const g2 = g * g;
  return (1 - g2) / Math.pow(Math.max(1 + g2 - 2 * g * cosTheta, 1e-4), 1.5);
}

/**
 * framePhaseWeight: `phase` is (forward g, back g, back weight, Lorenz-Mie mode); dual-lobe is a mix
 * of two HG lobes; mode > 0 mixes the 'lorenz-mie' table in by the mode (his code; only modes 0 and 1
 * reach it, 'lorenz-mie-rgb' takes the per-channel branch of {@link vdbFramePhaseColor}).
 * `rayDirection` camera → sample, `lightDirection` sample → light (both unit).
 */
export function vdbFramePhaseWeight(phase: Vec4, rayDirection: Vec3, lightDirection: Vec3): number {
  const c = dot3(rayDirection, lightDirection);
  const a = vdbFrameHenyeyGreenstein(c, phase[0]);
  const b = vdbFrameHenyeyGreenstein(c, phase[1]);
  const w = a + (b - a) * phase[2];
  return phase[3] > 0 ? w + (vdbFrameMiePhase(c) - w) * phase[3] : w;
}

/** frameHenyeyGreensteinColor: HG of each channel, g of each channel. */
export function vdbFrameHenyeyGreensteinColor(cosTheta: number, g: Vec3): Vec3 {
  return [vdbFrameHenyeyGreenstein(cosTheta, g[0]), vdbFrameHenyeyGreenstein(cosTheta, g[1]), vdbFrameHenyeyGreenstein(cosTheta, g[2])];
}

/**
 * framePhaseColor: the phase of each channel. `phaseSpectrum` is (red g scale, blue g scale, flag)
 * (`getFramePhaseSpectrum`): flag 0 the same phase for all channels; 'lorenz-mie-rgb' its three
 * tables; else per-channel g = clamp(g × scale, ±0.99) of both lobes (`phaseGExponent`).
 */
export function vdbFramePhaseColor(phase: Vec4, phaseSpectrum: Vec3, rayDirection: Vec3, lightDirection: Vec3): Vec3 {
  const c = dot3(rayDirection, lightDirection);
  if (phaseSpectrum[2] < 0.5) {
    const w = vdbFramePhaseWeight(phase, rayDirection, lightDirection);
    return [w, w, w];
  }
  if (phase[3] > vdbFrameMieRgbMode - 0.5) return vdbFrameMieRgbPhase(c);
  const scale: Vec3 = [phaseSpectrum[0], 1, phaseSpectrum[1]];
  const g = (base: number) => scale.map((s) => clamp(s * base, -0.99, 0.99)) as Vec3;
  const a = vdbFrameHenyeyGreensteinColor(c, g(phase[0]));
  const b = vdbFrameHenyeyGreensteinColor(c, g(phase[1]));
  return [0, 1, 2].map((i) => a[i]! + (b[i]! - a[i]!) * phase[2]) as Vec3;
}

/**
 * framePhaseWeightScaled: the phase with every g × gScale; the Lorenz-Mie modes use HG with the table
 * asymmetry (0.8526) × gScale, mixed in by min(mode, 1).
 */
export function vdbFramePhaseWeightScaled(phase: Vec4, cosTheta: number, gScale: number): number {
  const a = vdbFrameHenyeyGreenstein(cosTheta, phase[0] * gScale);
  const b = vdbFrameHenyeyGreenstein(cosTheta, phase[1] * gScale);
  const lobes = a + (b - a) * phase[2];
  const mie = vdbFrameHenyeyGreenstein(cosTheta, gScale * vdbFrameMieAsymmetry);
  return lobes + (mie - lobes) * Math.min(phase[3], 1);
}

/** frameShadowAbsorption: the part of the extinction the albedo (Rec. 709 luminance) does not scatter. */
export function vdbFrameShadowAbsorption(extinction: number, albedo: Vec3): number {
  return extinction * (1 - clamp(albedo[0] * 0.2126 + albedo[1] * 0.7152 + albedo[2] * 0.0722, 0, 1));
}

/** frameShadowStep: (extinction, effective extinction, reduced scattering) of one march step. */
export function vdbFrameShadowStep(extinction: number, absorption: number, multipleScattering: number): Vec3 {
  const absorbed = Math.min(absorption, extinction);
  const reduced = (extinction - absorbed) * multipleScattering;
  return [extinction, Math.sqrt(absorbed * 3 * (absorbed + reduced)), reduced];
}

/** frameShadowDepth: −log of the total transmittance (1D diffusion when multipleScattering > 0). */
export function vdbFrameShadowDepth(depths: Vec3, multipleScattering: number): number {
  return multipleScattering > 0 ? Math.min(depths[0], depths[1] + Math.log(depths[2] * 0.75 + 1)) : depths[0];
}

/** Octaves off: (count 0, …). */
export const vdbFrameOctavesOff: Vec4 = [0, 0, 0, 0];

/** Most octaves (his `frameMaxScatteringOctaves`). */
export const vdbFrameMaxScatteringOctaves = 8;

/** The renderer settings {@link vdbFrameScatteringOctaves} reads (his `frameRenderer.settings`). */
export interface VDBFrameOctaveSettings {
  multipleScattering: boolean | string;
  scatteringOctaves: number;
  scatteringOctaveExtinction: number;
  scatteringOctaveContribution: number;
  scatteringOctavePhase: number;
}

const clampUnit = (value: unknown, fallback: number) => Math.min(1, Math.max(0, Number.isFinite(Number(value)) ? Number(value) : fallback));

/**
 * getFrameScatteringOctaves: (count N, extinction a, contribution b, phase c) of multipleScattering
 * 'octaves' (octave n = 1..N has extinction × aⁿ, light × bⁿ, phase g × cⁿ); (0, 0, 0, 0) otherwise.
 */
export function vdbFrameScatteringOctaves(settings: VDBFrameOctaveSettings): Vec4 {
  if (settings.multipleScattering !== "octaves") return [0, 0, 0, 0];
  const count = Math.min(vdbFrameMaxScatteringOctaves, Math.max(1, Math.round(Number(settings.scatteringOctaves) || 1)));
  return [count, clampUnit(settings.scatteringOctaveExtinction, 0.5), clampUnit(settings.scatteringOctaveContribution, 0.5), clampUnit(settings.scatteringOctavePhase, 0.5)];
}

/** frameShadowMarchDepth: the optical depth that stops a march; with octaves the last octave's σ·aᴺ depth. */
export function vdbFrameShadowMarchDepth(depths: Vec3, multipleScattering: number, octaves: Vec4 = vdbFrameOctavesOff): number {
  return octaves[0] > 0 ? depths[0] * Math.pow(octaves[1], octaves[0]) : vdbFrameShadowDepth(depths, multipleScattering);
}

/** frameShadowTransmittance: (direct, total); with octaves the total slot holds 1 / (1 + optical depth). */
export function vdbFrameShadowTransmittance(depths: Vec3, multipleScattering: number, octaves: Vec4 = vdbFrameOctavesOff): Vec2 {
  return [Math.exp(-depths[0]), octaves[0] > 0 ? 1 / (depths[0] + 1) : Math.exp(-vdbFrameShadowDepth(depths, multipleScattering))];
}

/** frameShadowOpticalDepth: the optical depth back from the octave shadow value 1 / (1 + τ). */
export function vdbFrameShadowOpticalDepth(shadowY: number): number {
  return Math.max(1 / Math.max(shadowY, 1e-6) - 1, 0);
}

/** frameOctaveLight: Σₙ₌₁..N bⁿ · phase(g cⁿ) · exp(−aⁿ τ); `phase` null is isotropic (phase 1). */
export function vdbFrameOctaveLight(octaves: Vec4, opticalDepth: number, phase: Vec4 | null, cosTheta: number): number {
  let light = 0;
  let extinctionScale = 1;
  let contribution = 1;
  let gScale = 1;
  for (let i = 0; i < vdbFrameMaxScatteringOctaves; i++) {
    if (i + 1 > octaves[0]) break;
    extinctionScale *= octaves[1];
    contribution *= octaves[2];
    gScale *= octaves[3];
    const p = phase ? vdbFramePhaseWeightScaled(phase, cosTheta, gScale) : 1;
    light += contribution * p * Math.exp(-opticalDepth * extinctionScale);
  }
  return light;
}

/** frameShadowTotal: total transmittance of isotropic light from the y shadow value; octaves: exp(−τ) + Σ bⁿ exp(−aⁿ τ). */
export function vdbFrameShadowTotal(shadowY: number, octaves: Vec4 = vdbFrameOctavesOff): number {
  if (!(octaves[0] > 0)) return shadowY;
  const tau = vdbFrameShadowOpticalDepth(shadowY);
  return Math.exp(-tau) + vdbFrameOctaveLight(octaves, tau, null, 0);
}

/**
 * frameTwoSidedDiffusion (multipleScattering 'two-sided'): 1D slab diffusion with a collimated source
 * between the light side (front reduced depth f) and the far side (back b), the lowest side mode
 * (k² = 2 (π / (f + b + 2 z₀))²), Robin boundaries with z₀ = ⅔; the diffuse plus the reduced beam
 * exp(−f), times exp(−effective front), never below the direct light.
 */
export function vdbFrameTwoSidedDiffusion(direct: number, frontReduced: number, frontEffective: number, backReduced: number): number {
  const z0 = 2 / 3;
  const slab = frontReduced + backReduced;
  const k0 = (Math.SQRT2 * Math.PI) / (slab + 2 * z0);
  // k = 1 has no particular solution of this form; a small shift keeps the result continuous (step(0.01, |1 − k0²|)).
  const k = k0 + (1 - (Math.abs(1 - k0 * k0) >= 0.01 ? 1 : 0)) * 0.01;
  const p = -3 / (1 - k * k);
  const kl = k * slab;
  const coshKl = (Math.exp(kl) + Math.exp(-kl)) * 0.5;
  const sinhKl = (Math.exp(kl) - Math.exp(-kl)) * 0.5;
  const c12 = k * -z0;
  const r1 = p * -(1 + z0);
  const c21 = coshKl + k * z0 * sinhKl;
  const c22 = sinhKl + k * z0 * coshKl;
  const r2 = -p * Math.exp(-slab) * (1 - z0);
  const det = c22 - c12 * c21;
  const c = (r1 * c22 - c12 * r2) / det;
  const d = (r2 - c21 * r1) / det;
  const kf = k * frontReduced;
  const beam = Math.exp(-frontReduced);
  const diffuse = p * beam + c * (Math.exp(kf) + Math.exp(-kf)) * 0.5 + d * (Math.exp(kf) - Math.exp(-kf)) * 0.5;
  return Math.max(direct, (diffuse + beam) * Math.exp(-frontEffective));
}

/**
 * The shadow march of the inject pass (`volumeTransmittance`), over given extinction samples per unit
 * length already multiplied by the world step and absorbance (`stepScale`). Returns (direct, total,
 * reduced depth, effective depth), stopping when the march depth exceeds 8.
 */
export function vdbFrameShadowMarch(scaledExtinctions: readonly number[], albedo: Vec3, multipleScattering: number, octaves: Vec4 = vdbFrameOctavesOff): Vec4 {
  const depths: Vec3 = [0, 0, 0];
  for (const e of scaledExtinctions) {
    const s = vdbFrameShadowStep(e, vdbFrameShadowAbsorption(e, albedo), multipleScattering);
    depths[0] += s[0];
    depths[1] += s[1];
    depths[2] += s[2];
    if (vdbFrameShadowMarchDepth(depths, multipleScattering, octaves) > 8) break;
  }
  const t = vdbFrameShadowTransmittance(depths, multipleScattering, octaves);
  return [t[0], t[1], depths[2], depths[1]];
}

/**
 * frameShadowedLight (grey): phase × direct + scattered; the scattered light is (total − direct)
 * (isotropic), or with octaves Σ bⁿ phase(g cⁿ) exp(−aⁿ τ) from the octave shadow value.
 */
export function vdbFrameShadowedLight(phase: Vec4, rayDirection: Vec3, lightDirection: Vec3, shadow: Vec2, octaves: Vec4 = vdbFrameOctavesOff): number {
  const direct = vdbFramePhaseWeight(phase, rayDirection, lightDirection) * shadow[0];
  if (octaves[0] > 0) return direct + vdbFrameOctaveLight(octaves, vdbFrameShadowOpticalDepth(shadow[1]), phase, dot3(rayDirection, lightDirection));
  return direct + (shadow[1] - shadow[0]);
}

/** frameShadowedLight of each channel (per-channel phases: `phaseSpectrum`, see {@link vdbFramePhaseColor}). */
export function vdbFrameShadowedLightColor(phase: Vec4, phaseSpectrum: Vec3, octaves: Vec4, rayDirection: Vec3, lightDirection: Vec3, shadow: Vec2): Vec3 {
  const p = vdbFramePhaseColor(phase, phaseSpectrum, rayDirection, lightDirection);
  const scattered = octaves[0] > 0 ? vdbFrameOctaveLight(octaves, vdbFrameShadowOpticalDepth(shadow[1]), phase, dot3(rayDirection, lightDirection)) : shadow[1] - shadow[0];
  return [p[0] * shadow[0] + scattered, p[1] * shadow[0] + scattered, p[2] * shadow[0] + scattered];
}

/**
 * The view-ray pre-integration of one froxel in the inject pass: `opticalDepths[i]` is
 * extinction × opacity × fade × sub-step length at sub-step i (t = segX + seg × (i + jitter) / n).
 * Returns the slice alpha and the visibility-weighted mean t (the lit position), or null when the
 * froxel is discarded.
 */
export function vdbFrameInjectSlice(opticalDepths: readonly number[], segment: Vec2, jitter = 0.5): { alpha: number; t: number } | null {
  const n = opticalDepths.length;
  const seg = segment[1] - segment[0];
  let transmittance = 1;
  let visibleWeight = 0;
  let visibleTSum = 0;
  for (let i = 0; i < n; i++) {
    const t = segment[0] + seg * ((i + jitter) / n);
    const stepT = Math.exp(-opticalDepths[i]!);
    const w = transmittance * (1 - stepT);
    visibleWeight += w;
    visibleTSum += w * t;
    transmittance *= stepT;
  }
  const alpha = 1 - transmittance;
  if (visibleWeight <= 0 || alpha <= 1e-5) return null;
  return { alpha, t: visibleTSum / visibleWeight };
}

/** One premultiplied froxel of the atlas: rgb = radiance × alpha, a = alpha. */
export type VDBFrameFroxelSample = Vec4;

/**
 * The integrate pass for one pixel (`createFragmentNode`, grey): front to back over the slices
 * [first, end) of the log range, stopping at the scene depth. `froxels[k]` is the atlas sample of
 * slice `first + k`. Returns premultiplied (radiance, alpha), or null when discarded.
 */
export function vdbFrameIntegrate(
  froxels: readonly VDBFrameFroxelSample[],
  sliceRange: Vec2,
  range: Vec2,
  depth: number,
  sceneDepth: number,
): { radiance: Vec3; alpha: number } | null {
  const radiance: Vec3 = [0, 0, 0];
  let transmittance = 1;
  let sliceStart = vdbFrameSliceDepth(range, sliceRange[0] / depth);
  for (let i = 0; i < 1024; i++) {
    const slice = sliceRange[0] + i;
    if (slice >= sliceRange[1] || sliceStart >= sceneDepth || transmittance < 0.001) break;
    const sliceEnd = vdbFrameSliceDepth(range, (slice + 1) / depth);
    const coverage = (Math.min(sliceEnd, sceneDepth) - sliceStart) / (sliceEnd - sliceStart);
    const f = froxels[i] ?? [0, 0, 0, 0];
    const a = clamp(f[3], 0, 1);
    if (a > 0) {
      const sliceAlpha = coverage < 1 ? 1 - Math.pow(1 - a, coverage) : a;
      for (let c = 0; c < 3; c++) radiance[c] = radiance[c]! + transmittance * f[c]! * (sliceAlpha / a);
      transmittance *= 1 - sliceAlpha;
    }
    sliceStart = sliceEnd;
  }
  const alpha = 1 - transmittance;
  return alpha <= 0 ? null : { radiance, alpha };
}

/** sRGB OETF (three `sRGBTransferOETF`). */
export function vdbSrgbOetf(x: number): number {
  return x < 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 0.41666) - 0.055;
}

/** sRGB EOTF (three `sRGBTransferEOTF`). */
export function vdbSrgbEotf(x: number): number {
  return x < 0.04045 ? x * 0.0773993808 : Math.pow(x * 0.9478672986 + 0.0521327014, 2.4);
}

/** The integrate pass's sRGB blend (`srgbBlending`, clamp on): linear out. */
export function vdbFrameSrgbBlend(background: Vec3, radiance: Vec3, alpha: number, clampColors = true): Vec3 {
  const enc = (x: number) => (clampColors ? clamp(vdbSrgbOetf(x), 0, 1) : vdbSrgbOetf(x));
  return [0, 1, 2].map((c) => {
    const bg = enc(background[c]!);
    const vol = enc(radiance[c]! / alpha);
    return vdbSrgbEotf(bg + (vol - bg) * alpha);
  }) as Vec3;
}
