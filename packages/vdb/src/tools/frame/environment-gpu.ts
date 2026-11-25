// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameEnvironment.js (setTexels, getDirectionalLights, getShLights: reused through
//   VDBFrameEnvironmentLights in environment.ts), src/three-openvdb/frame/webgpu/FrameEnvironmentMapMaterial.js
//   (the 64 × 32 map of PMREM roughness 0.4 the lights are made from).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes (K6.2b, design §3.5, Q3): the host's equirect or cube texture is reduced by our kernels instead of three's
// PMREM: an exact-area (equirect) or supersampled (cube) 128 × 64 work map, a GGX roughness-0.4 convolution standing
// in for PMREM roughness 0.4, and SH L2 irradiance (Ramamoorthi–Hanrahan) for the 'outward' term. CPU twins of the
// core kernels (f64). Divergences: DIVERGES.md (K6.2b rows).

/**
 * The environment reduction of the core frame renderer (K6.2b) as f64 CPU twins of its WGSL kernels:
 *
 * - `vdbFrameEnvironmentAreaTexel` / `vdbFrameEnvironmentCubeTexel`: one texel of the 128 × 64 work map (map space,
 *   row 0 the zenith) from the host texture.
 * - `vdbFrameEnvironmentBlurTexel`: one texel of his 64 × 32 map (world space, his directions) as the GGX
 *   (roughness 0.4) convolution of the work map.
 * - `vdbFrameEnvironmentIrradianceSh`: the SH9 radiance projection of the work map in world space (three's basis),
 *   which `vdbFrameEnvironmentIrradiance` turns into irradiance (three's getShIrradianceAt).
 * - `vdbFrameEnvironmentReduce`: the environment buffer (lights and SH) from the read-back maps, through
 *   {@link VDBFrameEnvironmentLights} (his extraction, unchanged).
 *
 * Map convention of the host texture (equirect): u = atan2(m.z, m.x) / 2π + ½ along x, row 0 is +Y (the zenith),
 * m = R · d the map direction of a world direction d (R: a 3 × 3 column-major rotation, or a rotation with a
 * reflection: three's flipY equirect is diag(1, −1, 1) · R). Cube: WebGPU's face order and orientation.
 */

import { VDBFrameEnvironmentLights, frameEnvironmentMapHeight, frameEnvironmentMapWidth, frameEnvironmentModes, frameMaxEnvironmentLights, vdbFrameEnvironmentMapDirection } from "./environment.js";
import type { Vec3 } from "./index.js";

/** Work map size (design §3.3 envPrefilter). */
export const VDB_FRAME_ENV_WORK_WIDTH = 128;
export const VDB_FRAME_ENV_WORK_HEIGHT = 64;
/** GGX roughness of the blur (his PMREM roughness 0.4, `frameEnvironmentMapRoughness`). */
export const VDB_FRAME_ENV_ROUGHNESS = 0.4;
/** Most source texels per axis one work texel reads (larger maps are strided). */
export const VDB_FRAME_ENV_MAX_SPAN = 256;
/** Near-ties of the 'directional' extraction (brightest texel, nearest seed) go to the first candidate (f32 vs f64). */
export const VDB_FRAME_ENV_TIE = 1e-5;
/** Most cube sub-samples per axis of one work texel. */
export const VDB_FRAME_ENV_MAX_SUPERSAMPLE = 16;

/** A 3 × 3 column-major matrix (world → map). */
export type VDBFrameEnvRotation = ArrayLike<number>;

/** The identity map rotation. */
export const VDB_FRAME_ENV_IDENTITY: readonly number[] = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** m = R · d (R column-major). */
export function vdbFrameEnvToMap(r: VDBFrameEnvRotation, d: Vec3): Vec3 {
  return [r[0]! * d[0] + r[3]! * d[1] + r[6]! * d[2], r[1]! * d[0] + r[4]! * d[1] + r[7]! * d[2], r[2]! * d[0] + r[5]! * d[1] + r[8]! * d[2]];
}

/** d = Rᵀ · m (R orthogonal: its inverse). */
export function vdbFrameEnvToWorld(r: VDBFrameEnvRotation, m: Vec3): Vec3 {
  return [r[0]! * m[0] + r[1]! * m[1] + r[2]! * m[2], r[3]! * m[0] + r[4]! * m[1] + r[5]! * m[2], r[6]! * m[0] + r[7]! * m[1] + r[8]! * m[2]];
}

/** The map direction of an equirect point: u along x (u = atan2(z, x) / 2π + ½), v from the top row (+Y). */
export function vdbFrameEquirectDirection(u: number, v: number): Vec3 {
  const theta = v * Math.PI;
  const phi = (u - 0.5) * 2 * Math.PI;
  return [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
}

/** Solid angle of work map row `y` (exact: Δφ · (cos θ₀ − cos θ₁)). */
export function vdbFrameEnvWorkSolidAngle(y: number, width = VDB_FRAME_ENV_WORK_WIDTH, height = VDB_FRAME_ENV_WORK_HEIGHT): number {
  return ((2 * Math.PI) / width) * (Math.cos((y * Math.PI) / height) - Math.cos(((y + 1) * Math.PI) / height));
}

/** An RGBA float image, rows from the top (row 0 is the first `width · 4` floats). */
export interface VDBFrameEnvImage {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/**
 * One work map texel from an equirect source: the solid-angle mean of the source over the texel's box (exact
 * overlaps; sources wider than {@link VDB_FRAME_ENV_MAX_SPAN} texels per work texel are strided).
 */
export function vdbFrameEnvironmentAreaTexel(src: VDBFrameEnvImage, x: number, y: number, workWidth = VDB_FRAME_ENV_WORK_WIDTH, workHeight = VDB_FRAME_ENV_WORK_HEIGHT): [number, number, number, number] {
  const { width: W, height: H, data } = src;
  const u0 = x / workWidth;
  const u1 = (x + 1) / workWidth;
  const v0 = y / workHeight;
  const v1 = (y + 1) / workHeight;
  const i0 = Math.floor(u0 * W);
  const i1 = Math.min(W, Math.ceil(u1 * W));
  const j0 = Math.floor(v0 * H);
  const j1 = Math.min(H, Math.ceil(v1 * H));
  const si = Math.max(1, Math.ceil((i1 - i0) / VDB_FRAME_ENV_MAX_SPAN));
  const sj = Math.max(1, Math.ceil((j1 - j0) / VDB_FRAME_ENV_MAX_SPAN));
  const sum = [0, 0, 0];
  let weight = 0;
  for (let j = j0; j < j1; j += sj) {
    const a = Math.max(j / H, v0) * Math.PI;
    const b = Math.min((j + sj) / H, v1) * Math.PI;
    if (!(b > a)) continue;
    const wv = Math.cos(a) - Math.cos(b);
    for (let i = i0; i < i1; i += si) {
      const wu = Math.min((i + si) / W, u1) - Math.max(i / W, u0);
      if (!(wu > 0)) continue;
      const w = wu * wv;
      const at = (j * W + i) * 4;
      for (let c = 0; c < 3; c++) sum[c] = sum[c]! + data[at + c]! * w;
      weight += w;
    }
  }
  return weight > 0 ? [sum[0]! / weight, sum[1]! / weight, sum[2]! / weight, 1] : [0, 0, 0, 1];
}

/** The face and texel of a cube direction (WebGPU's face order +X, −X, +Y, −Y, +Z, −Z and orientation). */
export function vdbFrameCubeTexel(m: Vec3, size: number): { face: number; x: number; y: number } {
  const ax = Math.abs(m[0]);
  const ay = Math.abs(m[1]);
  const az = Math.abs(m[2]);
  let face: number;
  let ma: number;
  let sc: number;
  let tc: number;
  if (ax >= ay && ax >= az) {
    face = m[0] >= 0 ? 0 : 1;
    ma = ax;
    sc = m[0] >= 0 ? -m[2] : m[2];
    tc = -m[1];
  } else if (ay >= az) {
    face = m[1] >= 0 ? 2 : 3;
    ma = ay;
    sc = m[0];
    tc = m[1] >= 0 ? m[2] : -m[2];
  } else {
    face = m[2] >= 0 ? 4 : 5;
    ma = az;
    sc = m[2] >= 0 ? m[0] : -m[0];
    tc = -m[1];
  }
  const s = (sc / ma + 1) * 0.5;
  const t = (tc / ma + 1) * 0.5;
  return { face, x: Math.min(Math.max(Math.floor(s * size), 0), size - 1), y: Math.min(Math.max(Math.floor(t * size), 0), size - 1) };
}

/** Cube sub-samples per axis of a work texel for a face of `size` texels (about two per cube texel). */
export function vdbFrameEnvCubeSupersample(size: number, workHeight = VDB_FRAME_ENV_WORK_HEIGHT): number {
  return Math.min(VDB_FRAME_ENV_MAX_SUPERSAMPLE, Math.max(1, Math.ceil((4 * size) / workHeight)));
}

/**
 * One work map texel from a cube source (`faces[f]` an RGBA image of `size`²): k × k sub-samples of the texel's box,
 * each the nearest cube texel, weighted by sin θ (solid angle).
 */
export function vdbFrameEnvironmentCubeTexel(faces: readonly VDBFrameEnvImage[], size: number, x: number, y: number, workWidth = VDB_FRAME_ENV_WORK_WIDTH, workHeight = VDB_FRAME_ENV_WORK_HEIGHT): [number, number, number, number] {
  const k = vdbFrameEnvCubeSupersample(size, workHeight);
  const sum = [0, 0, 0];
  let weight = 0;
  for (let b = 0; b < k; b++) {
    const v = (y + (b + 0.5) / k) / workHeight;
    const w = Math.sin(v * Math.PI);
    for (let a = 0; a < k; a++) {
      const u = (x + (a + 0.5) / k) / workWidth;
      const t = vdbFrameCubeTexel(vdbFrameEquirectDirection(u, v), size);
      const f = faces[t.face]!;
      const at = (t.y * size + t.x) * 4;
      for (let c = 0; c < 3; c++) sum[c] = sum[c]! + f.data[at + c]! * w;
      weight += w;
    }
  }
  return weight > 0 ? [sum[0]! / weight, sum[1]! / weight, sum[2]! / weight, 1] : [0, 0, 0, 1];
}

/** The GGX weight of input direction cosine `c` around the output direction (n = v = r; D(h) · n·l). */
export function vdbFrameEnvGgxWeight(c: number, roughness = VDB_FRAME_ENV_ROUGHNESS): number {
  if (!(c > 0)) return 0;
  const a2 = roughness * roughness * roughness * roughness;
  const nh2 = (1 + c) * 0.5;
  const d = nh2 * (a2 - 1) + 1;
  return (a2 / (Math.PI * d * d)) * c;
}

/**
 * One texel (x, y) of his 64 × 32 map: the GGX (roughness 0.4) weighted mean of the work map around his direction
 * of the texel centre (`vdbFrameEnvironmentMapDirection`), the work map in map space (`rotation` world → map).
 */
export function vdbFrameEnvironmentBlurTexel(work: VDBFrameEnvImage, rotation: VDBFrameEnvRotation, x: number, y: number, roughness = VDB_FRAME_ENV_ROUGHNESS): [number, number, number, number] {
  const d = vdbFrameEnvironmentMapDirection((x + 0.5) / frameEnvironmentMapWidth, (y + 0.5) / frameEnvironmentMapHeight);
  const m = vdbFrameEnvToMap(rotation, d);
  const sum = [0, 0, 0];
  let weight = 0;
  for (let j = 0; j < work.height; j++) {
    const omega = vdbFrameEnvWorkSolidAngle(j, work.width, work.height);
    for (let i = 0; i < work.width; i++) {
      const mj = vdbFrameEquirectDirection((i + 0.5) / work.width, (j + 0.5) / work.height);
      const w = vdbFrameEnvGgxWeight(m[0] * mj[0] + m[1] * mj[1] + m[2] * mj[2], roughness) * omega;
      if (!(w > 0)) continue;
      const at = (j * work.width + i) * 4;
      for (let c = 0; c < 3; c++) sum[c] = sum[c]! + work.data[at + c]! * w;
      weight += w;
    }
  }
  return weight > 0 ? [sum[0]! / weight, sum[1]! / weight, sum[2]! / weight, 1] : [0, 0, 0, 1];
}

/** three's SH9 basis (SphericalHarmonics3.getBasisAt). */
export function vdbFrameSh9Basis(d: Vec3): number[] {
  const [x, y, z] = d;
  return [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)];
}

/** SH9 radiance coefficients (rgb each, three's basis) of the work map in world space: Σ L · Y(Rᵀ m) · Ω. */
export function vdbFrameEnvironmentIrradianceSh(work: VDBFrameEnvImage, rotation: VDBFrameEnvRotation): Vec3[] {
  const out: Vec3[] = Array.from({ length: 9 }, () => [0, 0, 0] as Vec3);
  for (let j = 0; j < work.height; j++) {
    const omega = vdbFrameEnvWorkSolidAngle(j, work.width, work.height);
    for (let i = 0; i < work.width; i++) {
      const d = vdbFrameEnvToWorld(rotation, vdbFrameEquirectDirection((i + 0.5) / work.width, (j + 0.5) / work.height));
      const basis = vdbFrameSh9Basis(d);
      const at = (j * work.width + i) * 4;
      for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) out[k]![c] = out[k]![c]! + Math.max(0, work.data[at + c]!) * basis[k]! * omega;
    }
  }
  return out;
}

/** Irradiance at a unit normal from SH9 radiance coefficients (three's getShIrradianceAt, Ramamoorthi–Hanrahan). */
export function vdbFrameEnvironmentIrradiance(sh: readonly Vec3[], n: Vec3): Vec3 {
  const [x, y, z] = n;
  const w = [0.886227, 2 * 0.511664 * y, 2 * 0.511664 * z, 2 * 0.511664 * x, 2 * 0.429043 * x * y, 2 * 0.429043 * y * z, 0.743125 * z * z - 0.247708, 2 * 0.429043 * x * z, 0.429043 * (x * x - y * y)];
  const out: Vec3 = [0, 0, 0];
  for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) out[c] = out[c]! + sh[k]![c]! * w[k]!;
  return [Math.max(out[0], 0), Math.max(out[1], 0), Math.max(out[2], 0)];
}

/** Environment modes as the buffer codes them. */
export const VDB_FRAME_ENV_MODE = { outward: 0, directional: 1, sphericalHarmonics: 2 } as const;

/** The environment buffer's content (`VdbFrameEnvironment`), as the twin computes it. */
export interface VDBFrameEnvironmentGpuResult {
  /** An environment with finite, positive energy. */
  valid: boolean;
  mode: number;
  count: number;
  directions: Vec3[];
  colors: Vec3[];
  /** SH9 radiance coefficients × intensity (irradiance through {@link vdbFrameEnvironmentIrradiance}). */
  sh: Vec3[];
}

/** The settings the reduction reads. */
export interface VDBFrameEnvironmentReduceSettings {
  environmentLighting: string;
  environmentSamples: number;
  environmentShOrder: number;
  /** Diffusion multiple scattering: the 'directional' seeds come in opposite pairs. */
  pairs: boolean;
  intensity: number;
}

/**
 * The `envReduce` kernel's output from the read-back 64 × 32 map (`blurred`, his directions, rows from v = 0) and the
 * work map SH (`sh`, before intensity): his lights ('directional' cells sorted by power, or 'spherical-harmonics'
 * quadrature lights) through {@link VDBFrameEnvironmentLights}, colours and SH × intensity.
 */
export function vdbFrameEnvironmentReduce(blurred: ArrayLike<number>, sh: readonly Vec3[], s: VDBFrameEnvironmentReduceSettings): VDBFrameEnvironmentGpuResult {
  const w = frameEnvironmentMapWidth;
  const h = frameEnvironmentMapHeight;
  const pixels = new Float64Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) pixels[i * 4 + c] = blurred[i * 4 + c]!;
    pixels[i * 4 + 3] = (Math.floor(i / w) + 0.5) / h;
  }
  const lights = new VDBFrameEnvironmentLights(() => Promise.resolve([]));
  const valid = lights.setTexels(pixels);
  const mode = s.environmentLighting === frameEnvironmentModes.directional ? VDB_FRAME_ENV_MODE.directional : s.environmentLighting === frameEnvironmentModes.sphericalHarmonics ? VDB_FRAME_ENV_MODE.sphericalHarmonics : VDB_FRAME_ENV_MODE.outward;
  const out: VDBFrameEnvironmentGpuResult = { valid, mode, count: 0, directions: [], colors: [], sh: sh.map((c) => c.map((v) => (valid ? v * s.intensity : 0)) as Vec3) };
  if (!valid || mode === VDB_FRAME_ENV_MODE.outward) return out;
  const list = mode === VDB_FRAME_ENV_MODE.directional ? lights.getDirectionalLights(s.environmentSamples, s.pairs, VDB_FRAME_ENV_TIE) : lights.getShLights(s.environmentShOrder);
  out.count = Math.min(list.length, frameMaxEnvironmentLights);
  out.directions = list.slice(0, out.count).map((l) => [...l.direction] as Vec3);
  out.colors = list.slice(0, out.count).map((l) => l.color.map((v) => v * s.intensity) as Vec3);
  return out;
}

/** The renderer's environment flags of 'frame' lighting (`VdbFrameUniforms.environment`, `lighting.z`, `lighting.w`). */
export interface VDBFrameEnvironmentFlags {
  /** The shadow grid is in use. */
  useShadows: boolean;
  /** The dominant 'directional' light has a shadow slot. */
  slot: boolean;
  /** The environment grid holds this frame's lights. */
  grid: boolean;
  /** Diffusion multiple scattering. */
  diffusion: boolean;
}

/**
 * The environment term of his density pass in the core ('frame' lighting, K6.2b): the irradiance E a volume adds as
 * albedo · E / π at its visible position. With environment-grid lights the grid (`gridAt`, null outside: then the
 * unshadowed sum of those lights); with only the slot light nothing (the lighting adds it); else 'outward' SH
 * irradiance along `outward`, shadowed between the sky down and up totals (diffusion: the outward and inward halves).
 */
export function vdbFrameEnvironmentDensityTerm(env: VDBFrameEnvironmentGpuResult, flags: VDBFrameEnvironmentFlags, outward: Vec3, sky: Vec3, skyTotals: [number, number], gridAt: () => Vec3 | null): Vec3 {
  if (!env.valid) return [0, 0, 0];
  const first = flags.slot ? 1 : 0;
  if (flags.useShadows && flags.grid && env.count > first) {
    const inside = gridAt();
    if (inside) return inside;
    const sum: Vec3 = [0, 0, 0];
    for (let i = first; i < env.count; i++) for (let c = 0; c < 3; c++) sum[c] = sum[c]! + env.colors[i]![c]!;
    return [Math.max(sum[0], 0), Math.max(sum[1], 0), Math.max(sum[2], 0)];
  }
  if (flags.useShadows && flags.slot && env.count > 0) return [0, 0, 0];
  const blend = (outward[0] * sky[0] + outward[1] * sky[1] + outward[2] * sky[2]) * 0.5 + 0.5;
  const [up, down] = skyTotals;
  const shadow = down + (up - down) * blend;
  const e = vdbFrameEnvironmentIrradiance(env.sh, outward);
  if (!flags.diffusion) return [e[0] * shadow, e[1] * shadow, e[2] * shadow];
  const inward = up + (down - up) * blend;
  const back = vdbFrameEnvironmentIrradiance(env.sh, [-outward[0], -outward[1], -outward[2]]);
  return [0, 1, 2].map((c) => (e[c]! * shadow + back[c]! * inward) * 0.5) as Vec3;
}
