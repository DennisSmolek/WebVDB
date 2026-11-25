/**
 * K6.R: the scene of the reference renderer (design K6-core-renderer.md §7.4): plain data shared by the GPU kernel
 * (`core/reference/`) and its CPU twin ({@link vdbReferenceRenderCpu}).
 *
 * The medium is the renderer's (three/frame/inject-material.ts, volume-source.ts `sampleExtinction`): each volume is
 * a local unit box [−0.5, 0.5]³ placed in the world by `localToWorld`, with extinction per world unit
 * σt(x) = `sigma` · value(uvw), value = density × mask, zero at or below `cutoff`, uvw = local + 0.5 (no wrap or
 * offset). The renderer's `sigma` is `densityScale · extinction / worldScale` ({@link vdbReferenceSigma}). Volumes
 * that overlap add their extinction (co-located media): a collision picks volume i with probability σᵢ / σ.
 * Scattering albedo: `albedo` × the source's albedo channel. Emission is the renderer's source term σt · Le with
 * Le = `emission.color` · (T / max(T₁, 1))⁴ · `intensity`, T = mix(T₀, T₁, emission value), 0 when T ≤ 0.
 *
 * Units are radiometric: directional lights carry irradiance, point and spot lights intensity (with no distance
 * falloff unless `inverseSquare`: the renderer's point and spot lights have none), the environment carries radiance.
 * {@link vdbReferenceLightsFromFrame} maps the renderer's light inputs (design §2.3 `VDBFrameLight`, three's
 * `color × intensity`) into these units.
 */

export type VDBRefVec3 = [number, number, number];

/**
 * Henyey-Greenstein (`backWeight` 0) or dual-lobe HG: (1 − w)·HG(g) + w·HG(backG); or, with `table`, a tabulated
 * phase (Lorenz-Mie, Mie RGB: {@link vdbReferenceMiePhaseTable}) that replaces the lobes.
 */
export interface VDBReferencePhase {
  /** Forward lobe asymmetry, (−1, 1). */
  g: number;
  /** Back lobe asymmetry (dual-lobe). Default 0. */
  backG?: number;
  /** Weight of the back lobe, [0, 1]. Default 0 (plain HG). */
  backWeight?: number;
  /** K6.R1: a tabulated phase per channel; `g`, `backG`, `backWeight` are then ignored. */
  table?: VDBReferencePhaseTable;
}

/**
 * K6.R1: a phase function per RGB channel tabulated as phase / isotropic (his Lorenz-Mie texel layout): `size`
 * angles θ = π u², u = i / (size − 1), read with linear blending in u (`vdbFrameMieTexel`). `cdf` is the sampling
 * table ({@link vdbReferencePhaseTableCdf}): sampling picks u from the piecewise-constant marginal of the channel
 * mean and weights each channel by its phase over that density, so the estimate is unbiased per channel.
 */
export interface VDBReferencePhaseTable {
  size: number;
  /** RGB per angle (`size` × 3). */
  values: Float32Array;
  /** Cumulative distribution of u at the `size` table angles (0 … 1). */
  cdf: Float32Array;
}

/** K6.R1: an emission colour table over log temperature (the renderer's blackbody LUT, `vdbReferenceBlackbodyTable`). */
export interface VDBReferenceColorTable {
  size: number;
  /** RGB per column (`size` × 3); column i at T = minTemperature · (max/min)^(i / (size − 1)). */
  values: Float32Array;
  minTemperature: number;
  maxTemperature: number;
}

/**
 * K6.R1: a per-cell bound of the density value (density × mask, before σ) over the volume's unit box: cell
 * (i, j, k) covers uvw ∈ [i/nx, (i+1)/nx] × … (x fastest). Tracking uses σ · bound of the cell the ray is in
 * (piecewise-constant majorants: empty and thin cells cost few null collisions). Built from the sparse pool's node
 * stats ({@link vdbReferenceMajorantGridSparse}) or the dense texels ({@link vdbReferenceMajorantGridDense}).
 */
export interface VDBReferenceMajorantGrid {
  dims: [number, number, number];
  values: Float32Array;
}

/** The renderer's emission model (volume-source.ts `sampleEmission`), white unless `color` is given. */
export interface VDBReferenceEmission {
  /** Colour of the emission (the renderer's `emissionColor` hook evaluated as a constant). Default white. */
  color?: VDBRefVec3;
  /** `emissiveIntensity`. Default 1. */
  intensity?: number;
  /** `temperatureRange` in kelvin. Default [0, 6500]. */
  temperatureRange?: [number, number];
  /** K6.R1: the colour hook as a table over temperature (blackbody), multiplied with `color`. Default white. */
  colorTable?: VDBReferenceColorTable;
}

/** One volume of the reference scene. */
export interface VDBReferenceVolume {
  /** Column-major 4x4: local unit box [−0.5, 0.5]³ → world. */
  localToWorld: ArrayLike<number>;
  /** Extinction per world unit per unit density value ({@link vdbReferenceSigma}). */
  sigma: number;
  /**
   * Bound of the density value (density × mask) inside the volume: the tracking majorant is `sigma · majorant`.
   * Default 1. A smaller bound than the data's maximum biases delta tracking (the CPU twin counts violations).
   */
  majorant?: number;
  /** `densityCutoff`: values at or below it read 0. Default 0. */
  cutoff?: number;
  /** Base colour (scattering albedo), multiplied by the source's albedo channel. Default (1, 1, 1). */
  albedo?: VDBRefVec3;
  /** Phase function. Default isotropic (g = 0). */
  phase?: VDBReferencePhase;
  /** Emission (read only when the source has an emission channel: `emits`). */
  emission?: VDBReferenceEmission;
  /** The source has an emission channel (the GPU source compiles `vdbSrcEmission` to 0 without one). Default false. */
  emits?: boolean;
  /** `'box'` (default): the unit box. `'sphere'`: the ball inscribed in it (analytic tests). */
  shape?: "box" | "sphere";
  /** The volume's slot in its source (`VDBVolumeSlot.v`; GPU only). Default 0. */
  v?: number;
  /** K6.R1: per-cell majorants (replace `majorant` inside the box). */
  majorantGrid?: VDBReferenceMajorantGrid;
}

/** A directional light: `direction` points toward the light (world, any length); `irradiance` in W/m². */
export interface VDBReferenceDirectionalLight {
  kind: "directional";
  direction: VDBRefVec3;
  irradiance: VDBRefVec3;
}

/**
 * K6.R1: three's `getDistanceAttenuation` (the renderer's 'frame' lighting): 1 / max(d^decay, 0.01), times
 * (clamp(1 − (d / cutoff)⁴))² when `cutoff` > 0.
 */
export interface VDBReferenceFalloff {
  decay: number;
  /** three's `light.distance` (0: no window). */
  cutoff?: number;
}

/**
 * A point light. Without `inverseSquare` or `falloff` the irradiance is `intensity` at any distance (the renderer's
 * 'volume' lighting); `inverseSquare` is 1/d², `falloff` three's attenuation ('frame' lighting).
 */
export interface VDBReferencePointLight {
  kind: "point";
  position: VDBRefVec3;
  intensity: VDBRefVec3;
  inverseSquare?: boolean;
  falloff?: VDBReferenceFalloff;
}

/** A spot light: `direction` from the light toward its target; smoothstep(cosOuter, cosInner, cos) attenuation. */
export interface VDBReferenceSpotLight {
  kind: "spot";
  position: VDBRefVec3;
  direction: VDBRefVec3;
  intensity: VDBRefVec3;
  cosOuter: number;
  cosInner: number;
  inverseSquare?: boolean;
  falloff?: VDBReferenceFalloff;
}

export type VDBReferenceLight = VDBReferenceDirectionalLight | VDBReferencePointLight | VDBReferenceSpotLight;

/** A hemisphere environment: radiance `sky` above the plane ⟂ `up`, `ground` below. */
export interface VDBReferenceHemisphere {
  sky: VDBRefVec3;
  ground: VDBRefVec3;
  up: VDBRefVec3;
}

/** An equirect radiance map (three's equirect mapping), read nearest (texel lookups, no filtering). */
export interface VDBReferenceEquirect {
  width: number;
  height: number;
  /** RGBA (alpha ignored) floats, row 0 at the top (v = 1: up), x fastest. */
  data: Float32Array;
  /** Radiance multiplier (`environmentIntensity`). Default 1. */
  intensity?: number;
  /** Row-major 3x3 world → map direction rotation. Default identity. */
  rotation?: ArrayLike<number>;
}

/** Radiance arriving from infinity: gathered when a path leaves the media after at least one scattering event. */
export interface VDBReferenceEnvironment {
  /** Uniform radiance from every direction. */
  uniform?: VDBRefVec3;
  /** Hemisphere radiances (summed). At most 4. */
  hemispheres?: VDBReferenceHemisphere[];
  /** An equirect map (summed). */
  equirect?: VDBReferenceEquirect;
  /**
   * K6.R1: a light probe: radiance as 9 RGB spherical-harmonic coefficients in three's `SphericalHarmonics3` order and
   * basis (27 numbers, coefficient-major), L(d) = Σ cᵢ Yᵢ(d), not clamped (the renderer's irradiance is the cosine
   * convolution of the same function). Summed.
   */
  sh?: ArrayLike<number>;
}

/** The camera: rays from the view origin through each pixel (perspective) or parallel (orthographic). */
export interface VDBReferenceCamera {
  /** Column-major 4x4 camera → world (three's `camera.matrixWorld`). */
  cameraToWorld: ArrayLike<number>;
  /** Column-major 4x4 inverse projection (three's `projectionMatrixInverse`; WebGL or WebGPU depth range). */
  projectionInverse: ArrayLike<number>;
  /** Orthographic projection: parallel rays from the ndc z = −1 plane. Default false (rays from the view origin). */
  orthographic?: boolean;
}

/** The render settings shared by the GPU kernel and the twin. */
export interface VDBReferenceSettings {
  width: number;
  height: number;
  /** Scattering events per path (1 = single scattering). Default 256; paths longer than this are cut (bias). */
  maxBounces?: number;
  /** Russian roulette on the path throughput from this depth on (survival = min(max(β), 1)). Default 4. */
  rrStart?: number;
  /** Seed of every random stream. Default 1. */
  seed?: number;
  /** Samples per pixel per batch (one dispatch). Default 1. */
  spp?: number;
  /** `'center'` (default): every sample through the pixel centre (as the froxel renderer's lookups); `'box'`: uniform in the pixel. */
  pixelFilter?: "center" | "box";
}

/** The whole reference scene. */
export interface VDBReferenceScene {
  camera: VDBReferenceCamera;
  volumes: VDBReferenceVolume[];
  lights?: VDBReferenceLight[];
  environment?: VDBReferenceEnvironment;
}

/** Most volumes and lights one GPU kernel takes (uniform arrays). */
export const VDB_REFERENCE_MAX_VOLUMES = 8;
export const VDB_REFERENCE_MAX_LIGHTS = 16;
export const VDB_REFERENCE_MAX_HEMISPHERES = 4;
/** Most tracking steps of one ray (a guard: an unbounded loop must never hang a GPU). */
export const VDB_REFERENCE_MAX_STEPS = 1 << 20;
/** Ratio tracking plays Russian roulette (survive 1/2, weight × 2) once its transmittance drops below this. */
export const VDB_REFERENCE_RATIO_RR = 0.1;

/**
 * The renderer's punctual light units → irradiance. 'volume' lighting adds `color · phasê · T` to the irradiance and
 * divides by π (inject-material.ts), with phasê the phase over the isotropic phase 1/(4π): the scattered radiance is
 * albedo · color · 4π·p · T / π = albedo · p · (4·color) · T. A directional light of three colour C is irradiance 4C
 * in the reference (three's Lambert surfaces see C: the renderer's volumes are 4× brighter under punctual lights
 * than its own ambient convention; a convention, not an approximation, so the reference follows it).
 */
export const VDB_REFERENCE_DIRECT_LIGHT_SCALE = 4;

/** The resolved settings with defaults. */
export function vdbReferenceSettings(s: VDBReferenceSettings): Required<VDBReferenceSettings> {
  return {
    width: s.width,
    height: s.height,
    maxBounces: s.maxBounces ?? 256,
    rrStart: s.rrStart ?? 4,
    seed: (s.seed ?? 1) >>> 0,
    spp: Math.max(1, Math.floor(s.spp ?? 1)),
    pixelFilter: s.pixelFilter ?? "center",
  };
}

/**
 * The renderer's extinction per world unit per density value: `densityScale` (÷ the mean non-zero density when
 * `normalizeDensity`) × `extinction` ÷ `worldScale`, the largest axis scale of the unit-box → world matrix
 * (three/frame/renderer.ts `setVolumeUniforms`).
 */
export function vdbReferenceSigma(o: { localToWorld: ArrayLike<number>; extinction: number; densityScale?: number; normalizeDensity?: boolean; meanNonZero?: number }): number {
  const m = o.localToWorld;
  const scale = Math.sqrt(Math.max(m[0]! ** 2 + m[1]! ** 2 + m[2]! ** 2, m[4]! ** 2 + m[5]! ** 2 + m[6]! ** 2, m[8]! ** 2 + m[9]! ** 2 + m[10]! ** 2));
  let density = o.densityScale ?? 1;
  if ((o.normalizeDensity ?? true) && (o.meanNonZero ?? 0) > 0) density /= o.meanNonZero!;
  return (density * o.extinction) / scale;
}

/**
 * A renderer light (design K6-core-renderer.md §2.3 `VDBFrameLight`, without ids, masks or shadow requests). Point and
 * spot lights carry three's `distance` (0: no window) and `decay` (default 2), used by 'frame' lighting only.
 */
export type VDBReferenceFrameLight =
  | { kind: "ambient"; color: VDBRefVec3 }
  | { kind: "hemisphere"; sky: VDBRefVec3; ground: VDBRefVec3; up: VDBRefVec3 }
  | { kind: "directional"; direction: VDBRefVec3; color: VDBRefVec3 }
  | { kind: "point"; position: VDBRefVec3; color: VDBRefVec3; distance?: number; decay?: number }
  | { kind: "spot"; position: VDBRefVec3; direction: VDBRefVec3; color: VDBRefVec3; cosAngle: number; cosPenumbra: number; distance?: number; decay?: number }
  | { kind: "probe"; sh: ArrayLike<number> };

/** Options of {@link vdbReferenceLightsFromFrame}. */
export interface VDBReferenceLightMapping {
  /**
   * The renderer's lighting mode the reference stands for (K6.R1). 'volume' (default): point and spot lights have no
   * distance falloff (inject-material.ts, core gather; V84). 'frame': three's `getDistanceAttenuation` with the
   * light's `distance` and `decay` (lighting-material.ts, core lighting.ts `vdbFrameDistanceAttenuation`).
   */
  lighting?: "volume" | "frame";
}

/**
 * The renderer's lights in reference units: punctual lights × {@link VDB_REFERENCE_DIRECT_LIGHT_SCALE}; ambient C is
 * a uniform radiance C/π and a hemisphere light's sky / ground are radiances S/π, G/π (three: an up-facing Lambert
 * surface gets irradiance C, and the renderer's isotropic light gives albedo · C / π); a light probe's SH are radiance
 * coefficients already (three's `shGetIrradianceAt` is their cosine convolution). `direction` of a directional light
 * points toward the light (his `position − target`), of a spot from the light toward its target. Point and spot
 * falloff follows the lighting mode ({@link VDBReferenceLightMapping}).
 */
export function vdbReferenceLightsFromFrame(lights: readonly VDBReferenceFrameLight[], o: VDBReferenceLightMapping = {}): { lights: VDBReferenceLight[]; environment: VDBReferenceEnvironment } {
  const k = VDB_REFERENCE_DIRECT_LIGHT_SCALE;
  const frame = (o.lighting ?? "volume") === "frame";
  const scale = (c: VDBRefVec3, s: number) => c.map((x) => x * s) as VDBRefVec3;
  const out: VDBReferenceLight[] = [];
  const environment: VDBReferenceEnvironment = {};
  for (const l of lights) {
    if (l.kind === "ambient") {
      const u = environment.uniform ?? [0, 0, 0];
      environment.uniform = u.map((x, i) => x + l.color[i]! / Math.PI) as VDBRefVec3;
    } else if (l.kind === "hemisphere") {
      (environment.hemispheres ??= []).push({ sky: scale(l.sky, 1 / Math.PI), ground: scale(l.ground, 1 / Math.PI), up: l.up });
    } else if (l.kind === "probe") {
      if (l.sh.length !== 27) throw new RangeError(`vdbReferenceLightsFromFrame: a probe needs 27 SH numbers, got ${l.sh.length}`);
      environment.sh = Array.from({ length: 27 }, (_, i) => (environment.sh?.[i] ?? 0) + l.sh[i]!);
    } else if (l.kind === "directional") {
      out.push({ kind: "directional", direction: l.direction, irradiance: scale(l.color, k) });
    } else {
      const falloff = frame ? { falloff: { decay: l.decay ?? 2, cutoff: l.distance ?? 0 } } : {};
      if (l.kind === "point") out.push({ kind: "point", position: l.position, intensity: scale(l.color, k), ...falloff });
      else out.push({ kind: "spot", position: l.position, direction: l.direction, intensity: scale(l.color, k), cosOuter: l.cosAngle, cosInner: l.cosPenumbra, ...falloff });
    }
  }
  return { lights: out, environment };
}

/** three's `getDistanceAttenuation` (Frostbite window; the renderer's 'frame' point and spot lights). */
export function vdbRefDistanceAttenuation(distance: number, falloff: VDBReferenceFalloff): number {
  const f = 1 / Math.max(Math.pow(distance, falloff.decay), 0.01);
  const cutoff = falloff.cutoff ?? 0;
  if (cutoff > 0) {
    const x = Math.min(Math.max(1 - Math.pow(distance / cutoff, 4), 0), 1);
    return f * x * x;
  }
  return f;
}

/** three's `SphericalHarmonics3` basis (Yᵢ, i = 0 … 8) at the unit direction `d`. */
export function vdbRefShBasis(d: readonly number[]): number[] {
  const x = d[0]!;
  const y = d[1]!;
  const z = d[2]!;
  return [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)];
}

// ── Small maths shared by the twin and the uniform packer ─────────────────────────────────────────────

/** Inverse of a column-major 4x4 (general). */
export function vdbRefInvert4(m: ArrayLike<number>): Float64Array {
  const a = Array.from({ length: 16 }, (_, i) => m[i]!);
  const inv = new Float64Array(16);
  inv[0] = a[5]! * a[10]! * a[15]! - a[5]! * a[11]! * a[14]! - a[9]! * a[6]! * a[15]! + a[9]! * a[7]! * a[14]! + a[13]! * a[6]! * a[11]! - a[13]! * a[7]! * a[10]!;
  inv[4] = -a[4]! * a[10]! * a[15]! + a[4]! * a[11]! * a[14]! + a[8]! * a[6]! * a[15]! - a[8]! * a[7]! * a[14]! - a[12]! * a[6]! * a[11]! + a[12]! * a[7]! * a[10]!;
  inv[8] = a[4]! * a[9]! * a[15]! - a[4]! * a[11]! * a[13]! - a[8]! * a[5]! * a[15]! + a[8]! * a[7]! * a[13]! + a[12]! * a[5]! * a[11]! - a[12]! * a[7]! * a[9]!;
  inv[12] = -a[4]! * a[9]! * a[14]! + a[4]! * a[10]! * a[13]! + a[8]! * a[5]! * a[14]! - a[8]! * a[6]! * a[13]! - a[12]! * a[5]! * a[10]! + a[12]! * a[6]! * a[9]!;
  inv[1] = -a[1]! * a[10]! * a[15]! + a[1]! * a[11]! * a[14]! + a[9]! * a[2]! * a[15]! - a[9]! * a[3]! * a[14]! - a[13]! * a[2]! * a[11]! + a[13]! * a[3]! * a[10]!;
  inv[5] = a[0]! * a[10]! * a[15]! - a[0]! * a[11]! * a[14]! - a[8]! * a[2]! * a[15]! + a[8]! * a[3]! * a[14]! + a[12]! * a[2]! * a[11]! - a[12]! * a[3]! * a[10]!;
  inv[9] = -a[0]! * a[9]! * a[15]! + a[0]! * a[11]! * a[13]! + a[8]! * a[1]! * a[15]! - a[8]! * a[3]! * a[13]! - a[12]! * a[1]! * a[11]! + a[12]! * a[3]! * a[9]!;
  inv[13] = a[0]! * a[9]! * a[14]! - a[0]! * a[10]! * a[13]! - a[8]! * a[1]! * a[14]! + a[8]! * a[2]! * a[13]! + a[12]! * a[1]! * a[10]! - a[12]! * a[2]! * a[9]!;
  inv[2] = a[1]! * a[6]! * a[15]! - a[1]! * a[7]! * a[14]! - a[5]! * a[2]! * a[15]! + a[5]! * a[3]! * a[14]! + a[13]! * a[2]! * a[7]! - a[13]! * a[3]! * a[6]!;
  inv[6] = -a[0]! * a[6]! * a[15]! + a[0]! * a[7]! * a[14]! + a[4]! * a[2]! * a[15]! - a[4]! * a[3]! * a[14]! - a[12]! * a[2]! * a[7]! + a[12]! * a[3]! * a[6]!;
  inv[10] = a[0]! * a[5]! * a[15]! - a[0]! * a[7]! * a[13]! - a[4]! * a[1]! * a[15]! + a[4]! * a[3]! * a[13]! + a[12]! * a[1]! * a[7]! - a[12]! * a[3]! * a[5]!;
  inv[14] = -a[0]! * a[5]! * a[14]! + a[0]! * a[6]! * a[13]! + a[4]! * a[1]! * a[14]! - a[4]! * a[2]! * a[13]! - a[12]! * a[1]! * a[6]! + a[12]! * a[2]! * a[5]!;
  inv[3] = -a[1]! * a[6]! * a[11]! + a[1]! * a[7]! * a[10]! + a[5]! * a[2]! * a[11]! - a[5]! * a[3]! * a[10]! - a[9]! * a[2]! * a[7]! + a[9]! * a[3]! * a[6]!;
  inv[7] = a[0]! * a[6]! * a[11]! - a[0]! * a[7]! * a[10]! - a[4]! * a[2]! * a[11]! + a[4]! * a[3]! * a[10]! + a[8]! * a[2]! * a[7]! - a[8]! * a[3]! * a[6]!;
  inv[11] = -a[0]! * a[5]! * a[11]! + a[0]! * a[7]! * a[9]! + a[4]! * a[1]! * a[11]! - a[4]! * a[3]! * a[9]! - a[8]! * a[1]! * a[7]! + a[8]! * a[3]! * a[5]!;
  inv[15] = a[0]! * a[5]! * a[10]! - a[0]! * a[6]! * a[9]! - a[4]! * a[1]! * a[10]! + a[4]! * a[2]! * a[9]! + a[8]! * a[1]! * a[6]! - a[8]! * a[2]! * a[5]!;
  const det = a[0]! * inv[0]! + a[1]! * inv[4]! + a[2]! * inv[8]! + a[3]! * inv[12]!;
  if (det === 0 || !Number.isFinite(det)) throw new RangeError("vdbRefInvert4: singular matrix");
  for (let i = 0; i < 16; i++) inv[i] = inv[i]! / det;
  return inv;
}

/** Column-major 4x4 × point (w = 1), divided by w. */
export function vdbRefTransformPoint(m: ArrayLike<number>, p: readonly number[]): VDBRefVec3 {
  const x = m[0]! * p[0]! + m[4]! * p[1]! + m[8]! * p[2]! + m[12]!;
  const y = m[1]! * p[0]! + m[5]! * p[1]! + m[9]! * p[2]! + m[13]!;
  const z = m[2]! * p[0]! + m[6]! * p[1]! + m[10]! * p[2]! + m[14]!;
  const w = m[3]! * p[0]! + m[7]! * p[1]! + m[11]! * p[2]! + m[15]!;
  return [x / w, y / w, z / w];
}

/** Column-major 4x4 × vector (w = 0). */
export function vdbRefTransformVector(m: ArrayLike<number>, v: readonly number[]): VDBRefVec3 {
  return [m[0]! * v[0]! + m[4]! * v[1]! + m[8]! * v[2]!, m[1]! * v[0]! + m[5]! * v[1]! + m[9]! * v[2]!, m[2]! * v[0]! + m[6]! * v[1]! + m[10]! * v[2]!];
}

/** Column-major 4x4 product a × b. */
export function vdbRefMultiply4(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r]! * b[c * 4 + k]!;
    o[c * 4 + r] = s;
  }
  return o;
}
