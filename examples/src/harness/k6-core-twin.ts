/**
 * K6.1 CPU twins of the core frame kernels, assembled from the tools/frame twins (f64): the gather kernel's
 * per-froxel work (his inject pass on the dense source's CPU sampler, then the 'mixture' / 'over' combine),
 * the resolve kernel's temporal blend and column scan, and the composite's two taps. `mutate` flips one term
 * so the probe must fail (fail-before, CLAUDE.md "correctness before timing").
 */
import {
  sampleWvdbDenseTexture,
  vdbFrameBoxSteps,
  vdbFrameDistanceFade,
  vdbFrameFroxel,
  vdbFrameGatherFroxel,
  vdbFrameInjectSlice,
  vdbFrameIntegratePrefix,
  vdbFrameIsotropicDiffusion,
  vdbFrameNoise,
  vdbFrameOctavesOff,
  vdbFrameReproject,
  vdbFrameScanColumn,
  vdbFrameShadowMarch,
  vdbFrameShadowedLightColor,
  vdbFrameSliceCoord,
  vdbFrameTemporalFroxel,
  vdbFrameViewPosition,
  vdbMat4Point,
  vdbShIrradianceAt,
  type VDBFrameFroxelMix,
  type VDBFrameVec2,
  type VDBFrameVec3,
  type VDBFrameVec4,
  type VDBFrameVolumeFroxel,
  type WvdbDenseVolume,
} from "../../../packages/vdb/src/tools/internal.js";

type V3 = VDBFrameVec3;
type V4 = VDBFrameVec4;

/** The terms `?mutate=` can break (each probe's spec checks one fails). */
export type TwinMutation = "none" | "mix" | "phase" | "extinction" | "fade" | "scan-stop" | "temporal-clamp" | "coverage" | "taps";

/** One volume as the gather kernel sees it (the renderer's packed record, in f64). */
export interface TwinVolume {
  volume: WvdbDenseVolume;
  rect: [number, number, number, number];
  first: number;
  count: number;
  viewToLocal: ArrayLike<number>;
  localToView: ArrayLike<number>;
  worldToLocal: ArrayLike<number>;
  densityScale: number;
  densityCutoff: number;
  extinction: number;
  worldScale: number;
  absorbance: number;
  opacity: number;
  baseColor: V3;
  scatterColor: V3;
  offset: V3;
  wrap: 0 | 1 | 2;
  phase: V4;
  phaseSpectrum: V3;
  multipleScattering: number;
  lights: number;
  shadowSteps: number;
  mipmaps: boolean;
  levels: number;
  emissiveIntensity: number;
  temperatureRange: [number, number];
}

/** The frame values the gather kernel reads. */
export interface TwinFrame {
  grid: V3;
  columns: number;
  range: VDBFrameVec2;
  projectionInverse: ArrayLike<number>;
  cameraWorld: ArrayLike<number>;
  jitter: number;
  fadeRadius: number;
  subSteps: number;
  ambient: V3;
  probe: V3[];
  hemi: { sky: V3; ground: V3; up: V3 }[];
  /** Packed like the uniforms: position (w kind), colour (w penumbra cos), spot (xyz target → light, w cone cos). */
  lights: { position: V4; color: V4; spot: V4 }[];
}

const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const add = (a: readonly number[], b: readonly number[]): V3 => [a[0]! + b[0]!, a[1]! + b[1]!, a[2]! + b[2]!];
const scale = (a: readonly number[], s: number): V3 => [a[0]! * s, a[1]! * s, a[2]! * s];
const mul3 = (a: readonly number[], b: readonly number[]): V3 => [a[0]! * b[0]!, a[1]! * b[1]!, a[2]! * b[2]!];
const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const len = (a: readonly number[]) => Math.hypot(a[0]!, a[1]!, a[2]!);
const norm = (a: readonly number[]): V3 => scale(a, 1 / len(a));
const mix3 = (a: readonly number[], b: readonly number[], t: number): V3 => [a[0]! + (b[0]! - a[0]!) * t, a[1]! + (b[1]! - a[1]!) * t, a[2]! + (b[2]! - a[2]!) * t];
const mat3 = (m: ArrayLike<number>, v: readonly number[]): V3 => [
  m[0]! * v[0]! + m[4]! * v[1]! + m[8]! * v[2]!,
  m[1]! * v[0]! + m[5]! * v[1]! + m[9]! * v[2]!,
  m[2]! * v[0]! + m[6]! * v[1]! + m[10]! * v[2]!,
];
const lodOf = (size: number, maxTexels: number) => Math.max(0, Math.log2(size * maxTexels));
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

function boxSegment(a: V3, b: V3): [number, number] {
  const d = sub(b, a);
  const t0 = [0, 1, 2].map((i) => (-0.5 - a[i]!) / d[i]!);
  const t1 = [0, 1, 2].map((i) => (0.5 - a[i]!) / d[i]!);
  const lo = [0, 1, 2].map((i) => Math.min(t0[i]!, t1[i]!));
  const hi = [0, 1, 2].map((i) => Math.max(t0[i]!, t1[i]!));
  return [Math.max(0, Math.max(...lo)), Math.min(1, Math.min(...hi))];
}

function boxExit(o: V3, d: V3): number {
  const far = [0, 1, 2].map((i) => Math.max((-0.5 - o[i]!) / d[i]!, (0.5 - o[i]!) / d[i]!));
  return Math.max(0, Math.min(...far));
}

const mirror = (x: number) => {
  const p = x - 2 * Math.floor(x / 2);
  return p > 1 ? 2 - p : p;
};

function gridUv(r: TwinVolume, local: V3): V3 {
  const uvw = add(add(local, [0.5, 0.5, 0.5]), r.offset);
  if (r.wrap === 2) return [mirror(uvw[0]), mirror(uvw[1]), mirror(uvw[2])];
  if (r.wrap === 1) return [uvw[0] - Math.floor(uvw[0]), uvw[1] - Math.floor(uvw[1]), uvw[2] - Math.floor(uvw[2])];
  return uvw;
}

function extinctionAt(r: TwinVolume, local: V3, lod: number, mutate: TwinMutation): number {
  const vol = r.volume;
  const mask = vol.mask ? sampleWvdbDenseTexture(vol.mask, add(local, [0.5, 0.5, 0.5]), lod) : 1;
  const value = sampleWvdbDenseTexture(vol.density, gridUv(r, local), lod) * mask;
  const e = ((value > r.densityCutoff ? value : 0) * r.densityScale * r.extinction) / r.worldScale;
  return mutate === "extinction" ? e * 1.02 : e;
}

function albedoAt(r: TwinVolume, local: V3, lod: number): V3 {
  const vol = r.volume;
  if (!vol.albedo) return [...r.baseColor] as V3;
  const uvw = gridUv(r, local);
  return [0, 1, 2].map((c) => r.baseColor[c]! * sampleWvdbDenseTexture(vol.albedo!, uvw, lod, c)) as V3;
}

function emissionAt(r: TwinVolume, local: V3, lod: number): V3 {
  const vol = r.volume;
  if (!vol.emission) return [0, 0, 0];
  const value = sampleWvdbDenseTexture(vol.emission, gridUv(r, local), lod);
  const [t0, t1] = r.temperatureRange;
  const temperature = t0 + (t1 - t0) * value;
  const power = Math.pow(temperature / Math.max(t1, 1), 4);
  const c = temperature > 0 ? 1 : 0;
  return [c * power * r.emissiveIntensity, c * power * r.emissiveIntensity, c * power * r.emissiveIntensity];
}

function march(r: TwinVolume, origin: V3, localDirection: V3, maxDistance: number, lod: number, maxTexels: number, albedo: V3, noise: number, mutate: TwinMutation): V4 {
  if (r.absorbance <= 0 || r.shadowSteps === 0) return [1, 1, 0, 0];
  const d = norm(localDirection);
  const distance = Math.min(boxExit(origin, d), maxDistance);
  if (distance <= 0) return [1, 1, 0, 0];
  const step = distance / r.shadowSteps;
  const stepLod = r.mipmaps ? Math.max(lod, lodOf(step, maxTexels)) : 0;
  const stepScale = step * len(mat3(r.localToView, d)) * r.absorbance;
  const samples: number[] = [];
  for (let k = 0; k < r.shadowSteps; k++) samples.push(extinctionAt(r, add(origin, scale(d, step * noise + step * k)), stepLod, mutate) * stepScale);
  return vdbFrameShadowMarch(samples, albedo, r.multipleScattering, vdbFrameOctavesOff);
}

/** One volume in one froxel (the WGSL `vdbFrameVolumeFroxel`), or null when it adds nothing. */
export function volumeFroxelTwin(frame: TwinFrame, r: TwinVolume, cell: V3, mutate: TwinMutation = "none"): VDBFrameVolumeFroxel | null {
  if (cell[0] < r.rect[0] || cell[1] < r.rect[1] || cell[0] >= r.rect[2] || cell[1] >= r.rect[3] || cell[2] < r.first || cell[2] >= r.first + r.count) return null;
  const tile: VDBFrameVec2 = [cell[2] % frame.columns, Math.floor(cell[2] / frame.columns)];
  const fragCoord: VDBFrameVec2 = [tile[0] * frame.grid[0] + cell[0] + 0.5, tile[1] * frame.grid[1] + cell[1] + 0.5];
  const f = vdbFrameFroxel(frame.grid, frame.columns, frame.range, frame.projectionInverse, fragCoord);
  const localNear = vdbMat4Point(r.viewToLocal, f.viewNear);
  const localFar = vdbMat4Point(r.viewToLocal, f.viewFar);
  const seg = boxSegment(localNear, localFar);
  if (seg[0] >= seg[1]) return null;
  const segment = seg[1] - seg[0];
  const dims = r.volume.dimensions;
  const maxTexels = Math.max(dims[0], dims[1], dims[2]);
  const middle = (f.sliceNear + f.sliceFar) * 0.5;
  const width = len(sub(vdbFrameViewPosition(frame.projectionInverse, [f.uv[0] + 1 / frame.grid[0], f.uv[1]], middle), vdbFrameViewPosition(frame.projectionInverse, f.uv, middle)));
  const widthLod = lodOf(width / r.worldScale, maxTexels);
  const sampledLod = r.mipmaps ? Math.min(widthLod, r.levels) : 0;
  const steps = vdbFrameBoxSteps(len(sub(localFar, localNear)) * segment, maxTexels, sampledLod, frame.subSteps);
  const subStepLength = (len(sub(f.viewFar, f.viewNear)) * segment) / steps;
  const lod = Math.max(widthLod, lodOf(subStepLength / r.worldScale, maxTexels));
  const readLod = r.mipmaps ? lod : 0;
  const jitter = vdbFrameNoise(frame.jitter, fragCoord);
  const depths: number[] = [];
  for (let i = 0; i < steps; i++) {
    const t = seg[0] + segment * ((i + jitter) / steps);
    const fade = mutate === "fade" ? 1 : vdbFrameDistanceFade(frame.fadeRadius, mix3(f.viewNear, f.viewFar, t));
    depths.push(extinctionAt(r, mix3(localNear, localFar, t), readLod, mutate) * r.opacity * fade * subStepLength);
  }
  const slice = vdbFrameInjectSlice(depths, seg, jitter);
  if (!slice) return null;
  const local = mix3(localNear, localFar, slice.t);
  const view = vdbMat4Point(r.localToView, local);
  const world = vdbMat4Point(frame.cameraWorld, view);
  const rayDirection = norm(mat3(frame.cameraWorld, sub(f.viewFar, f.viewNear)));
  const shadowNoise = vdbFrameNoise(frame.jitter, [fragCoord[0] + 17, fragCoord[1] + 31]);
  const albedo = albedoAt(r, local, readLod);
  const diffusion = r.multipleScattering > 0;
  const phase: V4 = mutate === "phase" ? [0, 0, 0, 0] : r.phase;
  const outward = len(local) > 1e-4 ? norm(mat3(frame.cameraWorld, mat3(r.localToView, local))) : ([0, 1, 0] as V3);
  let irradiance: V3 = [0, 0, 0];
  if (r.lights & 32) irradiance = add(add(irradiance, frame.ambient), vdbShIrradianceAt(outward, frame.probe));
  if (r.lights & 2) {
    for (const h of frame.hemi) {
      const up = mat3(r.worldToLocal, h.up);
      const front = march(r, local, up, 1e6, lod, maxTexels, albedo, shadowNoise, mutate);
      const back = march(r, local, scale(up, -1), 1e6, lod, maxTexels, albedo, shadowNoise, mutate);
      const sky = diffusion ? vdbFrameIsotropicDiffusion(front[2], back[2], front[3]) : front[1];
      const ground = diffusion ? vdbFrameIsotropicDiffusion(back[2], front[2], back[3]) : back[1];
      irradiance = add(irradiance, add(scale(h.sky, 0.5 * sky), scale(h.ground, 0.5 * ground)));
    }
  }
  for (const l of frame.lights) {
    const isDirectional = l.position[3] < 1.5;
    const isSpot = l.position[3] > 2.5;
    const bit = isDirectional ? 8 : isSpot ? 4 : 16;
    const toLightWorld = sub(l.position, world);
    const spot = isSpot ? smoothstep(l.spot[3], l.color[3], dot(norm(toLightWorld), l.spot)) : 1;
    if (!(r.lights & bit) || spot <= 0) continue;
    const toLight = sub(vdbMat4Point(r.worldToLocal, l.position), local);
    const direction = isDirectional ? mat3(r.worldToLocal, l.position) : toLight;
    const maxDistance = isDirectional ? 1e6 : len(toLight);
    const lightDirection = isDirectional ? ([l.position[0], l.position[1], l.position[2]] as V3) : norm(toLightWorld);
    const s = march(r, local, direction, maxDistance, lod, maxTexels, albedo, shadowNoise, mutate);
    const light = vdbFrameShadowedLightColor(phase, r.phaseSpectrum, vdbFrameOctavesOff, rayDirection, lightDirection, [s[0], s[1]]);
    irradiance = add(irradiance, scale(mul3([l.color[0], l.color[1], l.color[2]], light), spot));
  }
  const lit = scale(mul3(albedo, irradiance), 1 / Math.PI);
  const radiance = add([Math.max(r.scatterColor[0], lit[0]), Math.max(r.scatterColor[1], lit[1]), Math.max(r.scatterColor[2], lit[2])], emissionAt(r, local, readLod));
  return { radiance, alpha: slice.alpha };
}

/** The gather kernel's scatter texel at `cell`: every volume in record order, combined. */
export function gatherTwin(frame: TwinFrame, records: readonly TwinVolume[], cell: V3, mode: VDBFrameFroxelMix, mutate: TwinMutation = "none"): V4 {
  const lit: VDBFrameVolumeFroxel[] = [];
  for (const r of records) {
    const v = volumeFroxelTwin(frame, r, cell, mutate);
    if (v) lit.push(v);
  }
  const m: VDBFrameFroxelMix = mutate === "mix" ? (mode === "mixture" ? "over" : "mixture") : mode;
  return vdbFrameGatherFroxel(lit, m);
}

/** The resolve kernel for one column: (prefix boundaries, history′ per slice). */
export function resolveTwin(o: {
  column: readonly V4[];
  neighbours: (z: number) => V4[];
  history: ((z: number) => V4 | null) | null;
  blend: number;
  inSlices: (z: number) => boolean;
  mutate?: TwinMutation;
}): { prefix: V4[]; history: V4[] } {
  const D = o.column.length;
  const resolved: V4[] = [];
  for (let z = 0; z < D; z++) {
    const current = o.column[z]!;
    if (!o.inSlices(z)) {
      resolved.push(o.history ? [0, 0, 0, 0] : current);
      continue;
    }
    const h = o.history?.(z) ?? null;
    if (!h) {
      resolved.push(current);
      continue;
    }
    const n = [...o.neighbours(z)];
    if (z > 0) n.push(o.column[z - 1]!);
    if (z + 1 < D) n.push(o.column[z + 1]!);
    resolved.push(o.mutate === "temporal-clamp" ? (current.map((c, i) => h[i]! + (c - h[i]!) * o.blend) as V4) : vdbFrameTemporalFroxel(current, h, n, o.blend));
  }
  const prefix = vdbFrameScanColumn(resolved, o.mutate === "scan-stop" ? { stop: 0 } : {});
  return { prefix, history: resolved };
}

/** The composite at one pixel: (radiance, alpha) from the prefix column and the scene view depth. */
export function integrateTwin(boundaries: (k: number) => V4, viewDepth: number, range: VDBFrameVec2, depth: number, mutate: TwinMutation = "none"): V4 {
  const { k, c } = vdbFrameSliceCoord(viewDepth, range, depth);
  const cc = mutate === "coverage" ? Math.min(1, c * 1.1) : c;
  const r = vdbFrameIntegratePrefix(boundaries(k), boundaries(mutate === "taps" ? k : k + 1), cc);
  return [r.radiance[0], r.radiance[1], r.radiance[2], 1 - r.transmittance];
}

export { vdbFrameReproject };
