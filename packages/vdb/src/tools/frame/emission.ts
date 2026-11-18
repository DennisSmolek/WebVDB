// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameEmissionLight.js L3–15 (constants), L138–150 (emitter centre and radius),
//   L182–241 (updateProbeGrid); src/three-openvdb/frame/webgpu/FrameEmissionLight.js L59–60 (golden angle),
//   L168–237 (the direction pass), L259–279 (the reduce pass), L310–364 (the probe pass), L457–519 (the light
//   node's irradiance).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free TypeScript CPU twins (f64) of his emission-light maths; boxes and matrices are plain
// arrays (column-major). Divergences: none (twins of the ported TSL).

/**
 * K3.6: CPU twins of the emission light (his `FrameEmissionLight`): each emissive volume's light along 16
 * Fibonacci directions × 8² parallel rays (direction pass), its mean intensity and luminance-weighted centre
 * (reduce pass), the L1 irradiance probes of all volumes as point lights (probe pass), the probe cascades
 * (power-of-two snapped) and the irradiance a lit surface reads.
 */

import type { Vec3, Vec4 } from "./index.js";
import { vdbFrameFindCascade, vdbFrameGridUvw, vdbFrameSampleGrid, type VDBFrameAtlas, type VDBFrameGridLayout } from "./grid.js";

/** Directions of each volume's emitted light (his `frameEmissionDirections`). */
export const frameEmissionDirections = 16;
/** Rays on each side of the square that covers the volume (his `frameEmissionRays`). */
export const frameEmissionRays = 8;
/** Probes per axis, at most (the atlas is made for it; his `frameEmissionMaxResolution`). */
export const frameEmissionMaxResolution = 32;
export const frameEmissionMaxCascades = 4;
export const frameEmissionAtlasColumns = Math.ceil(Math.sqrt(frameEmissionMaxResolution * frameEmissionMaxCascades));
export const frameEmissionAtlasSize = frameEmissionMaxResolution * frameEmissionAtlasColumns;
/** Golden angle in radians: Fibonacci directions turn by it. */
export const frameEmissionGoldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));
/** Linear-sRGB luminance (three's `luminance()` with the working colour space's coefficients). */
export const frameEmissionLuminance: Vec3 = [0.2126, 0.7152, 0.0722];

const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (a: Vec3): Vec3 => {
  const l = Math.sqrt(dot(a, a));
  return [a[0] / l, a[1] / l, a[2] / l];
};
const apply = (m: ArrayLike<number>, p: Vec3): Vec3 => [
  m[0]! * p[0] + m[4]! * p[1] + m[8]! * p[2] + m[12]!,
  m[1]! * p[0] + m[5]! * p[1] + m[9]! * p[2] + m[13]!,
  m[2]! * p[0] + m[6]! * p[1] + m[10]! * p[2] + m[14]!,
];

/** Direction `index` (0–15) of the direction pass: a Fibonacci sphere point, from the volume to the viewer, with its tangent frame. */
export function vdbFrameEmissionDirection(index: number): { direction: Vec3; tangent: Vec3; bitangent: Vec3 } {
  const y = 1 - (index * 2 + 1) / frameEmissionDirections;
  const ring = Math.sqrt(Math.max(0, 1 - y * y));
  const phi = index * frameEmissionGoldenAngle;
  const direction: Vec3 = [Math.cos(phi) * ring, y, Math.sin(phi) * ring];
  const helper: Vec3 = Math.abs(direction[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
  const tangent = normalize(cross(helper, direction));
  return { direction, tangent, bitangent: cross(direction, tangent) };
}

/** His `boxSegment` of the local unit box ([−½, ½]³): the [t0, t1] ⊂ [0, 1] of the segment inside it. */
export function vdbFrameBoxSegment(start: Vec3, end: Vec3): [number, number] {
  let t0 = -Infinity;
  let t1 = Infinity;
  for (let c = 0; c < 3; c++) {
    const inv = 1 / (end[c]! - start[c]!);
    const a = (-0.5 - start[c]!) * inv;
    const b = (0.5 - start[c]!) * inv;
    t0 = Math.max(t0, Math.min(a, b));
    t1 = Math.min(t1, Math.max(a, b));
  }
  return [Math.max(0, t0), Math.min(1, t1)];
}

/** One emitter as the direction pass reads it. Samplers take local positions (unit box) and a mip lod. */
export interface VDBFrameEmitter {
  /** Extinction per world unit (the source's `sampleExtinction`). */
  sampleExtinction(local: Vec3, lod: number): number;
  /** Emitted radiance (the source's `sampleEmission`; the pass reads lod 0). */
  sampleEmission(local: Vec3, lod: number): Vec3;
  /** The source's `getLod` of a local size. */
  getLod(localSize: number): number;
  /** World → local unit box, column-major. */
  worldToLocal: ArrayLike<number>;
  center: Vec3;
  radius: number;
  /** World scale of the volume (`matrixWorld.getMaxScaleOnAxis()`). */
  worldScale: number;
  absorbance: number;
  steps: number;
}

/**
 * The emitter centre and bounding radius (FrameEmissionLight core L138–150): the translation of the volume's world
 * matrix (unit box → world) and the largest distance of a box corner from it.
 */
export function vdbFrameEmitterBounds(matrixWorld: ArrayLike<number>): { center: Vec3; radius: number; min: Vec3; max: Vec3 } {
  const center: Vec3 = [matrixWorld[12]!, matrixWorld[13]!, matrixWorld[14]!];
  let radius = 0;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i++) {
    const corner = apply(matrixWorld, [i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5]);
    radius = Math.max(radius, Math.hypot(corner[0] - center[0], corner[1] - center[1], corner[2] - center[2]));
    for (let c = 0; c < 3; c++) {
      min[c] = Math.min(min[c]!, corner[c]!);
      max[c] = Math.max(max[c]!, corner[c]!);
    }
  }
  return { center, radius, min, max };
}

/**
 * One texel of the direction pass (his FrameEmissionDirectionMaterial): the emitted light of the volume toward
 * direction `index`, summed over 8² parallel rays that cover its bounding sphere. Each ray adds emission × α × the
 * transmittance so far (absorbance scales the transmittance, as for scene lights) and stops below 1%. Returns output
 * 0 (intensity × ray area, radius) and output 1 (luminance-weighted world position × ray area, weight × area).
 */
export function vdbFrameEmissionDirectionTexel(emitter: VDBFrameEmitter, index: number): { intensity: Vec4; center: Vec4 } {
  const { direction, tangent, bitangent } = vdbFrameEmissionDirection(index);
  const { center: c, radius, steps } = emitter;
  const diameter = radius * 2;
  const spacing = diameter / frameEmissionRays;
  const intensity: Vec3 = [0, 0, 0];
  const center: Vec4 = [0, 0, 0, 0];
  for (let r = 0; r < frameEmissionRays * frameEmissionRays; r++) {
    const u = (((r % frameEmissionRays) + 0.5) / frameEmissionRays) * 2 - 1;
    const v = ((Math.floor(r / frameEmissionRays) + 0.5) / frameEmissionRays) * 2 - 1;
    const worldStart: Vec3 = [0, 1, 2].map((k) => c[k]! + direction[k]! * radius + tangent[k]! * u * radius + bitangent[k]! * v * radius) as Vec3;
    const worldEnd: Vec3 = [0, 1, 2].map((k) => worldStart[k]! - direction[k]! * diameter) as Vec3;
    const localStart = apply(emitter.worldToLocal, worldStart);
    const localEnd = apply(emitter.worldToLocal, worldEnd);
    const segment = vdbFrameBoxSegment(localStart, localEnd);
    if (!(segment[0] < segment[1])) continue;
    const segmentLength = segment[1] - segment[0];
    const stepLength = (segmentLength * diameter) / steps;
    const lod = emitter.getLod(stepLength / emitter.worldScale);
    let transmittance = 1;
    for (let s = 0; s < steps; s++) {
      const t = segment[0] + segmentLength * ((s + 0.5) / steps);
      const local: Vec3 = [0, 1, 2].map((k) => localStart[k]! + (localEnd[k]! - localStart[k]!) * t) as Vec3;
      const extinction = emitter.sampleExtinction(local, lod);
      if (extinction > 0) {
        const alpha = 1 - Math.exp(-extinction * stepLength);
        const e = emitter.sampleEmission(local, 0);
        const radiance: Vec3 = [e[0] * transmittance * alpha, e[1] * transmittance * alpha, e[2] * transmittance * alpha];
        const weight = dot(radiance, frameEmissionLuminance);
        for (let k = 0; k < 3; k++) {
          intensity[k] = intensity[k]! + radiance[k]!;
          center[k] = center[k]! + (worldStart[k]! + (worldEnd[k]! - worldStart[k]!) * t) * weight;
        }
        center[3] += weight;
        transmittance *= Math.exp(-extinction * emitter.absorbance * stepLength);
        if (transmittance < 0.01) break;
      }
    }
  }
  const area = spacing * spacing;
  return { intensity: [intensity[0] * area, intensity[1] * area, intensity[2] * area, radius], center: [center[0] * area, center[1] * area, center[2] * area, center[3] * area] };
}

/** The reduce pass: mean intensity of the 16 directions, and the centre of the visible emission ((0,0,0,0) when none). */
export function vdbFrameEmissionReduce(texels: readonly { intensity: Vec4; center: Vec4 }[]): { intensity: Vec4; center: Vec4 } {
  const intensity: Vec4 = [0, 0, 0, 0];
  const center: Vec4 = [0, 0, 0, 0];
  for (const t of texels) {
    for (let k = 0; k < 4; k++) {
      intensity[k] = intensity[k]! + t.intensity[k]!;
      center[k] = center[k]! + t.center[k]!;
    }
  }
  for (let k = 0; k < 4; k++) intensity[k] = intensity[k]! / frameEmissionDirections;
  const visible = center[3] > 0;
  const w = Math.max(center[3], 1e-12);
  return { intensity, center: visible ? [center[0] / w, center[1] / w, center[2] / w, 1] : [0, 0, 0, 0] };
}

/** One emitter of the probe pass: the reduce outputs (intensity rgb + radius, centre + visible flag). */
export interface VDBFrameEmitterLight {
  intensity: Vec4;
  center: Vec4;
}

/**
 * One probe of the probe pass: the L1 irradiance of every emitter as a point light at its emission centre,
 * E(n) = E/4 + E/2 · dot(l, n), softened near the centre by a quarter of the radius. Returns (red, green, blue),
 * each (constant, direction xyz).
 */
export function vdbFrameEmissionProbe(probe: Vec3, emitters: readonly VDBFrameEmitterLight[]): [Vec4, Vec4, Vec4] {
  const constant: Vec3 = [0, 0, 0];
  const dirs: [Vec3, Vec3, Vec3] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const e of emitters) {
    if (!(e.center[3] > 0)) continue;
    const toProbe: Vec3 = [probe[0] - e.center[0], probe[1] - e.center[1], probe[2] - e.center[2]];
    const d2 = dot(toProbe, toProbe);
    const soft = e.intensity[3] * 0.25;
    const sd2 = d2 + soft * soft;
    const irradiance: Vec3 = [e.intensity[0] / sd2, e.intensity[1] / sd2, e.intensity[2] / sd2];
    const scale = (1 / Math.sqrt(Math.max(d2, 1e-12))) * (d2 / sd2) * -0.5;
    const toLight: Vec3 = [toProbe[0] * scale, toProbe[1] * scale, toProbe[2] * scale];
    for (let k = 0; k < 3; k++) {
      constant[k] = constant[k]! + irradiance[k]! * 0.25;
      for (let a = 0; a < 3; a++) dirs[k]![a] = dirs[k]![a]! + toLight[a]! * irradiance[k]!;
    }
  }
  return [
    [constant[0], ...dirs[0]],
    [constant[1], ...dirs[1]],
    [constant[2], ...dirs[2]],
  ];
}

/** The light node's irradiance from (filtered) probe values: max(x + dot(yzw, n), 0) per channel, × falloff. */
export function vdbFrameEmissionIrradiance(probe: readonly [Vec4, Vec4, Vec4], normal: Vec3, falloff = 1): Vec3 {
  return probe.map((p) => Math.max(0, p[0] + p[1] * normal[0] + p[2] * normal[1] + p[3] * normal[2]) * falloff) as Vec3;
}

/** Settings of the probe cascades (his `FrameEmissionLight` props). */
export interface VDBFrameEmissionGridSettings {
  resolution: number;
  range: number;
  cascades: number;
  distance: number;
}

/** The probe cascades (cascade 0 finest), the centre of all emitters, and a key that changes with them. */
export interface VDBFrameEmissionGrid {
  resolution: number;
  count: number;
  min: Vec3[];
  size: Vec3[];
  center: Vec3;
  key: string;
}

/**
 * updateProbeGrid (FrameEmissionLight core L186–241): the last cascade is the box of all emitters grown by
 * (range − 1)/2 of its longest side on each side; finer cascades are power-of-two-voxel cubes around the camera,
 * each half the next, snapped to their voxel; a fine cascade that is not finer than the last or holds no emitter is
 * skipped.
 */
export function vdbFrameEmissionProbeGrid(settings: VDBFrameEmissionGridSettings, emitterBoxes: readonly { min: Vec3; max: Vec3 }[], cameraPosition: Vec3): VDBFrameEmissionGrid {
  const resolution = Math.min(frameEmissionMaxResolution, Math.max(2, Math.round(settings.resolution)));
  const cascades = Math.min(frameEmissionMaxCascades, Math.max(1, Math.round(settings.cascades)));
  const gMin: Vec3 = [Infinity, Infinity, Infinity];
  const gMax: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const b of emitterBoxes) {
    for (let k = 0; k < 3; k++) {
      gMin[k] = Math.min(gMin[k]!, b.min[k]!);
      gMax[k] = Math.max(gMax[k]!, b.max[k]!);
    }
  }
  const center: Vec3 = [(gMin[0] + gMax[0]) / 2, (gMin[1] + gMax[1]) / 2, (gMin[2] + gMax[2]) / 2];
  let size: Vec3 = [gMax[0] - gMin[0], gMax[1] - gMin[1], gMax[2] - gMin[2]];
  const grow = (Math.max(size[0], size[1], size[2]) * Math.max(0, settings.range - 1)) / 2;
  for (let k = 0; k < 3; k++) {
    gMin[k] = gMin[k]! - grow;
    gMax[k] = gMax[k]! + grow;
  }
  size = [Math.max(gMax[0] - gMin[0], 1e-6), Math.max(gMax[1] - gMin[1], 1e-6), Math.max(gMax[2] - gMin[2], 1e-6)];
  const lastSize = Math.max(size[0], size[1], size[2]);
  const fineHalfSize = Math.min(settings.distance, lastSize / 2);
  const min: Vec3[] = [];
  const sizes: Vec3[] = [];
  for (let cascade = 0; cascade < cascades - 1; cascade++) {
    const halfSize = fineHalfSize / Math.pow(2, cascades - 1 - cascade);
    const voxel = Math.pow(2, Math.ceil(Math.log2((halfSize * 2) / (resolution - 1))));
    const s = voxel * resolution;
    const bMin = [0, 1, 2].map((k) => Math.floor((cameraPosition[k]! - halfSize) / voxel) * voxel) as Vec3;
    const bMax = [bMin[0] + s, bMin[1] + s, bMin[2] + s] as Vec3;
    const intersects = bMax[0] >= gMin[0] && bMin[0] <= gMax[0] && bMax[1] >= gMin[1] && bMin[1] <= gMax[1] && bMax[2] >= gMin[2] && bMin[2] <= gMax[2];
    if (s >= lastSize || !intersects) continue;
    min.push(bMin);
    sizes.push([s, s, s]);
  }
  min.push([gMin[0], gMin[1], gMin[2]]);
  sizes.push(size);
  const key = [resolution];
  for (let i = 0; i < min.length; i++) key.push(...min[i]!, ...sizes[i]!);
  return { resolution, count: min.length, min, size: sizes, center, key: key.join(",") };
}

/**
 * Past the last cascade (the light node, L490–506): the clamped edge probe's light falls off as the squared distance
 * ratio |edge − centre|² / |p − centre|² (≤ 1).
 */
export function vdbFrameEmissionFalloff(edge: Vec3, position: Vec3, center: Vec3): number {
  const a: Vec3 = [edge[0] - center[0], edge[1] - center[1], edge[2] - center[2]];
  const b: Vec3 = [position[0] - center[0], position[1] - center[1], position[2] - center[2]];
  return Math.min(1, dot(a, a) / Math.max(dot(b, b), 1e-6));
}

/**
 * The light node's probe read (L457–507) on CPU atlases (the probe target's three attachments, rows bottom first):
 * the finest cascade that holds the point (half a probe inside), blended into the next one over the last tenth of
 * each side; outside every cascade, the last cascade's clamped edge with {@link vdbFrameEmissionFalloff}. Returns the
 * (red, green, blue) L1 probe values and the falloff; {@link vdbFrameEmissionIrradiance} then gives E(n).
 */
export function vdbFrameEmissionLookup(atlases: readonly [VDBFrameAtlas, VDBFrameAtlas, VDBFrameAtlas], grid: VDBFrameGridLayout, center: Vec3, position: Vec3): { probe: [Vec4, Vec4, Vec4]; falloff: number } {
  let probe: [Vec4, Vec4, Vec4] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  let falloff = 1;
  const sample = (cascade: number, uvw: Vec3, weight: number) => {
    probe = probe.map((p, k) => {
      const s = vdbFrameSampleGrid(atlases[k]!, grid, cascade, uvw);
      return p.map((v, c) => v + (s[c]! - v) * weight) as Vec4;
    }) as [Vec4, Vec4, Vec4];
  };
  if (grid.count > 0) {
    const cascade = vdbFrameFindCascade(grid, position, 0);
    if (cascade >= 0) {
      const uvw = vdbFrameGridUvw(grid, cascade, position);
      sample(cascade, uvw, 1);
      const side = Math.min(uvw[0], uvw[1], uvw[2], 1 - uvw[0], 1 - uvw[1], 1 - uvw[2]);
      const t = Math.min(1, Math.max(0, side / 0.1));
      const blend = 1 - t * t * (3 - 2 * t);
      if (blend > 0) {
        const next = vdbFrameFindCascade(grid, position, cascade + 1);
        if (next >= 0) sample(next, vdbFrameGridUvw(grid, next, position), blend);
      }
    } else {
      const last = grid.count - 1;
      const inset = 0.5 / grid.resolution;
      const uvw = vdbFrameGridUvw(grid, last, position).map((v) => Math.min(1 - inset, Math.max(inset, v))) as Vec3;
      const m = grid.gridMin[last]!;
      const s = grid.gridSize[last]!;
      sample(last, uvw, 1);
      falloff = vdbFrameEmissionFalloff([m[0] + uvw[0] * s[0], m[1] + uvw[1] * s[1], m[2] + uvw[2] * s[2]], position, center);
    }
  }
  return { probe, falloff };
}
