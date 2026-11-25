// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameVolumeShadows.js L42–53 (constants), L130–143 (getCascadeBreaks),
//   L248–290 (getCascadeLookup), L344–372 (the receiver transmittance), L704–731 (the cascade far distance of
//   fitCascades); src/three-openvdb/frame/webgpu/FrameVolumeShadowMaterial.js L27–48 (constants,
//   frameShadowDepth), L90–172 (the Beer shadow map pass).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free TypeScript CPU twins (f64) of his mesh-shadow maths; matrices are column-major
// arrays (three's `elements`). Divergences: none (twins of the ported TSL).

/**
 * K3.6: CPU twins of the volume shadows on meshes (his `FrameVolumeShadows` / `FrameVolumeShadowMaterial`):
 * the Beer shadow map pass (one texel: a light ray marched through the last grid cascade), the receiver
 * lookup (transmittance at a mesh point from the map), the directional-light cascade split and its lookup.
 */

import type { Vec2, Vec3, Vec4 } from "./index.js";

/** Optical depth below this casts no shadow (his `minOpticalDepth`). */
export const frameShadowMinOpticalDepth = 1e-4;
/** The march stops past this optical depth (transmittance below e⁻⁸; his `maxOpticalDepth`). */
export const frameShadowMaxOpticalDepth = 8.0;
/** Upper limit of `castShadowSteps` (his `frameMaxShadowSteps`). */
export const frameMaxCastShadowSteps = 256;
/** Directional lights: cascades of the view, tiles of one atlas (his `frameMaxShadowMapCascades`). */
export const frameMaxShadowMapCascades = 4;
/** Distance in front of the cascade bounds and the volume box, along the light (his `cascadeMargin`). */
export const frameShadowCascadeMargin = 1.0;
/** `castShadowCoarse`: map size and samples (his `coarseMapSize`, `coarseSteps`). */
export const frameShadowCoarseMapSize = 64;
export const frameShadowCoarseSteps = 8;

/** The ray uniforms of one shadow camera (his `createShadowRayUniforms`). */
export interface VDBFrameShadowRay {
  origin: Vec3;
  forward: Vec3;
  /** 1: depth is the distance from the origin (perspective). 0: along `forward` (orthographic). */
  perspective: number;
  /** Far plane of the shadow camera; map depths are divided by it. */
  far: number;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3): number => Math.sqrt(dot(a, a));
const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** frameShadowDepth: depth of a world point for a shadow camera (distance, or along the forward axis). */
export function vdbFrameShadowDepth(ray: Pick<VDBFrameShadowRay, "origin" | "forward" | "perspective">, world: Vec3): number {
  const offset = sub(world, ray.origin);
  return ray.perspective === 1 ? len(offset) : dot(offset, ray.forward);
}

/** Column-major 4×4 times (x, y, z, w). */
function mul4(m: ArrayLike<number>, v: Vec4): Vec4 {
  const out: Vec4 = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) out[r] = m[r]! * v[0] + m[4 + r]! * v[1] + m[8 + r]! * v[2] + m[12 + r]! * v[3];
  return out;
}

/**
 * The pass ray of one map texel (FrameVolumeShadowMaterial L90–109): two unprojected points of the pixel ray at
 * NDC depth 0.25 and 0.75 give the direction; perspective rays start at the camera, orthographic ones on the camera
 * plane. `ndc` is the pixel centre in [−1, 1]² (y up); matrices are the shadow camera's, column-major.
 */
export function vdbFrameShadowPassRay(projection: ArrayLike<number>, projectionInverse: ArrayLike<number>, cameraWorld: ArrayLike<number>, ndc: Vec2): { origin: Vec3; direction: Vec3 } {
  const unproject = (z: number): Vec3 => {
    const view = mul4(projectionInverse, [ndc[0], ndc[1], z, 1]);
    const world = mul4(cameraWorld, [view[0] / view[3], view[1] / view[3], view[2] / view[3], 1]);
    return [world[0], world[1], world[2]];
  };
  const near = unproject(0.25);
  const far = unproject(0.75);
  const d = sub(far, near);
  const l = len(d);
  const direction: Vec3 = [d[0] / l, d[1] / l, d[2] / l];
  const cameraPosition: Vec3 = [cameraWorld[12]!, cameraWorld[13]!, cameraWorld[14]!];
  const perspective = projection[15]! < 0.5;
  if (perspective) return { origin: cameraPosition, direction };
  const along = dot(sub(near, cameraPosition), direction);
  return { origin: [near[0] - direction[0] * along, near[1] - direction[1] * along, near[2] - direction[2] * along], direction };
}

/**
 * One texel of the Beer shadow map (FrameVolumeShadowMaterial L111–172): `steps` samples of the light ray through
 * `box` (the last grid cascade), each at the middle of its step; the march stops past optical depth 8. Returns
 * (τ·first, τ·last, 0, τ) with first / last the depths (÷ `ray.far`) of the first and last non-empty steps; empty
 * texels are 0. `extinctionAt` is the grid extinction (red channel of the shadow density grid) at a world point.
 * Spectral (`extinctionAt` returns rgb): output 0 is (τ̄·first, τ̄·last, 0, τ̄) with τ̄ the channel mean, output 1
 * is (τ rgb, 0).
 */
export function vdbFrameShadowMapTexel(
  extinctionAt: (world: Vec3) => number | Vec3,
  origin: Vec3,
  direction: Vec3,
  box: { min: Vec3; max: Vec3 } | null,
  steps: number,
  ray: VDBFrameShadowRay,
  spectral = false,
): { map: Vec4; spectral: Vec4 | null } {
  const depth: Vec3 = [0, 0, 0];
  const total = () => (spectral ? Math.min(depth[0], depth[1], depth[2]) : depth[0]);
  let first = 0;
  let last = 0;
  if (box) {
    const tMin: number[] = [];
    const tMax: number[] = [];
    for (let c = 0; c < 3; c++) {
      const inv = 1 / direction[c]!;
      const a = (box.min[c]! - origin[c]!) * inv;
      const b = (box.max[c]! - origin[c]!) * inv;
      tMin.push(Math.min(a, b));
      tMax.push(Math.max(a, b));
    }
    const tStart = Math.max(0, Math.max(tMin[0]!, Math.max(tMin[1]!, tMin[2]!)));
    const tEnd = Math.min(tMax[0]!, Math.min(tMax[1]!, tMax[2]!));
    if (tEnd > tStart) {
      const n = Math.max(steps, 1);
      const stepLength = (tEnd - tStart) / n;
      let firstT = -1;
      let lastT = -1;
      for (let i = 0; i < n; i++) {
        if (total() > frameShadowMaxOpticalDepth) break;
        const t = tStart + i * stepLength;
        const s = t + stepLength * 0.5;
        const e = extinctionAt([origin[0] + direction[0] * s, origin[1] + direction[1] * s, origin[2] + direction[2] * s]);
        const ext: Vec3 = typeof e === "number" ? [e, e, e] : e;
        const any = spectral ? ext[0] + ext[1] + ext[2] : ext[0];
        if (any > 0) {
          if (firstT < 0) firstT = t;
          lastT = t + stepLength;
          for (let c = 0; c < 3; c++) depth[c] = depth[c]! + ext[c]! * stepLength;
        }
      }
      if ((spectral ? depth[0] + depth[1] + depth[2] : depth[0]) > 0) {
        const at = (t: number): Vec3 => [origin[0] + direction[0] * t, origin[1] + direction[1] * t, origin[2] + direction[2] * t];
        first = vdbFrameShadowDepth(ray, at(firstT)) / ray.far;
        last = vdbFrameShadowDepth(ray, at(lastT)) / ray.far;
      }
    }
  }
  if (spectral) {
    const mean = (depth[0] + depth[1] + depth[2]) / 3;
    return { map: [first * mean, last * mean, 0, mean], spectral: [depth[0], depth[1], depth[2], 0] };
  }
  return { map: [first * depth[0], last * depth[0], 0, depth[0]], spectral: null };
}

/**
 * The receiver transmittance (FrameVolumeShadows L344–372): the optical depth grows linearly from the first to the
 * last non-empty depth of the (filtered) texel. `depth` is the receiver's {@link vdbFrameShadowDepth}, `far` the
 * drawn shadow camera's far. `spectralTexel` (rgb optical depth): per-channel transmittance; with `mean`, their mean
 * (a `ShadowMaterial` receiver). Returns 1 for an empty texel.
 */
export function vdbFrameShadowReceiver(texel: Vec4, depth: number, far: number, spectralTexel: Vec3 | null = null, mean = false): number | Vec3 {
  const opticalDepth = texel[3];
  if (!(opticalDepth > frameShadowMinOpticalDepth)) return spectralTexel && !mean ? [1, 1, 1] : 1;
  const first = (texel[0] / opticalDepth) * far;
  const last = (texel[1] / opticalDepth) * far;
  const along = clamp((depth - first) / Math.max(last - first, 1e-4), 0, 1);
  if (spectralTexel) {
    const t: Vec3 = [Math.exp(-spectralTexel[0] * along), Math.exp(-spectralTexel[1] * along), Math.exp(-spectralTexel[2] * along)];
    return mean ? (t[0] + t[1] + t[2]) / 3 : t;
  }
  return Math.exp(-opticalDepth * along);
}

/** getCascadeBreaks: far distance of each cascade, the practical split of three's CSMShadowNode (λ = 0.5). */
export function vdbFrameShadowCascadeBreaks(count: number, near: number, far: number): number[] {
  const breaks: number[] = [];
  for (let i = 1; i <= count; i++) {
    const uniformSplit = near + ((far - near) * i) / count;
    const logSplit = near * Math.pow(far / near, i / count);
    breaks.push(i === count ? far : uniformSplit + (logSplit - uniformSplit) * 0.5);
  }
  return breaks;
}

/**
 * The view distance the cascades cover (fitCascades L714–730): the far corner of the volume box plus one box
 * diagonal, rounded up to half octaves (so the splits move only at those steps), within [near·1.0001,
 * min(camera far, castShadowDistance)]. `boxFar` is the largest view depth of the box's corners.
 */
export function vdbFrameShadowCascadeFar(near: number, cameraFar: number, castShadowDistance: number, boxFar: number, boxDiagonal: number): number {
  const sceneFar = Math.max(near * 2.0, boxFar + boxDiagonal);
  const steppedFar = Math.pow(2.0, Math.ceil(Math.log2(sceneFar) * 2.0) / 2.0);
  return Math.max(near * 1.0001, Math.min(cameraFar, castShadowDistance, steppedFar));
}

/** The cascade of a receiver (getCascadeLookup L253–261): the number of cascade far distances in front of it. */
export function vdbFrameShadowCascadeIndex(viewDepth: number, breaks: readonly number[], count: number): number {
  let index = 0;
  for (let i = 0; i < frameMaxShadowMapCascades - 1; i++) if (viewDepth > breaks[i]! && i + 1 < count) index++;
  return index;
}

/**
 * The spot map uv of a receiver (FrameVolumeShadowNode, his 62a6e3e): `clip` is its clip position in the camera that
 * drew the map (the drawn view-projection, not three's current shadow matrix), with three's ShadowNode y flip.
 * Outside when behind that camera (w ≤ 0) or off the map.
 */
export function vdbFrameShadowSpotUv(clip: Vec4): { uv: Vec2; inside: boolean } {
  const uv: Vec2 = [(clip[0] / clip[3]) * 0.5 + 0.5, 0.5 - (clip[1] / clip[3]) * 0.5];
  return { uv, inside: clip[3] > 0 && uv[0] >= 0 && uv[0] <= 1 && uv[1] >= 0 && uv[1] <= 1 };
}

/**
 * The atlas uv of a receiver in cascade `index` (getCascadeLookup L266–282): `clip` is its clip position in the
 * cascade camera (x, y after the divide; orthographic, so w = 1). Half a texel in from each tile side. `inside`
 * also needs the view depth within the last break (checked by the caller).
 */
export function vdbFrameShadowCascadeUv(clip: Vec2, index: number, size: number): { uv: Vec2; insideTile: boolean } {
  const tileUv: Vec2 = [clip[0] * 0.5 + 0.5, 0.5 - clip[1] * 0.5];
  const inset = 0.5 / size;
  const u = clamp(tileUv[0], inset, 1 - inset);
  const v = clamp(tileUv[1], inset, 1 - inset);
  return {
    uv: [(u + index) / frameMaxShadowMapCascades, v],
    insideTile: tileUv[0] >= 0 && tileUv[0] <= 1 && tileUv[1] >= 0 && tileUv[1] <= 1,
  };
}
