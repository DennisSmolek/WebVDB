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

// ---------------------------------------------------------------------------------------------------------------
// K6.5: twins of the core volume shadow kernels (core/frame/wgsl/shadows.ts). The core draws each map texel from the
// inverse view-projection of the camera that drew it (spot, cascades) or from the cube face table (point); receivers
// read the map with that drawn view. Plain arrays, column-major.

/** Column-major 4×4 × (x, y, z, w). */
function apply4(m: ArrayLike<number>, x: number, y: number, z: number, w: number): Vec4 {
  return [m[0]! * x + m[4]! * y + m[8]! * z + m[12]! * w, m[1]! * x + m[5]! * y + m[9]! * z + m[13]! * w, m[2]! * x + m[6]! * y + m[10]! * z + m[14]! * w, m[3]! * x + m[7]! * y + m[11]! * z + m[15]! * w];
}

const normalize3 = (v: Vec3): Vec3 => {
  const l = len(v);
  return [v[0] / l, v[1] / l, v[2] / l];
};

/**
 * The map ray of one texel from the drawing camera's inverse view-projection (the core's `vdbFrameVolumeShadowCs`;
 * the same construction as {@link vdbFrameShadowPassRay}: two unprojected points at NDC depth 0.25 and 0.75 give the
 * direction; perspective rays start at the camera, orthographic ones on the camera plane). `ndc`: the texel centre,
 * x right, y up (texel row r ↔ ndc.y = 1 − 2 (r + ½) / size).
 */
export function vdbFrameShadowMapRay(inverseViewProjection: ArrayLike<number>, cameraPosition: Vec3, perspective: boolean, ndc: Vec2): { origin: Vec3; direction: Vec3 } {
  const unproject = (z: number): Vec3 => {
    const p = apply4(inverseViewProjection, ndc[0], ndc[1], z, 1);
    return [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
  };
  const near = unproject(0.25);
  const direction = normalize3(sub(unproject(0.75), near));
  if (perspective) return { origin: [...cameraPosition], direction };
  const along = dot(sub(near, cameraPosition), direction);
  return { origin: [near[0] - direction[0] * along, near[1] - direction[1] * along, near[2] - direction[2] * along], direction };
}

/**
 * The (unnormalized) direction of a cube map texel: face 0–5 (+x, −x, +y, −y, +z, −z), `a`, `b` in [−1, 1] the
 * face coordinates (s, t) × 2 − 1 with t down the rows. WebGPU's cube face selection (the Vulkan table): a cube view
 * sampled with this direction reads this texel.
 */
export function vdbFrameCubeFaceDirection(face: number, a: number, b: number): Vec3 {
  switch (face) {
    case 0:
      return [1, -b, -a];
    case 1:
      return [-1, -b, a];
    case 2:
      return [a, 1, b];
    case 3:
      return [a, -1, -b];
    case 4:
      return [a, -b, 1];
    default:
      return [-a, -b, -1];
  }
}

/** The inverse of {@link vdbFrameCubeFaceDirection}: the face and its (s, t) in [0, 1] (t down the rows) of a direction. */
export function vdbFrameCubeFaceUv(d: Vec3): { face: number; s: number; t: number } {
  const ax = Math.abs(d[0]);
  const ay = Math.abs(d[1]);
  const az = Math.abs(d[2]);
  let face: number;
  let sc: number;
  let tc: number;
  let ma: number;
  if (ax >= ay && ax >= az) {
    ma = ax;
    face = d[0] >= 0 ? 0 : 1;
    sc = d[0] >= 0 ? -d[2] : d[2];
    tc = -d[1];
  } else if (ay >= az) {
    ma = ay;
    face = d[1] >= 0 ? 2 : 3;
    sc = d[0];
    tc = d[1] >= 0 ? d[2] : -d[2];
  } else {
    ma = az;
    face = d[2] >= 0 ? 4 : 5;
    sc = d[2] >= 0 ? d[0] : -d[0];
    tc = -d[1];
  }
  return { face, s: (sc / ma + 1) / 2, t: (tc / ma + 1) / 2 };
}

/** General 4×4 inverse (f64; column-major). */
function invert4(m: ArrayLike<number>): Float64Array {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = Array.from({ length: 16 }, (_, i) => m[i]!) as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  const inv = det !== 0 ? 1 / det : 0;
  return Float64Array.from([
    (a11 * b11 - a12 * b10 + a13 * b09) * inv,
    (a02 * b10 - a01 * b11 - a03 * b09) * inv,
    (a31 * b05 - a32 * b04 + a33 * b03) * inv,
    (a22 * b04 - a21 * b05 - a23 * b03) * inv,
    (a12 * b08 - a10 * b11 - a13 * b07) * inv,
    (a00 * b11 - a02 * b08 + a03 * b07) * inv,
    (a32 * b02 - a30 * b05 - a33 * b01) * inv,
    (a20 * b05 - a22 * b02 + a23 * b01) * inv,
    (a10 * b10 - a11 * b08 + a13 * b06) * inv,
    (a01 * b08 - a00 * b10 - a03 * b06) * inv,
    (a30 * b04 - a31 * b02 + a33 * b00) * inv,
    (a21 * b02 - a20 * b04 - a23 * b00) * inv,
    (a11 * b07 - a10 * b09 - a12 * b06) * inv,
    (a00 * b09 - a01 * b07 + a02 * b06) * inv,
    (a31 * b01 - a30 * b03 - a32 * b00) * inv,
    (a20 * b03 - a21 * b01 + a22 * b00) * inv,
  ]);
}

/**
 * The drawing camera of a view-projection (spot lights, explicit cascades): its position and forward axis, and the
 * inverse. Perspective: the centre is VP⁻¹ (0, 0, 1, 0) (clip w = 0 on the camera); orthographic: the near plane centre
 * moved back by `near` along the forward axis. Forward: from the near plane centre to the far plane centre.
 */
export function vdbFrameShadowViewCamera(viewProjection: ArrayLike<number>, near: number, orthographic: boolean): { inverse: Float64Array; position: Vec3; forward: Vec3 } {
  const inverse = invert4(viewProjection);
  const at = (x: number, y: number, z: number, w: number): Vec3 => {
    const p = apply4(inverse, x, y, z, w);
    return [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
  };
  const nearCenter = at(0, 0, 0, 1);
  const forward = normalize3(sub(at(0, 0, 1, 1), nearCenter));
  const position: Vec3 = orthographic ? [nearCenter[0] - forward[0] * near, nearCenter[1] - forward[1] * near, nearCenter[2] - forward[2] * near] : at(0, 0, 1, 0);
  return { inverse, position, forward };
}

/** Inputs of {@link vdbFrameShadowFitCascades}: the view camera, the light and the casters (the last grid cascade). */
export interface VDBFrameShadowFitInputs {
  /** View → world of the camera (three's `camera.matrixWorld`). */
  cameraWorld: ArrayLike<number>;
  /** Clip → view of the camera (three's `camera.projectionMatrixInverse`). */
  projectionInverse: ArrayLike<number>;
  near: number;
  far: number;
  orthographic: boolean;
  castShadowDistance: number;
  /** Cascades (clamped to 1..4). */
  count: number;
  /** Map texels per side (the snap). */
  size: number;
  /** The way the light travels (normalized here). */
  lightDirection: Vec3;
  /** The box of all casters (the last grid cascade). */
  casterBox: { min: Vec3; max: Vec3 };
}

/** One fitted cascade camera: its view-projection (WebGPU depth 0..1), position (the ray origin) and far plane. */
export interface VDBFrameShadowCascadeCamera {
  viewProjection: Float64Array;
  origin: Vec3;
  far: number;
}

/**
 * His `fitCascades` (FrameVolumeShadows L692–778; three's CSMShadowNode method), engine-free: the practical split of
 * the view depth up to {@link vdbFrameShadowCascadeFar}; per slice a light-aligned orthographic camera sized from the
 * slice diagonal, its centre snapped to texels, its depth range over the slice and the casters (± 1). Returns the
 * cascades, the breaks (padded to 4 with the last ones, as his `cascadeBreaks`) and the light's forward axis.
 */
export function vdbFrameShadowFitCascades(o: VDBFrameShadowFitInputs): { count: number; breaks: [number, number, number, number]; forward: Vec3; cascades: VDBFrameShadowCascadeCamera[] } {
  const count = Math.min(frameMaxShadowMapCascades, Math.max(1, Math.round(o.count)));
  const view = invert4(o.cameraWorld);
  const box = o.casterBox;
  let boxFar = 0;
  for (let i = 0; i < 8; i++) {
    const p = apply4(view, i & 1 ? box.max[0] : box.min[0], i & 2 ? box.max[1] : box.min[1], i & 4 ? box.max[2] : box.min[2], 1);
    boxFar = Math.max(boxFar, -p[2]);
  }
  const far = vdbFrameShadowCascadeFar(o.near, o.far, o.castShadowDistance, boxFar, len(sub(box.max, box.min)));
  const breaks = vdbFrameShadowCascadeBreaks(count, o.near, far);

  // Light view: three's Matrix4.lookAt(origin, lightDirection, up): z = −direction, x = up × z, y = z × x
  const direction = normalize3(o.lightDirection);
  const up: Vec3 = Math.abs(direction[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const zAxis: Vec3 = [-direction[0], -direction[1], -direction[2]];
  const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const xAxis = normalize3(cross(up, zAxis));
  const yAxis = cross(zAxis, xAxis);
  // world → light (the transpose of the rotation) and back
  const toLight = (p: Vec3): Vec3 => [dot(p, xAxis), dot(p, yAxis), dot(p, zAxis)];
  const fromLight = (p: Vec3): Vec3 => [xAxis[0] * p[0] + yAxis[0] * p[1] + zAxis[0] * p[2], xAxis[1] * p[0] + yAxis[1] * p[1] + zAxis[1] * p[2], xAxis[2] * p[0] + yAxis[2] * p[1] + zAxis[2] * p[2]];
  const bounds = (points: Vec3[]) => {
    const lo: Vec3 = [Infinity, Infinity, Infinity];
    const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const q of points) {
      const p = toLight(q);
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k]!, p[k]!);
        hi[k] = Math.max(hi[k]!, p[k]!);
      }
    }
    return { lo, hi };
  };
  const caster = bounds(Array.from({ length: 8 }, (_, i) => [i & 1 ? box.max[0] : box.min[0], i & 2 ? box.max[1] : box.min[1], i & 4 ? box.max[2] : box.min[2]] as Vec3));

  // getSliceCorners: corners of the view at a depth, in world space (corners 0 and 2 are diagonal)
  const corners = (depth: number): Vec3[] =>
    (
      [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ] as const
    ).map(([x, y]) => {
      const q = apply4(o.projectionInverse, x, y, 0.5, 1);
      let p: Vec3 = [q[0] / q[3], q[1] / q[3], q[2] / q[3]];
      if (!o.orthographic) {
        const s = depth / -p[2];
        p = [p[0] * s, p[1] * s, p[2] * s];
      } else p = [p[0], p[1], -depth];
      const w = apply4(o.cameraWorld, p[0], p[1], p[2], 1);
      return [w[0], w[1], w[2]] as Vec3;
    });

  const cascades: VDBFrameShadowCascadeCamera[] = [];
  for (let i = 0; i < count; i++) {
    const nearCorners = corners(i === 0 ? o.near : breaks[i - 1]!);
    const farCorners = corners(breaks[i]!);
    const width = Math.max(len(sub(farCorners[0]!, farCorners[2]!)), len(sub(farCorners[0]!, nearCorners[2]!)));
    const texel = width / o.size;
    const slice = bounds([...nearCorners, ...farCorners]);
    const centerX = Math.floor((slice.lo[0] + slice.hi[0]) * 0.5 / texel) * texel;
    const centerY = Math.floor((slice.lo[1] + slice.hi[1]) * 0.5 / texel) * texel;
    const front = Math.max(slice.hi[2], caster.hi[2]) + frameShadowCascadeMargin;
    const back = Math.min(slice.lo[2], caster.lo[2]) - frameShadowCascadeMargin;
    const depth = front - back;
    const position = fromLight([centerX, centerY, front]);
    // camera world = [R | position]; its inverse: [Rᵀ | −Rᵀ position]
    const viewMatrix = [xAxis[0], yAxis[0], zAxis[0], 0, xAxis[1], yAxis[1], zAxis[1], 0, xAxis[2], yAxis[2], zAxis[2], 0, -dot(xAxis, position), -dot(yAxis, position), -dot(zAxis, position), 1];
    // three's makeOrthographic (WebGPU: depth 0..1): left/right/top/bottom ±width/2, near 0, far = depth
    const sx = 2 / width;
    const projection = [sx, 0, 0, 0, 0, sx, 0, 0, 0, 0, -1 / depth, 0, 0, 0, 0, 1];
    const vp = new Float64Array(16);
    for (let col = 0; col < 4; col++)
      for (let row = 0; row < 4; row++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += projection[k * 4 + row]! * viewMatrix[col * 4 + k]!;
        vp[col * 4 + row] = s;
      }
    cascades.push({ viewProjection: vp, origin: position, far: depth });
  }
  return {
    count,
    breaks: [breaks[Math.min(0, count - 1)]!, breaks[Math.min(1, count - 1)]!, breaks[Math.min(2, count - 1)]!, breaks[Math.min(3, count - 1)]!],
    forward: direction,
    cascades,
  };
}

/** The drawn view of a core volume shadow map, as `VdbVolumeShadowView` holds it (core/frame/wgsl/shadows.ts). */
export interface VDBFrameVolumeShadowViewCpu {
  /** 0: directional cascades, 1: spot, 2: point (cube). */
  kind: number;
  /** One per layer (cascades); spot: [0]. */
  viewProjection: readonly ArrayLike<number>[];
  /** Ray origin per layer (cascades) or [0]. */
  origin: readonly Vec3[];
  /** Far plane per layer (map depths ÷ far). */
  far: readonly number[];
  forward: Vec3;
  perspective: boolean;
  breaks: readonly number[];
  cascadeCount: number;
  size: number;
  enabled: boolean;
}

/**
 * The core receiver (`vdbVolumeShadow` / `vdbVolumeShadowCube`), CPU: picks the layer and uv from the drawn view
 * (cascade by view depth, spot by the drawn view-projection, point by the cube face of the direction from the drawn
 * light position), reads the map with `sample(layer, s, t)` (filtered; s, t in [0, 1], t down the rows) and applies
 * {@link vdbFrameShadowReceiver}. 1 outside the map or when disabled.
 */
export function vdbFrameVolumeShadowLookup(view: VDBFrameVolumeShadowViewCpu, world: Vec3, viewDepth: number, sample: (layer: number, s: number, t: number) => Vec4): number {
  if (!view.enabled) return 1;
  if (view.kind === 2) {
    const offset = sub(world, view.origin[0]!);
    const f = vdbFrameCubeFaceUv(offset);
    return vdbFrameShadowReceiver(sample(f.face, f.s, f.t), len(offset), view.far[0]!) as number;
  }
  let layer = 0;
  let uv: Vec2;
  if (view.kind === 0) {
    layer = vdbFrameShadowCascadeIndex(viewDepth, view.breaks, view.cascadeCount);
    const clip = mul4(view.viewProjection[layer]!, [world[0], world[1], world[2], 1]);
    const tile: Vec2 = [clip[0] * 0.5 + 0.5, 0.5 - clip[1] * 0.5];
    const lastBreak = view.breaks[Math.max(0, Math.min(3, view.cascadeCount - 1))]!;
    if (!(viewDepth <= lastBreak && tile[0] >= 0 && tile[0] <= 1 && tile[1] >= 0 && tile[1] <= 1)) return 1;
    const inset = 0.5 / view.size;
    uv = [clamp(tile[0], inset, 1 - inset), clamp(tile[1], inset, 1 - inset)];
  } else {
    const spot = vdbFrameShadowSpotUv(mul4(view.viewProjection[0]!, [world[0], world[1], world[2], 1]));
    if (!spot.inside) return 1;
    uv = spot.uv;
  }
  const depth = vdbFrameShadowDepth({ origin: view.origin[layer]!, forward: view.forward, perspective: view.perspective ? 1 : 0 }, world);
  return vdbFrameShadowReceiver(sample(layer, uv[0], uv[1]), depth, view.far[layer]!) as number;
}
