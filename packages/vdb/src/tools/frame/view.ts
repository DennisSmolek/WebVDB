// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameView.js (frustum clip of the volume boxes, clipped view depths,
//   getFroxelSlice, getFroxelSliceRange, getFroxelRect, getFroxelArea) and
//   src/three-openvdb/frame/core/frameLayout.js (getAtlasLayout, getFroxelSize);
//   three.js r187 Frustum.setFromProjectionMatrix / intersectsBox, Box3.applyMatrix4 (MIT).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: plain arrays instead of three's math classes (engine-free, K6.1); `getFroxelSize` takes the
// drawing-buffer and canvas sizes instead of a renderer; a union of two froxel areas (the core's dispatch
// box). Divergences: DIVERGES.md (K3.1 rows; K6.1 rows).

/**
 * Engine-free froxel view maths (K6.1; design §2.4, §3.2). The same numbers as
 * `@pmndrs/vdb/three` `frame/view.ts` and `frame/layout.ts` (a unit test runs both on random
 * cameras and boxes), on column-major `ArrayLike<number>` matrices. The core frame renderer imports
 * these; they are also the twins of its CPU work.
 */

import { vdbFrameSliceRange } from "./index.js";

/** Column-major 4x4. */
export type VDBMat4 = ArrayLike<number>;
type V3 = [number, number, number];
/** A plane `normal · p + constant = 0` (normalized). */
interface Plane {
  n: V3;
  c: number;
}

/** `a × b`, column-major 4x4. */
export function vdbMat4Multiply(a: VDBMat4, b: VDBMat4, out: Float64Array = new Float64Array(16)): Float64Array {
  const r = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let row = 0; row < 4; row++) {
      r[c * 4 + row] = a[row]! * b[c * 4]! + a[4 + row]! * b[c * 4 + 1]! + a[8 + row]! * b[c * 4 + 2]! + a[12 + row]! * b[c * 4 + 3]!;
    }
  }
  out.set(r);
  return out;
}

/** The inverse of a column-major 4x4 (three's `Matrix4.invert`; the zero matrix when singular). */
export function vdbMat4Invert(m: VDBMat4, out: Float64Array = new Float64Array(16)): Float64Array {
  const [n11, n21, n31, n41, n12, n22, n32, n42, n13, n23, n33, n43, n14, n24, n34, n44] = Array.from(m) as number[] as [
    number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number,
  ];
  const t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
  const t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
  const t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
  const t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;
  const det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
  if (det === 0) {
    out.fill(0);
    return out;
  }
  const d = 1 / det;
  const r = [
    t11 * d,
    (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * d,
    (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * d,
    (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * d,
    t12 * d,
    (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * d,
    (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * d,
    (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * d,
    t13 * d,
    (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * d,
    (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * d,
    (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * d,
    t14 * d,
    (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * d,
    (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * d,
    (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * d,
  ];
  out.set(r);
  return out;
}

/** `m · (x, y, z, 1)` divided by its w (three's `Vector3.applyMatrix4`). */
export function vdbMat4Point(m: VDBMat4, p: readonly number[]): V3 {
  const [x, y, z] = p as V3;
  const w = 1 / (m[3]! * x + m[7]! * y + m[11]! * z + m[15]!);
  return [(m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * w, (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * w, (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * w];
}

/** `m · (x, y, z, w)`. */
export function vdbMat4Vec4(m: VDBMat4, v: readonly number[]): [number, number, number, number] {
  const o: [number, number, number, number] = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) o[r] = m[r]! * v[0]! + m[4 + r]! * v[1]! + m[8 + r]! * v[2]! + m[12 + r]! * v[3]!;
  return o;
}

/** The largest axis scale of a column-major 4x4 (three's `Matrix4.getMaxScaleOnAxis`). */
export function vdbMat4MaxScale(m: VDBMat4): number {
  const sx = m[0]! * m[0]! + m[1]! * m[1]! + m[2]! * m[2]!;
  const sy = m[4]! * m[4]! + m[5]! * m[5]! + m[6]! * m[6]!;
  const sz = m[8]! * m[8]! + m[9]! * m[9]! + m[10]! * m[10]!;
  return Math.sqrt(Math.max(sx, sy, sz));
}

const plane = (x: number, y: number, z: number, w: number): Plane => {
  const inv = 1 / Math.hypot(x, y, z);
  return { n: [x * inv, y * inv, z * inv], c: w * inv };
};
const distance = (p: Plane, v: readonly number[]) => p.n[0] * v[0]! + p.n[1] * v[1]! + p.n[2] * v[2]! + p.c;

/** The frustum of a projection (or view-projection), three's `Frustum.setFromProjectionMatrix` (planes 0–5). */
export function vdbFrustumPlanes(m: VDBMat4, coordinateSystem: "webgl" | "webgpu" = "webgl", reversedDepth = false): Plane[] {
  const e = m;
  const planes = [
    plane(e[3]! - e[0]!, e[7]! - e[4]!, e[11]! - e[8]!, e[15]! - e[12]!),
    plane(e[3]! + e[0]!, e[7]! + e[4]!, e[11]! + e[8]!, e[15]! + e[12]!),
    plane(e[3]! + e[1]!, e[7]! + e[5]!, e[11]! + e[9]!, e[15]! + e[13]!),
    plane(e[3]! - e[1]!, e[7]! - e[5]!, e[11]! - e[9]!, e[15]! - e[13]!),
  ];
  if (reversedDepth) {
    planes.push(plane(e[2]!, e[6]!, e[10]!, e[14]!), plane(e[3]! - e[2]!, e[7]! - e[6]!, e[11]! - e[10]!, e[15]! - e[14]!));
  } else {
    planes.push(plane(e[3]! - e[2]!, e[7]! - e[6]!, e[11]! - e[10]!, e[15]! - e[14]!));
    planes.push(coordinateSystem === "webgl" ? plane(e[3]! + e[2]!, e[7]! + e[6]!, e[11]! + e[10]!, e[15]! + e[14]!) : plane(e[2]!, e[6]!, e[10]!, e[14]!));
  }
  return planes;
}

/** three's `Frustum.intersectsBox`. */
function intersectsBox(planes: readonly Plane[], min: V3, max: V3): boolean {
  for (const p of planes) {
    const v: V3 = [p.n[0] > 0 ? max[0] : min[0], p.n[1] > 0 ? max[1] : min[1], p.n[2] > 0 ? max[2] : min[2]];
    if (distance(p, v) < 0) return false;
  }
  return true;
}

/** The world box of the unit box [-0.5, 0.5]³ under `m` (three's `Box3.applyMatrix4`). */
export function vdbUnitBoxBounds(m: VDBMat4): { min: V3; max: V3 } {
  const min: V3 = [Infinity, Infinity, Infinity];
  const max: V3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i++) {
    const p = vdbMat4Point(m, [i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5]);
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a]!, p[a]!);
      max[a] = Math.max(max[a]!, p[a]!);
    }
  }
  return { min, max };
}

// NOTE Point where three planes meet (normal . p + constant = 0 for each)
function intersectPlanes(a: Plane, b: Plane, c: Plane): V3 {
  const cross = (u: V3, v: V3): V3 => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const ca = cross(b.n, c.n);
  const cb = cross(c.n, a.n);
  const cc = cross(a.n, b.n);
  const d = a.n[0] * ca[0] + a.n[1] * ca[1] + a.n[2] * ca[2];
  return [0, 1, 2].map((i) => (ca[i]! * -a.c + cb[i]! * -b.c + cc[i]! * -c.c) / d) as V3;
}

// NOTE Corners of each face of the unit box, in order around the face (corner bits: 1 x, 2 y, 4 z)
const BOX_FACES = [
  [0, 2, 6, 4],
  [1, 3, 7, 5],
  [0, 1, 5, 4],
  [2, 3, 7, 6],
  [0, 1, 3, 2],
  [4, 5, 7, 6],
];

const inUnitBox = (p: V3) => p[0] >= -0.5 && p[0] <= 0.5 && p[1] >= -0.5 && p[1] <= 0.5 && p[2] >= -0.5 && p[2] <= 0.5;

/**
 * The per-camera state of the volume clip (his `FrameVolumeCollector.collect` set-up): the world frustum
 * (WebGL convention, as his `frustum.setFromProjectionMatrix(projectionScreenMatrix)`), the view frustum
 * (the camera's coordinate system and depth direction) and its 8 corners.
 */
export interface VDBFrameClipCamera {
  readonly world: Plane[];
  readonly view: Plane[];
  readonly corners: V3[];
  readonly viewMatrix: VDBMat4;
  readonly projection: VDBMat4;
}

/**
 * @param view World → view (three's `camera.matrixWorldInverse`), column-major.
 * @param projection The projection (three's `camera.projectionMatrix`), column-major.
 * @param o The projection's coordinate system (default `'webgpu'`, NDC z in [0, 1]) and depth direction.
 */
export function vdbFrameClipCamera(view: VDBMat4, projection: VDBMat4, o: { coordinateSystem?: "webgl" | "webgpu"; reversedDepth?: boolean } = {}): VDBFrameClipCamera {
  const world = vdbFrustumPlanes(vdbMat4Multiply(projection, view), "webgl", false);
  const viewPlanes = vdbFrustumPlanes(projection, o.coordinateSystem ?? "webgpu", o.reversedDepth ?? false);
  const corners: V3[] = [];
  for (let i = 0; i < 8; i++) corners.push(intersectPlanes(viewPlanes[i & 1]!, viewPlanes[2 + ((i >> 1) & 1)]!, viewPlanes[4 + ((i >> 2) & 1)]!));
  return { world, view: viewPlanes, corners, viewMatrix: view, projection };
}

/** One volume after the clip: its matrices and the view depth range of its part inside the view frustum. */
export interface VDBFrameClippedVolume {
  readonly localToView: Float64Array;
  readonly viewToLocal: Float64Array;
  readonly worldToLocal: Float64Array;
  readonly viewNear: number;
  readonly viewFar: number;
}

/**
 * His `FrameVolumeCollector.addVolume` for one volume box (`matrixWorld`: the local unit box → world):
 * world-frustum culling, then the view depth range of the part of the box inside the view frustum
 * (faces clipped to the planes, plus the frustum corners inside the box). Null when no part is in view.
 */
export function vdbFrameClipVolume(cam: VDBFrameClipCamera, matrixWorld: VDBMat4): VDBFrameClippedVolume | null {
  const worldToLocal = vdbMat4Invert(matrixWorld);
  const box = vdbUnitBoxBounds(matrixWorld);
  if (!intersectsBox(cam.world, box.min, box.max)) return null;
  const localToView = vdbMat4Multiply(cam.viewMatrix, matrixWorld);
  const viewToLocal = vdbMat4Invert(localToView);
  const boxCorners: V3[] = [];
  for (let i = 0; i < 8; i++) boxCorners.push(vdbMat4Point(localToView, [i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5]));
  let near = Infinity;
  let far = -Infinity;
  for (const corner of cam.corners) {
    if (inUnitBox(vdbMat4Point(viewToLocal, corner))) {
      near = Math.min(near, -corner[2]);
      far = Math.max(far, -corner[2]);
    }
  }
  for (const face of BOX_FACES) {
    let input: V3[] = face.map((i) => boxCorners[i]!);
    for (let p = 0; p < 6 && input.length > 0; p++) {
      const pl = cam.view[p]!;
      const output: V3[] = [];
      for (let i = 0; i < input.length; i++) {
        const a = input[i]!;
        const b = input[(i + 1) % input.length]!;
        const da = distance(pl, a);
        const db = distance(pl, b);
        if (da >= 0) output.push([...a] as V3);
        if (da >= 0 !== db >= 0) {
          const t = da / (da - db);
          output.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
        }
      }
      input = output;
    }
    for (const v of input) {
      near = Math.min(near, -v[2]);
      far = Math.max(far, -v[2]);
    }
  }
  if (!(near <= far)) return null;
  return { localToView, viewToLocal, worldToLocal, viewNear: near, viewFar: far };
}

/** A froxel rectangle in cells: `[x0, y0, x1, y1)`, bottom-left origin. */
export type VDBFroxelRect = [number, number, number, number];

/**
 * His `getFroxelRect`: the screen rectangle of the volume box in froxel cells, one cell of padding for
 * the linear filter; the whole grid when a corner is behind the camera. Null when it covers no cell.
 */
export function vdbFrameFroxelRect(localToView: VDBMat4, projection: VDBMat4, width: number, height: number): VDBFroxelRect | null {
  const clip = vdbMat4Multiply(projection, localToView);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
    const c = vdbMat4Vec4(clip, [i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5, 1]);
    if (c[3] <= 1e-6) return [0, 0, width, height];
    minX = Math.min(minX, c[0] / c[3]);
    minY = Math.min(minY, c[1] / c[3]);
    maxX = Math.max(maxX, c[0] / c[3]);
    maxY = Math.max(maxY, c[1] / c[3]);
  }
  const x0 = Math.max(0, Math.floor((minX * 0.5 + 0.5) * width) - 1);
  const y0 = Math.max(0, Math.floor((minY * 0.5 + 0.5) * height) - 1);
  const x1 = Math.min(width, Math.ceil((maxX * 0.5 + 0.5) * width) + 1);
  const y1 = Math.min(height, Math.ceil((maxY * 0.5 + 0.5) * height) + 1);
  if (x1 <= x0 || y1 <= y0) return null;
  return [x0, y0, x1, y1];
}

/** His froxel area: the rectangle and slices of all volumes, one cell and one slice larger on each side. */
export interface VDBFroxelArea {
  rect: VDBFroxelRect;
  first: number;
  count: number;
}

/** The empty area (count 0). */
export const vdbFroxelAreaEmpty = (): VDBFroxelArea => ({ rect: [0, 0, 0, 0], first: 0, count: 0 });

/**
 * His `getFroxelArea` over volumes that each already have a rect (`vdbFrameFroxelRect`) and a slice range
 * (`vdbFrameSliceRange`); volumes without a rect are skipped. Count 0 when none covers a cell.
 */
export function vdbFrameFroxelArea(
  volumes: readonly { rect: VDBFroxelRect | null; first: number; count: number }[],
  width: number,
  height: number,
  depth: number,
): VDBFroxelArea {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  let first = Infinity;
  let last = -Infinity;
  for (const v of volumes) {
    if (!v.rect) continue;
    x0 = Math.min(x0, v.rect[0]);
    y0 = Math.min(y0, v.rect[1]);
    x1 = Math.max(x1, v.rect[2]);
    y1 = Math.max(y1, v.rect[3]);
    first = Math.min(first, v.first);
    last = Math.max(last, v.first + v.count - 1);
  }
  if (first > last) return vdbFroxelAreaEmpty();
  const f = Math.max(0, first - 1);
  return { rect: [Math.max(0, x0 - 1), Math.max(0, y0 - 1), Math.min(width, x1 + 1), Math.min(height, y1 + 1)], first: f, count: Math.min(depth - 1, last + 1) - f + 1 };
}

/** The slices of a volume (his `getFroxelSliceRange`): `{ first, count }`. */
export function vdbFrameVolumeSlices(viewNear: number, viewFar: number, range: readonly [number, number], depth: number): { first: number; count: number } {
  return vdbFrameSliceRange(viewNear, viewFar, [range[0], range[1]], depth);
}

/**
 * The bounding box of two areas (rect union, slice union), for a dispatch that must also clear what the
 * last dispatch wrote. An empty area does not widen it.
 */
export function vdbFroxelAreaUnion(a: VDBFroxelArea, b: VDBFroxelArea): VDBFroxelArea {
  if (b.count <= 0 || b.rect[2] <= b.rect[0] || b.rect[3] <= b.rect[1]) return { rect: [...a.rect], first: a.first, count: a.count };
  if (a.count <= 0 || a.rect[2] <= a.rect[0] || a.rect[3] <= a.rect[1]) return { rect: [...b.rect], first: b.first, count: b.count };
  const first = Math.min(a.first, b.first);
  const end = Math.max(a.first + a.count, b.first + b.count);
  return {
    rect: [Math.min(a.rect[0], b.rect[0]), Math.min(a.rect[1], b.rect[1]), Math.max(a.rect[2], b.rect[2]), Math.max(a.rect[3], b.rect[3])],
    first,
    count: end - first,
  };
}

/** His `getAtlasLayout`: tiles of tileWidth × tileHeight in an atlas that is about square. */
export function vdbFrameAtlasLayout(tileWidth: number, tileHeight: number, tileCount: number): { columns: number; width: number; height: number } {
  const columns = Math.ceil(Math.sqrt((tileCount * tileHeight) / tileWidth));
  const rows = Math.ceil(tileCount / columns);
  return { columns, width: tileWidth * columns, height: tileHeight * rows };
}

/** Held sizes of {@link vdbFrameFroxelSize} (his frameLayout.js state). */
export interface VDBFroxelHeldSize {
  canvasX: number;
  canvasY: number;
  bufferX: number;
  bufferY: number;
}

/** A fresh held size (nothing held). */
export const vdbFroxelHeldSize = (): VDBFroxelHeldSize => ({ canvasX: 0, canvasY: 0, bufferX: 0, bufferY: 0 });

// NOTE A size that changes by less than this share is kept (mobile address bars).
const STEADY_CHANGE = 0.15;
const steady = (size: number, held: number) => (held > 0 && Math.abs(size - held) <= held * STEADY_CHANGE ? held : size);

/**
 * His `getFroxelSize`: the settings' width and height, at most the drawing-buffer pixels and, with
 * `pixelsPerFroxel` N > 0, at most one froxel per N × N canvas pixels; `state` holds sizes through
 * small changes.
 */
export function vdbFrameFroxelSize(
  settings: { width: number; height: number; pixelsPerFroxel: number },
  viewport: { drawingBuffer: readonly [number, number]; canvas: readonly [number, number] },
  state: VDBFroxelHeldSize | null = null,
): { width: number; height: number } {
  let [bx, by] = viewport.drawingBuffer;
  let capX = bx;
  let capY = by;
  if (settings.pixelsPerFroxel > 0) {
    let [cx, cy] = viewport.canvas;
    if (state) {
      state.canvasX = steady(cx, state.canvasX);
      state.canvasY = steady(cy, state.canvasY);
      state.bufferX = steady(bx, state.bufferX);
      state.bufferY = steady(by, state.bufferY);
      [cx, cy, bx, by] = [state.canvasX, state.canvasY, state.bufferX, state.bufferY];
    }
    capX = Math.min(bx, Math.ceil(cx / settings.pixelsPerFroxel));
    capY = Math.min(by, Math.ceil(cy / settings.pixelsPerFroxel));
  } else if (state) {
    state.canvasX = state.canvasY = state.bufferX = state.bufferY = 0;
  }
  return { width: Math.max(1, Math.min(Math.round(settings.width), capX)), height: Math.max(1, Math.min(Math.round(settings.height), capY)) };
}

/**
 * The clip rectangle of a froxel cell rectangle (his `getClipRect`), `padding` cells larger on each side:
 * (x0, y0, x1, y1) in clip space; all 0 for an empty rectangle.
 */
export function vdbFrameClipRect(rect: VDBFroxelRect, width: number, height: number, padding: number): [number, number, number, number] {
  if (rect[2] <= rect[0] || rect[3] <= rect[1]) return [0, 0, 0, 0];
  return [
    (Math.max(0, rect[0] - padding) / width) * 2 - 1,
    (Math.max(0, rect[1] - padding) / height) * 2 - 1,
    (Math.min(width, rect[2] + padding) / width) * 2 - 1,
    (Math.min(height, rect[3] + padding) / height) * 2 - 1,
  ];
}
