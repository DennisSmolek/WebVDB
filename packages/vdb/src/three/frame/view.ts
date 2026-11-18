// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameView.js (all, 357 lines).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; the collector visits `isVDBFrameVolume` objects and has no shadow-grid
// casters' mip query beyond `hasMipmaps`. Divergences: DIVERGES.md (K3.1 rows).

/**
 * View code of the froxel renderer, the same for all backends (R1 bucket A: frustum clip of the
 * volume boxes, log slice ranges, froxel rectangle and area, distance fade radius) plus the scene
 * traversal (`FrameVolumeCollector`, bucket B). It makes no objects per frame. The engine-free part
 * moves to `core` in K3.8.
 */

import { Box3, Frustum, Matrix4, Vector3, Vector4, type Camera, type Object3D, type Plane, type Vector2Like } from "three";

/** One volume of a frame volume: the per-render state his `FrameVolume.volumes[i]` holds. */
export interface FrameVolumeRecord {
  readonly matrix: Matrix4;
  readonly matrixWorld: Matrix4;
  readonly localToView: Matrix4;
  readonly viewToLocal: Matrix4;
  readonly worldToLocal: Matrix4;
  readonly worldBox: Box3;
  viewNear: number;
  viewFar: number;
  ready: boolean;
}

/** What the collector needs of a frame volume. */
export interface FrameVolumeLike extends Object3D {
  readonly isVDBFrameVolume: true;
  readonly proxy: Object3D;
  readonly volumes: readonly FrameVolumeRecord[];
  readonly mipmaps: boolean;
}

export interface FrameVolumeEntry<V extends FrameVolumeLike = FrameVolumeLike> {
  frameVolume: V;
  volume: FrameVolumeRecord;
}

const unitBox = new Box3(new Vector3(-0.5, -0.5, -0.5), new Vector3(0.5, 0.5, 0.5));
const frustum = new Frustum();
const projectionScreenMatrix = new Matrix4();
const clipMatrix = new Matrix4();
const corner = new Vector4();
const viewFrustum = new Frustum();
const boxCorners = Array.from({ length: 8 }, () => new Vector3());
const clipPolygons = [Array.from({ length: 12 }, () => new Vector3()), Array.from({ length: 12 }, () => new Vector3())];
const clipDepths = { near: 0.0, far: 0.0 };
const frustumCorners = Array.from({ length: 8 }, () => new Vector3());
const localCorner = new Vector3();
const crossA = new Vector3();
const crossB = new Vector3();
const crossC = new Vector3();

// NOTE Point where three planes meet (normal . p + constant = 0 for each). Writes target.
const intersectPlanes = (a: Plane, b: Plane, c: Plane, target: Vector3) => {
  crossA.crossVectors(b.normal, c.normal);
  crossB.crossVectors(c.normal, a.normal);
  crossC.crossVectors(a.normal, b.normal);

  return target
    .copy(crossA)
    .multiplyScalar(-a.constant)
    .addScaledVector(crossB, -b.constant)
    .addScaledVector(crossC, -c.constant)
    .divideScalar(a.normal.dot(crossA));
};

// NOTE Corners of the view space frustum. The planes of Frustum are in pairs: x (0, 1), y (2, 3) and z (4, 5).
const setFrustumCorners = (planes: Plane[]) => {
  for (let i = 0; i < 8; i++) {
    intersectPlanes(planes[i & 1]!, planes[2 + ((i >> 1) & 1)]!, planes[4 + ((i >> 2) & 1)]!, frustumCorners[i]!);
  }
};

// NOTE Corners of each face of the unit box, in order around the face (corner bits: 1 x, 2 y, 4 z)
const boxFaces = [
  [0, 2, 6, 4],
  [1, 3, 7, 5],
  [0, 1, 5, 4],
  [2, 3, 7, 6],
  [0, 1, 3, 2],
  [4, 5, 7, 6],
];

const setBoxCorners = (localToView: Matrix4) => {
  for (let i = 0; i < 8; i++) {
    boxCorners[i]!.set(i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5).applyMatrix4(localToView);
  }
};

// NOTE View depth range of the part of the box inside the view space frustum. The faces of the box are clipped to the
// planes, so a box that reaches behind the camera or past the side of the view does not widen the range. The frustum
// corners inside the box are the other corners of that part: with the camera in the box the range starts at the near
// plane. Writes clipDepths and returns false when no part of the box is in the frustum.
const getClippedDepths = (clipFrustum: Frustum, viewToLocal: Matrix4) => {
  let near = Infinity;
  let far = -Infinity;

  for (let i = 0; i < 8; i++) {
    localCorner.copy(frustumCorners[i]!).applyMatrix4(viewToLocal);

    if (unitBox.containsPoint(localCorner)) {
      near = Math.min(near, -frustumCorners[i]!.z);
      far = Math.max(far, -frustumCorners[i]!.z);
    }
  }

  for (let f = 0; f < 6; f++) {
    let input = clipPolygons[0]!;
    let output = clipPolygons[1]!;
    let count = 4;

    for (let i = 0; i < 4; i++) {
      input[i]!.copy(boxCorners[boxFaces[f]![i]!]!);
    }

    for (let p = 0; p < 6 && count > 0; p++) {
      const plane = clipFrustum.planes[p]!;
      let outputCount = 0;

      for (let i = 0; i < count; i++) {
        const a = input[i]!;
        const b = input[(i + 1) % count]!;
        const da = plane.distanceToPoint(a);
        const db = plane.distanceToPoint(b);

        if (da >= 0.0) {
          output[outputCount++]!.copy(a);
        }

        if (da >= 0.0 !== db >= 0.0) {
          output[outputCount++]!.lerpVectors(a, b, da / (da - db));
        }
      }

      count = outputCount;
      [input, output] = [output, input];
    }

    for (let i = 0; i < count; i++) {
      near = Math.min(near, -input[i]!.z);
      far = Math.max(far, -input[i]!.z);
    }
  }

  clipDepths.near = near;
  clipDepths.far = far;

  return near <= far;
};

/**
 * Finds the ready volumes of the visible frame volumes (his `FrameVolumeCollector`). `volumes`: in
 * the view. `casters`: all, when `useGrid` (the 'frame' lighting shadow grid, K3.3). Entries and
 * arrays are reused by the next `collect()`.
 */
export class FrameVolumeCollector<V extends FrameVolumeLike = FrameVolumeLike> {
  volumes: FrameVolumeEntry<V>[] = [];
  casters: FrameVolumeEntry<V>[] = [];
  // NOTE View depth range of the volumes in the view
  near = 0.0;
  far = 0.0;

  entries: FrameVolumeEntry<V>[] = [];
  usedEntries = 0;
  camera: Camera | null = null;
  useGrid = false;
  prepareVolume: ((frameVolume: V, volume: FrameVolumeRecord) => void) | null = null;
  private readonly visitObject: (object: Object3D) => void;

  constructor() {
    this.visitObject = (object) => {
      const frameVolume = object as V;

      if (!frameVolume.isVDBFrameVolume || !frameVolume.proxy.layers.test(this.camera!.layers)) {
        return;
      }

      const { volumes } = frameVolume;

      for (let i = 0; i < volumes.length; i++) {
        this.addVolume(frameVolume, volumes[i]!);
      }
    };
  }

  getEntry(frameVolume: V, volume: FrameVolumeRecord): FrameVolumeEntry<V> {
    if (this.usedEntries === this.entries.length) {
      this.entries.push({ frameVolume, volume });
    }

    const entry = this.entries[this.usedEntries++]!;

    entry.frameVolume = frameVolume;
    entry.volume = volume;

    return entry;
  }

  addVolume(frameVolume: V, volume: FrameVolumeRecord): void {
    if (!volume.ready) {
      return;
    }

    const camera = this.camera!;

    this.prepareVolume?.(frameVolume, volume);

    volume.matrixWorld.multiplyMatrices(frameVolume.matrixWorld, volume.matrix);
    volume.worldToLocal.copy(volume.matrixWorld).invert();
    volume.worldBox.copy(unitBox).applyMatrix4(volume.matrixWorld);

    const entry = this.getEntry(frameVolume, volume);

    // NOTE Volumes out of the view can still cast shadows on visible volumes
    if (this.useGrid) {
      this.casters.push(entry);
    }

    if (!frustum.intersectsBox(volume.worldBox)) {
      return;
    }

    volume.localToView.multiplyMatrices(camera.matrixWorldInverse, volume.matrixWorld);
    volume.viewToLocal.copy(volume.localToView).invert();
    setBoxCorners(volume.localToView);

    if (!getClippedDepths(viewFrustum, volume.viewToLocal)) {
      return;
    }

    volume.viewNear = clipDepths.near;
    volume.viewFar = clipDepths.far;

    this.near = Math.min(this.near, volume.viewNear);
    this.far = Math.max(this.far, volume.viewFar);

    this.volumes.push(entry);
  }

  /** settings: near and far of the renderer settings. Returns this; `volumes` is empty when none is in view. */
  collect(
    scene: Object3D,
    camera: Camera & { near: number; far: number; reversedDepth?: boolean },
    settings: { near: number; far: number },
    useGrid: boolean,
    prepareVolume: ((frameVolume: V, volume: FrameVolumeRecord) => void) | null = null,
  ): this {
    this.volumes.length = 0;
    this.casters.length = 0;
    this.usedEntries = 0;
    this.near = Infinity;
    this.far = 0.0;
    this.camera = camera;
    this.useGrid = useGrid;
    this.prepareVolume = prepareVolume;

    projectionScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(projectionScreenMatrix);
    viewFrustum.setFromProjectionMatrix(camera.projectionMatrix, camera.coordinateSystem, camera.reversedDepth);
    setFrustumCorners(viewFrustum.planes);

    scene.traverseVisible(this.visitObject);

    // NOTE Fit the slices to the visible volumes. Then no slices are used for empty space.
    this.near = Math.max(this.near, settings.near, camera.near);
    this.far = Math.min(this.far, settings.far, camera.far);

    if (this.far <= this.near) {
      this.volumes.length = 0;
    }

    this.far = Math.max(this.far, this.near * 1.0001);
    this.camera = null;
    this.prepareVolume = null;

    return this;
  }

  /** True when a volume in the view or a caster uses mipmaps. */
  hasMipmaps(): boolean {
    return hasMipmaps(this.volumes) || hasMipmaps(this.casters);
  }
}

const hasMipmaps = (items: readonly FrameVolumeEntry[]) => {
  for (let i = 0; i < items.length; i++) {
    if (items[i]!.frameVolume.mipmaps) {
      return true;
    }
  }

  return false;
};

/** Uniform value of settings.fadeRadius. 0 turns the distance fade off. */
export const getFrameFadeRadius = (settings: { fadeRadius: number }) =>
  Number.isFinite(settings.fadeRadius) && settings.fadeRadius > 0.0 ? settings.fadeRadius : 0.0;

/** Fractional slice of a view depth. range: froxelRange (near, far). Slices are logarithmic. */
export const getFroxelSlice = (viewDepth: number, range: Vector2Like, depth: number) =>
  (depth * Math.log(Math.max(viewDepth, range.x) / range.x)) / Math.log(range.y / range.x);

/** First slice and slice count of a volume. Writes target ({ first, count }) and returns it. */
export const getFroxelSliceRange = (
  volume: { viewNear: number; viewFar: number },
  range: Vector2Like,
  depth: number,
  target: { first: number; count: number },
) => {
  const first = Math.max(0, Math.floor(getFroxelSlice(volume.viewNear, range, depth)));
  const last = Math.min(depth - 1, Math.ceil(getFroxelSlice(volume.viewFar, range, depth)));

  target.first = first;
  target.count = last - first + 1;

  return target;
};

/**
 * Screen rectangle of the volume in froxel cells (x0, y0, x1, y1), with one cell of padding for the
 * linear filter. Writes target. Returns false when the volume covers no cell.
 */
export const getFroxelRect = (
  volume: { localToView: Matrix4 },
  camera: Camera,
  width: number,
  height: number,
  target: Vector4,
) => {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  clipMatrix.multiplyMatrices(camera.projectionMatrix, volume.localToView);

  for (let i = 0; i < 8; i++) {
    corner.set(i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5, 1.0).applyMatrix4(clipMatrix);

    // NOTE A corner behind the camera: the volume can cover all of the screen
    if (corner.w <= 1e-6) {
      target.set(0.0, 0.0, width, height);

      return true;
    }

    minX = Math.min(minX, corner.x / corner.w);
    minY = Math.min(minY, corner.y / corner.w);
    maxX = Math.max(maxX, corner.x / corner.w);
    maxY = Math.max(maxY, corner.y / corner.w);
  }

  const x0 = Math.max(0, Math.floor((minX * 0.5 + 0.5) * width) - 1);
  const y0 = Math.max(0, Math.floor((minY * 0.5 + 0.5) * height) - 1);
  const x1 = Math.min(width, Math.ceil((maxX * 0.5 + 0.5) * width) + 1);
  const y1 = Math.min(height, Math.ceil((maxY * 0.5 + 0.5) * height) + 1);

  if (x1 <= x0 || y1 <= y0) {
    return false;
  }

  target.set(x0, y0, x1, y1);

  return true;
};

const volumeRect = new Vector4();
const volumeSlices = { first: 0, count: 0 };

export interface FroxelArea {
  rect: Vector4;
  first: number;
  count: number;
}

/**
 * Froxel area of all volumes: the union of their rectangles and slices, one cell and one slice
 * larger on each side. Writes area; count is 0 when no volume covers a cell.
 */
export const getFroxelArea = (
  volumes: readonly { volume: { localToView: Matrix4; viewNear: number; viewFar: number } }[],
  camera: Camera,
  width: number,
  height: number,
  range: Vector2Like,
  depth: number,
  area: FroxelArea,
) => {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  let first = Infinity;
  let last = -Infinity;

  for (let i = 0; i < volumes.length; i++) {
    const { volume } = volumes[i]!;

    if (!getFroxelRect(volume, camera, width, height, volumeRect)) {
      continue;
    }

    getFroxelSliceRange(volume, range, depth, volumeSlices);

    x0 = Math.min(x0, volumeRect.x);
    y0 = Math.min(y0, volumeRect.y);
    x1 = Math.max(x1, volumeRect.z);
    y1 = Math.max(y1, volumeRect.w);
    first = Math.min(first, volumeSlices.first);
    last = Math.max(last, volumeSlices.first + volumeSlices.count - 1);
  }

  if (first > last) {
    area.rect.set(0.0, 0.0, 0.0, 0.0);
    area.first = 0;
    area.count = 0;

    return area;
  }

  area.rect.set(Math.max(0, x0 - 1), Math.max(0, y0 - 1), Math.min(width, x1 + 1), Math.min(height, y1 + 1));
  area.first = Math.max(0, first - 1);
  area.count = Math.min(depth - 1, last + 1) - area.first + 1;

  return area;
};
