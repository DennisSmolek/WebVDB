/**
 * `NanoVDBGrid` — wraps one NanoVDB grid image (SPEC §3.1): creates the
 * `StorageBufferAttribute` GPU consumers bind, and exposes the metadata a
 * material needs to set up index-space raymarching (world bbox, index bbox,
 * the index<->world affine transform, and a bbox-fitted proxy box).
 *
 * This module owns none of the WGSL/TSL binding wiring itself (that's
 * `createVolumeRenderer` + the Phase 4 `NanoVDBVolumeMaterial`) — it is the
 * CPU-side "grid, GPU-ready" object the rest of the layer is built on.
 */

import * as THREE from "three";
import { StorageBufferAttribute } from "three/webgpu";
import type { GridMetadata, NanoVDBFile } from "nanovdb-wgsl";

/**
 * PNanoVDB `GridType` ids (NanoVDB.h `enum class GridType`) for the v1
 * supported subset (SPEC §2.1 "Quantized decode", §7 non-goals). Only the
 * types `nanovdb-wgsl`'s loader will hand back (`SUPPORTED_GRID_TYPES`:
 * Float/Fp8/FpN) are mapped; anything else is a loader bug or a future
 * extension, and `NanoVDBGrid` throws rather than guess.
 */
const GRID_TYPE_IDS: Readonly<Record<string, number>> = {
  Float: 1,
  Fp8: 14,
  FpN: 16,
};

export interface NanoVDBGridOptions {
  /** Flat u32 grid image — "a valid NanoVDB grid image in a flat u32 buffer" (SPEC §1). */
  image: Uint32Array;
  metadata: GridMetadata;
}

/**
 * Wraps one grid image for GPU use (SPEC §3.1). Construct directly with an
 * image + metadata pair, or via `NanoVDBGrid.fromFile(file, index)` from a
 * parsed `NanoVDBFile`.
 */
export class NanoVDBGrid {
  readonly image: Uint32Array;
  readonly metadata: GridMetadata;
  readonly gridTypeId: number;

  private _storageAttribute: StorageBufferAttribute | undefined;

  constructor(opts: NanoVDBGridOptions) {
    this.image = opts.image;
    this.metadata = opts.metadata;

    const gridTypeId = GRID_TYPE_IDS[opts.metadata.gridType];
    if (gridTypeId === undefined) {
      throw new Error(
        `NanoVDBGrid: unsupported grid type "${opts.metadata.gridType}" — v1 only handles ` +
          `${Object.keys(GRID_TYPE_IDS).join("/")} (FogVolume) grids (SPEC §2.1/§7).`,
      );
    }
    this.gridTypeId = gridTypeId;
  }

  /** Builds a grid from grid #`index` of an already-parsed `NanoVDBFile`. */
  static fromFile(file: NanoVDBFile, index = 0): NanoVDBGrid {
    const image = file.gridImage(index);
    const metadata = file.grids[index];
    if (!metadata) {
      throw new Error(`NanoVDBGrid.fromFile: index ${index} out of range (file has ${file.grids.length} grid(s))`);
    }
    return new NanoVDBGrid({ image, metadata });
  }

  /** Byte length of the grid image (matches `metadata.gridByteSize` for a well-formed file). */
  get byteLength(): number {
    return this.image.byteLength;
  }

  /**
   * The GPU-bindable storage buffer for this grid's image, lazily created
   * and cached — same pattern as demo 01's spike: `itemSize` 1, bound with
   * `'uint'` TSL semantics (`storage(attr, "uint", image.length)`), never
   * `'uint'`-typed by the attribute itself (three has no per-attribute GPU
   * type; the semantic lives at the TSL `storage()` call site).
   */
  get storageAttribute(): StorageBufferAttribute {
    this._storageAttribute ??= new StorageBufferAttribute(this.image, 1);
    return this._storageAttribute;
  }

  /** World-space bounding box (SPEC §3.1 "world bbox -> `Box3`"). */
  worldBBox(): THREE.Box3 {
    const { min, max } = this.metadata.worldBBox;
    return new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
  }

  /** Index-space bounding box, integer coords represented as floats. */
  indexBBox(): THREE.Box3 {
    const { min, max } = this.metadata.indexBBox;
    return new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
  }

  /**
   * Affine index -> world transform, derived from `metadata.voxelSize` (scale)
   * and the world/index bbox minimums (translation): for each axis,
   * `world = index * voxelSize + origin` where
   * `origin = worldBBox.min - indexBBox.min * voxelSize`.
   *
   * ASSUMPTION (documented, not general): this treats the grid's `Map` as
   * axis-aligned scale + translate, no rotation/shear — true for every
   * fixture and every primitive/cloud/EmberGen asset this project targets.
   * `nanovdb-wgsl`'s loader does not currently surface the full 3x3 `Map`
   * matrix (only `voxelSize` + the two bboxes, from `FileMetaData`/`GridData`
   * — see `packages/nanovdb-wgsl/src/nvdb-file.ts`), so a grid with a
   * rotated/sheared `Map` would silently produce a wrong transform here. If
   * that ever matters, the fix is: extend the loader to read
   * `GridData.mMap` (the full `Map` struct: `matF`/`invMatF`/`vecF`/`taperF`,
   * see `pnanovdb.wgsl`'s `pnanovdb_map_*` accessors) and build the matrix
   * from that instead of this bbox-derived approximation.
   *
   * NOTE: `metadata.worldBBox.max` is *not* `indexToWorld() * indexBBox.max`
   * — NanoVDB's `worldBBox` spans the full extent of the last active voxel
   * (its far face), i.e. it equals `indexToWorld() * (indexBBox.max + 1)`
   * per axis (confirmed against all baked fixtures, voxelSize 1: e.g.
   * `box_fog`'s indexBBox max `[40,20,30]` / worldBBox max `[41,21,31]`).
   * `indexBBox.max` itself is an inclusive voxel *coordinate*, not a
   * distance, so the point-transform below deliberately does not bake in
   * that +1 — it is a correct affine map for arbitrary index-space points
   * (ray positions, etc.), just not a max-corner-to-max-corner bbox map.
   */
  indexToWorld(): THREE.Matrix4 {
    const [sx, sy, sz] = this.metadata.voxelSize;
    const [ix, iy, iz] = this.metadata.indexBBox.min;
    const [wx, wy, wz] = this.metadata.worldBBox.min;
    const origin = new THREE.Vector3(wx - ix * sx, wy - iy * sy, wz - iz * sz);
    const scale = new THREE.Vector3(sx, sy, sz);
    return new THREE.Matrix4().compose(origin, new THREE.Quaternion(), scale);
  }

  /** World -> index transform: the exact inverse of `indexToWorld()`. */
  worldToIndex(): THREE.Matrix4 {
    return this.indexToWorld().clone().invert();
  }

  /**
   * A world-space box geometry fitted to `worldBBox()`, with its vertices
   * translated so the box sits at the grid's actual world position when
   * added to the scene with no additional transform (SPEC §3.1 usage:
   * `scene.add(new Mesh(grid.proxyGeometry(), cloud))`).
   */
  proxyGeometry(): THREE.BoxGeometry {
    const bbox = this.worldBBox();
    const size = bbox.getSize(new THREE.Vector3());
    const center = bbox.getCenter(new THREE.Vector3());
    const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
    geometry.translate(center.x, center.y, center.z);
    return geometry;
  }
}
