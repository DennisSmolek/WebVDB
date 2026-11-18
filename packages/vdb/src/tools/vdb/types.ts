/** Public `.vdb` parser types (docs/SPEC.md §4, docs/PLAN.md Phase 5). */

export interface VdbFile {
  fileVersion: number;
  grids: VdbGrid[];
}

export interface VdbTransformInfo {
  type: string;
  /** 4x4 row-major index->world affine matrix. */
  matrix: number[];
  voxelSize: [number, number, number];
}

export interface VdbBBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface VdbLeaf {
  origin: [number, number, number];
  values: Float32Array;
  /** 512-bit active mask as 16 little-endian u32 words. */
  valueMask: Uint32Array;
}

/**
 * An active tile above leaf level: a `dim^3` block of voxels that are all
 * active and all equal to `value`. `dim` is 8 for a lower-node (16^3) tile,
 * 128 for an upper-node (32^3) tile and 4096 for a root tile.
 */
export interface VdbTile {
  /** Index-space min corner, a multiple of `dim`. */
  origin: [number, number, number];
  dim: 8 | 128 | 4096;
  value: number;
}

export interface VdbGrid {
  name: string;
  gridType: string;
  transform: VdbTransformInfo;
  metadata: Record<string, unknown>;
  indexBBox: VdbBBox | null;
  /** Active voxels in leaves plus `dim^3` for every active tile. */
  activeVoxelCount: bigint;
  background: number;
  /**
   * Value and active state at an index coordinate: a leaf voxel, else the value
   * of an enclosing active tile, else the background (inactive).
   */
  readValue(ijk: [number, number, number]): { value: number; active: boolean };
  iterLeaves(): Iterable<VdbLeaf>;
  /** Active tiles at root, upper and lower level. Empty for fully leaf-resolved grids. */
  iterTiles(): Iterable<VdbTile>;
}
