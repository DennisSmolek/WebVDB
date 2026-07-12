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

export interface VdbGrid {
  name: string;
  gridType: string;
  transform: VdbTransformInfo;
  metadata: Record<string, unknown>;
  indexBBox: VdbBBox | null;
  activeVoxelCount: bigint;
  background: number;
  readValue(ijk: [number, number, number]): { value: number; active: boolean };
  iterLeaves(): Iterable<VdbLeaf>;
}
