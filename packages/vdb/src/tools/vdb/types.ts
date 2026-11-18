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

export type Vec3Tuple = [number, number, number];

/**
 * The OpenVDB value types this parser reads (the `<valueType>` of a
 * `Tree_<valueType>_5_4_3` grid type): `float`; the vectors `vec3s`, `vec3d`
 * (narrowed to f32) and `vec3i` (exact below 2^24); and `ptdataidx32`, the
 * point data grid (`points::PointDataGrid`).
 */
export type VdbValueType = "float" | "vec3s" | "vec3d" | "vec3i" | "ptdataidx32";

export interface VdbLeaf {
  origin: [number, number, number];
  /**
   * All 512 voxel values in leaf-offset order, `components` scalars each
   * (xyz interleaved for vector grids, so 1536 entries). For a point data
   * grid these are the voxels' point end offsets, see {@link VdbPointDataGrid}.
   */
  values: Float32Array;
  /** 512-bit active mask as 16 little-endian u32 words. */
  valueMask: Uint32Array;
}

/**
 * An active tile above leaf level: a `dim^3` block of voxels that are all
 * active and all equal to `value`. `dim` is 8 for a lower-node (16^3) tile,
 * 128 for an upper-node (32^3) tile and 4096 for a root tile.
 */
export interface VdbTile<V = number> {
  /** Index-space min corner, a multiple of `dim`. */
  origin: [number, number, number];
  dim: 8 | 128 | 4096;
  value: V;
}

interface VdbGridBase<V> {
  name: string;
  /** The grid type without the `_HalfFloat` suffix, e.g. `Tree_vec3s_5_4_3`. */
  gridType: string;
  transform: VdbTransformInfo;
  metadata: Record<string, unknown>;
  indexBBox: VdbBBox | null;
  /** Active voxels in leaves plus `dim^3` for every active tile. */
  activeVoxelCount: bigint;
  background: V;
  /**
   * Value and active state at an index coordinate: a leaf voxel, else the value
   * of an enclosing active tile, else the background (inactive).
   */
  readValue(ijk: [number, number, number]): { value: V; active: boolean };
  iterLeaves(): Iterable<VdbLeaf>;
  /** Active tiles at root, upper and lower level. Empty for fully leaf-resolved grids. */
  iterTiles(): Iterable<VdbTile<V>>;
}

/** A scalar `FloatGrid`. */
export interface VdbFloatGrid extends VdbGridBase<number> {
  valueType: "float";
  components: 1;
}

/** A `Vec3SGrid` / `Vec3DGrid` / `Vec3IGrid`. Values are `[x, y, z]`. */
export interface VdbVec3Grid extends VdbGridBase<Vec3Tuple> {
  valueType: "vec3s" | "vec3d" | "vec3i";
  components: 3;
}

/** Element type of a decoded point attribute. */
export type VdbPointAttributeArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array;

/** One attribute in a point data grid's descriptor. */
export interface VdbPointAttributeInfo {
  name: string;
  /** OpenVDB value type name, e.g. `vec3s`, `float`, `int32`, `uint8`. */
  type: string;
  /** OpenVDB codec name, e.g. `null`, `trnc`, `fxpt16`, `uvec`, `grp`, `str`. */
  codec: string;
  /** Position in the attribute set. */
  index: number;
}

/** Points decoded from a point data grid, in OpenVDB leaf and voxel order. */
export interface VdbPointList {
  count: number;
  /** World-space positions, xyz per point: `transform(voxel ijk + P)`. */
  positions: Float32Array;
  /**
   * Every other decodable attribute, `count * stride * components` values
   * (vectors interleaved). Real types decode to `Float32Array` (`double` to
   * `Float64Array`), whatever their codec; integer types keep their width.
   * Group attributes (codec `grp`) are not listed here; see `groups`.
   */
  attributes: Record<string, VdbPointAttributeArray>;
  /** Shape of each entry in `attributes`. */
  attributeShapes: Record<string, { components: number; stride: number }>;
  /** Group membership per point (0 or 1), by group name. */
  groups: Record<string, Uint8Array>;
  /** Attributes whose type/codec pair this decoder does not handle. */
  skipped: VdbPointAttributeInfo[];
}

/**
 * A `points::PointDataGrid` (`ptdataidx32`). The tree is parsed like any
 * other: each leaf voxel holds the *end* offset of its points within the
 * leaf's attribute arrays (so a voxel's points are `[prev end, end)`).
 * `readValue` returns that offset. {@link VdbPointDataGrid.points} decodes the
 * attribute arrays.
 */
export interface VdbPointDataGrid extends VdbGridBase<number> {
  valueType: "ptdataidx32";
  components: 1;
  /** Total points (the sum of every leaf's last end offset). */
  pointCount: number;
  /** The attribute descriptor of the first leaf (OpenVDB shares one per grid when it can). */
  attributes: VdbPointAttributeInfo[];
  /** Group names, from the descriptor. */
  groupNames: string[];
  /** Decodes every leaf's attribute arrays (cached after the first call). */
  points(): VdbPointList;
}

export type VdbGrid = VdbFloatGrid | VdbVec3Grid | VdbPointDataGrid;
