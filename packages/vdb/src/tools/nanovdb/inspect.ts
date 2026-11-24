/**
 * inspect.ts — a structural report for a NanoVDB grid image: grid type/class,
 * active voxel count, per-level node counts and a memory breakdown by section.
 * Everything is read straight from GridData/TreeData/RootData, so it works on
 * any FLOAT/Fp8/FpN/Vec3f image (ours or native `nanovdb_convert` output).
 *
 * The section sizes use the root/tile/upper/lower strides (shared by FLOAT,
 * Fp8 and FpN; Vec3f has its own); the leaf block is whatever remains up to
 * `mGridSize`, which correctly accounts for FpN's variable-size leaves.
 */

import {
  FLOAT_LAYOUT,
  GRID_TYPE_VEC3F,
  VEC3F_LAYOUT,
  GRID_OFF_GRID_CLASS,
  GRID_OFF_GRID_SIZE,
  GRID_OFF_GRID_TYPE,
  GRID_SIZE,
  ROOT_OFF_TABLE_SIZE,
  TREE_OFF_NODE_COUNT_LEAF,
  TREE_OFF_NODE_COUNT_LOWER,
  TREE_OFF_NODE_COUNT_UPPER,
  TREE_OFF_NODE_OFFSET_LEAF,
  TREE_OFF_NODE_OFFSET_ROOT,
  TREE_OFF_VOXEL_COUNT,
  TREE_SIZE,
} from "./bytes.js";

/** Structure and memory report of a NanoVDB grid image ({@link vdbInspect}). */
export interface VDBInspectReport {
  /** Grid value type name (`'Float'`, `'Fp8'`, `'FpN'`, `'Vec3f'`, ...; `'Unknown'` otherwise). */
  gridType: string;
  /** Grid class name (`'LevelSet'`, `'FogVolume'`, `'Staggered'`, `'Unknown'`). */
  gridClass: string;
  /** Active voxel count from the tree header. */
  voxelCount: number;
  /** Number of nodes per level. */
  nodeCounts: { upper: number; lower: number; leaf: number };
  /** Byte size of each grid section plus `total` (== mGridSize). */
  memoryBreakdown: Record<string, number>;
}

const GRID_TYPE_NAMES: Record<number, string> = {
  0: "Unknown",
  1: "Float",
  2: "Double",
  6: "Vec3f",
  9: "Half",
  13: "Fp4",
  14: "Fp8",
  15: "Fp16",
  16: "FpN",
};

const GRID_CLASS_NAMES: Record<number, string> = {
  0: "Unknown",
  1: "LevelSet",
  2: "FogVolume",
  3: "Staggered",
};

/**
 * Produces a structural + memory report for a NanoVDB grid image.
 *
 * @param image The grid image (little-endian u32 words, `GridData` at word 0).
 */
export function vdbInspect(image: Uint32Array): VDBInspectReport {
  const view = new DataView(image.buffer, image.byteOffset, image.byteLength);

  const gridTypeId = view.getUint32(GRID_OFF_GRID_TYPE, true);
  const gridClassId = view.getUint32(GRID_OFF_GRID_CLASS, true);
  const total = Number(view.getBigUint64(GRID_OFF_GRID_SIZE, true));

  const treeOff = GRID_SIZE;
  const voxelCount = Number(view.getBigUint64(treeOff + TREE_OFF_VOXEL_COUNT, true));
  const nLeaf = view.getUint32(treeOff + TREE_OFF_NODE_COUNT_LEAF, true);
  const nLower = view.getUint32(treeOff + TREE_OFF_NODE_COUNT_LOWER, true);
  const nUpper = view.getUint32(treeOff + TREE_OFF_NODE_COUNT_UPPER, true);

  const rootOff = treeOff + Number(view.getBigUint64(treeOff + TREE_OFF_NODE_OFFSET_ROOT, true));
  const nTiles = view.getUint32(rootOff + ROOT_OFF_TABLE_SIZE, true);
  const leafOff = treeOff + Number(view.getBigUint64(treeOff + TREE_OFF_NODE_OFFSET_LEAF, true));

  // Root/tile/upper/lower strides are shared by FLOAT/Fp8/FpN; Vec3f differs.
  const L = gridTypeId === GRID_TYPE_VEC3F ? VEC3F_LAYOUT : FLOAT_LAYOUT;
  const rootSection = L.rootSize + nTiles * L.rootTileSize;
  const upperSection = nUpper * L.upperSize;
  const lowerSection = nLower * L.lowerSize;
  // Leaf block is the remainder to end-of-grid — correct even for variable-size FpN leaves.
  const leafSection = Math.max(0, total - leafOff);

  return {
    gridType: GRID_TYPE_NAMES[gridTypeId] ?? `type#${gridTypeId}`,
    gridClass: GRID_CLASS_NAMES[gridClassId] ?? `class#${gridClassId}`,
    voxelCount,
    nodeCounts: { upper: nUpper, lower: nLower, leaf: nLeaf },
    memoryBreakdown: {
      gridData: GRID_SIZE,
      tree: TREE_SIZE,
      root: rootSection,
      upper: upperSection,
      lower: lowerSection,
      leaf: leafSection,
      total,
    },
  };
}
