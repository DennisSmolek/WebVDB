/**
 * inspect.ts — a structural report for a NanoVDB grid image: grid type/class,
 * active voxel count, per-level node counts and a memory breakdown by section.
 * Everything is read straight from GridData/TreeData/RootData, so it works on
 * any FLOAT/Fp8/FpN image (ours or native `nanovdb_convert` output).
 *
 * The section sizes use the shared root/tile/upper/lower strides (identical for
 * all three grid types); the leaf block is whatever remains up to `mGridSize`,
 * which correctly accounts for FpN's variable-size leaves.
 */

import {
  FLOAT_LAYOUT,
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

export interface InspectReport {
  gridType: string;
  gridClass: string;
  voxelCount: number;
  nodeCounts: { upper: number; lower: number; leaf: number };
  /** Byte size of each grid section plus `total` (== mGridSize). */
  memoryBreakdown: Record<string, number>;
}

const GRID_TYPE_NAMES: Record<number, string> = {
  0: "Unknown",
  1: "Float",
  2: "Double",
  14: "Fp8",
  16: "FpN",
};

const GRID_CLASS_NAMES: Record<number, string> = {
  0: "Unknown",
  1: "LevelSet",
  2: "FogVolume",
  3: "Staggered",
};

/** Produces a structural + memory report for a NanoVDB grid image. */
export function inspect(image: Uint32Array): InspectReport {
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

  const L = FLOAT_LAYOUT; // root/tile/upper/lower strides are shared across grid types
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
