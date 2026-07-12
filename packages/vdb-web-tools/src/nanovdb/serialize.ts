/**
 * serialize.ts — lays a {@link BuiltTree} out into a complete, valid NanoVDB
 * FLOAT grid image (a flat little-endian u32 array), and the public
 * `buildFromDense` entry point.
 *
 * ## Memory layout (32-byte aligned throughout)
 *
 *   [GridData 672][TreeData 64][RootData 64 + tiles*32][uppers][lowers][leaves]
 *
 * Every block size is a multiple of 32 (GridData 672, TreeData 64, RootData
 * 64 + 32/tile, upper 270400, lower 33856, leaf 2144), so concatenating from a
 * 32-aligned base keeps every block 32-aligned with no interior padding — this
 * matches the layout native `nanovdb_convert` emits (verified against
 * `sphere_fog_float.nvdb`: tree@672, root@736 (offset 64 from tree), uppers
 * after the root table, etc.).
 *
 * ## Build order
 *
 *   1. buildTree() — voxels -> leaves (value mask + values) -> lower nodes
 *      (child leaves + background tiles) -> upper nodes -> root tiles; stats and
 *      per-node bbox aggregated bottom-up in that pass.
 *   2. Compute block offsets, allocate the image.
 *   3. Write nodes leaf-first is unnecessary — offsets are known up front, so we
 *      write GridData, TreeData, RootData+tiles, uppers, lowers, leaves, patching
 *      child offsets (relative to the containing node) as we go.
 *
 * ## nanovdb_convert byte-parity (DEFERRED — needs native tools)
 *
 * This wave's correctness bar is *value* parity through the repo's proven CPU
 * (`read-value.ts`) and WGSL readers, plus structural conformance, NOT byte
 * identity with `nanovdb_convert`. Three things differ from a native encode and
 * are intentionally out of scope here: (a) active-tile compression of uniform
 * regions (see tree.ts), (b) the real Full/Partial CRC checksum (we write the
 * "disabled" sentinel — see bytes.ts), and (c) native's trailing grid padding.
 * All three are tracked for the byte-parity follow-up; none affects any value a
 * reader returns.
 */

import {
  CHECKSUM_DISABLED,
  GRID_CLASS_FOG_VOLUME,
  GRID_FLAGS_STATS_BBOX_BREADTH_FIRST,
  GRID_NAME_MAX,
  GRID_OFF_BLIND_METADATA_COUNT,
  GRID_OFF_BLIND_METADATA_OFFSET,
  GRID_OFF_CHECKSUM,
  GRID_OFF_DATA0,
  GRID_OFF_DATA1,
  GRID_OFF_DATA2,
  GRID_OFF_FLAGS,
  GRID_OFF_GRID_CLASS,
  GRID_OFF_GRID_COUNT,
  GRID_OFF_GRID_INDEX,
  GRID_OFF_GRID_NAME,
  GRID_OFF_GRID_SIZE,
  GRID_OFF_GRID_TYPE,
  GRID_OFF_MAGIC,
  GRID_OFF_MAP,
  GRID_OFF_VERSION,
  GRID_OFF_VOXEL_SIZE,
  GRID_OFF_WORLD_BBOX,
  GRID_SIZE,
  GridImageWriter,
  LEAF_OFF_BBOX_DIF_AND_FLAGS,
  LEAF_OFF_BBOX_MIN,
  LEAF_OFF_VALUE_MASK,
  LOWER_OFF_CHILD_MASK,
  LOWER_TABLE_COUNT,
  MAGIC_GRID,
  MAP_OFF_INVMATD,
  MAP_OFF_INVMATF,
  MAP_OFF_MATD,
  MAP_OFF_MATF,
  MAP_OFF_TAPERD,
  MAP_OFF_TAPERF,
  MAP_OFF_VECD,
  MAP_OFF_VECF,
  NODE_FLAG_STATS,
  NODE_OFF_BBOX_MAX,
  NODE_OFF_BBOX_MIN,
  NODE_OFF_FLAGS,
  ROOT_OFF_BBOX_MAX,
  ROOT_OFF_BBOX_MIN,
  ROOT_OFF_TABLE_SIZE,
  ROOT_TILE_OFF_CHILD,
  ROOT_TILE_OFF_KEY,
  ROOT_TILE_OFF_STATE,
  TREE_OFF_NODE_COUNT_LEAF,
  TREE_OFF_NODE_COUNT_LOWER,
  TREE_OFF_NODE_COUNT_UPPER,
  TREE_OFF_NODE_OFFSET_LEAF,
  TREE_OFF_NODE_OFFSET_LOWER,
  TREE_OFF_NODE_OFFSET_ROOT,
  TREE_OFF_NODE_OFFSET_UPPER,
  TREE_OFF_TILE_COUNT_LOWER,
  TREE_OFF_TILE_COUNT_ROOT,
  TREE_OFF_TILE_COUNT_UPPER,
  TREE_OFF_VOXEL_COUNT,
  TREE_SIZE,
  UPPER_OFF_CHILD_MASK,
  UPPER_TABLE_COUNT,
  VERSION_PACKED,
} from "./bytes.js";
import { FLOAT_LEAF_CODEC, type LeafCodec } from "./leaf-codec.js";
import { StatsAccumulator } from "./stats.js";
import { buildTree, type BuiltTree, type LowerNode, type UpperNode } from "./tree.js";

export interface BuildFromDenseOptions {
  /** Index-space min corner of the dense block. Default [0,0,0]. */
  origin?: [number, number, number];
  /** Uniform voxel size in world units. Default 1. */
  voxelSize?: number;
  /** World-space position of index coord `origin`. Default `origin*voxelSize`. */
  worldOrigin?: [number, number, number];
  /** Background (inactive) value. Default 0. */
  background?: number;
  /**
   * Active predicate. Omitted: active iff `value !== background` (exact).
   * Given: active iff `|value - background| > activeThreshold`.
   */
  activeThreshold?: number;
  /** Grid name (<256 bytes incl. NUL). Default "dense". */
  gridName?: string;
  /** Grid class. Default "FogVolume". */
  gridClass?: "FogVolume" | "Unknown";
}

/** Result of a build, exposing metadata alongside the raw image. */
export interface BuiltGrid {
  image: Uint32Array;
  gridName: string;
  voxelCount: number;
  indexBBox: { min: [number, number, number]; max: [number, number, number] };
  worldBBox: { min: [number, number, number]; max: [number, number, number] };
  voxelSize: number;
  nodeCounts: { leaf: number; lower: number; upper: number };
}

const GRID_CLASS_ID: Record<NonNullable<BuildFromDenseOptions["gridClass"]>, number> = {
  FogVolume: GRID_CLASS_FOG_VOLUME,
  Unknown: 0,
};

/**
 * Builds a complete FLOAT NanoVDB grid image from a dense array of shape
 * `[nx,ny,nz]` (x-major: idx = (x*ny + y)*nz + z).
 */
export function buildFromDense(
  values: Float32Array,
  dims: [number, number, number],
  opts: BuildFromDenseOptions = {},
): Uint32Array {
  return buildFromDenseDetailed(values, dims, opts).image;
}

/** As `buildFromDense`, but also returns computed metadata (used by writeNvdb). */
export function buildFromDenseDetailed(
  values: Float32Array,
  dims: [number, number, number],
  opts: BuildFromDenseOptions = {},
): BuiltGrid {
  const [nx, ny, nz] = dims;
  if (nx < 0 || ny < 0 || nz < 0 || !Number.isInteger(nx) || !Number.isInteger(ny) || !Number.isInteger(nz)) {
    throw new Error(`buildFromDense: dims must be non-negative integers, got [${nx},${ny},${nz}]`);
  }
  if (values.length !== nx * ny * nz) {
    throw new Error(
      `buildFromDense: values.length ${values.length} !== nx*ny*nz (${nx}*${ny}*${nz} = ${nx * ny * nz})`,
    );
  }

  const origin = opts.origin ?? [0, 0, 0];
  const voxelSize = opts.voxelSize ?? 1;
  const background = opts.background ?? 0;
  const gridName = opts.gridName ?? "dense";
  const gridClassId = GRID_CLASS_ID[opts.gridClass ?? "FogVolume"];
  const worldOrigin: [number, number, number] =
    opts.worldOrigin ?? [origin[0] * voxelSize, origin[1] * voxelSize, origin[2] * voxelSize];

  const nameBytes = new TextEncoder().encode(gridName);
  if (nameBytes.length + 1 > GRID_NAME_MAX) {
    throw new Error(
      `buildFromDense: gridName is ${nameBytes.length} bytes; must be < ${GRID_NAME_MAX} (incl. NUL). ` +
        `Long grid names (HasLongGridName + blind metadata) are not supported in v1.`,
    );
  }

  const codec = FLOAT_LEAF_CODEC;
  const tree = buildTree(values, dims, { origin, background, activeThreshold: opts.activeThreshold });

  const image = layoutGrid(tree, codec, {
    gridName,
    nameBytes,
    gridClassId,
    voxelSize,
    worldOrigin,
    origin,
    background,
  });

  const indexBBox = bboxOrEmpty(tree.root);
  const worldBBox = worldBBoxFromIndex(indexBBox, voxelSize, worldOrigin, origin);

  return {
    image,
    gridName,
    voxelCount: tree.activeVoxelCount,
    indexBBox,
    worldBBox,
    voxelSize,
    nodeCounts: { leaf: tree.leaves.length, lower: tree.lowers.length, upper: tree.uppers.length },
  };
}

interface LayoutParams {
  gridName: string;
  nameBytes: Uint8Array;
  gridClassId: number;
  voxelSize: number;
  worldOrigin: [number, number, number];
  origin: [number, number, number];
  background: number;
}

function layoutGrid(tree: BuiltTree, codec: LeafCodec, p: LayoutParams): Uint32Array {
  const L = codec.layout;
  const nTiles = tree.rootTiles.length;
  const nUpper = tree.uppers.length;
  const nLower = tree.lowers.length;
  const nLeaf = tree.leaves.length;

  // Block base offsets (all multiples of 32).
  const treeOff = GRID_SIZE; // 672
  const rootOff = treeOff + TREE_SIZE; // 736
  const rootBlockSize = L.rootSize + nTiles * L.rootTileSize;
  const upperOff = rootOff + rootBlockSize;
  const lowerOff = upperOff + nUpper * L.upperSize;
  const leafOff = lowerOff + nLower * L.lowerSize;
  const total = leafOff + nLeaf * L.leafSize;

  const w = new GridImageWriter(total);

  const upperAbs = (i: number): number => upperOff + i * L.upperSize;
  const lowerAbs = (i: number): number => lowerOff + i * L.lowerSize;
  const leafAbs = (i: number): number => leafOff + i * L.leafSize;

  // ---- GridData ----------------------------------------------------------
  writeGridData(w, tree, codec, p, total);

  // ---- TreeData ----------------------------------------------------------
  w.setU64(treeOff + TREE_OFF_NODE_OFFSET_LEAF, BigInt(leafOff - treeOff));
  w.setU64(treeOff + TREE_OFF_NODE_OFFSET_LOWER, BigInt(lowerOff - treeOff));
  w.setU64(treeOff + TREE_OFF_NODE_OFFSET_UPPER, BigInt(upperOff - treeOff));
  w.setU64(treeOff + TREE_OFF_NODE_OFFSET_ROOT, BigInt(rootOff - treeOff));
  w.setU32(treeOff + TREE_OFF_NODE_COUNT_LEAF, nLeaf);
  w.setU32(treeOff + TREE_OFF_NODE_COUNT_LOWER, nLower);
  w.setU32(treeOff + TREE_OFF_NODE_COUNT_UPPER, nUpper);
  // This builder emits no active tiles (all active voxels live in leaves), so
  // every active-tile count is 0. See tree.ts topology policy.
  w.setU32(treeOff + TREE_OFF_TILE_COUNT_LOWER, 0);
  w.setU32(treeOff + TREE_OFF_TILE_COUNT_UPPER, 0);
  w.setU32(treeOff + TREE_OFF_TILE_COUNT_ROOT, 0);
  w.setU64(treeOff + TREE_OFF_VOXEL_COUNT, BigInt(tree.activeVoxelCount));

  // ---- RootData + tiles --------------------------------------------------
  const ibox = bboxOrEmpty(tree.root);
  writeCoord(w, rootOff + ROOT_OFF_BBOX_MIN, ibox.min);
  writeCoord(w, rootOff + ROOT_OFF_BBOX_MAX, ibox.max);
  w.setU32(rootOff + ROOT_OFF_TABLE_SIZE, nTiles);
  codec.writeConstant(w, rootOff + L.rootOffBackground, p.background);
  writeStats(w, rootOff, L.rootOffMin, L.rootOffMax, L.rootOffAve, L.rootOffStdDev, tree.root);

  tree.rootTiles.forEach((tile, i) => {
    const tileOff = rootOff + L.rootSize + i * L.rootTileSize;
    w.setU64(tileOff + ROOT_TILE_OFF_KEY, tile.key);
    w.setI64(tileOff + ROOT_TILE_OFF_CHILD, BigInt(upperAbs(tile.child.memIndex) - rootOff));
    w.setU32(tileOff + ROOT_TILE_OFF_STATE, 0);
    codec.writeConstant(w, tileOff + L.rootTileOffValue, 0);
  });

  // ---- Upper nodes -------------------------------------------------------
  for (const upper of tree.uppers) {
    writeInternalNode(w, upperAbs(upper.memIndex), upper, {
      tableCount: UPPER_TABLE_COUNT,
      childMaskOff: UPPER_OFF_CHILD_MASK,
      tableOff: L.upperOffTable,
      tableStride: L.tableStride,
      offMin: L.upperOffMin,
      offMax: L.upperOffMax,
      offAve: L.upperOffAve,
      offStdDev: L.upperOffStdDev,
      background: p.background,
      codec,
      childAbs: lowerAbs,
    });
  }

  // ---- Lower nodes -------------------------------------------------------
  for (const lower of tree.lowers) {
    writeInternalNode(w, lowerAbs(lower.memIndex), lower, {
      tableCount: LOWER_TABLE_COUNT,
      childMaskOff: LOWER_OFF_CHILD_MASK,
      tableOff: L.lowerOffTable,
      tableStride: L.tableStride,
      offMin: L.lowerOffMin,
      offMax: L.lowerOffMax,
      offAve: L.lowerOffAve,
      offStdDev: L.lowerOffStdDev,
      background: p.background,
      codec,
      childAbs: leafAbs,
    });
  }

  // ---- Leaf nodes --------------------------------------------------------
  for (const leaf of tree.leaves) {
    const off = leafAbs(leaf.memIndex);
    const s = leaf.stats;
    const bmin = s.isEmpty ? leaf.origin : s.bboxMin;
    const bmax = s.isEmpty ? leaf.origin : s.bboxMax;
    writeCoord(w, off + LEAF_OFF_BBOX_MIN, bmin);
    const difX = clampByte(bmax[0] - bmin[0]);
    const difY = clampByte(bmax[1] - bmin[1]);
    const difZ = clampByte(bmax[2] - bmin[2]);
    w.setU32(
      off + LEAF_OFF_BBOX_DIF_AND_FLAGS,
      (difX | (difY << 8) | (difZ << 16) | (NODE_FLAG_STATS << 24)) >>> 0,
    );
    w.setMaskWords(off + LEAF_OFF_VALUE_MASK, leaf.valueMask);
    codec.encodeLeafValues(w, off, leaf.values, leaf.stats);
  }

  return w.u32;
}

interface InternalNodeWriteParams {
  tableCount: number;
  childMaskOff: number;
  tableOff: number;
  tableStride: number;
  offMin: number;
  offMax: number;
  offAve: number;
  offStdDev: number;
  background: number;
  codec: LeafCodec;
  /** Absolute byte offset of child with the given memIndex. */
  childAbs: (memIndex: number) => number;
}

function writeInternalNode(
  w: GridImageWriter,
  nodeOff: number,
  node: UpperNode | LowerNode,
  p: InternalNodeWriteParams,
): void {
  const s = node.stats;
  const bmin = s.isEmpty ? node.origin : s.bboxMin;
  const bmax = s.isEmpty ? node.origin : s.bboxMax;
  writeCoord(w, nodeOff + NODE_OFF_BBOX_MIN, bmin);
  writeCoord(w, nodeOff + NODE_OFF_BBOX_MAX, bmax);
  w.setU64(nodeOff + NODE_OFF_FLAGS, BigInt(NODE_FLAG_STATS));
  writeStats(w, nodeOff, p.offMin, p.offMax, p.offAve, p.offStdDev, s);

  // Table: child slots hold an i64 offset (relative to this node); every other
  // slot is an inactive background tile carrying the background value. The
  // value mask stays all-zero (this builder emits no active tiles).
  const childSlots = node.children;
  for (let n = 0; n < p.tableCount; n++) {
    const slotOff = nodeOff + p.tableOff + n * p.tableStride;
    const child = childSlots.get(n);
    if (child !== undefined) {
      w.setMaskBit(nodeOff + p.childMaskOff, n);
      w.setI64(slotOff, BigInt(p.childAbs(child.memIndex) - nodeOff));
    } else if (p.background !== 0) {
      // Skip when background is 0: the buffer is already zero-filled.
      p.codec.writeConstant(w, slotOff, p.background);
    }
  }
}

function writeStats(
  w: GridImageWriter,
  base: number,
  offMin: number,
  offMax: number,
  offAve: number,
  offStdDev: number,
  s: StatsAccumulator,
): void {
  w.setF32(base + offMin, s.isEmpty ? 0 : s.min);
  w.setF32(base + offMax, s.isEmpty ? 0 : s.max);
  w.setF32(base + offAve, s.average);
  w.setF32(base + offStdDev, s.stdDev);
}

function writeGridData(
  w: GridImageWriter,
  tree: BuiltTree,
  codec: LeafCodec,
  p: LayoutParams,
  total: number,
): void {
  w.setU64(GRID_OFF_MAGIC, MAGIC_GRID);
  w.setU64(GRID_OFF_CHECKSUM, CHECKSUM_DISABLED);
  w.setU32(GRID_OFF_VERSION, VERSION_PACKED);
  w.setU32(GRID_OFF_FLAGS, GRID_FLAGS_STATS_BBOX_BREADTH_FIRST);
  w.setU32(GRID_OFF_GRID_INDEX, 0);
  w.setU32(GRID_OFF_GRID_COUNT, 1);
  w.setU64(GRID_OFF_GRID_SIZE, BigInt(total));
  // Grid name: C string in the fixed 256-byte inline field.
  w.setBytes(GRID_OFF_GRID_NAME, p.nameBytes); // NUL terminator already zero
  writeMap(w, p);
  writeWorldBBox(w, tree, p);
  const vs = p.voxelSize;
  w.setF64(GRID_OFF_VOXEL_SIZE, vs);
  w.setF64(GRID_OFF_VOXEL_SIZE + 8, vs);
  w.setF64(GRID_OFF_VOXEL_SIZE + 16, vs);
  w.setU32(GRID_OFF_GRID_CLASS, p.gridClassId);
  w.setU32(GRID_OFF_GRID_TYPE, codec.layout.gridType);
  // No blind metadata: offset points at end-of-grid, count 0 (native
  // convention, verified on sphere_fog_float.nvdb).
  w.setU64(GRID_OFF_BLIND_METADATA_OFFSET, BigInt(total));
  w.setU32(GRID_OFF_BLIND_METADATA_COUNT, 0);
  w.setU32(GRID_OFF_DATA0, 0);
  w.setU64(GRID_OFF_DATA1, 0n);
  w.setU64(GRID_OFF_DATA2, 0n);
}

function writeMap(w: GridImageWriter, p: LayoutParams): void {
  const s = p.voxelSize;
  const inv = 1 / s;
  const t: [number, number, number] = [
    p.worldOrigin[0] - p.origin[0] * s,
    p.worldOrigin[1] - p.origin[1] * s,
    p.worldOrigin[2] - p.origin[2] * s,
  ];
  const mat = [s, 0, 0, 0, s, 0, 0, 0, s];
  const invMat = [inv, 0, 0, 0, inv, 0, 0, 0, inv];
  const mapBase = GRID_OFF_MAP;
  for (let i = 0; i < 9; i++) {
    w.setF32(mapBase + MAP_OFF_MATF + i * 4, mat[i]!);
    w.setF32(mapBase + MAP_OFF_INVMATF + i * 4, invMat[i]!);
    w.setF64(mapBase + MAP_OFF_MATD + i * 8, mat[i]!);
    w.setF64(mapBase + MAP_OFF_INVMATD + i * 8, invMat[i]!);
  }
  for (let i = 0; i < 3; i++) {
    w.setF32(mapBase + MAP_OFF_VECF + i * 4, t[i]!);
    w.setF64(mapBase + MAP_OFF_VECD + i * 8, t[i]!);
  }
  w.setF32(mapBase + MAP_OFF_TAPERF, 1);
  w.setF64(mapBase + MAP_OFF_TAPERD, 1);
}

function writeWorldBBox(w: GridImageWriter, tree: BuiltTree, p: LayoutParams): void {
  const ibox = bboxOrEmpty(tree.root);
  const wb = worldBBoxFromIndex(ibox, p.voxelSize, p.worldOrigin, p.origin);
  for (let a = 0; a < 3; a++) w.setF64(GRID_OFF_WORLD_BBOX + a * 8, wb.min[a]!);
  for (let a = 0; a < 3; a++) w.setF64(GRID_OFF_WORLD_BBOX + 24 + a * 8, wb.max[a]!);
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function writeCoord(w: GridImageWriter, off: number, c: readonly [number, number, number]): void {
  w.setI32(off, c[0]);
  w.setI32(off + 4, c[1]);
  w.setI32(off + 8, c[2]);
}

function clampByte(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function bboxOrEmpty(s: StatsAccumulator): {
  min: [number, number, number];
  max: [number, number, number];
} {
  if (s.isEmpty) return { min: [0, 0, 0], max: [-1, -1, -1] };
  return { min: [...s.bboxMin], max: [...s.bboxMax] };
}

function worldBBoxFromIndex(
  ibox: { min: [number, number, number]; max: [number, number, number] },
  voxelSize: number,
  worldOrigin: [number, number, number],
  origin: [number, number, number],
): { min: [number, number, number]; max: [number, number, number] } {
  const t = [
    worldOrigin[0] - origin[0] * voxelSize,
    worldOrigin[1] - origin[1] * voxelSize,
    worldOrigin[2] - origin[2] * voxelSize,
  ];
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    min[a] = ibox.min[a]! * voxelSize + t[a]!;
    // "+ one voxel" convention: world max corner is (indexMax + 1) * voxelSize.
    max[a] = (ibox.max[a]! + 1) * voxelSize + t[a]!;
  }
  return { min, max };
}
