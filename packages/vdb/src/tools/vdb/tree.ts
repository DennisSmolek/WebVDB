/**
 * tree.ts — the 5-4-3 tree: root -> upper (32^3 slots, log2dim 5) -> lower
 * (16^3 slots, log2dim 4) -> leaf (8^3 voxels, log2dim 3).
 *
 * Two-phase read, per both format references (and matching real OpenVDB's
 * `Tree::readTopology`/`Tree::readBuffers` split):
 *  1. **Topology** (single recursive pass): root tiles/children; each
 *     upper/lower node's `(childMask, valueMask, tileValues)` inline (file
 *     version >= 214, "internal-node compression" — always true for our
 *     218-224 window); each leaf's `valueMask` ONLY (its value buffer is
 *     deferred).
 *  2. **Buffers** (second pass, same recursion order, leaves only): each
 *     leaf re-serializes its `valueMask` (a genuine on-disk quirk — OpenVDB's
 *     `LeafNode::writeBuffers` re-saves the mask right before the buffer;
 *     confirmed against `vdb-rs`'s `read_tree_data`, which does the same
 *     double read) followed by its compressed voxel values.
 *
 * The `mjurczyk/openvdb` JS reader (this repo's other reference) never
 * actually implements leaf value decoding — `ChildNode.readValues()` returns
 * early for leaves with a mask-derived 0/1 placeholder array and
 * `GridDescriptor.readBuffers()` is a bare `// TODO`. That's fine for its
 * boolean-active demos but useless for real voxel data, which is the whole
 * point of this parser (handing Phase 5b's serializer real floats) — the
 * two-phase buffer reading here is lifted from `vdb-rs` instead.
 *
 * ## Active tiles (K4.1)
 *
 * Root, upper and lower nodes may hold *active tiles*: a slot whose child bit
 * is clear and whose value-mask bit is set stands for a uniform, active block
 * of `childDim^3` voxels (lower slot: one 8^3 leaf; upper slot: one 128^3 lower
 * node; root tile: one 4096^3 upper node). They are read from the formats'
 * own topology fields (OpenVDB `RootNode::readTopology` /
 * `InternalNode::readTopology`) and returned as {@link TileData}, so
 * `buildFromVdb` can emit them as NanoVDB tiles rather than expanding them.
 * Inactive tiles are still dropped (they read back as the grid background),
 * as before.
 *
 * Scope limits (throw `VdbUnsupportedError`, not silently wrong data):
 * multi-buffer trees.
 */

import { countBits, countZeroBits, iterOnes, testBit } from "./bit-utils.js";
import type { ByteReader } from "./byte-reader.js";
import type { CompressionFlags } from "./compression.js";
import { readCompressedValues } from "./compression.js";
import { VdbFormatError, VdbUnsupportedError } from "./errors.js";

export const LEAF_LOG2DIM = 3;
export const LOWER_LOG2DIM = 4;
export const UPPER_LOG2DIM = 5;

export const LEAF_DIM = 1 << LEAF_LOG2DIM; // 8
export const LEAF_NUM_VALUES = 1 << (3 * LEAF_LOG2DIM); // 512
const LOWER_NUM_VALUES = 1 << (3 * LOWER_LOG2DIM); // 4096
const UPPER_NUM_VALUES = 1 << (3 * UPPER_LOG2DIM); // 32768

const LOWER_TOTAL = LEAF_LOG2DIM; // 3
const UPPER_TOTAL = LEAF_LOG2DIM + LOWER_LOG2DIM; // 7

/** Edge length, in voxels, of an active tile at each level. */
export const LOWER_TILE_DIM = 8; // 1 << LOWER_TOTAL: a lower-node tile covers one leaf
export const UPPER_TILE_DIM = 128; // 1 << UPPER_TOTAL: an upper-node tile covers one lower node
export const ROOT_TILE_DIM = 4096; // 1 << (3 + 4 + 5): a root tile covers one upper node

export interface LeafData {
  origin: [number, number, number];
  values: Float32Array; // length 512
  valueMask: Uint32Array; // 16 x u32 = 512 bits
}

/** One active tile above leaf level: `dim^3` voxels, all active, all `value`. */
export interface TileData {
  /** Index-space min corner, a multiple of `dim`. */
  origin: [number, number, number];
  /** Edge length in voxels: 8 (lower-node tile), 128 (upper-node tile) or 4096 (root tile). */
  dim: 8 | 128 | 4096;
  value: number;
}

export interface TreeReadResult {
  background: number;
  leaves: LeafData[];
  /** Active tiles at root, upper and lower level (inactive tiles are dropped). */
  tiles: TileData[];
  /** Active voxels in leaves plus `dim^3` per active tile. */
  activeVoxelCount: number;
}

interface TreeCtx {
  reader: ByteReader;
  fileVersion: number;
  compression: CompressionFlags;
  useHalf: boolean;
  background: number;
  /** Active tiles collected during the topology pass. */
  tiles: TileData[];
}

function offsetToLocalCoord(offset: number, log2dim: number): [number, number, number] {
  const x = offset >> (2 * log2dim);
  const rem = offset & ((1 << (2 * log2dim)) - 1);
  const y = rem >> log2dim;
  const z = rem & ((1 << log2dim) - 1);
  return [x, y, z];
}

function offsetToGlobalCoord(
  offset: number,
  log2dim: number,
  total: number,
  origin: readonly [number, number, number],
): [number, number, number] {
  const [lx, ly, lz] = offsetToLocalCoord(offset, log2dim);
  return [origin[0] + (lx << total), origin[1] + (ly << total), origin[2] + (lz << total)];
}

/**
 * Reads `(childMask, valueMask, tileValues)` for an upper/lower node and
 * appends every active tile (child bit clear, value bit set) to `ctx.tiles`.
 *
 * Tile-value layout follows OpenVDB `InternalNode::readTopology`: from file
 * version 222 (node-mask compression) the stream holds a value for every slot,
 * indexed by slot; before 222 it holds one value per child-*off* slot, packed
 * in slot order.
 */
function readInternalNodeTopology(
  ctx: TreeCtx,
  origin: readonly [number, number, number],
  log2dim: number,
  childTotal: number,
  fullNumValues: number,
): { childMask: Uint32Array; valueMask: Uint32Array } {
  const childMask = ctx.reader.maskWords(fullNumValues);
  const valueMask = ctx.reader.maskWords(fullNumValues);
  const packed = ctx.fileVersion < 222;
  const numValuesForStream = packed ? countZeroBits(childMask, fullNumValues) : fullNumValues;
  const values = readCompressedValues(ctx.reader, {
    fileVersion: ctx.fileVersion,
    compression: ctx.compression,
    useHalf: ctx.useHalf,
    background: ctx.background,
    numValues: numValuesForStream,
    valueMask,
  });
  const dim = (1 << childTotal) as TileData["dim"];
  let packedIdx = 0;
  for (let i = 0; i < fullNumValues; i++) {
    if (testBit(childMask, i)) continue;
    const value = packed ? values[packedIdx++]! : values[i]!;
    if (!testBit(valueMask, i)) continue;
    ctx.tiles.push({ origin: offsetToGlobalCoord(i, log2dim, childTotal, origin), dim, value });
  }
  return { childMask, valueMask };
}

interface LeafPlaceholder {
  origin: [number, number, number];
}

interface LowerNode {
  origin: [number, number, number];
  childMask: Uint32Array;
  leaves: Map<number, LeafPlaceholder>;
}

interface UpperNode {
  origin: [number, number, number];
  childMask: Uint32Array;
  lowers: Map<number, LowerNode>;
}

function readLowerTopology(ctx: TreeCtx, origin: [number, number, number]): LowerNode {
  const { childMask } = readInternalNodeTopology(ctx, origin, LOWER_LOG2DIM, LOWER_TOTAL, LOWER_NUM_VALUES);
  const leaves = new Map<number, LeafPlaceholder>();
  for (const idx of iterOnes(childMask, LOWER_NUM_VALUES)) {
    const leafOrigin = offsetToGlobalCoord(idx, LOWER_LOG2DIM, LOWER_TOTAL, origin);
    // Leaf topology: value mask only, no child mask, no data (deferred).
    ctx.reader.maskWords(LEAF_NUM_VALUES);
    leaves.set(idx, { origin: leafOrigin });
  }
  return { origin, childMask, leaves };
}

function readUpperTopology(ctx: TreeCtx, origin: [number, number, number]): UpperNode {
  const { childMask } = readInternalNodeTopology(ctx, origin, UPPER_LOG2DIM, UPPER_TOTAL, UPPER_NUM_VALUES);
  const lowers = new Map<number, LowerNode>();
  for (const idx of iterOnes(childMask, UPPER_NUM_VALUES)) {
    const lowerOrigin = offsetToGlobalCoord(idx, UPPER_LOG2DIM, UPPER_TOTAL, origin);
    lowers.set(idx, readLowerTopology(ctx, lowerOrigin));
  }
  return { origin, childMask, lowers };
}

function readLeafBuffer(ctx: TreeCtx, placeholder: LeafPlaceholder): LeafData {
  if (ctx.fileVersion < 222) {
    ctx.reader.vec3i(); // origin (redundant with topology-derived origin)
    const numBuffers = ctx.reader.u8();
    if (numBuffers !== 1) {
      throw new VdbFormatError(`leaf node numBuffers ${numBuffers} !== 1 (multi-buffer leaves unsupported)`);
    }
  }
  // The value mask is re-serialized immediately before the buffer data — a
  // real on-disk quirk of LeafNode::writeBuffers, not a bug in this reader.
  const valueMask = ctx.reader.maskWords(LEAF_NUM_VALUES);
  const values = readCompressedValues(ctx.reader, {
    fileVersion: ctx.fileVersion,
    compression: ctx.compression,
    useHalf: ctx.useHalf,
    background: ctx.background,
    numValues: LEAF_NUM_VALUES,
    valueMask,
  });
  return { origin: placeholder.origin, values, valueMask };
}

export interface TreeReadOptions {
  fileVersion: number;
  compression: CompressionFlags;
  useHalf: boolean;
}

export function readTree(reader: ByteReader, options: TreeReadOptions): TreeReadResult {
  const bufferCount = reader.u32();
  if (bufferCount !== 1) {
    throw new VdbUnsupportedError(`multi-buffer trees are not supported (bufferCount=${bufferCount})`);
  }

  const background = reader.f32();
  const tiles: TileData[] = [];
  const ctx: TreeCtx = {
    reader,
    fileVersion: options.fileVersion,
    compression: options.compression,
    useHalf: options.useHalf,
    background,
    tiles,
  };

  const numTiles = reader.u32();
  const numRootChildren = reader.u32();

  // Root tiles: origin, value (always full-width ValueType, never half — see
  // OpenVDB `RootNode::readTopology`), active flag. Inactive tiles are dropped.
  for (let i = 0; i < numTiles; i++) {
    const origin = reader.vec3i();
    const value = reader.f32();
    const active = reader.bool();
    if (!active) continue;
    if (origin.some((c) => c % ROOT_TILE_DIM !== 0)) {
      throw new VdbFormatError(`root tile origin [${origin.join(", ")}] is not ${ROOT_TILE_DIM}-aligned`);
    }
    tiles.push({ origin, dim: ROOT_TILE_DIM, value });
  }

  const uppers: UpperNode[] = [];
  for (let i = 0; i < numRootChildren; i++) {
    const origin = reader.vec3i();
    uppers.push(readUpperTopology(ctx, origin));
  }

  const leaves: LeafData[] = [];
  let activeVoxelCount = 0;
  for (const upper of uppers) {
    for (const lower of upper.lowers.values()) {
      for (const placeholder of lower.leaves.values()) {
        const leaf = readLeafBuffer(ctx, placeholder);
        activeVoxelCount += countBits(leaf.valueMask);
        leaves.push(leaf);
      }
    }
  }

  for (const tile of tiles) activeVoxelCount += tile.dim ** 3;

  return { background, leaves, tiles, activeVoxelCount };
}
