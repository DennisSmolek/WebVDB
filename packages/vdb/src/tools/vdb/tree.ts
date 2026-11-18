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
 * Scope limits (throw `VdbUnsupportedError`, not silently wrong data):
 * active tiles above leaf resolution (root, upper, or lower level) and
 * multi-buffer trees. Narrow-band level sets and fully-leaf-resolved fog
 * volumes — the samples this parser targets — have neither.
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

export interface LeafData {
  origin: [number, number, number];
  values: Float32Array; // length 512
  valueMask: Uint32Array; // 16 x u32 = 512 bits
}

export interface TreeReadResult {
  background: number;
  leaves: LeafData[];
  activeVoxelCount: number;
}

interface TreeCtx {
  reader: ByteReader;
  fileVersion: number;
  compression: CompressionFlags;
  useHalf: boolean;
  background: number;
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

function assertNoActiveTiles(
  childMask: Uint32Array,
  valueMask: Uint32Array,
  fullNumValues: number,
  label: string,
): void {
  for (let i = 0; i < fullNumValues; i++) {
    if (!testBit(childMask, i) && testBit(valueMask, i)) {
      throw new VdbUnsupportedError(
        `active tile in ${label} at slot ${i} — constant-value regions above leaf resolution are not ` +
          "supported by this scope-limited FloatGrid parser; re-export as a fully leaf-resolved grid",
      );
    }
  }
}

/** Reads `(childMask, valueMask, tileValues)` for an upper/lower node and
 *  advances the stream correctly, discarding the tile values themselves
 *  (out of scope — see module docs) after confirming none are active. */
function readInternalNodeTopology(
  ctx: TreeCtx,
  log2dim: number,
  fullNumValues: number,
  label: string,
): { childMask: Uint32Array; valueMask: Uint32Array } {
  const childMask = ctx.reader.maskWords(fullNumValues);
  const valueMask = ctx.reader.maskWords(fullNumValues);
  const numValuesForStream =
    ctx.fileVersion < 222 ? countZeroBits(childMask, fullNumValues) : fullNumValues;
  // Consumed for correct stream position only — see module docs.
  readCompressedValues(ctx.reader, {
    fileVersion: ctx.fileVersion,
    compression: ctx.compression,
    useHalf: ctx.useHalf,
    background: ctx.background,
    numValues: numValuesForStream,
    valueMask,
  });
  assertNoActiveTiles(childMask, valueMask, fullNumValues, label);
  void log2dim;
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
  const { childMask } = readInternalNodeTopology(ctx, LOWER_LOG2DIM, LOWER_NUM_VALUES, "lower (16^3)");
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
  const { childMask } = readInternalNodeTopology(ctx, UPPER_LOG2DIM, UPPER_NUM_VALUES, "upper (32^3)");
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
  const ctx: TreeCtx = { reader, fileVersion: options.fileVersion, compression: options.compression, useHalf: options.useHalf, background };

  const numTiles = reader.u32();
  const numRootChildren = reader.u32();

  for (let i = 0; i < numTiles; i++) {
    reader.vec3i(); // origin
    reader.f32(); // value
    const active = reader.bool();
    if (active) {
      throw new VdbUnsupportedError(
        "active root-level tile — constant-value regions above leaf resolution are not supported " +
          "by this scope-limited FloatGrid parser; re-export as a fully leaf-resolved grid",
      );
    }
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

  return { background, leaves, activeVoxelCount };
}
