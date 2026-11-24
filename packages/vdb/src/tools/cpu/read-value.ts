/**
 * Pure-TypeScript CPU reference implementation of NanoVDB voxel lookup.
 *
 * This is a faithful, line-by-line transliteration of the root -> upper ->
 * lower -> leaf descent in `vendor/upstream/PNanoVDB.h` (the "PNanoVDB"
 * portable-C reference), restricted to grid types FLOAT (1), HALF (9; K7.H), FP4 (13), FP8 (14),
 * FP16 (15) and FPN (16). It exists to
 * de-risk the WGSL traversal math ahead of the Phase 2 GPU test suite: this
 * module and `pnanovdb.wgsl` should always agree, voxel for voxel.
 *
 * Byte-offset/layout constants are read from `vendor/stride-tables.json`
 * (extracted from the vendored header) rather than hardcoded — see
 * `stride-tables.ts`. Only structural bit-math constants that are the same
 * for every grid type (the leaf/lower/upper dim masks: 8, 128, 4096 voxels)
 * are literal here, matching PNanoVDB.h's own `pnanovdb_*_coord_to_offset`.
 *
 * PNanoVDB.h functions transliterated (by name, for cross-reference):
 *   pnanovdb_grid_get_magic, pnanovdb_grid_get_grid_type,
 *   pnanovdb_grid_get_tree, pnanovdb_tree_get_root,
 *   pnanovdb_coord_to_key, pnanovdb_root_get_tile_zero,
 *   pnanovdb_root_find_tile, pnanovdb_root_tile_get_key/get_child/get_state,
 *   pnanovdb_root_get_value_address_and_level (+ pnanovdb_root_is_active),
 *   pnanovdb_upper_coord_to_offset, pnanovdb_upper_get_child_mask,
 *   pnanovdb_upper_get_child, pnanovdb_upper_get_table_address,
 *   pnanovdb_upper_get_value_mask (+ ..._and_level),
 *   pnanovdb_lower_coord_to_offset, pnanovdb_lower_get_child_mask,
 *   pnanovdb_lower_get_child, pnanovdb_lower_get_table_address,
 *   pnanovdb_lower_get_value_mask (+ ..._and_level),
 *   pnanovdb_leaf_coord_to_offset, pnanovdb_leaf_get_value_mask,
 *   pnanovdb_leaf_get_table_address, pnanovdb_read_float,
 *   pnanovdb_leaf_fp_read_float, pnanovdb_leaf_fp8_read_float,
 *   pnanovdb_leaf_fpn_read_float, pnanovdb_read_half.
 */
import { VDBToolsError } from "../errors.js";
import { vdbHalfToFloat } from "../nanovdb/half.js";
import {
  defineBigInt,
  defineNumber,
  gridTypeConstantsFor,
  gridTypeIds,
  type GridTypeConstants,
} from "./stride-tables.js";

/** A voxel read: its decoded value and whether it is active. */
export interface VDBReadResult {
  /** The decoded value (the background value for an unset voxel). */
  value: number;
  /** Whether the voxel is active. */
  active: boolean;
}

type Coord = readonly [number, number, number];

const supportedGridTypeNames = ["FLOAT", "HALF", "FP4", "FP8", "FP16", "FPN"] as const;
type SupportedGridTypeName = (typeof supportedGridTypeNames)[number];

// ----------------------------- structural (non-per-grid-type) constants ---
// These mirror PNanoVDB.h's own literal bit masks/shifts (dims 8/16/32 for
// leaf/lower/upper nodes and the 32-bit mask-word stride) and are the same
// for every grid type, so unlike the byte offsets they are not looked up
// from stride-tables.json.
const MASK_WORD_BITS = 5; // 32 bits per pnanovdb_uint32_t mask word

// ----------------------------- defines pulled from stride-tables.json -----
const GRID_OFF_MAGIC = defineNumber("PNANOVDB_GRID_OFF_MAGIC");
const GRID_OFF_GRID_TYPE = defineNumber("PNANOVDB_GRID_OFF_GRID_TYPE");
const GRID_SIZE = defineNumber("PNANOVDB_GRID_SIZE");
const MAGIC_GRID = defineBigInt("PNANOVDB_MAGIC_GRID");

const TREE_OFF_NODE_OFFSET_ROOT = defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT");

const ROOT_OFF_TABLE_SIZE = defineNumber("PNANOVDB_ROOT_OFF_TABLE_SIZE");
const ROOT_TILE_OFF_KEY = defineNumber("PNANOVDB_ROOT_TILE_OFF_KEY");
const ROOT_TILE_OFF_CHILD = defineNumber("PNANOVDB_ROOT_TILE_OFF_CHILD");
const ROOT_TILE_OFF_STATE = defineNumber("PNANOVDB_ROOT_TILE_OFF_STATE");

const UPPER_OFF_VALUE_MASK = defineNumber("PNANOVDB_UPPER_OFF_VALUE_MASK");
const UPPER_OFF_CHILD_MASK = defineNumber("PNANOVDB_UPPER_OFF_CHILD_MASK");

const LOWER_OFF_VALUE_MASK = defineNumber("PNANOVDB_LOWER_OFF_VALUE_MASK");
const LOWER_OFF_CHILD_MASK = defineNumber("PNANOVDB_LOWER_OFF_CHILD_MASK");

const LEAF_OFF_VALUE_MASK = defineNumber("PNANOVDB_LEAF_OFF_VALUE_MASK");
const LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS = defineNumber(
  "PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS",
);
const LEAF_TABLE_NEG_OFF_MINIMUM = defineNumber("PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM");
const LEAF_TABLE_NEG_OFF_QUANTUM = defineNumber("PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM");

// ----------------------------- low-level buffer access ---------------------
// pnanovdb_buf_read_uint32 / pnanovdb_read_uint32 / pnanovdb_read_float, etc.
// `address` is a byte offset into `words`, word-aligned except for HALF values
// and min/max stats (2-byte aligned; read with readHalf).

function readU32(words: Uint32Array, address: number): number {
  return words[address >>> 2]!;
}

/** pnanovdb_read_uint64 (unsigned 64-bit; low word first, little-endian). */
function readU64(words: Uint32Array, address: number): bigint {
  const lo = readU32(words, address);
  const hi = readU32(words, address + 4);
  return (BigInt(hi) << 32n) | BigInt(lo);
}

/** pnanovdb_read_int64 (signed 64-bit two's complement). */
function readI64(words: Uint32Array, address: number): bigint {
  const u = readU64(words, address);
  return u >= 1n << 63n ? u - (1n << 64n) : u;
}

/** pnanovdb_read_float (bit-reinterpret, not a numeric cast). */
function readF32(words: Uint32Array, address: number): number {
  // Share the backing ArrayBuffer rather than allocating per-call.
  const f32 = getFloat32View(words);
  return f32[address >>> 2]!;
}

const float32Views = new WeakMap<Uint32Array, Float32Array>();
function getFloat32View(words: Uint32Array): Float32Array {
  let view = float32Views.get(words);
  if (!view) {
    view = new Float32Array(words.buffer, words.byteOffset, words.length);
    float32Views.set(words, view);
  }
  return view;
}

/** pnanovdb_read_half: the binary16 at a 2-byte-aligned address, decoded exactly. */
function readHalf(words: Uint32Array, address: number): number {
  return vdbHalfToFloat(readU32(words, address & ~3) >>> ((address & 2) << 3));
}

/** Bit `bitIndex` of a `pnanovdb_uint32_t mask[]` array starting at `maskBase`. */
function getMaskBit(words: Uint32Array, maskBase: number, bitIndex: number): boolean {
  const word = readU32(words, maskBase + 4 * (bitIndex >>> MASK_WORD_BITS));
  return ((word >>> (bitIndex & 31)) & 1) !== 0;
}

// ----------------------------- coord -> offset (structural, per PNanoVDB.h) -

/** pnanovdb_leaf_coord_to_offset: voxel index within an 8^3 leaf, 0..511. */
function leafCoordToOffset([x, y, z]: Coord): number {
  return (((x & 7) >> 0) << 6) + (((y & 7) >> 0) << 3) + ((z & 7) >> 0);
}

/** pnanovdb_lower_coord_to_offset: leaf-table index within a 16^3 lower, 0..4095. */
function lowerCoordToOffset([x, y, z]: Coord): number {
  return (((x & 127) >> 3) << 8) + (((y & 127) >> 3) << 4) + ((z & 127) >> 3);
}

/** pnanovdb_upper_coord_to_offset: lower-table index within a 32^3 upper, 0..32767. */
function upperCoordToOffset([x, y, z]: Coord): number {
  return (((x & 4095) >> 7) << 10) + (((y & 4095) >> 7) << 5) + ((z & 4095) >> 7);
}

/**
 * pnanovdb_coord_to_key, taking the `PNANOVDB_NATIVE_64` branch (we have
 * real 64-bit integers via BigInt, so there's no need for the 32-bit
 * low/high split PNanoVDB.h falls back to without native 64-bit ints).
 * `>>> 12` on a JS number already performs ToUint32 then a logical shift,
 * i.e. exactly `pnanovdb_int32_as_uint32(v) >> 12u`.
 */
function coordToKey([x, y, z]: Coord): bigint {
  const iu = BigInt(x >>> 12);
  const ju = BigInt(y >>> 12);
  const ku = BigInt(z >>> 12);
  return ku | (ju << 21n) | (iu << 42n);
}

// ----------------------------- leaf FP decode -------------------------------

/**
 * pnanovdb_leaf_fp_read_float: shared quantized decode for Fp4/Fp8/Fp16/FpN.
 * `tableAddress` is the (n-independent) start of the leaf's compressed value
 * table; `minimum`/`quantum` are read via the fixed NEG offsets relative to
 * that same address, per PNanoVDB.h.
 */
function leafFpReadFloat(
  words: Uint32Array,
  tableAddress: number,
  n: number,
  valueLogBits: number,
): number {
  const valueBits = 1 << valueLogBits; // 1,2,4,8,16
  const valueMask = (1 << valueBits) - 1;
  const valuesPerWordBits = 5 - valueLogBits;
  const valuesPerWordMask = (1 << valuesPerWordBits) - 1;

  const minimum = readF32(words, tableAddress - LEAF_TABLE_NEG_OFF_MINIMUM);
  const quantum = readF32(words, tableAddress - LEAF_TABLE_NEG_OFF_QUANTUM);
  const raw = readU32(words, tableAddress + ((n >> valuesPerWordBits) << 2));
  const valueCompressed = (raw >>> ((n & valuesPerWordMask) << valueLogBits)) & valueMask;
  // pnanovdb_uint32_to_float: numeric cast, not a bit-reinterpret.
  return valueCompressed * quantum + minimum;
}

/** pnanovdb_leaf_fp4_read_float: fixed 4-bit-per-value quantization. */
function leafFp4ReadFloat(words: Uint32Array, tableAddress: number, n: number): number {
  return leafFpReadFloat(words, tableAddress, n, 2);
}

/** pnanovdb_leaf_fp16_read_float: fixed 16-bit-per-value quantization. */
function leafFp16ReadFloat(words: Uint32Array, tableAddress: number, n: number): number {
  return leafFpReadFloat(words, tableAddress, n, 4);
}

/** pnanovdb_leaf_fp8_read_float: fixed 8-bit-per-value quantization. */
function leafFp8ReadFloat(words: Uint32Array, tableAddress: number, n: number): number {
  return leafFpReadFloat(words, tableAddress, n, 3);
}

/**
 * pnanovdb_leaf_fpn_read_float: bits-per-value is dynamic, stored in the top
 * 3 bits of the leaf's `bbox_dif_and_flags` byte (the "flags" byte), read via
 * the fixed NEG offset relative to the table address:
 *   flags = bbox_dif_and_flags >> 24
 *   value_log_bits = flags >> 5   // 0,1,2,3,4 -> 1,2,4,8,16 bits/value
 */
function leafFpnReadFloat(words: Uint32Array, tableAddress: number, n: number): number {
  const bboxDifAndFlags = readU32(
    words,
    tableAddress - LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS,
  );
  const flags = bboxDifAndFlags >>> 24;
  const valueLogBits = flags >>> 5;
  return leafFpReadFloat(words, tableAddress, n, valueLogBits);
}

// ----------------------------- grid type detection --------------------------

function gridTypeNameFor(id: number): SupportedGridTypeName {
  for (const name of supportedGridTypeNames) {
    if (gridTypeIds[name] === id) return name;
  }
  const knownName = Object.entries(gridTypeIds).find(([, v]) => v === id)?.[0];
  throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", 
    `readValue: unsupported grid type ${id}${knownName ? ` (${knownName})` : ""}; ` +
      `only FLOAT (${gridTypeIds.FLOAT}), HALF (${gridTypeIds.HALF}), FP4 (${gridTypeIds.FP4}), FP8 (${gridTypeIds.FP8}), FP16 (${gridTypeIds.FP16}) and FPN (${gridTypeIds.FPN}) are implemented` +
      (knownName === "VEC3F" ? "; use vdbReadValueVec3 for VEC3F grids" : ""),
  );
}

// ----------------------------- descent (root -> upper -> lower -> leaf) ----

/** Level at which the value/active bit for `ijk` was ultimately resolved. */
type Level = 0 | 1 | 2 | 3 | 4; // leaf, lower tile, upper tile, root tile, background

interface Resolved {
  level: Level;
  /** Byte address of the resolved value (or background) slot. */
  valueAddress: number;
  active: boolean;
  /** Leaf-local voxel offset (0..511), only meaningful when level === 0. */
  leafN: number;
}

function descendLeaf(
  words: Uint32Array,
  leafAddress: number,
  ijk: Coord,
  gt: GridTypeConstants,
): Resolved {
  const n = leafCoordToOffset(ijk);
  const active = getMaskBit(words, leafAddress + LEAF_OFF_VALUE_MASK, n);
  // pnanovdb_leaf_get_table_address: byte_offset = leaf_off_table + ((value_stride_bits * n) >> 3)
  const valueAddress = leafAddress + gt.leaf_off_table + ((gt.value_stride_bits * n) >> 3);
  return { level: 0, valueAddress, active, leafN: n };
}

function descendLower(
  words: Uint32Array,
  lowerAddress: number,
  ijk: Coord,
  gt: GridTypeConstants,
): Resolved {
  const n = lowerCoordToOffset(ijk);
  if (getMaskBit(words, lowerAddress + LOWER_OFF_CHILD_MASK, n)) {
    const tableAddress = lowerAddress + gt.lower_off_table + gt.table_stride * n;
    const childRel = readI64(words, tableAddress);
    const leafAddress = lowerAddress + Number(childRel);
    return descendLeaf(words, leafAddress, ijk, gt);
  }
  const valueAddress = lowerAddress + gt.lower_off_table + gt.table_stride * n;
  const active = getMaskBit(words, lowerAddress + LOWER_OFF_VALUE_MASK, n);
  return { level: 1, valueAddress, active, leafN: 0 };
}

function descendUpper(
  words: Uint32Array,
  upperAddress: number,
  ijk: Coord,
  gt: GridTypeConstants,
): Resolved {
  const n = upperCoordToOffset(ijk);
  if (getMaskBit(words, upperAddress + UPPER_OFF_CHILD_MASK, n)) {
    const tableAddress = upperAddress + gt.upper_off_table + gt.table_stride * n;
    const childRel = readI64(words, tableAddress);
    const lowerAddress = upperAddress + Number(childRel);
    return descendLower(words, lowerAddress, ijk, gt);
  }
  const valueAddress = upperAddress + gt.upper_off_table + gt.table_stride * n;
  const active = getMaskBit(words, upperAddress + UPPER_OFF_VALUE_MASK, n);
  return { level: 2, valueAddress, active, leafN: 0 };
}

function descendRoot(
  words: Uint32Array,
  rootAddress: number,
  ijk: Coord,
  gt: GridTypeConstants,
): Resolved {
  const tileCount = readU32(words, rootAddress + ROOT_OFF_TABLE_SIZE);
  const tile0 = rootAddress + gt.root_size;
  const key = coordToKey(ijk);

  let tileAddress = -1;
  for (let i = 0; i < tileCount; i++) {
    const candidate = tile0 + i * gt.root_tile_size;
    const tileKey = readU64(words, candidate + ROOT_TILE_OFF_KEY);
    if (tileKey === key) {
      tileAddress = candidate;
      break;
    }
  }

  if (tileAddress < 0) {
    // pnanovdb_root_find_tile returned null -> background, inactive.
    return {
      level: 4,
      valueAddress: rootAddress + gt.root_off_background,
      active: false,
      leafN: 0,
    };
  }

  const child = readI64(words, tileAddress + ROOT_TILE_OFF_CHILD);
  if (child === 0n) {
    const state = readU32(words, tileAddress + ROOT_TILE_OFF_STATE);
    return {
      level: 3,
      valueAddress: tileAddress + gt.root_tile_off_value,
      active: state !== 0,
      leafN: 0,
    };
  }

  const upperAddress = rootAddress + Number(child);
  return descendUpper(words, upperAddress, ijk, gt);
}

// ----------------------------- public API -----------------------------------

function checkMagic(gridWords: Uint32Array, fn: string): void {
  const magic = readU64(gridWords, GRID_OFF_MAGIC);
  if (magic !== MAGIC_GRID) {
    throw new VDBToolsError("VDB_BAD_IMAGE", 
      `${fn}: invalid NanoVDB grid magic 0x${magic.toString(16)} ` +
        `(expected 0x${MAGIC_GRID.toString(16)}); gridWords does not start at a GridData block`,
    );
  }
}

/**
 * Looks up voxel `ijk` in a NanoVDB grid image (little-endian u32 words,
 * `GridData` at word 0). Supports grid types FLOAT (1), HALF (9), FP4 (13), FP8 (14),
 * FP16 (15) and FPN (16); throws for anything else. HALF decodes exactly (binary16
 * at every level: voxels, tiles and the background).
 *
 * The value is the exact f64 decode (ground truth). It is NOT the GPU's
 * evaluation order: `vdbReadValueF32` (CPU twin of `vdbReadValue`) matches the
 * shader's f32 arithmetic, and the two can differ by 1 ulp (f32) on Fp grids.
 * Use `vdbReadValueF32` as the oracle for GPU parity.
 *
 * @param gridWords The grid image (little-endian u32 words, `GridData` at word 0).
 * @param ijk Integer index coordinates of the voxel.
 */
export function vdbReadValueCpu(gridWords: Uint32Array, ijk: [number, number, number]): VDBReadResult {
  checkMagic(gridWords, "readValue");

  const gridTypeId = readU32(gridWords, GRID_OFF_GRID_TYPE);
  const gridTypeName = gridTypeNameFor(gridTypeId);
  const gt = gridTypeConstantsFor(gridTypeName);

  const treeAddress = GRID_SIZE;
  const rootAddress = treeAddress + Number(readU64(gridWords, treeAddress + TREE_OFF_NODE_OFFSET_ROOT));

  const resolved = descendRoot(gridWords, rootAddress, ijk, gt);

  let value: number;
  if (gridTypeName === "FLOAT") {
    value = readF32(gridWords, resolved.valueAddress);
  } else if (gridTypeName === "HALF") {
    value = readHalf(gridWords, resolved.valueAddress);
  } else if (resolved.level === 0) {
    // Leaf level: quantized decode (pnanovdb_root_fp8/fpn_read_float, level==0 branch).
    const a = resolved.valueAddress;
    const n = resolved.leafN;
    value =
      gridTypeName === "FP4"
        ? leafFp4ReadFloat(gridWords, a, n)
        : gridTypeName === "FP8"
          ? leafFp8ReadFloat(gridWords, a, n)
          : gridTypeName === "FP16"
            ? leafFp16ReadFloat(gridWords, a, n)
            : leafFpnReadFloat(gridWords, a, n);
  } else {
    // Tile/background level: FP8/FPN store the constant value as a plain
    // float32 (pnanovdb_root_fp8/fpn_read_float, else branch).
    value = readF32(gridWords, resolved.valueAddress);
  }

  return { value, active: resolved.active };
}

/**
 * Returns the encoded per-leaf quantization step at `ijk`, or zero when the
 * value resolves to an unquantized FLOAT or HALF value or an internal/background tile.
 */
export function readLeafQuantum(gridWords: Uint32Array, ijk: [number, number, number]): number {
  const gridTypeId = readU32(gridWords, GRID_OFF_GRID_TYPE);
  const gridTypeName = gridTypeNameFor(gridTypeId);
  if (gridTypeName === "FLOAT" || gridTypeName === "HALF") return 0;

  const gt = gridTypeConstantsFor(gridTypeName);
  const treeAddress = GRID_SIZE;
  const rootAddress = treeAddress + Number(readU64(gridWords, treeAddress + TREE_OFF_NODE_OFFSET_ROOT));
  const resolved = descendRoot(gridWords, rootAddress, ijk, gt);
  if (resolved.level !== 0) return 0;

  return readF32(gridWords, resolved.valueAddress - LEAF_TABLE_NEG_OFF_QUANTUM);
}

/** A VEC3F voxel read: its three components and whether it is active. */
export interface VDBReadVec3Result {
  /** The decoded `[x, y, z]` value (the background for an unset voxel). */
  value: [number, number, number];
  /** Whether the voxel is active. */
  active: boolean;
}

/**
 * Looks up voxel `ijk` in a NanoVDB VEC3F (6) grid image: the same root ->
 * upper -> lower -> leaf descent as {@link vdbReadValueCpu} with the VEC3F layout
 * constants, then 3 f32 at the resolved address (`pnanovdb_read_vec3f`).
 * Throws for any other grid type.
 *
 * @param gridWords The VEC3F grid image (little-endian u32 words, `GridData` at word 0).
 * @param ijk Integer index coordinates of the voxel.
 */
export function vdbReadValueVec3(gridWords: Uint32Array, ijk: [number, number, number]): VDBReadVec3Result {
  checkMagic(gridWords, "vdbReadValueVec3");
  const gridTypeId = readU32(gridWords, GRID_OFF_GRID_TYPE);
  if (gridTypeId !== gridTypeIds.VEC3F) {
    const knownName = Object.entries(gridTypeIds).find(([, v]) => v === gridTypeId)?.[0];
    throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", 
      `vdbReadValueVec3: grid type ${gridTypeId}${knownName ? ` (${knownName})` : ""} is not VEC3F (${gridTypeIds.VEC3F})`,
    );
  }
  const gt = gridTypeConstantsFor("VEC3F");
  const treeAddress = GRID_SIZE;
  const rootAddress = treeAddress + Number(readU64(gridWords, treeAddress + TREE_OFF_NODE_OFFSET_ROOT));
  const resolved = descendRoot(gridWords, rootAddress, ijk, gt);
  const a = resolved.valueAddress;
  return {
    value: [readF32(gridWords, a), readF32(gridWords, a + 4), readF32(gridWords, a + 8)],
    active: resolved.active,
  };
}
