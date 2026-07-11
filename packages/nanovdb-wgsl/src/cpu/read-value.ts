/**
 * Pure-TypeScript CPU reference implementation of NanoVDB voxel lookup.
 *
 * This is a faithful, line-by-line transliteration of the root -> upper ->
 * lower -> leaf descent in `vendor/upstream/PNanoVDB.h` (the "PNanoVDB"
 * portable-C reference), restricted to grid types FLOAT (1), FP8 (14) and
 * FPN (16) — the three types the Phase 1 fixture bake produces. It exists to
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
 *   pnanovdb_leaf_fpn_read_float.
 */
import {
  defineBigInt,
  defineNumber,
  gridTypeConstantsFor,
  gridTypeIds,
  type GridTypeConstants,
} from "./stride-tables.js";

export interface ReadResult {
  value: number;
  active: boolean;
}

type Coord = readonly [number, number, number];

const supportedGridTypeNames = ["FLOAT", "FP8", "FPN"] as const;
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
// `address` is always a byte offset into `words`, always word-aligned (every
// struct/table stride in the vendored layout is a multiple of 4 bytes).

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
  throw new Error(
    `readValue: unsupported grid type ${id}${knownName ? ` (${knownName})` : ""}; ` +
      `only FLOAT (${gridTypeIds.FLOAT}), FP8 (${gridTypeIds.FP8}) and FPN (${gridTypeIds.FPN}) are implemented`,
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

/**
 * Looks up voxel `ijk` in a NanoVDB grid image (little-endian u32 words,
 * `GridData` at word 0). Supports grid types FLOAT (1), FP8 (14) and FPN
 * (16); throws for anything else.
 */
export function readValue(gridWords: Uint32Array, ijk: [number, number, number]): ReadResult {
  const magic = readU64(gridWords, GRID_OFF_MAGIC);
  if (magic !== MAGIC_GRID) {
    throw new Error(
      `readValue: invalid NanoVDB grid magic 0x${magic.toString(16)} ` +
        `(expected 0x${MAGIC_GRID.toString(16)}); gridWords does not start at a GridData block`,
    );
  }

  const gridTypeId = readU32(gridWords, GRID_OFF_GRID_TYPE);
  const gridTypeName = gridTypeNameFor(gridTypeId);
  const gt = gridTypeConstantsFor(gridTypeName);

  const treeAddress = GRID_SIZE;
  const rootAddress = treeAddress + Number(readU64(gridWords, treeAddress + TREE_OFF_NODE_OFFSET_ROOT));

  const resolved = descendRoot(gridWords, rootAddress, ijk, gt);

  let value: number;
  if (gridTypeName === "FLOAT") {
    value = readF32(gridWords, resolved.valueAddress);
  } else if (resolved.level === 0) {
    // Leaf level: quantized decode (pnanovdb_root_fp8/fpn_read_float, level==0 branch).
    value =
      gridTypeName === "FP8"
        ? leafFp8ReadFloat(gridWords, resolved.valueAddress, resolved.leafN)
        : leafFpnReadFloat(gridWords, resolved.valueAddress, resolved.leafN);
  } else {
    // Tile/background level: FP8/FPN store the constant value as a plain
    // float32 (pnanovdb_root_fp8/fpn_read_float, else branch).
    value = readF32(gridWords, resolved.valueAddress);
  }

  return { value, active: resolved.active };
}
