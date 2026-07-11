/**
 * CPU reference for the GPU parity harness: root -> upper -> lower -> leaf
 * descent (`readValueCpu`) plus trilinear sampling (`sampleTrilinearCpu`),
 * parameterized by the `ParsedWgslConstants` extracted at runtime from the
 * vendored WGSL text (`wgsl-constants.ts`) instead of
 * `vendor/stride-tables.json`.
 *
 * This is a transliteration of
 * `packages/nanovdb-wgsl/src/cpu/read-value.ts` and
 * `packages/nanovdb-wgsl/src/cpu/sample-trilinear.ts` (both already
 * validated 657/657 against native NanoVDB sidecars in Phase 1), NOT an
 * import of them — those modules pull their byte offsets from
 * `stride-tables.ts`, which loads `vendor/stride-tables.json` via
 * `node:fs.readFileSync` at import time. `node:fs` has no browser
 * equivalent, so importing that chain into this Vite page would fail at
 * module-eval time; duplicating the (small, already-proven) descent here
 * keeps this page self-contained. Behavior must stay identical to the
 * originals — do not "improve" this copy independently of them.
 */
import type { GridTypeConstants, ParsedWgslConstants, WgslScalarConstants } from "./wgsl-constants";

export type Coord = readonly [number, number, number];

export interface ReadResult {
  value: number;
  active: boolean;
}

// Structural (non-per-grid-type) bit-math constant — same for every grid
// type, so it's a literal here rather than something parsed out of the WGSL
// (mirrors read-value.ts's own MASK_WORD_BITS).
const MASK_WORD_BITS = 5;

// --------------------------------------------------------------- buffer IO -

function readU32(words: Uint32Array, address: number): number {
  return words[address >>> 2]!;
}

function readU64(words: Uint32Array, address: number): bigint {
  const lo = readU32(words, address);
  const hi = readU32(words, address + 4);
  return (BigInt(hi) << 32n) | BigInt(lo);
}

function readI64(words: Uint32Array, address: number): bigint {
  const u = readU64(words, address);
  return u >= 1n << 63n ? u - (1n << 64n) : u;
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
function readF32(words: Uint32Array, address: number): number {
  return getFloat32View(words)[address >>> 2]!;
}

function getMaskBit(words: Uint32Array, maskBase: number, bitIndex: number): boolean {
  const word = readU32(words, maskBase + 4 * (bitIndex >>> MASK_WORD_BITS));
  return ((word >>> (bitIndex & 31)) & 1) !== 0;
}

// ------------------------------------------------------- coord -> offset ---

function leafCoordToOffset([x, y, z]: Coord): number {
  return (((x & 7) >> 0) << 6) + (((y & 7) >> 0) << 3) + ((z & 7) >> 0);
}
function lowerCoordToOffset([x, y, z]: Coord): number {
  return (((x & 127) >> 3) << 8) + (((y & 127) >> 3) << 4) + ((z & 127) >> 3);
}
function upperCoordToOffset([x, y, z]: Coord): number {
  return (((x & 4095) >> 7) << 10) + (((y & 4095) >> 7) << 5) + ((z & 4095) >> 7);
}
function coordToKey([x, y, z]: Coord): bigint {
  const iu = BigInt(x >>> 12);
  const ju = BigInt(y >>> 12);
  const ku = BigInt(z >>> 12);
  return ku | (ju << 21n) | (iu << 42n);
}

// ------------------------------------------------------------ leaf FP ------

function leafFpReadFloat(
  words: Uint32Array,
  tableAddress: number,
  n: number,
  valueLogBits: number,
  negOffMinimum: number,
  negOffQuantum: number,
): number {
  const valueBits = 1 << valueLogBits;
  const valueMask = (1 << valueBits) - 1;
  const valuesPerWordBits = 5 - valueLogBits;
  const valuesPerWordMask = (1 << valuesPerWordBits) - 1;

  const minimum = readF32(words, tableAddress - negOffMinimum);
  const quantum = readF32(words, tableAddress - negOffQuantum);
  const raw = readU32(words, tableAddress + ((n >> valuesPerWordBits) << 2));
  const valueCompressed = (raw >>> ((n & valuesPerWordMask) << valueLogBits)) & valueMask;
  return valueCompressed * quantum + minimum;
}

function leafFp8ReadFloat(words: Uint32Array, tableAddress: number, n: number, s: WgslScalarConstants): number {
  return leafFpReadFloat(
    words,
    tableAddress,
    n,
    3,
    s.PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM,
    s.PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM,
  );
}

function leafFpnReadFloat(words: Uint32Array, tableAddress: number, n: number, s: WgslScalarConstants): number {
  const bboxDifAndFlags = readU32(
    words,
    tableAddress - s.PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS,
  );
  const flags = bboxDifAndFlags >>> 24;
  const valueLogBits = flags >>> 5;
  return leafFpReadFloat(
    words,
    tableAddress,
    n,
    valueLogBits,
    s.PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM,
    s.PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM,
  );
}

// ------------------------------------------------- descent (root..leaf) ----

type Level = 0 | 1 | 2 | 3 | 4;

interface Resolved {
  level: Level;
  valueAddress: number;
  active: boolean;
  leafN: number;
}

function descendLeaf(
  words: Uint32Array,
  leafAddress: number,
  ijk: Coord,
  s: WgslScalarConstants,
  gt: GridTypeConstants,
): Resolved {
  const n = leafCoordToOffset(ijk);
  const active = getMaskBit(words, leafAddress + s.PNANOVDB_LEAF_OFF_VALUE_MASK, n);
  const valueAddress = leafAddress + gt.leaf_off_table + ((gt.value_stride_bits * n) >> 3);
  return { level: 0, valueAddress, active, leafN: n };
}

function descendLower(
  words: Uint32Array,
  lowerAddress: number,
  ijk: Coord,
  s: WgslScalarConstants,
  gt: GridTypeConstants,
): Resolved {
  const n = lowerCoordToOffset(ijk);
  if (getMaskBit(words, lowerAddress + s.PNANOVDB_LOWER_OFF_CHILD_MASK, n)) {
    const tableAddress = lowerAddress + gt.lower_off_table + gt.table_stride * n;
    const childRel = readI64(words, tableAddress);
    const leafAddress = lowerAddress + Number(childRel);
    return descendLeaf(words, leafAddress, ijk, s, gt);
  }
  const valueAddress = lowerAddress + gt.lower_off_table + gt.table_stride * n;
  const active = getMaskBit(words, lowerAddress + s.PNANOVDB_LOWER_OFF_VALUE_MASK, n);
  return { level: 1, valueAddress, active, leafN: 0 };
}

function descendUpper(
  words: Uint32Array,
  upperAddress: number,
  ijk: Coord,
  s: WgslScalarConstants,
  gt: GridTypeConstants,
): Resolved {
  const n = upperCoordToOffset(ijk);
  if (getMaskBit(words, upperAddress + s.PNANOVDB_UPPER_OFF_CHILD_MASK, n)) {
    const tableAddress = upperAddress + gt.upper_off_table + gt.table_stride * n;
    const childRel = readI64(words, tableAddress);
    const lowerAddress = upperAddress + Number(childRel);
    return descendLower(words, lowerAddress, ijk, s, gt);
  }
  const valueAddress = upperAddress + gt.upper_off_table + gt.table_stride * n;
  const active = getMaskBit(words, upperAddress + s.PNANOVDB_UPPER_OFF_VALUE_MASK, n);
  return { level: 2, valueAddress, active, leafN: 0 };
}

function descendRoot(
  words: Uint32Array,
  rootAddress: number,
  ijk: Coord,
  s: WgslScalarConstants,
  gt: GridTypeConstants,
): Resolved {
  const tileCount = readU32(words, rootAddress + s.PNANOVDB_ROOT_OFF_TABLE_SIZE);
  const tile0 = rootAddress + gt.root_size;
  const key = coordToKey(ijk);

  let tileAddress = -1;
  for (let i = 0; i < tileCount; i++) {
    const candidate = tile0 + i * gt.root_tile_size;
    const tileKey = readU64(words, candidate + s.PNANOVDB_ROOT_TILE_OFF_KEY);
    if (tileKey === key) {
      tileAddress = candidate;
      break;
    }
  }

  if (tileAddress < 0) {
    return { level: 4, valueAddress: rootAddress + gt.root_off_background, active: false, leafN: 0 };
  }

  const child = readI64(words, tileAddress + s.PNANOVDB_ROOT_TILE_OFF_CHILD);
  if (child === 0n) {
    const state = readU32(words, tileAddress + s.PNANOVDB_ROOT_TILE_OFF_STATE);
    return { level: 3, valueAddress: tileAddress + gt.root_tile_off_value, active: state !== 0, leafN: 0 };
  }

  const upperAddress = rootAddress + Number(child);
  return descendUpper(words, upperAddress, ijk, s, gt);
}

// ------------------------------------------------------------ public API ---

/**
 * Looks up voxel `ijk` in a NanoVDB grid image (little-endian u32 words,
 * `GridData` at word 0). `gridTypeId` must be one of
 * `wc.scalars.PNANOVDB_GRID_TYPE_{FLOAT,FP8,FPN}`.
 */
export function readValueCpu(
  gridWords: Uint32Array,
  ijk: Coord,
  gridTypeId: number,
  wc: ParsedWgslConstants,
): ReadResult {
  const gt = wc.gridTypeConstants[gridTypeId];
  if (!gt) {
    throw new Error(`readValueCpu: no pnanovdb_grid_type_constants row for grid type id ${gridTypeId}`);
  }
  const s = wc.scalars;

  const treeAddress = s.PNANOVDB_GRID_SIZE;
  const rootAddress = treeAddress + Number(readU64(gridWords, treeAddress + s.PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT));
  const resolved = descendRoot(gridWords, rootAddress, ijk, s, gt);

  let value: number;
  if (gridTypeId === s.PNANOVDB_GRID_TYPE_FLOAT) {
    value = readF32(gridWords, resolved.valueAddress);
  } else if (resolved.level === 0) {
    value =
      gridTypeId === s.PNANOVDB_GRID_TYPE_FP8
        ? leafFp8ReadFloat(gridWords, resolved.valueAddress, resolved.leafN, s)
        : leafFpnReadFloat(gridWords, resolved.valueAddress, resolved.leafN, s);
  } else {
    value = readF32(gridWords, resolved.valueAddress);
  }

  return { value, active: resolved.active };
}

/**
 * Continuous index-space trilinear sample: floor to the base voxel, take 8
 * neighboring `readValueCpu` taps (ignoring `active` — the background value
 * participates in the blend like a texture sampler would), lerp x then y
 * then z. Must match `pnanovdb_sample_trilinear_typed` in pnanovdb.wgsl
 * exactly (same 8-tap order, same lerp nesting).
 */
export function sampleTrilinearCpu(
  gridWords: Uint32Array,
  xyz: Coord,
  gridTypeId: number,
  wc: ParsedWgslConstants,
): number {
  const bx = Math.floor(xyz[0]);
  const by = Math.floor(xyz[1]);
  const bz = Math.floor(xyz[2]);
  const fx = xyz[0] - bx;
  const fy = xyz[1] - by;
  const fz = xyz[2] - bz;

  const tap = (dx: 0 | 1, dy: 0 | 1, dz: 0 | 1): number =>
    readValueCpu(gridWords, [bx + dx, by + dy, bz + dz], gridTypeId, wc).value;

  const v000 = tap(0, 0, 0);
  const v100 = tap(1, 0, 0);
  const v010 = tap(0, 1, 0);
  const v110 = tap(1, 1, 0);
  const v001 = tap(0, 0, 1);
  const v101 = tap(1, 0, 1);
  const v011 = tap(0, 1, 1);
  const v111 = tap(1, 1, 1);

  const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

  const c00 = lerp(v000, v100, fx);
  const c10 = lerp(v010, v110, fx);
  const c01 = lerp(v001, v101, fx);
  const c11 = lerp(v011, v111, fx);
  const c0 = lerp(c00, c10, fy);
  const c1 = lerp(c01, c11, fy);
  return lerp(c0, c1, fz);
}
