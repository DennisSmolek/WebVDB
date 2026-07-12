/**
 * bytes.ts — the single home for every NanoVDB struct offset/size constant and
 * the low-level little-endian writer used by the serializer. No other module in
 * `nanovdb/` writes a raw byte offset: they go through the named constants and
 * the {@link GridImageWriter} helpers here.
 *
 * ## Provenance of the numbers
 *
 * Every constant below is the ABI-32.9.1 value extracted into
 * `packages/nanovdb-wgsl/vendor/stride-tables.json` (itself generated from the
 * vendored `PNanoVDB.h`). They are duplicated here as named literals rather than
 * read from that JSON at runtime on purpose: `vdb-web-tools` is the pure-TS,
 * browser-first CPU half (decision D3) and must not pull in `node:fs`. To keep
 * the two provably in lockstep, `test/bytes-constants.test.ts` reads
 * stride-tables.json and asserts each constant here equals the extracted value.
 *
 * Only the FLOAT grid type is wired up in this wave; the per-grid-type layout
 * (`FLOAT` block below) is isolated so Fp8/FpN can be added as sibling blocks in
 * the quantization wave without touching the common offsets.
 */

// ---------------------------------------------------------------------------
// Magic numbers, versioning, enums (PNANOVDB_* #defines)
// ---------------------------------------------------------------------------

/** Grid-image magic `NanoVDB1` (u64, little-endian). This is what the raw grid
 *  buffer starts with — what `readValue`/`NanoVDBFile` sniff for. */
export const MAGIC_GRID = 0x314244566f6e614en; // "NanoVDB1"
/** File-header magic `NanoVDB2` (u64) for the segment `.nvdb` container. */
export const MAGIC_FILE = 0x324244566f6e614en; // "NanoVDB2"

export const MAJOR_VERSION = 32;
export const MINOR_VERSION = 9;
export const PATCH_VERSION = 1;
/** Packed `Version`: major<<21 | minor<<10 | patch. */
export const VERSION_PACKED = (MAJOR_VERSION << 21) | (MINOR_VERSION << 10) | PATCH_VERSION;

export const GRID_TYPE_FLOAT = 1;
export const GRID_TYPE_FP8 = 14;
export const GRID_TYPE_FPN = 16;
export const GRID_CLASS_UNKNOWN = 0;
export const GRID_CLASS_FOG_VOLUME = 2;

/**
 * GridData.mFlags for a grid that carries a bbox and full min/max/ave/stddev
 * statistics and is laid out breadth-first — the exact value native
 * `nanovdb_convert` writes for a stats-bearing float FogVolume (verified:
 * `sphere_fog_float.nvdb` carries 0x3e). Bit0 (HasLongGridName) stays clear
 * because we reject names longer than the inline 256-byte field.
 *   bit1 HasBBox | bit2 HasMinMax | bit3 HasAverage | bit4 HasStdDeviation |
 *   bit5 IsBreadthFirst
 */
export const GRID_FLAGS_STATS_BBOX_BREADTH_FIRST = 0x3e;

/**
 * Internal-node (`mFlags`, u64) and leaf (`bbox_dif_and_flags` high byte) flag
 * value written by native `nanovdb_convert` on every stats-bearing node
 * (verified: uppers, lowers and leaves in `sphere_fog_float.nvdb` all carry 2).
 * Neither the CPU (`read-value.ts`) nor WGSL FLOAT readers consult it; we mirror
 * it for structural fidelity with native output.
 */
export const NODE_FLAG_STATS = 2;

// ---------------------------------------------------------------------------
// GridData (672 bytes)
// ---------------------------------------------------------------------------

export const GRID_SIZE = 672;
export const GRID_OFF_MAGIC = 0;
export const GRID_OFF_CHECKSUM = 8;
export const GRID_OFF_VERSION = 16;
export const GRID_OFF_FLAGS = 20;
export const GRID_OFF_GRID_INDEX = 24;
export const GRID_OFF_GRID_COUNT = 28;
export const GRID_OFF_GRID_SIZE = 32;
export const GRID_OFF_GRID_NAME = 40;
export const GRID_NAME_MAX = 256; // GridData::MaxNameSize
export const GRID_OFF_MAP = 296;
export const GRID_OFF_WORLD_BBOX = 560;
export const GRID_OFF_VOXEL_SIZE = 608;
export const GRID_OFF_GRID_CLASS = 632;
export const GRID_OFF_GRID_TYPE = 636;
export const GRID_OFF_BLIND_METADATA_OFFSET = 640;
export const GRID_OFF_BLIND_METADATA_COUNT = 648;
export const GRID_OFF_DATA0 = 652;
export const GRID_OFF_DATA1 = 656;
export const GRID_OFF_DATA2 = 664;

/**
 * GridData.mChecksum sentinel meaning "no checksum" (NanoVDB `ChecksumMode`
 * Disable == ~uint64_t(0)). We write this rather than a real CRC: neither the
 * CPU reader, the WGSL reader, nor `NanoVDBFile.fromArrayBuffer` validates the
 * checksum (verified by reading their source — only magic, version and
 * mGridSize are cross-checked), so a disabled checksum round-trips cleanly.
 * Computing the native CRC48/Full checksum is deferred to the nanovdb_convert
 * byte-parity item (see build-from-dense.ts module doc).
 */
export const CHECKSUM_DISABLED = 0xffffffffffffffffn;

// Map (affine transform), offsets relative to GRID_OFF_MAP.
export const MAP_OFF_MATF = 0; // float[9] row-major 3x3
export const MAP_OFF_INVMATF = 36; // float[9]
export const MAP_OFF_VECF = 72; // float[3] translation
export const MAP_OFF_TAPERF = 84; // float
export const MAP_OFF_MATD = 88; // double[9]
export const MAP_OFF_INVMATD = 160; // double[9]
export const MAP_OFF_VECD = 232; // double[3]
export const MAP_OFF_TAPERD = 256; // double
export const MAP_SIZE = 264;

// ---------------------------------------------------------------------------
// TreeData (64 bytes)
// ---------------------------------------------------------------------------

export const TREE_SIZE = 64;
export const TREE_OFF_NODE_OFFSET_LEAF = 0; // u64, bytes from TreeData to first leaf
export const TREE_OFF_NODE_OFFSET_LOWER = 8; // u64
export const TREE_OFF_NODE_OFFSET_UPPER = 16; // u64
export const TREE_OFF_NODE_OFFSET_ROOT = 24; // u64
export const TREE_OFF_NODE_COUNT_LEAF = 32; // u32
export const TREE_OFF_NODE_COUNT_LOWER = 36; // u32
export const TREE_OFF_NODE_COUNT_UPPER = 40; // u32
export const TREE_OFF_TILE_COUNT_LOWER = 44; // u32 (active tiles)
export const TREE_OFF_TILE_COUNT_UPPER = 48; // u32
export const TREE_OFF_TILE_COUNT_ROOT = 52; // u32
export const TREE_OFF_VOXEL_COUNT = 56; // u64 (active voxels)

// ---------------------------------------------------------------------------
// RootData + RootTile
// ---------------------------------------------------------------------------

export const ROOT_OFF_BBOX_MIN = 0; // i32[3]
export const ROOT_OFF_BBOX_MAX = 12; // i32[3]
export const ROOT_OFF_TABLE_SIZE = 24; // u32 (tile count)

export const ROOT_TILE_OFF_KEY = 0; // u64
export const ROOT_TILE_OFF_CHILD = 8; // i64 (bytes from RootData to child upper node)
export const ROOT_TILE_OFF_STATE = 16; // u32

// ---------------------------------------------------------------------------
// Internal node common offsets (same for upper and lower; masks differ in size)
// ---------------------------------------------------------------------------

export const NODE_OFF_BBOX_MIN = 0; // i32[3]
export const NODE_OFF_BBOX_MAX = 12; // i32[3]
export const NODE_OFF_FLAGS = 24; // u64
export const NODE_OFF_VALUE_MASK = 32; // bit array

export const UPPER_OFF_CHILD_MASK = 4128; // after 32768-bit value mask
export const UPPER_TABLE_COUNT = 32768; // 32^3
export const LOWER_OFF_CHILD_MASK = 544; // after 4096-bit value mask
export const LOWER_TABLE_COUNT = 4096; // 16^3

// ---------------------------------------------------------------------------
// Leaf common offsets
// ---------------------------------------------------------------------------

export const LEAF_OFF_BBOX_MIN = 0; // i32[3]
export const LEAF_OFF_BBOX_DIF_AND_FLAGS = 12; // u32: [difX,difY,difZ,flags] bytes
export const LEAF_OFF_VALUE_MASK = 16; // 512-bit array (64 bytes)
export const LEAF_TABLE_COUNT = 512; // 8^3

// ---------------------------------------------------------------------------
// Quantized-leaf (Fp8/FpN) header layout — `LeafFnBase` (NanoVDB.h). The 96-byte
// header is common to Fp4/Fp8/Fp16/FpN; the compressed code array follows at 96.
// Unlike FLOAT (which stores f32 stats), quantized leaves store the affine
// (mMinimum, mQuantum) pair as f32 and the min/max/ave/dev *statistics* as
// uint16 codes quantized against that same (min, quantum). Verified against the
// native box/sphere fp8+fpn fixtures.
// ---------------------------------------------------------------------------

export const LEAF_OFF_FP_MINIMUM = 80; // f32  mMinimum  (== value at code 0)
export const LEAF_OFF_FP_QUANTUM = 84; // f32  mQuantum  (== (max-min)/((1<<bits)-1))
export const LEAF_OFF_FP_STAT_MIN = 88; // u16  quantized min of active values
export const LEAF_OFF_FP_STAT_MAX = 90; // u16
export const LEAF_OFF_FP_STAT_AVG = 92; // u16
export const LEAF_OFF_FP_STAT_DEV = 94; // u16
export const LEAF_OFF_FP_CODES = 96; // start of the packed code array
export const FP_LEAF_HEADER_SIZE = 96;

/** FpN leaf `mFlags` (high byte of bbox_dif_and_flags): logBitWidth in bits 5-7,
 *  plus the has-bbox bit the native encoder always sets (0x02). Fp8 uses 0x02. */
export const LEAF_FLAG_HAS_BBOX = 2;

// ---------------------------------------------------------------------------
// Per-grid-type layout (FLOAT). Fp8/FpN add sibling blocks in the next wave.
// ---------------------------------------------------------------------------

/** Per-grid-type node sizes, stat offsets and table geometry (ABI 32.9.1). */
export interface GridTypeLayout {
  readonly gridType: number;
  readonly rootSize: number;
  readonly rootTileSize: number;
  readonly rootTileOffValue: number;
  readonly rootOffBackground: number;
  readonly rootOffMin: number;
  readonly rootOffMax: number;
  readonly rootOffAve: number;
  readonly rootOffStdDev: number;
  readonly tableStride: number; // bytes per internal-node table slot
  readonly valueStrideBits: number; // bits per leaf value
  readonly upperSize: number;
  readonly upperOffTable: number;
  readonly upperOffMin: number;
  readonly upperOffMax: number;
  readonly upperOffAve: number;
  readonly upperOffStdDev: number;
  readonly lowerSize: number;
  readonly lowerOffTable: number;
  readonly lowerOffMin: number;
  readonly lowerOffMax: number;
  readonly lowerOffAve: number;
  readonly lowerOffStdDev: number;
  readonly leafSize: number;
  readonly leafOffTable: number;
  readonly leafOffMin: number;
  readonly leafOffMax: number;
  readonly leafOffAve: number;
  readonly leafOffStdDev: number;
}

export const FLOAT_LAYOUT: GridTypeLayout = {
  gridType: GRID_TYPE_FLOAT,
  rootSize: 64,
  rootTileSize: 32,
  rootTileOffValue: 20,
  rootOffBackground: 28,
  rootOffMin: 32,
  rootOffMax: 36,
  rootOffAve: 40,
  rootOffStdDev: 44,
  tableStride: 8,
  valueStrideBits: 32,
  upperSize: 270400,
  upperOffTable: 8256,
  upperOffMin: 8224,
  upperOffMax: 8228,
  upperOffAve: 8232,
  upperOffStdDev: 8236,
  lowerSize: 33856,
  lowerOffTable: 1088,
  lowerOffMin: 1056,
  lowerOffMax: 1060,
  lowerOffAve: 1064,
  lowerOffStdDev: 1068,
  leafSize: 2144,
  leafOffTable: 96,
  leafOffMin: 80,
  leafOffMax: 84,
  leafOffAve: 88,
  leafOffStdDev: 92,
};

/**
 * Quantized (Fp8/FpN) grid layout. Root/upper/lower blocks are byte-for-byte
 * identical to FLOAT (their tiles/stats stay f32); only the leaf differs. For
 * Fp8 `leafSize` is a fixed 608 (96 header + 512 one-byte codes). For FpN the
 * leaf is variable-sized (96 + bitWidth*64), so `leafSize` here is the 96-byte
 * base — the real per-leaf size comes from the codec. The `leafOff*` stat
 * offsets are the u16 statistic slots (the codec writes them directly).
 */
function quantizedLayout(gridType: number, leafSize: number): GridTypeLayout {
  return {
    ...FLOAT_LAYOUT,
    gridType,
    valueStrideBits: 0, // not a uniform stride for quantized leaves
    leafSize,
    leafOffTable: LEAF_OFF_FP_CODES,
    leafOffMin: LEAF_OFF_FP_STAT_MIN,
    leafOffMax: LEAF_OFF_FP_STAT_MAX,
    leafOffAve: LEAF_OFF_FP_STAT_AVG,
    leafOffStdDev: LEAF_OFF_FP_STAT_DEV,
  };
}

/** Fp8: fixed 8-bit codes. Leaf = 96-byte header + 512 bytes = 608. */
export const FP8_LAYOUT: GridTypeLayout = quantizedLayout(GRID_TYPE_FP8, 608);
/** FpN: variable bit-width. `leafSize` is the 96-byte base (see codec). */
export const FPN_LAYOUT: GridTypeLayout = quantizedLayout(GRID_TYPE_FPN, FP_LEAF_HEADER_SIZE);

// ---------------------------------------------------------------------------
// coord -> key / coord -> offset (structural bit math, identical for all types)
// ---------------------------------------------------------------------------

/** pnanovdb_coord_to_key — root-tile key for a voxel coord (native-64 branch). */
export function coordToKey(x: number, y: number, z: number): bigint {
  const iu = BigInt(x >>> 12);
  const ju = BigInt(y >>> 12);
  const ku = BigInt(z >>> 12);
  return ku | (ju << 21n) | (iu << 42n);
}

/** pnanovdb_upper_coord_to_offset — lower-table slot within a 32^3 upper. */
export function upperCoordToOffset(x: number, y: number, z: number): number {
  return (((x & 4095) >> 7) << 10) + (((y & 4095) >> 7) << 5) + ((z & 4095) >> 7);
}

/** pnanovdb_lower_coord_to_offset — leaf-table slot within a 16^3 lower. */
export function lowerCoordToOffset(x: number, y: number, z: number): number {
  return (((x & 127) >> 3) << 8) + (((y & 127) >> 3) << 4) + ((z & 127) >> 3);
}

/** pnanovdb_leaf_coord_to_offset — voxel slot within an 8^3 leaf, 0..511. */
export function leafCoordToOffset(x: number, y: number, z: number): number {
  return ((x & 7) << 6) + ((y & 7) << 3) + (z & 7);
}

// ---------------------------------------------------------------------------
// GridImageWriter — typed little-endian struct writer over one grid image.
// ---------------------------------------------------------------------------

/**
 * A thin, endianness-explicit writer over a single grid image's backing
 * `ArrayBuffer`. All NanoVDB integers/floats are little-endian; every setter
 * here passes `true` so the code never repeats the flag. Mask helpers implement
 * `pnanovdb_*_mask` bit addressing (32-bit words, LSB-first within a word).
 */
export class GridImageWriter {
  readonly bytes: Uint8Array;
  readonly u32: Uint32Array;
  private readonly view: DataView;

  constructor(byteLength: number) {
    if (byteLength % 4 !== 0) {
      throw new Error(`GridImageWriter: byteLength ${byteLength} must be a multiple of 4`);
    }
    const buffer = new ArrayBuffer(byteLength);
    this.bytes = new Uint8Array(buffer);
    this.u32 = new Uint32Array(buffer);
    this.view = new DataView(buffer);
  }

  setU16(off: number, v: number): void {
    this.view.setUint16(off, v & 0xffff, true);
  }
  setU32(off: number, v: number): void {
    this.view.setUint32(off, v >>> 0, true);
  }
  setI32(off: number, v: number): void {
    this.view.setInt32(off, v | 0, true);
  }
  setU64(off: number, v: bigint): void {
    this.view.setBigUint64(off, v, true);
  }
  setI64(off: number, v: bigint): void {
    this.view.setBigInt64(off, v, true);
  }
  setF32(off: number, v: number): void {
    this.view.setFloat32(off, v, true);
  }
  setF64(off: number, v: number): void {
    this.view.setFloat64(off, v, true);
  }

  /** Writes `len` bytes of `src` at `off`, zero-filling nothing beyond it. */
  setBytes(off: number, src: Uint8Array): void {
    this.bytes.set(src, off);
  }

  /** Sets bit `bitIndex` of a mask word array whose word 0 starts at `maskBase`. */
  setMaskBit(maskBase: number, bitIndex: number): void {
    const wordOff = maskBase + 4 * (bitIndex >>> 5);
    this.view.setUint32(
      wordOff,
      (this.view.getUint32(wordOff, true) | (1 << (bitIndex & 31))) >>> 0,
      true,
    );
  }

  /** Copies a whole precomputed 32-bit mask (`words` u32s) to `maskBase`. */
  setMaskWords(maskBase: number, words: Uint32Array): void {
    for (let i = 0; i < words.length; i++) {
      this.view.setUint32(maskBase + i * 4, words[i]! >>> 0, true);
    }
  }
}
