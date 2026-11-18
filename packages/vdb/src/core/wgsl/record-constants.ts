/**
 * Pool header and grid-record word offsets the generated WGSL reads
 * (docs/design/K1-core.md §1.1–1.2, layout 1).
 *
 * TODO(K1.3a merge): replace this module with the WGSL constants generator in
 * `core/pool/layout.ts`, which owns the layout (`VDB_GPU_LAYOUT_VERSION`,
 * `encodeRecord`/`decodeRecord`). Keep this seam small: the generator only
 * needs `VDB_GPU_LAYOUT_VERSION` and `recordConstantsWgsl()`.
 */

/** Layout version the generated module targets (pool header word 1). */
export const VDB_GPU_LAYOUT_VERSION = 1;

/** Byte offset of the grid table in the pool (header is 32 bytes). */
export const POOL_TABLE_OFFSET_BYTES = 32;
/** Bytes per grid record. */
export const POOL_RECORD_BYTES = 160;

/** Header words (§1.1). */
export const POOL_HEADER_WORD = {
  magic: 0,
  layoutVersion: 1,
  gridCapacity: 2,
  gridCount: 3,
  tableOffset: 4,
  recordBytes: 5,
  usedBytes: 6,
} as const;

/** Record words (§1.2). */
export const RECORD_WORD = {
  imageOffset: 0,
  imageBytes: 1,
  gridType: 2,
  flags: 3,
  worldToIndex: 4,
  indexToWorld: 16,
  indexBBoxMin: 28,
  background: 31,
  indexBBoxMax: 32,
  valueMax: 35,
  majorantOffset: 36,
  majorantDims: 37,
} as const;

/** Record flag bits (word 3). */
export const RECORD_FLAG = {
  live: 1 << 0,
  hasStats: 1 << 1,
  hasBrickMajorant: 1 << 2,
  instanceTransform: 1 << 3,
} as const;

const constantLine = (name: string, value: number) => `const ${name}: u32 = ${value}u;`;

/**
 * Every layout constant as WGSL, one `const` per line. The generator keeps only
 * the lines whose names the module references.
 */
export function recordConstantsWgsl(): string[] {
  return [
    constantLine("VDB_POOL_LAYOUT_VERSION", VDB_GPU_LAYOUT_VERSION),
    constantLine("VDB_POOL_HEADER_GRID_COUNT", POOL_HEADER_WORD.gridCount),
    constantLine("VDB_POOL_TABLE_WORD", POOL_TABLE_OFFSET_BYTES / 4),
    constantLine("VDB_POOL_RECORD_WORDS", POOL_RECORD_BYTES / 4),
    constantLine("VDB_REC_IMAGE_OFFSET", RECORD_WORD.imageOffset),
    constantLine("VDB_REC_IMAGE_BYTES", RECORD_WORD.imageBytes),
    constantLine("VDB_REC_GRID_TYPE", RECORD_WORD.gridType),
    constantLine("VDB_REC_FLAGS", RECORD_WORD.flags),
    constantLine("VDB_REC_WORLD_TO_INDEX", RECORD_WORD.worldToIndex),
    constantLine("VDB_REC_INDEX_TO_WORLD", RECORD_WORD.indexToWorld),
    constantLine("VDB_REC_INDEX_BBOX_MIN", RECORD_WORD.indexBBoxMin),
    constantLine("VDB_REC_BACKGROUND", RECORD_WORD.background),
    constantLine("VDB_REC_INDEX_BBOX_MAX", RECORD_WORD.indexBBoxMax),
    constantLine("VDB_REC_VALUE_MAX", RECORD_WORD.valueMax),
    constantLine("VDB_REC_MAJORANT_OFFSET", RECORD_WORD.majorantOffset),
    constantLine("VDB_REC_MAJORANT_DIMS", RECORD_WORD.majorantDims),
    constantLine("VDB_REC_FLAG_LIVE", RECORD_FLAG.live),
    constantLine("VDB_REC_FLAG_HAS_STATS", RECORD_FLAG.hasStats),
    constantLine("VDB_REC_FLAG_HAS_BRICK_MAJORANT", RECORD_FLAG.hasBrickMajorant),
    constantLine("VDB_REC_FLAG_INSTANCE_TRANSFORM", RECORD_FLAG.instanceTransform),
  ];
}
