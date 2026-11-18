/**
 * Pool buffer layout 1 (docs/design/K1-core.md §1): a 32-byte header, a table of
 * fixed-stride 160-byte grid records, then 32-byte-aligned ranges (NanoVDB
 * images, brick-majorant grids).
 *
 * This file is the single source of truth for every header word, record word
 * and flag bit. The TypeScript encoders below and the WGSL constants returned by
 * {@link vdbPoolLayoutWgsl} are both derived from the tables in this file, so the
 * CPU writer and the GPU reader cannot drift apart. Any change to a word or its
 * meaning bumps {@link VDB_GPU_LAYOUT_VERSION} and is a breaking change (§1.8).
 */

import type { VDBVec3 } from "../../tools/vdb/types.js";
import { VDBError } from "../errors.js";

/** Layout version written to header word 1 and baked into generated WGSL. */
export const VDB_GPU_LAYOUT_VERSION = 1;

/** Header word 0: `"VDBP"` read as a little-endian u32. */
export const VDB_POOL_MAGIC = 0x50424456;

/** Every range in the pool (table, images, majorants) starts on this byte alignment (NanoVDB's data alignment). */
export const VDB_POOL_ALIGN = 32;

/** Header size in bytes. */
export const VDB_POOL_HEADER_BYTES = 32;
/** Header size in u32 words. */
export const VDB_POOL_HEADER_WORDS = VDB_POOL_HEADER_BYTES / 4;

/** Byte offset of record 0 (fixed in layout 1). */
export const VDB_POOL_TABLE_OFFSET = 32;

/** Grid record size in bytes. A multiple of 16 so `vec4` loads stay possible. */
export const VDB_POOL_RECORD_BYTES = 160;
/** Grid record size in u32 words. */
export const VDB_POOL_RECORD_WORDS = VDB_POOL_RECORD_BYTES / 4;

/** Header word indices (§1.1). */
export const VDB_HEADER_WORD = {
  /** {@link VDB_POOL_MAGIC}. */
  magic: 0,
  /** {@link VDB_GPU_LAYOUT_VERSION}. */
  layoutVersion: 1,
  /** Records in the table. */
  gridCapacity: 2,
  /** Highest live grid id + 1 (`vdbGridCount()` in WGSL). */
  gridCount: 3,
  /** {@link VDB_POOL_TABLE_OFFSET}, written for readers that don't know the version. */
  tableOffset: 4,
  /** {@link VDB_POOL_RECORD_BYTES}. */
  recordBytes: 5,
  /** Allocation high-water mark in bytes (diagnostics). */
  usedBytes: 6,
  /** Always 0 in layout 1. */
  reserved: 7,
} as const;

/** Grid record word indices (§1.2). Multi-word fields name their first word. */
export const VDB_RECORD_WORD = {
  /** u32: image byte offset from the start of the pool; a multiple of 32. */
  imageOffset: 0,
  /** u32: NanoVDB `GridData.mGridSize` (its high word must be 0). */
  imageBytes: 1,
  /** u32: NanoVDB `GridType` id (1 Float, 13 Fp4, 14 Fp8, 15 Fp16, 16 FpN). */
  gridType: 2,
  /** u32: see {@link VDB_RECORD_FLAG} and the class / brick-log2 bit fields. */
  flags: 3,
  /** f32 x12: world -> index, 3x4 row-major (`idx = M * [p, 1]`). */
  worldToIndex: 4,
  /** f32 x12: index -> world, 3x4 row-major. */
  indexToWorld: 16,
  /** i32 x3: root active bbox min. */
  indexBBoxMin: 28,
  /** f32: root background value. */
  background: 31,
  /** i32 x3: root active bbox max (inclusive). */
  indexBBoxMax: 32,
  /** f32: `max(root max stat, background)` or the caller's bound; the `global` majorant. */
  valueMax: 35,
  /** u32: brick-majorant range byte offset; 0 = none. */
  majorantOffset: 36,
  /** u32 x3: brick-majorant grid dimensions in cells. */
  majorantDims: 37,
} as const;

/** Single-bit record flags (record word 3). */
export const VDB_RECORD_FLAG = {
  /** b0: the record describes a live grid; a dead id reads 0 everywhere. */
  live: 1 << 0,
  /** b1: the image carries min/max stats (`GridData` flag `HasMinMax`). */
  hasStats: 1 << 1,
  /** b2: `majorantOffset`/`majorantDims` describe a brick-majorant range. */
  hasBrickMajorant: 1 << 2,
  /** b3: the affines include a host instance transform (`setTransform`). */
  instanceTransform: 1 << 3,
} as const;

/** Bit position of the 8-bit NanoVDB `GridClass` field in record word 3. */
export const VDB_RECORD_GRID_CLASS_SHIFT = 8;
/** Bit position of the 8-bit brick log2 field in record word 3. */
export const VDB_RECORD_BRICK_LOG2_SHIFT = 16;
/** Mask (after shifting) of the 8-bit class and brick-log2 fields. */
export const VDB_RECORD_BYTE_FIELD_MASK = 0xff;

/** Decoded pool header (§1.1). */
export interface VDBPoolHeader {
  magic: number;
  layoutVersion: number;
  gridCapacity: number;
  gridCount: number;
  tableOffset: number;
  recordBytes: number;
  usedBytes: number;
}

// One definition, shared with `@pmndrs/vdb/tools` (D32).
export type { VDBVec3 };

/**
 * Decoded grid record (§1.2). Affines hold the f32-rounded values the GPU sees;
 * 12 entries each, 3x4 row-major.
 */
export interface VDBGridRecord {
  imageOffset: number;
  imageBytes: number;
  gridType: number;
  live: boolean;
  hasStats: boolean;
  hasBrickMajorant: boolean;
  instanceTransform: boolean;
  /** NanoVDB `GridClass` id (0-255). */
  gridClass: number;
  /** log2 of the brick edge in voxels (0-255; 0 when there is no brick majorant). */
  brickLog2: number;
  worldToIndex: number[];
  indexToWorld: number[];
  indexBBoxMin: VDBVec3;
  indexBBoxMax: VDBVec3;
  background: number;
  valueMax: number;
  majorantOffset: number;
  majorantDims: VDBVec3;
}

/** Rounds `n` up to the pool alignment ({@link VDB_POOL_ALIGN}). */
export function align32(n: number): number {
  return Math.ceil(n / VDB_POOL_ALIGN) * VDB_POOL_ALIGN;
}

/** Byte offset of record `id`. */
export function recordByteOffset(id: number): number {
  return VDB_POOL_TABLE_OFFSET + id * VDB_POOL_RECORD_BYTES;
}

/** First byte after a table of `capacity` records, aligned: where ranges may start. */
export function dataStartOffset(capacity: number): number {
  return align32(VDB_POOL_TABLE_OFFSET + capacity * VDB_POOL_RECORD_BYTES);
}

/** Writes the header's 8 words into `out` at `wordOffset` and returns `out`. */
export function encodeHeader(
  header: Omit<VDBPoolHeader, "magic" | "layoutVersion" | "tableOffset" | "recordBytes">,
  out: Uint32Array = new Uint32Array(VDB_POOL_HEADER_WORDS),
  wordOffset = 0,
): Uint32Array {
  const w = VDB_HEADER_WORD;
  out[wordOffset + w.magic] = VDB_POOL_MAGIC;
  out[wordOffset + w.layoutVersion] = VDB_GPU_LAYOUT_VERSION;
  out[wordOffset + w.gridCapacity] = header.gridCapacity;
  out[wordOffset + w.gridCount] = header.gridCount;
  out[wordOffset + w.tableOffset] = VDB_POOL_TABLE_OFFSET;
  out[wordOffset + w.recordBytes] = VDB_POOL_RECORD_BYTES;
  out[wordOffset + w.usedBytes] = header.usedBytes;
  out[wordOffset + w.reserved] = 0;
  return out;
}

/** Reads a header from `words` at `wordOffset` (no validation; see {@link VDBPoolHeader}). */
export function decodeHeader(words: Uint32Array, wordOffset = 0): VDBPoolHeader {
  const w = VDB_HEADER_WORD;
  const at = (i: number) => words[wordOffset + i] ?? 0;
  return {
    magic: at(w.magic),
    layoutVersion: at(w.layoutVersion),
    gridCapacity: at(w.gridCapacity),
    gridCount: at(w.gridCount),
    tableOffset: at(w.tableOffset),
    recordBytes: at(w.recordBytes),
    usedBytes: at(w.usedBytes),
  };
}

function views(words: Uint32Array): { f32: Float32Array; i32: Int32Array } {
  return {
    f32: new Float32Array(words.buffer, words.byteOffset, words.length),
    i32: new Int32Array(words.buffer, words.byteOffset, words.length),
  };
}

function byteField(name: string, value: number): number {
  if (!Number.isInteger(value) || value < 0 || value > VDB_RECORD_BYTE_FIELD_MASK) {
    throw new VDBError("VDB_BAD_OPTION", `encodeRecord: ${name} must be an integer in 0..255, got ${value}.`);
  }
  return value;
}

/**
 * Writes `record` as 40 words into `out` at `wordOffset` and returns `out`.
 * Affines are rounded to f32 here; pass the f64 values you composed.
 */
export function encodeRecord(
  record: VDBGridRecord,
  out: Uint32Array = new Uint32Array(VDB_POOL_RECORD_WORDS),
  wordOffset = 0,
): Uint32Array {
  if (record.worldToIndex.length !== 12 || record.indexToWorld.length !== 12) {
    throw new VDBError("VDB_BAD_OPTION", "encodeRecord: worldToIndex and indexToWorld must each have 12 entries (3x4 row-major).");
  }
  if (out.length < wordOffset + VDB_POOL_RECORD_WORDS) {
    throw new VDBError(
      "VDB_BAD_OPTION",
      `encodeRecord: output has ${out.length} words; a record at word ${wordOffset} needs ${wordOffset + VDB_POOL_RECORD_WORDS}.`,
    );
  }
  const { f32, i32 } = views(out);
  const r = VDB_RECORD_WORD;
  const o = wordOffset;
  const F = VDB_RECORD_FLAG;
  let flags = 0;
  if (record.live) flags |= F.live;
  if (record.hasStats) flags |= F.hasStats;
  if (record.hasBrickMajorant) flags |= F.hasBrickMajorant;
  if (record.instanceTransform) flags |= F.instanceTransform;
  flags |= byteField("gridClass", record.gridClass) << VDB_RECORD_GRID_CLASS_SHIFT;
  flags |= byteField("brickLog2", record.brickLog2) << VDB_RECORD_BRICK_LOG2_SHIFT;

  out[o + r.imageOffset] = record.imageOffset;
  out[o + r.imageBytes] = record.imageBytes;
  out[o + r.gridType] = record.gridType;
  out[o + r.flags] = flags >>> 0;
  for (let k = 0; k < 12; k++) {
    f32[o + r.worldToIndex + k] = record.worldToIndex[k]!;
    f32[o + r.indexToWorld + k] = record.indexToWorld[k]!;
  }
  for (let k = 0; k < 3; k++) {
    i32[o + r.indexBBoxMin + k] = record.indexBBoxMin[k]!;
    i32[o + r.indexBBoxMax + k] = record.indexBBoxMax[k]!;
    out[o + r.majorantDims + k] = record.majorantDims[k]!;
  }
  f32[o + r.background] = record.background;
  f32[o + r.valueMax] = record.valueMax;
  out[o + r.majorantOffset] = record.majorantOffset;
  return out;
}

/** Reads the 40-word record at `wordOffset` of `words`. */
export function decodeRecord(words: Uint32Array, wordOffset = 0): VDBGridRecord {
  if (words.length < wordOffset + VDB_POOL_RECORD_WORDS) {
    throw new VDBError(
      "VDB_BAD_OPTION",
      `decodeRecord: input has ${words.length} words; a record at word ${wordOffset} needs ${wordOffset + VDB_POOL_RECORD_WORDS}.`,
    );
  }
  const { f32, i32 } = views(words);
  const r = VDB_RECORD_WORD;
  const o = wordOffset;
  const flags = words[o + r.flags]!;
  const F = VDB_RECORD_FLAG;
  const vec = (src: Int32Array | Uint32Array, at: number): VDBVec3 => [src[o + at]!, src[o + at + 1]!, src[o + at + 2]!];
  return {
    imageOffset: words[o + r.imageOffset]!,
    imageBytes: words[o + r.imageBytes]!,
    gridType: words[o + r.gridType]!,
    live: (flags & F.live) !== 0,
    hasStats: (flags & F.hasStats) !== 0,
    hasBrickMajorant: (flags & F.hasBrickMajorant) !== 0,
    instanceTransform: (flags & F.instanceTransform) !== 0,
    gridClass: (flags >>> VDB_RECORD_GRID_CLASS_SHIFT) & VDB_RECORD_BYTE_FIELD_MASK,
    brickLog2: (flags >>> VDB_RECORD_BRICK_LOG2_SHIFT) & VDB_RECORD_BYTE_FIELD_MASK,
    worldToIndex: Array.from(f32.subarray(o + r.worldToIndex, o + r.worldToIndex + 12)),
    indexToWorld: Array.from(f32.subarray(o + r.indexToWorld, o + r.indexToWorld + 12)),
    indexBBoxMin: vec(i32, r.indexBBoxMin),
    indexBBoxMax: vec(i32, r.indexBBoxMax),
    background: f32[o + r.background]!,
    valueMax: f32[o + r.valueMax]!,
    majorantOffset: words[o + r.majorantOffset]!,
    majorantDims: vec(words, r.majorantDims),
  };
}

/** `fooBarBaz` -> `FOO_BAR_BAZ`. */
function upperSnake(name: string): string {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toUpperCase();
}

function u32Hex(n: number): string {
  return `0x${(n >>> 0).toString(16)}u`;
}

/**
 * WGSL `const` declarations for layout {@link VDB_GPU_LAYOUT_VERSION}, generated
 * from the tables in this file. Pure and deterministic: same package version,
 * same text. Every name carries the `VDB_` prefix (D10, K1-core §3.4) and no
 * comment contains a back-tick (CLAUDE.md Traps).
 *
 * Emitted names (all `u32`):
 * - `VDB_GPU_LAYOUT_VERSION`, `VDB_POOL_MAGIC`
 * - `VDB_POOL_TABLE_OFFSET` (bytes), `VDB_POOL_TABLE_WORD` (= offset / 4),
 *   `VDB_POOL_RECORD_BYTES`, `VDB_POOL_RECORD_WORDS`, `VDB_POOL_ALIGN`
 * - `VDB_HEADER_<FIELD>` for every {@link VDB_HEADER_WORD} entry (word index),
 *   e.g. `VDB_HEADER_GRID_COUNT = 3u`
 * - `VDB_REC_<FIELD>` for every {@link VDB_RECORD_WORD} entry (word index within
 *   a record), e.g. `VDB_REC_IMAGE_OFFSET = 0u`, `VDB_REC_WORLD_TO_INDEX = 4u`
 * - `VDB_REC_FLAG_<NAME>` for every {@link VDB_RECORD_FLAG} bit, plus
 *   `VDB_REC_GRID_CLASS_SHIFT`, `VDB_REC_BRICK_LOG2_SHIFT`, `VDB_REC_BYTE_FIELD_MASK`
 *
 * Word `w` of record `g` is `vdb_load(VDB_POOL_TABLE_WORD + g * VDB_POOL_RECORD_WORDS + w)`.
 */
export function vdbPoolLayoutWgsl(): string {
  const lines: string[] = [
    `// @pmndrs/vdb pool layout ${VDB_GPU_LAYOUT_VERSION} (generated from core/pool/layout.ts; do not edit).`,
    `const VDB_GPU_LAYOUT_VERSION: u32 = ${VDB_GPU_LAYOUT_VERSION}u;`,
    `const VDB_POOL_MAGIC: u32 = ${u32Hex(VDB_POOL_MAGIC)};`,
    `const VDB_POOL_ALIGN: u32 = ${VDB_POOL_ALIGN}u;`,
    `const VDB_POOL_TABLE_OFFSET: u32 = ${VDB_POOL_TABLE_OFFSET}u;`,
    `const VDB_POOL_TABLE_WORD: u32 = ${VDB_POOL_TABLE_OFFSET / 4}u;`,
    `const VDB_POOL_RECORD_BYTES: u32 = ${VDB_POOL_RECORD_BYTES}u;`,
    `const VDB_POOL_RECORD_WORDS: u32 = ${VDB_POOL_RECORD_WORDS}u;`,
  ];
  for (const [name, word] of Object.entries(VDB_HEADER_WORD)) {
    lines.push(`const VDB_HEADER_${upperSnake(name)}: u32 = ${word}u;`);
  }
  for (const [name, word] of Object.entries(VDB_RECORD_WORD)) {
    lines.push(`const VDB_REC_${upperSnake(name)}: u32 = ${word}u;`);
  }
  for (const [name, bit] of Object.entries(VDB_RECORD_FLAG)) {
    lines.push(`const VDB_REC_FLAG_${upperSnake(name)}: u32 = ${u32Hex(bit)};`);
  }
  lines.push(
    `const VDB_REC_GRID_CLASS_SHIFT: u32 = ${VDB_RECORD_GRID_CLASS_SHIFT}u;`,
    `const VDB_REC_BRICK_LOG2_SHIFT: u32 = ${VDB_RECORD_BRICK_LOG2_SHIFT}u;`,
    `const VDB_REC_BYTE_FIELD_MASK: u32 = ${u32Hex(VDB_RECORD_BYTE_FIELD_MASK)};`,
  );
  return lines.join("\n") + "\n";
}
