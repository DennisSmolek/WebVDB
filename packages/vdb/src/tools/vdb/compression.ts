/**
 * compression.ts — per-node value array (de)serialization.
 *
 * This is the risky, easy-to-get-subtly-wrong part of the format: OpenVDB's
 * "node mask compression" scheme (file version >= 222) can omit inactive
 * values entirely, storing only the active ones plus up to two "inactive
 * fill" values and a second selection mask to tell them apart — then the
 * reader has to re-expand that sparse array back to the node's full
 * `dim^3` table. Algorithm and field order lifted (format facts only, not
 * code) from `vdb-rs`'s `read_compressed`/`read_compressed_data`, which is
 * the more careful of the two references (the `mjurczyk/openvdb` JS reader
 * never actually implements leaf value decoding at all — see grid.ts docs),
 * and checked against OpenVDB `io/Compression.h` (`readCompressedValues`,
 * `HalfReader`, `readData`).
 *
 * K4.2: values are no longer float-only. A {@link ValueFormat} describes the
 * grid's `ValueType` (float, vec3s, vec3d, vec3i, PointDataIndex32); decoded
 * values land in a `Float32Array` with `components` scalars per value.
 */

import { countBits, testBit } from "./bit-utils.js";
import type { ByteReader } from "./byte-reader.js";
import { decompressBlosc } from "./blosc.js";
import { VdbFormatError } from "./errors.js";
import { halfToFloat } from "./half-float.js";
import { inflateZlib } from "./inflate.js";

export interface CompressionFlags {
  zip: boolean;
  activeMask: boolean;
  blosc: boolean;
}

export function compressionFromBits(bits: number): CompressionFlags {
  return {
    zip: (bits & 0x1) !== 0,
    activeMask: (bits & 0x2) !== 0,
    blosc: (bits & 0x4) !== 0,
  };
}

/** `NodeMetaData` (openvdb/io/Compression.h) — how a node's value array was
 *  reduced before writing, when node-mask compression (v222+) is in play. */
const enum NodeMetaData {
  NoMaskOrInactiveVals = 0,
  NoMaskAndMinusBg = 1,
  NoMaskAndOneInactiveVal = 2,
  MaskAndNoInactiveVals = 3,
  MaskAndOneInactiveVal = 4,
  MaskAndTwoInactiveVals = 5,
  NoMaskAndAllVals = 6,
}

// ---------------------------------------------------------------------------
// Value formats
// ---------------------------------------------------------------------------

/** On-disk encoding of one value component at full precision. */
export type ComponentKind = "f32" | "f64" | "i32" | "u32";

/**
 * How a grid's `ValueType` is laid out on disk: `components` scalars of
 * `kind` each (`sizeof(ValueType) = components * bytes(kind)`). Decoded values
 * always land in a `Float32Array`, `components` entries per value (xyz
 * interleaved for vectors). That is exact for f32, and for i32/u32 below 2^24;
 * f64 is narrowed to f32.
 */
export interface ValueFormat {
  readonly components: 1 | 3;
  readonly kind: ComponentKind;
}

export const FLOAT_FORMAT: ValueFormat = { components: 1, kind: "f32" };
export const VEC3S_FORMAT: ValueFormat = { components: 3, kind: "f32" };
export const VEC3D_FORMAT: ValueFormat = { components: 3, kind: "f64" };
export const VEC3I_FORMAT: ValueFormat = { components: 3, kind: "i32" };
/** `PointDataIndex32`: internal-node and root values of a point data grid. */
export const PTDATAIDX32_FORMAT: ValueFormat = { components: 1, kind: "u32" };

const KIND_BYTES: Record<ComponentKind, number> = { f32: 4, f64: 8, i32: 4, u32: 4 };

/** `sizeof(ValueType)` at full precision. */
export function valueBytes(format: ValueFormat): number {
  return format.components * KIND_BYTES[format.kind];
}

/**
 * OpenVDB `RealToHalf<T>::isReal`: half-float storage applies to float,
 * double and their vector types only. An integer grid "saved as half" still
 * writes full-width values (`HalfReader<false, T>` is plain `readData`).
 */
export function isRealFormat(format: ValueFormat): boolean {
  return format.kind === "f32" || format.kind === "f64";
}

function readComponent(reader: ByteReader, kind: ComponentKind): number {
  switch (kind) {
    case "f32":
      return reader.f32();
    case "f64":
      return reader.f64();
    case "i32":
      return reader.i32();
    case "u32":
      return reader.u32();
  }
}

/** Reads one full-width `ValueType` (root background, root tiles, inactive fill values). */
export function readFullValue(reader: ByteReader, format: ValueFormat): number[] {
  const out = new Array<number>(format.components);
  for (let c = 0; c < format.components; c++) out[c] = readComponent(reader, format.kind);
  return out;
}

/** Decodes `scalars` little-endian scalars of `kind` (or halves) into a new `Float32Array`. */
function decodeScalars(raw: Uint8Array, scalars: number, kind: ComponentKind, half: boolean): Float32Array {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out = new Float32Array(scalars);
  if (half) {
    for (let i = 0; i < scalars; i++) out[i] = halfToFloat(view.getUint16(i * 2, true));
    return out;
  }
  switch (kind) {
    case "f32":
      for (let i = 0; i < scalars; i++) out[i] = view.getFloat32(i * 4, true);
      break;
    case "f64":
      for (let i = 0; i < scalars; i++) out[i] = view.getFloat64(i * 8, true);
      break;
    case "i32":
      for (let i = 0; i < scalars; i++) out[i] = view.getInt32(i * 4, true);
      break;
    case "u32":
      for (let i = 0; i < scalars; i++) out[i] = view.getUint32(i * 4, true);
      break;
  }
  return out;
}

/** Reads the raw (pre node-mask-expansion) value payload: `count` values
 *  (`count * components` scalars), each full width or half, through
 *  blosc/zip/no compression (`io::readData` / `io::HalfReader::read`). */
function readRawValues(
  reader: ByteReader,
  count: number,
  half: boolean,
  format: ValueFormat,
  compression: CompressionFlags,
): Float32Array {
  const scalarBytes = half ? 2 : KIND_BYTES[format.kind];
  const elementSize = scalarBytes * format.components;

  const decodeBytes = (raw: Uint8Array, n: number): Float32Array => {
    if (raw.length < n * elementSize) {
      throw new VdbFormatError(`value payload is ${raw.length} bytes, expected ${n * elementSize}`);
    }
    return decodeScalars(raw, n * format.components, format.kind, half);
  };

  const readPlain = (n: number): Float32Array => decodeBytes(reader.bytes_(n * elementSize), n);

  /** The `<= 0` sentinel: raw values follow, `-n / elementSize` of them. */
  const readSentinelRaw = (numBytes: number): Float32Array => {
    const rawCount = -numBytes / elementSize;
    if (!Number.isInteger(rawCount) || rawCount < 0 || rawCount > count) {
      throw new VdbFormatError(
        `raw-value sentinel ${numBytes} is not a whole number of ${elementSize}-byte values within the expected ${count}`,
      );
    }
    return readPlain(rawCount);
  };

  if (compression.blosc) {
    const numCompressedBytes = readI64AsNumber(reader);
    if (numCompressedBytes <= 0) return readSentinelRaw(numCompressedBytes);
    // OpenVDB pads buffers shorter than Blosc's 48-byte minimum before
    // compressing, so the frame can decode to more than `count` values.
    const maxBytes = Math.max(count * elementSize, 48);
    const decoded = decompressBlosc(reader.bytes_(numCompressedBytes), {
      maxBytes,
      expectedCbytes: numCompressedBytes,
    });
    if (decoded.length < count * elementSize) {
      throw new VdbFormatError(`blosc frame decoded to ${decoded.length} bytes, expected ${count * elementSize}`);
    }
    return decodeBytes(decoded.subarray(0, count * elementSize), count);
  }

  if (compression.zip) {
    const numZippedBytes = readI64AsNumber(reader);
    if (numZippedBytes <= 0) return readSentinelRaw(numZippedBytes);
    const zippedBytes = reader.bytes_(numZippedBytes);
    const inflated = inflateZlib(zippedBytes, count * elementSize);
    return decodeBytes(inflated.subarray(0, count * elementSize), count);
  }

  return readPlain(count);
}

/** Reads a signed 64-bit stream field narrowed to `number` (compressed byte
 *  counts are written as `int64_t` and can legitimately be <= 0 as a
 *  "not worth compressing, here's the raw size instead" sentinel). */
function readI64AsNumber(reader: ByteReader): number {
  const bytes = reader.bytes_(8);
  const view = new DataView(bytes.buffer, bytes.byteOffset, 8);
  const v = view.getBigInt64(0, true);
  if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < -BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VdbFormatError(`i64 value ${v} exceeds safe integer range`);
  }
  return Number(v);
}

export interface ReadCompressedOptions {
  fileVersion: number;
  compression: CompressionFlags;
  /** The grid was saved as half float (honoured for real formats only). */
  useHalf: boolean;
  /** The grid background, `format.components` entries. */
  background: ArrayLike<number>;
  /** Full logical size of this node's value table (32768 / 4096 / 512). */
  numValues: number;
  valueMask: Uint32Array;
  /** Value layout. Default: {@link FLOAT_FORMAT}. */
  format?: ValueFormat;
}

/** The generic "node value array" reader — used for internal-node tile
 *  values (Node5/Node4) and leaf voxel values alike; both go through the
 *  same on-disk encoding (`io::Compression::readCompressedValues`). Returns
 *  `numValues * format.components` scalars. */
export function readCompressedValues(reader: ByteReader, opts: ReadCompressedOptions): Float32Array {
  const { fileVersion, compression, numValues, valueMask } = opts;
  const format = opts.format ?? FLOAT_FORMAT;
  const comps = format.components;
  const half = opts.useHalf && isRealFormat(format);

  let meta: NodeMetaData = NodeMetaData.NoMaskOrInactiveVals;
  if (fileVersion >= 222) {
    meta = reader.u8() as NodeMetaData;
  }

  // Inactive fill values. They default to 0 here, not to OpenVDB's
  // `background` / `-background` (open issue K4.8); only `NoMaskAndMinusBg`
  // uses the background.
  let inactiveVal0: ArrayLike<number> = new Array<number>(comps).fill(0);
  let inactiveVal1: ArrayLike<number> = inactiveVal0;
  if (meta === NodeMetaData.NoMaskAndMinusBg) {
    inactiveVal0 = Array.from({ length: comps }, (_, c) => -opts.background[c]!);
  } else if (
    meta === NodeMetaData.NoMaskAndOneInactiveVal ||
    meta === NodeMetaData.MaskAndOneInactiveVal ||
    meta === NodeMetaData.MaskAndTwoInactiveVals
  ) {
    // Written at full `sizeof(ValueType)`, even for half-float grids.
    inactiveVal0 = readFullValue(reader, format);
    inactiveVal1 = meta === NodeMetaData.MaskAndTwoInactiveVals ? readFullValue(reader, format) : inactiveVal0;
  }

  let selectionMask: Uint32Array | null = null;
  if (
    meta === NodeMetaData.MaskAndNoInactiveVals ||
    meta === NodeMetaData.MaskAndOneInactiveVal ||
    meta === NodeMetaData.MaskAndTwoInactiveVals
  ) {
    selectionMask = reader.maskWords(numValues);
  }

  const onDiskCount =
    compression.activeMask && meta !== NodeMetaData.NoMaskAndAllVals && fileVersion >= 222
      ? countBits(valueMask)
      : numValues;

  // K4.7: `io::HalfReader<true, T>::read` returns before touching the stream
  // when `count < 1`, because `HalfWriter<true, T>::write` writes nothing for
  // an empty array, not even the blosc/zip size field. So a half-float node
  // whose mask compression leaves no values (an internal node without active
  // tiles, typically) has no payload at all. Full-width and integer arrays go
  // through plain `readData`, which still reads the size field for count 0.
  const raw =
    half && onDiskCount < 1 ? new Float32Array(0) : readRawValues(reader, onDiskCount, half, format, compression);

  if (raw.length !== onDiskCount * comps) {
    throw new VdbFormatError(`node value payload holds ${raw.length / comps} values, expected ${onDiskCount}`);
  }

  if (compression.activeMask && onDiskCount !== numValues) {
    const expanded = new Float32Array(numValues * comps);
    let readIdx = 0;
    if (comps === 1) {
      const fill0 = inactiveVal0[0]!;
      const fill1 = inactiveVal1[0]!;
      for (let i = 0; i < numValues; i++) {
        if (testBit(valueMask, i)) expanded[i] = raw[readIdx++]!;
        else expanded[i] = selectionMask && testBit(selectionMask, i) ? fill1 : fill0;
      }
      return expanded;
    }
    for (let i = 0; i < numValues; i++) {
      let src: ArrayLike<number> = inactiveVal0;
      let srcBase = 0;
      if (testBit(valueMask, i)) {
        src = raw;
        srcBase = readIdx++ * comps;
      } else if (selectionMask && testBit(selectionMask, i)) {
        src = inactiveVal1;
      }
      for (let c = 0; c < comps; c++) expanded[i * comps + c] = src[srcBase + c]!;
    }
    return expanded;
  }

  return raw;
}
