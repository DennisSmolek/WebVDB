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
 * never actually implements leaf value decoding at all — see grid.ts docs).
 */

import { countBits, testBit } from "./bit-utils.js";
import type { ByteReader } from "./byte-reader.js";
import { VdbFormatError, VdbUnsupportedError } from "./errors.js";
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

/** Reads the raw (post node-mask-expansion) value payload: `count` values,
 *  each either half or full float, through blosc/zip/no compression. */
function readRawValues(
  reader: ByteReader,
  count: number,
  useHalf: boolean,
  compression: CompressionFlags,
): Float32Array {
  const elementSize = useHalf ? 2 : 4;

  const readPlain = (n: number): Float32Array => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = useHalf ? reader.f16AsF32(halfToFloat) : reader.f32();
    }
    return out;
  };

  const decodeBytes = (raw: Uint8Array, n: number): Float32Array => {
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = useHalf ? halfToFloat(view.getUint16(i * 2, true)) : view.getFloat32(i * 4, true);
    }
    return out;
  };

  if (compression.blosc) {
    const numCompressedBytes = readI64AsNumber(reader);
    if (numCompressedBytes <= 0) {
      // Sentinel meaning "not worth compressing" — raw values follow instead.
      const rawCount = -numCompressedBytes / elementSize;
      return readPlain(rawCount);
    }
    reader.bytes_(numCompressedBytes); // consume so any sibling grid stays in sync
    throw new VdbUnsupportedError(
      "blosc-compressed .vdb input is not supported — re-export the file with zlib or no compression (per SPEC §4)",
    );
  }

  if (compression.zip) {
    const numZippedBytes = readI64AsNumber(reader);
    if (numZippedBytes <= 0) {
      const rawCount = -numZippedBytes / elementSize;
      return readPlain(rawCount);
    }
    const zippedBytes = reader.bytes_(numZippedBytes);
    const inflated = inflateZlib(zippedBytes);
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
  useHalf: boolean;
  background: number;
  /** Full logical size of this node's value table (32768 / 4096 / 512). */
  numValues: number;
  valueMask: Uint32Array;
}

/** The generic "node value array" reader — used for internal-node tile
 *  values (Node5/Node4) and leaf voxel values alike; both go through the
 *  same on-disk encoding (`io::Compression::readCompressedValues`). */
export function readCompressedValues(reader: ByteReader, opts: ReadCompressedOptions): Float32Array {
  const { fileVersion, compression, numValues, valueMask } = opts;

  let meta: NodeMetaData = NodeMetaData.NoMaskOrInactiveVals;
  if (fileVersion >= 222) {
    meta = reader.u8() as NodeMetaData;
  }

  let inactiveVal0 = 0;
  let inactiveVal1 = 0;
  if (meta === NodeMetaData.NoMaskAndMinusBg) {
    inactiveVal0 = -opts.background;
  } else if (
    meta === NodeMetaData.NoMaskAndOneInactiveVal ||
    meta === NodeMetaData.MaskAndOneInactiveVal ||
    meta === NodeMetaData.MaskAndTwoInactiveVals
  ) {
    inactiveVal0 = reader.f32();
    if (meta === NodeMetaData.MaskAndTwoInactiveVals) {
      inactiveVal1 = reader.f32();
    } else {
      inactiveVal1 = inactiveVal0;
    }
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

  const raw = readRawValues(reader, onDiskCount, opts.useHalf, compression);

  if (compression.activeMask && raw.length !== numValues) {
    const expanded = new Float32Array(numValues);
    let readIdx = 0;
    for (let i = 0; i < numValues; i++) {
      if (testBit(valueMask, i)) {
        expanded[i] = raw[readIdx++]!;
      } else if (selectionMask && testBit(selectionMask, i)) {
        expanded[i] = inactiveVal1;
      } else {
        expanded[i] = inactiveVal0;
      }
    }
    return expanded;
  }

  return raw;
}
