/**
 * metadata.ts — the generic `(name, type, byteLength, value)` metadata map
 * used both at file scope (`OpenVDBReader.readHeader`) and per-grid scope
 * (`GridDescriptor.readMetadata`) — same on-disk encoding both times, per
 * both format references.
 */

import type { ByteReader } from "./byte-reader.js";

export type MetadataValue =
  | string
  | boolean
  | number
  | bigint
  | [number, number, number]
  | Uint8Array;

function readMetadataValue(reader: ByteReader, type: string, byteLength: number): MetadataValue {
  const start = reader.pos;
  let value: MetadataValue;
  switch (type) {
    case "string":
      value = reader.fixedString(byteLength);
      break;
    case "bool":
      value = reader.bool();
      break;
    case "int32":
      value = reader.i32();
      break;
    case "int64":
      value = reader.u64(); // stored/consumed as bigint; real files use plain magnitudes
      break;
    case "float":
      value = reader.f32();
      break;
    case "double":
      value = reader.f64();
      break;
    case "vec3i":
      value = reader.vec3i();
      break;
    case "vec3s":
      value = [reader.f32(), reader.f32(), reader.f32()];
      break;
    case "vec3d":
      value = reader.vec3d();
      break;
    default:
      // Unknown metadata type (e.g. exotic user metadata) — keep the raw
      // bytes rather than failing the whole parse over cosmetic metadata.
      value = reader.bytes_(byteLength).slice();
      break;
  }
  const consumed = reader.pos - start;
  if (consumed !== byteLength) {
    // Some types (e.g. "string") legitimately consume exactly byteLength;
    // fixed-size types should too. If not, trust the declared length so a
    // format quirk here doesn't desync the rest of the stream.
    reader.seek(start + byteLength);
  }
  return value;
}

/** Reads a `count`-prefixed `(name, type, len, value)*` metadata map. */
export function readMetadataMap(reader: ByteReader): Record<string, MetadataValue> {
  const count = reader.u32();
  const map: Record<string, MetadataValue> = {};
  for (let i = 0; i < count; i++) {
    const name = reader.string();
    const type = reader.string();
    const byteLength = reader.u32();
    map[name] = readMetadataValue(reader, type, byteLength);
  }
  return map;
}
