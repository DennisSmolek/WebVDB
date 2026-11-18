/**
 * byte-reader.ts — a seekable little-endian cursor over a `.vdb` buffer.
 *
 * All primitive VDB stream fields (per the `OpenVDBReader`/`vdb-rs` format
 * facts, docs/handoffs and DECISIONS.md D3) are little-endian; strings are a
 * u32 length prefix + raw ASCII bytes (no terminator).
 */

import { VdbFormatError } from "./errors.js";

export class ByteReader {
  private readonly view: DataView;
  private readonly bytes: Uint8Array;
  pos = 0;

  constructor(buffer: ArrayBuffer) {
    this.bytes = new Uint8Array(buffer);
    this.view = new DataView(buffer);
  }

  get length(): number {
    return this.bytes.length;
  }

  private need(count: number): void {
    if (this.pos + count > this.bytes.length) {
      throw new VdbFormatError(
        `unexpected end of buffer (wanted ${count} bytes at offset ${this.pos}, have ${this.bytes.length})`,
      );
    }
  }

  seek(pos: number): void {
    if (pos < 0 || pos > this.bytes.length) {
      throw new VdbFormatError(`seek out of range: ${pos} (buffer length ${this.bytes.length})`);
    }
    this.pos = pos;
  }

  skip(count: number): void {
    this.seek(this.pos + count);
  }

  u8(): number {
    this.need(1);
    return this.bytes[this.pos++]!;
  }

  bool(): boolean {
    return this.u8() !== 0;
  }

  i32(): number {
    this.need(4);
    const v = this.view.getInt32(this.pos, true);
    this.pos += 4;
    return v;
  }

  u32(): number {
    this.need(4);
    const v = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }

  u64(): bigint {
    this.need(8);
    const v = this.view.getBigUint64(this.pos, true);
    this.pos += 8;
    return v;
  }

  /** u64 narrowed to `number` — safe for stream offsets/counts on real files
   *  (well under 2^53), throws rather than silently truncating otherwise. */
  u64AsNumber(): number {
    const v = this.u64();
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new VdbFormatError(`u64 value ${v} exceeds safe integer range`);
    }
    return Number(v);
  }

  f32(): number {
    this.need(4);
    const v = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return v;
  }

  f64(): number {
    this.need(8);
    const v = this.view.getFloat64(this.pos, true);
    this.pos += 8;
    return v;
  }

  f16AsF32(halfToFloat: (h: number) => number): number {
    this.need(2);
    const h = this.view.getUint16(this.pos, true);
    this.pos += 2;
    return halfToFloat(h);
  }

  bytes_(count: number): Uint8Array {
    this.need(count);
    const out = this.bytes.subarray(this.pos, this.pos + count);
    this.pos += count;
    return out;
  }

  /** u32 length prefix + that many raw ASCII bytes (VDB's `readString`). */
  string(): string {
    const len = this.u32();
    const raw = this.bytes_(len);
    let s = "";
    for (let i = 0; i < raw.length; i++) s += String.fromCharCode(raw[i]!);
    return s;
  }

  /** Fixed-length ASCII string (no length prefix) — used for the UUID. */
  fixedString(len: number): string {
    const raw = this.bytes_(len);
    let s = "";
    for (let i = 0; i < raw.length; i++) s += String.fromCharCode(raw[i]!);
    return s;
  }

  vec3i(): [number, number, number] {
    return [this.i32(), this.i32(), this.i32()];
  }

  vec3d(): [number, number, number] {
    return [this.f64(), this.f64(), this.f64()];
  }

  /** Reads `bitCount` mask bits as `bitCount/32` little-endian u32 words.
   *  (On disk these are u64 LE words; splitting each into two u32 LE words
   *  preserves bit-for-bit ordinal position — see bit-utils.ts.) */
  maskWords(bitCount: number): Uint32Array {
    if (bitCount % 32 !== 0) {
      throw new VdbFormatError(`mask bit count ${bitCount} is not a multiple of 32`);
    }
    const words = new Uint32Array(bitCount / 32);
    for (let i = 0; i < words.length; i++) words[i] = this.u32();
    return words;
  }
}
