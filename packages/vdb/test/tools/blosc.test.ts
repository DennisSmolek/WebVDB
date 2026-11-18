/**
 * Blosc1 decoding for OpenVDB node values (`bunny_cloud.vdb` is Blosc/LZ4).
 *
 * The LZ4 vector below was produced by the reference `lz4` CLI (v1.x, `-9`),
 * not by this code: 256 float32 values `(i % 37) * 0.25 - (i % 5)`, byte-
 * shuffled with typesize 4 the way Blosc does, compressed, and the single raw
 * block lifted out of the LZ4 frame. Decoding it through a Blosc frame and
 * getting the floats back exactly checks LZ4 sequences, overlapping matches and
 * the unshuffle together against an independent encoder.
 */

import { describe, expect, it } from "vitest";

import { decompressBlosc, decompressLz4Block, unshuffle } from "../../src/tools/vdb/blosc.js";
import { ByteReader } from "../../src/tools/vdb/byte-reader.js";
import { readCompressedValues } from "../../src/tools/vdb/compression.js";
import { VdbFormatError, VdbUnsupportedError } from "../../src/tools/vdb/errors.js";

const LZ4_VECTOR_HEX =
  "1f000100ffeef01940c01040a0008080e020e0808000704010c040a088603000c8b0988050f0d8c0a8900c0000306040290010002900105029001090290010b8290010e02900b004f0d8c0808000a00030c029001530290010a8290010d0290010f82900511080e02050290010102900106029001098290010c0290010e829005508f8e04070900010a090001000900010a090001020900020700029000fb90034fa0600bfbfc0c03f3fbebfbf403f3f3ebf4040403f3f400100604141c0c0c03f290010402900002200103e18000728006241404040c03e4b002100bf23001fbe4b0002012300123e4b0016006e000a4b0020c0c0900010c09000014b001f006e00001f40b9002f504040404041";

const N = 256;
const EXPECTED = Float32Array.from({ length: N }, (_, i) => (i % 37) * 0.25 - (i % 5));

function hex(s: string): Uint8Array {
  return Uint8Array.from(s.match(/../g)!.map((b) => parseInt(b, 16)));
}

function shuffle(src: Uint8Array, typesize: number): Uint8Array {
  const count = src.length / typesize;
  const out = new Uint8Array(src.length);
  for (let j = 0; j < count; j++) for (let i = 0; i < typesize; i++) out[i * count + j] = src[j * typesize + i]!;
  return out;
}

/** Builds a Blosc1 frame: header, one block start, then the given streams (each size-prefixed). */
function bloscFrame(flags: number, typesize: number, nbytes: number, streams: Uint8Array[]): Uint8Array {
  const body = streams.reduce((n, s) => n + 4 + s.length, 0);
  const frame = new Uint8Array(16 + 4 + body);
  const dv = new DataView(frame.buffer);
  frame[0] = 2; // format version
  frame[1] = 1; // codec format version
  frame[2] = flags;
  frame[3] = typesize;
  dv.setUint32(4, nbytes, true);
  dv.setUint32(8, nbytes, true); // one block
  dv.setUint32(12, frame.length, true);
  dv.setInt32(16, 20, true); // block start
  let p = 20;
  for (const s of streams) {
    dv.setInt32(p, s.length, true);
    frame.set(s, p + 4);
    p += 4 + s.length;
  }
  return frame;
}

const FLAG_SHUFFLE = 0x01;
const FLAG_MEMCPYED = 0x02;
const FLAG_DONT_SPLIT = 0x10;
const LZ4 = 1 << 5;

const asFloats = (bytes: Uint8Array): Float32Array =>
  new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

describe("Blosc1 decoder", () => {
  it("decodes an LZ4 block from the reference encoder", () => {
    const out = new Uint8Array(N * 4);
    expect(decompressLz4Block(hex(LZ4_VECTOR_HEX), out)).toBe(N * 4);
    const plain = new Uint8Array(N * 4);
    unshuffle(out, plain, 4);
    expect(asFloats(plain)).toEqual(EXPECTED);
  });

  it("decodes a shuffled, unsplit LZ4 frame (what OpenVDB writes)", () => {
    const frame = bloscFrame(FLAG_SHUFFLE | FLAG_DONT_SPLIT | LZ4, 4, N * 4, [hex(LZ4_VECTOR_HEX)]);
    expect(asFloats(decompressBlosc(frame))).toEqual(EXPECTED);
  });

  it("decodes a split frame whose streams are stored raw", () => {
    const shuffled = shuffle(new Uint8Array(EXPECTED.buffer), 4);
    const q = shuffled.length / 4;
    const streams = [0, 1, 2, 3].map((s) => shuffled.slice(s * q, (s + 1) * q));
    const frame = bloscFrame(FLAG_SHUFFLE | LZ4, 4, N * 4, streams);
    expect(asFloats(decompressBlosc(frame))).toEqual(EXPECTED);
  });

  it("decodes a memcpy'd frame", () => {
    const frame = new Uint8Array(16 + 8);
    const dv = new DataView(frame.buffer);
    frame[0] = 2;
    frame[2] = FLAG_MEMCPYED;
    frame[3] = 4;
    dv.setUint32(4, 8, true);
    dv.setUint32(8, 8, true);
    dv.setUint32(12, frame.length, true);
    dv.setFloat32(16, 1.5, true);
    dv.setFloat32(20, -2, true);
    expect(Array.from(asFloats(decompressBlosc(frame)))).toEqual([1.5, -2]);
  });

  it("decodes an LZ4 match that overlaps its own output", () => {
    // "abc", then copy 9 bytes from offset 3, then the literal "X".
    const block = Uint8Array.from([0x35, 0x61, 0x62, 0x63, 3, 0, 0x10, 0x58]);
    const out = new Uint8Array(13);
    expect(decompressLz4Block(block, out)).toBe(13);
    expect(new TextDecoder().decode(out)).toBe("abcabcabcabcX");
  });

  it("rejects other codecs, bit-shuffle and corrupt input clearly", () => {
    const zstd = bloscFrame(FLAG_SHUFFLE | (4 << 5), 4, 16, [new Uint8Array(3)]);
    expect(() => decompressBlosc(zstd)).toThrow(VdbUnsupportedError);
    expect(() => decompressBlosc(zstd)).toThrow(/Zstd/);
    const bitshuffle = bloscFrame(0x04 | LZ4, 4, 16, [new Uint8Array(3)]);
    expect(() => decompressBlosc(bitshuffle)).toThrow(/bit-shuffle/);
    expect(() => decompressBlosc(new Uint8Array(8))).toThrow(VdbFormatError);
    // A match offset pointing before the start of the output.
    expect(() => decompressLz4Block(Uint8Array.from([0x10, 0x61, 9, 0, 0x00]), new Uint8Array(16))).toThrow(
      /offset/,
    );
  });
});

describe("readCompressedValues with blosc", () => {
  function stream(meta: number, frame: Uint8Array): ArrayBuffer {
    const bytes = new Uint8Array(1 + 8 + frame.length);
    bytes[0] = meta;
    new DataView(bytes.buffer).setBigInt64(1, BigInt(frame.length), true);
    bytes.set(frame, 9);
    return bytes.buffer;
  }

  it("expands mask-compressed active values decoded from a Blosc frame", () => {
    // 256 active values out of a 512-slot leaf: every other slot is active.
    const valueMask = new Uint32Array(16).fill(0x55555555);
    const frame = bloscFrame(FLAG_SHUFFLE | FLAG_DONT_SPLIT | LZ4, 4, N * 4, [hex(LZ4_VECTOR_HEX)]);
    const values = readCompressedValues(new ByteReader(stream(0, frame)), {
      fileVersion: 223,
      compression: { zip: false, activeMask: true, blosc: true },
      useHalf: false,
      background: 0,
      numValues: 512,
      valueMask,
    });
    for (let i = 0; i < 512; i++) expect(values[i]).toBe(i % 2 === 0 ? EXPECTED[i / 2] : 0);
  });

  it("ignores OpenVDB's zero padding of buffers under Blosc's 48-byte minimum", () => {
    // Two floats (8 bytes) padded to 48 bytes before compression.
    const padded = new Uint8Array(48);
    new DataView(padded.buffer).setFloat32(0, 0.5, true);
    new DataView(padded.buffer).setFloat32(4, 0.75, true);
    const frame = new Uint8Array(16 + 48);
    const dv = new DataView(frame.buffer);
    frame[0] = 2;
    frame[2] = FLAG_MEMCPYED;
    frame[3] = 4;
    dv.setUint32(4, 48, true);
    dv.setUint32(8, 48, true);
    dv.setUint32(12, frame.length, true);
    frame.set(padded, 16);
    const values = readCompressedValues(new ByteReader(stream(6, frame)), {
      fileVersion: 223,
      compression: { zip: false, activeMask: true, blosc: true },
      useHalf: false,
      background: 0,
      numValues: 2,
      valueMask: new Uint32Array(1).fill(3),
    });
    expect(Array.from(values)).toEqual([0.5, 0.75]);
  });
});
