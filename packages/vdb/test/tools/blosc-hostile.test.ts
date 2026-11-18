/**
 * Hostile-input hardening for the pure-TS Blosc / zlib decompression path.
 *
 * Every crafted input must fail with a clean `VdbFormatError`, in well under
 * 200 ms, without ever allocating a typed array larger than the caller's
 * expected output (`maxBytes`). The real-frame vectors at the bottom were
 * produced by an independent encoder (python-blosc 1.21, i.e. c-blosc 1.21.x)
 * so the split rule is checked against upstream, not against this decoder.
 */

import { deflateSync } from "node:zlib";

import { describe, expect, it } from "vitest";

import { decompressBlosc, decompressLz4Block } from "../../src/tools/vdb/blosc.js";
import { ByteReader } from "../../src/tools/vdb/byte-reader.js";
import { readCompressedValues } from "../../src/tools/vdb/compression.js";
import { VdbFormatError } from "../../src/tools/vdb/errors.js";
import { inflateRaw, inflateZlib } from "../../src/tools/vdb/inflate.js";

const LZ4 = 1 << 5;
const ZLIB = 3 << 5;
const DONT_SPLIT = 0x10;
const SHUFFLE = 0x01;
const MAX = 4096; // the "expected output" cap used by most cases
const TIME_BUDGET_MS = 200;

function hex(s: string): Uint8Array {
  return Uint8Array.from(s.match(/../g)!.map((b) => parseInt(b, 16)));
}

/** Header-only frame builder with `extra` zero bytes after the 16-byte header. */
function hdr(flags: number, typesize: number, nbytes: number, blocksize: number, cbytes: number, extra = 0): Uint8Array {
  const b = new Uint8Array(16 + extra);
  const v = new DataView(b.buffer);
  b[0] = 2;
  b[1] = 1;
  b[2] = flags;
  b[3] = typesize;
  v.setUint32(4, nbytes, true);
  v.setUint32(8, blocksize, true);
  v.setUint32(12, cbytes, true);
  return b;
}

/** Runs `fn`, recording the largest numeric-length typed-array allocation made inside it. */
function withAllocSpy<T>(fn: () => T): { result: T | undefined; error: unknown; maxAlloc: number; ms: number } {
  const Real = globalThis.Uint8Array;
  let maxAlloc = 0;
  class Spy extends Real {
    constructor(...args: unknown[]) {
      if (typeof args[0] === "number" && args[0] > maxAlloc) maxAlloc = args[0];
      // @ts-expect-error spread into the overloaded base constructor
      super(...args);
    }
  }
  globalThis.Uint8Array = Spy as unknown as Uint8ArrayConstructor;
  const t = performance.now();
  let result: T | undefined;
  let error: unknown;
  try {
    result = fn();
  } catch (e) {
    error = e;
  } finally {
    globalThis.Uint8Array = Real;
  }
  return { result, error, maxAlloc, ms: performance.now() - t };
}

function expectRejected(fn: () => unknown, cap: number, message?: RegExp): void {
  const { error, maxAlloc, ms } = withAllocSpy(fn);
  expect(error).toBeInstanceOf(VdbFormatError);
  if (message) expect((error as Error).message).toMatch(message);
  expect(ms).toBeLessThan(TIME_BUDGET_MS);
  expect(maxAlloc).toBeLessThanOrEqual(cap);
}

describe("decompressBlosc rejects hostile frames", () => {
  const limits = { maxBytes: MAX };

  it("nbytes far beyond the expected output (S1)", () => {
    expectRejected(() => decompressBlosc(hdr(LZ4 | DONT_SPLIT, 4, 0x7ffffff0, 0x7ffffff0, 16), limits), MAX, /nbytes/);
    expectRejected(() => decompressBlosc(hdr(LZ4 | DONT_SPLIT, 4, 0xfffffff0, 0xfffffff0, 16), limits), MAX, /nbytes/);
    expectRejected(() => decompressBlosc(hdr(0x02, 4, 0xfffffff0, 16, 16), limits), MAX, /nbytes/);
    expectRejected(() => decompressBlosc(hdr(LZ4 | DONT_SPLIT, 4, 1 << 30, 1, 16), limits), MAX, /nbytes/);
  });

  it("the 1 GiB amplification frame (one stream shared by every block start)", () => {
    const bs = 65536;
    const nbytes = 1 << 30;
    const nblocks = nbytes / bs;
    const tableEnd = 16 + 4 * nblocks;
    const stream = Uint8Array.from([0x1f, 0, 1, 0, 255, 255, 255, 0x50, 0, 0, 0, 0, 0]);
    const frame = new Uint8Array(tableEnd + 4 + stream.length);
    const v = new DataView(frame.buffer);
    frame.set(hdr(LZ4 | DONT_SPLIT | SHUFFLE, 4, nbytes, bs, frame.length));
    for (let b = 0; b < nblocks; b++) v.setInt32(16 + 4 * b, tableEnd, true);
    v.setInt32(tableEnd, stream.length, true);
    frame.set(stream, tableEnd + 4);
    expectRejected(() => decompressBlosc(frame, { maxBytes: 1 << 20 }), 1 << 20, /nbytes/);
  });

  it("blocksize larger than the output, which used to size a huge scratch buffer (S2)", () => {
    const f = hdr(LZ4 | SHUFFLE, 4, 64, 1 << 30, 0, 8 + 64);
    const v = new DataView(f.buffer);
    v.setUint32(12, f.length, true);
    v.setInt32(16, 20, true);
    v.setInt32(20, 64, true);
    expectRejected(() => decompressBlosc(f, { maxBytes: 64 }), 64, /blocksize/);
    // Within maxBytes but larger than nbytes: still no allocation beyond the output.
    const g = hdr(LZ4 | SHUFFLE, 4, 64, 4096, 0, 8 + 64);
    const gv = new DataView(g.buffer);
    gv.setUint32(12, g.length, true);
    gv.setInt32(16, 20, true);
    gv.setInt32(20, 64, true);
    const { error, maxAlloc } = withAllocSpy(() => decompressBlosc(g, { maxBytes: 4096 }));
    expect(error).toBeUndefined();
    expect(maxAlloc).toBeLessThanOrEqual(64);
  });

  it("blocksize zero or beyond maxBytes", () => {
    expectRejected(() => decompressBlosc(hdr(LZ4, 4, 64, 0, 24, 8), limits), MAX, /blocksize/);
    expectRejected(() => decompressBlosc(hdr(LZ4, 4, 64, 0xffffffff, 24, 8), limits), MAX, /blocksize/);
  });

  it("block-start table larger than the frame (nblocks * 4 > cbytes - 16)", () => {
    expectRejected(() => decompressBlosc(hdr(LZ4 | DONT_SPLIT, 4, 4096, 1, 16 + 8, 8), limits), MAX, /block-start table/);
  });

  it("out-of-range block starts (S3)", () => {
    const frame = (bstart: number) => {
      const f = hdr(LZ4 | DONT_SPLIT, 4, 64, 64, 0, 4);
      const v = new DataView(f.buffer);
      v.setUint32(12, f.length, true);
      v.setInt32(16, bstart, true);
      return f;
    };
    for (const bstart of [-100, 0x7fffffff, 0, 16, 19, 20]) {
      expectRejected(() => decompressBlosc(frame(bstart), limits), MAX, /block-start\[0\]/);
    }
  });

  it("stream sizes past cbytes, even when the buffer holds more bytes than cbytes declares", () => {
    const f = hdr(LZ4 | DONT_SPLIT, 4, 64, 64, 24, 8);
    const v = new DataView(f.buffer);
    v.setInt32(16, 20, true);
    v.setInt32(20, 0x7ffffffc, true);
    expectRejected(() => decompressBlosc(f, limits), MAX, /stream size/);
    v.setInt32(20, -1, true);
    expectRejected(() => decompressBlosc(f, limits), MAX, /stream size/);
    // A stream that fits the buffer but not the declared cbytes (frame has 64 trailing bytes).
    const g = hdr(LZ4 | DONT_SPLIT, 4, 64, 64, 24, 8 + 64);
    const gv = new DataView(g.buffer);
    gv.setInt32(16, 20, true);
    gv.setInt32(20, 32, true);
    expectRejected(() => decompressBlosc(g, limits), MAX, /stream size/);
  });

  it("typesize zero, flags from the future, header/cbytes inconsistencies", () => {
    const f = hdr(LZ4 | SHUFFLE, 0, 64, 64, 0, 8 + 64);
    new DataView(f.buffer).setUint32(12, f.length, true);
    expectRejected(() => decompressBlosc(f, limits), MAX, /typesize/);
    expectRejected(() => decompressBlosc(hdr(0x02, 4, 8, 8, 20, 8), limits), MAX, /memcpy/); // cbytes != nbytes + 16
    expectRejected(() => decompressBlosc(hdr(0x02, 4, 8, 8, 40, 8), limits), MAX, /cbytes/); // cbytes beyond frame
    expectRejected(() => decompressBlosc(hdr(LZ4, 4, 8, 8, 4, 8), limits), MAX, /cbytes/); // cbytes < header
  });

  it("a frame whose cbytes differs from the byte count OpenVDB stored", () => {
    const f = hdr(0x02, 4, 8, 8, 24, 8);
    expectRejected(() => decompressBlosc(f, { maxBytes: MAX, expectedCbytes: 24 + 2 }), MAX, /does not match/);
    expect(decompressBlosc(f, { maxBytes: MAX, expectedCbytes: 24 }).length).toBe(8);
  });

  it("split block whose size does not divide into its streams", () => {
    // 4098 bytes, typesize 4, 1024 elements >= 128 and not flagged dont-split => 4 splits of 1024.5
    const f = hdr(LZ4, 4, 4098, 4098, 0, 8);
    new DataView(f.buffer).setUint32(12, f.length, true);
    new DataView(f.buffer).setInt32(16, 20, true);
    expectRejected(() => decompressBlosc(f, { maxBytes: 8192 }), 8192, /splits/);
  });

  it("zlib bomb inside a Blosc stream (S4)", () => {
    const bomb = deflateSync(Buffer.alloc(64 << 20), { level: 9 });
    const f = hdr(ZLIB | DONT_SPLIT, 4, 64, 64, 0, 8 + bomb.length);
    const v = new DataView(f.buffer);
    v.setUint32(12, f.length, true);
    v.setInt32(16, 20, true);
    v.setInt32(20, bomb.length, true);
    f.set(bomb, 24);
    expectRejected(() => decompressBlosc(f, { maxBytes: 64 }), 1 << 16, /exceeds/);
  });
});

describe("inflate output cap (S4)", () => {
  const bomb = deflateSync(Buffer.alloc(64 << 20), { level: 9 });

  it("inflateRaw and inflateZlib stop as soon as the cap is hit", () => {
    expectRejected(() => inflateZlib(new Uint8Array(bomb), 1000), 1 << 16, /exceeds/);
    const raw = new Uint8Array(bomb).subarray(2, bomb.length - 4);
    expectRejected(() => inflateRaw(raw, 1000), 1 << 16, /exceeds/);
  });

  it("an exact-size stream still inflates", () => {
    const data = new Uint8Array(1000).map((_, i) => i * 31);
    const z = new Uint8Array(deflateSync(data));
    expect(inflateZlib(z, 1000)).toEqual(data);
    expectRejected(() => inflateZlib(z, 999), 1 << 16, /exceeds/);
  });

  it("a code-length repeat that overruns the table is rejected", () => {
    // BFINAL=1, BTYPE=2, HLIT=0 (257), HDIST=0 (1), HCLEN=0 (4): code-length codes for 16,17,18,0 get
    // length 1,2,2,0 => symbol 16 = '0', 17 = '10', 18 = '11'. Then 3 x repeat-zero-138 overruns 258.
    const bits: number[] = [];
    const push = (v: number, n: number) => {
      for (let i = 0; i < n; i++) bits.push((v >> i) & 1);
    };
    push(1, 1);
    push(2, 2);
    push(0, 5);
    push(0, 5);
    push(0, 4);
    push(2, 3); // len(16)=2
    push(2, 3); // len(17)=2
    push(1, 3); // len(18)=1
    push(0, 3); // len(0)=0
    // canonical codes: 18 -> '0' (len 1), 16 -> '10', 17 -> '11'
    for (let k = 0; k < 3; k++) {
      push(0, 1); // sym 18
      push(127, 7); // repeat 138
    }
    while (bits.length % 8) bits.push(0);
    const bytes = new Uint8Array(bits.length / 8);
    bits.forEach((b, i) => (bytes[i >> 3]! |= b << (i & 7)));
    expectRejected(() => inflateRaw(bytes, 1000), 1 << 16, /overruns|too many/);
  });
});

describe("LZ4 block hostile cases", () => {
  it("offset before start, zero offset, overlong match, truncated or huge literal chains", () => {
    const out = () => new Uint8Array(32);
    expectRejected(() => decompressLz4Block(Uint8Array.from([0x10, 0x41, 0x05, 0x00, 0x00]), out()), 2048, /offset/);
    expectRejected(() => decompressLz4Block(Uint8Array.from([0x10, 0x41, 0x00, 0x00, 0x00]), out()), 2048, /offset/);
    expectRejected(() => decompressLz4Block(Uint8Array.from([0x1f, 0x41, 0x01, 0x00, 255, 255, 255, 10]), out()), 2048);
    expectRejected(() => decompressLz4Block(Uint8Array.from([0xf0, 255, 255]), out()), 2048);
    expectRejected(() => decompressLz4Block(Uint8Array.from([0xf0, ...new Array(1000).fill(255), 0]), out()), 2048);
  });
});

describe("readCompressedValues hostile payloads (S1, S4, N7)", () => {
  const maskAll = new Uint32Array(16).fill(0xffffffff);
  const opts = (blosc: boolean, zip: boolean) => ({
    fileVersion: 223,
    compression: { zip, activeMask: false, blosc },
    useHalf: false,
    background: 0,
    numValues: 512,
    valueMask: maskAll,
  });
  /** meta byte, i64 length field, then the payload. */
  function stream(field: bigint, payload: Uint8Array): ArrayBuffer {
    const bytes = new Uint8Array(1 + 8 + payload.length);
    new DataView(bytes.buffer).setBigInt64(1, field, true);
    bytes.set(payload, 9);
    return bytes.buffer;
  }
  const read = (field: bigint, payload: Uint8Array, blosc = true) =>
    readCompressedValues(new ByteReader(stream(field, payload)), opts(blosc, !blosc));

  it("blosc frame whose nbytes exceeds the node's expected size", () => {
    const f = hdr(LZ4 | DONT_SPLIT, 4, 1 << 30, 1 << 30, 16);
    expectRejected(() => read(BigInt(f.length), f), 1 << 20, /nbytes/);
  });

  it("blosc frame whose cbytes differs from the stored byte count", () => {
    const f = hdr(0x02, 4, 2048, 2048, 16 + 2048, 2048 + 8);
    expectRejected(() => read(BigInt(f.length), f), 1 << 20, /does not match/);
  });

  it("zip payload that inflates past the node's size", () => {
    const bomb = new Uint8Array(deflateSync(Buffer.alloc(64 << 20), { level: 9 }));
    expectRejected(() => read(BigInt(bomb.length), bomb, false), 1 << 20, /exceeds/);
  });

  it("zip payload that inflates to fewer bytes than the node needs", () => {
    const short = new Uint8Array(deflateSync(Buffer.alloc(100)));
    expectRejected(() => read(BigInt(short.length), short, false), 1 << 20, /expected/);
  });

  it("raw-value sentinel that is not a whole count within the node (N7)", () => {
    for (const field of [-3n, -2049n, -(2048n + 4n), -(1n << 40n), -2n]) {
      expectRejected(() => read(field, new Uint8Array(4096)), 1 << 20, /sentinel/);
      expectRejected(() => read(field, new Uint8Array(4096), false), 1 << 20, /sentinel/);
    }
  });

  it("a valid sentinel still reads raw values; a short one is rejected", () => {
    const raw = new Uint8Array(2048);
    new DataView(raw.buffer).setFloat32(0, 1.25, true);
    expect(read(-2048n, raw)[0]).toBe(1.25);
    expectRejected(() => read(0n, raw), 1 << 20, /expected 512/);
  });

  it("a compressed-size field larger than the buffer", () => {
    expectRejected(() => read(1n << 40n, new Uint8Array(8)), 1 << 20, /end of buffer/);
  });
});

describe("split decision follows c-blosc blosc_d (S5)", () => {
  const asBytes = (n: number, f: (i: number) => number) => Uint8Array.from({ length: n }, (_, i) => f(i));

  it("flag-less frame of 20 floats is not split (80/4 < 128), as pre-1.14 encoders wrote it", () => {
    const raw = asBytes(80, (i) => (i * 7) & 0xff);
    const f = hdr(LZ4, 4, 80, 80, 0, 8 + 80);
    const v = new DataView(f.buffer);
    v.setUint32(12, f.length, true);
    v.setInt32(16, 20, true);
    v.setInt32(20, 80, true);
    f.set(raw, 24);
    expect(decompressBlosc(f, { maxBytes: 80 })).toEqual(raw);
  });

  it("flag-less shuffled frame of 20 floats unshuffles correctly", () => {
    const values = Float32Array.from({ length: 20 }, (_, i) => i * 0.5 - 3);
    const plain = new Uint8Array(values.buffer);
    const shuffled = new Uint8Array(80);
    for (let j = 0; j < 20; j++) for (let b = 0; b < 4; b++) shuffled[b * 20 + j] = plain[j * 4 + b]!;
    const f = hdr(LZ4 | SHUFFLE, 4, 80, 80, 0, 8 + 80);
    const v = new DataView(f.buffer);
    v.setUint32(12, f.length, true);
    v.setInt32(16, 20, true);
    v.setInt32(20, 80, true);
    f.set(shuffled, 24);
    expect(decompressBlosc(f)).toEqual(plain);
  });

  it("flag-less typesize-24 frame is not split (typesize > MAX_SPLITS)", () => {
    const n = 24 * 512;
    const raw = asBytes(n, (i) => i & 0xff);
    const f = hdr(LZ4, 24, n, n, 0, 8 + n);
    const v = new DataView(f.buffer);
    v.setUint32(12, f.length, true);
    v.setInt32(16, 20, true);
    v.setInt32(20, n, true);
    f.set(raw, 24);
    expect(decompressBlosc(f, { maxBytes: n })).toEqual(raw);
  });

  // Real frames from python-blosc 1.21 (c-blosc 1.21.x), an independent encoder.
  it("decodes a c-blosc encoded 20-float frame (memcpy'd)", () => {
    const expected = Float32Array.from({ length: 20 }, (_, i) => Math.fround((i % 7) * 0.5 - 1 + i * 0.01));
    const out = decompressBlosc(hex("02013304500000005000000060000000000080bf48e1fabe0ad7a33c14ae073fb81e853f6666c63f0ad703407b146ebf3d0ad7beec51b83d9a99193f7b148e3f295ccf3fec510840f6285cbf3333b3be0ad7233e1f852b3f3d0a973fec51d83f"), { maxBytes: 80 });
    expect(new Float32Array(out.buffer, out.byteOffset, 20)).toEqual(expected);
  });

  it("decodes a c-blosc encoded split LZ4 frame (typesize 4, flags 0x21, 4 streams)", () => {
    const expected = Float32Array.from({ length: 4096 }, (_, i) => (i % 37) * 0.25 - (i % 5));
    const frame = hex("0201210400400000004000008e010000140000001a0000001f000100fffffffffffffffffffffffffffffff65000000000001a0000001f000100fffffffffffffffffffffffffffffff6500000000000b5000000f01a0040c01040a0008080e020e0808000704010c040a088603000c8b0988050f0d8c0a8900c0000306040290010002900105029001090290010b8290010e02900b004f0d8c0808000a00030c029001530290010a8290010d0290010f82900511080e02050290010102900106029006098805020e0c0290010e829005508f8e04070900010a090001000900010a090001020900020700029000090000fb900ffffffffffffffffffffffffffffff3a5088603000c881000000fa0600bfbfc0c03f3fbebfbf403f3f3ebf4040403f3f400100604141c0c0c03f290010402900002200103e22000728006241404040c03e4b002100bf23001fbe4b0002012300002900104029000523000a4b0020c0c0900010c09000014b001f006e0000104090000fb900ffffffffffffffffffffffffffffff3a504040404040");
    expect(frame[2]! & DONT_SPLIT).toBe(0);
    const out = decompressBlosc(frame, { maxBytes: 16384, expectedCbytes: frame.length });
    expect(new Float32Array(out.buffer, out.byteOffset, 4096)).toEqual(expected);
  });

  it("decodes a c-blosc encoded split zlib frame", () => {
    const expected = Float32Array.from({ length: 4096 }, (_, i) => (i % 37) * 0.25 - (i % 5));
    const out = decompressBlosc(hex("0201610400400000004000005c010000140000001a00000078daedc1010d000000c2a0f74f6d0f0714000000f06e100000011a00000078daedc1010d000000c2a0f74f6d0f0714000000f06e100000019c00000078daedcea119c2301884e11308444546e80819e11f212364848cf04b6465656425b23212894446229195e410d50c70febbe75e580b56e1dee7ee8e62a159bde588c77df5f479b56d9980980d2304c3c47061b833ec0c2f231ceb8ad818c633dc183e191e0cc3b849bf3030cc0c57868de19be1f5e8564e4fa507f4547a667a0a184270c105175c70c105175c70c105175c70c105ff17fe0512d4146a6800000078daedce410ac0200c05d19c2c891737395a135db4828b1e60842294e1fb2422d36c46b899867b5d7e9e31b212a9d03bec42fbffa71bf5a5eea90a3b99d7a995e89e92f7b573aa92f2647bd6945c5e73010e1c3870e0c08103070e1c3870e0c081ff853f51979b13"), { maxBytes: 16384 });
    expect(new Float32Array(out.buffer, out.byteOffset, 4096)).toEqual(expected);
  });

  it("decodes a c-blosc encoded typesize-24 frame, with and without the dont-split flag", () => {
    const expected = Uint8Array.from({ length: 24 * 600 }, (_, i) => (i * 7) % 251);
    const frame = hex("0201311840380000403800001a0200001400000002020000ffec00a85502aa5704ac5906ae5b08b05d0ab25f0cb4610eb66310b86512ba6714bc6916be6b18c06d1ac26f1cc4711ec67320c87522ca7724cc7926ce7b28d07d2ad27f2cd4812ed68330d88532da8734dc8936de8b38e08d3ae28f3ce4913ee69340e89542ea9744ec9946ee9b48f09d4af29f4cf4a14ef6a350f8a552faa75401a95603ab5805ad5a07af5c09b15e0bb3600db5620fb76411b96613bb6815bd6a17bf6c19c16e1bc3701dc5721fc77421c97623cb7825cd7a27cf7c29d17e2bd3802dd5822fd78431d98633db8835dd8a37df8c39e18e3be3903de5923fe79441e99643eb9845ed9a47ef9c49f19e4bf3a04df5a24ff7a451f9a653fb00ff4b0fd001ffbe0fc603750f9b04ffff320fcb02020f6b06ffa90fd001020f6108750f3609ffff1d0fcb02170f060bff940fd001170ffc0c750fd10dffff080fcb022c0fa10fff7f0fd0012c0f9711750f6c12fff20fcb02410f3c14ff6a0fd001410f3216750f0717ffdd0fcb02560fd718ff550fd001560fcd1a750fa21bffc80fcb026b0f6d1effff3c07cb020f3d20ffb307d0010f3322750f0823ffff270fcb020d0fd824ff9e0fd0010d0fce26750fa327ffff120fcb02220f7329ff890fd001220f692b750f3e2cfffc0fcb02370f0e2eff740fd001370f0430750fd500c20fd431ff710fa932ff5f0f9b04bf0fc603020f7435ffd20fcb025c50df8c39e18e");
    expect(frame[2]! & DONT_SPLIT).toBe(DONT_SPLIT);
    expect(decompressBlosc(frame, { maxBytes: 24 * 600 })).toEqual(expected);
    // The same frame as a pre-1.14 encoder wrote it: no flag, same layout, because 24 > MAX_SPLITS.
    const legacy = frame.slice();
    legacy[2] = legacy[2]! & ~DONT_SPLIT;
    expect(decompressBlosc(legacy, { maxBytes: 24 * 600 })).toEqual(expected);
  });
});
