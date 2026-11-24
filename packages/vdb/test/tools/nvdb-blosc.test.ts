/**
 * K7.C: the BloscLZ decoder and `.nvdb` Codec::BLOSC, valid and hostile.
 *
 * The BloscLZ vectors are OpenVDB v14.0.0's own (`nanovdb/nanovdb/unittest/TestBlosc.cc`,
 * Apache-2.0, "Generated with c-blosc 1.21.6, not with the decoder under test": input byte i is
 * i % 32, level 9, compressor blosclz, block size 256, one thread), so the decoder is checked
 * against the upstream encoder. Hostile `.nvdb` files start from the committed native fixture
 * (`fixtures/native-v14/sphere_fog_blosc.nvdb`) or from `writeNvdb` output re-framed as BLOSC with
 * memcpy'd frames (valid Blosc that any chunking can be built from). Every hostile case must fail
 * with a coded `VDBToolsError` in well under 200 ms without allocating past its cap.
 */

import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  NvdbCodec,
  NvdbFile,
  NvdbFormatError,
  NvdbStreamReader,
  VDBToolsError,
  VDBUnsupportedError,
  vdbBuildFromDense,
  writeNvdb,
} from "../../src/tools/internal.js";
import { decompressBlosc, decompressBloscInto } from "../../src/tools/vdb/blosc.js";
import { decompressBloscLzBlock } from "../../src/tools/vdb/blosclz.js";
import { VdbFormatError } from "../../src/tools/vdb/errors.js";

const TIME_BUDGET_MS = 200;

// ---------------------------------------------------------------------------
// OpenVDB v14.0.0 TestBlosc.cc vectors (c-blosc 1.21.6, BloscLZ).
// ---------------------------------------------------------------------------

const hex = (s: string): Uint8Array => Uint8Array.from(s.replace(/\s+/g, "").match(/../g)!.map((b) => parseInt(b, 16)));

/** plain: 1024 bytes, no shuffle, typesize 1. */
const PLAIN = hex(`
  02010001 00040000 00040000 43000000 14000000 2b000000 3f000102 03040506 0708090a 0b0c0d0e 0f101112
  13141516 1718191a 1b1c1d1e 1fe0ffff ffd71f02 1d1e1f`);

/** One shuffled block of the "shuffled" / "partial" vectors (typesize 8, 8 splits of 32 bytes). */
const BLOCK_STREAMS =
  "43000000 23000810 18e01303 03010911 19e01303 03020a12 1ae01303 03030b13 1be01303 03040c14 1ce01303 03050d15 " +
  "1de01303 03060e16 1ee01303 03070f17 1fe01103 01171f";

/** shuffled: 1024 bytes, byte shuffle, typesize 8 (four 256-byte blocks, each split in 8). */
const SHUFFLED = hex(`
  02011108 00040000 00010000 3c010000 20000000 67000000 ae000000 f5000000
  ${BLOCK_STREAMS} ${BLOCK_STREAMS} ${BLOCK_STREAMS} ${BLOCK_STREAMS}`);

/** partial: 1027 bytes, byte shuffle, typesize 8, a 3-byte leftover block stored raw. */
const PARTIAL = hex(`
  02011108 03040000 00010000 47010000 24000000 6b000000 b2000000 f9000000 40010000
  ${BLOCK_STREAMS} ${BLOCK_STREAMS} ${BLOCK_STREAMS} ${BLOCK_STREAMS} 03000000 000102`);

/** TestBlosc RejectBitShuffle: real bit-shuffled frames from c-blosc 1.21.6 (BloscLZ and LZ4). */
const BITSHUFFLE_BLOSCLZ = hex(`
  02010401 80000000 80000000 3c000000 14000000 24000000 23aaaaaa aae00303 00cce006 0000f0e0 06000100
  ffe00601 0100ffe0 0603e023 00010000`);
const BITSHUFFLE_LZ4 = hex(`
  02012401 80000000 80000000 38000000 14000000 20000000 1baa0100 1bcc0100 1bf00100 2b00ff02 002b00ff
  04000f02 00165000 00000000`);

function expectPayload(frame: Uint8Array, n: number): void {
  expect(frame.length).toBe(new DataView(frame.buffer, frame.byteOffset).getUint32(12, true));
  const out = decompressBlosc(frame, { maxBytes: n });
  expect(out.length).toBe(n);
  for (let i = 0; i < n; i++) if (out[i] !== i % 32) expect.fail(`byte ${i} is ${out[i]}, expected ${i % 32}`);
  // Exactly n: one byte less is refused (upstream checkPayload).
  expect(() => decompressBlosc(frame, { maxBytes: n - 1 })).toThrow(VdbFormatError);
}

describe("BloscLZ (TestBlosc.cc vectors)", () => {
  it("ReadBloscLZ: plain, shuffled and partial frames", () => {
    expectPayload(PLAIN, 1024);
    expectPayload(SHUFFLED, 1024);
    expectPayload(PARTIAL, 1027);
  });

  it("decodes into a caller's buffer without touching bytes past nbytes", () => {
    const dest = new Uint8Array(1026).fill(0xa5);
    expect(decompressBloscInto(PLAIN, dest.subarray(1, 1025))).toBe(1024);
    expect(dest[0]).toBe(0xa5);
    expect(dest[1025]).toBe(0xa5);
    expect(dest[1 + 33]).toBe(1);
  });

  it("RejectUnsupportedFormat: Zstd and an unknown BloscLZ format version are unsupported", () => {
    for (const input of [PLAIN, SHUFFLED]) {
      const zstd = input.slice();
      zstd[2] = (zstd[2]! & 0x1f) | (4 << 5);
      expect(() => decompressBlosc(zstd, { maxBytes: 1024 })).toThrow(VDBUnsupportedError);
      const version = input.slice();
      version[1] = 255;
      expect(() => decompressBlosc(version, { maxBytes: 1024 })).toThrow(/BloscLZ stream format version 255/);
    }
  });

  it("RejectInvalidBloscLZBlock: a truncated literal and a match before the output", () => {
    const truncated = Uint8Array.from([2, 1, 0x10, 1, 128, 0, 0, 0, 128, 0, 0, 0, 25, 0, 0, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0x3f]);
    expect(() => decompressBlosc(truncated, { maxBytes: 128 })).toThrow(/literal run runs past the input/);
    const invalidMatch = Uint8Array.from([
      2, 1, 0x10, 1, 128, 0, 0, 0, 128, 0, 0, 0, 29, 0, 0, 0, 20, 0, 0, 0, 5, 0, 0, 0, 0x20, 0x61, 0x20, 0xff, 0,
    ]);
    expect(() => decompressBlosc(invalidMatch, { maxBytes: 128 })).toThrow(/reaches before the start of the output/);
  });

  it("ReadUncompressedBloscLZBuffer: memcpy'd frames, also with the bit-shuffle flag", () => {
    const stored = new Uint8Array(48);
    stored.set([2, 1, 2, 1, 32, 0, 0, 0, 32, 0, 0, 0, 48, 0, 0, 0]);
    for (let i = 0; i < 32; i++) stored[16 + i] = i;
    expect([...decompressBlosc(stored, { maxBytes: 32 })]).toEqual([...Array(32).keys()]);
    stored[2]! |= 0x04;
    expect([...decompressBlosc(stored, { maxBytes: 32 })]).toEqual([...Array(32).keys()]);
  });

  it("RejectBitShuffle: real bit-shuffled frames fail and leave the output untouched", () => {
    for (const input of [BITSHUFFLE_BLOSCLZ, BITSHUFFLE_LZ4]) {
      const dest = new Uint8Array(128).fill(0xa5);
      expect(() => decompressBloscInto(input, dest)).toThrow(VDBUnsupportedError);
      expect(dest.every((b) => b === 0xa5)).toBe(true);
    }
  });
});

describe("decompressBloscLzBlock hostile streams", () => {
  const timed = (fn: () => unknown, message: RegExp) => {
    const t = performance.now();
    expect(fn).toThrow(VdbFormatError);
    expect(fn).toThrow(message);
    expect(performance.now() - t).toBeLessThan(TIME_BUDGET_MS);
  };

  it("empty input or output", () => {
    timed(() => decompressBloscLzBlock(new Uint8Array(0), new Uint8Array(8)), /is empty/);
    timed(() => decompressBloscLzBlock(Uint8Array.from([0]), new Uint8Array(0)), /is empty/);
  });

  it("a literal run past the output", () => {
    timed(() => decompressBloscLzBlock(Uint8Array.from([31, ...new Array(32).fill(7)]), new Uint8Array(16)), /past the output/);
  });

  it("a match-length chain past the output (no work per claimed byte)", () => {
    // literal 'a', then a long match (ctrl 0xe0: length 7) extended by a million 255 bytes.
    const src = new Uint8Array(1 << 20).fill(255);
    src.set([0, 0x61, 0xe0], 0);
    timed(() => decompressBloscLzBlock(src, new Uint8Array(4096)), /match runs past the output/);
  });

  it("a truncated 16-bit distance and a truncated match", () => {
    // ctrl 0x3f + 0xff is the all-ones 13-bit distance, the escape to a 16-bit one, which needs 2 more bytes.
    timed(() => decompressBloscLzBlock(Uint8Array.from([0, 0x61, 0x3f, 0xff, 0x00]), new Uint8Array(64)), /16-bit match distance is truncated/);
    timed(() => decompressBloscLzBlock(Uint8Array.from([0, 0x61, 0x20]), new Uint8Array(64)), /match distance is truncated/);
  });

  it("a run and an overlapping match decode like c-blosc", () => {
    // 'a', then a 3-byte run of the previous byte (distance 1), then literal 'b'.
    const run = new Uint8Array(5);
    expect(decompressBloscLzBlock(Uint8Array.from([0, 0x61, 0x20, 0x00, 0x00, 0x62]), run)).toBe(5);
    expect(new TextDecoder().decode(run)).toBe("aaaab");
    // 'ab', then a 4-byte match at distance 2 (overlapping its own output), then 'c'.
    const overlap = new Uint8Array(7);
    expect(decompressBloscLzBlock(Uint8Array.from([1, 0x61, 0x62, 0x40, 0x01, 0x00, 0x63]), overlap)).toBe(7);
    expect(new TextDecoder().decode(overlap)).toBe("ababab" + "c");
  });
});

// ---------------------------------------------------------------------------
// .nvdb Codec::BLOSC
// ---------------------------------------------------------------------------

const native = new Uint8Array(readFileSync(new URL("./fixtures/native-v14/sphere_fog_blosc.nvdb", import.meta.url)));
const dv = (b: Uint8Array) => new DataView(b.buffer, b.byteOffset, b.byteLength);
const ab = (b: Uint8Array): ArrayBuffer => b.slice().buffer;

/** Offsets in a one-grid segment: FileMetaData at 16, name, then the first chunk's u64 size. */
const META = 16;
const NAME_SIZE = dv(native).getUint32(META + 136, true);
const DATA = META + 176 + NAME_SIZE;

/** A small FLOAT ball image. */
function ball(dims: number): Uint32Array {
  const v = new Float32Array(dims ** 3);
  const c = (dims - 1) / 2;
  for (let i = 0; i < v.length; i++) {
    const x = Math.floor(i / (dims * dims));
    const y = Math.floor(i / dims) % dims;
    const z = i % dims;
    const d = Math.hypot(x - c, y - c, z - c) / (dims / 2);
    v[i] = d < 1 ? 1 - d : 0;
  }
  return vdbBuildFromDense(v, [dims, dims, dims], { gridName: "ball" });
}

/** A memcpy'd Blosc1 frame holding `bytes`. */
function memcpyFrame(bytes: Uint8Array): Uint8Array {
  const f = new Uint8Array(16 + bytes.length);
  const v = dv(f);
  f.set([2, 1, 0x02 | 0x01, 4]);
  v.setUint32(4, bytes.length, true);
  v.setUint32(8, bytes.length, true);
  v.setUint32(12, f.length, true);
  f.set(bytes, 16);
  return f;
}

/** `writeNvdb` output for `image`, re-framed as Codec::BLOSC with the image split at `cuts`. */
function bloscFile(image: Uint32Array, cuts: number[]): Uint8Array {
  const plain = new Uint8Array(writeNvdb([image]));
  const head = plain.slice(0, plain.length - image.byteLength);
  const img = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
  const edges = [0, ...cuts, img.length];
  const chunks: Uint8Array[] = [];
  for (let k = 0; k + 1 < edges.length; k++) {
    const frame = memcpyFrame(img.subarray(edges[k]!, edges[k + 1]!));
    const sized = new Uint8Array(8 + frame.length);
    dv(sized).setBigUint64(0, BigInt(frame.length), true);
    sized.set(frame, 8);
    chunks.push(sized);
  }
  const dataBytes = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(head.length + dataBytes);
  out.set(head);
  let at = head.length;
  for (const c of chunks) out.set(c, (at += c.length) - c.length);
  dv(out).setUint16(14, NvdbCodec.BLOSC, true);
  dv(out).setUint16(META + 168, NvdbCodec.BLOSC, true);
  dv(out).setBigUint64(META + 8, BigInt(dataBytes), true);
  return out;
}

/** Runs `fn`, recording the largest numeric-length Uint8Array allocated inside it. */
function withAllocSpy(fn: () => unknown): { error: unknown; maxAlloc: number; ms: number } {
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
  let error: unknown;
  try {
    fn();
  } catch (e) {
    error = e;
  } finally {
    globalThis.Uint8Array = Real;
  }
  return { error, maxAlloc, ms: performance.now() - t };
}

/** Both readers reject `file` with `cls` and `message`, fast and without allocating past `cap`. */
function expectBothReject(file: Uint8Array, message: RegExp, cls: typeof NvdbFormatError | typeof VDBUnsupportedError = NvdbFormatError, cap = 1 << 20): void {
  for (const read of [() => NvdbFile.fromArrayBuffer(ab(file)), () => new NvdbStreamReader().push(file)]) {
    const { error, maxAlloc, ms } = withAllocSpy(read);
    expect(error).toBeInstanceOf(cls);
    expect(error).toBeInstanceOf(VDBToolsError);
    expect((error as Error).message).toMatch(message);
    expect(ms).toBeLessThan(TIME_BUDGET_MS);
    expect(maxAlloc).toBeLessThanOrEqual(cap);
  }
}

describe(".nvdb Codec::BLOSC chunking", () => {
  const image = ball(20);
  const ref = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);

  it.each([
    ["one chunk", []],
    ["three chunks (NanoVDB splits at 1 GiB)", [1000, 5000]],
    ["a one-byte chunk", [1]],
  ])("%s: NvdbFile and the stream reader rebuild the image", (_, cuts) => {
    const file = bloscFile(image, cuts as number[]);
    const viaFile = NvdbFile.fromArrayBuffer(ab(file)).gridImage(0);
    expect(Buffer.from(viaFile.buffer, viaFile.byteOffset, viaFile.byteLength).equals(Buffer.from(ref))).toBe(true);
    const reader = new NvdbStreamReader();
    const grids = [];
    for (let at = 0; at < file.length; at += 333) grids.push(...reader.push(file.subarray(at, at + 333)));
    reader.end();
    expect(grids).toHaveLength(1);
    expect(Buffer.from(grids[0]!.image.buffer).equals(Buffer.from(ref))).toBe(true);
  });

  it("truncation anywhere in the chunks is an NvdbFormatError from both readers", () => {
    const file = bloscFile(image, [1000]);
    for (const cut of [DATA, DATA + 4, DATA + 8 + 10, file.length - 1]) {
      // Cut before most of the data, NvdbFile already knows the bytes left can't hold the grid.
      expect(() => NvdbFile.fromArrayBuffer(ab(file.subarray(0, cut)))).toThrow(/truncated|can't decode to more than/);
      expect(() => NvdbFile.fromArrayBuffer(ab(file.subarray(0, cut)))).toThrow(NvdbFormatError);
      const reader = new NvdbStreamReader();
      reader.push(file.subarray(0, cut));
      expect(() => reader.end()).toThrow(NvdbFormatError);
    }
  });

  it("chunks that add up past gridSize: the frame claims more than the grid still needs", () => {
    const file = bloscFile(image, [1000]);
    // Rewrite the second chunk's frame to claim one more byte than is left.
    const second = file.length - (ref.length - 1000) - 16;
    dv(file).setUint32(second + 4, ref.length - 1000 + 1, true);
    expectBothReject(file, /nbytes=\d+ exceeds the expected \d+-byte output/);
  });
});

describe(".nvdb Codec::BLOSC hostile input", () => {
  it("the native fixture is the valid baseline", () => {
    expect(NvdbFile.fromArrayBuffer(ab(native)).grids[0]!.name).toBe("density");
  });

  it("chunk size prefixes of 0, 15, 2^62 and past the grid's bound", () => {
    for (const size of [0n, 15n, 1n << 62n, BigInt(328672 + 17)]) {
      const bad = native.slice();
      dv(bad).setBigUint64(DATA, size, true);
      expectBothReject(bad, /-byte chunk, but a chunk holds a 16-byte Blosc header/);
    }
  });

  it("a 1 GiB gridSize in an 8 KiB file is refused before allocating (NvdbFile)", () => {
    const bad = native.slice();
    dv(bad).setBigUint64(META, 1n << 30n, true);
    const { error, maxAlloc, ms } = withAllocSpy(() => NvdbFile.fromArrayBuffer(ab(bad)));
    expect(error).toBeInstanceOf(NvdbFormatError);
    expect((error as Error).message).toMatch(/can't decode to more than/);
    expect(maxAlloc).toBeLessThan(1 << 20);
    expect(ms).toBeLessThan(TIME_BUDGET_MS);
    // The stream reader caps it with maxGridBytes instead.
    expect(() => new NvdbStreamReader({ maxGridBytes: 1 << 20 }).push(bad)).toThrow(/over maxGridBytes/);
  });

  it("a frame whose cbytes is not the chunk size", () => {
    const bad = native.slice();
    dv(bad).setUint32(DATA + 8 + 12, dv(bad).getUint32(DATA + 8 + 12, true) - 1, true);
    expectBothReject(bad, /cbytes=\d+ does not match the \d+ compressed bytes stored/);
  });

  it("a frame that decodes to 0 bytes", () => {
    const bad = native.slice();
    dv(bad).setUint32(DATA + 8 + 4, 0, true);
    expectBothReject(bad, /chunk that decodes to 0 bytes/);
  });

  it("an LZ4 stream with a corrupt byte is a format error, not a RangeError", () => {
    let failures = 0;
    for (let at = DATA + 8 + 16 + 40; at < native.length; at += 97) {
      const bad = native.slice();
      bad[at] = bad[at]! ^ 0xff;
      try {
        NvdbFile.fromArrayBuffer(ab(bad));
      } catch (e) {
        failures++;
        expect(e).toBeInstanceOf(VDBToolsError);
      }
    }
    expect(failures).toBeGreaterThan(0);
  });

  it("a Zstd frame is unsupported, naming the grid", () => {
    const bad = native.slice();
    bad[DATA + 8 + 2] = (bad[DATA + 8 + 2]! & 0x1f) | (4 << 5);
    expectBothReject(bad, /BLOSC grid #0 \("density"\): blosc codec Zstd is not supported/, VDBUnsupportedError);
  });

  it("the stream reader holds no more than the grid and one chunk", () => {
    const { maxAlloc } = withAllocSpy(() => new NvdbStreamReader().push(native));
    expect(maxAlloc).toBe(328672); // the grid image; the chunk (8 KiB) and scratch are smaller
  });
});

describe("raw NanoVDB1 grid buffers: the mGridSize walk (the 13.2 overflow class)", () => {
  const raw = (): Uint8Array => {
    const image = ball(12);
    return new Uint8Array(image.buffer.slice(image.byteOffset, image.byteOffset + image.byteLength));
  };

  it("the valid baseline reads", () => {
    expect(NvdbFile.fromArrayBuffer(ab(raw())).grids).toHaveLength(1);
  });

  it("mGridSize 0 with 2^32 - 1 grids is refused at once (it used to loop over empty grids)", () => {
    const bad = raw();
    dv(bad).setBigUint64(32, 0n, true);
    dv(bad).setUint32(28, 0xffffffff, true);
    const t = performance.now();
    expect(() => NvdbFile.fromArrayBuffer(ab(bad))).toThrow(/smaller than its 736-byte GridData \+ TreeData headers/);
    expect(performance.now() - t).toBeLessThan(TIME_BUDGET_MS);
  });

  it("mGridSize 2^63 is unrepresentable; past the buffer is truncated", () => {
    const huge = raw();
    dv(huge).setBigUint64(32, 1n << 63n, true);
    expect(() => NvdbFile.fromArrayBuffer(ab(huge))).toThrow(/unrepresentable gridSize/);
    const past = raw();
    dv(past).setBigUint64(32, BigInt(past.length + 8), true);
    expect(() => NvdbFile.fromArrayBuffer(ab(past))).toThrow(/truncated/);
  });
});
