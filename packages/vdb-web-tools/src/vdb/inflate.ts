/**
 * inflate.ts — pure-TypeScript DEFLATE (RFC 1951) + zlib (RFC 1950) decoder.
 *
 * Deviation from the original task brief: the brief called for `unzlibSync`
 * from `fflate`. `fflate` is a declared dependency of the sibling
 * `nanovdb-wgsl` package but NOT of `vdb-web-tools`, and this lane forbids
 * `pnpm install`/adding it to `package.json` (workspace symlinking needs an
 * install to take effect — editing `package.json` alone would not resolve).
 * Verified: `node -e "import('fflate')"` from `packages/vdb-web-tools` fails
 * with `ERR_MODULE_NOT_FOUND`; it is not hoisted to a location plain Node
 * resolution reaches either. Rather than block v1 zlib support on that, this
 * module implements RFC 1950/1951 directly — a self-contained ~250-line
 * decoder with zero runtime dependencies, which is arguably a *better* fit
 * for "pure TypeScript, zero wasm in the default install" (D3) than an
 * external codec would have been. See the phase handoff for the follow-up:
 * either wire in `fflate` as a real dependency later, or keep this.
 *
 * Algorithm follows the public-domain RFC 1951 reference approach (canonical
 * Huffman via per-length counts + a sorted symbol table, as in Mark Adler's
 * `puff.c`) — written fresh from the RFC text, not transcribed from any
 * existing implementation.
 */

import { VdbFormatError } from "./errors.js";

const MAX_BITS = 15;

class BitReader {
  private pos = 0;
  private bitBuf = 0;
  private bitCount = 0;

  constructor(private readonly data: Uint8Array) {}

  private byte(): number {
    if (this.pos >= this.data.length) {
      throw new VdbFormatError("inflate: unexpected end of compressed data");
    }
    return this.data[this.pos++]!;
  }

  /** Reads `n` bits (n <= 24), LSB-first accumulation per RFC 1951 §3.1.1. */
  bits(n: number): number {
    while (this.bitCount < n) {
      this.bitBuf |= this.byte() << this.bitCount;
      this.bitCount += 8;
    }
    const value = this.bitBuf & ((1 << n) - 1);
    this.bitBuf >>>= n;
    this.bitCount -= n;
    return value;
  }

  /** Discards any partial byte in the bit buffer (stored-block alignment). */
  alignToByte(): void {
    this.bitBuf = 0;
    this.bitCount = 0;
  }

  readU16LE(): number {
    return this.byte() | (this.byte() << 8);
  }

  get bytePos(): number {
    return this.pos;
  }

  /** Advances the raw byte cursor (only valid right after `alignToByte()`). */
  skipBytes(n: number): void {
    this.pos += n;
  }
}

interface HuffTable {
  counts: Uint16Array; // counts[len] = number of codes of that length
  symbols: Uint16Array; // symbols sorted by (length, code)
}

function buildHuffman(lengths: ArrayLike<number>): HuffTable {
  const counts = new Uint16Array(MAX_BITS + 1);
  for (let i = 0; i < lengths.length; i++) counts[lengths[i]!]!++;
  counts[0] = 0;

  const offsets = new Uint16Array(MAX_BITS + 2);
  for (let len = 1; len <= MAX_BITS; len++) {
    offsets[len + 1] = offsets[len]! + counts[len]!;
  }

  const symbols = new Uint16Array(lengths.length);
  const cursor = offsets.slice(0, MAX_BITS + 1);
  for (let sym = 0; sym < lengths.length; sym++) {
    const len = lengths[sym]!;
    if (len !== 0) symbols[cursor[len]!++] = sym;
  }
  return { counts, symbols };
}

/** Canonical-Huffman symbol decode (count/symbol-table walk, per `puff.c`). */
function decodeSymbol(br: BitReader, table: HuffTable): number {
  let code = 0;
  let first = 0;
  let index = 0;
  for (let len = 1; len <= MAX_BITS; len++) {
    code |= br.bits(1);
    const count = table.counts[len]!;
    if (code - first < count) return table.symbols[index + (code - first)]!;
    index += count;
    first += count;
    first <<= 1;
    code <<= 1;
  }
  throw new VdbFormatError("inflate: invalid Huffman code");
}

let fixedLitLen: HuffTable | undefined;
let fixedDist: HuffTable | undefined;

function getFixedLitLenTable(): HuffTable {
  if (!fixedLitLen) {
    const lengths = new Uint8Array(288);
    lengths.fill(8, 0, 144);
    lengths.fill(9, 144, 256);
    lengths.fill(7, 256, 280);
    lengths.fill(8, 280, 288);
    fixedLitLen = buildHuffman(lengths);
  }
  return fixedLitLen;
}

function getFixedDistTable(): HuffTable {
  if (!fixedDist) {
    fixedDist = buildHuffman(new Uint8Array(30).fill(5));
  }
  return fixedDist;
}

const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function readDynamicTables(br: BitReader): { litLen: HuffTable; dist: HuffTable } {
  const hlit = br.bits(5) + 257;
  const hdist = br.bits(5) + 1;
  const hclen = br.bits(4) + 4;

  const clLengths = new Uint8Array(19);
  for (let i = 0; i < hclen; i++) clLengths[CL_ORDER[i]!] = br.bits(3);
  const clTable = buildHuffman(clLengths);

  const lengths = new Uint8Array(hlit + hdist);
  let i = 0;
  while (i < lengths.length) {
    const sym = decodeSymbol(br, clTable);
    if (sym < 16) {
      lengths[i++] = sym;
    } else if (sym === 16) {
      if (i === 0) throw new VdbFormatError("inflate: repeat code with no previous length");
      const repeat = br.bits(2) + 3;
      const prev = lengths[i - 1]!;
      for (let r = 0; r < repeat; r++) lengths[i++] = prev;
    } else if (sym === 17) {
      const repeat = br.bits(3) + 3;
      for (let r = 0; r < repeat; r++) lengths[i++] = 0;
    } else if (sym === 18) {
      const repeat = br.bits(7) + 11;
      for (let r = 0; r < repeat; r++) lengths[i++] = 0;
    } else {
      throw new VdbFormatError(`inflate: invalid code-length symbol ${sym}`);
    }
  }

  return {
    litLen: buildHuffman(lengths.subarray(0, hlit)),
    dist: buildHuffman(lengths.subarray(hlit)),
  };
}

const LENGTH_BASE = [
  3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
  163, 195, 227, 258,
];
const LENGTH_EXTRA = [
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE = [
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049,
  3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA = [
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];

/** A growable byte buffer that supports LZ77's overlapping back-references. */
class ByteSink {
  private buf: Uint8Array;
  len = 0;

  constructor(initialCapacity = 1 << 16) {
    this.buf = new Uint8Array(initialCapacity);
  }

  private ensure(extra: number): void {
    if (this.len + extra <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.len + extra) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
  }

  pushByte(b: number): void {
    this.ensure(1);
    this.buf[this.len++] = b;
  }

  pushBytes(src: Uint8Array): void {
    this.ensure(src.length);
    this.buf.set(src, this.len);
    this.len += src.length;
  }

  /** Copies `length` bytes from `distance` bytes back — byte-by-byte, since
   *  `distance < length` (overlapping runs) is the normal LZ77 repeat case. */
  copyBack(distance: number, length: number): void {
    if (distance <= 0 || distance > this.len) {
      throw new VdbFormatError(`inflate: back-reference distance ${distance} exceeds output so far`);
    }
    this.ensure(length);
    let src = this.len - distance;
    for (let i = 0; i < length; i++) {
      this.buf[this.len++] = this.buf[src++]!;
    }
  }

  result(): Uint8Array {
    return this.buf.subarray(0, this.len);
  }
}

function inflateBlockData(br: BitReader, sink: ByteSink, litLen: HuffTable, dist: HuffTable): void {
  for (;;) {
    const sym = decodeSymbol(br, litLen);
    if (sym < 256) {
      sink.pushByte(sym);
    } else if (sym === 256) {
      return;
    } else {
      const li = sym - 257;
      if (li >= LENGTH_BASE.length) throw new VdbFormatError(`inflate: invalid length symbol ${sym}`);
      const length = LENGTH_BASE[li]! + br.bits(LENGTH_EXTRA[li]!);
      const dsym = decodeSymbol(br, dist);
      if (dsym >= DIST_BASE.length) throw new VdbFormatError(`inflate: invalid distance symbol ${dsym}`);
      const distance = DIST_BASE[dsym]! + br.bits(DIST_EXTRA[dsym]!);
      sink.copyBack(distance, length);
    }
  }
}

/** Raw DEFLATE (RFC 1951) decompression, no zlib/gzip framing. */
export function inflateRaw(data: Uint8Array): Uint8Array {
  const br = new BitReader(data);
  const sink = new ByteSink(Math.max(1 << 16, data.length * 3));

  let final = false;
  while (!final) {
    final = br.bits(1) === 1;
    const type = br.bits(2);
    if (type === 0) {
      br.alignToByte();
      const len = br.readU16LE();
      const nlen = br.readU16LE();
      if ((len ^ 0xffff) !== nlen) {
        throw new VdbFormatError("inflate: stored-block length/~length mismatch");
      }
      const start = br.bytePos;
      if (start + len > data.length) {
        throw new VdbFormatError("inflate: stored block runs past end of input");
      }
      sink.pushBytes(data.subarray(start, start + len));
      br.skipBytes(len);
    } else if (type === 1 || type === 2) {
      const { litLen, dist } = type === 1
        ? { litLen: getFixedLitLenTable(), dist: getFixedDistTable() }
        : readDynamicTables(br);
      inflateBlockData(br, sink, litLen, dist);
    } else {
      throw new VdbFormatError(`inflate: invalid block type ${type}`);
    }
  }
  return sink.result();
}

function adler32(data: Uint8Array): number {
  const MOD = 65521;
  const NMAX = 5552;
  let a = 1;
  let b = 0;
  let i = 0;
  const n = data.length;
  while (i < n) {
    const chunk = Math.min(NMAX, n - i);
    for (let j = 0; j < chunk; j++, i++) {
      a += data[i]!;
      b += a;
    }
    a %= MOD;
    b %= MOD;
  }
  return ((b << 16) | a) >>> 0;
}

/** zlib (RFC 1950) decompression: 2-byte header + DEFLATE stream + Adler-32. */
export function inflateZlib(data: Uint8Array): Uint8Array {
  if (data.length < 6) throw new VdbFormatError("zlib: stream too short to contain a header/trailer");
  const cmf = data[0]!;
  const flg = data[1]!;
  if ((cmf & 0x0f) !== 8) {
    throw new VdbFormatError(`zlib: unsupported compression method ${cmf & 0x0f} (expected 8/deflate)`);
  }
  if (((cmf << 8) + flg) % 31 !== 0) {
    throw new VdbFormatError("zlib: header checksum (FCHECK) failed");
  }
  let offset = 2;
  if (flg & 0x20) {
    throw new VdbFormatError("zlib: preset dictionaries (FDICT) are not supported");
  }
  const raw = inflateRaw(data.subarray(offset, data.length - 4));
  const expected =
    (data[data.length - 4]! << 24) |
    (data[data.length - 3]! << 16) |
    (data[data.length - 2]! << 8) |
    data[data.length - 1]!;
  const actual = adler32(raw) | 0;
  if (actual !== expected) {
    throw new VdbFormatError(
      `zlib: Adler-32 checksum mismatch (expected ${(expected >>> 0).toString(16)}, got ${(actual >>> 0).toString(16)}) — decompressed output is likely corrupt`,
    );
  }
  return raw;
}
