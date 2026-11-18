/**
 * blosc.ts — a pure-TS decoder for the Blosc1 frames OpenVDB writes.
 *
 * OpenVDB (`io/Compression.cc`, `bloscToStream`) compresses each node's value
 * buffer with `blosc_compress_ctx(clevel 9, byte shuffle, typesize
 * sizeof(ValueType), ..., "lz4", blocksize = whole buffer)`, writing the
 * compressed byte count as an i64 followed by one Blosc1 frame. The
 * openvdb.org `bunny_cloud.vdb` and `explosion.vdb` samples are stored this
 * way. This decoder covers the frames that writer produces, plus the
 * `zlib` codec and memcpy'd frames for safety; other codecs (BloscLZ, Snappy,
 * Zstd) and bit-shuffle throw `VdbUnsupportedError`.
 *
 * Frame layout (c-blosc 1.x `blosc.c`, format version 2):
 *
 *   [0] version  [1] versionlz  [2] flags  [3] typesize
 *   [4..8) nbytes (uncompressed)  [8..12) blocksize  [12..16) cbytes (whole frame)
 *   flags: 0x01 byte shuffle, 0x02 memcpy'd, 0x04 bit shuffle,
 *          0x10 blocks not split, bits 5-7 codec (0 BloscLZ, 1 LZ4/LZ4HC,
 *          2 Snappy, 3 zlib, 4 Zstd)
 *   memcpy'd: nbytes raw bytes follow the header.
 *   otherwise: nblocks i32 block starts (absolute frame offsets); each block
 *   holds `nsplits` streams (typesize streams when split, else 1; the last,
 *   short block is never split), each an i32 compressed size and its bytes. A
 *   stream whose size equals its uncompressed size is stored raw.
 *
 * Derived from the format description above, not from any third-party code.
 */

import { VdbFormatError, VdbUnsupportedError } from "./errors.js";
import { inflateZlib } from "./inflate.js";

const HEADER_SIZE = 16;
const FLAG_SHUFFLE = 0x01;
const FLAG_MEMCPYED = 0x02;
const FLAG_BITSHUFFLE = 0x04;
const FLAG_DONT_SPLIT = 0x10;

const CODEC_NAMES = ["BloscLZ", "LZ4", "Snappy", "zlib", "Zstd"];
const CODEC_LZ4 = 1;
const CODEC_ZLIB = 3;

/** Decompresses one Blosc1 frame and returns its `nbytes` uncompressed bytes. */
export function decompressBlosc(frame: Uint8Array): Uint8Array {
  if (frame.length < HEADER_SIZE) {
    throw new VdbFormatError(`blosc frame is ${frame.length} bytes, shorter than its ${HEADER_SIZE}-byte header`);
  }
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  const version = frame[0]!;
  const flags = frame[2]!;
  const typesize = frame[3]!;
  const nbytes = view.getUint32(4, true);
  const blocksize = view.getUint32(8, true);
  const cbytes = view.getUint32(12, true);
  if (version === 0 || version > 2) {
    throw new VdbUnsupportedError(`blosc frame format version ${version} (only Blosc1, versions 1-2, is supported)`);
  }
  if (cbytes > frame.length) {
    throw new VdbFormatError(`blosc frame declares ${cbytes} bytes but only ${frame.length} are present`);
  }

  const out = new Uint8Array(nbytes);
  if (nbytes === 0) return out;

  if (flags & FLAG_MEMCPYED) {
    if (HEADER_SIZE + nbytes > frame.length) {
      throw new VdbFormatError("memcpy'd blosc frame is truncated");
    }
    out.set(frame.subarray(HEADER_SIZE, HEADER_SIZE + nbytes));
    return out;
  }

  if (flags & FLAG_BITSHUFFLE) {
    throw new VdbUnsupportedError("blosc bit-shuffle is not supported (OpenVDB writes byte shuffle)");
  }
  const codec = (flags >> 5) & 0x7;
  if (codec !== CODEC_LZ4 && codec !== CODEC_ZLIB) {
    throw new VdbUnsupportedError(
      `blosc codec ${CODEC_NAMES[codec] ?? `#${codec}`} is not supported (LZ4 and zlib are; OpenVDB writes LZ4)`,
    );
  }
  if (blocksize === 0) throw new VdbFormatError("blosc frame has a zero block size");

  const shuffle = (flags & FLAG_SHUFFLE) !== 0 && typesize > 1;
  const dontSplit = (flags & FLAG_DONT_SPLIT) !== 0;
  const nblocks = Math.ceil(nbytes / blocksize);
  const scratch = shuffle ? new Uint8Array(blocksize) : null;

  for (let b = 0; b < nblocks; b++) {
    const bstart = view.getInt32(HEADER_SIZE + 4 * b, true);
    const leftover = b === nblocks - 1 && nbytes % blocksize !== 0;
    const bsize = leftover ? nbytes % blocksize : blocksize;
    const nsplits = !dontSplit && !leftover && typesize > 0 && bsize % typesize === 0 ? typesize : 1;
    const neblock = bsize / nsplits;
    const dest = out.subarray(b * blocksize, b * blocksize + bsize);
    const target = scratch ? scratch.subarray(0, bsize) : dest;

    let p = bstart;
    for (let s = 0; s < nsplits; s++) {
      if (p + 4 > frame.length) throw new VdbFormatError("blosc block is truncated");
      const csize = view.getInt32(p, true);
      p += 4;
      if (csize < 0 || p + csize > frame.length) {
        throw new VdbFormatError(`blosc stream size ${csize} runs past the frame`);
      }
      const src = frame.subarray(p, p + csize);
      const part = target.subarray(s * neblock, (s + 1) * neblock);
      if (csize === neblock) {
        part.set(src);
      } else if (codec === CODEC_LZ4) {
        const written = decompressLz4Block(src, part);
        if (written !== neblock) {
          throw new VdbFormatError(`LZ4 stream decoded to ${written} bytes, expected ${neblock}`);
        }
      } else {
        const inflated = inflateZlib(src);
        if (inflated.length !== neblock) {
          throw new VdbFormatError(`zlib stream decoded to ${inflated.length} bytes, expected ${neblock}`);
        }
        part.set(inflated);
      }
      p += csize;
    }

    if (scratch) unshuffle(scratch.subarray(0, bsize), dest, typesize);
  }
  return out;
}

/**
 * Reverses Blosc's byte shuffle: the shuffled block stores byte `i` of every
 * element together (`src[i * count + j]` is byte `i` of element `j`); trailing
 * bytes that do not fill a whole element are stored unshuffled at the end.
 */
export function unshuffle(src: Uint8Array, dest: Uint8Array, typesize: number): void {
  const count = Math.floor(src.length / typesize);
  for (let j = 0; j < count; j++) {
    for (let i = 0; i < typesize; i++) dest[j * typesize + i] = src[i * count + j]!;
  }
  const tail = count * typesize;
  dest.set(src.subarray(tail), tail);
}

/**
 * Decodes one raw LZ4 block (no frame header) into `dest`, returning the
 * number of bytes written. Each sequence is a token (high nibble literal
 * length, low nibble match length - 4, 15 meaning "more bytes follow, each
 * added, until one is < 255"), the literals, then a 2-byte little-endian
 * match offset and the match; the last sequence carries literals only.
 */
export function decompressLz4Block(src: Uint8Array, dest: Uint8Array): number {
  let ip = 0;
  let op = 0;
  const end = src.length;
  while (ip < end) {
    const token = src[ip++]!;

    let litLen = token >>> 4;
    if (litLen === 15) {
      let byte: number;
      do {
        if (ip >= end) throw new VdbFormatError("LZ4 literal length runs past the input");
        byte = src[ip++]!;
        litLen += byte;
      } while (byte === 255);
    }
    if (ip + litLen > end || op + litLen > dest.length) {
      throw new VdbFormatError("LZ4 literals run past the input or output");
    }
    dest.set(src.subarray(ip, ip + litLen), op);
    ip += litLen;
    op += litLen;
    if (ip === end) break; // last sequence: literals only

    if (ip + 2 > end) throw new VdbFormatError("LZ4 match offset is truncated");
    const offset = src[ip]! | (src[ip + 1]! << 8);
    ip += 2;
    if (offset === 0 || offset > op) throw new VdbFormatError(`LZ4 match offset ${offset} is out of range`);

    let matchLen = token & 0xf;
    if (matchLen === 15) {
      let byte: number;
      do {
        if (ip >= end) throw new VdbFormatError("LZ4 match length runs past the input");
        byte = src[ip++]!;
        matchLen += byte;
      } while (byte === 255);
    }
    matchLen += 4;
    if (op + matchLen > dest.length) throw new VdbFormatError("LZ4 match runs past the output");
    // Byte-by-byte: a match may overlap the bytes it is producing (offset < length).
    let from = op - offset;
    for (let k = 0; k < matchLen; k++) dest[op++] = dest[from++]!;
  }
  return op;
}
