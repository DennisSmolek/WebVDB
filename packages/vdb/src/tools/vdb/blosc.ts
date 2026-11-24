/**
 * blosc.ts — a pure-TS decoder for the Blosc1 frames OpenVDB and NanoVDB write.
 *
 * OpenVDB (`io/Compression.cc`, `bloscToStream`) compresses each node's value
 * buffer with `blosc_compress_ctx(clevel 9, byte shuffle, typesize
 * sizeof(ValueType), ..., "lz4", blocksize = whole buffer)`, writing the
 * compressed byte count as an i64 followed by one Blosc1 frame. The
 * openvdb.org `bunny_cloud.vdb` and `explosion.vdb` samples are stored this
 * way. NanoVDB's `Codec::BLOSC` `.nvdb` grids are LZ4 frames too
 * (`io::Internal::write`: clevel 9, byte shuffle, typesize 4, 256 KiB
 * blocks), and NVIDIA Warp writes `.nvdb` grids with python-blosc's defaults,
 * BloscLZ (K7.C; the stream decoder is `blosclz.ts`). This decoder covers
 * those frames, plus the `zlib` codec and memcpy'd frames for safety; other
 * codecs (Snappy, Zstd) and bit-shuffle throw `VDBUnsupportedError`.
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
 *   holds `nsplits` streams (typesize streams when split -- flag bit 4 clear,
 *   typesize <= 16, block has >= 128 elements, not the short last block --
 *   else 1), each an i32 compressed size and its bytes. A
 *   stream whose size equals its uncompressed size is stored raw.
 *
 * Derived from the format description above, not from any third-party code
 * (the BloscLZ stream decoder in `blosclz.ts` is a port; see its header). The
 * BloscLZ checks follow the Blosc that OpenVDB v14.0.0 vendors (`vdb_blosc.c`,
 * `validate_decompress_format`): a BloscLZ frame must carry stream format 1.
 */

import { decompressBloscLzBlock } from "./blosclz.js";
import { VdbFormatError, VDBUnsupportedError } from "./errors.js";
import { inflateZlib } from "./inflate.js";

const HEADER_SIZE = 16;
const FLAG_SHUFFLE = 0x01;
const FLAG_MEMCPYED = 0x02;
const FLAG_BITSHUFFLE = 0x04;
const FLAG_DONT_SPLIT = 0x10;
/** c-blosc `MAX_SPLITS` / `MIN_BUFFERSIZE`. */
const MAX_SPLITS = 16;
const MIN_BUFFERSIZE = 128;

const CODEC_NAMES = ["BloscLZ", "LZ4", "Snappy", "zlib", "Zstd"];
const CODEC_BLOSCLZ = 0;
const CODEC_LZ4 = 1;
const CODEC_ZLIB = 3;
/** `BLOSC_BLOSCLZ_VERSION_FORMAT`: the only BloscLZ stream format c-blosc 1.x writes and reads. */
const BLOSCLZ_VERSION_FORMAT = 1;

/** Default output cap when a caller does not know the expected size (1 GiB). */
const DEFAULT_MAX_BYTES = 1 << 30;

/** Bytes of a Blosc1 frame header. */
export const BLOSC_HEADER_SIZE = HEADER_SIZE;

export interface BloscLimits {
  /** Largest uncompressed size the caller will accept; checked before allocating. */
  maxBytes?: number;
  /** The frame's required `cbytes` header field (what the enclosing format says it stored). */
  expectedCbytes?: number;
}

/** The 16-byte Blosc1 frame header. */
export interface BloscHeader {
  version: number;
  versionlz: number;
  flags: number;
  typesize: number;
  /** Uncompressed bytes. */
  nbytes: number;
  blocksize: number;
  /** Bytes of the whole frame, header included. */
  cbytes: number;
}

/** Reads a Blosc1 frame header (the first 16 bytes) without validating it. */
export function readBloscHeader(frame: Uint8Array): BloscHeader {
  if (frame.length < HEADER_SIZE) {
    throw new VdbFormatError(`blosc frame is ${frame.length} bytes, shorter than its ${HEADER_SIZE}-byte header`);
  }
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  return {
    version: frame[0]!,
    versionlz: frame[1]!,
    flags: frame[2]!,
    typesize: frame[3]!,
    nbytes: view.getUint32(4, true),
    blocksize: view.getUint32(8, true),
    cbytes: view.getUint32(12, true),
  };
}

/**
 * Decompresses one Blosc1 frame and returns its `nbytes` uncompressed bytes.
 *
 * The frame is untrusted: every header field is validated against `maxBytes`
 * (and the frame's own `cbytes`) before any allocation or read, mirroring
 * c-blosc's `blosc_run_decompression_with_context` and `blosc_d` checks.
 */
export function decompressBlosc(frame: Uint8Array, limits: BloscLimits = {}): Uint8Array {
  const header = checkBloscHeader(frame, limits.maxBytes ?? DEFAULT_MAX_BYTES, limits.expectedCbytes);
  if (header.nbytes === 0) return new Uint8Array(0);
  if (header.flags & FLAG_MEMCPYED) return frame.slice(HEADER_SIZE, HEADER_SIZE + header.nbytes);
  const out = new Uint8Array(header.nbytes);
  decodeBlocks(frame, header, out);
  return out;
}

/**
 * Decompresses one Blosc1 frame into the start of `dest` and returns its
 * `nbytes`. `dest.length` is the output cap (a frame claiming more is
 * rejected before anything is decoded); the checks are those of
 * {@link decompressBlosc}, and the only allocation is the unshuffle scratch
 * (one block, never larger than the output).
 */
export function decompressBloscInto(frame: Uint8Array, dest: Uint8Array, expectedCbytes?: number): number {
  const header = checkBloscHeader(frame, dest.length, expectedCbytes);
  if (header.nbytes === 0) return 0;
  if (header.flags & FLAG_MEMCPYED) {
    dest.set(frame.subarray(HEADER_SIZE, HEADER_SIZE + header.nbytes));
    return header.nbytes;
  }
  decodeBlocks(frame, header, dest.subarray(0, header.nbytes));
  return header.nbytes;
}

/** Validates a frame header against the output cap and the stored byte count. */
function checkBloscHeader(frame: Uint8Array, maxBytes: number, expectedCbytes: number | undefined): BloscHeader {
  const header = readBloscHeader(frame);
  const { version, versionlz, flags, typesize, nbytes, blocksize, cbytes } = header;
  if (version === 0 || version > 2) {
    throw new VDBUnsupportedError(`blosc frame format version ${version} (only Blosc1, versions 1-2, is supported)`);
  }
  if (flags & 0x08) throw new VDBUnsupportedError("blosc frame has compressor flags from a future version");
  if (cbytes < HEADER_SIZE || cbytes > frame.length) {
    throw new VdbFormatError(`blosc header field cbytes=${cbytes} is outside the ${frame.length}-byte frame`);
  }
  if (expectedCbytes !== undefined && cbytes !== expectedCbytes) {
    throw new VdbFormatError(`blosc header field cbytes=${cbytes} does not match the ${expectedCbytes} compressed bytes stored`);
  }
  if (nbytes === 0) return header;
  if (nbytes > maxBytes) {
    throw new VdbFormatError(`blosc header field nbytes=${nbytes} exceeds the expected ${maxBytes}-byte output`);
  }
  if (blocksize === 0 || blocksize > maxBytes) {
    throw new VdbFormatError(`blosc header field blocksize=${blocksize} is zero or exceeds the expected ${maxBytes}-byte output`);
  }
  if (typesize === 0) throw new VdbFormatError("blosc header field typesize is zero");

  if (flags & FLAG_MEMCPYED) {
    if (nbytes + HEADER_SIZE !== cbytes) {
      throw new VdbFormatError(`memcpy'd blosc frame: cbytes=${cbytes} != nbytes(${nbytes}) + ${HEADER_SIZE}`);
    }
    return header;
  }

  if (flags & FLAG_BITSHUFFLE) {
    throw new VDBUnsupportedError("blosc bit-shuffle is not supported (OpenVDB writes byte shuffle)");
  }
  const codec = (flags >> 5) & 0x7;
  if (codec !== CODEC_LZ4 && codec !== CODEC_ZLIB && codec !== CODEC_BLOSCLZ) {
    throw new VDBUnsupportedError(
      `blosc codec ${CODEC_NAMES[codec] ?? `#${codec}`} is not supported (LZ4, BloscLZ and zlib are; OpenVDB writes LZ4)`,
    );
  }
  if (codec === CODEC_BLOSCLZ && versionlz !== BLOSCLZ_VERSION_FORMAT) {
    throw new VDBUnsupportedError(`BloscLZ stream format version ${versionlz} (only version ${BLOSCLZ_VERSION_FORMAT} exists)`);
  }
  return header;
}

/** Decodes every block of a compressed (not memcpy'd) frame into `out`, which is exactly `nbytes` long. */
function decodeBlocks(frame: Uint8Array, header: BloscHeader, out: Uint8Array): void {
  const { flags, typesize, nbytes, blocksize, cbytes } = header;
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  const codec = (flags >> 5) & 0x7;

  const nblocks = Math.ceil(nbytes / blocksize);
  const tableEnd = HEADER_SIZE + 4 * nblocks;
  if (tableEnd > cbytes) {
    throw new VdbFormatError(`blosc block-start table (${nblocks} blocks) does not fit in cbytes=${cbytes}`);
  }

  const shuffle = (flags & FLAG_SHUFFLE) !== 0 && typesize > 1;
  const dontSplit = (flags & FLAG_DONT_SPLIT) !== 0;
  // Never larger than the output: blocksize <= maxBytes, and a block never exceeds nbytes.
  const scratch = shuffle ? new Uint8Array(Math.min(blocksize, nbytes)) : null;

  for (let b = 0; b < nblocks; b++) {
    const bstart = view.getInt32(HEADER_SIZE + 4 * b, true);
    if (bstart < tableEnd || bstart + 4 > cbytes) {
      throw new VdbFormatError(`blosc block-start[${b}]=${bstart} is outside [${tableEnd}, ${cbytes - 4}]`);
    }
    const leftover = b === nblocks - 1 && nbytes % blocksize !== 0;
    const bsize = leftover ? nbytes % blocksize : blocksize;
    // c-blosc blosc_d: split only if the frame allows it, typesize <= MAX_SPLITS (16),
    // the block holds >= MIN_BUFFERSIZE (128) elements, and it is not the leftover block.
    // Frames from before the dont-split flag (c-blosc < 1.14) follow the same rule.
    const nsplits =
      !dontSplit && typesize <= MAX_SPLITS && Math.floor(bsize / typesize) >= MIN_BUFFERSIZE && !leftover
        ? typesize
        : 1;
    const neblock = bsize / nsplits;
    if (!Number.isInteger(neblock)) {
      throw new VdbFormatError(`blosc block ${b} of ${bsize} bytes does not divide into ${nsplits} splits`);
    }
    const dest = out.subarray(b * blocksize, b * blocksize + bsize);
    const target = scratch ? scratch.subarray(0, bsize) : dest;

    let p = bstart;
    for (let s = 0; s < nsplits; s++) {
      if (p + 4 > cbytes) throw new VdbFormatError(`blosc block ${b} stream ${s} size field is outside cbytes=${cbytes}`);
      const csize = view.getInt32(p, true);
      p += 4;
      if (csize < 0 || csize > cbytes - p) {
        throw new VdbFormatError(`blosc stream size ${csize} (block ${b}, stream ${s}) runs past cbytes=${cbytes}`);
      }
      const src = frame.subarray(p, p + csize);
      const part = target.subarray(s * neblock, (s + 1) * neblock);
      if (csize === neblock) {
        part.set(src);
      } else if (codec === CODEC_LZ4 || codec === CODEC_BLOSCLZ) {
        const written = codec === CODEC_LZ4 ? decompressLz4Block(src, part) : decompressBloscLzBlock(src, part);
        if (written !== neblock) {
          throw new VdbFormatError(`${CODEC_NAMES[codec]} stream decoded to ${written} bytes, expected ${neblock}`);
        }
      } else {
        const inflated = inflateZlib(src, neblock);
        if (inflated.length !== neblock) {
          throw new VdbFormatError(`zlib stream decoded to ${inflated.length} bytes, expected ${neblock}`);
        }
        part.set(inflated);
      }
      p += csize;
    }

    if (scratch) unshuffle(scratch.subarray(0, bsize), dest, typesize);
  }
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
