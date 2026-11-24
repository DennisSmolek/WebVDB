/**
 * blosclz.ts — the BloscLZ block decoder, for Blosc1 frames whose codec is BloscLZ (K7.C).
 *
 * NVIDIA Warp writes `.nvdb` grids with python-blosc's defaults (BloscLZ), and NanoVDB 14 reads them
 * (OpenVDB v14.0.0 CHANGES: "Read BloscLZ-compressed .nvdb files written by Warp"). The `.vdb` writers
 * use LZ4; this decoder is read-only, like OpenVDB's.
 *
 * Ported from the decoder OpenVDB v14.0.0 vendors (`ext/blosc/openvdb/ext/blosc/blosclz.c`, "Decoder
 * extracted from c-blosc v1.21.6 (BloscLZ 2.5.1)", bounds checked before any read or write), which is
 * BSD-3-Clause and based on FastLZ (MIT):
 *
 *   Blosc - Blocked Shuffling and Compression Library
 *   Copyright (c) 2009-2018 Francesc Alted <francesc@blosc.org>
 *   Copyright (c) 2019-present Blosc Development Team <blosc@blosc.org>
 *   License: BSD 3-Clause (see THIRD_PARTY_NOTICES.md)
 *
 *   FastLZ - Byte-aligned LZ77 compression library
 *   Copyright (C) 2005-2020 Ariya Hidayat <ariya.hidayat@gmail.com>
 *   License: MIT (see THIRD_PARTY_NOTICES.md)
 *
 * Stream format: the first byte's low 5 bits are a literal run (`ctrl & 31`). Each later control byte
 * is a literal run of `ctrl + 1` bytes when `ctrl < 32`, else a match: length `(ctrl >> 5) + 2`
 * (`ctrl >> 5 == 7` adds following bytes until one is not 255), distance `((ctrl & 31) << 8) + next
 * byte + 1`, and the all-ones 13-bit distance (8191) escapes to a 16-bit big-endian distance + 8191.
 * Every failure the C decoder reports by returning 0 throws a `VdbFormatError` here. DIVERGES V68:
 * same behaviour; the C code's 8-byte wild copy becomes a byte loop.
 */

import { VdbFormatError } from "./errors.js";

/** BloscLZ's longest 13-bit match distance (`MAX_DISTANCE`). */
const MAX_DISTANCE = 8191;

function fail(why: string): never {
  throw new VdbFormatError(`BloscLZ stream ${why}`);
}

/**
 * Decodes one BloscLZ stream (one Blosc split) into `dest`, returning the bytes written. Never reads
 * past `src` or writes past `dest`. As in c-blosc, a stream that ends right after a match header stops
 * there without copying that match (encoders always end on literals).
 */
export function decompressBloscLzBlock(src: Uint8Array, dest: Uint8Array): number {
  const ipLimit = src.length;
  const opLimit = dest.length;
  if (ipLimit <= 0 || opLimit <= 0) fail("is empty");
  let ip = 0;
  let op = 0;
  let ctrl = src[ip++]! & 31;

  for (;;) {
    if (ctrl >= 32) {
      // Match.
      let len = (ctrl >> 5) - 1;
      let ofs = (ctrl & 31) << 8;
      let code: number;
      if (len === 7 - 1) {
        do {
          if (ipLimit - ip < 2) fail("match length runs past the input");
          code = src[ip++]!;
          if (opLimit - op < len + code + 3) fail("match runs past the output");
          len += code;
        } while (code === 255);
      } else if (ipLimit - ip < 2) {
        fail("match distance is truncated");
      }
      code = src[ip++]!;
      len += 3;
      ofs += code;
      // 16-bit distance escape.
      if (code === 255 && ofs === (31 << 8) + 255) {
        if (ipLimit - ip < 2) fail("16-bit match distance is truncated");
        ofs = (src[ip]! << 8) + src[ip + 1]! + MAX_DISTANCE;
        ip += 2;
      }
      if (len > opLimit - op) fail("match runs past the output");
      if (ofs >= op) fail(`match distance ${ofs + 1} reaches before the start of the output (at ${op})`);
      if (ip >= ipLimit) break;
      ctrl = src[ip++]!;

      let ref = op - ofs - 1;
      if (ofs === 0) {
        // A run of the previous byte.
        dest.fill(dest[ref]!, op, op + len);
        op += len;
      } else {
        // Byte by byte: the match may overlap the bytes it produces.
        for (; len > 0; --len) dest[op++] = dest[ref++]!;
      }
    } else {
      // Literal run.
      ctrl++;
      if (ctrl > opLimit - op) fail("literal run runs past the output");
      if (ctrl > ipLimit - ip) fail("literal run runs past the input");
      dest.set(src.subarray(ip, ip + ctrl), op);
      op += ctrl;
      ip += ctrl;
      if (ip >= ipLimit) break;
      ctrl = src[ip++]!;
    }
  }
  return op;
}
