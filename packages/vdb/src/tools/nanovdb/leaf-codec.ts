/**
 * leaf-codec.ts — the leaf value-table encoder abstraction.
 *
 * The leaf is the only block whose *value representation* varies by grid type:
 * FLOAT stores 512 plain f32s; Fp8/FpN store a per-leaf (min, quantum) pair plus
 * a packed bit-stream of quantized codes. Everything else about a grid image
 * (topology, masks, internal-node tables, stats, bbox, GridData) is identical
 * across those types — the internal-node/root tiles and background stay plain
 * f32 even for quantized grids (verified against native fp8/fpn fixtures). So the
 * serializer is parameterised over a {@link LeafCodec}: `serialize.ts` lays out
 * the common blocks and hands each leaf to the codec, which owns the full leaf
 * write (bbox, flags, value mask, value table and per-leaf stats).
 *
 * Three codecs ship:
 *   - {@link FLOAT_LEAF_CODEC} — 512 raw f32 values + f32 stats.
 *   - {@link FP8_LEAF_CODEC} — fixed 8-bit quantization (`LeafData<Fp8>`).
 *   - {@link makeFpNLeafCodec} — variable bit-width (`LeafData<FpN>`); a
 *     per-leaf oracle picks the smallest of 1/2/4/8/16 bits meeting a tolerance.
 *
 * ## Encode authority
 *
 * The quantization mirrors `nanovdb::tools::CreateNanoGrid` (NanoVDB.h
 * `LeafFnBase`/`LeafData<Fp8|FpN>` + CreateNanoGrid.h `processLeafs`) and must
 * satisfy the repo's proven decoders (`read-value.ts` `leafFp*ReadFloat` and
 * `pnanovdb.wgsl`). Per-leaf `mMinimum`/`mQuantum` = min / (max-min)/((1<<bits)-1)
 * over ALL 512 values (active and inactive). A code is
 * `floor(encode*(v-min) + 0.5)` — round-to-nearest, matching native's
 * dithering-off `DitherLUT` (constant 0.5). Codes are packed LSB-first, value `n`
 * at global bit `n*bits`, identical to the reader's word/shift addressing.
 */

import {
  FLOAT_LAYOUT,
  FP8_LAYOUT,
  FP_LEAF_HEADER_SIZE,
  FPN_LAYOUT,
  type GridTypeLayout,
  LEAF_FLAG_HAS_BBOX,
  LEAF_OFF_BBOX_DIF_AND_FLAGS,
  LEAF_OFF_BBOX_MIN,
  LEAF_OFF_FP_CODES,
  LEAF_OFF_FP_MINIMUM,
  LEAF_OFF_FP_QUANTUM,
  LEAF_OFF_FP_STAT_AVG,
  LEAF_OFF_FP_STAT_DEV,
  LEAF_OFF_FP_STAT_MAX,
  LEAF_OFF_FP_STAT_MIN,
  LEAF_TABLE_COUNT,
  NODE_FLAG_STATS,
} from "./bytes.js";
import type { GridImageWriter } from "./bytes.js";
import type { StatsAccumulator } from "./stats.js";

/** The leaf data a codec needs: origin, all 512 values, mask, active-value stats. */
export interface EncodableLeaf {
  origin: [number, number, number];
  values: Float32Array;
  valueMask: Uint32Array;
  stats: StatsAccumulator;
}

/** Per-leaf encoding plan, computed once by `planLeaf` and reused by `encodeLeaf`. */
export interface LeafPlan {
  /** Total byte size of this leaf (header + value table). */
  readonly byteSize: number;
}

export interface LeafCodec {
  readonly layout: GridTypeLayout;
  /**
   * Computes a leaf's encoding plan (its byte size, and for FpN the chosen
   * bit-width) from its 512 values. Called once before layout so variable-size
   * (FpN) leaves can be placed at cumulative offsets.
   */
  planLeaf(values: Float32Array): LeafPlan;
  /**
   * Writes one leaf's entire block — bbox, `bbox_dif_and_flags`, value mask,
   * value table and per-leaf stats — into the image at absolute byte offset
   * `leafOff`. `plan` is the value returned by {@link planLeaf} for this leaf.
   */
  encodeLeaf(w: GridImageWriter, leafOff: number, leaf: EncodableLeaf, plan: LeafPlan): void;
  /** Writes a single constant value (an internal-node tile / background). */
  writeConstant(w: GridImageWriter, off: number, value: number): void;
}

// ---------------------------------------------------------------------------
// Shared leaf-header write (bbox + dif/flags + value mask) — common to all codecs
// ---------------------------------------------------------------------------

function clampByte(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** Writes bbox_min, bbox_dif_and_flags (with `flagsByte` in the high byte) and
 *  the 512-bit value mask. Returns nothing; the codec then writes its table. */
function writeLeafHeader(
  w: GridImageWriter,
  leafOff: number,
  leaf: EncodableLeaf,
  flagsByte: number,
): void {
  const s = leaf.stats;
  const bmin = s.isEmpty ? leaf.origin : s.bboxMin;
  const bmax = s.isEmpty ? leaf.origin : s.bboxMax;
  w.setI32(leafOff + LEAF_OFF_BBOX_MIN, bmin[0]);
  w.setI32(leafOff + LEAF_OFF_BBOX_MIN + 4, bmin[1]);
  w.setI32(leafOff + LEAF_OFF_BBOX_MIN + 8, bmin[2]);
  const difX = clampByte(bmax[0] - bmin[0]);
  const difY = clampByte(bmax[1] - bmin[1]);
  const difZ = clampByte(bmax[2] - bmin[2]);
  w.setU32(
    leafOff + LEAF_OFF_BBOX_DIF_AND_FLAGS,
    (difX | (difY << 8) | (difZ << 16) | (flagsByte << 24)) >>> 0,
  );
  w.setMaskWords(leafOff + 16, leaf.valueMask);
}

// ---------------------------------------------------------------------------
// FLOAT
// ---------------------------------------------------------------------------

/** FLOAT leaf codec: 512 raw f32 values, f32 stats. */
export const FLOAT_LEAF_CODEC: LeafCodec = {
  layout: FLOAT_LAYOUT,
  planLeaf() {
    return { byteSize: FLOAT_LAYOUT.leafSize };
  },
  encodeLeaf(w, leafOff, leaf) {
    writeLeafHeader(w, leafOff, leaf, NODE_FLAG_STATS);
    const L = FLOAT_LAYOUT;
    const values = leaf.values;
    for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
      w.setF32(leafOff + L.leafOffTable + n * 4, values[n]!);
    }
    // Stats block. For an all-inactive leaf the accumulator is empty; write
    // zeros, matching "no active values".
    const s = leaf.stats;
    w.setF32(leafOff + L.leafOffMin, s.isEmpty ? 0 : s.min);
    w.setF32(leafOff + L.leafOffMax, s.isEmpty ? 0 : s.max);
    w.setF32(leafOff + L.leafOffAve, s.average);
    w.setF32(leafOff + L.leafOffStdDev, s.stdDev);
  },
  writeConstant(w, off, value) {
    w.setF32(off, value);
  },
};

// ---------------------------------------------------------------------------
// Quantized (Fp8 / FpN) shared helpers
// ---------------------------------------------------------------------------

const fround = Math.fround;

/** Min and max over all 512 leaf values (the affine origin for quantization). */
function leafMinMax(values: Float32Array): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
    const v = values[n]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

/** `floor(encode*(v-min) + 0.5)` in float32, clamped to `[0, mask]`.
 *  Matches native `uint(encode*(v-min) + 0.5f)` with dithering disabled. */
function quantizeF32(v: number, min: number, encode: number, mask: number): number {
  const code = Math.floor(fround(fround(encode * fround(v - min)) + 0.5));
  return code < 0 ? 0 : code > mask ? mask : code;
}

/** `floor(encode*(v-min) + 0.5)` in double, clamped — the 16-bit FpN path. */
function quantizeF64(v: number, min: number, encode: number, mask: number): number {
  const code = Math.floor(encode * (v - min) + 0.5);
  return code < 0 ? 0 : code > mask ? mask : code;
}

/** Writes mMinimum, mQuantum (f32) and the four u16 quantized statistics. */
function writeFpHeaderStats(
  w: GridImageWriter,
  leafOff: number,
  min: number,
  quantum: number,
  stats: StatsAccumulator,
): void {
  w.setF32(leafOff + LEAF_OFF_FP_MINIMUM, min);
  w.setF32(leafOff + LEAF_OFF_FP_QUANTUM, quantum);
  // Quantized active-value statistics (uint16), per LeafFnBase::set{Min,Max,Avg,Dev}.
  const qInv = quantum > 0 ? 1 / quantum : 0;
  const q16 = (x: number): number => {
    const c = Math.floor(x + 0.5);
    return c < 0 ? 0 : c > 0xffff ? 0xffff : c;
  };
  if (stats.isEmpty) {
    w.setU16(leafOff + LEAF_OFF_FP_STAT_MIN, 0);
    w.setU16(leafOff + LEAF_OFF_FP_STAT_MAX, 0);
    w.setU16(leafOff + LEAF_OFF_FP_STAT_AVG, 0);
    w.setU16(leafOff + LEAF_OFF_FP_STAT_DEV, 0);
    return;
  }
  w.setU16(leafOff + LEAF_OFF_FP_STAT_MIN, q16((stats.min - min) * qInv));
  w.setU16(leafOff + LEAF_OFF_FP_STAT_MAX, q16((stats.max - min) * qInv));
  w.setU16(leafOff + LEAF_OFF_FP_STAT_AVG, q16((stats.average - min) * qInv));
  w.setU16(leafOff + LEAF_OFF_FP_STAT_DEV, q16(stats.stdDev * qInv));
}

/**
 * Packs 512 codes at `bits` bits/value, LSB-first, into the image at
 * `leafOff + 96`. Value `n` occupies global bit range `[n*bits, n*bits+bits)`,
 * exactly what `leafFpReadFloat`'s word/shift addressing reads back.
 */
function packCodes(
  w: GridImageWriter,
  leafOff: number,
  bits: number,
  codeAt: (n: number) => number,
): void {
  const base = leafOff + LEAF_OFF_FP_CODES;
  const bytes = w.bytes;
  if (bits === 8) {
    for (let n = 0; n < LEAF_TABLE_COUNT; n++) bytes[base + n] = codeAt(n);
    return;
  }
  if (bits === 16) {
    for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
      const c = codeAt(n);
      bytes[base + 2 * n] = c & 0xff;
      bytes[base + 2 * n + 1] = (c >>> 8) & 0xff;
    }
    return;
  }
  // 1/2/4 bits: sub-byte packing (buffer is zero-initialised, so OR in).
  for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
    const c = codeAt(n);
    let g = n * bits;
    for (let b = 0; b < bits; b++) {
      if ((c >>> b) & 1) bytes[base + (g >>> 3)]! |= 1 << (g & 7);
      g++;
    }
  }
}

// ---------------------------------------------------------------------------
// Fp8 — fixed 8-bit
// ---------------------------------------------------------------------------

export const FP8_LEAF_CODEC: LeafCodec = {
  layout: FP8_LAYOUT,
  planLeaf() {
    return { byteSize: FP8_LAYOUT.leafSize };
  },
  encodeLeaf(w, leafOff, leaf) {
    writeLeafHeader(w, leafOff, leaf, LEAF_FLAG_HAS_BBOX);
    const { min, max } = leafMinMax(leaf.values);
    const range = fround(max - min);
    const mask = 255;
    const quantum = range > 0 ? fround(range / mask) : 0;
    writeFpHeaderStats(w, leafOff, min, quantum, leaf.stats);
    const encode = range > 0 ? fround(mask / range) : 0;
    const values = leaf.values;
    packCodes(w, leafOff, 8, (n) => quantizeF32(values[n]!, min, encode, mask));
  },
  writeConstant(w, off, value) {
    w.setF32(off, value);
  },
};

// ---------------------------------------------------------------------------
// FpN — variable bit-width chosen per leaf by an absolute-difference oracle
// ---------------------------------------------------------------------------

/** Native's default absolute tolerance for a FogVolume (`AbsDiff::init`). */
export const FPN_DEFAULT_TOLERANCE = 0.01;

const LOG_BITS_MAX = 4; // logBitWidth in [0,4] -> 1,2,4,8,16 bits

/**
 * Picks the smallest logBitWidth (0..4 -> 1,2,4,8,16 bits) whose float32 encode
 * keeps every one of the 512 values within `tolerance` of its dequantized
 * approximation — the exact loop in `CreateNanoGrid::preProcess<FpN>`.
 */
function chooseFpNLogBits(values: Float32Array, min: number, max: number, tolerance: number): number {
  const range = fround(max - min);
  if (!(range > 0)) return 0; // constant leaf -> 1 bit (quantum 0)
  for (let logBits = 0; logBits < LOG_BITS_MAX; logBits++) {
    const mask = (1 << (1 << logBits)) - 1;
    const encode = fround(mask / range);
    const decode = fround(range / mask);
    let ok = true;
    for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
      const exact = values[n]!;
      const code = quantizeF32(exact, min, encode, mask);
      const approx = fround(fround(code * decode) + min);
      if (fround(Math.abs(exact - approx)) > tolerance) {
        ok = false;
        break;
      }
    }
    if (ok) return logBits;
  }
  return LOG_BITS_MAX;
}

interface FpNPlan extends LeafPlan {
  readonly logBits: number;
}

/** Builds an FpN codec for a given absolute tolerance (default: native's 0.01). */
export function makeFpNLeafCodec(tolerance: number = FPN_DEFAULT_TOLERANCE): LeafCodec {
  return {
    layout: FPN_LAYOUT,
    planLeaf(values): FpNPlan {
      const { min, max } = leafMinMax(values);
      const logBits = chooseFpNLogBits(values, min, max, tolerance);
      const bits = 1 << logBits;
      return { byteSize: FP_LEAF_HEADER_SIZE + bits * 64, logBits };
    },
    encodeLeaf(w, leafOff, leaf, plan) {
      const logBits = (plan as FpNPlan).logBits;
      const bits = 1 << logBits;
      writeLeafHeader(w, leafOff, leaf, (logBits << 5) | LEAF_FLAG_HAS_BBOX);
      const { min, max } = leafMinMax(leaf.values);
      const range = max - min;
      const mask = (1 << bits) - 1;
      const quantum = range > 0 ? (bits >= 16 ? range / mask : fround(range / mask)) : 0;
      writeFpHeaderStats(w, leafOff, min, quantum, leaf.stats);
      const values = leaf.values;
      if (bits >= 16) {
        const encode = range > 0 ? 65535 / range : 0;
        packCodes(w, leafOff, bits, (n) => quantizeF64(values[n]!, min, encode, mask));
      } else {
        const encode = range > 0 ? fround(mask / fround(range)) : 0;
        packCodes(w, leafOff, bits, (n) => quantizeF32(values[n]!, min, encode, mask));
      }
    },
    writeConstant(w, off, value) {
      w.setF32(off, value);
    },
  };
}
