// Portions ported from OpenVDB nanovdb/nanovdb/math/HalfLite.h (v14.0.0), Copyright Contributors to the OpenVDB
// Project, Apache-2.0; which derives HalfToFloat / FloatToHalf from Imath's half.h, Copyright Contributors to the
// OpenEXR Project, BSD-3-Clause (primary original authors Florian Kainz, Rod Bogart). See THIRD_PARTY_NOTICES.md.
/**
 * half.ts — IEEE 754 binary16 ("half") conversions for NanoVDB Half grids
 * (`GridType::Half` = 9, raw 16-bit values; NanoVDB 13.2, K7.H).
 *
 * Both directions are ports of NanoVDB's `nanovdb/math/HalfLite.h`
 * (`HalfToFloat`, `FloatToHalf`; derived from Imath's `half.h`), which
 * `tools::createNanoGrid` uses to convert float grids to Half grids. So a value
 * we write is the bit pattern native NanoVDB writes for the same f32:
 *
 * - `vdbFloatToHalf`: round to nearest, ties to even; values past the largest
 *   half (65504, or exactly: `|v| > 0x477fefff` as f32 bits) become signed
 *   infinity; tiny values become signed zero or a subnormal; NaN keeps its sign
 *   and top 10 payload bits (forced non-zero). The input is rounded to f32 first,
 *   as `createNanoGrid` converts a double to float before calling it.
 * - `vdbHalfToFloat`: exact (every half, subnormals included, is an f32).
 */

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** The value of binary16 bit pattern `bits` (low 16 bits used). Exact. */
export function vdbHalfToFloat(bits: number): number {
  const h = bits & 0xffff;
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return sign * man * 2 ** -24;
  if (exp === 31) return man === 0 ? sign * Infinity : NaN;
  return sign * (1024 + man) * 2 ** (exp - 25);
}

/** The binary16 bit pattern of `value` (rounded to f32 first): NanoVDB `math::FloatToHalf`. */
export function vdbFloatToHalf(value: number): number {
  f32[0] = value;
  const bits = u32[0]!;
  const ui = bits & 0x7fffffff;
  let h = (bits >>> 16) & 0x8000;

  if (ui >= 0x38800000) {
    // Infinity or NaN: keep the sign and the available payload bits.
    if (ui >= 0x7f800000) {
      h |= 0x7c00;
      if (ui === 0x7f800000) return h;
      const m = (ui & 0x7fffff) >>> 13;
      return h | m | (m === 0 ? 1 : 0);
    }
    // Overflow (rounds past 65504).
    if (ui > 0x477fefff) return h | 0x7c00;
    // Normal: rebias, round to nearest even.
    const r = ui - 0x38000000;
    return h | ((r + 0x00000fff + ((r >>> 13) & 1)) >>> 13);
  }

  // Zero or a value that rounds to zero.
  if (ui < 0x33000001) return h;

  // Subnormal half.
  const e = ui >>> 23;
  const shift = 0x7e - e;
  const m = 0x800000 | (ui & 0x7fffff);
  const r = (m << (32 - shift)) >>> 0;
  h |= m >>> shift;
  if (r > 0x80000000 || (r === 0x80000000 && (h & 1) !== 0)) h++;
  return h;
}

/** `value` rounded to the nearest half (as a number): `vdbHalfToFloat(vdbFloatToHalf(value))`. */
export function vdbRoundToHalf(value: number): number {
  return vdbHalfToFloat(vdbFloatToHalf(value));
}
