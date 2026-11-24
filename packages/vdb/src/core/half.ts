/**
 * IEEE 754 binary16 -> number, for the few Half fields the pool reads from a
 * grid header (root background and min/max; K7.H). The core never imports
 * `tools` (D20), so this restates `tools`' `vdbHalfToFloat`;
 * `test/core/half.test.ts` checks the two agree on all 65536 bit patterns.
 *
 * Exact: every binary16 value (subnormals, infinities, NaN) is representable
 * as an f32 and as a JS number. Same result as NanoVDB `math::HalfToFloat`
 * (`nanovdb/math/HalfLite.h`, from Imath) and WGSL `unpack2x16float`.
 */
export function halfBitsToNumber(bits: number): number {
  const h = bits & 0xffff;
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return sign * man * 2 ** -24;
  if (exp === 31) return man === 0 ? sign * Infinity : NaN;
  return sign * (1024 + man) * 2 ** (exp - 25);
}
