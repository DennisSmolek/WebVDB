/**
 * half-float.ts — IEEE 754 binary16 -> binary32 (f32) decode.
 *
 * OpenVDB grids can opt into storing leaf/tile value arrays as half floats
 * (`GridDescriptor`'s `_HalfFloat` type-name suffix / the `is_saved_as_half_float`
 * grid metadata bool) even when the grid's nominal value type is `float`. We
 * only ever need half -> f32 (read direction); no encode path.
 */
export function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exponent = (h & 0x7c00) >> 10;
  const mantissa = h & 0x03ff;

  if (exponent === 0) {
    // Subnormal (or zero): value = (-1)^sign * mantissa * 2^-24.
    return (sign ? -1 : 1) * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) {
    // Inf / NaN.
    return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  // Normal: value = (-1)^sign * (1 + mantissa/1024) * 2^(exponent-15).
  return (sign ? -1 : 1) * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}
