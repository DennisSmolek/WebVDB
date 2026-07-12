/**
 * bit-utils.ts — bit-mask helpers shared by tree/compression code.
 *
 * OpenVDB node masks (child mask, value mask, selection mask) are serialized
 * as arrays of 64-bit words, LSB-first within each word. We store them as
 * `Uint32Array` instead of pairs of `bigint`s: splitting each on-disk u64 LE
 * word into two u32 LE words preserves bit-for-bit ordinal position (word
 * `i`'s bit `b` covers the same logical slot either way), and plain 32-bit
 * ops are both faster and simpler in JS than `bigint` bit-twiddling.
 */

/** Population count (number of set bits) of a 32-bit unsigned integer. */
export function popcount32(x: number): number {
  let v = x >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

/** Whether bit `index` is set in a mask stored as 32-bit words. */
export function testBit(words: Uint32Array, index: number): boolean {
  const word = words[index >>> 5] ?? 0;
  return ((word >>> (index & 31)) & 1) !== 0;
}

/** Total number of set bits across every word (i.e. `countOn()`). */
export function countBits(words: Uint32Array): number {
  let total = 0;
  for (let i = 0; i < words.length; i++) total += popcount32(words[i]!);
  return total;
}

/** Total number of unset bits across `bitCount` logical bits. */
export function countZeroBits(words: Uint32Array, bitCount: number): number {
  return bitCount - countBits(words);
}

/** Indices (ascending) of every set bit, 0..bitCount-1. */
export function* iterOnes(words: Uint32Array, bitCount: number): Iterable<number> {
  for (let i = 0; i < bitCount; i++) {
    if (testBit(words, i)) yield i;
  }
}
