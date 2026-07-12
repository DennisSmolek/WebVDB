/**
 * leaf-codec.ts — the leaf value-table encoder abstraction.
 *
 * The leaf is the only block whose *value representation* varies by grid type:
 * FLOAT stores 512 plain f32s; Fp8/FpN store a per-leaf (min, quantum) pair plus
 * a packed bit-stream of quantized codes. Everything else about a grid image
 * (topology, masks, internal-node tables, stats, bbox, GridData) is identical
 * across those types. So the serializer is parameterised over a {@link LeafCodec}
 * and this wave ships exactly one — {@link FLOAT_LEAF_CODEC}. The Fp8/FpN wave
 * adds sibling codecs (quantizing `values` into the packed table and writing the
 * min/quantum at the leaf's negative offsets) with no change to `serialize.ts`.
 */

import {
  FLOAT_LAYOUT,
  type GridTypeLayout,
  LEAF_TABLE_COUNT,
} from "./bytes.js";
import type { GridImageWriter } from "./bytes.js";
import type { StatsAccumulator } from "./stats.js";

export interface LeafCodec {
  readonly layout: GridTypeLayout;
  /**
   * Writes one leaf's value table and its min/max/ave/stddev block into the
   * image at absolute byte offset `leafOff`. `values` holds all 512 voxel
   * values in `pnanovdb_leaf_coord_to_offset` order (inactive slots included,
   * carrying their source value for exact round-trip). `stats` is the leaf's
   * active-value accumulator.
   */
  encodeLeafValues(
    w: GridImageWriter,
    leafOff: number,
    values: Float32Array,
    stats: StatsAccumulator,
  ): void;
  /** Writes a single constant value (an internal-node tile / background). */
  writeConstant(w: GridImageWriter, off: number, value: number): void;
}

/** FLOAT leaf codec: 512 raw f32 values, f32 stats. */
export const FLOAT_LEAF_CODEC: LeafCodec = {
  layout: FLOAT_LAYOUT,
  encodeLeafValues(w, leafOff, values, stats) {
    const L = FLOAT_LAYOUT;
    for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
      w.setF32(leafOff + L.leafOffTable + n * 4, values[n]!);
    }
    // Stats block. For an all-inactive leaf (possible only under a custom
    // activeThreshold that rejects every voxel in a materialised block) the
    // accumulator is empty; write zeros, matching "no active values".
    w.setF32(leafOff + L.leafOffMin, stats.isEmpty ? 0 : stats.min);
    w.setF32(leafOff + L.leafOffMax, stats.isEmpty ? 0 : stats.max);
    w.setF32(leafOff + L.leafOffAve, stats.average);
    w.setF32(leafOff + L.leafOffStdDev, stats.stdDev);
  },
  writeConstant(w, off, value) {
    w.setF32(off, value);
  },
};
