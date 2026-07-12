/**
 * vdb-web-tools — the CPU half, pure TypeScript first (decision D3).
 *
 * Phase 0 stub: API surface only. The `.vdb` parser, NanoVDB serializer,
 * quantization, affine transforms, and `.nvdb` I/O are Phase 5
 * deliverables, validated byte/value-wise against official
 * `nanovdb_convert` output on the fixture corpus (docs/SPEC.md §4).
 * WASM is an optional escalation rung (W1), never the foundation.
 */

export type { VdbBBox, VdbFile, VdbGrid, VdbLeaf, VdbTransformInfo } from "./vdb/index.js";
export { VdbFormatError, VdbUnsupportedError } from "./vdb/index.js";

/**
 * Parse a `.vdb` container: FloatGrid, 5-4-3 tree, uncompressed/zlib
 * (blosc raises a clear "not supported" error, per SPEC §4). Phase 5 —
 * see `src/vdb/` for the reader implementation.
 */
export { parseVdb } from "./vdb/index.js";

/**
 * Build a NanoVDB FLOAT grid image from a parsed `.vdb` grid — streams the
 * parser's leaves into the serializer (no dense array), carrying the `.vdb`
 * uniform scale+translate transform into the NanoVDB Map.
 */
export { buildFromVdb, buildFromVdbDetailed } from "./nanovdb/index.js";
export type { BuildFromVdbOptions } from "./nanovdb/index.js";

/**
 * NanoVDB serializer (Phase 5b): build FLOAT grid images from dense
 * arrays and write `.nvdb` segment files — see `src/nanovdb/` for the
 * layout/stats/validation details and the LeafCodec seam Fp8/FpN slot into.
 */
export {
  buildFromDense,
  buildFromDenseDetailed,
  buildFromLeavesDetailed,
  FLOAT_LEAF_CODEC,
  FP8_LEAF_CODEC,
  makeFpNLeafCodec,
} from "./nanovdb/index.js";
export type {
  BuildFromDenseOptions,
  BuildFromLeavesOptions,
  BuiltGrid,
  LeafCodec,
} from "./nanovdb/index.js";

/** Quantize a FLOAT grid image to Fp8/FpN (per-leaf min/quantum + packed codes). */
export { quantize, quantizeDetailed } from "./nanovdb/index.js";
export type { QuantizeMode } from "./nanovdb/index.js";

/** Affine transform — a metadata-only Map edit, no voxel churn (uniform scale+translate). */
export { transform } from "./nanovdb/index.js";
export type { TransformInput, TransformSpec } from "./nanovdb/index.js";

/** Tree stats, per-level counts, memory breakdown. */
export { inspect } from "./nanovdb/index.js";
export type { InspectReport } from "./nanovdb/index.js";

export { writeNvdb } from "./nanovdb/index.js";
export type { WriteNvdbOptions } from "./nanovdb/index.js";

/** Read `.nvdb` files. Phase 5 follow-up (the `nanovdb-wgsl` package's NanoVDBFile covers reading today). */
export function readNvdb(_buffer: ArrayBuffer): never {
  throw new Error("readNvdb: not implemented (Phase 5)");
}
