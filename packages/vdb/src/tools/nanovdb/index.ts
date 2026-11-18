/**
 * nanovdb/ — pure-TypeScript NanoVDB serializer (Phase 5, decision D3).
 *
 * Public surface for the orchestrator to re-export from the package root:
 *   - `vdbBuildFromDense(values, dims, opts?)` -> a complete, valid FLOAT NanoVDB
 *     grid image (flat u32 array) that the repo's proven readers accept.
 *   - `writeNvdb(images, opts?)` -> a segment-format `.nvdb` file ArrayBuffer.
 *   - `vdbBuildFromDenseDetailed(...)` -> image + computed metadata.
 *
 * Module map: bytes.ts (offsets + writer), tree.ts (dense -> sparse topology),
 * stats.ts (GridStats-style node stats), leaf-codec.ts (leaf value encoding —
 * the abstraction Fp8/FpN slot into next wave), serialize.ts (layout), and
 * write-nvdb.ts (file framing).
 */

export {
  vdbBuildFromDense,
  vdbBuildFromDenseDetailed,
  vdbBuildFromLeavesDetailed,
} from "./serialize.js";
export type { VDBBuildFromDenseOptions, VDBBuildFromLeavesOptions, VDBBuiltGrid } from "./serialize.js";
export { writeNvdb } from "./write-nvdb.js";
export type { WriteNvdbOptions } from "./write-nvdb.js";
export type { VDBLeafCodec } from "./leaf-codec.js";
export {
  VDB_FLOAT_LEAF_CODEC,
  VDB_FP4_LEAF_CODEC,
  VDB_FP8_LEAF_CODEC,
  VDB_FP16_LEAF_CODEC,
  vdbMakeFpNLeafCodec,
  FPN_DEFAULT_TOLERANCE,
  VDB_VEC3F_LEAF_CODEC,
} from "./leaf-codec.js";
export { buildFromVdb, buildFromVdbDetailed } from "./build-from-vdb.js";
export type { BuildFromVdbOptions } from "./build-from-vdb.js";
export { vdbCollapseGrid, vdbQuantize, vdbQuantizeDetailed } from "./quantize.js";
export type { VDBQuantizeMode, VDBQuantizeOptions } from "./quantize.js";
export { vdbResampleGrid, vdbResampleError, vdbResampleDense } from "./resample.js";
export type {
  VDBResampleErrorOptions,
  VDBResampleErrorResult,
  VDBResampledGrid,
  VDBResampleFilter,
  VDBResampleOptions,
  VDBResampleDenseOptions,
  VDBResampleSource,
} from "./resample.js";
export { vdbTransform } from "./transform.js";
export type { VDBGridTransformInput, VDBTransformSpec } from "./transform.js";
export { vdbBakeCoarseFirst } from "./coarse-first.js";
export type { VDBBakeCoarseFirstOptions } from "./coarse-first.js";
export { vdbInspect } from "./inspect.js";
export type { VDBInspectReport } from "./inspect.js";
