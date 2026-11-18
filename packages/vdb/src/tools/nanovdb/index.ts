/**
 * nanovdb/ — pure-TypeScript NanoVDB serializer (Phase 5, decision D3).
 *
 * Public surface for the orchestrator to re-export from the package root:
 *   - `buildFromDense(values, dims, opts?)` -> a complete, valid FLOAT NanoVDB
 *     grid image (flat u32 array) that the repo's proven readers accept.
 *   - `writeNvdb(images, opts?)` -> a segment-format `.nvdb` file ArrayBuffer.
 *   - `buildFromDenseDetailed(...)` -> image + computed metadata.
 *
 * Module map: bytes.ts (offsets + writer), tree.ts (dense -> sparse topology),
 * stats.ts (GridStats-style node stats), leaf-codec.ts (leaf value encoding —
 * the abstraction Fp8/FpN slot into next wave), serialize.ts (layout), and
 * write-nvdb.ts (file framing).
 */

export {
  buildFromDense,
  buildFromDenseDetailed,
  buildFromLeavesDetailed,
} from "./serialize.js";
export type { BuildFromDenseOptions, BuildFromLeavesOptions, BuiltGrid } from "./serialize.js";
export { writeNvdb } from "./write-nvdb.js";
export type { WriteNvdbOptions } from "./write-nvdb.js";
export type { LeafCodec } from "./leaf-codec.js";
export { FLOAT_LEAF_CODEC, FP8_LEAF_CODEC, makeFpNLeafCodec, FPN_DEFAULT_TOLERANCE } from "./leaf-codec.js";
export { buildFromVdb, buildFromVdbDetailed } from "./build-from-vdb.js";
export type { BuildFromVdbOptions } from "./build-from-vdb.js";
export { quantize, quantizeDetailed } from "./quantize.js";
export type { QuantizeMode } from "./quantize.js";
export { transform } from "./transform.js";
export type { TransformInput, TransformSpec } from "./transform.js";
export { inspect } from "./inspect.js";
export type { InspectReport } from "./inspect.js";
