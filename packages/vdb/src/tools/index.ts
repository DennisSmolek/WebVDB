/**
 * @pmndrs/vdb/tools — the CPU half, pure TypeScript first (decision D3).
 *
 * Phase 0 stub: API surface only. The `.vdb` parser, NanoVDB serializer,
 * quantization, affine transforms, and `.nvdb` I/O are Phase 5
 * deliverables, validated byte/value-wise against official
 * `nanovdb_convert` output on the fixture corpus (docs/SPEC.md §4).
 * WASM is an optional escalation rung (W1), never the foundation.
 */

export type { VdbBBox, VdbFile, VdbGrid, VdbLeaf, VdbTile, VdbTransformInfo } from "./vdb/index.js";
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

/** Read `.nvdb` files. Phase 5 follow-up (`NanoVDBFile` in this module covers reading today). */
export function readNvdb(_buffer: ArrayBuffer): never {
  throw new Error("readNvdb: not implemented (Phase 5)");
}

/**
 * `.nvdb` file loader (from the former `nanovdb-wgsl` package).
 */
export {
  FILE_HEADER_SIZE,
  FILE_METADATA_SIZE,
  GRID_DATA_SIZE,
  MAGIC_ASCII,
  Codec,
  SUPPORTED_GRID_TYPES,
  NanoVDBFile,
} from "./nvdb-file.js";
export type { GridMetadata } from "./nvdb-file.js";

/**
 * Pure-TypeScript CPU reference for NanoVDB voxel lookup/sampling (Phase 1/2,
 * docs/PLAN.md) — a byte-for-byte transliteration of the vendored
 * `PNanoVDB.h` root -> upper -> lower -> leaf descent, restricted to grid
 * types FLOAT/FP8/FPN. Reads layout constants from a baked, browser-safe
 * generated module (`./cpu/stride-tables.generated.ts`), so these are safe to
 * import from any Vite/browser page as well as Node.
 */
export { readValue } from "./cpu/read-value.js";
export type { ReadResult } from "./cpu/read-value.js";
export { sampleNearest, sampleTrilinear } from "./cpu/sample-trilinear.js";
export type { SampleOptions } from "./cpu/sample-trilinear.js";

/**
 * CPU twins of the `march` WGSL profile (K1.4b, D11): the vendored HDDA, the
 * node size at a voxel, node stats (with the Fp leaf-stats fix), the
 * `vdbNodeMax` bound, the ray clip and the canonical HDDA walk.
 */
export {
  defaultValueMax,
  getDim,
  gridHasStats,
  hddaInit,
  hddaStep,
  hddaUpdate,
  hddaWalk,
  indexBBox,
  isActive,
  leafVoxels,
  nodeMax,
  nodeStats,
  rayClip,
  readValueF32,
  treeNodes,
} from "./cpu/march.js";
export type { Hdda, HddaSegment, HddaWalkOptions, NodeLevel, NodeMaxOptions, NodeStats, TreeNode, Vec3 } from "./cpu/march.js";
export { probeCoords, probePoints } from "./cpu/probe-coords.js";
export type { ProbeOpts } from "./cpu/probe-coords.js";

/**
 * Layout-constant accessors underlying `readValue` — exposed for callers
 * that need to walk NanoVDB tree structure generically (any grid type, not
 * just FLOAT/FP8/FPN) without decoding leaf values themselves, e.g.
 * enumerating LEAF node origins for a visualization. See `./cpu/
 * stride-tables.ts` for the full accessor surface and its data source.
 */
export { defineNumber, gridTypeConstantsForId } from "./cpu/stride-tables.js";
export type { GridTypeConstants } from "./cpu/stride-tables.js";
