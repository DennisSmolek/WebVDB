/**
 * vdb-web-tools — the CPU half, pure TypeScript first (decision D3).
 *
 * Phase 0 stub: API surface only. The `.vdb` parser, NanoVDB serializer,
 * quantization, affine transforms, and `.nvdb` I/O are Phase 5
 * deliverables, validated byte/value-wise against official
 * `nanovdb_convert` output on the fixture corpus (docs/SPEC.md §4).
 * WASM is an optional escalation rung (W1), never the foundation.
 */

export interface InspectReport {
  gridType: string;
  gridClass: string;
  voxelCount: number;
  nodeCounts: { upper: number; lower: number; leaf: number };
  memoryBreakdown: Record<string, number>;
}

export type { VdbBBox, VdbFile, VdbGrid, VdbLeaf, VdbTransformInfo } from "./vdb/index.js";
export { VdbFormatError, VdbUnsupportedError } from "./vdb/index.js";

/**
 * Parse a `.vdb` container: FloatGrid, 5-4-3 tree, uncompressed/zlib
 * (blosc raises a clear "not supported" error, per SPEC §4). Phase 5 —
 * see `src/vdb/` for the reader implementation.
 */
export { parseVdb } from "./vdb/index.js";

/** Build a NanoVDB grid image from a parsed `.vdb` grid. Phase 5. */
export function buildFromVdb(_grid: unknown): never {
  throw new Error("buildFromVdb: not implemented (Phase 5)");
}

/**
 * NanoVDB serializer (Phase 5b): build FLOAT grid images from dense
 * arrays and write `.nvdb` segment files — see `src/nanovdb/` for the
 * layout/stats/validation details and the LeafCodec seam Fp8/FpN slot
 * into next.
 */
export { buildFromDense, buildFromDenseDetailed, FLOAT_LEAF_CODEC } from "./nanovdb/index.js";
export type { BuildFromDenseOptions, BuiltGrid, LeafCodec } from "./nanovdb/index.js";

/** Quantize a grid to Fp8/FpN. Phase 5. */
export function quantize(_grid: unknown, _mode: "fp8" | "fpn", _tolerance?: number): never {
  throw new Error("quantize: not implemented (Phase 5)");
}

/** Affine transform — a metadata-only Map edit, no voxel churn. Phase 5. */
export function transform(_grid: unknown, _matrix: Float32Array): never {
  throw new Error("transform: not implemented (Phase 5)");
}

/** Tree stats, per-level counts, memory breakdown. Phase 5 (explorer). */
export function inspect(_grid: unknown): InspectReport {
  throw new Error("inspect: not implemented (Phase 5)");
}

export { writeNvdb } from "./nanovdb/index.js";
export type { WriteNvdbOptions } from "./nanovdb/index.js";

/** Read `.nvdb` files. Phase 5 follow-up (the `nanovdb-wgsl` package's NanoVDBFile covers reading today). */
export function readNvdb(_buffer: ArrayBuffer): never {
  throw new Error("readNvdb: not implemented (Phase 5)");
}
