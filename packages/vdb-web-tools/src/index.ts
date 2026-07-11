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

/** Parse a `.vdb` container (zlib via fflate; blosc via optional codec). Phase 5. */
export function parseVdb(_buffer: ArrayBuffer): never {
  throw new Error("parseVdb: not implemented (Phase 5)");
}

/** Build a NanoVDB grid image from a parsed `.vdb` grid. Phase 5. */
export function buildFromVdb(_grid: unknown): never {
  throw new Error("buildFromVdb: not implemented (Phase 5)");
}

/** Build a NanoVDB grid image from a dense array. Phase 5 (demo 07). */
export function buildFromDense(
  _values: Float32Array,
  _dims: [number, number, number],
): never {
  throw new Error("buildFromDense: not implemented (Phase 5)");
}

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

/** Read/write `.nvdb` files. Phase 5. */
export function readNvdb(_buffer: ArrayBuffer): never {
  throw new Error("readNvdb: not implemented (Phase 5)");
}
export function writeNvdb(_grid: unknown): never {
  throw new Error("writeNvdb: not implemented (Phase 5)");
}
