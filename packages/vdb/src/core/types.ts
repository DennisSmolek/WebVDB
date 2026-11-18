/**
 * Types and tables shared by the pool (`pool/`) and the WGSL generator
 * (`wgsl/`). One home so the two cannot drift (K1 integration).
 */

/** Grid value types the pool can hold and the generated WGSL can decode (NanoVDB `GridType` ids 1, 13-16). */
export type VDBGridTypeName = "float" | "fp4" | "fp8" | "fp16" | "fpn";

/**
 * NanoVDB `GridType` id per name (`PNANOVDB_GRID_TYPE_*` in the vendored module).
 * `test/core/types.test.ts` checks it against the vendored stride tables, the
 * vendored WGSL constants and `tools`.
 */
export const VDB_GRID_TYPE_IDS: Readonly<Record<VDBGridTypeName, number>> = {
  float: 1,
  fp4: 13,
  fp8: 14,
  fp16: 15,
  fpn: 16,
};

/** Canonical order; the generator sorts `types` into it so equal sets give equal bytes. */
export const VDB_GRID_TYPE_NAMES: readonly VDBGridTypeName[] = ["float", "fp4", "fp8", "fp16", "fpn"];

/** Where a kernel binds the pool: `@group(group) @binding(firstBinding)`. Layout 1 uses one binding (D10). */
export interface VDBSlot {
  readonly group: number;
  readonly firstBinding: number;
}
