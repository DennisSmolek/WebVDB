/**
 * Types and tables shared by the pool (`pool/`) and the WGSL generator
 * (`wgsl/`). One home so the two cannot drift (K1 integration).
 */

/**
 * Grid value types the pool can hold and the generated WGSL can decode (NanoVDB
 * `GridType` ids 1, 13-16 and 9). `half` is NanoVDB's `GridType::Half` (raw IEEE
 * binary16 values, implemented in NanoVDB 13.2; K7.H).
 */
export type VDBGridTypeName = "float" | "fp4" | "fp8" | "fp16" | "fpn" | "half";

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
  half: 9,
};

/**
 * Canonical order; the generator sorts `types` into it so equal sets give equal
 * bytes. `half` came last (K7.H), so the order of the older types is unchanged.
 */
export const VDB_GRID_TYPE_NAMES: readonly VDBGridTypeName[] = ["float", "fp4", "fp8", "fp16", "fpn", "half"];

/**
 * Every scalar grid type the pool can hold, for
 * `vdbModuleWgsl({ types: VDB_ALL_SCALAR_TYPES, ... })`: a module that reads
 * any grid a host may add later, so it never has to be regenerated and its
 * pipeline rebuilt. The price is the largest module (every decoder); a host
 * that knows its types up front should list just those. Same contents and
 * order as {@link VDB_GRID_TYPE_NAMES}.
 */
export const VDB_ALL_SCALAR_TYPES: readonly VDBGridTypeName[] = Object.freeze([...VDB_GRID_TYPE_NAMES]);

/** Where a kernel binds the pool: `@group(group) @binding(firstBinding)`. Layout 1 uses one binding (D10). */
export interface VDBSlot {
  /** The bind group index. */
  readonly group: number;
  /** The binding index of the pool buffer within the group. */
  readonly firstBinding: number;
}
