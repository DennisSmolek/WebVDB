/**
 * Thin shim (K2.3 part A): the v1 WGSL assembly moved into
 * `legacy-material.ts`. What remains is used by `compute.ts` (K2.3 part B
 * ports it onto the core) and the legacy material. Delete this file once
 * neither imports it.
 */

/** PNanoVDB `GridType` ids for the v1 subset (`pnanovdb.wgsl`'s `PNANOVDB_GRID_TYPE_*`). */
export const GRID_TYPE_FLOAT = 1;
export const GRID_TYPE_FP8 = 14;
export const GRID_TYPE_FPN = 16;

/**
 * Throws unless `pnanovdbSource` holds the vendored library's `nanovdb_buffer`
 * read site (catches an empty/wrong string before it becomes a WGSL error).
 */
export function assertHasBufferGlobal(pnanovdbSource: string): void {
  if (!/\bnanovdb_buffer\b/.test(pnanovdbSource)) {
    throw new Error(
      "assertHasBufferGlobal: `pnanovdbSource` does not contain the `nanovdb_buffer` read site every " +
        "consumer of the vendored library must satisfy — pass the real vendored pnanovdb.wgsl text " +
        "(packages/vdb/vendor/pnanovdb.wgsl), not an empty/wrong string.",
    );
  }
}
