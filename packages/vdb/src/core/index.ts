/**
 * @pmndrs/vdb (core) - renderer-agnostic NanoVDB traversal WGSL module.
 *
 * The WGSL module itself is vendored in `vendor/pnanovdb.wgsl` (see
 * vendor/VENDOR.md for the fork policy). Everything usable without a GPU
 * (the `.nvdb` loader, the CPU oracle, the `.vdb` tooling) lives in
 * `@pmndrs/vdb/tools`. The core never imports `three` (test/boundaries.test.ts).
 */

/**
 * Resolvable URL of the vendored WGSL module, for runtimes that fetch it
 * (Node tests, non-bundler setups). Vite/bundler consumers should import
 * `@pmndrs/vdb/pnanovdb.wgsl?raw` instead.
 */
export const pnanovdbWgslUrl = new URL("../../vendor/pnanovdb.wgsl", import.meta.url);

export { VDBError } from "./errors.js";
export type { VDBErrorCode } from "./errors.js";

// K1.4a: WGSL module generator (docs/design/K1-core.md §3).
export { checkModule, vdbModuleEntries, vdbModuleWgsl, VDB_MODULE_BANNER } from "./wgsl/generate.js";
export { VDB_GRID_TYPE_IDS, VDB_GRID_TYPE_NAMES } from "./wgsl/types.js";
export type { VDBGridTypeName, VDBModuleOptions, VDBModulePoolLike, VDBModuleProfile, VDBSlot } from "./wgsl/types.js";
