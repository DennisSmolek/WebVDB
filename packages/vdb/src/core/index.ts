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
