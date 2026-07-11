/**
 * nanovdb-wgsl — renderer-agnostic NanoVDB traversal WGSL module + TS loader.
 *
 * The WGSL module itself is vendored in `vendor/pnanovdb.wgsl` (see
 * vendor/VENDOR.md for the fork policy). The `.nvdb` file loader
 * (`NanoVDBFile`, Phase 1 of docs/PLAN.md) lives in `./nvdb-file.ts`; this
 * module just re-exports its public surface.
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
 * Resolvable URL of the vendored WGSL module, for runtimes that fetch it
 * (Node tests, non-bundler setups). Vite/bundler consumers should import
 * `nanovdb-wgsl/pnanovdb.wgsl?raw` instead.
 */
export const pnanovdbWgslUrl = new URL("../vendor/pnanovdb.wgsl", import.meta.url);
