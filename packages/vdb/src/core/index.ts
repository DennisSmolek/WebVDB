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

export { VDBGridPool } from "./pool/pool.js";
export type {
  VDBAddOptions,
  VDBBrickMajorant,
  VDBGridInfo,
  VDBGridPoolOptions,
  VDBSlot,
  VDBTransformInput,
} from "./pool/pool.js";
export type { VDBGridTypeName } from "./pool/image-header.js";
export {
  VDB_GPU_LAYOUT_VERSION,
  VDB_HEADER_WORD,
  VDB_POOL_ALIGN,
  VDB_POOL_HEADER_BYTES,
  VDB_POOL_MAGIC,
  VDB_POOL_RECORD_BYTES,
  VDB_POOL_RECORD_WORDS,
  VDB_POOL_TABLE_OFFSET,
  VDB_RECORD_FLAG,
  VDB_RECORD_WORD,
  decodeHeader,
  decodeRecord,
  encodeRecord,
  vdbPoolLayoutWgsl,
} from "./pool/layout.js";
export type { VDBGridRecord, VDBPoolHeader, VDBVec3 } from "./pool/layout.js";
