/**
 * @pmndrs/vdb core internals (D32): NOT a package export.
 *
 * Everything public (`./index.ts`) plus what tests, harnesses and scripts need that a user doesn't:
 * the pool's binary layout (constants, header/record codecs, the layout WGSL), the module banner, the
 * pyramid kernel's WGSL and the vendored module's URL. Import it by path (`src/core/internal.ts`, or
 * `dist/core/internal.js` from a script after `pnpm build`); its names can change without notice.
 */

export * from "./index.js";

/** Resolvable URL of the vendored WGSL module (`@pmndrs/vdb/pnanovdb.wgsl`), for Node tests. */
export const pnanovdbWgslUrl = new URL("../../vendor/pnanovdb.wgsl", import.meta.url);

export {
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
export type { VDBGridRecord, VDBPoolHeader } from "./pool/layout.js";
export { VDB_MODULE_BANNER } from "./wgsl/generate.js";
export { boxPyramidWgsl } from "./compute/index.js";
// K6.1: the group-2 source contract (design §2.1); public in K6.8.
export type { VDBVolumeSlot } from "./source/types.js";
// K6.1: the core frame renderer (public in K6.8).
export * from "./frame/index.js";
