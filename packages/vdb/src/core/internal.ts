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
export {
  VDB_FRAME_MULTI_SCALE_PARAMS_BYTES,
  VDB_FRAME_MULTI_SCALE_SAMPLER,
  multiScaleMipsWgsl,
  vdbCreateMultiScaleMips,
  vdbFrameMultiScaleParams,
  vdbFrameMultiScaleWgsl,
} from "./compute/multi-scale.js";
export type {
  VDBFrameMultiScaleParams,
  VDBFrameMultiScaleWgslOptions,
  VDBMultiScaleFormat,
  VDBMultiScaleMips,
  VDBMultiScaleMipsOptions,
  VDBMultiScaleRecordOptions,
  VDBMultiScaleResizeOptions,
} from "./compute/multi-scale.js";
// K6.7: the core sparse source (public in K6.8).
export { vdbSparseSourceGpu } from "./source/sparse.js";
export type {
  VDBSparseIndexBox,
  VDBSparseLodGpu,
  VDBSparseSourceGpu,
  VDBSparseSourceGpuOptions,
  VDBSparseSourceUpdate,
  VDBSparseVolumeGpu,
  VDBSparseVolumeGpuOptions,
  VDBSparseVolumeGrids,
} from "./source/sparse.js";
export type { VDBVolumeSlot } from "./source/types.js";
export {
  VDB_SPARSE_RECORD_BYTES,
  VDB_SPARSE_RECORD_FIELDS,
  SPARSE_FLAG,
  packSparseRecord,
  sparseBindings,
  sparseRecordOffset,
  sparseRecordStructWgsl,
  sparseSourceWgsl,
} from "./source/sparse-wgsl.js";
// K6.R: the reference renderer (brute-force path tracer over a volume source; internal only).
export * from "./reference/index.js";
