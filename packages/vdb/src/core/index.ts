/// <reference types="@webgpu/types" preserve="true" />
/**
 * @pmndrs/vdb (core) - the engine-free WebGPU core: a pool of NanoVDB grids in one storage buffer,
 * the generated `vdb*` WGSL module that reads it, and compute tools that run on it.
 *
 * Everything usable without a GPU (the `.nvdb` loader, the CPU oracle, the `.vdb` tooling) lives in
 * `@pmndrs/vdb/tools`. The core never imports `three` (test/boundaries.test.ts).
 *
 * This barrel is the public API (D32). The pool's binary layout helpers and other test oracles live in
 * `./internal.ts`, imported by path by tests and harnesses; it is not a package export.
 */

export { VDBError } from "./errors.js";
export type { VDBErrorCode } from "./errors.js";

export { VDBGridPool } from "./pool/pool.js";
export type {
  VDBAddOptions,
  VDBBrickMajorant,
  VDBGridInfo,
  VDBGridPoolOptions,
  VDBTransformInput,
} from "./pool/pool.js";
/** The pool's GPU layout version, for `vdbCheckModule({ layoutVersion })`. */
export { VDB_GPU_LAYOUT_VERSION } from "./pool/layout.js";
export type { VDBVec3 } from "./pool/layout.js";
// K1.4a: WGSL module generator (docs/design/K1-core.md §3).
export { vdbCheckModule, vdbModuleEntries, vdbModuleWgsl } from "./wgsl/generate.js";
export type {
  VDBBufferModuleOptions,
  VDBModuleCommonOptions,
  VDBModuleHelper,
  VDBModuleOptions,
  VDBModulePoolLike,
  VDBModuleProfile,
  VDBModuleTypeCheck,
  VDBSlotModuleOptions,
} from "./wgsl/types.js";
export { VDB_ALL_SCALAR_TYPES, VDB_GRID_TYPE_IDS, VDB_GRID_TYPE_NAMES } from "./types.js";
export type { VDBGridTypeName, VDBSlot } from "./types.js";
// K2.3: compute tools on the pool. The GPU tools carry the `Gpu` suffix: `@pmndrs/vdb/three` wraps them
// as `vdbGridStats` / `vdbDecodeToAtlas` (renderer in, promise out) and `@pmndrs/vdb/tools` has the
// `*Cpu` versions (D32).
export { vdbApplyValueTransform, vdbDecodeToAtlasGpu, vdbGridStatsGpu, VDB_VALUE_TRANSFORM_PRESETS } from "./compute/index.js";
// K3.2a: the sparse source's coarse pyramid built on the GPU.
export { vdbCreateBoxPyramid } from "./compute/index.js";
export type { VDBBoxPyramidGpu, VDBBoxPyramidGpuOptions, VDBBoxPyramidRecordOptions } from "./compute/index.js";
export type { VDBBoxPyramidLayout, VDBPyramidAtlasLevel } from "../tools/frame/pyramid-layout.js";
export type {
  VDBAtlasFilter,
  VDBAtlasFormat,
  VDBAtlas,
  VDBAtlasReadback,
  VDBDecodeToAtlasOptions,
  VDBGridStats,
  VDBGridStatsOptions,
  VDBReadback,
  VDBValueTransform,
  VDBValueTransformPreset,
} from "./compute/index.js";
// K3.0: the volume-source seam (R1 §4.2) and its dense `.wvdb` source.
export { vdbDenseSource } from "./source/dense.js";
export type { VDBDenseSource, VDBDenseSourceOptions } from "./source/dense.js";
export type { VDBVolumeChannels, VDBVolumeSource, VDBVolumeSourceInfo, VDBVolumeSourceWgslOptions } from "./source/types.js";
