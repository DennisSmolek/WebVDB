/**
 * three-nanovdb — TSL/three.js layer over the nanovdb-wgsl core.
 *
 * `NanoVDBGrid` (grid image -> StorageBufferAttribute + metadata -> Box3/
 * Matrix4/proxy geometry, SPEC §3.1) and `createVolumeRenderer` (device-first
 * WebGPURenderer bootstrap, decision D4, SPEC §3.4) are the Phase 3 core.
 * `NanoVDBVolumeMaterial` (the fragment-raymarch cloud material, SPEC §3.2),
 * exported below, is built on top of this surface.
 */

export { NanoVDBGrid } from "./grid.js";
export type { NanoVDBGridOptions } from "./grid.js";

export { createVolumeRenderer, computeRequiredLimits, nextPow2 } from "./renderer.js";
export type { VolumeRendererOptions, CapabilityReport, RequiredLimits } from "./renderer.js";

export { NanoVDBVolumeMaterial } from "./material.js";
export type { NanoVDBVolumeMaterialParameters } from "./material.js";

export { NanoVDBSequence } from "./sequence.js";
export type {
  NanoVDBSequenceOptions,
  NanoVDBSequenceStats,
  SequenceTarget,
  FrameLoader,
} from "./sequence.js";

export {
  assembleVolumeWgsl,
  rewriteBufferGlobal,
  samplerForGridType,
  assertHasBufferGlobal,
  GRID_TYPE_FLOAT,
  GRID_TYPE_FP8,
  GRID_TYPE_FPN,
  DEFAULT_BUFFER_NAME,
  DEFAULT_ENTRY_NAME,
} from "./wgsl.js";
export type { VolumeWgslOptions, AssembledVolumeWgsl } from "./wgsl.js";

export {
  gridStats,
  valueTransform,
  decodeToAtlas,
  buildComputeShaderSource,
  buildValueTransformShaderSource,
  resolveTransformBody,
  computeAtlasDims,
  bboxSize,
  VALUE_TRANSFORM_PRESETS,
} from "./compute.js";
export type {
  GridStatsOptions,
  GridStatsResult,
  ValueTransformResult,
  DecodeToAtlasOptions,
  DecodeToAtlasResult,
  AtlasFilter,
  AtlasFormat,
  IndexBBox,
} from "./compute.js";
