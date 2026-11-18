/**
 * @pmndrs/vdb/three — TSL/three.js layer over the core + tools.
 *
 * `NanoVDBGrid` (grid image -> StorageBufferAttribute + metadata -> Box3/
 * Matrix4/proxy geometry, SPEC §3.1) and `createVolumeRenderer` (device-first
 * WebGPURenderer bootstrap, decision D4, SPEC §3.4) are the Phase 3 core.
 * `NanoVDBVolumeMaterial` (the fragment-raymarch cloud material, SPEC §3.2),
 * exported below, is built on top of this surface.
 */

// K2.1/K2.2: the adapter over the engine-free core (docs/design/K1-core.md §6).
export { getDevice, getGPUTexture } from "./device.js";
export type { ThreeBackendLike, ThreeRendererLike } from "./device.js";
export { VDBGridNode, vdbReadValue, vdbSample } from "./grid-node.js";
export type { VDBGridNodeOptions, VDBGridNodePoolLike, VDBSampleOptions, VDBTslNode } from "./grid-node.js";

// v1 (ported onto VDBGridNode in K2.3).
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

// K2.3: compute wrappers over the core tools (renderer in, promise out).
export { gridStats, decodeToAtlas, valueTransform } from "./compute.js";
export type { GridStatsOptions, GridStatsResult, DecodeToAtlasOptions, DecodeToAtlasResult } from "./compute.js";
