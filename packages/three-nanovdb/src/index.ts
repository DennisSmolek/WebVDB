/**
 * three-nanovdb — TSL/three.js layer over the nanovdb-wgsl core.
 *
 * `NanoVDBGrid` (grid image -> StorageBufferAttribute + metadata -> Box3/
 * Matrix4/proxy geometry, SPEC §3.1) and `createVolumeRenderer` (device-first
 * WebGPURenderer bootstrap, decision D4, SPEC §3.4) are the Phase 3 core.
 * `NanoVDBVolumeMaterial` (the fragment-raymarch cloud material, SPEC §3.2)
 * is a separate, later piece of work built on top of this surface.
 */

export { NanoVDBGrid } from "./grid.js";
export type { NanoVDBGridOptions } from "./grid.js";

export { createVolumeRenderer, computeRequiredLimits, nextPow2 } from "./renderer.js";
export type { VolumeRendererOptions, CapabilityReport, RequiredLimits } from "./renderer.js";

export { NanoVDBVolumeMaterial } from "./material.js";
export type { NanoVDBVolumeMaterialParameters } from "./material.js";

export {
  assembleVolumeWgsl,
  rewriteBufferGlobal,
  samplerForGridType,
  GRID_TYPE_FLOAT,
  GRID_TYPE_FP8,
  GRID_TYPE_FPN,
  DEFAULT_BUFFER_NAME,
  DEFAULT_ENTRY_NAME,
} from "./wgsl.js";
export type { VolumeWgslOptions, AssembledVolumeWgsl } from "./wgsl.js";
