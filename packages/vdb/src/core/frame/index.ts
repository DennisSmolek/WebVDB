/**
 * The core frame renderer (K6, D34): compute kernels on 3D textures on the host's device. K6.1–K6.7
 * export through `core/internal.ts` only; the public surface lands in K6.8 (design §2.1).
 */

export { vdbCreateFrameRenderer, vdbFrameRendererGrid, vdbFrameRendererTextures } from "./renderer.js";
export type { VDBFrameRendererGpu, VDBFrameRendererGpuOptions } from "./renderer.js";
export { vdbFrameDefaultSettings } from "./settings.js";
export type { VDBFrameSettings, VDBFramePhaseName, VDBFrameMultipleScatteringMode } from "./settings.js";
export { VDB_FRAME_ALL_LIGHTS, VDB_FRAME_LIGHT_BITS, vdbFrameDefaultVolumeParams } from "./types.js";
export type {
  VDBFrameCamera,
  VDBFrameEnvironment,
  VDBFrameInput,
  VDBFrameLight,
  VDBFrameMat4,
  VDBFrameOccluder,
  VDBFrameOutputs,
  VDBFrameRedrawStats,
  VDBFrameTargets,
  VDBFrameUpdate,
  VDBFrameVec3,
  VDBFrameVolumeGpu,
  VDBFrameVolumeParams,
  VDBFrameWrap,
  VDBFroxelLayout,
  VDBVolumeShadowRequest,
} from "./types.js";
export {
  VDB_FRAME_COMPOSITE_COLOR_FN_WGSL,
  VDB_FRAME_COMPOSITE_COLOR_WGSL,
  VDB_FRAME_SRGB_EOTF_WGSL,
  VDB_FRAME_SRGB_OETF_WGSL,
  VDB_FRAME_INTEGRATE_FUNCTIONS,
  VDB_FRAME_INTEGRATE_TAPS_WGSL,
  VDB_FRAME_SLICE_COORD_WGSL,
  vdbFrameCompositeWgsl,
  vdbFrameIntegrateEntries,
  vdbFrameIntegrateWgsl,
} from "./wgsl/integrate.js";
export type { VDBFrameIntegrateWgslOptions } from "./wgsl/integrate.js";
export { frameChunksWgsl } from "./wgsl/chunks.js";
export type { VDBFrameChunk } from "./wgsl/chunks.js";
export {
  CHUNK_PROBE_INPUTS,
  CHUNK_PROBE_OUTPUTS,
  GROUP0_LAYOUT,
  SOURCE_SLOT,
  vdbFrameChunkProbeSpec,
  vdbFrameCompositeKernelWgsl,
  vdbFrameCompositeSpec,
  vdbFrameGatherKernelWgsl,
  vdbFrameGatherSpec,
  vdbFrameGridDensitySpec,
  vdbFrameResolveKernelWgsl,
  vdbFrameResolveSpec,
  vdbFrameTransmittanceSpec,
} from "./kernels.js";
export type { VDBFrameCompositeKey, VDBFrameGatherKey, VDBFrameGridDensityKey } from "./kernels.js";
export { FRAME_GATHER_BINDINGS, GRID_DENSITY_BINDINGS, TRANSMITTANCE_BINDINGS, VDB_FRAME_SUM_WORDS } from "./wgsl/lighting.js";
export { computePipeline, renderPipeline, prepareCompute, prepareRender, vdbHashText } from "./pipelines.js";
export type { VDBCachedPipeline, VDBComputeSpec, VDBRenderSpec } from "./pipelines.js";
export { VDBRetireQueue, createFroxelTexture } from "./resources.js";
export {
  VDB_FRAME_GROUP,
  VDB_FRAME_GROUP_STRIDE,
  VDB_FRAME_LIGHTS,
  VDB_FRAME_MAX_CASCADES,
  VDB_FRAME_MAX_HEMISPHERE_LIGHTS,
  VDB_FRAME_SHADOW_SLOTS,
  VDB_FRAME_MAX_LIGHTS,
  VDB_FRAME_MAX_VOLUMES,
  VDB_FRAME_UNIFORMS,
  VDB_FRAME_VOLUME,
  VDB_FRAME_VOLUME_BYTES,
  VDBUniformWriter,
  vdbFrameUniformsWgsl,
  vdbUniformStruct,
} from "./uniforms.js";
export type { VDBUniformField, VDBUniformSlot, VDBUniformStruct, VDBUniformType } from "./uniforms.js";
export { vdbFrameMultipleScatteringValue, vdbFramePhaseSpectrum, vdbFramePhaseVector } from "./phase.js";
export { GATHER_BINDINGS } from "./wgsl/gather.js";
export { RESOLVE_BINDINGS } from "./wgsl/resolve.js";
