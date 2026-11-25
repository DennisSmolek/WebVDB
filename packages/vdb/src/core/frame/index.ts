/**
 * The core frame renderer (K6, D34): compute kernels on 3D textures on the host's device. K6.1–K6.7
 * export through `core/internal.ts` only; the public surface lands in K6.8 (design §2.1).
 */

export { vdbCreateFrameRenderer, vdbFrameRendererEmission, vdbFrameRendererEnvironment, vdbFrameRendererGrid, vdbFrameRendererScattering, vdbFrameRendererTextures } from "./renderer.js";
// K6.5: volume shadow maps, emission colour data + hooks, the emission probes and their host WGSL
export { VDBFrameShadowMaps } from "./shadows.js";
export type { VDBFrameShadowInputs } from "./shadows.js";
export { VDBFrameEmissionColors, VDBFrameEmissionProbeSet, vdbEmissionHookProblems, vdbValidateEmissionHook } from "./emission.js";
export type { VDBEmissionColorCode, VDBFrameEmissionPlan, VDBFrameEmitterInput } from "./emission.js";
export {
  VDB_FRAME_SHADOW_PIECE,
  VDB_FRAME_SHADOW_PIECE_STRIDE,
  VDB_VOLUME_SHADOW_MAX_CASCADES,
  VDB_VOLUME_SHADOW_VIEW,
  VOLUME_SHADOW_BINDINGS,
  vdbFrameVolumeShadowWgsl,
  vdbVolumeShadowProbeWgsl,
  vdbVolumeShadowWgsl,
} from "./wgsl/shadows.js";
export {
  EMISSION_DIRECTION_BINDINGS,
  EMISSION_PROBE_BINDINGS,
  EMISSION_REDUCE_BINDINGS,
  VDB_EMISSION_GRID,
  VDB_FRAME_EMISSION_DIRECTIONS,
  VDB_FRAME_EMISSION_LUT_KELVIN,
  VDB_FRAME_EMISSION_LUT_SIZE,
  VDB_FRAME_EMISSION_MAX_CASCADES,
  VDB_FRAME_EMISSION_RAYS,
  VDB_FRAME_EMITTERS,
  VDB_FRAME_EMITTERS_STRIDE,
  VDB_FRAME_MAX_EMITTERS,
  vdbEmissionIrradianceProbeWgsl,
  vdbEmissionProbeWgsl,
  vdbFrameEmissionColorWgsl,
} from "./wgsl/emission.js";
export { VDBFrameEnvironmentState, vdbFrameEnvironmentProjection } from "./environment.js";
export type { VDBFrameEnvironmentInfo } from "./environment.js";
export {
  ENV_BLUR_BINDINGS,
  ENV_GRID_BINDINGS,
  ENV_PREFILTER_BINDINGS,
  ENV_REDUCE_BINDINGS,
  vdbFrameEnvAreaWgsl,
  vdbFrameEnvBlurWgsl,
  vdbFrameEnvCubeWgsl,
  vdbFrameEnvReduceWgsl,
  vdbFrameEnvironmentGridWgsl,
} from "./wgsl/environment.js";
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
  VDBVolumeShadowMap,
  VDBVolumeShadowDrawn,
  VDBEmissionColor,
  VDBFrameEmissionLightSettings,
  VDBFrameEmissionProbes,
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
  ENV_GROUP0_LAYOUT,
  GROUP0_LAYOUT,
  SOURCE_SLOT,
  vdbFrameEnvBlurSpec,
  vdbFrameEnvPrefilterSpec,
  vdbFrameEnvReduceSpec,
  vdbFrameEnvironmentGridSpec,
  vdbFrameChunkProbeSpec,
  vdbFrameCompositeKernelWgsl,
  vdbFrameCompositeSpec,
  vdbFrameGatherKernelWgsl,
  vdbFrameGatherSpec,
  vdbFrameGridDensitySpec,
  vdbFrameGridMipSpec,
  vdbFrameResolveKernelWgsl,
  vdbFrameResolveSpec,
  vdbFrameTransmittanceSpec,
  vdbFrameVolumeShadowSpec,
  vdbFrameEmissionDirectionSpec,
  vdbFrameEmissionReduceSpec,
  vdbFrameEmissionProbeSpec,
} from "./kernels.js";
export type { VDBFrameCompositeKey, VDBFrameGatherKey, VDBFrameGridDensityKey } from "./kernels.js";
export {
  FRAME_GATHER_BINDINGS,
  GRID_DENSITY_BINDINGS,
  GRID_MIP_BINDINGS,
  TRANSMITTANCE_BINDINGS,
  TRANSMITTANCE_MLP_BINDINGS,
  TRANSMITTANCE_MULTI_SCALE_BINDINGS,
  VDB_FRAME_MLP_MAX_LAYERS,
  VDB_FRAME_MLP_MAX_WIDTH,
  VDB_FRAME_SUM_WORDS,
} from "./wgsl/lighting.js";
export type { VDBFrameScatteringKernel } from "./wgsl/lighting.js";
export { VDBFrameScattering } from "./scattering.js";
export type { VDBFrameMlp, VDBFrameMlpLayer } from "../../tools/frame/mlp.js";
export { computePipeline, renderPipeline, prepareCompute, prepareRender, vdbHashText } from "./pipelines.js";
export type { VDBCachedPipeline, VDBComputeSpec, VDBRenderSpec } from "./pipelines.js";
export { VDBRetireQueue, createFroxelTexture } from "./resources.js";
export {
  VDB_FRAME_ENV_PARAMS,
  VDB_FRAME_ENVIRONMENT,
  VDB_FRAME_GROUP,
  VDB_FRAME_GROUP_STRIDE,
  VDB_FRAME_MAX_ENVIRONMENT_LIGHTS,
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
