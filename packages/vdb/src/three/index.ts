/**
 * @pmndrs/vdb/three — the three.js adapter over the engine-free core (D8).
 *
 * `getDevice` / `getGPUTexture` reach the renderer's WebGPU handles, so the
 * pool and the other pmndrs tools share one device. `VDBGridNode` puts a
 * `VDBGridPool` into three's node system (`vdbSample`, `vdbReadValue`, or the
 * module as a `wgslFn` include). `VDBVolumeMaterial` raymarches one pooled
 * grid; `VDBSequence` plays a frame sequence into one pool id with
 * `pool.replace`. `vdbGridStats` / `vdbDecodeToAtlas` run the core compute
 * tools through the renderer's device.
 *
 * There is no renderer bootstrap: create the device yourself
 * (`adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) })`)
 * and pass it, `new WebGPURenderer({ device })`, so the pool outlives
 * `renderer.dispose()`.
 */

// K2.1/K2.2: the adapter over the engine-free core (docs/design/K1-core.md §6).
export { getDevice, getGPUTexture } from "./device.js";
export type { ThreeBackendLike, ThreeRendererLike } from "./device.js";
export { VDBGridNode, vdbReadValue, vdbSample } from "./grid-node.js";
export type { VDBGridNodeOptions, VDBGridNodePoolLike, VDBSampleOptions, VDBTslNode } from "./grid-node.js";

// K2.3: the volume material and the sequence player on the pool.
export {
  VDBVolumeMaterial,
  vdbProxyGeometry,
  VDB_VOLUME_MAX_STEPS_CAP,
  VDB_VOLUME_SHADOW_STEPS_CAP,
  VDB_VOLUME_SAMPLE_BUDGET_CAP,
} from "./material.js";
export type { VDBVolumeMaterialParameters, VDBProxyGridInfo, VDBUintUniform } from "./material.js";
export { VDBSequence } from "./sequence.js";

// K3.7 (R1 §6 Step 7): level-set surfaces traced on the sparse pool, and point grids as sprites.
export { VDBLevelSetMaterial, VDB_LEVEL_SET_SAMPLES_CAP } from "./level-set-material.js";
export type { VDBLevelSetMaterialParameters } from "./level-set-material.js";
export { VDBPoints, vdbAttenuatedPointSize, vdbPointDiscTexture } from "./points.js";
export type { VDBPointsData, VDBPointsOptions } from "./points.js";
export type {
  VDBSequenceOptions,
  VDBSequenceStats,
  VDBSequenceTarget,
  VDBSequencePoolLike,
  VDBFrameLoader,
} from "./sequence.js";

// K2.3: compute wrappers over the core tools (renderer in, promise out). The
// core's `gridStats` / `decodeToAtlas` take a device and an encoder; the value
// transform is the core's `applyValueTransform` (CPU).
export { vdbGridStats, vdbDecodeToAtlas } from "./compute.js";
export type { VDBThreeGridStatsOptions, VDBGridStatsResult, VDBThreeDecodeToAtlasOptions, VDBDecodeToAtlasResult } from "./compute.js";

// K3.0: the dense `.wvdb` volume source (R1 §4.2) in his createVolumeNodes() shape.
export { vdbDenseTextures, vdbVolumeUniforms, vdbVolumeNodes, vdbMinComponent, vdbMaxComponent } from "./volume-source.js";
export type {
  VDBDenseChannelTextures,
  VDBThreeDenseSource,
  VDBVolumeNodes,
  VDBVolumeNodesOptions,
  VDBVolumeUniformParameters,
  VDBVolumeUniforms,
  VDBThreeVolumeSource,
  VDBThreeSourceFactory,
} from "./volume-source.js";

// K3.2: the sparse volume source (a VDBGridPool grid behind the same seam) with its LOD strategies.
export { vdbSparseSource } from "./sparse-source.js";
export type {
  VDBSparseLod,
  VDBSparseSourceGrids,
  VDBSparseSourceOptions,
  VDBThreeSparseSource,
  VDBSparseVolumeUniforms,
  VDBSparsePyramid,
  VDBIndexBox,
} from "./sparse-source.js";

// K4.16: coarse-first `.nvdb` streaming into one pool id (add the preview with
// the full size reserved, then replace). Engine-free glue; here because tools
// may not import the core.
export { vdbStreamCoarseFirst } from "./coarse-first.js";
export type {
  VDBCoarseFirstEvent,
  VDBCoarseFirstOptions,
  VDBCoarseFirstPoolLike,
  VDBCoarseFirstResult,
  VDBCoarseFirstSource,
} from "./coarse-first.js";

// K3.1: the froxel renderer with 'volume' lighting (his FrameVolume / FrameRenderer) on a volume source.
export {
  VDBFrameVolume,
  VDBFrameRenderer,
  vdbFrameRenderer,
  VDBFrameInjectMaterial,
  VDBFrameIntegrateMaterial,
  VDBFrameShadowDensityMaterial,
  VDBFrameTransmittanceMaterial,
  VDBFrameDensityMaterial,
  VDBFrameLightingMaterial,
  VDBFrameTemporalMaterial,
  VDBFrameEnvironmentMapMaterial,
  VDBFrameEnvironmentGridMaterial,
  VDBFrameGridMipMaterial,
  vdbFrameLights,
  vdbFrameAllLights,
  vdbFrameChunks,
  // His helper names stay inside frame/ (diffable with upstream); the public names follow D17.
  getFramePhase as vdbFramePhase,
  getFramePhaseSpectrum as vdbFramePhaseSpectrum,
  getFramePhaseAsymmetry as vdbFramePhaseAsymmetry,
  getFrameMultipleScattering as vdbFrameMultipleScattering,
  getFrameScatteringOctaves as vdbFrameScatteringOctavesInto,
  getFrameTwoSidedDiffusion as vdbFrameTwoSidedMode,
  getAtlasLayout as vdbFrameAtlasLayout,
  getFroxelSize as vdbFroxelSize,
  getFroxelSlice as vdbFroxelSlice,
  getFroxelSliceRange as vdbFroxelSliceRange,
  getFroxelRect as vdbFroxelRect,
  getFroxelArea as vdbFroxelArea,
} from "./frame/index.js";
export type {
  VDBFrameVolumeProps,
  VDBFrameRendererSettings,
  VDBFramePassHook,
  VDBFramePass,
  VDBFrameRedrawStats,
  VDBFrameInjectMaterialOptions,
  VDBFramePhase,
  VDBFrameMultipleScattering,
  FrameVolumeRecord,
  FroxelArea,
} from "./frame/index.js";
