/**
 * @pmndrs/vdb/three — the three.js adapter over the engine-free core (D8).
 *
 * `getDevice` / `getGPUTexture` reach the renderer's WebGPU handles, so the
 * pool and the other pmndrs tools share one device. `VDBGridNode` puts a
 * `VDBGridPool` into three's node system (`vdbSample`, `vdbReadValue`, or the
 * module as a `wgslFn` include). `VDBVolumeMaterial` raymarches one pooled
 * grid; `VDBSequence` plays a frame sequence into one pool id with
 * `pool.replace`. `vdbGridStats` / `vdbDecodeToAtlas` run the core compute
 * tools through the renderer's device. `VDBFrameVolume` / `VDBFrameRenderer`
 * are mjurczyk's froxel renderer on a dense or sparse volume source.
 *
 * There is no renderer bootstrap: create the device yourself
 * (`adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) })`)
 * and pass it, `new WebGPURenderer({ device })`, so the pool outlives
 * `renderer.dispose()`.
 *
 * This barrel is the public API (D32). The frame renderer's pass materials, shader chunks and three
 * internals hooks live in `./internal.ts`, imported by path by tests and harnesses; it is not a package
 * export.
 */

// K2.1/K2.2: the adapter over the engine-free core (docs/dev/design/K1-core.md §6). `getDevice` and
// `getGPUTexture` keep the helper names the other pmndrs tools use (D32).
export { getDevice, getGPUTexture } from "./device.js";
export type { VDBThreeBackendLike, VDBThreeRendererLike } from "./device.js";
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
export type {
  VDBSequenceLoopMode,
  VDBSequenceOptions,
  VDBSequenceRange,
  VDBSequenceStats,
  VDBSequenceTarget,
  VDBSequencePoolLike,
  VDBFrameLoader,
} from "./sequence.js";

// K3.7 (R1 §6 Step 7): level-set surfaces traced on the sparse pool, and point grids as sprites.
export { VDBLevelSetMaterial, VDB_LEVEL_SET_SAMPLES_CAP } from "./level-set-material.js";
export type { VDBLevelSetMaterialParameters } from "./level-set-material.js";
export { VDBPoints } from "./points.js";
export type { VDBPointsData, VDBPointsOptions } from "./points.js";

// K2.3: compute wrappers over the core tools (renderer in, promise out). The
// core's `vdbGridStatsGpu` / `vdbDecodeToAtlasGpu` take a device and an encoder;
// the value transform is the core's `vdbApplyValueTransform` (CPU).
export { vdbGridStats, vdbDecodeToAtlas } from "./compute.js";
export type { VDBThreeGridStatsOptions, VDBGridStatsResult, VDBThreeDecodeToAtlasOptions, VDBDecodeToAtlasResult } from "./compute.js";

// K3.0: the dense `.wvdb` volume source (R1 §4.2) in his createVolumeNodes() shape.
export { vdbDenseTextures, vdbVolumeUniforms, vdbVolumeNodes } from "./volume-source.js";
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
export { vdbCoarseFirstApply, vdbStreamCoarseFirst } from "./coarse-first.js";
export type {
  VDBCoarseFirstBind,
  VDBCoarseFirstBindPoolLike,
  VDBCoarseFirstEvent,
  VDBCoarseFirstOptions,
  VDBCoarseFirstPoolLike,
  VDBCoarseFirstResult,
  VDBCoarseFirstSource,
} from "./coarse-first.js";

// K3.1–K3.6: the froxel renderer (his FrameVolume / FrameRenderer) on a volume source: the volume, the
// renderer, the light-mask flags, the measured media, the emission light and its blackbody colour hook.
export {
  VDBFrameVolume,
  VDBFrameRenderer,
  vdbFrameRenderer,
  vdbFrameLights,
  vdbFrameAllLights,
  // His helper name stays inside frame/ (diffable with upstream); the public name follows D17.
  getFrameMediumProps as vdbFrameMedium,
  VDBFrameEmissionLight,
  vdbFrameBlackbodyEmission,
} from "./frame/index.js";
export type {
  VDBFrameVolumeProps,
  VDBFrameRendererSettings,
  VDBFramePassHook,
  VDBFramePass,
  VDBFrameRedrawStats,
  VDBFramePhase,
  VDBFrameMultipleScattering,
  VDBFrameMediumVolumeProps,
  VDBFrameEmissionLightProps,
  VDBFrameBlackbodyHook,
} from "./frame/index.js";
