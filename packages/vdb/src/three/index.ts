/**
 * @pmndrs/vdb/three — the three.js adapter over the engine-free core (D8).
 *
 * `getDevice` / `getGPUTexture` reach the renderer's WebGPU handles, so the
 * pool and the other pmndrs tools share one device. `VDBGridNode` puts a
 * `VDBGridPool` into three's node system (`vdbSample`, `vdbReadValue`, or the
 * module as a `wgslFn` include). `VDBVolumeMaterial` raymarches one pooled
 * grid; `VDBSequence` plays a frame sequence into one pool id with
 * `pool.replace`.
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
export type { VDBVolumeMaterialParameters, VDBProxyGridInfo } from "./material.js";
export { VDBSequence } from "./sequence.js";
export type {
  VDBSequenceOptions,
  VDBSequenceStats,
  VDBSequenceTarget,
  VDBSequencePoolLike,
  VDBFrameLoader,
} from "./sequence.js";

// Convenience bootstrap (a renderer on a device the caller's code creates; D8 allows it in an adapter).
export { createVolumeRenderer, computeRequiredLimits, nextPow2 } from "./renderer.js";
export type { VolumeRendererOptions, CapabilityReport, RequiredLimits } from "./renderer.js";

// K2.3: compute wrappers over the core tools (renderer in, promise out).
export { gridStats, decodeToAtlas, valueTransform } from "./compute.js";
export type { GridStatsOptions, GridStatsResult, DecodeToAtlasOptions, DecodeToAtlasResult } from "./compute.js";

// K3.0: the dense `.wvdb` volume source (R1 §4.2) in his createVolumeNodes() shape.
export { vdbDenseTextures, vdbVolumeUniforms, vdbVolumeNodes, vdbMinComponent, vdbMaxComponent } from "./volume-source.js";
export type {
  VDBDenseChannelTextures,
  VDBThreeDenseSource,
  VDBVolumeNodes,
  VDBVolumeNodesOptions,
  VDBVolumeUniformParameters,
  VDBVolumeUniforms,
} from "./volume-source.js";
