/**
 * K3.1: mjurczyk's froxel pipeline with `'volume'` lighting (R1 §6 Step 1), ported as three r187
 * TSL fragment passes over the VDBVolumeSource seam. See docs/handoffs/K3-1.md.
 */
export { VDBFrameVolume } from "./volume.js";
export type { VDBFrameVolumeProps } from "./volume.js";
export { VDBFrameRenderer, vdbFrameRenderer, setVolumeUniforms } from "./renderer.js";
export type { VDBFrameRendererSettings, VDBFramePassHook } from "./renderer.js";
export { VDBFrameInjectMaterial } from "./inject-material.js";
export type { VDBFrameInjectMaterialOptions } from "./inject-material.js";
export { VDBFrameIntegrateMaterial } from "./integrate-material.js";
export { vdbFrameLights, vdbFrameAllLights } from "./lights.js";
export {
  getFramePhase,
  getFramePhaseSpectrum,
  getFramePhaseAsymmetry,
  getFrameMultipleScattering,
  isFrameDiffusion,
} from "./phase.js";
export type { VDBFramePhase, VDBFrameMultipleScattering, VDBFramePhaseSource } from "./phase.js";
export { getAtlasLayout, getFroxelSize } from "./layout.js";
export { FrameVolumeCollector, getFroxelSlice, getFroxelSliceRange, getFroxelRect, getFroxelArea, getFrameFadeRadius } from "./view.js";
export type { FrameVolumeRecord, FroxelArea } from "./view.js";
export * as vdbFrameChunks from "./chunks.js";
