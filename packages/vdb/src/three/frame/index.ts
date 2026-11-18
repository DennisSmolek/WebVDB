/**
 * K3.1: mjurczyk's froxel pipeline with `'volume'` lighting (R1 §6 Step 1), ported as three r187
 * TSL fragment passes over the VDBVolumeSource seam. See docs/handoffs/K3-1.md.
 */
export { VDBFrameVolume } from "./volume.js";
export type { VDBFrameVolumeProps } from "./volume.js";
export { VDBFrameRenderer, vdbFrameRenderer, setVolumeUniforms } from "./renderer.js";
export type { VDBFrameRendererSettings, VDBFramePassHook, VDBFramePass, VDBFrameRedrawStats } from "./renderer.js";
// K3.3: 'frame' lighting passes
export { VDBFrameShadowDensityMaterial } from "./shadow-density-material.js";
export { VDBFrameTransmittanceMaterial, frameShadowLightSlots } from "./transmittance-material.js";
export { VDBFrameDensityMaterial, createDensityShadowUniforms, frameDensityEnvironment } from "./density-material.js";
export { VDBFrameLightingMaterial } from "./lighting-material.js";
export { VDBFrameTemporalMaterial } from "./temporal-material.js";
export { VDBFrameEnvironmentMaterial, VDBFrameEnvironmentMapMaterial } from "./environment-material.js";
export { VDBFrameEnvironmentGridMaterial } from "./environment-grid-material.js";
// K3.4: phase functions and multiple scattering (two-sided, octaves, neural)
export { VDBFrameGridMipMaterial } from "./grid-mip-material.js";
export { createNeuralNodes } from "./neural.js";
export { frameNeuralMipLevels } from "./transmittance-material.js";
export { VDBFrameInjectMaterial } from "./inject-material.js";
export type { VDBFrameInjectMaterialOptions } from "./inject-material.js";
export { VDBFrameIntegrateMaterial } from "./integrate-material.js";
export { vdbFrameLights, vdbFrameAllLights } from "./lights.js";
export {
  getFramePhase,
  getFramePhaseSpectrum,
  getFramePhaseAsymmetry,
  getFrameMultipleScattering,
  getFrameScatteringOctaves,
  getFrameTwoSidedDiffusion,
  getFrameMieTexels,
  isFrameDiffusion,
} from "./phase.js";
export type { VDBFramePhase, VDBFrameMultipleScattering, VDBFramePhaseSource } from "./phase.js";
export { getAtlasLayout, getFroxelSize } from "./layout.js";
// K3.5: spectral media and presets
export { getFrameMediumProps } from "./medium.js";
export type { VDBFrameMediumVolumeProps } from "./medium.js";
export { frameSpectrumWavelengths, frameAngstromExponents, getFrameAngstromExponent, getFrameExtinctionSpectrum, frameSpectralGroupCount, getFrameSpectralGroups } from "./spectrum.js";
export { FrameVolumeCollector, getFroxelSlice, getFroxelSliceRange, getFroxelRect, getFroxelArea, getFrameFadeRadius } from "./view.js";
export type { FrameVolumeRecord, FroxelArea } from "./view.js";
export * as vdbFrameChunks from "./chunks.js";
// K3.6: volume shadows on meshes, the emission light, the CIE blackbody emission colour
export { VDBFrameVolumeShadows, VDBFrameVolumeShadowNode, frameMaxShadowMapCascades } from "./volume-shadows.js";
export type { VDBFrameShadowEntry } from "./volume-shadows.js";
export { VDBFrameVolumeShadowMaterial, createShadowRayUniforms, frameShadowDepth, frameMaxCastShadowSteps } from "./volume-shadow-material.js";
export { VDBFrameEmissionLight, VDBFrameEmissionLightNode, VDBFrameEmissionDirectionMaterial, VDBFrameEmissionReduceMaterial, VDBFrameEmissionProbeMaterial } from "./emission-light.js";
export type { VDBFrameEmissionLightProps } from "./emission-light.js";
export { vdbFrameBlackbodyEmission, getFrameBlackbodyMap, frameBlackbodyColor } from "./blackbody.js";
export type { VDBFrameBlackbodyHook } from "./blackbody.js";
export { vdbFrameCheckLightReset, VDB_FRAME_THREE_REVISION } from "./three-internals.js";
