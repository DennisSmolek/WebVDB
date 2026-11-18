/**
 * @pmndrs/vdb/three internals (D32): NOT a package export.
 *
 * Everything public (`./index.ts`) plus what tests and harnesses need that a user doesn't: the frame
 * renderer's pass materials (inject, integrate, 'frame' lighting, shadows, emission), its TSL shader
 * chunks, the uniform packers and froxel layout helpers behind the settings, the volume shadow system
 * and the three-internals hooks. Import it by path (`src/three/internal.ts`); its names can change
 * without notice.
 */

export * from "./index.js";

export { vdbAttenuatedPointSize, vdbPointDiscTexture } from "./points.js";
export { vdbMinComponent, vdbMaxComponent } from "./volume-source.js";
export {
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
  vdbFrameChunks,
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
  getFrameExtinctionSpectrum as vdbFrameExtinctionSpectrumInto,
  getFrameAngstromExponent as vdbFrameAngstromExponent,
  frameAngstromExponents as vdbFrameAngstromExponents,
  VDBFrameVolumeShadows,
  VDBFrameVolumeShadowMaterial,
  VDBFrameEmissionLightNode,
  VDBFrameEmissionDirectionMaterial,
  VDBFrameEmissionReduceMaterial,
  VDBFrameEmissionProbeMaterial,
  vdbFrameCheckLightReset,
} from "./frame/index.js";
export type {
  VDBFrameInjectMaterialOptions,
  FrameVolumeRecord,
  FroxelArea,
  VDBFrameShadowEntry,
} from "./frame/index.js";
