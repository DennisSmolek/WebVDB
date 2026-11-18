/**
 * @pmndrs/vdb/tools internals (D32): NOT a package export.
 *
 * Everything public (`./index.ts`) plus what tests, harnesses and scripts need that a user doesn't:
 * the CPU twins of the WGSL march profile and the frame-renderer passes (the parity oracles), the
 * `.nvdb` / `.wvdb` layout constants and chunk codecs, the stride-table accessors, the sparse source's
 * CPU pyramid, and the frame lighting maths. Import it by path (`src/tools/internal.ts`, or
 * `dist/tools/internal.js` from a script after `pnpm build`); its names can change without notice.
 */

export * from "./index.js";

export { FILE_HEADER_SIZE, FILE_METADATA_SIZE, GRID_DATA_SIZE, MAGIC_ASCII } from "./nvdb-file.js";
export {
  cellBoundTrilinear,
  defaultValueMax,
  getDim,
  gridHasStats,
  hddaCellEnd,
  hddaInit,
  hddaStep,
  hddaUpdate,
  hddaWalk,
  indexBBox,
  isActive,
  leafVoxels,
  nodeAverage,
  nodeMax,
  nodeStats,
  rayClip,
  rayClipSupport,
  treeNodes,
} from "./cpu/march.js";
export type {
  Hdda,
  HddaSegment,
  HddaWalkOptions,
  NodeLevel,
  NodeMaxOptions,
  NodeStats,
  RayClipSupportOptions,
  TreeNode,
  Vec3,
} from "./cpu/march.js";
export { probeCoords, probePoints } from "./cpu/probe-coords.js";
export type { ProbeOpts } from "./cpu/probe-coords.js";
export {
  levelSetGradient,
  levelSetHit,
  levelSetHitReference,
  levelSetNormal,
  sampleTrilinearF32,
} from "./cpu/level-set.js";
export type { LevelSetHit, LevelSetHitOptions } from "./cpu/level-set.js";
export {
  atlasDims,
  atlasFromRaw,
  bboxSize,
  GRID_STATS_SUM_SCALE,
  rootStats,
  transformLeafValues,
} from "./cpu/compute.js";
export type { AtlasFromRawOptions, IndexBBox } from "./cpu/compute.js";
export { defineNumber, gridTypeConstantsForId } from "./cpu/stride-tables.js";
export type { GridTypeConstants } from "./cpu/stride-tables.js";
export {
  brickCounts as wvdbBrickCounts,
  brickOrigin as wvdbBrickOrigin,
  brickTotal as wvdbBrickTotal,
  decodeMask as decodeWvdbMask,
  decodePage as decodeWvdbPage,
  decodePoints as decodeWvdbPoints,
  encodeMask as encodeWvdbMask,
  encodePage as encodeWvdbPage,
  encodePoints as encodeWvdbPoints,
  inflateWvdbChunk,
  normalizeWvdbClass,
  wvdbChunks,
  wvdbChunkByteLimit,
  downsampleWvdbLevel,
  wvdbDensityStats,
  sampleWvdbDense,
  sampleWvdbDenseBox,
  sampleWvdbLevelSetSurfaceMean,
  wvdbFinestDimensions,
  sha256Hex,
  wvdbTexels,
  wvdbTexelStats,
  wvdbMipAtlas,
  wvdbMipLevelCount,
  wvdbMipLevelSize,
  wvdbToByte,
  wvdbDensityScale,
  sampleWvdbDenseTexture,
} from "./wvdb/index.js";
export type {
  WvdbBrickInput,
  DecodedPage as WvdbDecodedPage,
  DecodedPoints as WvdbDecodedPoints,
  WvdbLevelSetSource,
} from "./wvdb/index.js";
export {
  vdbFrameSliceDepth,
  vdbFrameFroxelSlice,
  vdbFrameSliceRange,
  vdbFrameViewPosition,
  vdbFrameFroxel,
  vdbFrameDistanceFade,
  vdbFrameBoxSteps,
  vdbFrameNoise,
  vdbFrameHenyeyGreenstein,
  vdbFramePhaseWeight,
  vdbFrameShadowAbsorption,
  vdbFrameShadowStep,
  vdbFrameShadowDepth,
  vdbFrameShadowTransmittance,
  vdbFrameShadowMarch,
  vdbFrameShadowedLight,
  vdbFrameInjectSlice,
  vdbFrameIntegrate,
  vdbSrgbOetf,
  vdbSrgbEotf,
  vdbFrameSrgbBlend,
  vdbFrameHenyeyGreensteinColor,
  vdbFramePhaseColor,
  vdbFramePhaseWeightScaled,
  vdbFrameOctavesOff,
  vdbFrameMaxScatteringOctaves,
  vdbFrameScatteringOctaves,
  vdbFrameShadowMarchDepth,
  vdbFrameShadowOpticalDepth,
  vdbFrameOctaveLight,
  vdbFrameShadowTotal,
  vdbFrameTwoSidedDiffusion,
  vdbFrameShadowedLightColor,
} from "./frame/index.js";
export type {
  VDBFrameFroxelSample,
  VDBFrameOctaveSettings,
  Vec2 as VDBFrameVec2,
  Vec3 as VDBFrameVec3,
  Vec4 as VDBFrameVec4,
} from "./frame/index.js";
export {
  vdbPyramidBaseLevel,
  vdbBoxMipAtlasF32,
  vdbBoxPyramid,
  vdbFrameNodeAverageSample,
  vdbFrameSkipEmpty,
} from "./frame/sparse.js";
export type { VDBFloatMipAtlas, VDBBoxPyramidOptions, VDBBoxPyramid } from "./frame/sparse.js";
export { vdbBoxPyramidLayout, vdbPyramidLevelSize } from "./frame/pyramid-layout.js";
export type { VDBBoxPyramidLayout, VDBPyramidAtlasLevel } from "./frame/pyramid-layout.js";
export {
  vdbFrameGridCascades,
  vdbFrameGridRange,
  vdbFrameCascadeCasters,
  vdbFrameGridSignatureState,
  vdbFrameUpdateGridSignatures,
  vdbSameValues,
  vdbSignaturePushVector,
  vdbSignaturePushElements,
  vdbFrameGridUvw,
  vdbFrameInsideGrid,
  vdbFrameFindCascade,
  vdbFrameGridCell,
  vdbFrameAtlasBilinear,
  vdbFrameSampleTiles,
  vdbFrameSampleGrid,
  vdbFrameGridExtinction,
  vdbFrameGridMarch,
  vdbFrameIsotropicDiffusion,
  vdbFrameTransmittanceVoxel,
  vdbFrameReproject,
  vdbFrameLightingFroxel,
  vdbFrameEnvironmentGridVoxel,
  vdbFrameSampleShadows,
  vdbFrameDensityFroxel,
  vdbFrameDensityFroxelSteps,
} from "./frame/grid.js";
export type {
  VDBVec3Like,
  VDBBox3Like,
  VDBFrameGridSignatureState,
  VDBFrameGridLayout,
  VDBFrameAtlas,
  VDBFrameShadowLight,
  VDBFrameLightingInputs,
  VDBFrameDensityInputs,
  VDBFrameTransmittanceModes,
} from "./frame/grid.js";
export {
  VDBFrameEnvironmentLights,
  vdbFrameEnvironmentMapDirection,
  vdbFrameEnvironmentMode,
  frameEnvironmentModes as vdbFrameEnvironmentModes,
  frameMaxEnvironmentLights as vdbFrameMaxEnvironmentLights,
  frameEnvironmentMapWidth as vdbFrameEnvironmentMapWidth,
  frameEnvironmentMapHeight as vdbFrameEnvironmentMapHeight,
} from "./frame/environment.js";
export type {
  VDBFrameEnvironmentMode,
  VDBFrameEnvironmentSettings,
  VDBFrameEnvironmentScene,
  VDBFrameEnvironmentLight,
  VDBFrameEnvironmentResult,
} from "./frame/environment.js";
export {
  vdbFrameMieTable,
  vdbFrameMieTexels,
  vdbFrameMieTexel,
  vdbFrameMiePhase,
  vdbFrameMieRgbPhase,
  vdbFrameMieFormat,
  vdbFrameMieTableSize,
  vdbFrameMieAsymmetry,
  vdbFrameMieRgbMode,
  vdbFrameMieChannel,
  vdbFrameMieRgbChannels,
} from "./frame/mie.js";
export type { VDBFrameMieTableOptions, VDBFrameMieTableResult } from "./frame/mie.js";
export {
  vdbFrameNeuralLayout,
  vdbFrameNeuralCheck,
  vdbFrameNeuralForward,
  vdbFrameNeuralSampleLevel,
  vdbFrameNeuralUnit,
  vdbFrameNeuralInputs,
  vdbFrameNeuralLight,
  vdbFrameTileAtlasLayout,
  vdbFrameGridMipLevel,
  vdbFrameNeuralMipResolution,
} from "./frame/neural.js";
export type { VDBFrameNeuralNetwork, VDBFrameNeuralMip } from "./frame/neural.js";
export { vdbFrameNeuralWeights } from "./frame/neural-weights.js";
export { vdbFrameMediumMeasurements, vdbFrameFitPhaseGExponent } from "./frame/medium.js";
export {
  vdbFrameSpectrumWavelengths,
  vdbFrameSpectralGroupCount,
  vdbFrameSpectralGroups,
  vdbFrameExtinctionSpectrum,
  vdbFrameMeanComponent,
  vdbFrameSpectralShadowUnit,
  vdbFrameSpectralShadowDepths,
  vdbFrameSpectralShadowStep,
  vdbFrameSpectralGridAlbedo,
  vdbFrameSpectralShadowDepth,
  vdbFrameSpectralShadowMarchDepth,
  vdbFrameSpectralShadowTransmittance,
  vdbFrameSpectralOctaveLight,
  vdbFrameSpectralShadowTotal,
  vdbFrameSpectralShadowedLight,
  vdbFrameUnpackSpectralGroup,
  vdbFramePackSpectralGroup,
  vdbFrameSpectralGridSample,
  vdbFrameSpectralGridMarch,
  vdbFrameSpectralTransmittanceVoxel,
  vdbFrameSpectralEnvironmentGridVoxel,
  vdbFrameSpectralDensityFroxel,
  vdbFrameSpectralLightingFroxel,
  vdbFrameSpectralGroupsAt,
  vdbShIrradianceAt,
  vdbFrameInjectIsotropicIrradiance,
  vdbFrameSpectralShadowMarch,
  vdbFrameSpectralInjectSlice,
  vdbFrameSpectralIntegrate,
} from "./frame/spectral.js";
export type {
  VDBFrameSpectralDepths,
  VDBFrameSpectralShadow,
  VDBFrameSpectralMarch,
  VDBFrameSpectralDensityInputs,
  VDBFrameSpectralLightingInputs,
  VDBFrameInjectMarch,
  VDBFrameInjectLightInputs,
} from "./frame/spectral.js";
export {
  frameShadowMinOpticalDepth as vdbFrameShadowMinOpticalDepth,
  frameShadowMaxOpticalDepth as vdbFrameShadowMaxOpticalDepth,
  frameMaxCastShadowSteps as vdbFrameMaxCastShadowSteps,
  frameMaxShadowMapCascades as vdbFrameMaxShadowMapCascades,
  vdbFrameShadowDepth as vdbFrameShadowCameraDepth,
  vdbFrameShadowPassRay,
  vdbFrameShadowMapTexel,
  vdbFrameShadowReceiver,
  vdbFrameShadowCascadeBreaks,
  vdbFrameShadowCascadeFar,
  vdbFrameShadowCascadeIndex,
  vdbFrameShadowCascadeUv,
} from "./frame/shadows.js";
export type { VDBFrameShadowRay } from "./frame/shadows.js";
export {
  frameEmissionDirections as vdbFrameEmissionDirections,
  frameEmissionRays as vdbFrameEmissionRays,
  frameEmissionMaxResolution as vdbFrameEmissionMaxResolution,
  frameEmissionMaxCascades as vdbFrameEmissionMaxCascades,
  frameEmissionAtlasSize as vdbFrameEmissionAtlasSize,
  vdbFrameEmissionDirection,
  vdbFrameBoxSegment,
  vdbFrameEmitterBounds,
  vdbFrameEmissionDirectionTexel,
  vdbFrameEmissionReduce,
  vdbFrameEmissionProbe,
  vdbFrameEmissionIrradiance,
  vdbFrameEmissionProbeGrid,
  vdbFrameEmissionFalloff,
  vdbFrameEmissionLookup,
} from "./frame/emission.js";
export type {
  VDBFrameEmitter,
  VDBFrameEmitterLight,
  VDBFrameEmissionGridSettings,
  VDBFrameEmissionGrid,
} from "./frame/emission.js";
export {
  vdbFrameBlackbodyRgb,
  vdbFrameBlackbodyTable,
  vdbFrameEmissionSpectrum,
  frameBlackbodyWavelengthStart as vdbFrameBlackbodyWavelengthStart,
  frameBlackbodyWavelengthStep as vdbFrameBlackbodyWavelengthStep,
  frameBlackbodyWavelengthCount as vdbFrameBlackbodyWavelengthCount,
  frameBlackbodyTableSize as vdbFrameBlackbodyTableSize,
  frameBlackbodyTableRows as vdbFrameBlackbodyTableRows,
} from "./frame/blackbody.js";
