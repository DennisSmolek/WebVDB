/**
 * @pmndrs/vdb/tools — the CPU half, pure TypeScript first (decision D3).
 *
 * Phase 0 stub: API surface only. The `.vdb` parser, NanoVDB serializer,
 * quantization, affine transforms, and `.nvdb` I/O are Phase 5
 * deliverables, validated byte/value-wise against official
 * `nanovdb_convert` output on the fixture corpus (docs/SPEC.md §4).
 * WASM is an optional escalation rung (W1), never the foundation.
 */

export type {
  VdbBBox,
  VdbFile,
  VdbFloatGrid,
  VdbGrid,
  VdbLeaf,
  VdbPointAttributeArray,
  VdbPointAttributeInfo,
  VdbPointDataGrid,
  VdbPointList,
  VdbTile,
  VdbTransformInfo,
  VdbValueType,
  VdbVec3Grid,
  Vec3Tuple,
} from "./vdb/index.js";
export { VdbFormatError, VdbUnsupportedError } from "./vdb/index.js";

/**
 * Parse a `.vdb` container: 5-4-3 trees of float, vec3s/vec3d/vec3i and point
 * data (`ptdataidx32`) grids; uncompressed, zlib or Blosc (LZ4/zlib); half-float
 * storage; active tiles. See `src/vdb/` for the reader implementation.
 */
export { parseVdb } from "./vdb/index.js";

/**
 * Build a NanoVDB grid image from a parsed `.vdb` grid (FLOAT, or Vec3f for
 * vector grids) — streams the parser's leaves into the serializer (no dense
 * array), carrying the `.vdb` uniform scale+translate transform into the
 * NanoVDB Map.
 */
export { buildFromVdb, buildFromVdbDetailed } from "./nanovdb/index.js";
export type { BuildFromVdbOptions } from "./nanovdb/index.js";

/**
 * NanoVDB serializer (Phase 5b): build FLOAT grid images from dense
 * arrays and write `.nvdb` segment files — see `src/nanovdb/` for the
 * layout/stats/validation details and the LeafCodec seam Fp8/FpN slot into.
 */
export {
  buildFromDense,
  buildFromDenseDetailed,
  buildFromLeavesDetailed,
  FLOAT_LEAF_CODEC,
  FP8_LEAF_CODEC,
  makeFpNLeafCodec,
  VEC3F_LEAF_CODEC,
} from "./nanovdb/index.js";
export type {
  BuildFromDenseOptions,
  BuildFromLeavesOptions,
  BuiltGrid,
  LeafCodec,
} from "./nanovdb/index.js";

/** Quantize a FLOAT grid image to Fp8/FpN (per-leaf min/quantum + packed codes). */
export { collapseGrid, quantize, quantizeDetailed } from "./nanovdb/index.js";
export type { QuantizeMode, QuantizeOptions } from "./nanovdb/index.js";

/**
 * Resample-to-resolution bake (K4.4): a NanoVDB image or parsed `.vdb` grid
 * resampled over a world box with a point, box (default) or trilinear filter,
 * sparse, optionally quantized; and the comparison report's error metric.
 */
export { resampleGrid, resampleError, resampleDense } from "./nanovdb/index.js";
export type {
  ResampleErrorOptions,
  ResampleErrorResult,
  ResampledGrid,
  ResampleFilter,
  ResampleOptions,
  ResampleDenseOptions,
  ResampleSource,
} from "./nanovdb/index.js";

/** Affine transform — a metadata-only Map edit, no voxel churn (uniform scale+translate). */
export { transform } from "./nanovdb/index.js";
export type { TransformInput, TransformSpec } from "./nanovdb/index.js";

/** Tree stats, per-level counts, memory breakdown. */
export { inspect } from "./nanovdb/index.js";
export type { InspectReport } from "./nanovdb/index.js";

export { writeNvdb } from "./nanovdb/index.js";
export type { WriteNvdbOptions } from "./nanovdb/index.js";

/** Read `.nvdb` files. Phase 5 follow-up (`NanoVDBFile` in this module covers reading today). */
export function readNvdb(_buffer: ArrayBuffer): never {
  throw new Error("readNvdb: not implemented (Phase 5)");
}

/**
 * `.nvdb` file loader (from the former `nanovdb-wgsl` package).
 */
export {
  FILE_HEADER_SIZE,
  FILE_METADATA_SIZE,
  GRID_DATA_SIZE,
  MAGIC_ASCII,
  Codec,
  SUPPORTED_GRID_TYPES,
  NanoVDBFile,
} from "./nvdb-file.js";
export type { GridMetadata } from "./nvdb-file.js";

/**
 * Coarse-first `.nvdb` delivery (K4.16, the K4.5 design's option a): bake
 * `[<name>.lod48, <name>]` into one file, and read a `.nvdb` from a stream,
 * emitting each grid as soon as its bytes (ZIP or raw) are complete.
 */
export { bakeCoarseFirst } from "./nanovdb/index.js";
export type { BakeCoarseFirstOptions } from "./nanovdb/index.js";
export { NvdbFormatError, NvdbStreamReader, nvdbLodName, nvdbResponseLength, parseNvdbLodName, readNvdbStream } from "./nvdb-stream.js";
export type { NvdbProgress, NvdbStreamGrid, NvdbStreamReaderOptions, NvdbStreamSource } from "./nvdb-stream.js";

/**
 * Pure-TypeScript CPU reference for NanoVDB voxel lookup/sampling (Phase 1/2,
 * docs/PLAN.md) — a byte-for-byte transliteration of the vendored
 * `PNanoVDB.h` root -> upper -> lower -> leaf descent, restricted to grid
 * types FLOAT/FP8/FPN. Reads layout constants from a baked, browser-safe
 * generated module (`./cpu/stride-tables.generated.ts`), so these are safe to
 * import from any Vite/browser page as well as Node.
 */
export { readValue, readValueVec3 } from "./cpu/read-value.js";
export type { ReadResult, ReadVec3Result } from "./cpu/read-value.js";
export { sampleNearest, sampleTrilinear, sampleTrilinearVec3 } from "./cpu/sample-trilinear.js";
export type { SampleOptions } from "./cpu/sample-trilinear.js";

/**
 * CPU twins of the `march` WGSL profile (K1.4b, D11): the vendored HDDA, the
 * node size at a voxel, node stats (with the Fp leaf-stats fix), the
 * `vdbNodeMax` bound, the ray clip and the canonical HDDA walk, plus the
 * opt-in helpers (K1.7a): `cellBoundTrilinear`, `hddaCellEnd`,
 * `rayClipSupport`.
 */
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
  readValueF32,
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

/**
 * K3.7: CPU twins of the opt-in `'levelSet'` WGSL helper (the level-set
 * zero-crossing finder and its gradient normal), the brute-force oracle they
 * are tested against, the marching-cubes mesh of a level set and the
 * render-ready point list of a point data grid (his `createPointData`).
 */
export { levelSetGradient, levelSetHit, levelSetHitReference, levelSetNormal, sampleTrilinearF32 } from "./cpu/level-set.js";
export type { LevelSetHit, LevelSetHitOptions } from "./cpu/level-set.js";
export { vdbPointRenderData, VDB_POINTS_DEFAULT_MAX } from "./vdb/point-render-data.js";
export type { VDBPointRenderData, VDBPointRenderDataOptions } from "./vdb/point-render-data.js";

/**
 * CPU twins of the core compute tools (K2.3): `gridStatsCpu`,
 * `decodeToAtlasCpu` (+ `atlasFromRaw`, the shared post-readback step), and
 * `transformLeafValues`, the CPU value transform behind `applyValueTransform`.
 */
export {
  atlasDims,
  atlasFromRaw,
  bboxSize,
  decodeToAtlasCpu,
  GRID_STATS_SUM_SCALE,
  gridStatsCpu,
  rootStats,
  transformLeafValues,
} from "./cpu/compute.js";
export type {
  AtlasFilter,
  AtlasFormat,
  AtlasFromRawOptions,
  DecodeToAtlasCpuOptions,
  GridStatsCpuOptions,
  GridStatsCpuResult,
  IndexBBox,
} from "./cpu/compute.js";

/**
 * Layout-constant accessors underlying `readValue` — exposed for callers
 * that need to walk NanoVDB tree structure generically (any grid type, not
 * just FLOAT/FP8/FPN) without decoding leaf values themselves, e.g.
 * enumerating LEAF node origins for a visualization. See `./cpu/
 * stride-tables.ts` for the full accessor surface and its data source.
 */
export { defineNumber, gridTypeConstantsForId } from "./cpu/stride-tables.js";
export type { GridTypeConstants } from "./cpu/stride-tables.js";

/**
 * `.wvdb` — mjurczyk's WebVDB delivery format (spec v0.5.x, D14): read
 * (whole file, prefix, byte range, or stream; coarse LOD first), write, and
 * bake from a parsed `.vdb` or a NanoVDB grid (point-sampled, like his CLI).
 */
export {
  WVDB_BRICK_DIM,
  WVDB_BRICK_VOXELS,
  WVDB_SUPPORTED_VERSIONS,
  WvdbFormatError,
  brickCounts as wvdbBrickCounts,
  brickOrigin as wvdbBrickOrigin,
  brickTotal as wvdbBrickTotal,
  decodeMask as decodeWvdbMask,
  decodePage as decodeWvdbPage,
  decodePoints as decodeWvdbPoints,
  encodeMask as encodeWvdbMask,
  encodePage as encodeWvdbPage,
  encodePoints as encodeWvdbPoints,
  readWvdb,
  readWvdbHeader,
  decodeWvdbLod,
  inflateWvdbChunk,
  normalizeWvdbClass,
  wvdbChunks,
  wvdbLodByteRange,
  wvdbPrefixLength,
  WvdbStreamReader,
  wvdbChunkByteLimit,
  writeWvdb,
  downsampleWvdbLevel,
  wvdbDensityStats,
  wvdbPointLods,
  bakeWvdb,
  sampleWvdbDense,
  sampleWvdbDenseBox,
  sampleWvdbLevelSetMjurczyk,
  wvdbFinestDimensions,
  wvdbSourceFromVdbGrid,
  wvdbSourceFromNanoVdb,
  vdbToWvdb,
  nanoVdbToWvdb,
  sha256Hex,
  wvdbDenseVolume,
  wvdbTexels,
  wvdbTexelStats,
  wvdbMipAtlas,
  wvdbMipLevelCount,
  wvdbMipLevelSize,
  wvdbToByte,
  wvdbDensityScale,
  sampleWvdbDenseTexture,
  wvdbDenseToNanoVdb,
} from "./wvdb/index.js";
export type {
  WvdbBake,
  WvdbBits,
  WvdbBox,
  WvdbChunkRef,
  WvdbGridClass,
  WvdbGridHeader,
  WvdbHeader,
  WvdbLayout,
  WvdbPageRef,
  WvdbPointsGridHeader,
  WvdbPointsLodHeader,
  WvdbPointsPageRef,
  WvdbSource,
  WvdbVec3,
  WvdbVolumeGridHeader,
  WvdbVolumeLodHeader,
  WvdbBrickInput,
  DecodedPage as WvdbDecodedPage,
  DecodedPoints as WvdbDecodedPoints,
  WvdbHeaderInfo,
  ReadWvdbHeaderOptions,
  ReadWvdbOptions,
  WvdbChunkInfo,
  WvdbFile,
  WvdbGrid,
  WvdbVolumeGrid,
  WvdbPointsGrid,
  WvdbLevel,
  WvdbVolumeLevel,
  WvdbPointsLevel,
  WvdbDenseVolume,
  WvdbDenseTexture,
  WvdbDenseOptions,
  WvdbMipAtlas,
  WvdbTexelStats,
  WvdbLodEvent,
  WvdbStreamReaderOptions,
  WvdbDenseLevel,
  WvdbVolumeGridInput,
  WvdbPointsGridInput,
  WvdbGridInput,
  WriteWvdbOptions,
  WriteWvdbResult,
  WvdbSampleSource,
  WvdbSampling,
  WvdbLevelSetSampler,
  WvdbIndexAccess,
  WvdbLevelSetSource,
  BakeWvdbOptions,
  VdbToWvdbOptions,
  WvdbDenseToNanoVdbOptions,
  WvdbNanoVdbGrid,
} from "./wvdb/index.js";

/**
 * K3.1: CPU twins of the froxel pipeline maths (mjurczyk's `'volume'` lighting), the test oracles
 * of the `./three` frame passes: slice mapping, HG phase, shadow march, inject pre-integration,
 * integrate composite, sRGB blend.
 */
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
} from "./frame/index.js";
export type { VDBFrameFroxelSample } from "./frame/index.js";

/**
 * K3.2: the sparse volume source's LOD strategies on the CPU (the coarse pyramid build, his box mip
 * atlas in float) and the CPU twins of its WGSL (node-average tap, HDDA empty-space query).
 */
export { vdbPyramidBaseLevel, vdbBoxMipAtlasF32, vdbBoxPyramid, vdbFrameNodeAverageSample, vdbFrameSkipEmpty } from "./frame/sparse.js";
export type { VDBFloatMipAtlas, VDBBoxPyramidOptions, VDBBoxPyramid } from "./frame/sparse.js";
// K3.2a: the pyramid's layout, shared with the core's GPU build (`createBoxPyramid`).
export { vdbBoxPyramidLayout, vdbPyramidLevelSize } from "./frame/pyramid-layout.js";
export type { VDBBoxPyramidLayout, VDBPyramidAtlasLevel } from "./frame/pyramid-layout.js";

/**
 * K3.3: mjurczyk's 'frame' lighting, engine-free: the world shadow grid (cascades, caster ranges, change
 * signatures), CPU twins of the grid passes (atlas cell, cascade search, slice-atlas sampling, the
 * transmittance march, isotropic diffusion, temporal reprojection) and the environment lights
 * ('directional' / 'spherical-harmonics' from a lat-long readback).
 */
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
} from "./frame/grid.js";
export type { VDBVec3Like, VDBBox3Like, VDBFrameGridSignatureState, VDBFrameGridLayout, VDBFrameAtlas, VDBFrameShadowLight, VDBFrameLightingInputs, VDBFrameDensityInputs } from "./frame/grid.js";
export {
  VDBFrameEnvironmentLights,
  vdbFrameEnvironmentMapDirection,
  vdbFrameEnvironmentMode,
  frameEnvironmentModes as vdbFrameEnvironmentModes,
  frameMaxEnvironmentLights as vdbFrameMaxEnvironmentLights,
  frameEnvironmentMapWidth as vdbFrameEnvironmentMapWidth,
  frameEnvironmentMapHeight as vdbFrameEnvironmentMapHeight,
} from "./frame/environment.js";
export type { VDBFrameEnvironmentMode, VDBFrameEnvironmentSettings, VDBFrameEnvironmentScene, VDBFrameEnvironmentLight, VDBFrameEnvironmentResult } from "./frame/environment.js";

/**
 * K3.4: mjurczyk's phase functions and multiple-scattering modes as CPU twins: per-channel and
 * Lorenz-Mie phases (the Mie table generator, a port of his BHMIE script, and the table read),
 * scattering octaves, two-sided slab diffusion, the octave shadow total, and the neural indirect
 * network (forward pass, inputs, grid mips). His trained weights are a separate data module
 * (`vdbFrameNeuralWeights`, MIT; provenance in THIRD_PARTY.md).
 */
export {
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
export type { VDBFrameOctaveSettings, Vec2 as VDBFrameVec2, Vec3 as VDBFrameVec3, Vec4 as VDBFrameVec4 } from "./frame/index.js";
export type { VDBFrameTransmittanceModes } from "./frame/grid.js";
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
