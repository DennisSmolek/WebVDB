/**
 * @pmndrs/vdb/tools — the CPU half, pure TypeScript (D3): parse `.vdb`, build, quantize, resample and
 * transform NanoVDB grid images, read and write `.nvdb` and `.wvdb` (whole, streamed, coarse first),
 * and read voxels on the CPU. No GPU, no `three`; Node and browsers.
 *
 * This barrel is the public API (D32): what a user calls. The CPU twins of the WGSL and the frame
 * passes, layout accessors and other test oracles live in `./internal.ts`, which tests and harnesses
 * import by path; it is not a package export.
 *
 * Naming (D32): library API is `VDB*` / `vdb*`; a type or class bound to one file format keeps that
 * format's prefix (`Vdb*` for OpenVDB `.vdb`, `Nvdb*` for `.nvdb`, `Wvdb*` for `.wvdb`), and a function
 * that reads, writes or converts one format may use the verb-and-format form (`parseVdb`, `writeNvdb`,
 * `readWvdb`, `vdbToWvdb`).
 */

/** The one error hierarchy (D32): every error on bad input is a `VDBToolsError` with a `code`. */
export { VDBToolsError, VdbFormatError, NvdbFormatError, WvdbFormatError, VDBUnsupportedError } from "./errors.js";
export type { VDBToolsErrorCode } from "./errors.js";

/**
 * Parse a `.vdb` container: 5-4-3 trees of float, vec3s/vec3d/vec3i and point
 * data (`ptdataidx32`) grids; uncompressed, zlib or Blosc (LZ4/zlib); half-float
 * storage; active tiles. See `src/vdb/` for the reader implementation.
 */
export { parseVdb } from "./vdb/index.js";
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
  VDBVec3,
} from "./vdb/index.js";
/** The render-ready point list of a point data grid (his `createPointData`). */
export { vdbPointRenderData, VDB_POINTS_DEFAULT_MAX } from "./vdb/point-render-data.js";
export type { VDBPointRenderData, VDBPointRenderDataOptions } from "./vdb/point-render-data.js";

/**
 * Build a NanoVDB grid image from a parsed `.vdb` grid (FLOAT, or Vec3f for
 * vector grids) — streams the parser's leaves into the serializer (no dense
 * array), carrying the `.vdb` uniform scale+translate transform into the
 * NanoVDB Map.
 */
export { buildFromVdb, buildFromVdbDetailed } from "./nanovdb/index.js";
export type { BuildFromVdbOptions } from "./nanovdb/index.js";

/**
 * NanoVDB serializer (Phase 5b): build grid images from dense arrays or your own
 * leaves through a leaf codec (Float, Fp4/Fp8/Fp16/FpN, Vec3f).
 */
export {
  vdbBuildFromDense,
  vdbBuildFromDenseDetailed,
  vdbBuildFromLeavesDetailed,
  VDB_FLOAT_LEAF_CODEC,
  VDB_FP4_LEAF_CODEC,
  VDB_FP8_LEAF_CODEC,
  VDB_FP16_LEAF_CODEC,
  vdbMakeFpNLeafCodec,
  VDB_VEC3F_LEAF_CODEC,
} from "./nanovdb/index.js";
export type { VDBBuildFromDenseOptions, VDBBuildFromLeavesOptions, VDBBuiltGrid, VDBLeafCodec } from "./nanovdb/index.js";

/** Quantize a FLOAT grid image to Fp4/Fp8/Fp16/FpN (per-leaf min/quantum + packed codes); fold constant leaves. */
export { vdbCollapseGrid, vdbQuantize, vdbQuantizeDetailed } from "./nanovdb/index.js";
export type { VDBQuantizeMode, VDBQuantizeOptions } from "./nanovdb/index.js";

/**
 * Resample-to-resolution bake (K4.4): a NanoVDB image or parsed `.vdb` grid
 * resampled over a world box with a point, box (default) or trilinear filter,
 * sparse, optionally quantized; and the comparison report's error metric.
 */
export { vdbResampleGrid, vdbResampleError, vdbResampleDense } from "./nanovdb/index.js";
export type {
  VDBResampleErrorOptions,
  VDBResampleErrorResult,
  VDBResampledGrid,
  VDBResampleFilter,
  VDBResampleOptions,
  VDBResampleDenseOptions,
  VDBResampleSource,
} from "./nanovdb/index.js";

/** Affine transform — a metadata-only Map edit, no voxel churn (uniform scale+translate). */
export { vdbTransform } from "./nanovdb/index.js";
export type { VDBGridTransformInput, VDBTransformSpec } from "./nanovdb/index.js";

/** Tree stats, per-level counts, memory breakdown. */
export { vdbInspect } from "./nanovdb/index.js";
export type { VDBInspectReport } from "./nanovdb/index.js";

/** `.nvdb` files: read (whole or streamed), write, and the coarse-first bake (K4.16). */
export { NvdbFile, NvdbCodec, NVDB_SUPPORTED_GRID_TYPES } from "./nvdb-file.js";
export type { NvdbGridMetadata } from "./nvdb-file.js";
export { writeNvdb } from "./nanovdb/index.js";
export type { WriteNvdbOptions } from "./nanovdb/index.js";
export { NvdbStreamReader, nvdbLodName, nvdbResponseLength, parseNvdbLodName, readNvdbStream } from "./nvdb-stream.js";
export type { NvdbProgress, NvdbStreamGrid, NvdbStreamReaderOptions, NvdbStreamSource } from "./nvdb-stream.js";
export { vdbBakeCoarseFirst } from "./nanovdb/index.js";
export type { VDBBakeCoarseFirstOptions } from "./nanovdb/index.js";

/**
 * CPU voxel access on a grid image (Phase 1/2): the vendored `PNanoVDB.h`
 * root -> upper -> lower -> leaf descent in TypeScript, and the nearest and
 * trilinear samplers on it. `vdbReadValueCpu` carries the `Cpu` suffix because
 * `@pmndrs/vdb/three` exports the GPU `vdbReadValue` (D32).
 */
export { vdbReadValueCpu, vdbReadValueVec3 } from "./cpu/read-value.js";
export type { VDBReadResult, VDBReadVec3Result } from "./cpu/read-value.js";
export { vdbReadValueF32 } from "./cpu/march.js";
export { vdbSampleNearest, vdbSampleTrilinear, vdbSampleTrilinearVec3 } from "./cpu/sample-trilinear.js";
export type { VDBSampleCpuOptions } from "./cpu/sample-trilinear.js";

/**
 * CPU versions of the core compute tools (K2.3): grid stats and the decode to a
 * dense atlas (the WebGL fallback path). The core's GPU versions are
 * `vdbGridStatsGpu` / `vdbDecodeToAtlasGpu`.
 */
export { vdbDecodeToAtlasCpu, vdbGridStatsCpu } from "./cpu/compute.js";
export type {
  VDBAtlasFilter,
  VDBAtlasFormat,
  VDBDecodeToAtlasCpuOptions,
  VDBGridStatsCpuOptions,
  VDBGridStatsCpuResult,
} from "./cpu/compute.js";

/**
 * `.wvdb` — mjurczyk's WebVDB delivery format (spec v0.5.x, D14): read
 * (whole file, prefix, byte range, or stream; coarse LOD first), write, and
 * bake from a parsed `.vdb` or a NanoVDB grid; the dense volume the frame
 * renderer's dense source uploads, and the dense volume back to a grid image.
 */
export {
  WVDB_BRICK_DIM,
  WVDB_BRICK_VOXELS,
  WVDB_SUPPORTED_VERSIONS,
  readWvdb,
  readWvdbHeader,
  decodeWvdbLod,
  wvdbLodByteRange,
  wvdbPrefixLength,
  WvdbStreamReader,
  writeWvdb,
  wvdbPointLods,
  bakeWvdb,
  wvdbSourceFromVdbGrid,
  wvdbSourceFromNvdb,
  vdbToWvdb,
  nvdbToWvdb,
  wvdbDenseVolume,
  wvdbDenseToNvdb,
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
  BakeWvdbOptions,
  VdbToWvdbOptions,
  WvdbDenseToNvdbOptions,
  WvdbNvdbGrid,
} from "./wvdb/index.js";

/**
 * K3.5: mjurczyk's measured media for the frame renderer (`VDBFrameVolume` props): the presets with the
 * cited measurements behind them, a preset's props by name, and a medium from your own measurements.
 * `@pmndrs/vdb/three` `vdbFrameMedium(name)` gives the same props as three colours and vectors.
 */
export { vdbFrameMediumPresets, vdbFrameMediumNames, vdbFrameMediumProps, vdbFrameMediumFromMeasurements } from "./frame/medium.js";
export type { VDBFrameMediumPreset, VDBFrameMediumProps, VDBFrameMediumMeasurement, VDBFrameMeasured } from "./frame/medium.js";
/**
 * K3.6: the blackbody colour of a temperature (the emission colour the frame renderer's 'planck' / 'soot' hook uses),
 * and (D33) the CIE 1931 2° colour matching functions it integrates: by default the analytic fit of Wyman, Sloan &
 * Shirley (JCGT 2(2), 2013), `vdbCie1931Fit`; any `VDBColorMatchingFunctions` may be passed instead.
 */
export { vdbFrameBlackbodyColor, vdbCie1931Fit } from "./frame/blackbody.js";
export type { VDBColorMatchingFunctions } from "./frame/blackbody.js";
