# Changelog

All notable changes to `@pmndrs/vdb` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/) (while 0.x, minor releases may break).

## [Unreleased]

First public release (0.1.0). The earlier v1 unscoped packages (`webvdb` and siblings) were never published to npm;
`@pmndrs/vdb` is the first published name. MIT licensed. Peer: `three` r187 (optional, only for `@pmndrs/vdb/three`).
The renderer, `.wvdb` format and level-set and point code derive from
[mjurczyk/openvdb](https://github.com/mjurczyk/openvdb) (MIT); the WGSL traversal derives from
emcfarlane/webgpu-nanovdb (Apache-2.0). See `THIRD_PARTY_NOTICES.md`.

### Added

One package with three entry points plus `@pmndrs/vdb/pnanovdb.wgsl`: `@pmndrs/vdb` (engine-free WebGPU core),
`@pmndrs/vdb/tools` (pure-TS CPU tools, no GPU, no `three`) and `@pmndrs/vdb/three` (three.js adapter).

**Core (`@pmndrs/vdb`)**

- `VDBGridPool`: one storage buffer holding a grid table and any number of NanoVDB grid images (add, replace,
  remove, setTransform, grow, version and revision counters, host-buffer mode, dispose) with a versioned GPU layout.
- Generated WGSL modules (`vdbModuleWgsl`, `vdbModuleEntries`, `vdbCheckModule`) per profile (`read`, `sample`,
  `march`) for Float, Fp4, Fp8, Fp16 and FpN grids, every symbol prefixed `vdb`, slot and buffer binding modes, with
  a size budget guarded in CI. Opt-in helpers: `fastTrilinear`, `levelSet`, `cellBoundTrilinear`, `hddaCellEnd`,
  `rayClipSupport`, `nodeAverage`.
- GPU compute: `vdbGridStatsGpu`, `vdbDecodeToAtlasGpu`, the box pyramid `vdbCreateBoxPyramid`, and
  `vdbApplyValueTransform`; `vdbDenseSource` for dense volume sources.
- `VDBError` with stable `code`s.

**Tools (`@pmndrs/vdb/tools`)**

- `parseVdb`: OpenVDB `.vdb` files including Blosc and zlib compression, active tiles, vector grids, point grids
  and half-float storage.
- Build and quantize: `vdbBuildFromDense`, `buildFromVdb`, `vdbBuildFromLeavesDetailed`, `vdbQuantize` to Fp4, Fp8,
  Fp16 and FpN; `vdbResampleGrid`, `vdbCollapseGrid`, `vdbTransform`, `vdbInspect`.
- `.nvdb`: `NvdbFile` (NONE and ZIP codecs), `NvdbStreamReader` / `readNvdbStream` (emits each grid as it
  completes), `writeNvdb` (with the ZIP codec), and coarse-first baking with `vdbBakeCoarseFirst`.
- `.wvdb` (mjurczyk/openvdb spec): `readWvdb`, `writeWvdb`, `WvdbStreamReader`, `bakeWvdb`, and converters
  (`vdbToWvdb`, `nvdbToWvdb`, `wvdbDenseToNvdb`).
- CPU reads, samplers, stats and atlas: `vdbReadValueCpu`, `vdbReadValueF32`, `vdbSampleTrilinear`,
  `vdbSampleNearest`, `vdbGridStatsCpu`, `vdbDecodeToAtlasCpu`.
- Medium presets (`vdbFrameMediumPresets`, `vdbFrameMediumFromMeasurements`) and blackbody colours from an analytic
  CIE 1931 fit (`vdbFrameBlackbodyColor`, `vdbCie1931Fit`).
- One error hierarchy: `VDBToolsError` with `VdbFormatError`, `NvdbFormatError`, `WvdbFormatError` and
  `VDBUnsupportedError`.

**three.js adapter (`@pmndrs/vdb/three`)**

- `getDevice(renderer)` and `getGPUTexture` so tools share one GPU device; `VDBGridNode` with the `vdbSample` and
  `vdbReadValue` TSL helpers.
- `VDBVolumeMaterial` (with `vdbProxyGeometry`), `VDBSequence` (frame playback on `pool.replace`), and compute
  wrappers (`vdbGridStats`, `vdbDecodeToAtlas`).
- Coarse-first streaming: `vdbStreamCoarseFirst` and `vdbCoarseFirstApply`, with `bind` to update material, mesh and
  node after each swap.
- The froxel renderer, ported from mjurczyk/openvdb and re-based on the sparse pool: `VDBFrameVolume` and
  `VDBFrameRenderer` (`vdbFrameRenderer`) with `'volume'` and `'frame'` lighting, temporal accumulation, phase
  functions including Lorenz-Mie, multiple scattering (two-sided, octaves, neural), spectral media and measured
  presets (`vdbFrameMedium`), mesh shadows, and emission light (`VDBFrameEmissionLight`, blackbody).
- Volume sources with LOD: `vdbDenseSource` / `vdbDenseTextures` (dense) and `vdbSparseSource` (coarse pyramid,
  node-average LOD, optional HDDA empty-space skipping).
- `VDBLevelSetMaterial` (sphere-traced level-set surfaces lit by three's lights and shadows) and `VDBPoints`.

### Changed

- `vdbDecodeToAtlasGpu` / `vdbDecodeToAtlas` / `vdbDecodeToAtlasCpu` sample voxel centres: cell `i` reads index
  position `min + (i + 0.5) * (max - min + 1) / n - 0.5`, so at full resolution it is voxel `min + i` (v1 sampled
  half a voxel past, and the last cell read the background). Atlases span the box `[min - 0.5, max + 0.5]` instead
  of `[min, max + 1]`.

### Notes

- `vdbReadValueCpu` returns the exact f64 decode; `vdbReadValueF32` matches the GPU evaluation order. The two can
  differ by 1 ulp on Fp grids.
- Public names follow D32: library API is `VDB*` / `VDB_*` / `vdb*`; anything bound to one file format keeps its
  prefix (`Vdb*` for `.vdb`, `Nvdb*` for `.nvdb`, `Wvdb*` for `.wvdb`). Each entry point exports only the
  user-facing API.
- The `majorant` WGSL profile is not in 0.1 (`VDB_BAD_OPTION`).
- `@webgpu/types` is a dependency so the shipped declarations type-check in your project.
- The vendored `pnanovdb.wgsl` keeps its Apache-2.0 licence and NOTICE, shipped in the package, as does
  `THIRD_PARTY_NOTICES.md`.
