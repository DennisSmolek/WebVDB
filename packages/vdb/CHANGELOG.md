# Changelog

All notable changes to `@pmndrs/vdb` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/) (while 0.x, minor releases may break).

## [Unreleased]

First public release (0.1.0). Our earlier v1 unscoped packages were never published (the npm `webvdb` package is mjurczyk's, not ours);
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
  Fp16 and FpN; `vdbResampleGrid`, `vdbCollapseGrid`, `vdbTransform`, `vdbInspect`. `vdbBuildFromDense` and
  `vdbBuildFromLeavesDetailed` take `voxelSize` as a number or `[x, y, z]` (an anisotropic, diagonal Map; G15a).
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
- Coarse-first into the froxel renderer without a rebuild (G16a): `vdbSparseSource(..., { followImage: true })`
  re-reads its boxes and rebuilds its pyramid after a `pool.replace` changes the grid's box (boxes, sizes and LOD levels
  are uniforms then), and a `VDBFrameVolume` whose box is the source's follows it. `VDBCoarseFirstEvent.types` lists
  the family's grid types for `new VDBGridNode(renderer, pool, { types })`, so one node, source and volume take the
  preview and the full grid with no shader or pipeline built at the swap.
- The froxel renderer, ported from mjurczyk/openvdb and re-based on the sparse pool: `VDBFrameVolume` and
  `VDBFrameRenderer` (`vdbFrameRenderer`) with `'volume'` and `'frame'` lighting, temporal accumulation, phase
  functions including Lorenz-Mie, multiple scattering (two-sided, octaves, neural), spectral media and measured
  presets (`vdbFrameMedium`), mesh shadows, and emission light (`VDBFrameEmissionLight`, blackbody).
- Volume sources with LOD: `vdbDenseSource` / `vdbDenseTextures` (dense) and `vdbSparseSource` (coarse pyramid,
  node-average LOD, optional HDDA empty-space skipping).
- `VDBLevelSetMaterial` (sphere-traced level-set surfaces lit by three's lights and shadows) and `VDBPoints`.
- Froxel renderer API (K5.9): one place for the phase (the renderer's `settings.phase*` is every volume's; a volume's
  `phase*` props are optional `'volume'`-lighting overrides, `volume.getPhase()`); `volume.setMedium(name | props |
  null)` applies a measured medium in one call (albedo, extinction spectrum, the renderer's phase, spectral on);
  `densityMean` fixes the mean `normalizeDensity` divides by (one mean for a sequence); `volume.bbox` is the resolved
  box; the volume follows a `source.replace()` by itself (`syncSource`, per draw).
- `VDBSequence<TFrame>` plays any frame type; `start({ source })` swaps dense `.wvdb` frames into a
  `vdbDenseTextures` source. A dense `replace` reads a channel the frame lacks as its absent value (emission 0, mask
  and albedo 1) and ignores a channel the source lacks. A `seek` while stopped shows the frame once decoded.
- `VDBSequence` frame range and ping-pong: `range: [start, end]` (inclusive; default all frames) limits playback, and
  `loopMode: 'repeat' | 'pingpong'` (default `'repeat'`) turns around at the range end (each end frame shown once per
  turn). `setRange()` / `setLoopMode()` change them while playing; the prefetch ring follows the playback order. New
  types `VDBSequenceRange`, `VDBSequenceLoopMode`.
- `wvdbDenseToNvdb(...).sparseSourceOptions` (`@pmndrs/vdb/tools`): the index box, dense stats and box that line a
  sparse copy of a `.wvdb` level up with the dense source, spread into `vdbSparseSource`.

### Changed

- `vdbDecodeToAtlasGpu` / `vdbDecodeToAtlas` / `vdbDecodeToAtlasCpu` sample voxel centres: cell `i` reads index
  position `min + (i + 0.5) * (max - min + 1) / n - 0.5`, so at full resolution it is voxel `min + i` (v1 sampled
  half a voxel past, and the last cell read the background). Atlases span the box `[min - 0.5, max + 0.5]` instead
  of `[min, max + 1]`.

### Fixed

- `wvdbDenseToNvdb` gave the grid one voxel size (the x side's), so a `.wvdb` level with non-cubic voxels (any
  default bake: each side's texel count rounds on its own) read back shifted and stretched in world space. Each axis
  now gets its own voxel size; smoke.vdb's default 128 bake measures 42.3% with `vdbResampleError` (was 45.4%; a
  `dimensions: [128, 128, 128]` bake 37.4%, was 92.4%) (G15a).
- `VDBFrameVolume.densityScale` had no effect on a sparse source: the density uniform's per-render stats refresh
  overwrote the per-draw value. A new mean from a replaced image is now applied once, when its readback lands (K5.9a).
- Mesh shadows with `multipleScattering: 'neural'` and temporal/jitter in `'frame'` lighting failed WGSL validation
  in `VDBFrameTransmittanceMaterial` ("unresolved value 'nodeUniform0'"): the network's layout function is now built
  per material build (K5.9b).

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
