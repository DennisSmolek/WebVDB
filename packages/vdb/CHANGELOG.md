# Changelog

All notable changes to `@pmndrs/vdb` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/) (while 0.x, minor releases may break).

## [Unreleased]

First public release (0.1.0). The earlier v1 unscoped packages (`webvdb` and
siblings) were never published to npm; `@pmndrs/vdb` is the first published name.

### Changed

- `vdbDecodeToAtlasGpu` / `vdbDecodeToAtlas` / `vdbDecodeToAtlasCpu` sample voxel
  centres: cell `i` reads index position `min + (i + 0.5) * (max - min + 1) / n - 0.5`,
  so at full resolution it is voxel `min + i` (v1 sampled half a voxel past, and the
  last cell read the background). Atlases span the box `[min - 0.5, max + 0.5]`
  instead of `[min, max + 1]`.

### Added

- One package with three entry points: `@pmndrs/vdb` (engine-free WebGPU core),
  `@pmndrs/vdb/tools` (pure-TS CPU tools) and `@pmndrs/vdb/three` (three.js
  adapter, `three` is an optional peer at r187). Also `@pmndrs/vdb/pnanovdb.wgsl`.
- Core: `VDBGridPool`, one storage buffer holding a grid table plus any number of
  NanoVDB grid images (add, replace, remove, setTransform, grow, version and
  revision counters, host-buffer mode, dispose), with a versioned GPU layout.
- Core: generated WGSL modules per feature set (`read`, `sample`, `march`
  profiles; Float, Fp8 and FpN grids), with every symbol prefixed `vdb`, slot and
  buffer binding modes, a size budget and snapshot guarded in CI. Compute tools
  `vdbGridStatsGpu`, `vdbDecodeToAtlasGpu`, `vdbApplyValueTransform` and the GPU
  coarse pyramid `vdbCreateBoxPyramid`.
- three.js adapter: `VDBGridNode` and the `vdbSample()` TSL helper,
  `VDBVolumeMaterial` (with `vdbProxyGeometry`), `VDBSequence` (frame playback on
  `pool.replace`), compute helpers (`vdbGridStats`, `vdbDecodeToAtlas`), and
  `getDevice(renderer)` / `getGPUTexture` so tools share one GPU device.
- Tools: `.nvdb` reader and writer; `.vdb` (OpenVDB) parser including Blosc and
  zlib compression, active tiles, vector grids and point grids; build from dense
  data or from a `.vdb`; Fp8/FpN quantize; inspect and transform; resample-to-
  resolution bake (point, box and trilinear filters).
- Tools: `.wvdb` read and write per the mjurczyk/openvdb spec (v0.5, plus points).
- Raw-WebGPU example with no engine, and the three.js demos.

### Notes

- `vdbReadValueCpu` returns the exact f64 decode; `vdbReadValueF32` matches the
  GPU's evaluation order. The two can differ by 1 ulp on Fp grids.
- Public names follow D32: library API is `VDB*` / `VDB_*` / `vdb*`; anything bound
  to one file format keeps its prefix (`Vdb*` for `.vdb`, `Nvdb*` for `.nvdb`,
  `Wvdb*` for `.wvdb`), and format readers and writers may read as a verb and the
  format (`parseVdb`, `writeNvdb`, `readWvdb`). Each entry point exports only the
  user-facing API; the CPU twins of the WGSL used as test oracles are not
  exported.
- Every error `@pmndrs/vdb/tools` throws on bad input is a `VDBToolsError` with a
  `code` (`VdbFormatError`, `NvdbFormatError`, `WvdbFormatError` and
  `VDBUnsupportedError` extend it); the core throws `VDBError`.
- Code is MIT licensed. The vendored `pnanovdb.wgsl` keeps its Apache-2.0 licence
  and NOTICE, which are shipped in the package.
