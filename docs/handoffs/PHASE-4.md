# Handoff: Phase 4 — compute toolset + fallback

Gate: **PASSED** — demo 04 renders the same cloud via
compute→atlas→volume sampling, and the stats utility validates against
sidecars (exact activeVoxelCount, min/max/mean ≤1e-3 vs full CPU scan).
Suite at close: 177 unit / 7 e2e (goldens stable ×2) / clean tsc.

## What exists

`packages/three-nanovdb/src/compute.ts` — all raw-WebGPU compute (not
TSL), 3D dispatch, shared library assembly with the material path:

| API | Notes |
|---|---|
| `gridStats(device, grid, pnanovdbSource, opts?)` | root-stats pre-read → non-negative atomic fast path (bitcast min/max, carry-checked u64 sum, 256-bin histogram) or dump+CPU-reduce fallback; `usedAtomicFastPath` reported |
| `valueTransform(device, grid, pnanovdbSource, wgslBody)` | Float-only, leaf voxels only (tile-backed uniform regions untouched — enforced by test); returns a fresh image; baked node stats go stale (documented) |
| `decodeToAtlas(device, grid, pnanovdbSource, opts?)` | dense decode over the index bbox (maxDim clamp), trilinear/nearest, optional [min,max]→u8 normalization; returns typed array + dims + range |

All three take `pnanovdbSource` explicitly (package never does `?raw`
imports itself — same convention as `NanoVDBVolumeMaterial`).

## Environment findings (retest on real hardware)

- `Data3DTexture` + TSL `texture3D()` fails Dawn validation in this
  sandbox (2D view vs 3D texture, on upload and submit) — demo 04
  samples the atlas from a storage buffer via `wgslFn` instead;
  `decodeToAtlas` itself is unaffected. The SPEC-pictured
  `VolumeNodeMaterial` bridge should be retried on hardware.
- three r185 `NodeMaterial.setupLighting()` short-circuits without ≥1
  light in the scene — a zero-intensity `AmbientLight` un-sticks it.
- Both utilities iterate the dense index bbox (fine at fixture scale);
  leaf-walking dispatch is the known upgrade for huge sparse grids.

## Phase 5 state (parallel work, this same wave)

- 5a `.vdb` parser: DONE — four openvdb.org samples parse, voxel counts
  match file metadata exactly; dependency-free inflate (decision to
  keep vs swap to fflate is open); blosc/tiles/non-float throw clearly.
- 5b serializer: DONE — buildFromDense + writeNvdb, full-voxel
  round-trip through the proven readers, structural checker passes on
  native fixtures; LeafCodec seam ready.
- REMAINING (wave 2): `buildFromVdb` (parser→serializer), Fp8/FpN
  quantization via LeafCodec, `inspect`, affine `transform`, demo 06/07
  cores, and the deferred native anchor: `nanovdb_convert` byte/value
  parity on a machine with the toolchain.
