# @pmndrs/vdb spec (draft)

Status: draft, 2026-11-18. It replaces the v1 spec ([docs/SPEC.md](../v1/SPEC.md)) for new work. Names
marked *(sketch)* are proposals for K1/K2 design tasks to settle; this text is a starting point, not a contract.

## 1. What it is

Sparse volumes (OpenVDB/NanoVDB) on WebGPU, for any engine. Three layers, one published package (D9):

```
@pmndrs/vdb/tools   CPU, pure TypeScript, runs in Node and the browser
  parse .vdb · build NanoVDB · quantize Fp8/FpN · transform · inspect
  read/write .nvdb · read/write .wvdb · CPU oracle (readValue, sampleTrilinear, majorant walk)
        │  the contract: NanoVDB grid images (Uint32Array), unchanged from v1
        ▼
@pmndrs/vdb         WebGPU core, engine-free
  VDBGridPool: one storage buffer, grid table + images
  generated WGSL modules: read · sample · march · majorant
  compute: stats · value edits · dense decode · majorant build
  sequence playback · shared shading WGSL (phase, scattering, spectral; K3)
        │  GPUDevice + GPUBuffer + WGSL strings
        ▼
@pmndrs/vdb/three   three.js adapter (later ./babylon, ./pixi)
  getDevice(renderer) · VDBGridNode · vdbSample() TSL · volume material · sequence binding
  renderer ported from three-openvdb (D18): froxels, scattering, mesh shadows, emission
```

## 2. Rules of the core

From D8 and D10, following `@pmndrs/bvh`, upscaler, denoiser and downsampler:

- It takes the host's `GPUDevice` and never creates or destroys one. `isSupported(device)` and
  `requiredLimits(adapter)` let the host size its device: the storage binding size, plus `shader-f16` where it
  pays off.
- It records into the host's `GPUCommandEncoder`. A self-submitting convenience method may exist, as in
  upscaler.
- It owns only what it allocated: `dispose()` frees exactly that, and a host buffer passed in is "checked, never
  destroyed" (downsampler).
- Async creation (`await VDBGridPool.create(device, options)`). Errors carry a code and an actionable message.
- No `three` import anywhere under core or tools (lint rule, K0.3).

## 3. GPU contract (D10)

### 3.1 The pool buffer *(sketch)*

```
[header: magic, layoutVersion, gridCount, tableOffset]   16-byte aligned
[grid table: per grid { imageOffset, imageSize, gridType, flags, worldToIndex 3x4 row-major,
                         majorantOffset? }]
[grid image 0][pad][grid image 1][pad] …                  NanoVDB images, unchanged bytes
[optional majorant grids]
```

- **One** `array<u32>` storage binding; every address is relative to the image's offset. The v1 traversal
  already reads through `pnanovdb_buf_t { byte_offset, size_in_words }`, so a base offset per grid is native to
  the fork.
- `GPU_LAYOUT_VERSION` is bumped on any layout change; a `version` counter on the pool tells hosts to rebuild
  bind groups when the buffer is reallocated.
- Per-grid transforms live in the table so instances can move a grid without touching its image (the pathtracer
  needs this).

### 3.2 Module generator *(sketch)*

```ts
const wgsl = vdbModuleWgsl({
  slot: { group: 2, firstBinding: 20 },  // the host chooses; pathtracer group 2 takes 20+
  types: ['float', 'fp8'],               // decoders for other types are compiled out
  profile: 'sample',                      // 'read' | 'sample' | 'march' | 'majorant'
  sampler: 'trilinear',                   // or 'nearest'
});
const entries = vdbModuleEntries({ slot: { group: 2, firstBinding: 20 } }); // GPUBindGroupLayoutEntry[]
```

WGSL surface, all `vdb`-prefixed *(sketch)*:

| Profile | Adds |
|---|---|
| `read` | `vdbGridCount()`, `vdbAccessorInit(g)`, `vdbReadValue(acc, ijk) -> f32`, `vdbIsActive(acc, ijk) -> bool` |
| `sample` | `vdbWorldToIndex(g, p)`, `vdbSample(acc, xyzIndex) -> f32` (trilinear or nearest; zero outside) |
| `march` | HDDA (`vdbHddaInit/Step`), ray clip, node stats (`vdbNodeMax`, …) |
| `majorant` | `vdbMajorantSegment(ray, tMin, tMax) -> {t0, t1, sigmaMaj}` for delta and ratio tracking |

The size budget per profile is set in K1.1 with the pathtracer. Its kernels are 41–55 KB; a 10 KB reader is the
case it flagged. Every symbol has a CPU twin in `tools` with the same semantics (D11).

### 3.3 Compute

`gridStats`, `valueTransform`, `decodeToAtlas` (v1, moved off three), plus `buildMajorantGrid` (K3.2).
Each takes a pool and grid id, records into the host's encoder, and returns its outputs or a promise for
readbacks outside timed regions.

## 4. Adapters

### 4.1 three.js (`./three`)

Shaped like the upscaler and denoiser adapters:

- `getDevice(renderer)` and `getGPUTexture(renderer, tex)`: the same helpers the other tools use, so every tool
  shares one device. The error reads "requires an initialized WebGPURenderer (await renderer.init())".
- `VDBGridNode`: binds a pool's buffer into TSL; `.module` is the WGSL to include and `.update()` re-binds when
  `pool.version` changes.
- `vdbSample(gridNode, positionNode)`: the lowercase TSL helper.
- `VDBVolumeMaterial` (v1's `NanoVDBVolumeMaterial`) and `VDBSequence` (v1's `NanoVDBSequence`), now on the
  core.
- three is a peer dependency, typed structurally where possible (downsampler's `ThreeRendererLike`).

### 4.2 Renderer (K3, D18)

The `./three` renderer is mjurczyk's `FrameVolume`/`FrameRenderer` (three-openvdb, MIT), ported and re-based:

- It reads density through a **volume source** *(sketch)*: `VDBGridPool` (sparse, full resolution) or `.wvdb`
  levels (dense, his format). Existing `.wvdb` users keep their assets.
- Engine-free shading (phase functions, spectral media, scattering approximations, froxel math) is generated
  WGSL in `core`; the three.js layer adds depth, mesh shadows and emission onto meshes.
- The sparse source adds HDDA empty-space skipping.
- Each technique is A/B-tested against his output on the same asset (D11); ported files carry attribution.

### 4.3 Later

`./babylon` and `./pixi` follow upscaler's and bvh's adapters (R6). A raw-WebGPU example is the reference for
the engine contract (K1.7).

## 5. Formats (D14)

| Format | Role | Read | Write |
|---|---|---|---|
| `.nvdb` | Exact sparse grids; the Houdini, Blender and EmberGen export | ✅ v1 | ✅ v1 (float, Fp8, FpN) |
| `.vdb` | Source files | ✅ v1, float only; active tiles, vectors and points in K4 | R4 (browser `.vdb` writer) |
| `.wvdb` | mjurczyk's delivery format, spec v0.5: resampled, 8-bit, LOD-streamed | K4.3 | K4.3 |

## 6. Testing

- CPU oracle tests (vitest, Node) for every WGSL entry and every format path.
- GPU parity in headless Chromium on SwiftShader in CI (v1 harness), and on hardware in quiet passes.
- Goldens for demos; re-baselining needs a DIVERGES entry.
- A WGSL size snapshot guarded in CI (K1.5).

## 7. Open questions

1. ~~The package scope and repo home~~: `@pmndrs/vdb`, `pmndrs/vdb` (D17). When to move (K0.7).
2. MIT for our code (D13, K0.1).
3. The WGSL size budget per profile (K1.1).
4. Majorants: node-max HDDA or brick grid, settled by A/B (K3.2).
5. How coarse-first `.nvdb` delivery should work: a separate preview grid, or a NanoVDB-native LOD (K4.5).
6. The volume-source interface between sparse and dense sources (K3.1).
