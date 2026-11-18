# KumoVDB spec (draft)

Status: draft, 2026-11-18. It replaces the v1 spec ([docs/SPEC.md](../docs/SPEC.md)) for new work. Names
marked *(sketch)* are proposals for K1/K2 design tasks to settle; this text is a starting point, not a contract.

## 1. What it is

Sparse volumes (OpenVDB/NanoVDB) on WebGPU, for any engine. Three layers, one published package (D9):

```
@pmndrs/kumovdb/tools   CPU, pure TypeScript, runs in Node and the browser
  parse .vdb · build NanoVDB · quantize Fp8/FpN · transform · inspect
  read/write .nvdb · read/write .wvdb · CPU oracle (readValue, sampleTrilinear, majorant walk)
        │  the contract: NanoVDB grid images (Uint32Array), unchanged from v1
        ▼
@pmndrs/kumovdb         WebGPU core, engine-free
  KumoGridPool: one storage buffer, grid table + images
  generated WGSL modules: read · sample · march · majorant
  compute: stats · value edits · dense decode · majorant build
  sequence playback · reference volume renderer (K5)
        │  GPUDevice + GPUBuffer + WGSL strings
        ▼
@pmndrs/kumovdb/three   three.js adapter (later ./babylon, ./pixi)
  getDevice(renderer) · KumoGridNode · kumoSample() TSL · volume material · sequence binding
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
- Async creation (`await KumoGridPool.create(device, options)`). Errors carry a code and an actionable message.
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
const wgsl = kumoModuleWgsl({
  slot: { group: 2, firstBinding: 20 },  // the host chooses; pathtracer group 2 takes 20+
  types: ['float', 'fp8'],               // decoders for other types are compiled out
  profile: 'sample',                      // 'read' | 'sample' | 'march' | 'majorant'
  sampler: 'trilinear',                   // or 'nearest'
});
const entries = kumoModuleEntries({ slot: { group: 2, firstBinding: 20 } }); // GPUBindGroupLayoutEntry[]
```

WGSL surface, all `kumo`-prefixed *(sketch)*:

| Profile | Adds |
|---|---|
| `read` | `kumoGridCount()`, `kumoAccessorInit(g)`, `kumoReadValue(acc, ijk) -> f32`, `kumoIsActive(acc, ijk) -> bool` |
| `sample` | `kumoWorldToIndex(g, p)`, `kumoSample(acc, xyzIndex) -> f32` (trilinear or nearest; zero outside) |
| `march` | HDDA (`kumoHddaInit/Step`), ray clip, node stats (`kumoNodeMax`, …) |
| `majorant` | `kumoMajorantSegment(ray, tMin, tMax) -> {t0, t1, sigmaMaj}` for delta and ratio tracking |

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
- `KumoGridNode`: binds a pool's buffer into TSL; `.module` is the WGSL to include and `.update()` re-binds when
  `pool.version` changes.
- `kumoSample(gridNode, positionNode)`: the lowercase TSL helper.
- `KumoVolumeMaterial` (v1's `NanoVDBVolumeMaterial`) and `KumoSequence` (v1's `NanoVDBSequence`), now on the
  core.
- three is a peer dependency, typed structurally where possible (downsampler's `ThreeRendererLike`).

### 4.2 Later

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

1. The package scope, the repo home, and when to move (D9, K0.1).
2. MIT for our code (D13, K0.1).
3. The WGSL size budget per profile (K1.1).
4. Majorants: node-max HDDA or brick grid, settled by A/B (K3.2).
5. How coarse-first `.nvdb` delivery should work: a separate preview grid, or a NanoVDB-native LOD (K4.5).
6. Whether the joint project takes the WebVDB name (D7, R10).
