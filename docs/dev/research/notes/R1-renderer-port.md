# R1 — Port plan for mjurczyk's volume renderer

Feeds K3 (D18). Written 2026-11-18 from a read-only study of the upstream source: nothing of it was installed,
built or run. Line counts are `wc -l` at the pinned commit. Upstream paths in §3–§5 are relative to
`src/three-openvdb/` unless they start with `src/` or `scripts/`; our paths are relative to this repo.

## 0. Summary

- **His renderer is a froxel pipeline, not a ray marcher.** Every pass is a fragment pass into 2D atlases of
  z-slices (WebGL heritage). Default (`lighting: 'frame'`): volumes are injected into one **world-space dense
  shadow grid** (64³, up to 4 cascades), transmittance is marched through *that grid*, then froxels are lit
  from it. Only **four passes read volume data**. Everything else (multiple scattering, phase functions,
  spectral, neural, mesh shadows, environment) reads the grid or the froxels, so it does not depend on the
  volume source.
- **The seam is one factory:** `createVolumeNodes()` in `frame/webgpu/frameShaderChunks.js` (L834–1028). It
  returns `sampleExtinction / sampleAlbedo / sampleEmission(localPos, lod)` plus box helpers. Its callers are
  `FrameShadowDensityMaterial`, `FrameDensityMaterial`, `FrameInjectMaterial` and `FrameEmissionLight`.
- **The LOD (`lod` argument) is the hard part, not trilinear.** Every read passes a mip level derived from the
  footprint (grid voxel, froxel width or step length). Sparse level-0-only sampling would alias badly in the
  shadow-grid injection. A sparse source needs a prefiltered coarse representation. Two candidates are in §4.3,
  to be settled by an A/B.
- **No delta tracking, no majorants, no gradients** in the fog renderer. K1's `majorant` profile is not needed
  for K3; only the pathtracer needs it. HDDA (`march`) pays off in `'volume'` lighting's shadow march and in
  empty-froxel early-outs.
- **Bucket sizes** (`frame/core` + `frame/webgpu`, 11,479 lines): **A** engine-free shading ≈ 5,170 (≈ 300 of it
  generated data), **B** three.js integration ≈ 4,770, **C** dense data path ≈ 1,390 (≈ 1,670 with
  `wvdb/WVDBLoader.js`), D 145. Outside these directories, **D** (other) is the WebGL backend (8,136), level sets,
  points, loaders, the parser and the CLI.
- **Shading language:** the WebGPU path is TSL only, with no `wgslFn`. The WebGL path is GLSL strings and the
  same algorithms, so it is the closer text to translate into WGSL. Lifting into the core means rewriting the
  A-bucket as WGSL with CPU twins, then calling it from TSL through `wgslFn`, as v1's `packages/vdb/src/three/material.ts`
  already does.
- **Proposed port order** (§6): dense-source parity harness → froxels with `'volume'` lighting → sparse source
  plus LOD A/B → `'frame'` lighting and shadow grid → phase functions and all multiple-scattering modes →
  spectral → mesh shadows and emission → level sets and points. This moves multiple scattering and phase
  functions ahead of mesh shadows and emission (see §6 for why).

## 1. Pinned source

| | |
|---|---|
| Repo | https://github.com/mjurczyk/openvdb (shallow clone, read only) |
| Commit | `75e73454f2d6f4384c9ffd57b704319767499360` — "Release v0.26.0", 2026-11-18 06:55 UTC |
| Version | 0.26.0 (workspace and all three packages) |
| Licence | MIT in all three `package.json`; no LICENSE file in the repo. Authors: `mjurczyk`; contributor `notchris` |
| three | peer `>=0.162.0` (three-openvdb); examples lockfile pins **three 0.187.0**. We are on r185 |

Packages (npm workspaces under `src/`):

| Package | Path | Role | Entry points |
|---|---|---|---|
| `openvdb` | `src/openvdb` (4.1k lines + an 8k-line `dependencies.js` bundle) | `.vdb` reader, `GridDescriptor`, noise, primitives, `.wvdb` format codec | `.` → `index.js` |
| `three-openvdb` | `src/three-openvdb` | the renderer, `.wvdb` loaders, level sets, points | `.` → `index.js` (**WebGPURenderer**, `three/webgpu` + TSL); `./webgl` → `webgl.js` (WebGLRenderer, GLSL) |
| `webvdb` | `src/wvdb` (`cli.js` 435, `bake.js` 658, `zip.js` 145) | `.vdb` → `.wvdb` baking CLI | bin |

Both entries export the same API. `setFrameBackend()` (`frame/core/frameBackend.js`) picks `frame/webgpu/` or
`frame/webgl/`, and the shared code is `frame/core/`. Upstream also has a Playwright suite of **187 scenarios with
JPEG references** (`tests/validation.spec.js`, `tests/validation-set/`; a failure is a channel difference above
48 on more than 1% of pixels), which we can reuse for the A/B (§6).

## 2. How the renderer works

`FrameVolume` (a `THREE.Group`) **resamples every grid on the CPU into a dense 8-bit `Data3DTexture`**. The
default `resolution` is 100 texels on the longest box side. Values are `clamp(value / scale, 0, 1) * 255`, and
level sets become 1 on active voxels. It also builds an optional CPU mip atlas (`frameMipmaps.js`, box or
Gaussian). A proxy mesh's `onBeforeRender` calls `frameRenderer.render()` once per frame. The passes for
`lighting: 'frame'` (the default), in `FrameRenderer.renderFrameLighting()`:

1. **Shadow density** (`FrameShadowDensityMaterial`): for each world shadow-grid voxel in each caster's box,
   `sampleExtinction(local, lod = log2(gridVoxel / texel))` → (extinction, absorption) into an RGBA16F z-slice
   atlas. It is redrawn only when a signature changes (`frameGrid.updateGridSignatures`). **Reads the source.**
2. **Transmittance** (`FrameTransmittanceMaterial`): per grid voxel, a 1-voxel-step march through the grid toward
   ≤ 6 light slots plus sky up and down. Multiple scattering (diffusion, two-sided, octaves, neural) is applied
   here. The output is total and direct transmittance per slot.
3. **Density** (`FrameDensityMaterial`): per froxel per volume (instanced tile quads, one per slice), up to
   `subSteps` samples of `sampleExtinction` across the slice. It also samples albedo and emission at the
   visibility-weighted position, plus the grid shadows there. Output is 4 MRT targets. **Reads the source.**
4. **Lighting** (`FrameLightingMaterial`): each froxel is lit once from the combined data, using the phase,
   the slots and PMREM irradiance.
5. **Temporal** (`FrameTemporalMaterial`): reprojects into the previous froxel grid, with a neighbourhood
   clamp and `temporalBlend` 0.2.
6. **Integrate** (`FrameIntegrateMaterial`, on the proxy): front-to-back over the log-spaced slices per pixel.
   It stops at `viewportDepthTexture()` and composites, with optional sRGB blending.

With `lighting: 'volume'` the first four steps are replaced by **`FrameInjectMaterial`**: each volume lights its
own froxels and marches its own shadows **through the source texture** (`shadowSteps` 32, with the LOD growing
with step length). This mode **reads the source** in the march. Mesh shadows (`FrameVolumeShadows`) and the
neural inputs read the grid's last cascade. **`FrameEmissionLight` reads the source** for its direction pass.

Defaults: froxels 512×512×24, `subSteps` 32, `shadowResolution` 64, `shadowCascades` 1, `multipleScattering`
`'neural'`, `phase` HG g = 0.8, `temporal` on, `jitter` off, `castShadows` off.

## 3. Module map (WebGPU path plus shared code)

Bucket legend: **A** engine-free shading → WGSL in `core`; **B** three.js integration; **C** dense-texture data
path, replaced by the volume source; **D** other. "Split" lists the parts that belong in a second bucket.

### 3.1 `src/three-openvdb/frame/core/` (plain JS, imports `three` for math types only unless noted)

| File | Lines | What it does | Bucket | Split / notes |
|---|---:|---|---|---|
| `FrameVolume.js` | 651 | Group + proxy; CPU resample of grids to 8-bit 3D textures; density stats and `meanDensity`; instances (`SharedVolume`) | **C** | instance/props API → B |
| `frameMipFilter.js` | 191 | CPU Gaussian mip levels (Kallweit 2017 §6), cached, 12 ms slices | C | |
| `framePhase.js` | 129 | phase-mode → uniform packing; asymmetry; `getFrameMultipleScattering` (1−g); octave params | A | |
| `frameSpectrum.js` | 42 | Ångström exponents, per-channel extinction, spectral slot groups | A | |
| `frameMedium.js` | 90 | measured medium presets (cloud, fog, ice, soot, smokes, haze, dust) with citations | A | data |
| `frameMieTable.js` / `frameMieRgbTable.js` | 8 / 7 (18 KB / 53 KB) | generated Lorenz-Mie tables, 2048 texels, u = √(θ/π) | A | generated (§7) |
| `frameBlackbody.js` | 172 | CIE 1931 CMF table → Planck/soot emission colour LUT (256×16) | A | CIE data (§7) |
| `frameNeuralIndirect.js` | 235 | generated MLP weights 20→32→32→1 | A | generated, trained (§7) |
| `frameGrid.js` | 223 | shadow-grid cascade layout (power-of-two snapped), caster ranges, change signatures | A | uses `Box3` |
| `frameView.js` | 357 | frustum clip of volume boxes, log slice ranges, froxel area, distance fade | A | `FrameVolumeCollector` (scene traversal) → B |
| `frameLayout.js` | 55 | froxel grid size with steady-size hysteresis, atlas layout | A | takes sizes from renderer → host passes them |
| `frameEnvironment.js` | 331 | env → ≤12 directional lights (SH quadrature) from a 64×32 readback | A | `extract()` readback → B |
| `FrameEmissionLight.js` | 287 | `THREE.Light` base: emitter slots, probe cascades | B | cascade math → A |
| `FrameVolumeAnimation.js` | 267 | frame playback over instances | B | ↔ our `VDBSequence` |
| `frameLights.js` | 9 | light-type bitmask | B | |
| `VDBAnimationLoader.js` | 127 | `.vdb` sequence loader | D | |
| `frameBackend.js` | 18 | WebGL/WebGPU switch | D | dropped |

### 3.2 `src/three-openvdb/frame/webgpu/` (TSL `NodeMaterial`s; no raw WGSL, no compute)

| File | Lines | What it does | Bucket | Split / notes |
|---|---:|---|---|---|
| `frameShaderChunks.js` | 1228 | shared TSL: froxel maths, IGN noise, HG/dual-lobe/Mie phase, shadow step/depth, 1D/2D diffusion, octaves, spectral variants, grid sampling and cascades, light uniforms, **`createVolumeNodes`** | **split** | A ≈ 715 (L82–622, L1030–1206); C ≈ 340 (L688–1028); B ≈ 165 (frag-coord, light uniforms, tile vertex) |
| `FrameDensityMaterial.js` | 539 | froxel density pass, grey and spectral variants | A | env irradiance → B; calls source |
| `FrameInjectMaterial.js` | 535 | `'volume'` lighting: per-volume lighting and shadow march | A | calls source in loops |
| `FrameShadowDensityMaterial.js` | 132 | inject casters into the world grid | A | calls source |
| `FrameTransmittanceMaterial.js` | 460 | grid march to light slots, MS modes, spectral groups | A | grid only |
| `FrameLightingMaterial.js` | 438 | froxel lighting from combined data | A | PMREM → B |
| `FrameTemporalMaterial.js` | 219 | froxel reprojection and clamp | A | |
| `FrameEnvironmentGridMaterial.js` | 283 | env lights without a slot, in the last cascade | A | |
| `FrameGridMipMaterial.js` | 69 | box mips of the last cascade (neural inputs) | A | |
| `frameNeuralChunks.js` | 131 | MLP evaluation, weights as shader constants | A | |
| `FrameRenderer.js` | 1787 | orchestration: render targets, scenes, pools, signatures, light collection, settings | **B** | the pass graph is what a core compute version would replace |
| `FrameIntegrateMaterial.js` | 295 | slice march and composite against `viewportDepthTexture`, sRGB blend emulation | B | slice march ≈ 100 → A |
| `FrameVolumeShadows.js` | 1027 | Beer shadow maps on three lights via `filterNode` and `LightsNode` internals | B | |
| `FrameVolumeShadowMaterial.js` | 174 | renders Beer maps from the grid | B | march → A |
| `FrameEmissionLight.js` | 658 | `AnalyticLightNode`; direction/reduce/probe passes | B | ≈ 250 of pass kernels → A; calls source |
| `FrameEnvironmentMaterial.js` / `…MapMaterial.js` | 56 / 44 | PMREM access, env readback | B | |
| `frameMipmaps.js` | 205 | CPU box mip atlas (levels side by side on x) | C | |

### 3.3 Everything else

| Path | Lines | Bucket | Notes |
|---|---:|---|---|
| `frame/webgl/*` | 8136 | D | GLSL `ShaderMaterial` twin of the WebGPU path. Not ported, but read it when writing WGSL (`frameShaderChunks.js` there is 1017 lines of GLSL) |
| `wvdb/WVDBLoader.js` | 285 | C | `WVDBVolumeData`: swaps `Data3DTexture`s per LOD, `attach()` |
| `wvdb/WVDBReader.js`, `wvdbKernel.js`, `WVDBAnimationLoader.js` | 613 / 346 / 257 | D | streaming decode in workers → K4.3 reference |
| `levelset/*` | 984 | D | marching cubes (CPU) and `LevelSetSurfaceMaterial` (192, TSL sphere trace of an 8-bit SDF texture inside `MeshStandardNodeMaterial`) |
| `points/*` | 226 | D | `VDBPoints`: instanced sprites (`PointsNodeMaterial`) |
| `primitives/MeshVolume*.js` | 315 | D | mesh → volume via three-mesh-bvh |
| `VDBLoader.js`, `index.js`, `webgl.js` | 174 | D | |
| `src/openvdb/**` | 4104 | D | parser (K4 reference), `math/noise.js` (CPU Perlin/simplex/Worley, ported from noisejs), primitives |
| `scripts/mie-phase-table.mjs`, `scripts/neural-indirect/*` | 245 / 1638 | D | generators for the A-bucket tables and weights |

**Bucket totals** (`frame/core` + `frame/webgpu`, 11,479 lines): **A ≈ 5,170**, **B ≈ 4,770**, **C ≈ 1,390**
(≈ 1,670 with `WVDBLoader.js`), **D 145** (`VDBAnimationLoader`, `frameBackend`). The WebGL path is not counted.

**Lifting difficulty by language:** A-bucket code is TSL closures that build node graphs over shared `uniform()`
objects. No tool turns TSL into engine-free WGSL, so each function is rewritten. The translation is mechanical
(`Fn`/`If`/`Loop` → WGSL, `.toVar()` → `var`), and the GLSL twin is a second reading of each function. The
uniform objects become one `struct` per pass. The `three` adapter wraps the core WGSL with
`wgslFn` + `code()`, as v1's `packages/vdb/src/three/material.ts` does today.

## 4. The seam: where density is read

### 4.1 Every read of volume data

All of them go through `createVolumeNodes(volumeUniforms, useMipmaps)` (`frameShaderChunks.js` L834):

| Function | Coordinates | Filtering / LOD | Composition |
|---|---|---|---|
| `sampleExtinction(local, lod)` | local unit box [−0.5, 0.5]³ → `gridUv` = wrap(local + 0.5 + `offset3D`), wrap modes clamp/repeat/mirror done in shader | `texture.sample(uvw).level(0)`: **hardware trilinear on R8 unorm**. With mips: manual blend of two levels from the CPU mip atlas (`sampleMipmapped`, `sampleMipAtlas`) | `density × mask` (mask at `local + 0.5`, **no offset or wrap**), `> densityCutoff`, `× densityScale/meanDensity × extinction / worldScale` |
| `sampleAlbedo(local, lod)` | `gridUv` | same, RGBA8 | `baseColor × map.rgb` (1 when absent) |
| `sampleEmission(local, lod)` | `gridUv` | same, R8 | value → temperature → blackbody colour × (T/Tmax)⁴ × intensity |
| `getLod(localSize)` | — | `log2(localSize × max(texels))` | footprint → level; `getSampledLod` clamps to the levels that exist; `getBoxSteps` = 2 steps per voxel of that level, capped at `subSteps` |
| `insideBox`, `boxExit`, `boxSegment` | local box | — | pure geometry (A) |

Callers and the LOD each one passes:

| Caller | Where | LOD source | Count per frame |
|---|---|---|---|
| `FrameShadowDensityMaterial.createFragmentNode` | L81–126 | grid voxel size (coarse: a 64³ grid over a 1000³ asset is about lod 4) | grid voxels × casters, **only when the grid signature changes** |
| `FrameDensityMaterial.buildNodes` / `buildSpectralNodes` | L158–230, L375–457 | `max(froxel width, sub-step length)` | froxels in area × ≤ `subSteps` + 2 |
| `FrameInjectMaterial` (`'volume'` lighting) | L171–312, L521 | froxel lod, then `max(lod, getLod(step))` in the shadow march | froxels × (`subSteps` + lights × `shadowSteps`) |
| `FrameEmissionLight` direction pass | `webgpu/FrameEmissionLight.js` L153–215 | step length for extinction; **level 0 for emission** (T⁴ would be lost in mips) | 16 directions × 8² rays × `steps`, on change |

No reads anywhere use gradients, normals, majorants or delta tracking. `LevelSetSurfaceMaterial` reads its own
SDF texture and takes central differences for normals; that is separate from the fog seam.

### 4.2 Proposed interface

The renderer's shading (A) keeps the composition: cutoff, scale, extinction, wrap, blackbody. A **source** supplies
raw normalized channels at a uvw and a LOD, and a few march queries. This mirrors `createVolumeNodes`, so his
materials port with call-site edits only.

```ts
// core: engine-free
export interface VDBVolumeSource {
  readonly kind: 'sparse' | 'dense';
  readonly texels: [number, number, number];  // voxels across the box at lod 0 (sparse: index bbox extent)
  readonly lodLevels: number;                 // prefiltered levels available (0 = level 0 only)
  readonly channels: { mask: boolean; albedo: boolean; emission: boolean };
  readonly stats: { meanNonZero: number; max: number; activeVoxels: number }; // normalizeDensity; bounds
  readonly boxToSource: Float32Array;         // 4x4: local unit box → source space (sparse: index space)
  readonly version: number;                   // bumps when bindings must be rebuilt (LOD swap, pool growth)
  wgsl(o: { slot: VDBSlot; profile: 'sample' | 'march'; channels?: Partial<VDBVolumeSource['channels']> }): string;
  layoutEntries(slot: VDBSlot): GPUBindGroupLayoutEntry[];
  bindEntries(slot: VDBSlot): GPUBindGroupEntry[];
}
export function vdbSparseSource(pool: VDBGridPool, grids: { density: number; mask?: number; albedo?: number;
  emission?: number }, o?: { lod?: 'none' | 'coarse-pyramid' | 'node-average'; pyramidMax?: number }): VDBVolumeSource;
export function vdbDenseSource(device: GPUDevice, levels: WVDBLevels /* K4.3 */, o?: { mips?: 'box' | 'gaussian' }):
  VDBVolumeSource;
```

```wgsl
// Generated per source; the renderer's kernels call only these. uvw ∈ [0,1]³ of the source box, already
// offset and wrapped by the caller. lod in levels (log2 of footprint in lod-0 voxels).
fn vdbSrcDensity (v: u32, uvw: vec3f, lod: f32) -> f32;   // sparse: ≥ 0, unclamped; dense: [0, 1]
fn vdbSrcMask    (v: u32, uvw: vec3f, lod: f32) -> f32;   // compiled to 1.0 when absent
fn vdbSrcAlbedo  (v: u32, uvw: vec3f, lod: f32) -> vec3f; // compiled to vec3(1) when absent
fn vdbSrcEmission(v: u32, uvw: vec3f, lod: f32) -> f32;   // compiled to 0.0 when absent
fn vdbSrcTexels  (v: u32) -> vec3f;
fn vdbSrcLodLevels(v: u32) -> f32;
// 'march' profile only. Dense versions are trivial (return t0 / stats.max).
fn vdbSrcSkipEmpty(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32) -> f32; // next t that may hold density (HDDA)
fn vdbSrcMaxIn   (v: u32, uvwMin: vec3f, uvwMax: vec3f) -> f32;        // upper bound in a box (node max)
```

In `./three`, `vdbVolumeNodes(source, uniforms)` returns `{ getLod, sampleExtinction, sampleAlbedo,
sampleEmission, insideBox, boxExit, boxSegment }`, the exact shape of `createVolumeNodes`. The port then changes
one import per material. `v` indexes the pool's grid table for sparse sources. Dense sources bind one 3D texture
(plus a mip atlas) per channel. That breaks D10's "no textures" only for kernels compiled with a dense source, so
it is a compile-time choice.

### 4.3 What sparse cannot cheaply match

| Dense-path assumption | Sparse cost | Plan |
|---|---|---|
| Hardware trilinear (one fetch) on R8 | 8 taps through the read accessor; leaf-coherent, but about 8–20× the ALU and loads | Shared-leaf stencil fetch (one tree walk, then neighbours by offset when in the same leaf); K1 `sample` profile |
| **Mip levels at any `lod`** (box or Gaussian atlas) | NanoVDB has no mips. Node stats give averages only at levels 3 (leaf 8³), 7 (lower 128³) and 12 | **A/B two options** (CLAUDE.md "build both, then decide"): **(P) coarse pyramid**, compute-built dense levels from level L₀ = ⌈log2(maxTexels / `pyramidMax`)⌉ up (v1 `decodeToAtlas`), r16f in the pool buffer, manual trilinear, about 2 MB at `pyramidMax` 128, with levels 1…L₀−1 blended toward L₀; **(N) node averages**, trilinear over neighbouring leaf averages for level 3 and lower averages for level 7, needing stats in every grid. Baseline: `'none'` (level 0 only, aliases) |
| 8-bit `clamp(v/scale,0,1)` data, mean of the clamped values | Sparse returns true floats, so dense values above 1 differ | `normalize: 'clamp01'` compat flag for the parity A/B; default unclamped (DIVERGES row) |
| `RepeatWrapping`/`MirroredRepeat` + `offset3D` (`wrapUv`) | uvw math is source-independent; index wrap needs the bbox | S: same `wrapUv` in shading, modulo of the index bbox in the source |
| Per-volume aux grids at the **same dims** (mask, albedo, emission share the density mip layout) | Sparse aux grids have their own transforms | per-channel `boxToSource`; no shared-layout assumption |

**HDDA gains** (K1 `march` profile) are largest in `FrameInjectMaterial`'s shadow march (32 steps, mostly
empty for sparse assets). They are moderate in the density pass: `vdbSrcSkipEmpty` before the sub-step loop, and
a discard when `vdbSrcMaxIn(froxel box) == 0`. In shadow density, `vdbSrcMaxIn(grid voxel)` discards early.

## 5. Techniques

| Technique | How it works (brief) | Depends on | Perf knobs | Port onto sparse |
|---|---|---|---|---|
| **Froxels, depth-aware** | log slices fitted to the visible volumes' clipped depth range (`frameView`); instanced slice quads per volume; integrate stops at `viewportDepthTexture`; area-clipped passes | source (density pass), three depth texture | `width/height` 512, `pixelsPerFroxel`, `depth` 24, `subSteps` 32, `near/far`, `fadeRadius` | **M**: only the density pass touches the source, but froxels × substeps × 8 taps is the hot loop |
| **`'volume'` lighting** | per-volume inject with its own shadow march through the source | source, phase, MS step maths | `shadowSteps` 32, march cut at depth 8 | **M**: best place for HDDA |
| **World shadow grid** (`'frame'`) | dense cascaded grid (RGBA16F z-slice atlas); transmittance march at 1 voxel per step; signature caching skips redraws | source (injection only) | `shadowResolution` 64, `shadowCascades` 1–4, `shadowDistance` | **S** march (source-free), **M** injection (needs the LOD fix) |
| **Temporal + jitter** | golden-ratio jitter, IGN noise, froxel reprojection with neighbourhood clamp | camera matrices | `temporal`, `temporalBlend`, `jitter` | S (source-free) |
| **Environment** | `'outward'`, `'directional'` (≤ 12 lights from a 64×32 PMREM readback, SH quadrature), `'spherical-harmonics'`; env grid of the last cascade | PMREM (B), grid | `environmentSamples` 8, `environmentShOrder` | M (readback is B; maths A) |
| **MS: 1D diffusion** (`true`) | per step (σt, √(3σa(σa+σs′)), σs′) with σs′ = σs(1−g); T = max(e^{−σt}, e^{−eff}/(1+0.75·red)) | shadow march | none | S |
| **MS: two-sided** | 2D slab diffusion with Robin BCs and a side-leak mode (`frameTwoSidedDiffusion`); one more march per light | grid march | | S |
| **MS: octaves** | N ≤ 8 octaves: σ·aⁿ, bⁿ, g·cⁿ (Wrenninge-style) | phase | `scatteringOctave*` | S |
| **MS: neural** (default) | MLP 20→32→32→1 per grid voxel per light; inputs: 6 box mips of the last cascade at ±h·2ᵏ along the light, direct optical depth, √(σa/σt); runs only on grid redraw; grey `'frame'` only | grid mips, weights | `shadowResolution` drives cost (64³ × lights × ≈ 1.7k MAC) | **M**: source-free; weights packaging and compile time on D3D are the work (§8) |
| **Phase functions** | HG; dual-lobe (mix by weight); Lorenz-Mie 550 nm table and RGB tables (2048 texels, `u = √(θ/π)`, `load` + lerp); per-channel g via `phaseGExponent` | Mie table texture (or a buffer in core) | | S |
| **Spectral media + presets** | per-channel extinction `scale·(λ/550)^−α`; per-channel albedo; MS per channel; 3 transmittance targets per slot group; presets from Dubovik 2002, Hess 1998, Bond 2006 | everything above, MRT | `spectral` (about 1.5–3× transmittance passes) | M (target count, packing) |
| **Mesh shadows** | Beer shadow maps (CSM tiles, spot, point cube) marched through the grid's last cascade, multiplied in through each three light's `filterNode`; budgeted `castShadowUpdatesPerFrame` | three `LightsNode.setupLightsNode` internals, ShadowNode | `castShadowMapSize` 512, `castShadowSteps` 32, cascades | **L** (three-version-fragile; source-free) |
| **Emission lighting meshes** | per-volume emission along 16 directions × 8² rays → reduce → L1 SH probe cascades (≤ 32³, ≤ 4) read by an `AnalyticLightNode` | source (emission at lod 0), three light-node API | `resolution`, `cascades`, `steps` | **L** (B-heavy; source reads at lod 0 are fine on sparse) |
| **Emission spectra** | `'approximate'` (Shadertoy, §7), `'planck'`/`'soot'` from the CIE LUT | LUT | | S (drop or re-derive `'approximate'`) |
| **Level-set surfaces** | sphere trace of the dense SDF texture inside `MeshStandardNodeMaterial`; mesh via CPU marching cubes | dense SDF | `resolution` | **M** natively on sparse (HDDA + zero crossing on the NanoVDB level set), D otherwise |
| **Points** | instanced round sprites from `P`/`Cd` | point grids (K4) | `maxPoints` | S once K4 parses point grids |
| **Distance fade, sRGB blend emulation** | FogExp2-style fade; reads scene colour to blend in sRGB | | `fadeRadius`, `srgbBlending` | S |

## 6. Port order for K3, with an A/B per step

PHASES K3 lists: march and lighting basics → froxels → mesh shadows and emission → multiple scattering →
spectral and phase → level sets and points. **The code argues for a different order.** There is no march apart
from the froxel pipeline: `'volume'` lighting *is* the basics. Phase and multiple scattering are a few hundred
lines of source-free maths inside every lighting pass, and the default output already uses `'neural'`. Mesh
shadows and emission are about 1.9k lines of three-internals-coupled B code. The cloud system, our first
consumer, needs the A-bucket items, not mesh shadows.

| Step | Scope | A/B asset and scene | Image metric | Perf metric |
|---|---|---|---|---|
| **Step 0 Harness + dense source** | `vdbDenseSource` from our `.wvdb` reader (K4.3); capture rig for his renderer on the owner's machine (not the cloud sandbox): PNG, `jitter` off, `temporal` off, fixed camera | `bunny_cloud.wvdb` (his demo-1 default), `smoke.wvdb` | sanity: his capture vs his JPEG reference within his own threshold | — |
| **Step 1 Froxels, `'volume'` lighting** | froxel maths, inject, integrate, HG, MS off then `true`, one directional light | demo-1 `lighting-volume`, `multiple-scattering-off`, `lights-directional` on **dense** source | identical data, so strict: PSNR ≥ 40 dB and ≤ 0.5% pixels > 8/255 (linear, pre-tonemap) | GPU ms per pass (timestamp queries), median of 300 frames after 60 warm-up, 1080p, named preset |
| **Step 2 Sparse source + LOD A/B** | `vdbSparseSource` with `lod: none \| coarse-pyramid \| node-average`; HDDA in the inject march | the same `.wvdb` LOD **converted to NanoVDB with identical voxels** (sampler-only diff), then `smoke.vdb` full-res and the WDAS quarter cloud | dense vs sparse on identical voxels: PSNR ≥ 38 dB; flicker metric: mean frame-to-frame diff on a slow orbit (aliasing) | per-pass ms vs dense; GPU bytes (source + pyramid) |
| **Step 3 `'frame'` lighting** | shadow density, grid cascades, transmittance, density, lighting, temporal, env `'directional'` | demo-1 `default` (MS → `true` first), `shadow-cascades-4`, `shadow-grid-high`, `two-bunnies`, `environment-sky-midday` | dense: strict as Step 1; sparse: ≥ 35 dB vs dense | redraw-skip rate of the grid signature; ms per pass |
| **Step 4 Phase + MS modes** | dual-lobe, Mie, Mie RGB, `phaseGExponent`; two-sided, octaves, neural | demo-1 `frame-phase-*`, `multiple-scattering-*`, `octave*` | strict on dense; plus the white-furnace check (`src/docs/examples/src/utils/furnace.js`) | neural pass ms; WGSL compile ms (FXC/DXC on Windows) |
| **Step 5 Spectral + presets** | spectral targets and groups, `getFrameMediumProps` | demo-1/2 with `spectral` on, each preset | strict on dense | pass count, ms |
| **Step 6 Mesh shadows + emission** | `FrameVolumeShadows`, `FrameEmissionLight` on r185 | demo-1 `cast-shadow*`, `explosion-emission*` (ground explosion sequence) | strict on dense; shadow-only diff on a `ShadowMaterial` ground | shadow-map update ms within budget |
| **Step 7 Level sets + points** | sparse level-set tracer (A/B against his dense SDF tracer) and `VDBPoints` | demo-5 `venus-statue`, `utah-teapot`; demo-7 points | silhouette IoU ≥ 0.99, normal-angle error | ms per frame |

Rules for every step:
- Reference images come from **his renderer at the pinned commit**, rendered locally to PNG. His committed JPEGs
  (q90, loose 48/255 threshold) are a smoke check only.
- Parity runs use `jitter` off, `temporal` off, `normalize: 'clamp01'`, and matched `resolution`, so the dense
  source equals his texture byte for byte.
- Each step lands both sources behind a setting (CLAUDE.md "build both").
- The K3 exit-gate perf number uses the WDAS quarter cloud on **sparse** at a named preset.
- Each departure gets a DIVERGES row: LOD strategy, clamp, Shadertoy blackbody, WGSL-vs-TSL numerics.

Core-vs-adapter timing: Steps 1–5 first port the passes as three fragment passes, the fastest route to parity. As
each step goes green, its kernels move into `core` as WGSL, with CPU twins in `tools` for the maths: phase,
diffusion closed forms, octaves, MLP, froxel slice maths. The three adapter keeps the render targets. A
**core compute version of the pass graph** (3D storage textures instead of slice atlases) is a later item,
gated on the cloud system needing it, and A/B-tested against the ported fragment graph.

## 7. Licensing and attribution

**File header** for every ported or translated file (WGSL, TS or TSL):

```
// Portions ported from mjurczyk/openvdb (three-openvdb) v0.26.0, commit 75e7345,
// <upstream path(s)>. Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License
// (see THIRD_PARTY.md). Changes: <one line>. Divergences: DIVERGES.md <row ids>.
```

**THIRD_PARTY.md entry:** "mjurczyk/openvdb (`openvdb`, `three-openvdb`, `webvdb`) — MIT — https://github.com/mjurczyk/openvdb at
`75e73454f2d6f4384c9ffd57b704319767499360`. The upstream repository has no LICENSE file; MIT is declared in each
package's `package.json` and confirmed by the owner (D18)." Below it goes the full MIT text, with the copyright
line "Copyright (c) mjurczyk/openvdb contributors". Don't invent a year; ask him for one in R10. List the ported
file map (upstream path → our path).

Generated and data files need provenance lines:

| File | Provenance | Action |
|---|---|---|
| `frameMieTable.js`, `frameMieRgbTable.js` | his `scripts/mie-phase-table.mjs` (his BHMIE implementation after Bohren & Huffman; Deirmendjian C1 droplet distribution) | port the **generator** into `tools` and regenerate, so the table is a build output with the script as source; header cites the script and commit |
| `frameBlackbody.js` | CIE 1931 2° CMF from `files.cie.co.at` (public dataset) plus his Planck/soot code | cite CIE in the header |
| `frameNeuralIndirect.js` | his `scripts/neural-indirect/train.mjs`. Training data: procedural media **plus `bunny_cloud.wvdb` and `smoke.wvdb`**, which derive from openvdb.org samples (CC BY-SA 4.0 per our CLAUDE.md). Trained at HG g = 0.8. No separate licence on the weights | treat as MIT with the repo; **flag for the owner and for R10**: ask him to confirm the weights are MIT. Fallback: retrain with his script on procedural-only media on our machine (§9 Q2) |
| `frameMedium.js` | published measurements (Dubovik 2002, Hess 1998, Bond 2006, Pope & Fry 1997, Baran 2012) | keep the citations |
| **`blackBody()` in `frameShaderChunks.js` L973** | "Source – shadertoy.com/view/4tdGWM". Shadertoy's default licence is **CC BY-NC-SA 3.0** unless the shader states otherwise | **do not port verbatim.** Use the CIE `'planck'` path for `'approximate'`, or re-derive; DIVERGES row |
| IGN noise (L179) | Jimenez 2014 formula | cite; fine |
| `openvdb/math/noise.js`, `levelset/marchingCubesTables.js` | noisejs (Gustavson), Paul Bourke tables | not ported in K3 |

## 8. Risks

**Couplings to dense textures**
- *LOD everywhere* (§4.3). The shadow-grid injection samples at lod ≈ log2(grid voxel / texel). Without
  prefiltering, a sparse hero asset aliases and the cascades shimmer as they snap. Mitigation: the Step 2 A/B; the
  coarse pyramid is the likely default.
- *The lighting resolution is the shadow grid's, not the asset's.* At 64³ (1 cascade), full-resolution sparse data
  only improves primary visibility, and that is capped by the froxel grid (≤ 512² × 24 slices × `subSteps`). The
  "sparse advantage" in K3 needs `'volume'` lighting (marching the source with HDDA), finer cascades, or both.
  Otherwise his 100-texel resample and our full-resolution grid will look alike. Measure it in Steps 2–3 and say
  so in the K3 handoff.
- *Hot-loop cost:* software trilinear in the density pass. Mitigation: shared-leaf stencil, `getBoxSteps` on the
  sampled level, HDDA skip, and `pixelsPerFroxel`.

**three.js version** (his pin is 0.187, ours r185; his peer range claims `>=0.162`)
- Internals he uses: `renderer._nodes.nodeFrame.renderId` (`FrameRenderer.collectVolumes`),
  `renderer._precompilationState` (`FrameEmissionLight` L434), `new LightsNode().setupLightsNode(...)` to reset
  light shadow nodes (`FrameVolumeShadows` L175–212, which already notes "other three.js versions can have other
  internals"), `AnalyticLightNode` subclassing, `ShadowNode` `filterNode` and `NodeUpdateType` timing.
- Check on r185: `camera.reversedDepth` (`frameView.collect`), MRT `RenderTarget({count})` +
  `outputStruct`, `viewportDepthTexture`, `PointsNodeMaterial` sprites, and `texture3D().sample().level()` in
  loops.
- The v0.26.0 diff itself was a three-version fix: the CUBE_UV env path in the WebGL GLSL.
- Mitigation: port B-bucket files last (Step 6), behind a three-version probe that turns the feature off with a
  warning, as he does. Add an r185/r187 CI matrix for `./three`.
- *Compile time:* the neural weights are inlined as shader constants (≈ 1.8k floats). On Windows FXC/DXC that is
  a known slow-compile pattern (see the `webgpu-tsl-compile` notes). Put the weights in a uniform or storage
  buffer in core, and measure compile ms in Step 4.

**Release velocity**
- 17 releases in 19 days (v0.10.0 on 2026-10-31 → v0.26.0 on 2026-11-18).
- `frame/core` + `frame/webgpu` churn: +7,226/−668 lines since v0.12.0 (Nov 2), +420/−50 since v0.21.0
  (Nov 10), and 0 lines from v0.25 to v0.26. The renderer is stabilizing.
- His `CHANGELOG.md` has no headings after v0.5.1; everything sits under "Unreleased". **Track by tag diff, not
  by changelog.**
- Plan: the pinned commit goes in every ported header and in a `docs/dev/research/upstream.md` table (upstream path →
  our path → pinned commit). A small read-only script lists upstream changes to ported paths between the pin
  and the latest tag. Review at each K3 step and before each of our releases; bump the pin deliberately, with a
  DIVERGES review.

**Quality concerns** (he has said agents wrote much of it; the code is consistent and heavily commented)
1. **Density clamp:** `FrameVolume.js` `sampleDensity` stores `clamp(value / scale, 0, 1)` in 8 bits, with
   `scale` 1 for density. Grids with densities above 1 saturate silently, and `meanDensity` is computed from the
   clamped data. This is not a crash bug but a fidelity limit, and it bounds our parity tests.
2. **Frame identity heuristic:** `FrameRenderer.render()` dedupes per frame with `renderer.info.render.calls`
   plus the camera. Any draw between two `FrameVolume` proxies changes the count, which would re-render the
   whole frame. `renderOrder 1e6` makes this unlikely, not impossible. Use an explicit frame id.
3. **Stale comment:** `getFrameMultipleScattering` says "Only mode true" but returns 1−g for `'two-sided'` and
   `'neural'` too, via `isFrameDiffusion`. The code looks right.
4. **Mask sampling:** the mask is sampled at `local + 0.5` without `offset3D` or wrap, while density, albedo and
   emission use `gridUv`. This is probably intentional (the mask is fixed in the box) but undocumented. Confirm
   with him before porting.
5. **Two hand-synced backends** (GLSL 8.1k, TSL 8.3k) with real differences: WebGL uses hardware mips with
   floor sizes; WebGPU uses a CPU mip atlas with ceil sizes (`frameMipFilter.getLevelSize`). Diff both when
   porting a function, and treat WebGPU as the reference.
6. **CPU conversion cost:** about 1M `grid.getValue()` calls in JS per 100³ volume (12 ms slices). This is the
   main thing our GPU-resident sparse source removes. It is worth stating in the joint-project pitch.
7. **Neural generalization:** trained at HG g = 0.8 only and grey only. Other phases or albedos are
   out-of-distribution. His own numbers on held-out scenes: neural 37% mean error (+2% bias), `true` 65%
   (−51%), two-sided 59% (+10%), octaves 127% (+48%). Keep `true` as our documented fallback, and A/B neural vs
   diffusion in Step 4.

## 9. Questions for the owner

1. **Port order change** (§6: phase and MS before mesh shadows and emission). *Suggest yes.* Pro: the
   source-free A-bucket reaches the cloud system early, and it matches his default output (neural). Con:
   PHASES text changes. *Settles it:* none needed; it is a sequencing choice. Update PHASES K3 when K3 starts.
2. **Neural weights provenance.** *Suggest:* ship his weights after R10 confirms MIT; meanwhile record the
   training-data note in THIRD_PARTY. Pro: identical output to his. Con: CC BY-SA training-asset ambiguity.
   *Settles it:* his confirmation, or a procedural-only retrain whose Step 4 A/B shows ≤ 2 pp change in his
   compare metric.
3. **Default LOD strategy for sparse** (coarse pyramid vs node averages). *Suggest* building both in Step 2 and
   deciding by the flicker plus per-pass-ms A/B on the WDAS cloud. The pyramid costs about 2 MB per grid; node
   averages need stats in every grid we write.
4. **K1 `majorant` profile.** K3 does not need it (no delta tracking). *Suggest* deferring it after the
   `march` profile, with the pathtracer as its only consumer (D19). *Settles it:* this note's §4.1 caller table.
