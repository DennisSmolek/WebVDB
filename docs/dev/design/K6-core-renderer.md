# K6 design note: the renderer in the core

Status: design, rewritten 2026-11-25 for the owner's answers (Q1 = B: compute on 3D textures; Q2–Q10, N1–N3). Ready
for K6.1 ([K6.1-brief.md](./K6.1-brief.md)). Plan context: [PHASES.md](../PHASES.md) K6/K7/P1,
[TASKS.md](../TASKS.md) K6.x, [DECISIONS.md](../DECISIONS.md) D8, D10, D11, D20, D21, D25, D32 and the D34 text in §10.
Inputs: [R1](../research/notes/R1-renderer-port.md), [K1-core](./K1-core.md), the K3 handoffs in
[docs/archive/handoffs/](../../archive/handoffs/), [DIVERGES.md](../DIVERGES.md) V23–V73, his post-0.26 fixes (K7.1–K7.3).

Project goal (owner, 2026-11-25): the best VDB library on the web, WebGPU-first. mjurczyk's renderer (our K3 port at
`75e7345`) is a benchmark and a collaborator, not the spec. Native OpenVDB/NanoVDB, papers (Hillaire 2015/2020,
Wronski 2014, Disney "Deep Scattering", Novák et al. 2018) and other engines are fair sources. Every departure from
his renderer gets a DIVERGES row (§11).

## 0. Decisions at a glance

| Topic | Design |
|---|---|
| Who runs the passes | The core, on the host's device, into the host's encoder (Q2). `./three` passes three's device, collects scene data, binds core textures into three by the backend-record swap, composites through a material. |
| Pass style | Compute kernels on 3D textures (Q1 = B). Fragment only at host seams: composite (pass or WGSL function), level-set `frag_depth`, points. |
| Froxels | `texture_3d` written through `texture_storage_3d`. One thread per froxel gathers the visible volumes; a per-column z-scan writes integrated scattering and optical depth; the composite reads that prefix texture at the scene depth (§3.4). |
| 'frame' lighting | Density and lighting fused into one kernel; no 4 × rgba16float intermediates. Shadow grid, transmittance and env grid are stacked 3D textures read with hardware trilinear. |
| Environment | Host equirect or cube `GPUTexture`. The core prefilters it, projects SH (irradiance) and extracts dominant lights on the GPU into a buffer later passes read. No CPU readback (Q3). |
| Shadows | Volume→mesh Beer maps read with the view they were drawn with; mesh→volume from the host's depth maps (default on when given); volume self-shadowing. All in K6 (Q10). |
| Multiple scattering | A pluggable term. Default chosen by quality against the reference renderer (K6.R), between diffusion and K7.3 multi-scale. The MLP is an experimental generic evaluator (Q5). |
| Emission colour | Data (named spectrum or 256-texel LUT) plus an optional user WGSL hook (Q9). |
| Device baseline | WebGPU core features and default limits. `texture-formats-tier1`, `shader-f16`, `subgroups` are optional fast paths with fallbacks (N2). |
| Gates | Correctness and quality only: kernel ↔ CPU twin, frames vs the TSL path and his K3 captures by tolerance, a brute-force reference for "better" (N1). No timing gates; timing A/Bs are deferred (§5.4). |
| TSL path | Stays as the reference behind an internal `backend: 'tsl' \| 'core'` during K6.1–K6.7; deleted in K6.8 (N3). |

## 1. Goals, non-goals, constraints

**Goals.** (1) A WebGPU host with no engine renders the froxel renderer ('volume' and 'frame' lighting, phase,
multiple scattering, spectral, temporal), volume shadows on its meshes, mesh shadows on volumes, emission light,
level sets and points (PHASES K6 exit gate). (2) `./three` draws through the same kernels. (3) Every kernel has a
CPU twin in `tools` (D11). (4) The three bug classes of K3–K7 can't happen in the core by construction (§4.3).
(5) Quality is measured against a reference, not only against his renderer.

**Non-goals for K6.** Delta tracking and majorants in the froxel path (P1's `majorant` profile, D22). Other adapters
(R6; K6 makes them possible). Removing the dense source. Shading level-set and point surfaces with a core BRDF
inside `./three` (three shades them; §3.9). Any speed claim (no quiet machine; §5.4).

**Constraints.**
- D8: the host owns the device and the encoder. The core records and never submits. It frees only what it made.
- D20: `core` never imports `three`; it may import engine-free maths and twins from `tools/frame` (precedent:
  `core/compute/pyramid.ts` → `tools/frame/pyramid-layout.ts`). `tools` never imports `core`.
- D10: grid data reaches a kernel through one storage buffer (the pool). A source-free kernel may use its one
  storage buffer for other data (MLP weights, the environment result). No kernel binds two.
- D21: WGSL bytes are budgeted and snapshotted (§5.3). One VDB module per shader.
- D32: new core objects take `Gpu` where `./three` has the plain name (Q4). K6.1–K6.7 export through
  `core/internal.ts` only; the public surface lands in K6.8 with one reviewed `api-surface` diff.

**Device limits and formats (N2).** Verified 2026-11-25 against the WebGPU editor's draft
(<https://gpuweb.github.io/gpuweb/>: §3.6.2 limits, §25 features, §26.1.1 plain color formats) and WGSL
(<https://gpuweb.github.io/gpuweb/wgsl/>: §6.6.5 storage textures, `readonly_and_readwrite_storage_textures`).

| Fact (core = default device, `core-features-and-limits`) | Value | What it means here |
|---|---|---|
| `rgba16float` storage | write-only and read-only in core; read-write needs `texture-formats-tier2` | Froxels, prefix, history, grids, maps: all rgba16float, written write-only. |
| `rg16float`, `r16float`, `r8unorm` storage | only with `texture-formats-tier1` (read-write: `r16float`/`r8unorm` with tier2; `rg16float` never) | rg16float shadow grid is a fast path; fallback rgba16float (same `.xy`). |
| `r32float`, `r32uint`, `r32sint` | read-write in core | In-place accumulation of multi-group partials as packed `pack2x16float` pairs in `r32uint` (§3.3). |
| `rgba32float` storage | write-only/read-only in core; filterable only with `float32-filterable` | Environment map (HDR suns exceed f16's 65504); read with `textureLoad`, never filtered. |
| `rgba16float` sampling | `"float"` (filterable) in core | Hardware trilinear on every froxel and grid texture. |
| 3D storage textures | `texture_storage_3d` exists; layouts forbid only `cube`/`cube-array` storage views | Froxels and grids are 3D; cube shadow maps are 2D arrays of 6 layers. |
| `GPUTextureView` | exposes no dimension or format (`interface GPUTextureView {}`) | Environment input is a `GPUTexture` plus a projection (§2.3). |
| `maxTextureDimension3D` | 2048 | Froxel width/height/depth+1 ≤ 2048 (clamped, warned). Stacked grids tile in 3D (§3.2). |
| `maxStorageTexturesPerShaderStage` | 4 (fragment: `maxStorageTexturesInFragmentStage` 4) | Stacking keeps every kernel at ≤ 3 storage textures (spectral gather: scatter, scatterAlpha, partials); fragment seams use none. |
| `maxSampledTexturesPerShaderStage` / `maxSamplersPerShaderStage` | 16 / 16 | Stacking keeps sampled counts ≤ 12 incl. a dense source's 8 (4 maps + 4 mip atlases). |
| `maxStorageBuffersPerShaderStage` | 8 (fragment 8; compat 4) | We use ≤ 1 (D10). |
| `maxUniformBufferBindingSize` | 65,536 B (compat 16,384) | The visible-volume array: 512 B each → ≤ 128 volumes per frame (compat 32), warned beyond. |
| `maxUniformBuffersPerShaderStage` | 12 | ≤ 5 used (frame, lights, environment, volumes, group). |
| `maxComputeInvocationsPerWorkgroup` | 256 (compat 128) | All kernels use 64 (fits compat). |
| `maxComputeWorkgroupSizeX/Y/Z` | 256 / 256 / 64 (compat 128 / 128 / 64) | Shapes 8×8×1 and 4×4×4. |
| `maxComputeWorkgroupStorageSize` | 16,384 B | Reductions use ≤ 8 KiB (256 × vec4f × 2). His MLP (1,761 f32 = 6.9 KiB) fits a workgroup cache (deferred A/B). |
| `maxComputeWorkgroupsPerDimension` | 65,535 | Not binding (512 / 8 = 64). |
| `maxStorageBufferBindingSize` / `maxBufferSize` | 128 MiB / 256 MiB | Per pool (demo 13's one pool per large grid stays). |
| `maxBindGroups` | 4 | Groups 0–2 used, no gaps (Traps). |
| `shader-f16` / `subgroups` | WGSL `enable f16;` / `enable subgroups;` only with the feature | Optional fast paths, off until a deferred A/B (§5.4). |

Compatibility-mode devices are best effort: kernels fit compat's workgroup limits; the volume cap drops to 32; the
sparse source still needs a storage buffer in compute (always available).

## 2. Public core API

### 2.1 Objects and names (D32, Q4)

| Core export | Kind | Notes |
|---|---|---|
| `vdbCreateFrameRenderer(device, options)` → `VDBFrameRendererGpu` | function / type | The kernel graph and its resources. |
| `VDBFrameVolumeGpu` | type | Handle from `renderer.addVolume(slot, params)`. |
| `vdbSparseSourceGpu(pool, options)` → `VDBSparseSourceGpu` | function / type | K6.7: one source for every volume of one pool. `source.volume({ density, mask?, albedo?, emission? })` returns a `VDBVolumeSlot`. Pyramids share one packed r16float atlas (one binding), `followImage`, `skipEmpty`. |
| `vdbDenseSource` | function | Exists (K3.0). A dense source is its own `VDBVolumeSlot` (`v` = 0). |
| `VDBFrameSettings`, `VDBFrameVolumeParams`, `VDBFrameInput`, `VDBFrameCamera`, `VDBFrameLight`, `VDBFrameEnvironment`, `VDBFrameOccluder`, `VDBFrameTargets`, `VDBFrameOutputs`, `VDBVolumeShadowRequest`, `VDBVolumeShadowMap`, `VDBEmissionColor`, `VDBFrameMlp` | types | Plain data. One declaration each; `./three` re-exports. |
| `vdbFrameIntegrateWgsl(options)`, `vdbVolumeShadowWgsl(options)`, `vdbEmissionProbeWgsl(options)`, `vdbLevelSetSurfaceWgsl(options)` | functions | WGSL for hosts that composite or receive in their own shaders (§2.5). |
| `vdbFrameLights`, phase packing, `VDBSequence` | values / class | Move from `./three` (Q7: `VDBSequence` is engine-free already); one declaration, re-exported. |

Group-2 source contract (frozen in K6.1, used by K6.7): `wgsl`, `layoutEntries`, `bindEntries` as today
(`core/source/types.ts`), and every `vdbSrc*` function takes `v: u32`, the source-local slot of the volume, which the
renderer copies from `VDBVolumeSlot.v` into the volume's uniform record. Volumes that share a source object form one
**source group**: one bind group, one dispatch.

### 2.2 Creation, settings, volumes

```ts
interface VDBFrameRendererGpuOptions {
  settings?: Partial<VDBFrameSettings>;      // his names and v0.26 defaults (V67), plus the K6 fields below
  types?: readonly VDBGridTypeName[];         // grid types sparse sources may hold (default VDB_ALL_SCALAR_TYPES); compiled once (G16a)
  composite?: { colorFormat: GPUTextureFormat; depthFormat: GPUTextureFormat } | null; // the core composite pass (§2.5)
  features?: { tier1?: boolean; f16?: boolean; subgroups?: boolean }; // default: what the device has (f16/subgroups off until §5.4)
  label?: string;
}
interface VDBFrameRendererGpu {
  readonly device: GPUDevice;
  readonly settings: VDBFrameSettings;        // mutable; read at the next update()
  readonly outputs: VDBFrameOutputs;          // §2.5
  readonly redrawStats: VDBFrameRedrawStats;  // V46
  addVolume(slot: VDBVolumeSlot, params?: Partial<VDBFrameVolumeParams>): VDBFrameVolumeGpu;
  prepare(options?: { spectral?: boolean[]; lighting?: ("volume" | "frame")[] }): Promise<void>; // createComputePipelineAsync for every reachable variant
  update(encoder: GPUCommandEncoder, frame: VDBFrameInput, options?: { timestampWrites?: VDBFrameTimestamps; froxels?: boolean }): VDBFrameUpdate;
  render(encoder: GPUCommandEncoder, targets: VDBFrameTargets): void; // the core composite (needs options.composite)
  dispose(): void;
}
interface VDBFrameVolumeGpu {
  params: VDBFrameVolumeParams;   // his FrameVolume props: matrixWorld, bbox, extinction, extinctionScale, angstromExponent,
                                  // baseColor, scatterColor, densityScale, normalizeDensity, densityMean, densityCutoff,
                                  // absorbance, opacity, emissiveIntensity, temperatureRange, lights (bitmask),
                                  // shadowSteps, mipmaps, hdda, phase overrides (V65), wrap/offset, emissionColor (§3.8)
  visible: boolean;               // drawn this frame (host culling, layers)
  castsShadow: boolean;           // in the shadow grid ('frame' lighting), also when not visible
  remove(): void;
}
```

New settings (K6 fields, beside his): `froxelMix: 'mixture' | 'over'` (§3.3; `'over'` is internal, for the TSL
A/B, removed in K6.8), `occlusion: boolean` (default true: host occluder maps shadow volumes, §3.7),
`multipleScattering` adds `'multi-scale'` (K7.3) and `'mlp'` (experimental, §3.6). `params` has one writer (the
host); `update()` packs it into the volume's uniform record (the K5.9a class, §4.3).

### 2.3 Frame input

```ts
interface VDBFrameInput {
  frameId: number;                     // monotonic; jitter index, temporal validity (replaces his render-call heuristic)
  camera: VDBFrameCamera;              // view, projection (column-major), near, far, orthographic, reversedDepth, id?
  viewport: { drawingBuffer: [number, number]; canvas: [number, number] }; // froxel size, pixelsPerFroxel
  lights: readonly VDBFrameLight[];    // only lights in the rendered scene (the shadow budget counts these, K7.1 c13f2d5)
  environment?: VDBFrameEnvironment | null;
  occluders?: readonly VDBFrameOccluder[]; // host depth shadow maps (§3.7)
}
type VDBFrameLight = { id: number | string; mask?: number } & (
  | { kind: "ambient"; color: VDBVec3 }
  | { kind: "hemisphere"; sky: VDBVec3; ground: VDBVec3; up: VDBVec3 }
  | { kind: "probe"; sh: Float32Array }                            // 9 × rgb, three's SH9 convention
  | { kind: "directional"; direction: VDBVec3; color: VDBVec3; shadow?: VDBVolumeShadowRequest }
  | { kind: "point"; position: VDBVec3; color: VDBVec3; distance: number; decay: number; shadow?: VDBVolumeShadowRequest }
  | { kind: "spot"; position: VDBVec3; direction: VDBVec3; color: VDBVec3; distance: number; decay: number;
      cosAngle: number; cosPenumbra: number; shadow?: VDBVolumeShadowRequest });
interface VDBFrameEnvironment {
  id: unknown; revision: number;       // a change re-runs the environment kernels
  texture: GPUTexture;                 // linear radiance; any filterable or unfilterable float format (read by textureLoad)
  projection?: "equirect" | "cube";    // default: 6 square layers → cube, else equirect (a GPUTextureView has no readable dimension)
  rotation?: Float32Array;             // 3x3 world → map
  intensity: number;
}
interface VDBFrameOccluder {           // a host depth shadow map
  light: number | string;              // the VDBFrameLight id it belongs to
  depth: GPUTextureView;               // depth or r32float, 2d or 2d-array (cascades / cube faces as layers)
  viewProjection: Float32Array[];      // one per layer; the matrices the map was drawn with
  orthographic: boolean; bias: number; reversedDepth: boolean;
}
```

Light slots (`getSortedLight`, ≤ 6 shadow slots, the dominant environment light in the next free slot) are CPU work
in `update()`. The environment slot's direction and colour are not known on the CPU: the slot points at environment
light 0 in the GPU environment buffer (§3.5), so no readback is needed. The 16-light and 4-hemisphere caps stay.

The environment input fits a host's compute-PMREM output or a pmndrs/sky texture directly: any linear float
equirect or cube texture with `TEXTURE_BINDING`, mips optional (the prefilter reads level 0 by area). three's
CUBE_UV atlas is not a core format; the adapter passes `scene.environment`'s source texture, or converts a CUBE_UV-only
input with its existing env-map pass.

### 2.4 Lifecycle

```ts
const frame = vdbCreateFrameRenderer(device, { composite: { colorFormat, depthFormat } });
const sparse = vdbSparseSourceGpu(pool, { followImage: true });
const volume = frame.addVolume(sparse.volume({ density: id }), { extinction: 100 });
await frame.prepare();                            // optional warm-up

const enc = device.createCommandEncoder();        // per frame
pool.update(enc);                                 // host pool first (Traps: bind groups after the pool's update)
const u = frame.update(enc, { frameId, camera, viewport, lights, environment, occluders });
drawOpaque(enc, u.outputs);                        // receivers sample outputs.volumeShadows / emissionProbes
frame.render(enc, { color, depth, sceneColor });  // or vdbFrameIntegrate in the host's own pass
device.queue.submit([enc.finish()]);
```

- **update** does the CPU work (visible volumes and casters, frustum clip, slice range, froxel area, cascades and
  signatures, light slots, uniform packing), rebuilds bind groups whose inputs changed (`pool.version`,
  `source.version`, reallocated textures), and records the kernels of §3.1. Uniforms go through
  `queue.writeBuffer`; the contract is one `update()` per submitted encoder (the pool's contract, K1-core §1.4).
- **render** draws one triangle per pixel area into `targets.color`, scissored to the froxel screen rectangle,
  reading `targets.depth` (`texture_depth_2d`, or `r32float`) and, for `srgbBlending` or spectral, `targets.sceneColor`
  (else the core copies `color` into its scratch; needs `COPY_SRC`). Single-sample; MSAA hosts composite after
  their resolve or use the function.
- **Resize** is implicit: froxel size from settings and viewport (`getFroxelSize`, 15 % hysteresis), grid sizes from
  settings. A change reallocates, retires the old textures (§4.2), bumps `outputs.version`, drops history.
- **dispose** destroys every texture, buffer and query set the renderer made, current and retired. It never touches
  the device, pools or sources.

### 2.5 Outputs and the composite / MRT seam

```ts
interface VDBFrameOutputs {
  readonly version: number;                 // +1 when a texture below is a new object: rebuild host bind groups
  readonly froxels: { integrated: GPUTexture; integratedAlpha: GPUTexture | null; layout: VDBFroxelLayout } | null;
  readonly volumeShadows: ReadonlyMap<number | string, VDBVolumeShadowMap>;  // by light id
  readonly emissionProbes: { texture: GPUTexture; grid: VDBFrameGridLayout } | null; // stacked 3D, 3 sub-volumes
  readonly uniforms: GPUBuffer;             // integrate / receiver uniforms, for host shaders
}
```

The core allocates every output (storage textures need `STORAGE_BINDING`, which three render targets lack; Q2).
Hosts that composite themselves bind `outputs.froxels.integrated` (`texture_3d<f32>`, linear sampler) and call
`vdbFrameIntegrateWgsl({ spectral })`:

```wgsl
struct VdbFrameIntegrated { radiance: vec3f, transmittance: vec3f }   // grey: transmittance.x == .y == .z
fn vdbFrameIntegrate(screenUv: vec2f, sceneViewDepth: f32) -> VdbFrameIntegrated;
fn vdbFrameComposite(i: VdbFrameIntegrated, sceneColor: vec4f, srgbBlending: bool) -> vec4f;
```

This is the seam that removes the K7.2 class (his 720c38f): kernels run in compute passes no host MRT or `pass()`
setting reaches; the host places only a function. Receivers get `fn vdbVolumeShadow(light: u32, world: vec3f,
viewDepth: f32) -> vec3f` and `fn vdbEmissionIrradiance(world: vec3f, normal: vec3f) -> vec3f`.

### 2.6 Volume shadows on meshes

```ts
interface VDBVolumeShadowRequest {
  size: number;                                  // castShadowMapSize
  view?: { viewProjection: Float32Array; near: number; far: number; orthographic: boolean }   // spot, point face
       | { cascades: { viewProjection: Float32Array; near: number; far: number }[]; breaks: number[] } // directional CSM
       | "fit";                                  // fitted around the casters (tools/frame/shadows.ts)
}
interface VDBVolumeShadowMap {
  texture: GPUTexture;                           // rgba16float 2d-array: cascades or 6 cube faces as layers (+ spectral layers)
  drawn: { viewProjection: Float32Array[]; origin: VDBVec3; forward: VDBVec3; far: number; perspective: boolean; breaks: number[] };
  revision: number;
}
```

The host passes the camera its own depth map uses, so both maps share one texel grid. Receivers project with
`drawn`, the state the map was drawn with: his 62a6e3e (lookups with the current `shadowCoord`) is impossible. The
per-update budget (`castShadowUpdatesPerFrame`, oldest piece first) counts only lights in this frame's `lights`
(his c13f2d5).

### 2.7 Sequences and coarse-first

`VDBSequence` moves to core (Q7) and calls `pool.replace` (sparse) or `source.replace` (dense). `update()` polls: a
grid whose `imageRevision` moved marks its pyramid stale (rebuilt in the same encoder after `pool.update`) and
moves the caster signature. `densityMean` stays a param (V66). Coarse-first: `followImage` re-reads boxes and the
pyramid size once the new image is resident; boxes, sizes and LOD ranges are uniforms and `types` is fixed at
creation, so a swap creates zero pipelines and zero shader modules (G16a's test moves to the core harness).

## 3. Kernel graph

### 3.1 Bindings and order

| Group | Contents |
|---|---|
| 0 frame | `uniform VdbFrameUniforms` (camera, froxel layout, area, previous area, jitter, fade, settings, phase, octaves, MS), `uniform VdbFrameLights`, `uniform VdbFrameEnvironment` (the GPU environment buffer bound as uniform), samplers (linear clamp, nearest clamp, comparison), LUTs (Mie / Mie RGB, emission colour) |
| 1 pass | `uniform VdbFrameVolumes` (array, 512 B per visible volume), `uniform VdbFrameGroup` (dynamic offset: first, count, isFirst, isLast), the pass's sampled and storage textures, occluder maps |
| 2 source | `source.layoutEntries({ group: 2, firstBinding: 0 })`: the pool (one storage buffer) + packed pyramid atlas, or dense textures. Source-free kernels: their one storage buffer here (MLP weights, environment result) or nothing |

Per update, 'volume' lighting: pyramids → [environment] → `froxelGather` per source group → `froxelResolve`.
'frame' lighting: pyramids → [environment] → `gridDensity` per caster group (stale cascades) → [grid mips] →
`gridTransmittance` → `environmentGrid` → `volumeShadow` (budgeted) → emission → `froxelGather` per source group
(the last fuses lighting) → `froxelResolve`. Brackets: only on a signature or revision change. Stale signature
caches skip kernels exactly as today (V44 fields + light ids + environment revision + occluder revisions).

### 3.2 Texture conventions

- **Froxel textures** are `W × H × D` 3D (`D` = slices; the prefix has `D + 1` boundary layers). Texel (x, y, z) is
  his froxel (x, y) of slice z. The jitter noise keeps his formula on the virtual atlas coordinate during the A/B
  (bit-equal noise), so parity is not lost to noise.
- **Stacked textures**: several logical 3D grids of one size (cascades × targets) are sub-volumes of one 3D texture,
  tiled along x, then y, then z within 2048 per axis (`vdbFrameStackLayout`, twin in `tools/frame`). Reads clamp the
  coordinate to the sub-volume's half-texel box, which equals clamp-to-edge per sub-volume. One binding per stack
  keeps storage writes ≤ 2 and sampled reads ≤ 12 per kernel.
- **Zero outside the area**: each froxel kernel dispatches over the bounding box of this frame's and last frame's
  areas and writes 0 outside this frame's area. Texels outside both were zeroed earlier, so "zero outside the area"
  holds every frame without clears (his targets were cleared; same values).
- **Multi-group partials**: when a frame has more than one source group, groups before the last accumulate additive
  sums in place in a stacked `r32uint` texture of `pack2x16float` pairs (read-write is core for `r32uint`). The last
  group reads them, adds its own, and writes the result. One group allocates nothing.

### 3.3 Kernel table

WG = workgroup size. All formats are core unless marked; "→ fallback" is used without the feature.

| Kernel | Dispatch, WG | Reads | Writes (format) | When | CPU twin (`tools/frame`) | Replaces |
|---|---|---|---|---|---|---|
| `pyramid` (exists) | per grid, 4×4×4 | pool | packed r16float atlas (copy) | `imageRevision` moved | `vdbBoxPyramid` | — |
| `froxelGather` 'volume' | area bbox × D, 8×8×1 (a screen tile of one slice: coherent rays) | source, lights, env buffer, occluders | grey: scatter rgba16float (premultiplied rgb, α); spectral: + scatterAlpha rgba16float (α r, b); multi-group: partials r32uint | per source group | `vdbFrameInjectSlice`, `vdbFrameShadowMarch`, **new** `vdbFrameGatherFroxel` (mixture) | `inject-material.ts` |
| `froxelGather` 'frame' (fused) | same | source, transmittance stack, env grid, LUTs, occluders, partials | scatter (+ scatterAlpha); non-last groups: partials | per source group | `vdbFrameDensityFroxel` + `vdbFrameLightingFroxel` (+ spectral), **new** `vdbFrameFusedFroxel` | `density-material.ts`, `lighting-material.ts` |
| `froxelResolve` | area bbox columns, 8×8×1 | scatter (load, 6 neighbours), history (trilinear) | history' rgba16float (if `temporal`), integrated rgba16float (L rgb, opacity 1 − T; V80) (+ integratedAlpha: opacity r, b → rg16float tier1, fallback rgba16float) | every update | `vdbFrameReproject`, **new** `vdbFrameTemporalFroxel`, `vdbFrameScanColumn` | `temporal-material.ts`, the slice loop of `integrate-material.ts` |
| `gridDensity` | cascade R³, 4×4×4 | source at grid lod | shadow grid stack rg16float tier1 → rgba16float (spectral: 2 stacks rgba16float); multi-group partials | stale cascade, per caster group | `vdbFrameShadowAbsorption`, **new** `vdbFrameGridDensityVoxel` | `shadow-density-material.ts` |
| `gridMips` | per level, 4×4×4 | last cascade | mip stack (rg16float tier1 → rgba16float) | MS needs mips, on grid redraw | `vdbFrameGridMipLevel`; K7.3's twin for `'multi-scale'` | `grid-mip-material.ts` |
| `gridTransmittance` | cascades × R³, 4×4×4 | grid stack, mips, occluders, MLP weights (its one storage buffer) | transmittance stack rgba16float (4 targets × cascades; spectral 3 per slot group) | stale | `vdbFrameTransmittanceVoxel`, spectral twin, `vdbFrameNeuralLight` → `vdbFrameMlpForward` | `transmittance-material.ts`, `neural.ts` |
| `envPrefilter` | 128 × 64, 8×8×1 | host env texture (`textureLoad`, exact-area box) | env work map rgba32float | env revision | **new** `vdbFrameEnvironmentPrefilter` | `environment-material.ts` (PMREM sampling) |
| `envReduce` | 1 workgroup of 256 | work map | env buffer (storage; later bound as uniform): SH9 irradiance, ≤ 12 lights | env revision or env settings | `VDBFrameEnvironmentLights` (directional, SH lights), **new** `vdbFrameEnvironmentSh` | the CPU readback path |
| `environmentGrid` | last cascade R³, 4×4×4 | grid stack, env buffer | env grid rgba16float (spectral stack) | stale | `vdbFrameEnvironmentGridVoxel` (+ spectral) | `environment-grid-material.ts` |
| `volumeShadow` | size² × layers, 8×8×1 | last cascade | map rgba16float 2d-array (τ·first, τ·last, 0, τ; + spectral layers) | budgeted per piece | `vdbFrameShadowMapTexel`, `vdbFrameShadowReceiver` | `volume-shadow-material.ts`, half of `volume-shadows.ts` |
| `emissionDirection`, `emissionReduce`, `emissionProbe` | rays 8×8×1; 1 WG of 256 per slot; probes 4×4×4 | source emission + extinction; reduce; emitters | rgba32float; probe stack rgba16float (3 sub-volumes) | volume or cascade change | `vdbFrameEmissionDirectionTexel`, `vdbFrameEmissionReduce`, `vdbFrameEmissionProbe` | the pass half of `emission-light.ts` |
| composite (fragment) / `vdbFrameIntegrate` | pixel rect | integrated (2 taps), depth, scene colour | host colour | `render()` or host | **new** `vdbFrameIntegratePrefix`, `vdbFrameSrgbBlend` | `integrate-material.ts` |
| `levelSetSurface` (fragment), points (vertex/fragment) | per draw | pool / point buffer | host colour + `frag_depth` | per draw | `levelSetHit`, `levelSetNormal`, **new** sprite-size twin | §3.9 |

Workgroup shapes are chosen for locality (8×8 screen tiles share view rays and source leaves; 4³ grid blocks share
cells) and 64 lanes fit every vendor's SIMD width and compat. Tuning them is deferred (§5.4).

### 3.4 Froxels: gather, mixture, scan, composite

**Gather (K6.1).** One thread per froxel loops over its source group's volumes (`VdbFrameGroup.first..count`): a
rect and slice-range early-out from the volume record, then his pre-integration across the slice (sub-steps at the
froxel lod), the visible position, and the light loop. The shadow march is one WGSL function,
`vdbFrameShadowMarch(v, o, d, steps, lod) -> vec4f`, called from one loop over the light list and the isotropic
directions, so `vdbSrcDensity` appears at two call sites per entry (sub-steps and march). The K3.7 level-set trace
showed why: six inlined sample sites took 98–168 s to compile on SwiftShader, one site 10–19 s. Light terms are
loops over uniform counts (Q8); V55's compiled-in sets go away.

**Mixture.** His inject blends each volume "over" in draw order, which is scene-traversal order: two volumes in one
froxel give different colours if the scene lists them the other way round. The core default (`froxelMix:
'mixture'`) treats volumes sharing a froxel as co-located media: it sums `τᵢ` and `τᵢ·Lᵢ` (Lᵢ the volume's
unpremultiplied slice radiance) and writes α = 1 − exp(−Στᵢ), premultiplied L = (ΣτᵢLᵢ / Στᵢ)·α. It is
order-independent and makes multi-group accumulation additive. One volume per froxel gives identical values. `'over'`
stays internal for the TSL A/B; K6.R decides with the reference, and K6.8 deletes the loser.

**Fusion ('frame' lighting, K6.2).** The density sums (his 4 targets: extinction-weighted albedo, emission, visible t,
shadow slots, environment) live in registers while the thread loops its group; the last group then lights the
froxel (his lighting pass) and writes scatter. No intermediate targets for one group; multi-group frames use the
packed partials (§3.2).

**Resolve: temporal + z-scan.** One thread per (x, y) column walks z = 0..D−1. With `temporal`, each froxel is his
reprojection (`vdbFrameReproject`: the previous froxel grid sampled with hardware trilinear, which equals his
bilinear-in-tile + linear-between-slices with his clamps), clamped to the 6 current neighbours (x±1, y±1 by
`textureLoad`; z±1 from the column), blended, and written to history'. Then the column accumulates his integrate
loop: stop adding once T < 0.001 (his break), and write boundary k+1 = (Σ radiance, opacity 1 − T). Boundary 0
is (0, 0). As built (K6.1, V80) the prefix stores the opacity 1 − T, not the optical depth τ: bilinear filtering of
the prefix then averages T (the coverage-correct box filter at silhouettes); f16 keeps T to 2.4e-4 near T = 1.
History' is the next frame's history; the scatter texture is only read here.

**Composite.** Scene depth → fractional slice s (his log mapping), k = ⌊s⌋, c = s − k. Two taps of the prefix at the
boundary layers k and k + 1 (bilinear in x, y; exactly on z texel centres): A, B (A.T = 1 − A.opacity). With
α = 1 − B.T / A.T (the slice's own opacity):

    radiance = A.L + (B.L − A.L) · (1 − (1 − α)^c) / α        (→ c as α → 0)
    T        = A.T · (1 − α)^c

which is his `sliceAlpha` coverage formula. One difference remains: x, y filtering now acts on the composited prefix
instead of on per-slice froxels before compositing (they differ at volume silhouettes by sub-froxel amounts; DIVERGES
row). A one-tap linear lookup (Hillaire 2015) is a deferred timing option, not the default. Spectral reads the
second prefix for the opacity r, b and composites against the scene colour itself (one hardware alpha), as his.

**Memory** at his 512 × 512 × 24, grey: scatter 48 MiB + history 2 × 48 + prefix 50 = 194 MiB. His 'frame' lighting
used atlas 48 + history 96 + density targets 192 = 336 MiB. Multi-group frames add 192 MiB of partials ('frame') or
48 ('volume'). Spectral adds the alpha textures (≈ +50 %).

### 3.5 'frame' lighting grids and environment (K6.2)

- **Shadow grid.** `gridDensity` gathers casters per voxel per stale cascade into the stacked grid (extinction,
  absorption). Cascades, ranges and signatures stay CPU code (`tools/frame/grid.ts`).
- **Transmittance.** Source-free; marches the grid stack with hardware trilinear toward 6 slots and sky up/down; 16
  values per voxel → 4 sub-volumes per cascade in one stack (spectral: 3 per slot group, one dispatch per group).
- **Environment (Q3).** `envPrefilter` reduces the host texture to a 128 × 64 rgba32float work map by exact texel
  area (equirect) or direction-binned `textureLoad` (cube), then a separable cone blur standing in for his PMREM
  roughness 0.4. `envReduce` (one workgroup) projects SH L2 with solid-angle weights (irradiance by the Ramamoorthi–
  Hanrahan cosine convolution) and runs his 'directional' extraction (brightest texel, fixed sky seeds, nearest-seed
  binning, energy centroids) and 'spherical-harmonics' lights on the 64 × 32 downsample, then writes one
  `VdbFrameEnvironment` struct. Every later pass binds that buffer as a uniform. No readback: the first frame already
  has its environment lights (the 'outward' first-frame fallback trap goes away). 'outward' irradiance comes from the
  SH, not from PMREM roughness 1 (DIVERGES row).
- **Environment grid.** As his, from the env buffer.

*K6.2 as built (handoff K6-2.md):* shadow grid and transmittance are rgba16float stacks (`vdbFrameStackLayout`: R³
sub-volumes along z, then x, then y; the tier1 rg16float grid is not used yet); the grid layout, stack origins, the
'frame' phase and the multiple-scattering value are fields of `VdbFrameUniforms` (no new binding); the group record
gains `casters` (first, count, role, index): in a source group the records are visible non-casters, visible casters,
then casters that are not drawn. Multi-group 'frame' partials hold the 16 density sums as 8 `pack2x16float` words per
froxel (r32uint, 8·D layers); the grid's partials are one word per voxel. The environment (`envPrefilter`,
`envReduce`, `environmentGrid`, the environment light slot and 'outward' SH) is K6.2b.

*K6.2b as built (handoff K6-2b.md):* `envPrefilter` is two kernels: the exact-area (equirect) or k×k supersampled
(cube) 128 × 64 rgba32float work map **in map space**, then a GGX roughness-0.4 convolution (D(h)·n·l, three's PMREM
filter, not a cone blur) straight into his 64 × 32 map in world space (the rotation applies there and in the SH).
`envReduce` is one workgroup of **64** (compat), writes `VdbFrameEnvironment` (count, mode, valid; SH9 × intensity; 12
directions and colours). The environment kernels have their own group 0 (`VdbFrameEnvParams`), because the buffer they
write is group 0 binding 3 (uniform) of every other kernel. Texel directions and solid angles come from f32 tables in
the WGSL (WGSL sin / cos are 2^-11 accurate); near-ties of the extraction use a 1e-5 tolerance. `VdbFrameUniforms`
gains `environment` (slot in use, first grid light, grid holds lights, present); slot, grid and 'outward' are decided
on the GPU from the buffer's count. 'volume' lighting marches every environment light (or 'outward' SH) in the
gather's one march loop. The `./three` adapter passes `scene.environment`'s texture with three's rotation (and the
flipY / cube reflection); a CUBE_UV atlas is drawn once into a 1024 × 512 lat-long.

### 3.6 Phase and multiple scattering (K6.3)

- **Phase**: HG, dual-lobe, Lorenz-Mie and Mie RGB from the committed tables as LUTs (V47); a uniform.
- **Multiple scattering is a pluggable term**: `false`, `true` (1D diffusion), `'two-sided'`, `'octaves'`,
  `'multi-scale'` (K7.3: binomial grid mips + 5 taps per level, written by K7.3 as core compute over a 3D texture,
  format-agnostic; K6.3 wires its kernel into `gridMips`/`gridTransmittance`), and `'mlp'` (experimental). The family
  is a pipeline key; its numbers are uniforms. The default is picked in K6.3 by quality: K6.R's reference decides
  between diffusion and multi-scale on the K3 scene set and the WDAS cloud; his 'neural' default is not matched.
  *K7.3 as built (V75):* `vdbCreateMultiScaleMips(device, { size, levels ≤ 8 })`; per shadow-grid redraw
  `mips.record(encoder, gridTexture, { origin: [0, 0, R·lastCascade] })` (only on signature change); `mips.resize`
  on resolution change, rebuild bind groups on `mips.version`. The kernel writes f16 through a storage buffer and
  `copyBufferToTexture`, so r16float/rg16float mips need no tier1. Shader: prepend `vdbFrameMultiScaleWgsl()`, bind
  `base: texture_3d<f32>` (filterable), `mips = mips.view`, a sampler from `VDB_FRAME_MULTI_SCALE_SAMPLER`, and a
  64 B `VdbFrameMultiScale` uniform from `vdbFrameMultiScaleParams(...)`; per voxel and light
  `vdbFrameMultiScaleLight(base, mips, smp, ms, world, dirToLight, maxDistance)`, then `min(direct + scattered, 1)`.
  Grey only (spectral and 'volume' fall back to diffusion). Builds from two origins need two submits (the origin
  is a `writeBuffer` uniform). Edge mode (zero vs his clamp) and σ = 2^k levels are K6.3 A/B candidates.
- **MLP (experimental, Q5).** A generic small-MLP evaluator: layer count, widths and activations are uniforms;
  weights and biases are a storage buffer (the transmittance kernel is source-free, so this is its one storage
  buffer, D10). Width ≤ 64 per layer (two `array<f32, 64>` ping-pong locals). A new network is new data: no rebuild,
  no code change. API: `settings.multipleScattering = 'mlp'` with `renderer.mlp: VDBFrameMlp | null` (default: his
  weights, V48, credit kept), documented as experimental. Twin: `vdbFrameMlpForward` (generalises
  `vdbFrameNeuralForward`).

### 3.7 Shadows (K6.5, K6.5b)

- **Volume → mesh (K6.5).** `volumeShadow` writes Beer maps from the last cascade per requested light piece,
  budgeted; receivers use `vdbVolumeShadow` with `drawn` (§2.6).
- **Mesh → volume (K6.5b, new).** `occluders` are the host's depth shadow maps. Separation of terms: the
  transmittance stack keeps the slot's *direct* transmittance unoccluded and multiplies the *scattered/total* term by
  the voxel's occluder visibility (so light diffusing into a mesh shadow is smooth at grid resolution); the fused
  gather multiplies the direct term by the occluder visibility at the froxel's visible position (sharp at froxel
  resolution); 'volume' lighting multiplies both terms of its march at the lit position. A map test at the start of
  a march toward the light is the march stopping at the occluder. Up to 4 maps bound per kernel (fixed slots,
  placeholder depth textures, count in a uniform: no pipeline key); more are warned and ignored. Default on when the
  host passes maps (`occlusion: true`). Twin: **new** `vdbFrameOccluderVisibility` (PCF as the host's filter: 1 tap
  or 2×2 comparison).
- **Self-shadowing**: the volume's own march ('volume') or the shadow grid ('frame'), as now.

### 3.8 Emission (K6.5)

Colour is data: `emissionColor: 'white' | { spectrum: 'planck' | 'soot'; exponent } | { lut: Float32Array /* 256 × rgb */ }`,
packed into one row of the emission LUT (D33's analytic CMF; harnesses inject the exact CIE table as a LUT). An optional
hook `{ wgsl: string }` defines `fn vdbUserEmissionColor(t: f32, emission: f32) -> vec3f` (t = normalised temperature);
every top-level name must start with `vdbUser`. The renderer validates it once (names, reserved words, back-ticks,
then `createShaderModule` + `getCompilationInfo` under `pushErrorScope('validation')`) and throws
`VDBError('VDB_BAD_WGSL')` naming the line. A hook is a pipeline key (hash of its text).

### 3.9 Level sets and points (K6.6)

The trace is core WGSL already (`vdbLevelSetHit`, `vdbLevelSetNormal`). K6.6 adds `vdbLevelSetSurfaceWgsl()`, a
fragment wrapper returning `{ hit, t, world, normal, ndcDepth }` with the pixel-centre ray (V57), and a raw reference
pass that writes `frag_depth` plus a Lambert + ambient shade or a G-buffer, for engine-free hosts and demo 17.
`VDBLevelSetMaterial` calls the same function through `wgslFn`. Points: the sprite size and attenuation become core
WGSL (`vdbPointSpriteVertex`) for a raw pass and `VDBPoints`' TSL vertex.

## 4. Resources, ownership, caching

### 4.1 Who owns what

| Resource | Owner | Lifetime |
|---|---|---|
| Device, encoder, submits | host | — |
| Pools, sources (incl. the packed pyramid atlas) | host creates; the source owns its textures | `source.dispose()` |
| Environment texture, occluder maps, composite targets | host | — |
| Froxel, history, prefix, partial, grid, transmittance, env, shadow-map, emission textures; uniform and env buffers; LUTs; query sets | renderer | until retired or `dispose()` |
| Pipelines, layouts, modules | per-device cache (WeakMap) | the device |

The Q2 allocator hook is gone: storage textures must be core-allocated. A host that wants its own memory wraps the
outputs (`./three` does, §6).

### 4.2 Retire queue

A replaced texture or buffer is kept in a list tagged with the update count and destroyed two updates later (the
pool's contract). three r187 destroys at once (K3.5/RV(5)); the core never does. `dispose()` drains the list.

### 4.3 How the K3–K7 lessons map

| Lesson | In the core |
|---|---|
| K5.9a: a uniform callback overwrote the per-draw density scale | No callbacks; `params` packed once per update, one writer per field. Test: `densityScale` moves the froxels on dense and sparse. |
| K5.9b: three caches a layout function per backend across builders | Plain WGSL text; pipelines keyed by the text hash + layout + formats. |
| RV(4): source listeners outlived volumes | Nothing registered; `update()` polls `version`/`revision`/`imageRevision`. |
| RV(5): `texture.dispose()` destroys immediately | Retire queue. |
| V53: TextureNode bindings shared by uuid | Explicit bind groups; placeholders are their own textures. |
| Traps: bind group after `pool.update` | `update()` builds groups after reading `pool.version`; documented host order. |
| One VDB module per shader | The renderer builds one module per pipeline from the union of helpers the sources ask for. |
| G16a: a grid swap rebuilt five pipelines | `types` at creation, boxes in uniforms; a test counts `create*Pipeline*` / `createShaderModule` across a swap (0). |
| K7.1 (62a6e3e, c13f2d5) | Drawn-view receivers; budget over this frame's lights. |
| K7.2 (720c38f) | Compute kernels + composite function (§2.5). |
| The 'outward' first-frame environment fallback | GPU environment reduction, no readback (§3.5). |

Temporal history is invalid after a layout change, a camera `id` change or a skipped `frameId`.

## 5. Shader composition and budgets

### 5.1 Text of one pipeline

```
vdbModuleWgsl({ slot: { group: 2, firstBinding: 0 }, types, profile: 'march', helpers })   // source kernels only
+ source.wgsl({ slot: { group: 2, firstBinding: 0 }, profile, channels })                   // vdbSrc*
+ frameUniformsWgsl()                                                                       // group 0, generated from the packer's field list
+ frameChunksWgsl(roots)                                                                    // closure of vdbFrame* functions, fixed order
+ kernelWgsl(kernel, features)                                                              // group 1 + entry
```

Chunks declare dependencies; the closure is emitted in a fixed order (same features, same bytes). Names: `vdbFrame*`
/ `VdbFrame*` and `vdbSrc*`; a test runs the reserved-word list (`active`, `set`, `filter`, `target`), a duplicate
symbol scan and the back-tick check over every composed pipeline. Uniform structs come from one field list that
generates both the WGSL struct and the TS packer; a unit test checks offsets.

### 5.2 Pipeline keys (compile time) vs uniforms

| Pipeline key | Uniform |
|---|---|
| kernel; lighting mode; spectral; MS family (incl. `'mlp'`); `froxelMix` (until K6.8); `temporal`; source layout (kind, channels, LOD mode, `hdda`, `types`); emission on/off + hook hash; storage formats (tier1 or fallback); multi-group (first/middle/last role); composite formats + `srgbBlending` | light counts and kinds, occluder count, phase and its parameters, MS numbers, MLP shape, octaves, jitter, cascade count, grid sizes, boxes, LOD ranges, all volume params, areas |

A first 'frame' frame with one sparse source compiles ≈ 9 compute pipelines (gridDensity, transmittance,
envPrefilter, envReduce, environmentGrid, fused gather, resolve, + volumeShadow / emission when used) and 1 render
pipeline (composite). `prepare()` creates them with `create*PipelineAsync`, every creation under
`pushErrorScope('validation')`.

### 5.3 Budgets (D21)

Sizes are measured at K6.1 and committed as `core/frame/size-snapshot.json` (CI fails on uncommitted growth), with
these caps in KiB of minified WGSL, module included: source-free kernels 24 (transmittance with MLP 28); source
kernels (gather, gridDensity, emissionDirection) with `march` + source helpers 56; `levelSet` surface: its budget + 4.
Sizes are bytes, not timings, so they gate.

### 5.4 Deferred: timing

No quiet machine is available (owner, 2026-11-25; D25). Every choice below is made on design or correctness grounds
now; the A/Bs are one deferred task (K6.T), run when a quiet machine exists, each as a paired run under a setting:

1. Workgroup shapes per kernel (8×8×1 vs 4×4×4 vs 16×16×1).
2. Composite: two-tap exact partial slice (default) vs one-tap linear.
3. Temporal fused into the scan (default) vs a separate kernel.
4. Light terms: uniform-count loops (default, Q8) vs compiled-in light-kind sets (V55).
5. Per-froxel volume loop (default) vs a tile/cluster volume list.
6. HDDA in the gather and grid kernels (D30/D31 re-measure); `vdbSampleFast` in compute.
7. MLP: weights from storage vs a workgroup cache; `shader-f16` arithmetic (precision gate first).
8. `subgroups` in `envReduce` and `emissionReduce`.
9. `texture-formats-tier1` rg16float grid vs the rgba16float fallback (bandwidth).
10. Multi-group partials: packed `r32uint` in place (default) vs rgba16float ping-pong or tier2 read-write.
11. Stacked textures vs separate bindings.
12. Compile time per pipeline (Metal; D3D12/DXC); `prepare()` cost.
13. Core vs TSL frame time on WDAS and demo 10 (the former Q1 second gate); his ≥ 60 fps WDAS target (K3.9).
14. Default froxel size / `pixelsPerFroxel`; `castShadowUpdatesPerFrame`.

### 5.5 Composing with P1

P1 adds the `majorant` profile and the lean reader. The renderer takes only opt-in helpers, so default module bytes
don't move under it. Shared file: `size-snapshot.json`, regenerated on conflict, never hand-merged.

## 6. The three adapter after K6 (K6.8)

**What stays in `./three`** (≈ 2k lines from ≈ 11.4k):
- `VDBFrameVolume` (Group + proxy box) holding a `VDBFrameVolumeGpu`; props map onto `params`.
- `VDBFrameRenderer`: one `VDBFrameRendererGpu` per three renderer on `getDevice(renderer)` (Q2). In the first proxy's
  `onBeforeRender` it collects volumes and casters, converts lights (and `shadow.camera` → requests, `light.shadow.map`
  → occluders), passes `scene.environment`'s texture, records `update()` into its own encoder and submits; three's
  frame submits after, so queue order is right.
- **Binding core textures into three** (Q2): a stand-in `Data3DTexture` / `DataArrayTexture` per output whose backend
  record's `texture` is set to the core `GPUTexture` before three first sees it, re-set when `outputs.version` moves;
  `backend.destroyTexture` is guarded so three never destroys a core texture. This is `VDBGridNode`'s pool-buffer
  pattern for textures, self-tested once per backend (`VDB_ENGINE_MISMATCH` on another three) and counted by tests
  that skip the self-test objects.
- The integrate proxy: a `NodeMaterial` whose fragment is `wgslFn(vdbFrameIntegrate)` over `viewportDepthTexture` /
  `viewportTexture`; three's MRT and `pass()` see one ordinary material (K7.2).
- Mesh shadows: the `filterNode` install (V60) sampling core maps through `vdbVolumeShadowWgsl`; emission: the
  `AnalyticLightNode` through `vdbEmissionProbeWgsl`.
- `vdbSparseSource` / `vdbDenseTextures` become thin wrappers; `vdbVolumeNodes` stays for `VDBVolumeMaterial`.
- Re-exports: `VDBSequence` (core), `VDBLevelSetMaterial`, `VDBPoints`, `VDBVolumeMaterial`, `VDBGridNode`.

**0.2.0 changes (Q6: break freely, no 0.1.x users, no aliases):** pass materials and their `any`-typed fields
removed; `VDBFrameVolumeShadows` / `VDBFrameVolumeShadowNode` removed; `emissionColor` takes `VDBEmissionColor`;
`neuralNetwork` becomes `mlp` (experimental); `passHook` brackets recorded kernels via timestamp writes;
`renderFroxels`/`holdFroxels` map to `update(..., { froxels })`; settings gain the K6 fields; new
`VDBFrameRenderer.core`.

## 7. Verification (N1)

1. **Kernel ↔ CPU twin.** Each kernel runs alone in the raw-WebGPU harness (`examples/src/harness/k6-core.{html,ts}`,
   no three) on fixed seeds; N ≥ 512 texels read back and compared to the `tools/frame` twin within 2 f16 ulp, or a
   bound justified in the step handoff (e.g. SwiftShader's ~12-bit division on sharp data, Traps). Every new twin
   gets a fail-before mutation (break one term, the probe must fail). Spec: `e2e/k6-core.spec.ts`.
2. **Frames vs the TSL path.** While both exist, the K3 harnesses take `?backend=core`. Per step, its K3 scenes
   (Step 1 for K6.1 … Step 7 for K6.6), dense and sparse: PSNR ≥ 45 dB and ≤ 0.2 % pixels > 4/255 against the TSL
   frame, same device.
3. **Frames vs his K3 captures** (`pnpm k3:parity` through `backend: 'core'`): his strict bar (PSNR ≥ 40 dB, ≤ 0.5 %
   pixels > 8/255) on every scene that passes it today; Step 7 IoU ≥ 0.99 and 9,802 points. A regression report lists
   each scene's numbers for both backends. Intentional improvements (mixture, SH irradiance, GPU environment lights,
   MS default, mesh→volume shadows) are listed with their DIVERGES row and the reference result, and excluded from
   the bar. Bit-identity (the old 48/48) is not a gate.
4. **Quality reference (K6.R).** A brute-force march: per pixel, a fixed fine step (≤ ½ voxel) through the source at
   level 0, a full-resolution shadow march per light per sample (no grid, no froxels), the environment from the
   reference SH and lights, and multiple scattering by many-sample random walks with ratio tracking against the
   source's `stats.max`, accumulated in rgba32float over frames until the change per frame is < 1e-4. CPU twin on
   64 × 64 crops. "Better" means lower error against it (PSNR and mean relative luminance error per scene); the
   pathtracer replaces it after P1.
5. **Structural.** `test/boundaries.test.ts`; `api-surface.test.ts` (one reviewed diff in K6.8); a limits test that
   builds every pipeline descriptor of the key matrix and checks §1's limits; `pnpm wgsl:variants` over every
   variant under `pushErrorScope('validation')`; G16a's zero-build swap in the core harness; demo goldens 02, 04–09
   and demos 10–16 structural specs after K6.8 (re-baselined only with a listed reason).

## 8. Migration order and sizes

| Task | Content | Size | Depends on | Parallel with |
|---|---|---|---|---|
| K6.1 | Skeleton (options, settings, lifecycle, retire queue, pipeline cache, uniform packers), froxel layout/view maths in `tools/frame`, 'volume' lighting: gather + resolve + composite pass + `vdbFrameIntegrate`; dense source; `k6-core` harness; `backend` A/B in `./three`. [Brief](./K6.1-brief.md). | L | — | K6.7, K6.R (after K6.1's first commit freezes the group-2 contract) |
| K6.7 | `vdbSparseSourceGpu`: one source per pool, `vdbSrc*(v, …)`, packed pyramid atlas, `followImage`, `skipEmpty` | M | group-2 contract | K6.1, K6.R |
| K6.R | Reference renderer (§7.4) + CPU twin + report script | L | group-2 contract | K6.1, K6.7 |
| K6.2 | 'frame' lighting: gridDensity, transmittance (stacked), envPrefilter + envReduce (GPU), env grid, fused gather, multi-group partials | L | K6.1, K6.7 | K6.6 |
| K6.3 | Phase LUTs; MS family incl. K7.3 `'multi-scale'` and the experimental MLP; grid mips; pick the MS default with K6.R | M | K6.2, **K7.3**, K6.R | K6.5, K6.5b |
| K6.4 | Spectral: keys, stacked spectral targets, alpha prefix | M | K6.3 | K6.5, K6.5b |
| K6.5 | Volume→mesh Beer maps (drawn view, budget), emission kernels, emission data + WGSL hook | M core + L adapter | K6.2 | K6.3, K6.4, K6.5b |
| K6.5b | Mesh→volume occluder maps in transmittance, fused gather and the 'volume' march | M | K6.2 (K6.5's light/shadow types) | K6.3, K6.4 |
| K6.6 | Level-set surface wrapper + raw pass; point sprite WGSL + raw pass | S–M | K6.1 | everything after K6.1 |
| K6.8 | `./three` thin (§6): delete the TSL passes and `backend`, `VDBSequence` to core, the 0.2.0 API, timing harness on timestamp writes | L | K6.1–K6.7, K6.5b | — |
| K6.9 | Raw-WebGPU demo 17 (WDAS, 'frame', mesh shadows both ways, emission) + docs "Rendering without three" | M | K6.5, K6.5b | K6.8 |
| K6.10 | Gate review (separate agent) | S | all | — |
| K6.T | Deferred timing A/Bs (§5.4) | M | a quiet machine | — |

K7.1 is merged into the TSL path (V74); its repro (`e2e/k7-volume-shadow-stale.spec.ts`) becomes a K6.5 test. K7.2 is
superseded by the composite seam. K7.3 writes `'multi-scale'` as core compute with a CPU twin over a 3D texture input (rg16float or
rgba16float, channel selectable); K6.3 wires it. P1 touches only the generator and its snapshot.

## 9. Owner decisions recorded (2026-11-25)

| Q | Decision | Why |
|---|---|---|
| Q1 | B: compute on 3D textures, gather, fused density + lighting, z-scan, fragment only at seams | WebGPU-native; matches the compute-only consumers (cloud system, pathtracer); removes atlas tiles, blending scatter and the 32 B/sample limit. |
| Q2 | The core runs the kernels on the host's (three's) device and allocates its storage textures; `./three` binds them by backend-record swap with a self-test | Storage textures need `STORAGE_BINDING`; one orchestration for every host; the same pattern as the other pmndrs libraries and `VDBGridNode`. |
| Q3 | Environment: a host equirect or cube texture; GPU prefilter, SH projection and light extraction, no readback | Fits compute PMREM and pmndrs/sky outputs; removes the first-frame fallback. |
| Q4 | `Gpu` suffix for core objects | D32's rule. |
| Q5 | The MLP is an experiment: generic evaluator, shapes as uniforms, weights as data; MS default by quality | New networks need no rebuild; his default is not the spec. |
| Q6 | Break `./three` freely in 0.2.0 | No 0.1.x users. |
| Q7 | `VDBSequence` moves to core | Engine-free already; raw hosts get sequences. |
| Q8 | Light terms as uniform-count loops; compiled-in sets deferred | Fewer variants, no rebuild per light; the alternative needs timing. |
| Q9 | Emission colour as data + an optional validated WGSL hook | Engine-free, still extensible. |
| Q10 | Volume→mesh, mesh→volume and self-shadowing all land in K6 | Shadows are first-class. |
| N1–N3 | Correctness/quality gates; core features + default limits baseline; TSL path is the reference until K6.8 | §7, §1, §8. |

**Notes, not blocking.** (a) A `GPUTextureView` exposes no dimension in JS (WebGPU IDL), so the environment input is
a `GPUTexture` with an optional `projection`, inferred from the layer count. (b) `VDB_BAD_WGSL` is a new
`VDBErrorCode` for the emission hook (a type change, no new export). (c) The `'mixture'` froxel mix is a planned
departure decided by K6.R; until then both modes exist.

No new owner questions block K6.1.

## 10. D34 text (for DECISIONS.md)

| ID | Date | Decision | Why |
|---|---|---|---|
| D34 | 2026-11-25 | **The K6 core renderer is compute on 3D textures** ([docs/dev/design/K6-core-renderer.md](./design/K6-core-renderer.md)). The froxel renderer, the 'frame' lighting grids, the environment reduction, volume shadow maps and emission run in `core` as WGSL compute kernels on the host's device, recorded into the host's encoder; the core allocates its storage textures. Froxels, shadow grids, transmittance and environment grids are `texture_3d` (written through `texture_storage_3d`, stacked when several grids share a size), read with hardware trilinear. One thread per froxel gathers the visible volumes (overlapping volumes mix order-independently); 'frame' lighting fuses density and lighting; a per-column z-scan (with temporal reprojection) writes integrated radiance and optical depth, which the composite reads at the scene depth. Fragment shaders exist only at host seams: the composite (a core pass, or the WGSL function `vdbFrameIntegrate` the host places), level-set `frag_depth`, points. Inputs are plain data: camera, lights, a host equirect/cube environment texture (reduced on the GPU to SH and lights, no readback), host depth shadow maps (mesh→volume shadows), shadow requests (volume→mesh maps read with the view they were drawn with). Multiple scattering is pluggable (default chosen by quality against a brute-force reference; the MLP is experimental, weights as data). Emission colour is data plus an optional validated WGSL hook. Baseline: WebGPU core features and default limits; `texture-formats-tier1`, `shader-f16` and `subgroups` are fast paths with fallbacks. Gates: every kernel matches its CPU twin (≤ 2 f16 ulp or a justified bound) with a fail-before mutation; frames match the TSL path and his K3 captures within tolerance, with intentional improvements listed; a reference renderer judges "better". K3 bit-identity is no longer a gate; timing A/Bs wait for a quiet machine (D25). The TSL path is the reference during K6.1–K6.7 and is deleted in K6.8; `./three` 0.2.0 becomes a thin adapter. Core objects take `Gpu` (D32); `VDBSequence` moves to core. Owner decisions 2026-11-25 (Q1–Q10, N1–N3). | Owner: the best VDB library on the web, WebGPU-first, with mjurczyk's renderer as a benchmark and collaborator, not the spec. Compute on 3D textures is the WebGPU-native design and what the compute-only consumers (the cloud system, the pathtracer) need; it removes his WebGL-era atlas tiles, scatter-by-blending, the 4 × rgba16float density targets and the 32 B/sample limit. His post-0.26 fixes (62a6e3e, c13f2d5, 720c38f) and our K5.9/RV bugs came from engine coupling, which plain-data inputs, explicit bind groups, a retire queue and a composite function remove. No quiet machine is available, so the design is chosen on correctness and quality, and every timing question is one deferred task. |

## 11. Planned DIVERGES rows

One line each; the implementing step writes the row (source: mjurczyk `75e7345` unless noted).

1. K6.1 — Froxels are 3D textures written by compute, not his z-slice 2D atlas drawn by instanced fragment passes (`FrameRenderer`, `frameLayout.js`).
2. K6.1 — One thread per froxel gathers the volumes of a source group instead of one blended draw per volume (`FrameInjectMaterial` blending).
3. K6.1 — Overlapping volumes in a froxel mix order-independently (`froxelMix: 'mixture'`) instead of draw-order premultiplied "over"; decided by K6.R.
4. K6.1 — Integration is a per-column prefix of (radiance, optical depth) and the composite reads two boundary taps; x, y filtering acts on the prefix, not on per-slice froxels (`FrameIntegrateMaterial` L219–294).
5. K6.1 — Temporal reprojection runs inside the z-scan; the history is sampled with hardware trilinear (same clamps as his bilinear + slice lerp) (`FrameTemporalMaterial`).
6. K6.1 — Froxel size is capped at `maxTextureDimension3D` (2048) per axis; at most 128 visible volumes per frame (uniform binding size), warned.
7. K6.1 — Light terms are loops over uniform counts (reverses V55's compiled-in light sets; values unchanged).
8. K6.2 — 'frame' lighting fuses the density and lighting passes in one kernel; no 4 × rgba16float density targets (`FrameDensityMaterial`, `FrameLightingMaterial`).
9. K6.2 — Shadow grid, transmittance and environment grid are stacked 3D textures with hardware trilinear, not 2D atlases with manual z lerp (`frameGrid.js`).
10. K6.2 — Without `texture-formats-tier1` the shadow grid and grid mips are rgba16float (his RG half float); values in `.xy` unchanged (platform).
11. K6.2 — Environment input is a host equirect/cube texture prefiltered by our kernel, not three's PMREM at roughness 0.4 (`FrameRenderer` env map).
12. K6.2 — 'outward' environment irradiance is SH L2 (Ramamoorthi–Hanrahan), not PMREM roughness 1 × π.
13. K6.2 — Environment lights ('directional', 'spherical-harmonics') are extracted on the GPU into a buffer; no CPU readback, no first-frame 'outward' fallback (`frameEnvironment.js`).
14. K6.3 — Multiple scattering default is chosen by quality against K6.R between diffusion and multi-scale; his `'neural'` default is not followed (V67 note).
15. K6.3 — The neural network runs in a generic MLP evaluator (shapes as uniforms, weights in a storage buffer), experimental, `'mlp'` (`frameNeuralChunks.js`; extends V48).
16. K6.3 — `'multi-scale'` comes from K7.3's core compute on 3D textures (his 4e2d46f is past the pin).
17. K6.5 — Volume shadow receivers read with the view the map was drawn with (his 62a6e3e, by construction) and the update budget counts only lights in the frame (his c13f2d5).
18. K6.5 — Emission colour is data (named spectrum or LUT) plus an optional validated WGSL hook (supersedes V24/V62 for the core).
19. K6.5b — Meshes shadow volumes through the host's depth shadow maps (not in his renderer): direct term at froxel resolution, scattered term in the grid.
20. K6.7 — One sparse source serves every volume of a pool, with pyramids packed in one atlas (replaces V30/V42's one material per source).
21. K6.8 — The MRT/post seam is a composite function in the host's material (his 720c38f solved structurally); `passHook` times recorded kernels via timestamp writes (extends V34/V46/V49).
22. K6.8 — `VDBSequence` lives in core (engine-free), re-exported by `./three`.
