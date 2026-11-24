# K6 design note: the renderer in the core

Status: design only, 2026-11-24 (K6.0). Awaiting owner review before K6.1 starts. Plan context:
[PHASES.md](../PHASES.md) K6/K7/P1, [TASKS.md](../TASKS.md) K6.0–K6.10, [DECISIONS.md](../DECISIONS.md) D8, D10, D20,
D21, D28–D33. Inputs: [R1](../research/notes/R1-renderer-port.md), [K1-core](./K1-core.md) (pool, generator, budgets),
the K3 handoffs ([K3-1](../../archive/handoffs/K3-1.md), [K3-2a](../../archive/handoffs/K3-2a.md),
[K3-2d](../../archive/handoffs/K3-2d.md), [K3-3](../../archive/handoffs/K3-3.md), [K3-6](../../archive/handoffs/K3-6.md),
[K3-7](../../archive/handoffs/K3-7.md)), [DIVERGES.md](../DIVERGES.md) V23–V67, and the 2026-11-24 research on his
post-0.26 fixes (K7.1, K7.2, K7.3).

## 0. Decisions at a glance

| Topic | Proposal |
|---|---|
| Who runs the passes | The core, into the host's encoder on the host's device. `./three` collects scene data, owns the proxies and composites through a material. |
| Pass style | Fragment passes on today's 2D z-slice atlases, formats and blend states, checkable texel for texel against the TSL path. Compute/3D textures: a later A/B (Q1). |
| Inputs | Plain data: camera matrices, a light list (directional, point, spot, ambient, hemisphere, probe), environment textures, a frame id. No engine objects. |
| Outputs | Textures the host reads (froxel atlas, volume shadow maps, emission probes), core-allocated or from a host allocator hook (the pyramid's `targets`, generalised). |
| MRT / post seam | Composite by a core pass into host targets, or by the WGSL function `vdbFrameIntegrate` in the host's own pass or material. `./three` uses the function: three's MRT and `pass()` reach the composite, never the froxel passes. |
| Shadow maps | The host lists lights with their shadow cameras (or asks the core to fit them); each map comes back with the view it was drawn with, which receivers use instead of the host's `shadowCoord` (K7.1 by construction). |
| Shader composition | One `vdb` module (union of helpers) + `vdbSrc*` + the `vdbFrame*` chunk closure + the entry. Keyed by a hash of the text; boxes in uniforms, so a grid swap builds nothing (G16a). |
| Ownership | The core destroys only what it allocated, two updates after retiring it (the pool's contract), and registers no listeners: it polls `version` / `revision` / `imageRevision`. |

## 1. Goals, non-goals, constraints

**Goals.** (1) A WebGPU host with no engine renders the froxel renderer ('volume' and 'frame' lighting, phase,
multiple scattering incl. neural and multi-scale, spectral, temporal), volume shadows on its meshes, emission
light, level sets and points (PHASES K6 exit gate: the WDAS cloud, 'frame' lighting, mesh shadows, emission, without
three). (2) `./three` draws through the same passes and keeps the K3 parity suite at 48/48 and the demo goldens.
(3) Every core pass has a CPU twin in `tools` (D11). (4) The three-specific bug classes found in K3–K5 can't happen
in the core by construction (§4).

**Non-goals for K6.** A new lighting model or new defaults (V67 stays). Delta tracking and majorants (P1's
`majorant` profile, D22). Other adapters (R6; K6 only has to make them possible). A compute rewrite of the pass
graph (Q1). Removing the dense source or the 2D atlases. Shading level-set and point surfaces with the core's
own BRDF inside `./three` (three shades them; §3.6).

**Constraints.**
- D8: the host owns the `GPUDevice` and the encoder. The core records into the host's encoder and never submits;
  the environment readback copies in the host's encoder and maps its core-owned buffer two updates later (the
  pool's submit contract), so a map never precedes the host's submit (Traps: `read()` before `submitted()`). `dispose()` frees only what the core allocated.
- D20: `core` never imports `three`; it may import engine-free maths and twins from `tools/frame` (as
  `core/compute/pyramid.ts` already imports `tools/frame/pyramid-layout.ts`); `tools` never imports `core`.
- D10: grid data reaches a kernel through **one** storage buffer (the pool) at a slot. A pass reads one source
  at a time, so passes over volumes in different pools (demo 13's one pool per large grid) work: each draw binds
  its source's group. Dense sources bind textures, a compile-time choice (R1 §4.2).
- D21: WGSL bytes are budgeted and snapshotted (§5.3). One VDB module per shader (Traps).
- WebGPU default limits the pass graph must fit, asserted per pipeline by a unit test: 4 bind groups, 10 storage
  buffers per stage (Apple), 16 sampled textures per stage, 8 colour attachments, **32 colour bytes per
  sample** (the grey density pass and transmittance pass are 4 × rgba16float = 32 B, exactly at the limit),
  `maxStorageBufferBindingSize` 128 MiB per pool (Traps: demo 13), 64 KiB uniform bindings. The spectral
  lighting pass reads 4 groups × 3 transmittance targets: in the core these are layers of one 2D-array texture
  (one binding), which keeps it under 16 textures.
- Compatibility-mode devices may expose no fragment-stage storage buffers (K1-core §6 item 4): the sparse source
  then throws `VDB_UNSUPPORTED_DEVICE`, saying so.

## 2. Public core API sketch

### 2.1 Objects and names (D32)

`./three` keeps the plain names (`VDBFrameRenderer`, `VDBFrameVolume`, `vdbSparseSource`), so the core's
encoder-level objects take `Gpu`, as `VDBBoxPyramidGpu` / `vdbCreateBoxPyramid` already do (Q4).

| Core export (`.`) | Kind | Notes |
|---|---|---|
| `vdbCreateFrameRenderer(device, options)` → `VDBFrameRendererGpu` | function / type | The pass graph and its resources. |
| `VDBFrameVolumeGpu` | type | Handle from `renderer.addVolume(source, params)`. |
| `vdbSparseSourceGpu(pool, grids, options)` → `VDBVolumeSource` | function | K6.7 (lifts K3.2b): the sparse source over a pool slot, the GPU pyramid, `followImage`. |
| `vdbDenseSource` | function | Exists (K3.0). |
| `VDBFrameSettings`, `VDBFrameVolumeParams`, `VDBFrameInput`, `VDBFrameCamera`, `VDBFrameLight`, `VDBFrameEnvironment`, `VDBFrameTargets`, `VDBFrameOutputs`, `VDBVolumeShadowRequest`, `VDBVolumeShadowMap`, `VDBTextureAllocator` | types | Plain data. `VDBFrameRendererSettings` (three) becomes `VDBFrameSettings`, one declaration re-exported (D32). |
| `vdbFrameIntegrateWgsl(options)`, `vdbVolumeShadowWgsl(options)`, `vdbEmissionProbeWgsl(options)` | functions | WGSL text for hosts that composite or receive in their own shaders (§2.5). |
| `vdbFrameLights`, `vdbFrameAllLights`, phase packing | values | Move from `./three` (`lights.ts`, `phase.ts` are already engine-free); one declaration each. |
| `VDBSequence` | class | Already engine-free (it imports only `core` errors and `tools`); moves to core, re-exported by `./three` (Q7). |

### 2.2 Creation, settings, volumes

```ts
interface VDBFrameRendererGpuOptions {
  settings?: Partial<VDBFrameSettings>;      // his names and v0.26 defaults (V67), as VDBFrameRendererSettings today
  types?: readonly VDBGridTypeName[];         // grid types any sparse source may hold (default VDB_ALL_SCALAR_TYPES): compiled once (G16a)
  composite?: { colorFormat: GPUTextureFormat; depthFormat: GPUTextureFormat; sampleCount?: 1 } | null; // the core composite pass (§2.5)
  allocator?: VDBTextureAllocator;            // host-allocated outputs (§4.1)
  neuralNetwork?: VDBFrameNeuralNetwork | null; // default: his weights (V48); a uniform buffer, not shader constants (Q5)
  label?: string;
}

interface VDBFrameRendererGpu {
  readonly device: GPUDevice;
  readonly settings: VDBFrameSettings;        // mutable; read at the next update()
  readonly outputs: VDBFrameOutputs;          // §2.5; stable objects, their textures change with `outputs.version`
  readonly redrawStats: VDBFrameRedrawStats;  // V46
  addVolume(source: VDBVolumeSource, params?: Partial<VDBFrameVolumeParams>): VDBFrameVolumeGpu;
  prepare(options?: { spectral?: boolean[]; lighting?: ("volume" | "frame")[] }): Promise<void>; // warm every pipeline the settings and sources can need
  update(encoder: GPUCommandEncoder, frame: VDBFrameInput, options?: { timestampWrites?: VDBFrameTimestamps }): VDBFrameUpdate;
  render(encoder: GPUCommandEncoder, targets: VDBFrameTargets): void;   // the core composite (needs options.composite)
  dispose(): void;
}

interface VDBFrameVolumeGpu {
  readonly source: VDBVolumeSource;
  params: VDBFrameVolumeParams;   // matrixWorld, bbox, extinction, extinctionScale, angstromExponent, baseColor,
                                  // scatterColor, densityScale, normalizeDensity, densityMean, densityCutoff, absorbance,
                                  // opacity, emissiveIntensity, temperatureRange, lights (bitmask), shadowSteps, mipmaps,
                                  // hdda, phase overrides ('volume' lighting, V65), wrap/offset: his FrameVolume props;
                                  // emissionColor: 'white' | { spectrum: 'planck' | 'soot', exponent } | { lut } (Q9)
  visible: boolean;               // drawn this frame (host culling and layers)
  castsShadow: boolean;           // injected into the shadow grid ('frame' lighting), also when not visible (his casters)
  remove(): void;
}
```

`params` is plain data with one writer (the host). The renderer packs it into the per-volume uniform block in
`update()`; nothing else writes those bytes (the K5.9a class, §4.3).

### 2.3 Frame input: camera, lights, environment, time

```ts
interface VDBFrameInput {
  frameId: number;                     // monotonic; frame identity (jitter index, temporal, dedupe). Replaces his
                                       // renderer.info.render.calls + camera heuristic (R1 §8 quality 2)
  camera: VDBFrameCamera;
  viewport: { drawingBuffer: [number, number]; canvas: [number, number] }; // getFroxelSize, pixelsPerFroxel
  lights: readonly VDBFrameLight[];    // only lights in the rendered scene: the shadow budget counts these (K7.1 c13f2d5)
  environment?: VDBFrameEnvironment | null;
  time?: number;                       // seconds; recorded in redraw signatures only, no pass reads it
}
interface VDBFrameCamera {
  view: Float32Array;                  // world → view, column-major 4x4
  projection: Float32Array;            // and its inverse is computed in f64 on the CPU
  near: number; far: number;
  orthographic: boolean;
  reversedDepth: boolean;              // three's camera.reversedDepth / renderer.reversedDepthBuffer
  id?: unknown;                        // history is valid only for the same id (his historyCamera === camera)
}
type VDBFrameLight = { id: number | string; mask?: number } & (   // mask: his vdbFrameLights type bit, default by kind
  | { kind: "ambient"; color: VDBVec3 }                            // colour × intensity, linear
  | { kind: "hemisphere"; sky: VDBVec3; ground: VDBVec3; up: VDBVec3 }
  | { kind: "probe"; sh: Float32Array }                            // 9 × rgb, three's SH9 convention
  | { kind: "directional"; direction: VDBVec3; color: VDBVec3; shadow?: VDBVolumeShadowRequest } // direction: toward the light
  | { kind: "point"; position: VDBVec3; color: VDBVec3; distance: number; decay: number; shadow?: VDBVolumeShadowRequest }
  | { kind: "spot"; position: VDBVec3; direction: VDBVec3; color: VDBVec3; distance: number; decay: number;
      cosAngle: number; cosPenumbra: number; shadow?: VDBVolumeShadowRequest });
interface VDBFrameEnvironment {
  id: unknown; revision: number;       // a change redraws the env map, env grid and readback
  radiance: GPUTextureView;            // equirect, linear, rgba16float/rgba32float, sampled with a linear sampler
  irradiance?: GPUTextureView;         // equirect, PMREM-roughness-1 radiance × π (his IBL irradiance); default: the core prefilters `radiance`
  rotation?: Float32Array;             // 3x3, world → map (backgroundRotation / environmentRotation)
  intensity: number;
}
```

Light ordering and slot assignment (`getSortedLight`, the first ≤ 6 shadow slots, the dominant directional
environment light in the next free slot, the rest to the environment grid) move unchanged into the core's
`update()`; the three adapter only builds the list. The 16-light and 4-hemisphere caps stay and warn once.

**Environment.** Today the passes sample three's PMREM (CUBE_UV) through `pmremTexture` at roughness 1 and the CPU
derives ≤ 12 environment lights from a 64 × 32 lat-long readback (`tools/frame/environment.ts`, engine-free). In the
core the host gives equirect textures; the core renders the 64 × 32 map from them, reads it back into its own
buffer and runs the same `VDBFrameEnvironmentLights` (its `settled` promise stays). The three adapter bakes
`irradiance` and `radiance` from the PMREM once per environment change with the existing env-map pass (Q3).

### 2.4 Lifecycle and the host's frame

```ts
// once
const frame = vdbCreateFrameRenderer(device, { composite: { colorFormat, depthFormat }, types: VDB_ALL_SCALAR_TYPES });
const volume = frame.addVolume(vdbSparseSourceGpu(pool, { density: id }, { followImage: true }), { extinction: 100 });
await frame.prepare();                           // optional warm-up (createRenderPipelineAsync for every variant)

// per frame, one encoder
const enc = device.createCommandEncoder();
pool.update(enc);                                // host-owned pool first (Traps: bind groups after the pool's update)
const u = frame.update(enc, { frameId, camera, viewport, lights, environment });
//   records: source pyramids that are stale, shadow density + grid mips + transmittance + env grid (signature-cached),
//   volume shadow maps (budgeted), emission probes, the froxel passes, temporal.
drawOpaque(enc, u.outputs);                      // receivers sample u.outputs.volumeShadows / emissionProbes
frame.render(enc, { color, depth, sceneColor }); // composite (or call vdbFrameIntegrate in the host's own pass)
device.queue.submit([enc.finish()]);
```

- **update** does all CPU work (collection of visible volumes and casters, frustum clip and slice range
  (`view.ts`), grid cascades and signatures (`tools/frame/grid.ts`), light slots, uniform packing), rebuilds bind
  groups whose inputs changed (pool `version`, source `version`, outputs reallocated), then records passes. It
  returns `VDBFrameUpdate { outputs, froxelArea, drawn: VDBFramePass[], skipped: VDBFramePass[] }`.
  `options.timestampWrites` maps pass names (V34/V46/V49) to timestamp query slots, replacing three's
  render-pass timestamps for timing (`k3-pass-timer.ts` moves to it).
- **render** draws a fullscreen triangle per pixel area into `targets.color` (load, premultiplied blend), reading
  `targets.depth` as `texture_depth_2d` (or a sampled `r32float` view) and, for `srgbBlending` or spectral, the
  scene colour from `targets.sceneColor`. Without `sceneColor` the core copies `color` into its own scratch
  first (needs `COPY_SRC`). Single-sample only; MSAA hosts composite after their resolve or use the WGSL function.
- **Resize** is implicit: froxel sizes come from `settings` and `viewport` each update (`getFroxelSize` with its
  15 % hysteresis); grid sizes from `shadowResolution` / `shadowCascades`. A size change reallocates, retires the
  old textures (§4.2), bumps `outputs.version` and invalidates the history.
- **dispose** destroys every texture, buffer and query set the renderer allocated, current and retired; calls
  `allocator.release` for host-allocated ones; never touches the device, the pools or the sources (their
  creators dispose them; `VDBVolumeSource.dispose()` exists).

### 2.5 Outputs and the MRT / post-processing seam

```ts
interface VDBFrameOutputs {
  readonly version: number;                 // +1 when any texture below is a new object: rebuild host bind groups
  readonly froxels: { color: GPUTexture; alpha: GPUTexture | null; layout: VDBFroxelLayout } | null; // atlas + spectral alpha
  readonly volumeShadows: ReadonlyMap<number | string, VDBVolumeShadowMap>;  // by light id
  readonly emissionProbes: { maps: [GPUTexture, GPUTexture, GPUTexture]; grid: VDBFrameGridLayout } | null;
  readonly uniforms: GPUBuffer;             // the integrate / receiver uniform block, for host shaders
}
interface VDBTextureAllocator {             // optional: the host allocates the outputs it samples
  create(descriptor: GPUTextureDescriptor, role: "froxels" | "froxelAlpha" | "volumeShadow" | "emissionProbe"): GPUTexture;
  release(texture: GPUTexture, role: string): void;  // after the retire delay; the core never destroys these
}
```

Composite options: (a) `frame.render(encoder, targets)`, the core's own pass; (b) `vdbFrameIntegrateWgsl({
spectral, srgbBlending })` returns `fn vdbFrameIntegrate(pixel: vec2f, sceneViewZ: f32, sceneColor: vec3f) ->
vec4f` plus its bindings (`vdbFrameIntegrateEntries(slot)`) for a host that composites in its own pass, its own
MRT layout, or inside a material. (b) is the seam that removes the K7.2 class: the froxel passes are raw passes
that no host MRT or `pass()` setting can reach, and the only code in the host's pass is a function the host
places. The same holds for receivers: `vdbVolumeShadowWgsl()` gives `fn vdbVolumeShadow(light: u32, world: vec3f,
viewDepth: f32) -> vec3f` (grey in `.x`, spectral in `.xyz`; twin `vdbFrameShadowReceiver`), and
`vdbEmissionProbeWgsl()` gives `fn vdbEmissionIrradiance(world: vec3f, normal: vec3f) -> vec3f` (twin
`vdbFrameEmissionLookup` + `vdbFrameEmissionIrradiance`).

### 2.6 Volume shadows on meshes

```ts
interface VDBVolumeShadowRequest {
  size: number;                                  // castShadowMapSize
  view?: { viewProjection: Float32Array; near: number; far: number; orthographic: boolean }   // spot / point face: the host's shadow camera
       | { cascades: { viewProjection: Float32Array; near: number; far: number }[]; breaks: number[] } // directional CSM
       | "fit";                                  // core fits around the casters (tools/frame/shadows.ts cascade maths)
}
interface VDBVolumeShadowMap {
  texture: GPUTexture; spectral: GPUTexture | null;   // rgba16float (τ·first depth, τ·last depth, 0, τ); cube: 6 layers
  drawn: { viewProjection: Float32Array[]; origin: VDBVec3; forward: VDBVec3; far: number; perspective: boolean;
           breaks: number[] } ; // the state the map was drawn with
  revision: number;                              // bumps when any piece was redrawn
}
```

The host passes the same camera its own depth shadow map uses, so both maps share one texel grid; the core never
reads the host's depth map (the receiver multiplies the host's opaque shadow by `vdbVolumeShadow`, as three's
`filterNode` does today). Receivers project the world position with `drawn`, so a map drawn last frame is read
with last frame's camera: his 62a6e3e (spot/point looked up with three's current `shadowCoord`, not the light state
the map was drawn with; our `volume-shadows.ts` L618 has the same pattern) is impossible. The per-update budget
(`castShadowUpdatesPerFrame`, oldest piece first, the oldest always drawn) counts only lights in this frame's
`lights` (his c13f2d5).

The K6 brief had the host hand over its depth shadow map too. Volume-on-mesh shadows don't need it: the Beer map
stores the first and last volume depth along each texel's ray and the receiver compares its own depth. It would be
needed for meshes shadowing volumes, which his renderer does not do; `VDBVolumeShadowRequest` leaves room for an
`occluder: { depth: GPUTextureView }` field (Q10).

### 2.7 Sequences and coarse-first

Sequences stay host-driven: `VDBSequence` calls `pool.replace` (sparse) or `source.replace` (dense). The renderer
polls in `update()`: a grid whose `imageRevision` moved marks its pyramid stale (rebuilt in the same encoder after
`pool.update`, which is valid in one encoder; the three-only rule "rebuild after the update was submitted" goes
away) and moves the caster signature; a dense source's `version` bump re-reads its box (RV(8)). The sequence-wide
mean stays a param (`densityMean`, V66). Coarse-first: `vdbSparseSourceGpu(..., { followImage: true })` re-reads
boxes, transform and pyramid size from `pool.info` once the new image is resident; boxes, sizes and the LOD range
are uniforms, and `types` was fixed at renderer creation, so a swap creates zero pipelines and zero shader
modules (G16a's test moves to the core harness).

## 3. Pass graph

### 3.1 Shared bindings

Every pipeline layout is groups 0..2 with no gaps (Traps: a skipped `@group` silently zeroes outputs); group 3 is
unused.

| Group | Binding | Contents |
|---|---|---|
| 0 frame | 0 | `uniform VdbFrameUniforms` (camera, froxel layout, range, jitter, fade, settings, phase, octaves, ms, grid layout) |
| | 1 | `uniform VdbFrameLights` (`frameMaxLights` 16, hemisphere 4, probe SH9, environment 12, shadow slots 6) |
| | 2 | `uniform VdbFrameVolume`, dynamic offset, 512 B per volume (matrices, colours, extinction spectrum, flags) |
| | 3, 4 | linear-clamp and nearest-clamp samplers |
| | 5, 6 | LUTs: Mie / Mie RGB (`texture_2d<f32>`, 2048 texels); emission colour (256 texels, one row per distinct `emissionColor`, row index in the volume block) |
| | 7 | `uniform VdbFrameNeural`, his 20 → 32 → 32 → 1 weights, 1,761 f32 (≈ 7 KiB) |
| 1 pass | 0.. | the pass's inputs (previous passes' textures) |
| 2 source | `source.layoutEntries({ group: 2, firstBinding: 0 })` | sparse: the pool (1 storage buffer) + pyramid base/atlas (r16float); dense: channel textures + mip atlases |

Storage buffers per stage: at most 1 (the pool). Uniform structs are packed by one TS packer per struct, checked
by a unit test against the WGSL struct's offsets (the struct text is generated from the same field list).

### 3.2 Pass list

Atlas = `getAtlasLayout(tileW, tileH, tiles)` z-slice atlas (unchanged). "Cached" = redrawn only when its
signature changes (`tools/frame/grid.ts`). All TSL files are under `packages/vdb/src/three/frame/`.

| Pass (WGSL entry) | Reads | Writes (format, blend) | When | Replaces | CPU twin (`tools/frame`) |
|---|---|---|---|---|---|
| `pyramid` (exists, compute) | pool grid | r16float base + atlas | grid `imageRevision` moved | — (core already) | `vdbBoxPyramid` |
| `inject` (`vdbFrameInjectFs`), 'volume' lighting | source (froxel lod, march lod), lights, env lights | froxel atlas rgba16float (+ rg16float alpha spectral), his blend | per visible volume, instanced slices | `inject-material.ts` | `vdbFrameInjectSlice`, `vdbFrameShadowMarch`, `vdbFrameInjectIsotropicIrradiance`, spectral twins |
| `shadowDensity` (`vdbFrameShadowDensityFs`) | source at grid-voxel lod | grid atlas rg16float (spectral 2 × rgba16float), additive | per caster per cascade, cached | `shadow-density-material.ts` | `vdbFrameShadowAbsorption` + source twin; **new** voxel twin |
| `gridMips` | last cascade | 4 levels rg16float | neural, on grid redraw | `grid-mip-material.ts` | `vdbFrameGridMipLevel` |
| `gridMipsMultiScale` (K7.3, compute) | last cascade | binomial mips | `'multi-scale'`, on redraw | — (K7.3 writes it in core) | K7.3's twin |
| `transmittance` (`vdbFrameTransmittanceFs`) | grid atlas, mips, neural weights | 4 × rgba16float (spectral: 3 layers per slot group of one 2D array) | cached | `transmittance-material.ts`, `neural.ts` | `vdbFrameTransmittanceVoxel`, `vdbFrameSpectralTransmittanceVoxel`, `vdbFrameNeuralLight` |
| `environmentMap` | `environment.radiance` | 64 × 32 rgba32float + readback | environment revision | `environment-material.ts` (map) | `vdbFrameEnvironmentMapDirection`, `VDBFrameEnvironmentLights` |
| `environmentGrid` | last cascade, env lights | rgba16float R³ atlas | cached | `environment-grid-material.ts` | `vdbFrameEnvironmentGridVoxel` (+ spectral) |
| `density` (`vdbFrameDensityFs`) | source (froxel lod), transmittance, env grid, irradiance | 4 × rgba16float (spectral 3), additive, nearest | per visible volume, instanced slices | `density-material.ts` | `vdbFrameDensityFroxel`, `vdbFrameSpectralDensityFroxel` |
| `lighting` (`vdbFrameLightingFs`) | density targets, transmittance array (spectral), LUTs | froxel atlas (+ alpha), no blend | once | `lighting-material.ts` | `vdbFrameLightingFroxel`, `vdbFrameSpectralLightingFroxel` |
| `temporal` | atlas, history | history ping-pong (2 × atlas) | `temporal` | `temporal-material.ts` | `vdbFrameReproject` |
| `composite` / `vdbFrameIntegrate` | atlas (or history), depth, scene colour | host colour, premultiplied or self-blended | `render()` or host pass | `integrate-material.ts` | `vdbFrameIntegrate`, `vdbFrameSrgbBlend`, `vdbFrameSpectralIntegrate` |
| `volumeShadow` (`vdbFrameVolumeShadowFs`) | last cascade | rgba16float tiles / cube layers (+ spectral) | budgeted per light piece | `volume-shadow-material.ts`, the A half of `volume-shadows.ts` | `vdbFrameShadowMapTexel`, `vdbFrameShadowReceiver`, cascade twins |
| `emissionDirection`, `emissionReduce`, `emissionProbe` | source emission (lod 0) + extinction; reduce; emitters | rgba32float 2 × (16 directions × slots; 8² rays each), 2 × (1 × slots); probes 3 × rgba16float atlas | on volume / cascade change | the pass half of `emission-light.ts` | `vdbFrameEmissionDirectionTexel`, `vdbFrameEmissionReduce`, `vdbFrameEmissionProbe`, `vdbFrameEmissionProbeGrid` |
| `levelSet` (`vdbLevelSetSurfaceFs`) | pool (level set) | host colour + depth (`frag_depth`), or a G-buffer | per level-set draw | trace part of `level-set-material.ts` (§3.6) | `levelSetHit`, `levelSetNormal` |
| `points` (`vdbPointsVs/Fs`) | point vertex buffer | host colour + depth | per draw | `points.ts` (§3.6) | **new** sprite size twin |

Order inside `update()` for 'frame' lighting: pyramids → shadowDensity → gridMips → transmittance →
environmentMap → environmentGrid → volumeShadow → emission → density → lighting → temporal. 'volume' lighting:
pyramids → inject → temporal (+ the shadow grid only when `castShadows` needs it, as today's `usesShadowGrid`).

### 3.3 'volume' lighting (K6.1)

The inject entry lights each froxel of one volume: sub-steps across the slice at the froxel lod
(`vdbFrameBoxSteps`), then per light a shadow march through the source. In TSL every march was inlined (V55: up to
8 marches per shader). In WGSL the march is one function, `vdbFrameShadowMarch(v, o, d, steps, lod) -> vec4f`,
called from one loop over the light list and the isotropic directions, so the source's `vdbSample` / `vdbSampleFast`
appears at two call sites per entry (the slice sub-steps and the march), not one per light. The level-set trace taught why it matters: six inlined sample sites took
98–168 s to compile on SwiftShader, one site 10–19 s ([K3-7](../../archive/handoffs/K3-7.md), `core/wgsl/level-set.ts`).
Light terms are loops over uniform counts rather than compiled-in sets (V55's per-scene rebuild goes away); if a
hardware A/B shows the zero-count loops cost time, the light-kind mask becomes a pipeline key.

### 3.4 'frame' lighting (K6.2)

Ported pass for pass. The CPU side is already engine-free: cascades, caster ranges and signatures
(`tools/frame/grid.ts`), environment lights (`tools/frame/environment.ts`). The signature cache keeps V44's fields
plus the light list's ids. `prepareShadowGrid` (the shadow grid drawn early for mesh-shadow receivers, K3.6)
becomes an ordering fact of `update()`: grid passes always precede `volumeShadow`.

### 3.5 Scattering, spectral, emission (K6.3–K6.5)

- **Phase:** HG, dual-lobe, Lorenz-Mie and Mie RGB from the committed tables (`tools/frame/mie-table.ts`, V47) as
  LUT textures with `textureLoad` + lerp, as his `load`; `phaseGExponent` per channel.
- **Multiple scattering:** `false`, `true` (1D diffusion), `'two-sided'`, `'octaves'`, `'neural'` (weights in group 0
  binding 7; a swapped network rewrites the buffer and redraws, no pipeline build), and `'multi-scale'` (K7.3,
  written once in core, before K6.3). The mode family is a pipeline key; its numbers are uniforms.
- **Spectral:** a pipeline key. Slot-group transmittance targets become layers of one `texture_2d_array`
  (4 groups × 3); render passes write three layers as three colour attachments.
- **Emission colour:** a 256-texel LUT (temperature → rgb), built on the CPU by the blackbody twin
  (`tools/frame/blackbody.ts`, D33's analytic CMF) or given by the host (`{ lut }`, which is how the parity harness
  injects the exact CIE table). It replaces the TSL `emissionColor` hook (Q9).

### 3.6 Level sets and points (K6.6)

The trace is already core WGSL (`vdbLevelSetHit`, `vdbLevelSetNormal`, opt-in helper `levelSet`). Surfaces are
lit by the host's own BRDF, lights and shadows, so the core does not take over three's shading. It adds:
`vdbLevelSetSurfaceWgsl()`, a fragment-stage wrapper `fn vdbLevelSetSurface(g, pixel, invViewProjection, camera)
-> VdbSurfaceHit { hit, t, world, normal, ndcDepth }` with the pixel-centre ray of V57; and a reference raw pass
(`vdbCreateLevelSetPassGpu`) that writes `frag_depth` plus a Lambert + ambient shade or a G-buffer (normal,
depth), for engine-free hosts and the K6.9 demo. `VDBLevelSetMaterial` stays a `MeshStandardNodeMaterial` calling
the same function through `wgslFn`. Points: the vertex layout from `vdbPointRenderData` and the sprite size and
attenuation maths (his `pointsCore`) become core WGSL (`vdbPointSpriteVertex`), used by a raw sprite pass and by
`VDBPoints`' TSL vertex; the disc and its mipmapped map stay in the adapters.

## 4. Resources, ownership, caching

### 4.1 Who owns what

| Resource | Owner | Lifetime |
|---|---|---|
| Device, encoder, queue submits | host | — |
| Pools, sources | host (sources from `vdbSparseSourceGpu` / `vdbDenseSource`) | the source owns its pyramid textures and scratch (K3.2a) |
| Froxel atlas, history, density MRT, grid atlases, mips, transmittance, env grid/map, readback buffer, emission targets, volume shadow maps, uniform buffers, LUTs, query sets | renderer | until reallocated (retired) or `dispose()` |
| Outputs created through `allocator` | host | the core calls `release`, never `destroy` |
| Pipelines, layouts, shader modules | per-device cache in the core (WeakMap) | the device |

### 4.2 The retire queue

A texture or buffer replaced in `update()` (resize, spectral toggle, a pyramid of a new size under `followImage`,
a cascade count change) is not destroyed at once: three r187 destroys at once and K3.5/RV(5) had to defer it
(Traps). The core keeps it in a retire list tagged with the update count and destroys (or `release`s) it two
updates later, the pool's contract: "submit each update's encoder before the second update after it" (demo 08,
K1-core §1.4). `dispose()` drains the list.

### 4.3 How the K5.9 / RV lessons map

| Lesson | In the core |
|---|---|
| K5.9a: a uniform's `onRenderUpdate` overwrote the per-draw density scale | No callbacks. Per-volume bytes are packed once per update from `params` and the source's stats; one writer per field. A test changes `densityScale` on dense and sparse and checks the froxel atlas moves on both (the bug's regression). |
| K5.9b: three caches a `Fn().setLayout()` body per backend across builders | WGSL is plain text; nothing is cached but pipelines, keyed by the hash of the full text plus layout and target formats. |
| RV(4): source listeners outlived their volumes | The renderer registers nothing on sources or pools; it compares `pool.version`, `pool.revision`, `info(id).imageRevision`, `source.version`, `environment.revision` each update. Nothing to unsubscribe. |
| RV(5) / Traps: `texture.dispose()` destroys immediately | Retire queue (§4.2). |
| V53: three's `TextureNode` uniform hash shares bindings by texture uuid | Bind groups are explicit; unused slot groups bind their own placeholder (or are layers of one array texture). |
| Traps: build the bind group after `pool.update` | `update()` builds groups after reading `pool.version`; the documented host order puts `pool.update` first. |
| Traps: one VDB module per shader; the source needs the union of helpers | The renderer computes the union for each pipeline (`march` + `cellBoundTrilinear`, `hddaCellEnd`, `nodeAverage`, `fastTrilinear` as the source asks) and generates one module. |
| G16a: a grid swap rebuilt five pipelines | `types` at creation; boxes and LOD range in uniforms; a test counts `create*Pipeline*` / `createShaderModule` calls across a swap (must be 0). |

Caches: the signature caches stay as today (V44's fields plus light ids). The temporal history is invalid after a
layout change, a camera `id` change or a skipped frame (`frameId` not previous + 1).

## 5. Shader composition

### 5.1 Text of one pipeline

```
vdbModuleWgsl({ slot: { group: 2, firstBinding: 0 }, types, profile: 'march', helpers })   // only for source passes
+ source.wgsl({ slot: { group: 2, firstBinding: 0 }, profile, channels })                   // vdbSrc* (exists for dense)
+ frameUniformsWgsl()                                                                       // group 0 structs + bindings
+ frameChunksWgsl(roots)                                                                    // closure of the vdbFrame* functions the entry calls
+ passWgsl(pass, features)                                                                  // group 1 bindings + entry
```

`frameChunksWgsl` works like the generator: each chunk declares its dependencies, the closure is emitted in a fixed
order, so the same features give the same bytes and Dawn's cache hits. Names: `vdbFrame*` / `VdbFrame*` functions
and structs, `vdbSrc*` for sources; both sit inside the reserved `vdb`/`Vdb` prefixes and never collide with the
generated module (a test runs bvh's `wgslNameProblems` port, the reserved-word list (`active`, `set`, `filter`,
`target`) and a duplicate-symbol scan over every composed pipeline). Back-ticks in WGSL comments are rejected
by the same test (Traps).

### 5.2 Compile-out (D10)

| Pipeline key (compile time) | Runtime (uniform) |
|---|---|
| lighting mode, spectral, MS family, `temporal` (its own pass), source kind + channels + LOD mode + `hdda` + grid types, emission on/off, composite format and `srgbBlending` | light counts and kinds, phase mode and parameters, MS numbers, octaves, jitter, cascade count, grid sizes, boxes, LOD range, densities, all volume params |

A first-frame 'frame' scene with one sparse source compiles about 11 pipelines (pyramid already warm: shadowDensity,
gridMips, transmittance, environmentMap, environmentGrid, density, lighting, temporal, composite, volumeShadow,
emission ×3 when used). A second volume with the same source layout reuses them (text-keyed; V30/V42's
one-material-per-source becomes one pipeline per source *layout*). `prepare()` creates every variant the settings
and current sources reach with `createRenderPipelineAsync`, so hosts can warm before the first frame, and toggling
`spectral` or `lighting` at runtime can be pre-warmed with `prepare({ spectral: [false, true] })`.

### 5.3 Budgets (D21)

Measured at K6.1 and committed as `core/frame/size-snapshot.json` (CI fails on growth not committed), with these
caps to start, in KiB of minified WGSL, module included:

| Pass class | Cap | Why |
|---|---|---|
| source-free (lighting, transmittance, env grid, temporal, composite, volume shadow, grid mips) | 24 (transmittance with neural 28) | the neural forward is ~3 KiB of code once the weights leave the text |
| source passes, `march` + source helpers, Float + Fp8 + FpN (inject, density, shadowDensity, emissionDirection) | 52 | `march+nodeAverage+fastTrilinear` is budgeted at 34 KiB (`size-budgets.json`) + `vdbSrc*` ≈ 3 + pass ≈ 10–14 |
| `levelSet` surface | `march+levelSet` budget + 4 | |

Compile time is measured per pass (Metal; D3D12/DXC on the owner's Windows machine when available) and recorded as
provisional (D25). The rule that pays most: as few source-sample call sites per entry as the pass allows (§3.3).

### 5.4 Composing with P1

P1 adds the `majorant` profile and the lean reader to the generator in parallel. The renderer does not use either
and takes only opt-in helpers, so default module bytes don't move under it (Traps). The one shared file is
`size-snapshot.json`; conflicts there are regenerated, never hand-merged. The renderer's own snapshot is separate.

## 6. The three adapter after K6

**What stays in `./three`** (estimate ≈ 2k lines, from ≈ 11.4k today: 9.8k in `three/frame`, 1.7k in the two sources):
- `VDBFrameVolume` (`Group` + proxy box), holding a `VDBFrameVolumeGpu`; props map onto `params`.
- `VDBFrameRenderer`: wraps one `VDBFrameRendererGpu` per three renderer (`getDevice(renderer)`); in the first
  proxy's `onBeforeRender` (as today) it collects volumes and casters (`FrameVolumeCollector`, three traversal and
  layers), extracts camera, lights (three light → `VDBFrameLight`, `shadow.camera` → `VDBVolumeShadowRequest`),
  environment (PMREM → equirect bake on change), creates an encoder, calls `update()`, and submits. three submits its
  own frame after that, so the queue order is correct; the pool's update was already submitted by `grids.update()`.
- The integrate proxy material: a `NodeMaterial` whose fragment is `wgslFn(vdbFrameIntegrate)` over three
  `viewportDepthTexture` / `viewportTexture`. Because it is an ordinary material in the scene pass, three's MRT
  and post-processing see one material, not a nested renderer (K7.2).
- Mesh shadows: the `filterNode` install and light reset (internals 1–2, self-tested, V60) sampling the core's
  maps through `vdbVolumeShadowWgsl` with the drawn view. Emission: the `AnalyticLightNode` reading the core's
  probes through `vdbEmissionProbeWgsl`.
- Output textures: the adapter is the `allocator`: it creates three `RenderTarget`s (which three allocates with
  `RENDER_ATTACHMENT | TEXTURE_BINDING`), `renderer.initRenderTarget`, and returns `getGPUTexture`; a self-test
  checks the usage flags (`VDB_ENGINE_MISMATCH` otherwise).
- `vdbSparseSource(grids, ...)` and `vdbDenseTextures` become thin wrappers over `vdbSparseSourceGpu` /
  `vdbDenseSource`; `vdbVolumeNodes` / `vdbVolumeUniforms` stay for `VDBVolumeMaterial` and custom TSL users.
- `VDBLevelSetMaterial`, `VDBPoints`, `VDBVolumeMaterial`, `VDBGridNode`, `VDBSequence` re-export.

**What changes for users (0.2.0; 0.x minors may break, no aliases, D32):**
- Unchanged: `VDBFrameVolume` props, `VDBFrameRenderer.settings` (names, defaults), `vdbFrameRenderer()`,
  `vdbSparseSource` / `vdbDenseTextures` signatures, `VDBFrameEmissionLight` props, `VDBLevelSetMaterial`,
  `VDBPoints`, `VDBSequence`.
- Removed: pass materials and their fields (`injectMaterial`, `densityMaterial`, `shadowDensityMaterial`,
  `emissionDirectionMaterial` on `VDBFrameVolume`; the TSL uniform fields typed `any` on `VDBFrameRenderer` and
  `VDBFrameEmissionLight`, the K5.5 follow-up); `VDBFrameVolumeShadows` / `VDBFrameVolumeShadowNode` exports
  (`three/internal.ts`).
- Changed: `emissionColor` takes `VDBEmissionColor` (data or LUT) instead of a TSL hook; `vdbFrameBlackbodyEmission`
  returns that data (Q9). `passHook` timings come from the core's timestamp writes, so `begin`/`end` bracket recorded
  passes, not three renders; `VDBFramePass` names stay. `renderFroxels` / `holdFroxels` (V34) map onto
  `update(..., { froxels: false })`.
- New: `VDBFrameRenderer.core` (the `VDBFrameRendererGpu`, for hosts mixing raw passes).

## 7. Verification plan

1. **CPU twins first (D11).** Each core pass gets a GPU-vs-twin probe in a raw-WebGPU harness
   (`examples/src/harness/k6-core.{html,ts}`, no three): render the pass alone into its target, read back N texels
   (N ≥ 512, fixed seeds), compare to the `tools/frame` twin within the f16 bounds of today's `k3-frame-lighting`
   spec. Twins missing today: the shadow-density voxel (source sample at grid lod → extinction/absorption), an
   end-to-end inject froxel (composes existing twins), the point sprite size. Each new twin gets a fail-before
   mutation, as the K3.6 specs do.
2. **Pixel A/B, TSL vs core, while both exist.** During K6.1–K6.7 `VDBFrameRenderer` gets a temporary setting
   `backend: 'tsl' | 'core'` (internal, not exported, removed in K6.8). The harness renders the same scene both ways
   on one device and compares every intermediate target texel by texel (bar: ≤ 2 f16 ulp on ≥ 99.9 % of texels; the
   rest explained by fma/ordering) and the final image (max |Δ| ≤ 1/255, PSNR ≥ 60 dB), on the K3 scene list of
   that step (Step 1 for K6.1 … Step 7 for K6.6), dense and sparse.
3. **Gates.** `pnpm k3:parity` against his captures at `75e7345` through the core-backed adapter: 48/48 strict
   (46 bit-identical today; any newly non-identical scene needs an explanation in the step handoff), Step 7 IoU
   ≥ 0.99 and 9,802 points. Demo goldens (02, 04–09 snapshots) unchanged; demos 10–16 structural specs unchanged;
   demo 16's zero-build swap still 0; `test/boundaries.test.ts`, `api-surface.test.ts` (with the reviewed diff).
4. **Which specs move.** `k3-frame*.spec.ts` keep testing through `./three` (they become adapter tests) and gain
   core twins in `e2e/k6-core.spec.ts`; `k3-pyramid` and `core-parity` stay; `k3-levelset` gains a raw-pass variant;
   the timing harness (`k3-timing`, `time-hardware.mjs`) switches to core timestamp writes in K6.8; new
   `e2e/demo-17.spec.ts` (K6.9 raw demo, readback presentation, golden on SwiftShader).
5. **Limits.** A unit test builds every pipeline descriptor for the key matrix and checks the §1 limits; WGSL
   validity of every variant runs in `pnpm wgsl:variants` (extended), under `pushErrorScope('validation')`.

## 8. Migration order, sizes, risks

| Task | Content | Size | Depends on | Risk |
|---|---|---|---|---|
| K6.1 | Renderer skeleton (options, settings, lifecycle, retire queue, pipeline cache, group 0 packer, allocator hook, composite + `vdbFrameIntegrate`), froxel layout/view maths to core, inject, temporal; dense source; A/B harness; `backend` setting | L | K6.0 approval | Skeleton churn: every later step builds on it. Mitigation: K6.1 also runs the Q2 spike. |
| K6.7 | `vdbSparseSourceGpu` (K3.2b): `vdbSrc*` over the pool, pyramid, `followImage`, `skipEmpty` | M | K6.1 group 2 contract | Low: the WGSL exists inside the TSL source's `wgslFn`s. Runs **in parallel** with K6.1 after the source contract lands. |
| K6.2 | shadow density, cascades/signatures, transmittance, environment map + readback, env grid, density, lighting | L | K6.1, K6.7 | Env bake (Q3) parity. |
| K6.3 | phase LUTs, MS modes, neural in a uniform buffer, grid mips; wire K7.3 multi-scale | M | K6.2, **K7.3** | Neural ulps (Q5). |
| K6.4 | spectral: pipeline key, array targets | M | K6.3 | 32 B/sample and 16-texture limits. |
| K6.5 | volume shadow maps and emission passes in core; drawn-view receivers; the three filterNode / AnalyticLightNode over them | M core + L adapter | K6.2 | Highest: three internals (V60), budget behaviour. |
| K6.6 | level-set surface wrapper + raw pass; point sprite WGSL + raw pass | S–M | K6.1 | Low. |
| K6.8 | Adapter thin: delete TSL passes, `backend` setting gone, API changes of §6, timing harness | L | K6.1–K6.7 | Parity drift found late; mitigated by per-step A/B. |
| K6.9 | Raw-WebGPU demo 17 (WDAS, 'frame', mesh shadows on a ground, emission from the explosion) + docs "Rendering without three" | M | K6.5 | — |
| K6.10 | Gate review (separate agent) | S | all | — |

**Parallel tracks.** K7.1 and K7.2 fix the shipping TSL path first (users have it now); their repro tests become
K6.5 / K6.8 tests and the core design makes them structural. K7.3 writes `'multi-scale'` as core compute next to
`core/compute/pyramid.ts`, reading the shadow grid as a `GPUTexture` with the atlas layout of
`tools/frame/grid.ts` and writing textures from an allocator or `targets`; the TSL path binds them through
`getGPUTexture` until K6.3 consumes them natively. P1 touches the generator and its snapshot only (§5.4). K6.6 can
run any time after K6.1.

## 9. Open questions for the owner

**Q1. Fragment passes on 2D atlases, or compute on 3D textures?** *Suggest:* fragment/atlas for K6; a compute
graph later behind a setting, A/B'd. Pro: texel-identical A/B against the TSL path, the same additive blending,
the 48/48 bar is reachable step by step. Con: keeps his WebGL-era layout (manual z lerp, tile edges) and more
render passes. *Settles it:* the K6.1 A/B shows ≤ 2 f16 ulp on the froxel atlas; then a quiet-machine timing A/B
(D25) of a compute prototype of the density + lighting passes on WDAS.

**Q2. Core runs the passes on three's device (own encoder, submitted from the first proxy's `onBeforeRender`), or
three keeps orchestrating and only the shading becomes `wgslFn(coreWgsl)`?** *Suggest:* core runs them. Pro: one
orchestration for every host, the K7.2 class gone, the adapter shrinks to ≈ 2k lines. Con: a submit inside three's
render call, output textures need the allocator hook (three internals for usage flags), timing moves off three's
timestamps. *Settles it:* a K6.1 spike: demo 10 in 'volume' lighting through the core under a three `pass()` with
MRT (GTAO) renders with no validation errors and equals the TSL path ≤ 1/255.

**Q3. Environment input: baked equirect textures, or SH only?** *Suggest:* the host gives equirect radiance (and
optionally irradiance); `./three` bakes PMREM roughness-1 radiance × π into a 128 × 64 rgba16float equirect on
environment change. Pro: one input any engine has; no CUBE_UV decoding in the core. Con: one bilinear resample of
a very smooth function; a bake pass per environment change. *Settles it:* Step 3 `*-env-isolated` scenes through
the bake vs today's PMREM path: PSNR ≥ 50 dB and the strict parity bar still met.

**Q4. Core names: `VDBFrameRendererGpu` / `vdbCreateFrameRenderer` / `vdbSparseSourceGpu`, or new nouns
(`VDBFroxelRenderer`)?** *Suggest:* the `Gpu` suffix (D32's rule, `VDBBoxPyramidGpu` precedent). Pro: one concept,
one name, search finds both. Con: `Gpu` on a class reads oddly. *Settles it:* the api-surface naming test plus a
docs read-through of "Rendering without three".

**Q5. Neural weights in a uniform buffer instead of shader constants?** *Suggest:* yes. Pro: ~7 KiB less WGSL per
transmittance variant, a swapped network needs no rebuild, constants were the slow-compile pattern on DXC (R1 §8).
Con: values may move by f32 ulps (constant folding vs loads). *Settles it:* `vdbFrameNeuralLight` twin parity and
Step 4 strict parity unchanged; transmittance compile ms before/after (provisional).

**Q6. Accept the `./three` breaks of §6 in 0.2.0?** *Suggest:* yes, with CHANGELOG migration notes, no aliases
(D32). Pro: removes `any`-typed internals from the public d.ts; the user-facing props stay. Con: anyone who
touched pass materials or the TSL `emissionColor` hook must change code. *Settles it:* the reviewed
`api-surface/three.txt` diff and demos 10–16 compiling unchanged (they use only the user-facing API).

**Q7. Move `VDBSequence` to the core?** *Suggest:* yes, in K6.8 (one declaration, re-exported by `./three`). Pro:
raw hosts and the cloud system get sequences, ping-pong and prefetch. Con: export churn in two snapshots.
*Settles it:* `sequence.test.ts` and demos 05/12 unchanged; `boundaries.test.ts` green.

**Q8. Light terms as loops over uniform counts, or compiled-in light sets (V55)?** *Suggest:* loops. Pro: no
rebuild when a light or environment appears; fewer variants. Con: possible cost of empty loops in the hottest pass.
*Settles it:* inject and density pass ms on the Apple GPU with 0 vs 4 hemisphere/env lights, loops vs compiled-in
(provisional, D25); switch to a light-kind key only if loops cost > 5 %.

**Q9. Emission colour as data (named spectrum or a 256-texel LUT) instead of a TSL hook?** *Suggest:* yes. Pro:
engine-free; the parity harness still injects the exact CIE table as a LUT (D33). Con: arbitrary per-voxel colour
functions are no longer possible without a custom pass. *Settles it:* Step 6 emission scenes (`explosion-emission*`)
strict parity with the LUT path, and the LUT twin within 1/1024 of `vdbFrameBlackbodyEmission`.

**Q10. Meshes shadowing volumes (the host's depth shadow map stopping the grid and inject marches)?** *Suggest:*
not in K6; a follow-up after K6.5 behind a setting, default off. Pro: K6 stays a port with a parity gate; the
interface already has the field. Con: hosts with a solid occluder between sun and cloud keep an unshadowed cloud.
*Settles it:* a test scene (a box between the sun and the bunny cloud) darkens the cloud behind the box only, and
the 48 parity scenes are bit-unchanged with the setting off.

## 10. DECISIONS entry draft (for the owner; not in DECISIONS.md)

| ID | Date | Decision | Why |
|---|---|---|---|
| D34 | 2026-11-?? | **Proposed: the K6 core renderer** ([docs/dev/design/K6-core-renderer.md](./K6-core-renderer.md)). The froxel renderer, volume shadow maps, emission probes and the level-set/point surface functions run in `core` as WGSL fragment passes on the K3 atlases and formats, recorded into the host's encoder on the host's device, with camera, lights, environment and shadow cameras as plain data and outputs in core- or host-allocated textures. Composite is either a core pass or the WGSL function `vdbFrameIntegrate` the host places (the MRT seam); volume shadows are read with the view they were drawn with. `./three` becomes an adapter: it collects scene data, runs `update()` on its own encoder before three's frame, composites through a material, and installs shadow filters and the emission light node over core outputs. Core names take `Gpu` where `./three` has the plain name. Each pass ports with a CPU twin and a texel A/B against the TSL path; K3 parity 48/48 and demo goldens gate K6.8. | D8 and D18 intended shared foundations in `core`; the cloud system and other adapters (R6) need the renderer without three; his post-0.26 fixes (62a6e3e, c13f2d5, 720c38f) and our K5.9/RV bugs all came from engine coupling (shadow coordinates, light collection, MRT inheritance, uniform callbacks, cached TSL layouts, immediate texture destruction), which plain-data inputs, explicit bind groups and a retire queue remove. Fragment passes on the same atlases keep the port checkable texel by texel. |
