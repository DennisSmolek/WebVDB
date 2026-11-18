# @pmndrs/vdb

Sparse NanoVDB volumes on WebGPU, from the [WebVDB](../../README.md) repo. One package, three subpaths:

| Import | What it is | Needs a GPU / three? |
|---|---|---|
| `@pmndrs/vdb` | the engine-free core: the vendored `pnanovdb.wgsl` module | no engine |
| `@pmndrs/vdb/tools` | CPU side: `.nvdb` loader, CPU reads (`vdbReadValueCpu`, `vdbSampleTrilinear`), `.vdb` parser, NanoVDB builder, quantizer, writer | no |
| `@pmndrs/vdb/three` | the three.js / TSL adapter: `VDBGridNode` (a pool in three's node system), `VDBVolumeMaterial`, `VDBSequence`, compute utilities | `three` (optional peer, `>=0.187.0 <0.188.0`) |
| `@pmndrs/vdb/pnanovdb.wgsl` | the raw WGSL text (use `?raw` in Vite) | no |

```sh
npm install @pmndrs/vdb          # core + tools
npm install @pmndrs/vdb three    # to use @pmndrs/vdb/three
```

Boundaries: `core` and `tools` never import `three`; `tools` never imports `core` (enforced by
`test/boundaries.test.ts`).

## `@pmndrs/vdb` (core) and the raw WGSL module

Renderer-agnostic NanoVDB traversal/sampling **WGSL module** + a
TypeScript **`.nvdb` file loader**. This is the low-level layer of
[WebVDB](../../README.md): it doesn't know about three.js or any
particular renderer — just how to get NanoVDB grid bytes off disk/network
and how to walk them, in WGSL or in plain TS.

### What's in the box

- **`vendor/pnanovdb.wgsl`** — a WGSL module with the NanoVDB root ->
  upper -> lower -> leaf tree traversal, HDDA ray marching, and
  Float/Fp4/Fp8/Fp16/FpN value decoders. Import it as raw text (`?raw` in Vite,
  or read the file yourself) and splice it into your own shader.
- **`NvdbFile`** (`@pmndrs/vdb/tools`) — parses a `.nvdb` file (header + per-grid metadata +
  grid images), handling the NONE/ZIP codecs.
- **A pure-TypeScript CPU reference** (`@pmndrs/vdb/tools`: `vdbReadValueCpu`, `vdbReadValueF32`,
  `vdbSampleTrilinear`) — the exact same tree descent as the WGSL
  module, so you can sanity-check GPU results or read voxels with no GPU
  at all (Node, workers, tests).

### Quickstart: load a file and read a voxel

```ts
import { NvdbFile, vdbReadValueCpu } from "@pmndrs/vdb/tools";

const file = await NvdbFile.fromURL("/fixtures/sphere_fog_float.nvdb");
const gridImage = file.gridImage(0); // flat Uint32Array — the grid, as-is on disk
const meta = file.grids[0]!;
console.log(meta.name, meta.gridType, meta.voxelCount);

const { value, active } = vdbReadValueCpu(gridImage, [12, 4, -3]);
```

`gridImage` is a zero-copy view into the file's buffer when possible
(`file.isZeroCopy(0)` tells you which) — upload it straight into a GPU
`storage` buffer, no reshaping needed.

### Loading the raw WGSL module

```ts
import pnanovdbSource from "@pmndrs/vdb/pnanovdb.wgsl?raw"; // Vite/bundler `?raw` import
// or, outside a bundler: fetch(import.meta.resolve("@pmndrs/vdb/pnanovdb.wgsl"))
```

`pnanovdbSource` is the library's text, unmodified. It's a genuine C-style
library, not a drop-in snippet: its reads all bottom out in one function,
`pnanovdb_buf_read_uint32`, that expects the *consumer* to declare a
module-scope `var<storage, read> nanovdb_buffer: array<u32>` binding (see
`vendor/VENDOR.md` for the exact contract) — plus `pnanovdb_buf_t`/address
plumbing and (for point-sampling) an accessor struct. You rarely need it
directly: the core's generator (`vdbModuleWgsl`) builds a pool-aware `vdb*`
module from it. Integrations to copy from instead of re-deriving the binding:

- **`examples/src/demos/08-webgpu-smoke`**: raw WebGPU on the core
  (`VDBGridPool` + `vdbModuleWgsl`), no engine.
- **`@pmndrs/vdb/three`** ([`src/three`](./src/three)): `VDBGridNode` binds
  the pool through three's node system (below).
- **`examples/src/demos/03-gpu-read`** is a from-scratch, heavily-commented
  raw-WebGPU walkthrough (no three.js) that hand-derives a FLOAT-only
  traversal function equivalent to this library's, teaching the byte
  layout end to end.

### Supported grid types / codecs

| Grid type | Read (WGSL + CPU) | Notes |
|---|---|---|
| `Float` | yes | lossless, 4 bytes/voxel |
| `Fp8` | yes | per-leaf min/quantum, 1 byte/voxel |
| `FpN` | yes | per-leaf variable bit-width (oracle-matched to native) |
| other `GridType`s | rejected | `NvdbFile` throws a clear "unsupported grid type" error — v1 targets FogVolume rendering (SPEC §2.1) |

| File codec | Support |
|---|---|
| `NONE` | yes |
| `ZIP` (zlib, via `fflate`) | read and write (`writeNvdb(images, { codec: 'zip' })`) |
| `BLOSC` | not supported — throws (see [D3](../../docs/DECISIONS.md#d3--cpu-half-pure-typescript-first-wasm-as-targeted-escalation)) |

### The vendoring story

The WGSL module is not written from scratch: per
[decision D2](../../docs/DECISIONS.md#d2--gpu-traversal-base), it's
**vendored as a pinned, audited fork** of
[emcfarlane/webgpu-nanovdb](https://github.com/emcfarlane/webgpu-nanovdb)'s
`pnanovdb.wgsl` — Apache-2.0, NOTICE preserved, every local change logged
in-tree. We're prepared to diverge permanently and happy to upstream
fixes if the author engages. Full pin/hash/diff-log details:
[`vendor/VENDOR.md`](./vendor/VENDOR.md).


## `@pmndrs/vdb/tools`

The CPU half of [WebVDB](../../README.md): parse `.vdb` containers, build
NanoVDB grid images (from a dense array or straight from a parsed `.vdb`),
quantize to Fp4/Fp8/Fp16/FpN, apply affine transforms, inspect memory layout, and
write `.nvdb` files — all **pure TypeScript, zero runtime dependencies,
zero WASM** ([decision D3](../../docs/DECISIONS.md#d3--cpu-half-pure-typescript-first-wasm-as-targeted-escalation)).
WASM stays an optional escalation rung, adopted only on demonstrated need
— nothing in this package requires it.

### Quickstart: the demo 07 pipeline

The full authoring round-trip — generate/obtain voxels, build a NanoVDB
grid, optionally quantize, write a real `.nvdb` file — trimmed from
[`examples/src/demos/07-builder`](../../examples/src/demos/07-builder):

```ts
import { vdbBuildFromDenseDetailed, vdbQuantize, writeNvdb } from "@pmndrs/vdb/tools";

const dim = 32;
const values = new Float32Array(dim * dim * dim); // fill with your density field
// ... write nonzero values into `values` at your active voxels ...

const built = vdbBuildFromDenseDetailed(values, [dim, dim, dim], {
  gridName: "my_grid",
  background: 0,
});
// built: { image: Uint32Array, voxelCount, indexBBox, worldBBox, voxelSize, nodeCounts }

const fp8Image = vdbQuantize(built.image, "fp8"); // or "fpn"; per-leaf min/quantum, native-exact rounding

const fileBuffer = writeNvdb([fp8Image]); // ArrayBuffer — a real NanoVDB2 file, ready to save/upload
```

`fileBuffer` round-trips through `NvdbFile.fromArrayBuffer`
byte-for-byte (that loader is this subpath's read path, see above).

### Quickstart: parse a real `.vdb` and build NanoVDB from it

The drop-in-a-file pipeline, trimmed from
[`examples/src/demos/06-explorer`](../../examples/src/demos/06-explorer):

```ts
import { parseVdb, buildFromVdbDetailed, vdbInspect } from "@pmndrs/vdb/tools";

const buffer = await fetch("/fixtures/vdb-samples/utahteapot.vdb").then((r) => r.arrayBuffer());
const vdbFile = parseVdb(buffer); // { fileVersion, grids: VdbGrid[] }

const built = buildFromVdbDetailed(vdbFile.grids[0]!); // streams leaves, no dense allocation
const report = vdbInspect(built.image);
console.log(report.nodeCounts, report.memoryBreakdown);
```

`buildFromVdb`/`buildFromVdbDetailed` stream the parser's leaves straight
into the NanoVDB serializer (proven on a 7M-voxel teapot with no dense
array), carrying the `.vdb` file's uniform scale+translate transform into
the NanoVDB `Map`.

### API surface

| Function | What it does |
|---|---|
| `parseVdb(buffer)` | Reads a `.vdb` container: FloatGrid, 5-4-3 tree, uncompressed or zlib. Throws a clear error for blosc. |
| `vdbBuildFromDense` / `vdbBuildFromDenseDetailed` | Dense `Float32Array` -> NanoVDB FLOAT grid image |
| `vdbBuildFromLeavesDetailed` | Lower-level: build from an already-leaf-shaped source (what `buildFromVdb` and `vdbQuantize` are built on) |
| `buildFromVdb` / `buildFromVdbDetailed` | Parsed `.vdb` grid -> NanoVDB FLOAT grid image, leaf-streamed |
| `vdbQuantize` / `vdbQuantizeDetailed` | FLOAT grid image -> Fp4/Fp8/Fp16/FpN (per-leaf min/quantum; FpN bit-width picked per leaf) |
| `vdbTransform` | Metadata-only affine edit (uniform scale + translate) on a copy; throws with a GPU-resample pointer for rotation/shear ([D6](../../docs/DECISIONS.md#d6--browser-first-companion-service-is-an-interim-crutch-to-eliminate)) |
| `vdbInspect` | Node counts + memory-by-section breakdown, summing to the grid's total byte size |
| `writeNvdb` | Serializes one or more grid images into a `.nvdb` file `ArrayBuffer` (NanoVDB2, `io::Segment` layout; `{ codec: 'zip' }` for NanoVDB's ZIP codec) |
| `vdbBakeCoarseFirst` | Coarse-first `.nvdb`: a small preview grid (`<name>.lod48`, box, FpN) then the full grid, ZIP by default |
| `NvdbStreamReader` / `readNvdbStream` | Reads a `.nvdb` from pushed bytes or a stream, emitting each grid as soon as its data is complete; hostile-input checks before allocation |
| `NvdbFile` | Reads a `.nvdb` held in memory or fetched (`fromArrayBuffer` / `fromURL`); there is no separate `readNvdb` |

### Supported `.vdb` subset

| | Support |
|---|---|
| Grid type | `FloatGrid` |
| Tree configuration | 5-4-3 (OpenVDB's default) |
| Compression | none, zlib |
| Compression | blosc — throws a clear "not supported" error |
| Half-float leaf storage | yes |
| Transform | uniform scale + translate (affine `Map`); non-uniform/rotated throws on `vdbTransform()`, parses fine on `parseVdb()`/`buildFromVdb()` |

### Native parity

Every function above is validated byte/value-wise against real
`nanovdb_convert`/OpenVDB output on the project's fixture corpus (four
openvdb.org samples + primitives). The one anchor that needs a native
machine to close out (this package's numbers already match at every
tested coordinate) is tracked in
[`docs/handoffs/PHASE-5.md`](../../docs/handoffs/PHASE-5.md#native-machine-checklist-the-deferred-anchor)
— see that doc for the full parity checklist and known semantic edge
cases (root background vs. near-field tile fill, active-tile compression,
CRC).


## `@pmndrs/vdb/three`

The three.js adapter over the engine-free core (D8), on three r187 (D23). The
grids live in a `VDBGridPool` on the renderer's `GPUDevice`; `VDBGridNode`
puts that pool into three's node system without copying it (three binds the
pool's own buffer); `VDBVolumeMaterial` raymarches one pooled grid, and
`VDBSequence` plays frames into one pool id.

### Quickstart: render a cloud

This mirrors [`examples/src/demos/02-cloud`](../../examples/src/demos/02-cloud),
trimmed to the essentials (no camera controls, no live uniforms):

```ts
import * as THREE from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { NvdbFile } from "@pmndrs/vdb/tools";
import { VDBGridNode, VDBVolumeMaterial, getDevice, vdbProxyGeometry } from "@pmndrs/vdb/three";

// The host owns the device: the pool's limits, passed to the renderer so the
// pool outlives renderer.dispose() (a renderer that made its own device destroys it).
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter!.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter!) });
const renderer = new THREE.WebGPURenderer({ device, canvas });
await renderer.init();

const file = await NvdbFile.fromURL("/fixtures/wdas_cloud_quarter_fp8.nvdb");
const pool = await VDBGridPool.create(getDevice(renderer));
const id = pool.add(file.gridImage(0));
pool.flush(); // writes the grid (or record pool.update(encoder) into your own encoder)

// One VDBGridNode per pool, renderer and shader (one WGSL module per shader).
// The material needs the support-box ray clip helper.
const grids = new VDBGridNode(renderer, pool, { helpers: ["rayClipSupport"] });
const material = new VDBVolumeMaterial(renderer, grids, id, { densityScale: 40 });
const scene = new THREE.Scene();
scene.add(new THREE.Mesh(vdbProxyGeometry(pool.info(id)), material));

// Every frame: the pool's update first (submitted), then the node, then three.
pool.flush();
grids.update(); // rebinds only when a growth replaced pool.buffer
await renderer.renderAsync(scene, camera);
```

- **`VDBGridNode(renderer, pool, { types?, profile?, sampler?, isActive?, outside?, helpers?, name? })`**:
  the module options minus the binding (three assigns it). `types` defaults to
  the pool's types now; pass every type you will add (or
  `VDB_ALL_SCALAR_TYPES`) when the pool fills later. TSL helpers:
  `vdbSample(grids, id, position, { space: 'world' | 'index' })` and
  `vdbReadValue(grids, id, ijk)`; or call the WGSL API from your own
  `wgslFn(src, [grids.module])` (demo 01 does both). A node binds through the
  renderer it was made with: a shader another renderer builds, or a shader
  holding a second node's module, throws `VDB_BAD_OPTION` (make one node per
  renderer; read every grid of a shader through one node, by id). The first
  node on a renderer self-tests three's attribute internals before the pool's
  buffer reaches three (`VDB_ENGINE_MISMATCH` if three would destroy it); a
  three other than r187 warns once, and so does a renderer that made its own
  device (its `dispose()` destroys the pool's buffer).
- **`VDBVolumeMaterial(renderer, grids, gridId, params)`**: needs a node with
  profile `sample` or `march` and `helpers: ['rayClipSupport']`. The march is
  one `wgslFn` over the public API: world ray to index space through the pool
  record (`pool.setTransform` instances work), clipped to the trilinear
  support box, fixed-step density with `vdbSample`, Henyey-Greenstein sun with
  a shadow march, ambient, and a per-fragment sample budget. Every tunable is
  a live uniform (`material.densityScale.value = 45`, `material.stepSize.value`,
  `maxSteps`, `sunDirection`, `sunColor`, `sunIntensity`, `anisotropy`,
  `shadowSteps`, `shadowDensity`, `ambient`, `jitter`); `gridId` is a `uint`
  uniform when given as a number, so `material.gridId.value = other` switches
  grids without a rebuild (give the mesh a new `vdbProxyGeometry` for the
  other grid, and after `pool.setTransform`). Caps (`maxStepsCap`,
  `shadowStepsCap`, `sampleBudgetCap`) are compiled in. The march runs in index
  space: `stepSize` is in index voxels and opacity integrates per index voxel,
  so scaling a grid with `setTransform` changes its size on screen, not its
  opacity. Perspective cameras only. `grids` must come from the same renderer.
- **`vdbProxyGeometry(pool.info(id), { pad? })`**: the world box to put the
  material on (add the mesh with an identity transform). `pad: 1` also covers
  the min-side half voxel of the trilinear support.
- **`getDevice(renderer)`, `getGPUTexture(renderer, texture)`**: the raw
  WebGPU handles, as in `@pmndrs/upscaler` and `@pmndrs/denoiser`.
- Gone since v1: `NanoVDBGrid` (use `pool.info(id)` and `vdbProxyGeometry`),
  `NanoVDBVolumeMaterial` (use `VDBVolumeMaterial`), `NanoVDBSequence` (use
  `VDBSequence`), `assembleVolumeWgsl`/`rewriteBufferGlobal`, and the
  `createVolumeRenderer` bootstrap with `computeRequiredLimits`/`nextPow2`
  (create the device yourself with `VDBGridPool.requiredLimits(adapter)`, as
  in the quickstart).

### Compute utilities

Compute passes over a grid in a `VDBGridPool` (K2.3). The engine-free core
records into your encoder and hands back a readback you resolve after
submitting; the three wrappers take the renderer and return promises:

```ts
import { VDBGridPool, vdbGridStatsGpu, vdbDecodeToAtlasGpu, vdbApplyValueTransform } from "@pmndrs/vdb";

const id = pool.add(image);
pool.update(encoder); // or pool.flush()
const stats = vdbGridStatsGpu(device, pool, id, { encoder, histogramBins: 256 });
const atlas = vdbDecodeToAtlasGpu(device, pool, id, { encoder, maxDim: 128, format: "uint8" });
device.queue.submit([encoder.finish()]);
// With your encoder, say it was submitted before reading (read() earlier rejects with VDB_BAD_OPTION):
await stats.submitted().read();  // { min, max, mean, histogram, activeVoxelCount, usedAtomicFastPath, range }
await atlas.submitted().read();  // { data: Uint8Array | Float32Array, dims, range }; atlas.buffer is the raw f32 GPU output

// Per-voxel edits run on the CPU (the pool is read-only to shaders) and are staged with pool.replace:
vdbApplyValueTransform(pool, id, image, "double"); // or (v) => v * 2; Float grids only
```

```ts
import { vdbGridStats, vdbDecodeToAtlas } from "@pmndrs/vdb/three";

const stats = await vdbGridStats(renderer, pool, id);
const { data, dims, range, texture } = await vdbDecodeToAtlas(renderer, pool, id, { maxDim: 128 }); // texture: Data3DTexture
```

The three wrappers submit their own encoder. The value transform has no three
wrapper: call the core `vdbApplyValueTransform`.

Each tool frees only the buffers it allocated (in `read()` or `dispose()`).
`vdbGridStatsGpu` bins over the grid's root min/max statistics (a grid with a
negative root min takes the dump path, 8 B per bbox voxel, and is refused up
front when that exceeds the device's buffer limits); `vdbDecodeToAtlasGpu`
normalizes to them by default. Atlas cell `i` of an axis with `n` cells over `[min, max]` samples index position
`min + (i + 0.5) * (max - min + 1) / n - 0.5`, so at full resolution cell `i` is
voxel `min + i` (the last cell is `max`); the texture spans the voxel-centred box
`[min - 0.5, max + 0.5]`. `vdbApplyValueTransform` rewrites leaf values
(not tiles) and does not recompute the image's statistics. CPU versions live in
`@pmndrs/vdb/tools` (`vdbGridStatsCpu`, `vdbDecodeToAtlasCpu`). See
[`examples/src/demos/04-atlas-fallback`](../../examples/src/demos/04-atlas-fallback)
for the atlas fallback path through a stock `Data3DTexture`.

### Coarse-first loading: `vdbStreamCoarseFirst`

A file baked with `vdbBakeCoarseFirst` (`@pmndrs/vdb/tools`) holds a ~12 KiB preview grid before the full grid
(docs/design/K4-5-coarse-first.md). `vdbStreamCoarseFirst` reads the response body as it arrives, adds the
preview to the pool with the full grid's size reserved, and replaces it in place when the full grid is in:

```ts
import { vdbStreamCoarseFirst } from "@pmndrs/vdb/three";

const { id } = await vdbStreamCoarseFirst(pool, await fetch("/smoke.coarse.nvdb"), {
  // After each swap: densityScale = base x voxelRatio (the material marches in index voxels), the proxy box
  // from pool.info(id), pool.flush() + grids.update(), and `majorant(event)` re-applied (replace drops it).
  bind: { material, mesh, grids },
  onGrid: (e) => console.log(e.final ? "full" : "preview"),
});
```

Every `bind` member is optional. `vdbCoarseFirstApply(pool, event, bind)` is the same work for an async `onGrid`.
Call `pool.update(encoder)` / `pool.flush()` and `grids.update()` in your frame loop as usual.

### Sequences: `VDBSequence`

`VDBSequence` plays per-frame `.nvdb` URLs into ONE pool grid with
`pool.replace(id, frame)`: no material rebuild, no shader recompile, and with
the capacity reserved, no buffer move either:

```ts
import { VDBSequence } from "@pmndrs/vdb/three";

// Reserve room for the largest frame: every replace then writes in place
// (no pool.version change, no VDBGridNode rebind, no new bind group).
const id = pool.add(firstFrame, { capacity: largestFrameBytes });
const material = new VDBVolumeMaterial(renderer, grids, id);

const sequence = new VDBSequence({
  urls: (i) => `/fixtures/embergen/pack/frame_${i}.nvdb`,
  frameCount: 48,
  fps: 24,
});
await sequence.preload();
sequence.start({ pool, id });
// each animation frame:
sequence.update(performance.now()); // may pool.replace(id, frame)
pool.flush();                       // or pool.update(encoder) in your encoder, submitted before three renders
grids.update();
await renderer.renderAsync(scene, camera);
```

A frame larger than the reserved capacity still plays: the pool moves it and
may grow, which is why `grids.update()` runs after the pool's update every
frame. The sequence never blocks the render loop: if a frame isn't decoded
when its time comes, it holds the last frame, counts a stall, and keeps going
(`sequence.stats`: `frame`, `decodedAhead`, `stalls`, `swaps`, `replaceMs`,
`failed`). A frame whose loader rejects fires `onError` once and is not
retried until `seek()` or `start()`; playback skips it. After `dispose()`,
late decodes are dropped and `start`/`seek`/`resume`/`preload` throw
`VDB_DISPOSED`.

Frame arrays are held by reference: the pool keeps an image until the update
that writes it, and until `pool.settled()` resolves while a growth is
unconfirmed. The player never writes into them; a custom `loader` that
recycles frame buffers must not reuse one before `await pool.settled()` after
the update that wrote it.

### Device ownership (D8)

The host creates the `GPUDevice` (with `VDBGridPool.requiredLimits(adapter)`)
and passes it to `WebGPURenderer({ device })`; the pool and every other pmndrs
tool take `getDevice(renderer)`. A renderer built without `{ device }` owns
its device and destroys it, with the pool's buffer, in `renderer.dispose()`.

## Third-party code

This package is MIT (see [LICENSE](./LICENSE)). The vendored `pnanovdb.wgsl` and `PNanoVDB.h` are Apache-2.0.
The published package ships `vendor/NOTICE` and `vendor/LICENSE` (Apache-2.0) alongside the vendored WGSL, and
they must stay in the package and any redistribution. The `.wvdb` reader and writer port logic from
mjurczyk/openvdb (MIT). Test fixtures are CC0, and `fflate` is an MIT runtime dependency. The full list, with
licences and file locations, is in [THIRD_PARTY.md](https://github.com/pmndrs/vdb/blob/main/THIRD_PARTY.md).

## Packaging note (for contributors)

The published export map points only at built `dist/` output (`dist/{core,tools,three}`, one `tsc` build).
Workspace Vite/Vitest/TypeScript configs alias `@pmndrs/vdb`, `@pmndrs/vdb/tools`, `@pmndrs/vdb/three` and
`@pmndrs/vdb/pnanovdb.wgsl` to `src/` for source-level development without exposing TypeScript source paths
to npm consumers.

Each subpath's `index.ts` is its public API, frozen by `test/api-surface.test.ts` (an export-list snapshot
and the D32 naming rule). Test oracles and other internals live in `src/{core,tools,three}/internal.ts`, which
re-exports the public API and is imported by path from tests, harnesses and scripts; it is never a package
export.
