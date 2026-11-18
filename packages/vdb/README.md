# @pmndrs/vdb

Sparse NanoVDB volumes on WebGPU, from the [WebVDB](../../README.md) repo. One package, three subpaths:

| Import | What it is | Needs a GPU / three? |
|---|---|---|
| `@pmndrs/vdb` | the engine-free core: the vendored `pnanovdb.wgsl` module and generated constants | no engine |
| `@pmndrs/vdb/tools` | CPU side: `.nvdb` loader, CPU oracle (`readValue`, `sampleTrilinear`), `.vdb` parser, NanoVDB builder, quantizer, writer | no |
| `@pmndrs/vdb/three` | the three.js / TSL adapter: grid wrapper, volume material, compute utilities, sequences | `three` (optional peer, `>=0.178.0`) |
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
  Float/Fp8/FpN value decoders. Import it as raw text (`?raw` in Vite,
  or read the file yourself) and splice it into your own shader.
- **`NanoVDBFile`** (`@pmndrs/vdb/tools`) — parses a `.nvdb` file (header + per-grid metadata +
  grid images), handling the NONE/ZIP codecs.
- **A pure-TypeScript CPU reference** (`@pmndrs/vdb/tools`: `readValue`, `sampleTrilinear`,
  `probeCoords`/`probePoints`) — the exact same tree descent as the WGSL
  module, so you can sanity-check GPU results or read voxels with no GPU
  at all (Node, workers, tests).

### Quickstart: load a file and read a voxel

```ts
import { NanoVDBFile, readValue } from "@pmndrs/vdb/tools";

const file = await NanoVDBFile.fromURL("/fixtures/sphere_fog_float.nvdb");
const gridImage = file.gridImage(0); // flat Uint32Array — the grid, as-is on disk
const meta = file.grids[0]!;
console.log(meta.name, meta.gridType, meta.voxelCount);

const { value, active } = readValue(gridImage, [12, 4, -3]);
```

`gridImage` is a zero-copy view into the file's buffer when possible
(`file.isZeroCopy(0)` tells you which) — upload it straight into a GPU
`storage` buffer, no reshaping needed.

### Loading the raw WGSL module

```ts
import pnanovdbSource from "@pmndrs/vdb/pnanovdb.wgsl?raw"; // Vite/bundler `?raw` import
// or: fetch(new URL("@pmndrs/vdb/pnanovdb.wgsl", import.meta.url)) — see the
// package's `pnanovdbWgslUrl` export for a resolvable URL in non-bundler runtimes
```

`pnanovdbSource` is the library's text, unmodified. It's a genuine C-style
library, not a drop-in snippet: its reads all bottom out in one function,
`pnanovdb_buf_read_uint32`, that expects the *consumer* to declare a
module-scope `var<storage, read> nanovdb_buffer: array<u32>` binding (see
`vendor/VENDOR.md` and the header comment of `@pmndrs/vdb/three`'s
`src/wgsl.ts` for the exact contract) — plus `pnanovdb_buf_t`/address
plumbing and (for point-sampling) an accessor struct. Two real, working
integrations to copy from instead of re-deriving the binding yourself:

- **`@pmndrs/vdb/three`** ([`src/three`](./src/three)) splices
  this exact file into a TSL fragment shader — `NanoVDBVolumeMaterial`
  handles the buffer-global rewrite and entry-point assembly for you.
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
| other `GridType`s | rejected | `NanoVDBFile` throws a clear "unsupported grid type" error — v1 targets FogVolume rendering (SPEC §2.1) |

| File codec | Support |
|---|---|
| `NONE` | yes |
| `ZIP` (zlib, via `fflate`) | yes |
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
quantize to Fp8/FpN, apply affine transforms, inspect memory layout, and
write `.nvdb` files — all **pure TypeScript, zero runtime dependencies,
zero WASM** ([decision D3](../../docs/DECISIONS.md#d3--cpu-half-pure-typescript-first-wasm-as-targeted-escalation)).
WASM stays an optional escalation rung, adopted only on demonstrated need
— nothing in this package requires it.

### Quickstart: the demo 07 pipeline

The full authoring round-trip — generate/obtain voxels, build a NanoVDB
grid, optionally quantize, write a real `.nvdb` file — trimmed from
[`examples/src/demos/07-builder`](../../examples/src/demos/07-builder):

```ts
import { buildFromDenseDetailed, quantize, writeNvdb } from "@pmndrs/vdb/tools";

const dim = 32;
const values = new Float32Array(dim * dim * dim); // fill with your density field
// ... write nonzero values into `values` at your active voxels ...

const built = buildFromDenseDetailed(values, [dim, dim, dim], {
  gridName: "my_grid",
  background: 0,
});
// built: { image: Uint32Array, voxelCount, indexBBox, worldBBox, voxelSize, nodeCounts }

const fp8Image = quantize(built.image, "fp8"); // or "fpn"; per-leaf min/quantum, native-exact rounding

const fileBuffer = writeNvdb([fp8Image]); // ArrayBuffer — a real NanoVDB2 file, ready to save/upload
```

`fileBuffer` round-trips through `NanoVDBFile.fromArrayBuffer`
byte-for-byte (that loader is this subpath's read path, see above).

### Quickstart: parse a real `.vdb` and build NanoVDB from it

The drop-in-a-file pipeline, trimmed from
[`examples/src/demos/06-explorer`](../../examples/src/demos/06-explorer):

```ts
import { parseVdb, buildFromVdbDetailed, inspect } from "@pmndrs/vdb/tools";

const buffer = await fetch("/fixtures/vdb-samples/utahteapot.vdb").then((r) => r.arrayBuffer());
const vdbFile = parseVdb(buffer); // { fileVersion, grids: VdbGrid[] }

const built = buildFromVdbDetailed(vdbFile.grids[0]!); // streams leaves, no dense allocation
const report = inspect(built.image);
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
| `buildFromDense` / `buildFromDenseDetailed` | Dense `Float32Array` -> NanoVDB FLOAT grid image |
| `buildFromLeavesDetailed` | Lower-level: build from an already-leaf-shaped source (what `buildFromVdb` and `quantize` are built on) |
| `buildFromVdb` / `buildFromVdbDetailed` | Parsed `.vdb` grid -> NanoVDB FLOAT grid image, leaf-streamed |
| `quantize` / `quantizeDetailed` | FLOAT grid image -> Fp8/FpN (per-leaf min/quantum; FpN bit-width picked per leaf) |
| `transform` | Metadata-only affine edit (uniform scale + translate) on a copy; throws with a GPU-resample pointer for rotation/shear ([D6](../../docs/DECISIONS.md#d6--browser-first-companion-service-is-an-interim-crutch-to-eliminate)) |
| `inspect` | Node counts + memory-by-section breakdown, summing to the grid's total byte size |
| `writeNvdb` | Serializes one or more grid images into a `.nvdb` file `ArrayBuffer` (NanoVDB2 layout) |
| `readNvdb` | Stub — reading `.nvdb` is `NanoVDBFile`, exported from this same subpath (`@pmndrs/vdb/tools`) |

### Supported `.vdb` subset

| | Support |
|---|---|
| Grid type | `FloatGrid` |
| Tree configuration | 5-4-3 (OpenVDB's default) |
| Compression | none, zlib |
| Compression | blosc — throws a clear "not supported" error |
| Half-float leaf storage | yes |
| Transform | uniform scale + translate (affine `Map`); non-uniform/rotated throws on `transform()`, parses fine on `parseVdb()`/`buildFromVdb()` |

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

The three.js/TSL layer of [WebVDB](../../README.md): wraps a NanoVDB grid
image (from `@pmndrs/vdb/tools`) as a renderable object, ships
a from-scratch fragment-raymarch volume material, a device-first WebGPU
renderer bootstrap, and a handful of GPU compute utilities for stats,
value remapping, and dense-atlas decoding.

### Quickstart: render a cloud

This mirrors [`examples/src/demos/02-cloud`](../../examples/src/demos/02-cloud),
trimmed to the essentials (no camera controls, no live uniforms):

```ts
import * as THREE from "three";
import { NanoVDBFile } from "@pmndrs/vdb/tools";
import pnanovdbSource from "@pmndrs/vdb/pnanovdb.wgsl?raw";
import { NanoVDBGrid, NanoVDBVolumeMaterial, createVolumeRenderer } from "@pmndrs/vdb/three";

const buffer = await fetch("/fixtures/wdas_cloud_quarter_fp8.nvdb").then((r) => r.arrayBuffer());
const file = NanoVDBFile.fromArrayBuffer(buffer);
const grid = NanoVDBGrid.fromFile(file, 0);

// Device-first bootstrap (decision D4): requests the adapter and creates
// the GPUDevice itself (raising maxStorageBufferBindingSize to fit `grid`),
// then hands that device to `WebGPURenderer` — rather than letting the
// renderer create its own device with default limits.
const { renderer } = await createVolumeRenderer({ gridBytes: grid.byteLength });

const scene = new THREE.Scene();
const material = new NanoVDBVolumeMaterial({ grid, pnanovdbSource });
const mesh = new THREE.Mesh(grid.proxyGeometry(), material);
scene.add(mesh);

const worldBox = grid.worldBBox();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, worldBox.getSize(new THREE.Vector3()).length() * 100);
camera.position.set(2, 1.5, 2);
camera.lookAt(worldBox.getCenter(new THREE.Vector3()));

await renderer.renderAsync(scene, camera);
```

`NanoVDBGrid.fromFile(file, i)` wraps grid `i`'s image as a
`StorageBufferAttribute` + metadata, and derives a `Box3`/proxy
`BufferGeometry` for it. `NanoVDBVolumeMaterial` assembles the vendored
`pnanovdb.wgsl` traversal into a TSL `NodeMaterial` fragment shader (fog
raymarch, sun shadow, per-material live uniforms for density/step/etc. —
see the material's own doc comment for the full uniform list).

Live uniforms are plain three.js `Uniform`s exposed as properties, e.g.
`material.densityScale.value = 45`, `material.stepSize.value = 0.4`.

### Compute utilities

Standalone GPU compute passes over a `NanoVDBGrid` — each takes the
`GPUDevice` (from `createVolumeRenderer`) and the same vendored
`pnanovdbSource` text the material uses:

```ts
import { gridStats, valueTransform, decodeToAtlas } from "@pmndrs/vdb/three";

const stats = await gridStats(device, grid, pnanovdbSource);
// { min, max, mean, histogram: Uint32Array(256) }

const { image } = await valueTransform(device, grid, pnanovdbSource, "return v * 2.0;");
// or a named preset: VALUE_TRANSFORM_PRESETS has "double" | "half" | "negate" | "clamp01"

const atlas = await decodeToAtlas(device, grid, pnanovdbSource, { format: "uint8", maxDim: 128 });
// { data: Uint8Array | Float32Array, dims: [x, y, z], range: { min, max } }
```

`valueTransform` is Float-grid-only in v1 (the address math assumes a
4-byte, non-quantized value stride) and returns a **new** image — it
never mutates the input. `decodeToAtlas` works for Float/Fp8/FpN and is
the fallback path for renderers that sample a dense atlas instead of
doing per-fragment sparse traversal (see
[`examples/src/demos/04-atlas-fallback`](../../examples/src/demos/04-atlas-fallback),
which cross-checks both against a CPU `readValue` pass).

### Sequences: `NanoVDBSequence`

`NanoVDBSequence` plays a list of per-frame `.nvdb` URLs through a single
material via an in-place `rebindGrid()` call — no material rebuild, no
shader recompile, every frame:

```ts
import { NanoVDBSequence } from "@pmndrs/vdb/three";

const material = new NanoVDBVolumeMaterial({
  grid: firstFrameGrid,
  pnanovdbSource,
  maxGridBytes: 8 * 1024 * 1024, // headroom: the largest frame in the sequence
});

const sequence = new NanoVDBSequence({
  urls: (i) => `/fixtures/embergen/pack/frame_${i}.nvdb`,
  frameCount: 48,
  fps: 24,
});
sequence.start({ rebindGrid: (grid) => material.rebindGrid(grid) });
// each animation frame:
sequence.update(performance.now());
```

`maxGridBytes` pre-sizes the material's storage buffer so later frames
(which may be a different byte size than the first) can rebind into the
same GPU buffer without reallocating — set it to the largest frame's byte
length up front, or rebinding a bigger grid throws with the exact value to
use. The sequence never blocks the render loop waiting on a fetch: if a
frame isn't decoded yet when its time comes, it holds the last frame,
counts a stall, and keeps going (see `sequence.stats`).

### Why device-first (D4)

Every entry point that needs a `GPUDevice` — `createVolumeRenderer`, and
by extension the whole material/compute surface — requests the adapter
and creates the device **itself** (with `requiredLimits` sized to the
grid) and hands it to `WebGPURenderer` at construction, rather than
letting the renderer create its own device with default limits and
finding out later a grid doesn't fit. See
[decision D4](../../docs/DECISIONS.md#d4--targets-and-device-creation).


## Packaging note (for contributors)

The published export map points only at built `dist/` output (`dist/{core,tools,three}`, one `tsc` build).
Workspace Vite/Vitest/TypeScript configs alias `@pmndrs/vdb`, `@pmndrs/vdb/tools`, `@pmndrs/vdb/three` and
`@pmndrs/vdb/pnanovdb.wgsl` to `src/` for source-level development without exposing TypeScript source paths
to npm consumers.
