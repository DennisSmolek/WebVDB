# three-nanovdb

The three.js/TSL layer of [WebVDB](../../README.md): wraps a NanoVDB grid
image (from [`nanovdb-wgsl`](../nanovdb-wgsl)) as a renderable object, ships
a from-scratch fragment-raymarch volume material, a device-first WebGPU
renderer bootstrap, and a handful of GPU compute utilities for stats,
value remapping, and dense-atlas decoding.

## Install

```sh
npm install three-nanovdb three nanovdb-wgsl
```

`three` (`>=0.178.0`) and `nanovdb-wgsl` are peer dependencies — you bring
your own three.js version.

## Quickstart: render a cloud

This mirrors [`examples/src/demos/02-cloud`](../../examples/src/demos/02-cloud),
trimmed to the essentials (no camera controls, no live uniforms):

```ts
import * as THREE from "three";
import { NanoVDBFile } from "nanovdb-wgsl";
import pnanovdbSource from "nanovdb-wgsl/pnanovdb.wgsl?raw";
import { NanoVDBGrid, NanoVDBVolumeMaterial, createVolumeRenderer } from "three-nanovdb";

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

## Compute utilities

Standalone GPU compute passes over a `NanoVDBGrid` — each takes the
`GPUDevice` (from `createVolumeRenderer`) and the same vendored
`pnanovdbSource` text the material uses:

```ts
import { gridStats, valueTransform, decodeToAtlas } from "three-nanovdb";

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

## Sequences: `NanoVDBSequence`

`NanoVDBSequence` plays a list of per-frame `.nvdb` URLs through a single
material via an in-place `rebindGrid()` call — no material rebuild, no
shader recompile, every frame:

```ts
import { NanoVDBSequence } from "three-nanovdb";

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

## Why device-first (D4)

Every entry point that needs a `GPUDevice` — `createVolumeRenderer`, and
by extension the whole material/compute surface — requests the adapter
and creates the device **itself** (with `requiredLimits` sized to the
grid) and hands it to `WebGPURenderer` at construction, rather than
letting the renderer create its own device with default limits and
finding out later a grid doesn't fit. See
[decision D4](../../docs/DECISIONS.md#d4--targets-and-device-creation).

## Packaging note (for contributors)

Like `nanovdb-wgsl`, this package's `"."` export has a `"development"`
condition (`./src/index.ts`) for in-workspace resolution and
`"import"`/`"types"` conditions pointing at the built `dist/` for
published consumers.
