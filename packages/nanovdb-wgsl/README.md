# nanovdb-wgsl

Renderer-agnostic NanoVDB traversal/sampling **WGSL module** + a
TypeScript **`.nvdb` file loader**. This is the low-level layer of
[WebVDB](../../README.md): it doesn't know about three.js or any
particular renderer — just how to get NanoVDB grid bytes off disk/network
and how to walk them, in WGSL or in plain TS.

## What's in the box

- **`vendor/pnanovdb.wgsl`** — a WGSL module with the NanoVDB root ->
  upper -> lower -> leaf tree traversal, HDDA ray marching, and
  Float/Fp8/FpN value decoders. Import it as raw text (`?raw` in Vite,
  or read the file yourself) and splice it into your own shader.
- **`NanoVDBFile`** — parses a `.nvdb` file (header + per-grid metadata +
  grid images), handling the NONE/ZIP codecs.
- **A pure-TypeScript CPU reference** (`readValue`, `sampleTrilinear`,
  `probeCoords`/`probePoints`) — the exact same tree descent as the WGSL
  module, so you can sanity-check GPU results or read voxels with no GPU
  at all (Node, workers, tests).

## Install

```sh
npm install nanovdb-wgsl
```

## Quickstart: load a file and read a voxel

```ts
import { NanoVDBFile, readValue } from "nanovdb-wgsl";

const file = await NanoVDBFile.fromURL("/fixtures/sphere_fog_float.nvdb");
const gridImage = file.gridImage(0); // flat Uint32Array — the grid, as-is on disk
const meta = file.grids[0]!;
console.log(meta.name, meta.gridType, meta.voxelCount);

const { value, active } = readValue(gridImage, [12, 4, -3]);
```

`gridImage` is a zero-copy view into the file's buffer when possible
(`file.isZeroCopy(0)` tells you which) — upload it straight into a GPU
`storage` buffer, no reshaping needed.

## Loading the raw WGSL module

```ts
import pnanovdbSource from "nanovdb-wgsl/pnanovdb.wgsl?raw"; // Vite/bundler `?raw` import
// or: fetch(new URL("nanovdb-wgsl/pnanovdb.wgsl", import.meta.url)) — see the
// package's `pnanovdbWgslUrl` export for a resolvable URL in non-bundler runtimes
```

`pnanovdbSource` is the library's text, unmodified. It's a genuine C-style
library, not a drop-in snippet: its reads all bottom out in one function,
`pnanovdb_buf_read_uint32`, that expects the *consumer* to declare a
module-scope `var<storage, read> nanovdb_buffer: array<u32>` binding (see
`vendor/VENDOR.md` and the header comment of `three-nanovdb`'s
`src/wgsl.ts` for the exact contract) — plus `pnanovdb_buf_t`/address
plumbing and (for point-sampling) an accessor struct. Two real, working
integrations to copy from instead of re-deriving the binding yourself:

- **`three-nanovdb`** ([`packages/three-nanovdb`](../three-nanovdb)) splices
  this exact file into a TSL fragment shader — `NanoVDBVolumeMaterial`
  handles the buffer-global rewrite and entry-point assembly for you.
- **`examples/src/demos/03-gpu-read`** is a from-scratch, heavily-commented
  raw-WebGPU walkthrough (no three.js) that hand-derives a FLOAT-only
  traversal function equivalent to this library's, teaching the byte
  layout end to end.

## Supported grid types / codecs

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

## The vendoring story

The WGSL module is not written from scratch: per
[decision D2](../../docs/DECISIONS.md#d2--gpu-traversal-base), it's
**vendored as a pinned, audited fork** of
[emcfarlane/webgpu-nanovdb](https://github.com/emcfarlane/webgpu-nanovdb)'s
`pnanovdb.wgsl` — Apache-2.0, NOTICE preserved, every local change logged
in-tree. We're prepared to diverge permanently and happy to upstream
fixes if the author engages. Full pin/hash/diff-log details:
[`vendor/VENDOR.md`](./vendor/VENDOR.md).

## Packaging note (for contributors)

This package exports both a `"development"` condition (`./src/index.ts`,
used inside the WebVDB workspace by Vite/Vitest/tsc for fast, source-level
resolution) and the usual `"import"`/`"types"` conditions pointing at the
built `dist/`. `npm install nanovdb-wgsl` outside this workspace always
gets `dist/`.
