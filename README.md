# WebVDB

OpenVDB/NanoVDB volumes on the web, **browser-first**: traverse and render
sparse volumes directly in WGSL on WebGPU (three.js `WebGPURenderer` + TSL),
with pure-TypeScript CPU tooling for parsing, building, and inspecting grids.
No dense-bake tricks, no WASM in the default install.

**Status: Phases 0–7 complete (CI scope)** — workspace scaffold, vendored
WGSL fork, fixture pipeline, WebGPU traversal, TSL volume rendering,
pure-TS CPU tooling (parse/build/quantize/transform/inspect), and
sequence playback all ship and are e2e-gated on `main`. Suite: 234 unit /
11 e2e (goldens byte-stable) / clean tsc. See
[STATUS.md](./STATUS.md) and [docs/PLAN.md](./docs/PLAN.md) for the phase
gates, and [docs/handoffs/](./docs/handoffs) for the phase-by-phase
handoff notes. Remaining work needs a native/hardware machine or network
access this environment doesn't have (native parity sweep, WDAS perf
table, EmberGen fixtures) — see STATUS.md's "Needs a native/hardware
machine" line.

## Packages

| Package | What it is |
|---|---|
| [`nanovdb-wgsl`](./packages/nanovdb-wgsl) | Renderer-agnostic NanoVDB traversal/sampling WGSL module (vendored Apache-2.0 fork of [emcfarlane/webgpu-nanovdb](https://github.com/emcfarlane/webgpu-nanovdb)'s `pnanovdb.wgsl`, pinned + audited) and the TS `.nvdb` loader |
| [`three-nanovdb`](./packages/three-nanovdb) | TSL/three.js layer: `NanoVDBGrid`, volume materials, compute utilities, device-first renderer bootstrap |
| [`vdb-web-tools`](./packages/vdb-web-tools) | Pure-TS CPU tooling: `.vdb` parse, NanoVDB build, Fp8/FpN quantize, affine transform, inspect (WASM only as an opt-in escalation) |
| [`examples/`](./examples) | Vite demo site — one demo per phase gate |

## Demos

Each demo is a phase gate (see [docs/SPEC.md §5](./docs/SPEC.md)); run
`pnpm dev` and open the listed path.

| # | Demo | Proves |
|---|---|---|
| 01 | [`hello-nvdb`](./examples/src/demos/01-hello-nvdb) — load `.nvdb`, compute-shader probe of known voxels vs. native ground truth | contract + traversal correctness |
| 02 | [`cloud`](./examples/src/demos/02-cloud) — Fp8 fog volume fragment-raymarched with sun lighting, orbit camera, live controls | **main goal** |
| 03 | [`gpu-read`](./examples/src/demos/03-gpu-read) — minimal annotated "read VDBs on the GPU" example (one file, heavily commented, raw WebGPU) | **main goal (educational)** |
| 04 | [`atlas-fallback`](./examples/src/demos/04-atlas-fallback) — same style of volume through compute → dense atlas → stock volume material | mobile/compat path |
| 05 | [`embergen-sequence`](./examples/src/demos/05-embergen-sequence) — animated volume playback via `NanoVDBSequence`'s `rebindGrid` | animation |
| 06 | [`explorer`](./examples/src/demos/06-explorer) — drag-drop `.nvdb`/`.vdb`: metadata panel, node-bbox wireframes, slice view, histogram, memory breakdown | technical tool |
| 07 | [`builder`](./examples/src/demos/07-builder) — author a grid in-browser (procedural dense → `vdb-web-tools` build → render, byte-verified round trip) | `vdb-web-tools` v1 round-trip |

## Founding documents

The feasibility study, spec, decisions, and build plan live in
[`docs/`](./docs). **[docs/DECISIONS.md](./docs/DECISIONS.md) (D1–D6) is
binding** — browser-first mission, vendored WGSL fork, pure-TS CPU path,
desktop-only v1 with device-first `GPUDevice` creation, and a native
companion service kept strictly as an interim crutch with a
browser-parity retirement roadmap.

## Getting started

```sh
pnpm install
pnpm typecheck    # tsc --build across the workspace (project references)
pnpm test         # Vitest across the workspace (234 unit tests)
pnpm test:e2e     # Playwright against the examples app (11 e2e specs)
pnpm dev          # examples dev server → http://localhost:5173
pnpm build        # tsc --build each package's dist/ (for publishing)
pnpm pack:dry     # npm pack --dry-run in each package, tarball contents only
pnpm fixtures     # download test volumes into fixtures/ (git-ignored)
```

`typecheck`/`test`/`dev` all resolve cross-package imports (`nanovdb-wgsl`,
`three-nanovdb`, `vdb-web-tools`) straight to TypeScript source via each
package's `"development"` export condition — no build step needed for
day-to-day work. `pnpm build` is only needed to produce the `dist/` a
published package ships (see each package's README "packaging note").

Fixtures (WDAS cloud, EmberGen free pack) are **never committed** — see
[fixtures/README.md](./fixtures/README.md) for the pipeline and licenses.

## License

Project code: Apache-2.0. Vendored third-party code keeps its upstream
license and attribution — see
[`packages/nanovdb-wgsl/vendor/NOTICE`](./packages/nanovdb-wgsl/vendor/NOTICE).
