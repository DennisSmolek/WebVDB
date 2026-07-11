# WebVDB

OpenVDB/NanoVDB volumes on the web, **browser-first**: traverse and render
sparse volumes directly in WGSL on WebGPU (three.js `WebGPURenderer` + TSL),
with pure-TypeScript CPU tooling for parsing, building, and inspecting grids.
No dense-bake tricks, no WASM in the default install.

**Status: Phase 0** — workspace scaffold, vendored WGSL fork, fixture
pipeline, test wiring. See [docs/PLAN.md](./docs/PLAN.md) for the phase gates.

## Packages

| Package | What it is |
|---|---|
| [`nanovdb-wgsl`](./packages/nanovdb-wgsl) | Renderer-agnostic NanoVDB traversal/sampling WGSL module (vendored Apache-2.0 fork of [emcfarlane/webgpu-nanovdb](https://github.com/emcfarlane/webgpu-nanovdb)'s `pnanovdb.wgsl`, pinned + audited) and the TS `.nvdb` loader |
| [`three-nanovdb`](./packages/three-nanovdb) | TSL/three.js layer: `NanoVDBGrid`, volume materials, compute utilities, device-first renderer bootstrap |
| [`vdb-web-tools`](./packages/vdb-web-tools) | Pure-TS CPU tooling: `.vdb` parse, NanoVDB build, Fp8/FpN quantize, affine transform, inspect (WASM only as an opt-in escalation) |
| [`examples/`](./examples) | Vite demo site — one demo per phase gate |

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
pnpm test         # Vitest across the workspace (Phase 0 gate)
pnpm test:e2e     # Playwright smoke against the examples app
pnpm dev          # examples dev server → http://localhost:5173
pnpm fixtures     # download test volumes into fixtures/ (git-ignored)
```

Fixtures (WDAS cloud, EmberGen free pack) are **never committed** — see
[fixtures/README.md](./fixtures/README.md) for the pipeline and licenses.

## License

Project code: Apache-2.0. Vendored third-party code keeps its upstream
license and attribution — see
[`packages/nanovdb-wgsl/vendor/NOTICE`](./packages/nanovdb-wgsl/vendor/NOTICE).
