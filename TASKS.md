# Tasks

Open work by phase ([PHASES.md](PHASES.md)). Owners:
- `owner`: Dennis
- `coord`: the coordinating session
- `T3`: Opus agent, for novel or risky work
- `T2`: Sonnet agent, for specified implementation
- `T1`: Haiku agent, for mechanical work
- `agent:<area>`: an agent working in a consumer repo

Status: ⬜ open · 🟨 in progress · ✅ done · ⏸ blocked. When you finish a task, mark it done, add follow-ups, and
record departures in [DIVERGES.md](DIVERGES.md).

## K0 — Re-foundation

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K0.0 | Reconcile the owner's newer local checkout with `origin`: push its unpushed work, then audit this plan against it | owner | ⬜ | Blocks K0.1+. The pathtracer notes record a WDAS 103.96 fps run that is not in this repo (R2). |
| K0.1 | Confirm D9 (package name and scope, repo home) and D13 (MIT) | owner | ⬜ | Until confirmed, the coordinator's recommendation stands. |
| K0.2 | Restructure into `packages/{core,tools,three}`, with private workspace packages bundled into one published package | T2 | ⬜ | Moves only. `git mv` to keep history. The nanovdb-wgsl loader and CPU oracle go to `tools`; the WGSL fork and constants go to `core`. |
| K0.3 | Lint boundary: `core` and `tools` must not import `three` (bvh's eslint `no-restricted-imports` rule) | T1 | ⬜ | |
| K0.4 | Build and exports: subpaths `.`, `./tools`, `./three`; vite or tsc build with dts; `verify:packed-subpaths` (upscaler) | T2 | ⬜ | Keep the `development` condition so workspace imports resolve to source. |
| K0.5 | Rebrand README, STATUS and package READMEs to KumoVDB; add a "superseded by PHASES.md" banner to `docs/PLAN.md`, `docs/SPEC.md`, `docs/DECISIONS.md` | T1 | ⬜ | Held out of the plan push so the owner's local branch merges cleanly. |
| K0.6 | Apply the license (D13) to code headers, LICENSE and NOTICE; keep the vendored Apache-2.0 files | T1 | ⬜ | After K0.1. |
| K0.7 | Update the pathtracer importer (`bench/scenes/volumes/vdb-grid.ts`, which reads `packages/vdb-web-tools/dist/index.js` via `WEBVDB_DIR`) | agent:pathtracer | ⬜ | Same change window as K0.2; follow the pathtracer's CLAUDE.md. |
| K0.8 | Gate run, plus `docs/handoffs/K0.md` | T2 | ⬜ | 234 unit / 11 e2e / goldens byte-identical / pack dry-run. |

## K1 — Engine-free core

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K1.1 | Set the WGSL budget with the pathtracer: bytes and compile time per profile on its reference device | coord + agent:pathtracer | ⬜ | Its kernels are 41–55 KB, and a 10 KB reader is the case it called a problem. |
| K1.2 | Design note: `KumoGridPool` layout (grid table, alignment, layout version), the `{group, firstBinding}` slot API, symbol prefix | T3 | ⬜ | `spec/SPEC-draft.md` §3 is the starting sketch. bvh's `traceModuleWgsl` and `GPU_LAYOUT_VERSION` are the precedent. |
| K1.3 | `KumoGridPool`: create, add, remove, update(encoder), version, dispose; host-owned device | T2 | ⬜ | |
| K1.4 | WGSL module generator with `read`, `sample`, `march` and `majorant` profiles; grid types compiled in or out | T3 | ⬜ | Split the vendored fork into composable pieces. The fork's diff log records every change (D2). |
| K1.5 | Size table and CI guard (pathtracer `size-snapshot` style) | T1 | ⬜ | |
| K1.6 | Move the GPU parity harness onto generated modules: every profile × Float/Fp8/FpN | T2 | ⬜ | Fresh-eyes verification agent afterwards (D11). |
| K1.7 | Raw-WebGPU example: smoke rendered with no engine | T2 | ⬜ | Upscaler's `examples/webgpu/hello` is the precedent. |

## K2 — three.js adapter

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K2.1 | `getDevice(renderer)` and `getGPUTexture`, matching upscaler and denoiser so all tools share one device | T2 | ⬜ | |
| K2.2 | `KumoGridNode` (`.module`, `.update()`) and the `kumoSample()` TSL helper | T3 | ⬜ | Replaces the v1 `setName` and buffer-rewrite trick. |
| K2.3 | Port the volume material, `gridStats`, `valueTransform`, `decodeToAtlas` and `NanoVDBSequence` | T2 | ⬜ | |
| K2.4 | Port demos 01–07 and keep their goldens | T2 | ⬜ | Re-baseline only with a DIVERGES entry. |

## K3 — Pathtracer integration

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K3.1 | Point-sample API in its convention (voxel centres, trilinear, zero outside) | T2 | ⬜ | |
| K3.2 | Majorant DDA, two ways behind a setting: NanoVDB node-max HDDA, and 8³ brick majorants | T3 | ⬜ | Pathtracer P5.v8/v10 measure the brick version. |
| K3.3 | Several grids per kernel through the grid table, each with its own transform | T2 | ⬜ | |
| K3.4 | Land in `pmndrs/pathtracer` behind a setting; A/B against the dense texture on `vol-vdb-smoke` | agent:pathtracer | ⬜ | Its exit gate: agree with pbrt-v4 at the noise floor. |

## K4 — Tools and interop

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K4.1 | Parse active tiles (`bunny_cloud.vdb`) | T2 | ⬜ | mjurczyk's MIT reader is a reference; attribute anything ported. |
| K4.2 | Vector grids, then point grids | T2 | ⬜ | |
| K4.3 | `.wvdb` read and write per spec v0.5 | T2 | ⬜ | A decoder written from the spec already worked in the comparison report. |
| K4.4 | Resample-to-resolution bake | T2 | ⬜ | Measured: our FpN at his 128³ is 474 KiB against his 790 KiB. |
| K4.5 | Coarse-first `.nvdb` delivery | T3 | ⬜ | Design first; his LOD0 is 14 KiB. |
| K4.6 | Quantize keeps uniform regions as tiles (the ~12% Fp8 gap) | T2 | ⬜ | |

## K5 — General renderer

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K5.1 | Renderer design note from R1 | T3 | ⬜ | Blocked on R1. |
| K5.2 | HDDA empty-space skipping in the march (deferred from v1) | T3 | ⬜ | |
| K5.3 | Depth compositing with the scene | T2 | ⬜ | |
| K5.4 | Shadows and a multiple-scattering approximation, A/B-tested | T3 | ⬜ | |

## K6 — Release

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K6.1 | pmndrs/docs MDX site with compiled snippets | T2 | ⬜ | |
| K6.2 | Release workflow: manual, npm trusted publishing, CHANGELOG | T1 | ⬜ | |

## Research

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| R1 | `FrameRenderer` study, written to `research/notes/R1-frame-renderer.md` | T3 | ⬜ | Read his source statically; don't run it in the cloud sandbox. |
| R2 | Recover the WDAS 103.96 fps run and commit the perf table | owner | ⬜ | Likely in the owner's local checkout (K0.0). |
| R3 | Native `nanovdb_convert` parity sweep | owner | ⬜ | docs/handoffs/PHASE-5.md checklist. |
| R9 | Retest the v1 sandbox workarounds on hardware | owner | ⬜ | docs/ON-DEVICE.md §6. |
| R10 | Outreach to mjurczyk; `.nvdb` loader PR offer | owner | ⬜ | Comparison report: https://claude.ai/artifact/FV7TNDJVmhuLxynaPf8X9o |
