# Tasks

Open work by phase ([PHASES.md](PHASES.md)). Owners:
- `owner`: Dennis
- `coord`: the coordinating session
- `T3`: Opus agent, for novel or risky work
- `T2`: Sonnet agent, for specified implementation
- `T1`: Haiku agent, for mechanical work

Status: ⬜ open · 🟨 in progress · ✅ done · ⏸ blocked. When you finish a task, mark it done, add follow-ups, and
record departures in [DIVERGES.md](DIVERGES.md).

## K0 — Re-foundation

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K0.0 | Reconcile the owner's local checkout with `origin` and audit this plan against it | coord | ✅ | 2026-11-18: on-device work pushed to `main` (`eee81bf`); plan merged (`c3732d0`); stale items fixed in this revision. |
| K0.1 | Confirm the name, scope and repo home (D17) and the licence (D13) | owner | 🟨 | D17 confirmed 2026-11-18. D13 (MIT) still open; v1 packages currently carry Apache-2.0 `LICENSE` files. |
| K0.2 | Tag `v1-final`, then restructure into `packages/{core,tools,three}`, with private workspace packages bundled into one published `@pmndrs/vdb` | T2 | ⬜ | Moves only. `git mv` to keep history. The nanovdb-wgsl loader and CPU oracle go to `tools`; the WGSL fork and constants go to `core`. |
| K0.3 | Lint boundary: `core` and `tools` must not import `three` (bvh's eslint `no-restricted-imports` rule) | T1 | ⬜ | |
| K0.4 | Build and exports: subpaths `.`, `./tools`, `./three`; build with dts; `verify:packed-subpaths` (upscaler) | T2 | ⬜ | Exports point at `dist/` only. Keep v1's source resolution (tsconfig `paths` + Vite/Vitest aliases); do **not** add a `development` export condition. |
| K0.5 | Rebrand README, STATUS and package READMEs to `@pmndrs/vdb`; add a "superseded by PHASES.md" banner to `docs/PLAN.md`, `docs/SPEC.md`, `docs/DECISIONS.md` | T1 | ⬜ | |
| K0.6 | Apply the licence (D13) to code headers, LICENSE and NOTICE; keep the vendored Apache-2.0 files | T1 | ⏸ | Blocked on K0.1 (D13). |
| K0.7 | Move the repo to `pmndrs/vdb` (GitHub transfer keeps redirects) | owner | ⬜ | Timing is the owner's call; can follow K0.2. |
| K0.8 | Gate run, plus `docs/handoffs/K0.md` | T2 | ⬜ | 249 unit / 11 e2e / goldens byte-identical / pack dry-run / packed-consumer install. Note the new tools path for the pathtracer's importer. |

## K1 — Engine-free core

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K1.1 | Set the WGSL size budget per profile: bytes and compile time on one reference device | coord | ⬜ | Use the pathtracer's published numbers as inputs (41–55 KB kernels; a 10 KB reader was its problem case). Not a gate on its team. |
| K1.2 | Design note: `VDBGridPool` layout (grid table, alignment, layout version), the `{group, firstBinding}` slot API, symbol prefix | T3 | ⬜ | `spec/SPEC-draft.md` §3 is the starting sketch. bvh's `traceModuleWgsl` and `GPU_LAYOUT_VERSION` are the precedent. |
| K1.3 | `VDBGridPool`: create, add, remove, update(encoder), version, dispose; host-owned device | T2 | ⬜ | |
| K1.4 | WGSL module generator with `read`, `sample`, `march` and `majorant` profiles; grid types compiled in or out | T3 | ⬜ | Split the vendored fork into composable pieces. The fork's diff log records every change (D2). |
| K1.5 | Size table and CI guard | T1 | ⬜ | |
| K1.6 | Move the GPU parity harness onto generated modules: every profile × Float/Fp8/FpN | T2 | ⬜ | Fresh-eyes verification agent afterwards (D11). |
| K1.7 | Raw-WebGPU example: smoke rendered with no engine | T2 | ⬜ | Upscaler's `examples/webgpu/hello` is the precedent. |

## K2 — three.js adapter, v1 parity

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K2.1 | `getDevice(renderer)` and `getGPUTexture`, matching upscaler and denoiser so all tools share one device | T2 | ⬜ | |
| K2.2 | `VDBGridNode` (`.module`, `.update()`) and the `vdbSample()` TSL helper | T3 | ⬜ | Replaces the v1 `setName` and buffer-rewrite trick. |
| K2.3 | Port the v1 volume material, `gridStats`, `valueTransform`, `decodeToAtlas` and the sequence player | T2 | ⬜ | |
| K2.4 | Port demos 01–07 and keep their goldens | T2 | ⬜ | Re-baseline only with a DIVERGES entry. |
| K2.5 | Drop the sandbox-only workarounds from the default path (DIVERGES V7/V8) | T2 | ⬜ | R9 showed hardware passes without them; keep them only if CI's SwiftShader still needs them, behind a flag. |

## K3 — Renderer: port and extend

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K3.1 | Volume-source interface: sparse `VDBGridPool` and dense `.wvdb` levels behind one sampling API | T3 | ⬜ | Blocked on R1 and K2.2. |
| K3.2 | Port march and lighting basics; A/B against his renderer on the same asset | T2 | ⬜ | Attribution on every ported file. |
| K3.3 | Depth-aware froxels; shadows on meshes; emission onto meshes | T3 | ⬜ | |
| K3.4 | Multiple scattering (his diffusion, octaves and network variants), then spectral media and phase functions | T3 | ⬜ | Uses the `majorant` profile where delta tracking applies. |
| K3.5 | HDDA empty-space skipping in the march (deferred since v1) | T3 | ⬜ | The sparse advantage; A/B against fixed-step. |
| K3.6 | Level-set surfaces and point grids | T2 | ⬜ | Needs K4.2. |
| K3.7 | Lift engine-free shading WGSL into `core` | T2 | ⬜ | So other adapters and the cloud system reuse it. |
| K3.8 | Gate run: his scenes on our source, WDAS at a named quality preset ≥ 60 fps 1080p, goldens per lighting mode | T2 | ⬜ | Quiet machine, paired runs (D11). |

## K4 — Tools and interop

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K4.1 | Parse active tiles (`bunny_cloud.vdb`) | T2 | ⬜ | mjurczyk's MIT reader is a reference; attribute anything ported. |
| K4.2 | Vector grids, then point grids, then level-set specifics | T2 | ⬜ | |
| K4.3 | `.wvdb` read and write per his spec (v0.5, plus points from his v0.25.0) | T2 | ⬜ | A decoder written from the spec already worked in the comparison report. |
| K4.4 | Resample-to-resolution bake | T2 | ⬜ | Measured: our FpN at his 128³ is 474 KiB against his 790 KiB. |
| K4.5 | Coarse-first `.nvdb` delivery | T3 | ⬜ | Design first; his LOD0 is 14 KiB. |
| K4.6 | Quantize keeps uniform regions as tiles (the ~12% Fp8 gap) | T2 | ⬜ | |

## K5 — Release

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K5.1 | pmndrs/docs MDX site with compiled snippets | T2 | ⬜ | |
| K5.2 | Release workflow: manual, npm trusted publishing, CHANGELOG | T1 | ⬜ | |
| K5.3 | Cloud-system showcase example | owner | ⬜ | |

## After 0.1

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| P1 | Pathtracer integration: point-sample API, majorant DDA two ways (node-max HDDA vs 8³ bricks), several grids per kernel, landed behind a setting | owner | ⬜ | D19. Update its importer (`bench/scenes/volumes/vdb-grid.ts`) to the K0 layout first. |

## Research

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| R1 | Renderer port plan, written to `research/notes/R1-renderer-port.md` | T3 | ⬜ | Read his source statically; don't run it in the cloud sandbox. Track his releases (0.26.0 on 2026-11-18). |
| R2 | WDAS perf on hardware | owner | ✅ | 2026-07-12; `docs/handoffs/PHASE-3.md`. |
| R3 | Native `nanovdb_convert` parity sweep | owner | ✅ | 2026-07-12; `pnpm parity:native`, `docs/handoffs/PHASE-5.md`. |
| R9 | Retest the v1 sandbox workarounds on hardware | owner | ✅ | 2026-07-12; follow-up is K2.5. |
| R10 | Outreach to mjurczyk: invite him in, offer the `.nvdb` loader PR; nudge him to add the LICENSE file (not blocking: it's MIT) | owner | ⬜ | Comparison report: https://claude.ai/artifact/FV7TNDJVmhuLxynaPf8X9o |
