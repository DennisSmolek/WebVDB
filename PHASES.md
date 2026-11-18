# Phases

`@pmndrs/vdb` plan v2, set 2026-11-18 and revised the same day (D17–D19). It replaces v1 phases 0–9
([docs/PLAN.md](docs/PLAN.md), frozen by D15). v1 delivered the parts these phases rearrange: the loader, the
audited WGSL fork, the CPU oracle, the three.js material, the compute tools, the TS parser and serializer, and the
sequence player (see [docs/handoffs/](docs/handoffs/)). Its on-device gates closed on 2026-07-12
([docs/ON-DEVICE.md](docs/ON-DEVICE.md)).

**First consumer:** the owner's cloud system. **After our release:** the pmndrs pathtracer (D19).

Status: ⬜ not started · 🟨 in progress · ✅ done. Each phase ends at a machine-checkable **exit gate** plus a
handoff in `docs/handoffs/K<n>.md`. Task IDs (`K1.3`, `R2`) live in [TASKS.md](TASKS.md).

```
K0 re-foundation ──► K1 core ──► K2 three adapter ──► K3 renderer ──► K5 release 0.1 ──► after 0.1:
                                  (v1 parity)       (port + extend)                     pathtracer, other adapters
        └──► K4 tools + interop  (CPU only, runs in parallel from K0)
Research: R1 (renderer port plan) runs during K1/K2 and feeds K3.
```

## K0 — Re-foundation ⬜

Rename to `@pmndrs/vdb` (D17) and reshape the repo into core plus adapters (D8). **No behaviour changes:** this
phase moves code, it does not rewrite it.

- Workspace: `packages/core` (WebGPU, from `nanovdb-wgsl` minus the loader), `packages/tools` (CPU: the loader,
  parser, serializer, CPU oracle; from `vdb-web-tools` plus the CPU half of `nanovdb-wgsl`), `packages/three`
  (from `three-nanovdb`). Published as one package, `@pmndrs/vdb`, with subpaths `.`, `./tools`, `./three`.
- A lint rule: `core` and `tools` may not import `three`.
- Workspace source resolution stays as v1 finished it: published exports point at `dist/` only, and in-repo
  imports resolve to source through tsconfig `paths` and Vite/Vitest aliases. Do **not** reintroduce a
  `development` export condition (a consumer running with `--conditions=development` would resolve to unpublished
  `src/`).
- The v1 unscoped packages (`nanovdb-wgsl`, `three-nanovdb`, `vdb-web-tools`) are **not published**. Tag the
  last v1 commit `v1-final` before the moves.
- Process docs adopted (this set), the license decision applied (D13), and the README rebranded.
- The pathtracer's path import of `packages/vdb-web-tools/dist/index.js` will break. That is accepted (D19);
  note the new path in the handoff.

**Exit gate:** 249 unit tests and 11 e2e specs green under the new layout; goldens byte-identical; `pnpm
pack:dry` shows one tarball with `.`, `./tools` and `./three`; the packed tarball installs into a clean external
consumer and all three subpaths import from `dist/`.

## K1 — Engine-free WebGPU core ⬜

Everything a WebGPU host needs to hold and read sparse grids, with no engine (D8, D10).

- `VDBGridPool.create(device, options)`: one packed storage buffer with a grid table and N NanoVDB images,
  `add()` / `remove()`, `update(encoder)`, a `version` counter for re-binding, and `dispose()`.
- `vdbModuleWgsl({ slot, types, profile })` and `vdbModuleEntries()`: generated WGSL, `vdb` prefix, features
  compiled out when off. Profiles, smallest first:
  - `read`: value and active state at an index
  - `sample`: plus world-to-index and trilinear
  - `march`: plus HDDA and stats
  - `majorant`: plus a majorant DDA for delta tracking (the K3 renderer's multiple scattering and, later, the
    pathtracer both need it)
- A WGSL size table per profile and grid-type set, measured as bytes and as compile time on one reference
  device, and guarded in CI.
- The GPU parity harness moves onto the module generator. A raw-WebGPU example with no engine is the reference
  for the engine contract.

Design inputs, not gates: the pathtracer's published constraints (Apple Chrome allows 10 storage buffers per
stage; its kernels are 41–55 KB of WGSL) are why the contract is one binding and features compile out.

**Exit gate:** the parity suite is green for every profile × {Float, Fp8, FpN} against the CPU twins; the size
table is committed; the `read` profile for Float is ≤ 4 KB (provisional, confirmed in K1.1); and the raw-WebGPU
example renders the smoke.

## K2 — three.js adapter, v1 parity ⬜

Rebuild `./three` as a thin layer over K1, in the upscaler, denoiser and bvh style. This phase reaches v1
parity on the new core; the new renderer is K3.

- `getDevice(renderer)` reads the renderer's device, the same helper the other pmndrs tools use.
- `VDBGridNode` binds a `VDBGridPool` buffer into TSL, with a `.module` and `.update()`. `vdbSample()` is the
  lowercase TSL helper.
- The v1 volume material, compute tools (`gridStats`, `valueTransform`, `decodeToAtlas`) and the sequence player
  are re-expressed on the core.
- The `setName` and one-token-rewrite binding trick from v1 is replaced by the slot-based module.
- The sandbox-only workarounds (offscreen blit, storage-buffer atlas, swizzle and keep-alive shims) are dropped
  from the default path; hardware passes without them (R9, DIVERGES V7/V8).

**Exit gate:** demos 01–07 run on the new adapter; the e2e suite is green; goldens are unchanged, or
re-baselined with a DIVERGES entry that says why.

## K3 — Renderer: port and extend ⬜

Compete on rendering (D18). Bring mjurczyk's renderer (three-openvdb, MIT) in as the basis of our three.js
renderer, re-based on the sparse core, and lift its engine-independent parts into shared foundations that every
adapter uses.

- **Port plan first (R1):** map his `FrameVolume`/`FrameRenderer` modules into three buckets: engine-free
  shading math (phase functions, spectral media, scattering approximations, froxel logic), three.js
  integration (depth, mesh shadows, emission onto meshes), and the dense-texture data path.
- **Volume source interface:** the renderer reads density through one interface with two sources: the sparse
  `VDBGridPool` (ours) and dense `.wvdb` levels (his format, D14). Users of his renderer keep their assets.
- **Port in order,** each step A/B-tested against his renderer on the same asset (D11): march and lighting
  basics → depth-aware froxels → shadows on meshes and emission → multiple scattering → spectral media and phase
  functions → level-set surfaces and points.
- **Sparse advantages:** HDDA empty-space skipping in the march (deferred since v1), and full-resolution hero
  assets with memory proportional to active voxels.
- **Shared foundations:** the shading WGSL lives in `core` (engine-free), so later adapters (R6) and the
  owner's cloud system reuse it.
- Every ported file carries attribution; every departure gets a DIVERGES row.

**Exit gate:** his demo scenes render on our sparse source within an agreed image-diff threshold of his output;
the WDAS quarter cloud runs at ≥ 60 fps at 1080p on the reference desktop **at a named quality preset** (v1's
104 fps used `stepSize=4`, `shadowSteps=4`; its quality preset ran at 16.8 fps); goldens per lighting mode; and
the A/B results are recorded.

## K4 — Tools and interop ⬜

CPU-side breadth, runnable in parallel from K0 since it touches no GPU code. His MIT `.vdb` reader is the
reference for parser coverage.

- Parser coverage: active tiles (fixes `bunny_cloud.vdb`), then vector grids, then point grids, then level-set
  specifics the K3 renderer needs.
- `.wvdb` read and write per spec v0.5 (D14), tracking his spec as it moves (points landed in his v0.25.0).
- A resample-to-resolution bake option.
- Coarse-first delivery for `.nvdb`: a low-resolution preview grid sent first.
- Keep uniform regions as tiles when quantizing, which closes the ~12% Fp8 size gap with `nanovdb_convert`.

**Exit gate:** `bunny_cloud.vdb` parses and round-trips through the CPU oracle; his committed `.wvdb` files
decode exactly per the spec; a `.wvdb` we write loads in his `WVDBLoader`; and the first paint of a streamed
`.nvdb` is no larger than his 14 KiB smoke preview.

## K5 — Release 0.1 ⬜

- Docs in pmndrs/docs MDX format, with compiled snippets (`docs:check`, as in bvh).
- Examples for raw WebGPU and three.js, plus the owner's cloud system as the showcase consumer.
- A CHANGELOG, and a manual release workflow using npm trusted publishing (the bvh and upscaler model).

**Exit gate:** `@pmndrs/vdb@0.1.0` published from `pmndrs/vdb`, with docs live and the examples linked.

## After 0.1

- **Pathtracer integration (owner-led, D19).** Point sampling in its convention (voxel centres, trilinear, zero
  outside); the majorant DDA built two ways and A/B-tested (NanoVDB node-max HDDA against its 8³ brick
  majorants); several grids per kernel through the grid table; landed behind a setting next to its dense path.
  Its gate: `vol-vdb-smoke` agrees with the pbrt-v4 reference at the noise floor using exactly one storage
  binding.
- **Other adapters** (R6): babylon, pixi.

## Research track (R)

Work that is not on the critical path. Each item ends in a written note in `research/`, a DECISIONS entry, or a
promotion into a K phase.

| ID | Item | Status | Feeds |
|---|---|---|---|
| R1 | Port plan for mjurczyk's renderer: module map (engine-free / three.js / dense data path), what each technique costs, and the port order. | ⬜ | K3 |
| R2 | WDAS perf on hardware | ✅ 2026-07-12: 103.96 fps at 1080p with `stepSize=4`, `shadowSteps=4`; table in `docs/handoffs/PHASE-3.md` | K3 gate |
| R3 | `nanovdb_convert` parity on a native machine | ✅ 2026-07-12: strict FLOAT/FP8 parity on all four pinned sources (`pnpm parity:native`) | K4 |
| R4 | The companion service and its browser successors (D6; v1 phases 6b and 9) | ⬜ | later |
| R5 | WASM escalation: not triggered (D3). Keep watch only. | — | — |
| R6 | Other adapters: babylon, pixi (upscaler and bvh precedent) | ⬜ | after K5 |
| R7 | Bindless and texture-table grids, once a portable WebGPU feature exists | ⬜ | pathtracer |
| R8 | PicoVDB: re-evaluate on the D16 triggers | — | — |
| R9 | Retest the v1 sandbox workarounds on hardware | ✅ 2026-07-12: direct presentation, stock 3D texture and no-shim lanes pass (`pnpm hardware:perf`) | K2 |
| R10 | Bring mjurczyk into the project: outreach, the `.nvdb` loader PR offer (and a nudge to add the LICENSE file) | ⬜ owner | K3 |
