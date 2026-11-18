# Phases

KumoVDB plan v2, set 2026-11-18. It replaces v1 phases 0–9 ([docs/PLAN.md](docs/PLAN.md), frozen by D15). v1
delivered the parts these phases rearrange: the loader, the audited WGSL fork, the CPU oracle, the three.js
material, the compute tools, the TS parser and serializer, and the sequence player (see
[docs/handoffs/](docs/handoffs/)).

Status: ⬜ not started · 🟨 in progress · ✅ done. Each phase ends at a machine-checkable **exit gate** plus a
handoff in `docs/handoffs/K<n>.md`. Task IDs (`K1.3`, `R2`) live in [TASKS.md](TASKS.md).

```
K0 re-foundation ──► K1 core ──► K2 three adapter ──► K5 general renderer ──► K6 release 0.1
                       │                                    ▲
                       ├──► K3 pathtracer integration ──────┤
                       └──► K4 tools + interop  (CPU only, runs in parallel from K0)
Research track R1–R10 runs alongside; R1 feeds K5.
```

## K0 — Re-foundation ⬜

Rename to KumoVDB (D7) and reshape the repo for the core-plus-adapters layout (D8, D9). **No behaviour
changes:** this phase moves code, it does not rewrite it.

- Workspace: `packages/core` (WebGPU, from `nanovdb-wgsl` minus the loader), `packages/tools` (CPU: the loader,
  parser, serializer, CPU oracle; from `vdb-web-tools` plus the CPU half of `nanovdb-wgsl`), `packages/three`
  (from `three-nanovdb`). Published as one package with subpaths (D9).
- A lint rule: `core` and `tools` may not import `three`.
- Process docs adopted (this set), the license decision applied (D13), and the README rebranded.
- Pathtracer compatibility: it imports `packages/vdb-web-tools/dist/index.js` by path today. Update its importer
  in the same change.

**Exit gate:** 234 unit tests and 11 e2e specs green under the new layout; goldens byte-identical; `pnpm
pack:dry` shows one tarball with `.`, `./tools` and `./three`; the pathtracer's `vdb-grid.ts` still converts
`smoke.vdb`.

## K1 — Engine-free WebGPU core ⬜

The heart of the plan: everything a WebGPU host needs to hold and read sparse grids, with no engine (D8, D10).

- `KumoGridPool.create(device, options)`: one packed storage buffer with a grid table and N NanoVDB images,
  `add()` / `remove()`, `update(encoder)`, a `version` counter for re-binding, and `dispose()`.
- `kumoModuleWgsl({ slot, types, features })` and `kumoModuleEntries()`: generated WGSL, `kumo` prefix,
  features compiled out when off. Profiles, smallest first:
  - `read`: value and active state at an index
  - `sample`: plus world-to-index and trilinear
  - `march`: plus HDDA and stats
  - `majorant`: plus a majorant DDA for delta tracking
- A WGSL size table per profile and grid-type set, measured as bytes and as compile time on one reference
  device, and guarded in CI like the pathtracer's `size-snapshot`.
- The GPU parity harness moves onto the module generator. A raw-WebGPU example with no engine is the reference
  for the engine contract.

**Exit gate:** the parity suite is green for every profile × {Float, Fp8, FpN} against the CPU twins; the size
table is committed; the `read` profile for Float meets the budget set in K1.1 (provisional target ≤ 4 KB, to be
confirmed against the pathtracer's numbers); and the raw-WebGPU example renders the smoke.

## K2 — three.js adapter on the core ⬜

Rebuild `./three` as a thin layer over K1, in the upscaler, denoiser and bvh style.

- `getDevice(renderer)` reads the renderer's device, the same helper the other tools use.
- `KumoGridNode` binds a `KumoGridPool` buffer into TSL, with a `.module` and `.update()`. `kumoSample()` is the
  lowercase TSL helper.
- The volume material, compute tools (`gridStats`, `valueTransform`, `decodeToAtlas`) and `NanoVDBSequence` are
  re-expressed on the core.
- The `setName` and one-token-rewrite binding trick from v1 is replaced by the slot-based module.

**Exit gate:** demos 01–07 run on the new adapter; the e2e suite is green; goldens are unchanged, or
re-baselined with a DIVERGES entry that says why.

## K3 — Pathtracer integration ⬜

Make KumoVDB the pathtracer's sparse VDB path (its volumes design doc, open problems 1, 2 and 5).

- Point sampling at an index or a world position, matching its convention: voxel centres, trilinear, zero
  outside.
- A majorant grid plus a DDA segment iterator for delta and ratio tracking. Build it both ways and A/B them:
  NanoVDB node max stats walked by HDDA, against its own 8³ brick majorants.
- Several grids per kernel through the grid table, each with its own world-to-index transform.
- Land it in `pmndrs/pathtracer` behind a setting, next to the dense texture path, and A/B the two.

**Exit gate:** `vol-vdb-smoke` agrees with its pbrt-v4 reference at the reference's noise floor; the sparse path
uses exactly one storage binding; the WGSL growth is within the pathtracer's budget; and a fog volume too large
to be dense renders.

## K4 — Tools and interop ⬜

CPU-side breadth, runnable in parallel from K0 since it touches no GPU code.

- Parser coverage: active tiles (fixes `bunny_cloud.vdb`), then vector grids, then point grids.
- `.wvdb` read and write per spec v0.5 (D14).
- A resample-to-resolution bake option.
- Coarse-first delivery for `.nvdb`: a low-resolution preview grid sent first.
- Keep uniform regions as tiles when quantizing, which closes the ~12% Fp8 size gap with `nanovdb_convert`.

**Exit gate:** `bunny_cloud.vdb` parses and round-trips through the CPU oracle; mjurczyk's committed `.wvdb`
files decode exactly per the spec; a `.wvdb` we write loads in his `WVDBLoader`; and the first paint of a
streamed `.nvdb` is no larger than his 14 KiB smoke preview.

## K5 — General volume renderer ⬜

An engine-free reference renderer on the core, with a three.js integration. It starts once R1 has written up
mjurczyk's `FrameRenderer`.

- HDDA empty-space skipping in the march, depth compositing with the scene, shadows, and a multiple-scattering
  approximation. Candidates are chosen from R1 and the papers it cites.
- Each technique is built behind a setting and A/B-tested (D11).

**Exit gate:** the WDAS quarter cloud at ≥ 60 fps at 1080p on the reference desktop (quiet pass); goldens for
each lighting mode; and an A/B against his renderer on the same smoke asset, recorded.

## K6 — Release 0.1 ⬜

- Docs in pmndrs/docs MDX format, with compiled snippets (`docs:check`, as in bvh).
- Examples for raw WebGPU, three.js and the pathtracer.
- A CHANGELOG, and a manual release workflow using npm trusted publishing (the bvh and upscaler model).

**Exit gate:** `@pmndrs/kumovdb@0.1.0` published, with docs live and the examples linked.

## Research track (R)

Work that is not on the critical path. Each item ends in a written note in `research/`, a DECISIONS entry, or a
promotion into a K phase.

| ID | Item | Feeds |
|---|---|---|
| R1 | Study mjurczyk's `FrameRenderer`: froxels, multiple scattering (diffusion, octaves, neural), spectral, mesh shadows, emission lights. Write it up as a design note; he has said agents wrote much of it. | K5 |
| R2 | WDAS perf on hardware: the pathtracer's notes record 103.96 fps at 1080p, which is not in this repo yet. Recover that run and the perf table. | K5 gate |
| R3 | `nanovdb_convert` value-parity sweep on a native machine (v1 Phase 5 checklist). | K4 |
| R4 | The companion service and its browser successors (D6; v1 phases 6b and 9). | later |
| R5 | WASM escalation: not triggered (D3). Keep watch only. | — |
| R6 | Other adapters: babylon, pixi (upscaler and bvh precedent). | after K2 |
| R7 | Bindless and texture-table grids, once a portable WebGPU feature exists. | K3 follow-up |
| R8 | PicoVDB: re-evaluate on the D16 triggers. | — |
| R9 | Retest the v1 sandbox workarounds on hardware: canvas present, `Data3DTexture`, the swizzle and keep-alive shims. | K2 |
| R10 | The joint project with mjurczyk: outreach, a `.nvdb` loader PR to `three-openvdb`, and whether to adopt the WebVDB name (D7). | owner |
