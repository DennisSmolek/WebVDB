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
| K0.1 | Confirm the name, scope and repo home (D17) and the licence (D13) | owner | ✅ | D17 confirmed 2026-11-18. D13 (MIT) still open; v1 packages currently carry Apache-2.0 `LICENSE` files. |
| K0.2 | Tag `v1-final`, then restructure into `packages/vdb/src/{core,tools,three}`, one `tsc` build, published as `@pmndrs/vdb` (D20) | T2 | ✅ | Moves only. `git mv` to keep history. The nanovdb-wgsl loader and CPU oracle go to `tools`; the WGSL fork and constants go to `core`. |
| K0.3 | Boundary test: `core` never imports `three`; `tools` never imports `core` or `three` | T2 | ✅ | `packages/vdb/test/boundaries.test.ts` (no ESLint in the repo). |
| K0.4 | Build and exports: subpaths `.`, `./tools`, `./three`; build with dts; `verify:packed-subpaths` (upscaler) | T2 | ✅ | `pnpm verify:packed-subpaths` + CI step. Exports point at `dist/` only. Keep v1's source resolution (tsconfig `paths` + Vite/Vitest aliases); do **not** add a `development` export condition. |
| K0.5 | Rebrand README, STATUS and package READMEs to `@pmndrs/vdb`; add a "superseded by PHASES.md" banner to `docs/PLAN.md`, `docs/SPEC.md`, `docs/DECISIONS.md` | T1 | ✅ | |
| K0.6 | Apply the licence (D13) to code headers, LICENSE and NOTICE; keep the vendored Apache-2.0 files | T1 | ✅ | MIT applied; vendored Apache-2.0 files and NOTICE unchanged and shipped. |
| K0.7 | Move the repo to `pmndrs/vdb` (GitHub transfer keeps redirects) | owner | ⬜ | Timing is the owner's call; can follow K0.2. |
| K0.8 | Gate run, plus `docs/handoffs/K0.md` | T2 | ✅ | 249 unit / 11 e2e / goldens byte-identical / pack dry-run / packed-consumer install. Note the new tools path for the pathtracer's importer. |

## K1 — Engine-free core

Spec: [docs/design/K1-core.md](docs/design/K1-core.md) (§8 has the acceptance per task). Order: K1.3a → (K1.3b ∥ K1.4a) → K1.4b → K1.5 ∥ K1.6 → K1.7.

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K1.1 | WGSL size budget per profile | coord | ✅ | D21: budgets from measurement (K1-core §3.6). Compile-time column needs the owner's reference device. |
| K1.2 | Design note | T3 | ✅ | docs/design/K1-core.md |
| K1.3a | Pool layout: header + record encode/decode, `VDB_GPU_LAYOUT_VERSION`, WGSL record constants from one source | T3 | ✅ | Reviewed (Opus); fixes landed (pool lifecycle 616b850; generator fixes in K1.4b). |
| K1.3b | `VDBGridPool`: create/add/replace/remove/setTransform/update/version/revision/dispose; host-buffer mode | T3 | ✅ | e2e readback of nine fixtures byte-identical. Reviewed (Opus); fixes landed (pool lifecycle 616b850; generator fixes in K1.4b). |
| K1.4a | WGSL generator: chunk extraction from the vendored file, `read` + `sample` profiles, slot/buffer modes, `checkModule` | T3 | ✅ | 72/72 variants compile. Reviewed (Opus); fixes landed (pool lifecycle 616b850; generator fixes in K1.4b). |
| K1.4b | `march` profile (HDDA, stats) + the Fp leaf-stats fix in the fork; `tools` twins (`hddaWalk`, `nodeStats`) | T3 | ✅ | 132/132 variants compile; GPU probe bit-exact; review fixes folded in. Majorant parts deferred (D22). |
| K1.5 | Size snapshot + CI guard | T1 | ✅ | 132 variants; KiB budgets (one-Fp column re-baselined, D21 amendment). `pnpm wgsl:size --write` to update. |
| K1.6 | GPU parity harness on generated modules: every profile × Float/Fp8/FpN | T2 | ✅ | `e2e/core-parity.spec.ts`: 19 variants, one growing pool, 0 ulp on SwiftShader. Fresh-eyes verification passed (D11): independent reader agrees with sidecars and twins at 0 ulp. |
| K1.7 | Raw-WebGPU example: smoke rendered with no engine | T2 | ✅ | `examples/src/demos/08-webgpu-smoke`, `e2e/demo-08.spec.ts`. Goldens made on macOS SwiftShader: confirm on Linux CI. API feedback in docs/handoffs/K1-7.md. |
| K1.7a | Core API follow-ups from K1.7 (vdbModuleEntries visibility default, a trilinear-safe cell bound helper, HDDA cell-exit helper, types-up-front guidance) | T3 | ✅ | docs/handoffs/K1-7.md "K1.7a". |
| K1.9 | `readValue` (f64 decode) vs `readValueF32` (GPU order) differ by 1 ulp on Fp grids: document or align; make the parity gate fail (not skip) in CI when fixtures are missing | T2 | ⬜ | From the K1 verification. |
| K1.8 | Optional `reader: 'lean'` (≤ 4 KB `read`/Float), A/B vs vendored accessor | T3 | ⬜ | Not gating (D21). |

## K2 — three.js adapter, v1 parity

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K2.0 | Move the repo to three r187 (D23) | T2 | ✅ | No code changes needed; e2e unchanged. Peer range `>=0.187.0 <0.188.0`. |
| K2.1 | `getDevice(renderer)` and `getGPUTexture`, matching upscaler and denoiser so all tools share one device | T2 | ✅ | `three/device.ts`; see docs/handoffs/K2.md. |
| K2.2 | `VDBGridNode` (`.module`, `.update()`) and the `vdbSample()` TSL helper | T3 | ✅ | `three/grid-node.ts` + `vdbReadValue`; e2e `three-grid-node.spec.ts` (bit-exact vs f32 twins on SwiftShader, growth rebind, survives `renderer.dispose()`). v1 material still on the old trick until K2.3; K2.3 plan in docs/handoffs/K2.md. |
| K2.3 | Port the v1 volume material, `gridStats`, `valueTransform`, `decodeToAtlas` and the sequence player | T2 | ✅ | Part A ✅: `VDBVolumeMaterial` (+ `vdbProxyGeometry`), `VDBSequence` on `pool.replace`, `VDBGridNode` `helpers` option, `storageAttribute` gone, `wgsl.ts` a 2-symbol shim (docs/handoffs/K2.md "K2.3 part A"). Part B ✅: compute tools (docs/handoffs/K2-3B.md). `three/legacy-material.ts`, `three/wgsl.ts` and `NanoVDBGrid` deleted at K2 close (docs/handoffs/K2.md "K2 close"). |
| K2.4 | Port demos 01–07 and keep their goldens | T2 | ✅ | Part A ✅: 01, 02, 05, 07 ported; 03 kept as the hand-written raw-WebGPU teaching demo (comments point at the core). No snapshot changed. Part B: 04, 06. |
| K2.5 | Drop the sandbox-only workarounds from the default path (DIVERGES V7/V8) | T2 | ✅ | Part A ✅ for 02/05/07: direct canvas by default, `?presentation=readback` opt-in for e2e, swizzle/keep-alive shims deleted. Demos 04/06 (V8, shims) are part B's. `pnpm hardware:smoke` not rerun (needs hardware Chrome); its 02/05 checks were updated. |
| K2.6 | `decodeToAtlas`: sample voxel centres (re-baseline demo 04 golden + DIVERGES row) | T2 | ⬜ | Today texel `i` samples `min + (i + 0.5) / dims * size`, half a voxel past the voxel centres (v1's convention, documented in core `atlas.ts` and three `compute.ts`). Move to voxel centres (`min - 0.5 + (i + 0.5) / dims * size`) in the kernel and the CPU twin together. |
| K2.7 | Independent review fixes of `./three` | T3 | ✅ | One renderer per `VDBGridNode` and one VDB module per shader enforced at shader build; attribute self-test + r187 warning; own-device warning; host-encoder `job.submitted()` before `read()`; `VDBSequence` failed set + disposal generation; D17 names (`vdbGridStats`, `vdbDecodeToAtlas`; `createVolumeRenderer`, `computeRequiredLimits`, `nextPow2`, the three `valueTransform` gone); dump-path size check; no sample behind the eye. docs/handoffs/K2.md "K2 review fixes". |

## K3 — Renderer: port and extend

Plan: [research/notes/R1-renderer-port.md](research/notes/R1-renderer-port.md) §6 (D24). Pinned upstream: mjurczyk/openvdb `75e7345` (v0.26.0).

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K3.0 | Step 0: capture rig for his renderer (owner's machine) + dense `.wvdb` source | T2 | 🟨 | 2026-11-18, reduced scope (no third-party execution): dense source (tools/core/three) byte-identical to a static re-implementation of his upload path, our-side harness + deterministic e2e, `scripts/k3/compare.mjs`, `capture-reference.mjs` written, not run. [docs/handoffs/K3-0.md](docs/handoffs/K3-0.md). Open: K3.0a. |
| K3.0a | Owner: run `scripts/k3/capture-reference.mjs` on a hardware GPU; check the manifest's sanity rows (his own JPEG bar) | owner | ⬜ | Steps in the script header and the K3.0 handoff. Also load our `.wvdb` files in his `WVDBLoader` (K4.3 owner check). |
| K3.1 | Step 1: froxels with `'volume'` lighting on the dense source; strict parity (PSNR ≥ 40 dB) | T3 | ⬜ | Brief: docs/handoffs/K3-0.md §Step 1. Needs K3.0a captures for the gate. |
| K3.2 | Step 2: sparse `VDBGridPool` source + LOD A/B (none / coarse pyramid / node average); HDDA in the inject march | T3 | ⬜ | Needs K1.4b `march`. |
| K3.3 | Step 3: `'frame'` lighting (shadow grid, cascades, transmittance, temporal, env) | T3 | ⬜ | |
| K3.4 | Step 4: phase functions + multiple-scattering modes (two-sided, octaves, neural) | T3 | ⬜ | Neural weights pending R10. |
| K3.5 | Step 5: spectral media + presets | T2 | ⬜ | |
| K3.6 | Step 6: mesh shadows + emission (three internals, r187) | T3 | ⬜ | |
| K3.7 | Step 7: level sets + points | T3 | ⬜ | Needs K4.2. |
| K3.8 | Lift each green step's shading into `core` WGSL with CPU twins | T3 | ⬜ | Rolling, after each step. |
| K3.9 | Gate run: WDAS quarter cloud on sparse at a named preset ≥ 60 fps 1080p; goldens per lighting mode; A/B record | T2 | ⬜ | Timing part deferred to a quiet machine (D25); goldens and A/B image metrics still gate. |

## K4 — Tools and interop

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K4.1 | Parse active tiles (`bunny_cloud.vdb`) | T3 | ✅ | Also added a pure-TS Blosc1 decoder (bunny_cloud is LZ4-compressed). 0 oracle mismatches over 19.2M voxels. Reviewed; Blosc/zlib hardened against hostile input (29 tests). mjurczyk's MIT reader is a reference; attribute anything ported. |
| K4.2 | Vector grids, then point grids, then level-set specifics | T3 | ✅ | Reviewed + hardened (38 hostile-input tests). K4.7 fixed in the same change. | Level-set inactive-value and tile work split out to K4.8 and K4.9. Vector and point grids still rejected by the parser. |
| K4.3 | `.wvdb` read and write per his spec (v0.5, plus points from his v0.25.0) | T3 | ✅ | Prerequisite for K3.0. A decoder written from the spec already worked in the comparison report. Owner check: load our files in his `WVDBLoader` examples page; byte-identity vs his CLI under the same Node zlib. |
| K4.4 | Resample-to-resolution bake | T2 | ✅ | `resampleGrid` (point / box (default) / trilinear, sparse, anisotropic `dimensions`, optional Fp8/FpN) and `resampleError`. Smoke at his 128³ with FpN: box 477 KiB / 18.3%, point 456 KiB / 17.3% (report: 474 KiB / ~18%). [docs/handoffs/K4-4.md](docs/handoffs/K4-4.md). |
| K4.5 | Coarse-first `.nvdb` delivery | T3 | ✅ | Design done: [docs/design/K4-5-coarse-first.md](docs/design/K4-5-coarse-first.md). Recommends a preview grid first in the same `.nvdb` (smoke first paint 11.9 KiB vs his 14.8 KiB at equal error). Implemented in K4.16 (+ K4.18 ZIP writes); native check K4.17; filter A/B K4.19. |
| K4.6 | Quantize keeps uniform regions as tiles (the ~12% Fp8 gap) | T2 | ⬜ | K4.1 made quantize keep source tiles: Fp8 sizes now equal native on the primitives. What remains: collapse uniform leaves in our own builds. |
| K4.7 | Parse OpenVDB 12.1 half-float grids that omit empty value arrays (`bunny.vdb`, `icosahedron.vdb`, `fire.vdb` fail) | T2 | ✅ | Found in K4.1. |
| K4.8 | Level sets: inactive leaf values are ±background, not 0 (`compression.ts`) | T2 | ⬜ | Changes leaf bytes; check demo goldens. |
| K4.9 | Keep inactive internal tiles (level-set interior −background) | T2 | ⬜ | |
| K4.10 | Native reference values for `bunny_cloud` (pyopenvdb or `nanovdb_convert` via `run-native-parity`) | owner | ⬜ | Needs Docker. Bunny derivatives are outside the pinned-SHA manifest (`fixtures/ATTRIBUTION.md`). |
| K4.11 | Document FpN size vs native (ours ~1.85× smaller: different tolerance) and decide | T2 | ⬜ | |
| K4.12 | Root-tile key order: sort unsigned like NanoVDB keys | T1 | ⬜ | |
| K4.13 | `.wvdb` follow-ups: prefiltered resample option (box/area; his bake is point-sampled, A/B per CLAUDE.md), port `sampleLevelSet`, page-by-page decode within a LOD (show a LOD before all its pages arrive, §12.2) | T2 | ⬜ | Wording from [docs/handoffs/K4-3.md](docs/handoffs/K4-3.md) (Follow-ups). |
| K4.14 | Parser hardening leftovers: reject file versions > 225 (as OpenVDB does); `readMetadataMap` must not write `__proto__` into a plain object (use a null-prototype object or Map) | T2 | ⬜ | From the K4.2 review. |
| K4.15 | Independent oracle for point attributes (Cd, v, id per point via OpenVDB Python bindings on waterfall_points) and native Vec3f parity | owner | ⬜ | Needs pyopenvdb / Docker. |
| K4.16 | Implement coarse-first (K4.5 option a): `NvdbStreamReader` (push bytes, emit each grid when complete), a bake helper writing `[<name>.lod48, <name>]`, an example on `pool.add(preview, { capacity })` → `pool.replace`, and an e2e that counts compressed bytes at preview emission (gate: ≤ 14.8 KiB on smoke) | T2 | ✅ | `bakeCoarseFirst`, `NvdbStreamReader` / `readNvdbStream`, `vdbStreamCoarseFirst` (`./three`), e2e `coarse-first.spec.ts`. Smoke first paint: 12.3 KiB with ZIP and no HTTP compression, 12.0 KiB with codec NONE + gzip; final frame equals the full grid alone. `writeNvdb` and `NanoVDBFile` now follow NanoVDB's `io::Segment` layout for multi-grid files. [docs/handoffs/K4-16.md](docs/handoffs/K4-16.md). |
| K4.17 | Native check of a multi-grid preview `.nvdb`: `nanovdb_print`, `nanovdb_convert`, a name-based `io::readGrid`, and EmberGen; record which grid grid-0 readers show | owner | ⬜ | Needs Docker (parity image). Fallback if a grid-0 reader matters: K4.5 option (b), separate preview file. |
| K4.18 | `writeNvdb` with `Codec.ZIP` per grid segment (our reader decodes it), so first paint doesn't depend on host compression | T2 | ✅ | `writeNvdb(images, { codec: 'zip', zipLevel })`, `io::Internal::write` framing. Smoke coarse-first: 666.3 KiB ZIP vs 1,693.3 KiB raw (654.5 KiB raw + gzip 9). Native read of our ZIP files is part of K4.17. |
| K4.19 | Preview filter A/B: render box vs point (vs trilinear) previews against the full-res render (PSNR) and under a half-voxel box shift | T3 | ⬜ | Pointwise error favours point at small ratios (smoke 128³: 17.3% vs 18.3%); box conserves mass. Decide the preview default from this. |
| K4.20 | Coarse-first follow-ups: a demo page (06 explorer or a new one) that streams a coarse-first file over the network with a live preview → full swap; `NvdbStreamReader` progress (`bytesTotal` from `FileMetaData.fileSize`); decide whether `NanoVDBFile` errors become `VdbFormatError` like the stream reader's | T2 | ⬜ | From [docs/handoffs/K4-16.md](docs/handoffs/K4-16.md). |

## K5 — Release

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| K5.1 | pmndrs/docs MDX site with compiled snippets | T2 | ⬜ | |
| K5.2 | Release workflow: manual, npm trusted publishing, CHANGELOG | T1 | ⬜ | |
| K5.3 | Cloud-system showcase example | owner | ⬜ | |

## After 0.1

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| P1 | Pathtracer integration: point-sample API, the `majorant` profile + brick builder (D22), majorant DDA two ways (node-max HDDA vs 8³ bricks), several grids per kernel, landed behind a setting | owner | ⬜ | D19. Update its importer (`bench/scenes/volumes/vdb-grid.ts`) to the K0 layout first. |

## Research

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| R1 | Renderer port plan, written to `research/notes/R1-renderer-port.md` | T3 | ✅ | Read his source statically; don't run it in the cloud sandbox. Track his releases (0.26.0 on 2026-11-18). |
| R2 | WDAS perf on hardware | owner | ✅ | 2026-07-12; `docs/handoffs/PHASE-3.md`. |
| R3 | Native `nanovdb_convert` parity sweep | owner | ✅ | 2026-07-12; `pnpm parity:native`, `docs/handoffs/PHASE-5.md`. |
| R9 | Retest the v1 sandbox workarounds on hardware | owner | ✅ | 2026-07-12; follow-up is K2.5. |
| R10 | Outreach to mjurczyk: invite him in, offer the `.nvdb` loader PR; nudge him to add the LICENSE file (not blocking: it's MIT). Also confirm the neural-scattering weights are MIT, and note his `'approximate'` blackbody cites a Shadertoy (default CC BY-NC-SA) — we won't port that function verbatim | owner | ⬜ | Comparison report: https://claude.ai/artifact/FV7TNDJVmhuLxynaPf8X9o |
