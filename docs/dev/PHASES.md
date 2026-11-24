# Phases

`@pmndrs/vdb` plan v2, set 2026-11-18 and revised the same day (D17–D19). It replaces v1 phases 0–9
([docs/archive/v1/PLAN.md](../archive/v1/PLAN.md), frozen by D15). v1 delivered the parts these phases rearrange: the loader, the
audited WGSL fork, the CPU oracle, the three.js material, the compute tools, the TS parser and serializer, and the
sequence player (see [docs/archive/handoffs/](../archive/handoffs/)). Its on-device gates closed on 2026-07-12
([docs/archive/v1/ON-DEVICE.md](../archive/v1/ON-DEVICE.md)).

**First consumer:** the owner's cloud system. **After our release:** the pmndrs pathtracer (D19).

Status: ⬜ not started · 🟨 in progress · ✅ done. Each phase ends at a machine-checkable **exit gate** plus a
handoff in `docs/archive/handoffs/K<n>.md`. Task IDs (`K1.3`, `R2`) live in [TASKS.md](TASKS.md).

```
K0 re-foundation ──► K1 core ──► K2 three adapter ──► K3 renderer ──► K5 release 0.1 ─┬─► K6 renderer in the core (pure WebGPU)
                                  (v1 parity)       (port + extend)                    ├─► K7 upstream sync (OpenVDB 14, mjurczyk 0.29)
        └──► K4 tools + interop  (CPU only, runs in parallel from K0)                  └─► P1 pathtracer integration
Research: R1 (renderer port plan) runs during K1/K2 and feeds K3.
```

**After 0.1 (planned 2026-11-24).** Three tracks run side by side: K6 moves the renderer out of three.js into the
engine-free core, K7 keeps us current with OpenVDB/NanoVDB and mjurczyk/openvdb (the goal: every file native tools
write opens here), and P1 brings the grid pool into pmndrs/pathtracer. They touch different code: K6 adds renderer
passes to `core`, K7 the tools and the WGSL type table, P1 a `majorant` profile and a lean reader. K7.3 (multi-scale
scattering) lands before K6.3 so it is written once, in the core.

## K0 — Re-foundation ✅

Done 2026-11-18 ([docs/archive/handoffs/K0.md](../archive/handoffs/K0.md)), except the repo move to `pmndrs/vdb` (K0.7, owner). Licence applied: MIT (D13).

Rename to `@pmndrs/vdb` (D17) and reshape the repo into core plus adapters (D8). **No behaviour changes:** this
phase moves code, it does not rewrite it.

- Workspace (D20): one package, `packages/vdb`, with `src/core` (WebGPU/WGSL), `src/tools` (CPU: the loader,
  parser, serializer, CPU oracle; from `vdb-web-tools` plus the CPU half of `nanovdb-wgsl`) and `src/three`
  (from `three-nanovdb`), built by one `tsc`. Published as `@pmndrs/vdb` with subpaths `.`, `./tools`, `./three`.
- A boundary test: `core` never imports `three`; `tools` never imports `core` or `three`.
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

## K1 — Engine-free WebGPU core ✅

Done 2026-11-18 ([docs/archive/handoffs/K1.md](../archive/handoffs/K1.md), K1-4a, K1-6, K1-7): pool, generator (`read`/`sample`/`march`),
size snapshot, GPU parity gate (verified fresh-eyes), raw-WebGPU example. Compile-time column deferred (D25);
K1.7a API polish and K1.8 lean reader open.

Everything a WebGPU host needs to hold and read sparse grids, with no engine (D8, D10).

- `VDBGridPool.create(device, options)`: one packed storage buffer with a grid table and N NanoVDB images,
  `add()` / `remove()`, `update(encoder)`, a `version` counter for re-binding, and `dispose()`.
- `vdbModuleWgsl({ slot, types, profile })` and `vdbModuleEntries()`: generated WGSL, `vdb` prefix, features
  compiled out when off. Profiles, smallest first:
  - `read`: value and active state at an index
  - `sample`: plus world-to-index and trilinear
  - `march`: plus HDDA and stats
  - `majorant`: a majorant DDA for delta tracking — **deferred to after 0.1** (D22); the record keeps its
    fields. Design: [docs/dev/design/K1-core.md](./design/K1-core.md) §5
- A WGSL size table per profile and grid-type set, measured as bytes and as compile time on one reference
  device, and guarded in CI.
- The GPU parity harness moves onto the module generator. A raw-WebGPU example with no engine is the reference
  for the engine contract.

Design inputs, not gates: the pathtracer's published constraints (Apple Chrome allows 10 storage buffers per
stage; its kernels are 41–55 KB of WGSL) are why the contract is one binding and features compile out.

**Exit gate:** the parity suite is green for every profile × {Float, Fp8, FpN} against the CPU twins; the size
snapshot is committed and every variant is within the budgets of K1-core §3.6 (D21; the ≤ 4 KB lean reader is
the optional K1.8); and the raw-WebGPU example renders the smoke.

## K2 — three.js adapter, v1 parity ✅

Done 2026-11-18 ([docs/archive/handoffs/K2.md](../archive/handoffs/K2.md), K2-0, K2-3B): three r187; `getDevice`, `VDBGridNode`,
`vdbSample`/`vdbReadValue`; `VDBVolumeMaterial`, `VDBSequence`, compute on the core; demos 01–07 on the new
adapter with goldens unchanged; v1 binding trick and sandbox workarounds removed (V7, V8 closed).

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

## K3 — Renderer: port and extend ✅

Parity (2026-11-19, captures of his renderer at 75e7345 on the owner's Apple GPU): all 7 gating scenes (Steps 1 and 3,
isolated) pass the strict bar; ours on hardware is bit-identical to his in 6/7 (PSNR ∞), two-bunnies 52.6 dB.
Done 2026-11-19. Parity against his renderer at 75e7345 on the Apple GPU (Chrome, metal-3): all 48 gating scenes
(Steps 1, 3, 4, 5, 6) pass the strict bar; 46/48 bit-identical (white smoke max 1/255; two-bunnies 52.6 dB, his
timer-driven second bunny). Step 7: level-set silhouette IoU 0.994 / 0.996, point count 9,802 = his. Timing gates
deferred (D25); HDDA off (D30/D31).

Compete on rendering (D18). Bring mjurczyk's renderer (three-openvdb, MIT) in as the basis of our three.js
renderer, re-based on the sparse core, and lift its engine-independent parts into shared foundations that every
adapter uses.

- **Port plan:** [docs/dev/research/notes/R1-renderer-port.md](research/notes/R1-renderer-port.md) (done). His renderer is
  a froxel pipeline written in TSL; density is read in one place (`createVolumeNodes()`), by four materials.
- **Volume source interface:** the renderer reads density through one interface with two sources: the sparse
  `VDBGridPool` (ours) and dense `.wvdb` levels (his format, D14). Users of his renderer keep their assets.
- **Port in order (D24),** each step A/B-tested against his renderer at the pinned commit on the same asset
  (D11): 0 dense-source harness (needs K4.3 `.wvdb`) → 1 froxels with `'volume'` lighting → 2 sparse source +
  LOD A/B → 3 `'frame'` lighting → 4 phase + multiple scattering → 5 spectral → 6 mesh shadows + emission →
  7 level sets + points.
- **Sparse advantages:** HDDA empty-space skipping in the march (deferred since v1), and full-resolution hero
  assets with memory proportional to active voxels. The hard part is LOD: his shadow grid samples prefiltered
  mips, so the sparse source needs a coarse pyramid or node averages (A/B in Step 2).
- **Shared foundations:** the shading WGSL lives in `core` (engine-free), so later adapters (R6) and the
  owner's cloud system reuse it.
- Every ported file carries attribution; every departure gets a DIVERGES row.

**Exit gate** (timing items deferred to a quiet machine, D25): his demo scenes render on our sparse source within an agreed image-diff threshold of his output;
the WDAS quarter cloud runs at ≥ 60 fps at 1080p on the reference desktop **at a named quality preset** (v1's
104 fps used `stepSize=4`, `shadowSteps=4`; its quality preset ran at 16.8 fps); goldens per lighting mode; and
the A/B results are recorded.

## K4 — Tools and interop ✅

Done 2026-11-18 except owner-machine checks (K4.10, K4.15, K4.17: native OpenVDB/NanoVDB oracles; our `.wvdb` in his
`WVDBLoader`). Exit gate met: `bunny_cloud.vdb` parses (Blosc, tiles) and round-trips through the oracle; his `.wvdb`
files decode exactly; coarse-first `.nvdb` first paint 12.3 KiB on smoke (gate 14.8). Also: vector and point grids,
level-set inactive values/tiles, resample bake, collapse, hostile-input hardening throughout.

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

## K5 — Release 0.1 🟨

`@pmndrs/vdb@0.1.0` published 2026-11-24 (manual first publish from 5328b27, tag `v0.1.0`; trusted publishing set up
for later releases). Docs live at https://pmndrs.github.io/vdb/ since 2026-11-24 (Pages deploys on every push to
`main`), linked both ways with the examples (K5.7 close-out). **Remaining for the exit gate:** the examples live, which
needs the owner's asset release (`examples-assets-v1`, RELEASING §4). K5.3 cloud-system showcase moves to after 0.1.

- Docs in pmndrs/docs MDX format, with compiled snippets (`docs:check`, as in bvh).
- Examples for raw WebGPU and three.js, plus the owner's cloud system as the showcase consumer.
- A CHANGELOG, and a manual release workflow using npm trusted publishing (the bvh and upscaler model).

**Exit gate:** `@pmndrs/vdb@0.1.0` published from `pmndrs/vdb`, with docs live and the examples linked.

## K6 — Renderer in the core (pure WebGPU) 🟨

**Status 2026-11-24:** K6.0 design written ([design/K6-core-renderer.md](design/K6-core-renderer.md), D34 draft in §10);
waiting on the owner's answers to its open questions Q1–Q10 before K6.1 starts.

The froxel renderer, level-set tracer and points live in `./three` as TSL today. K6 moves them into the engine-free
core as WGSL passes, as D8 intended: the core takes the host's `GPUDevice`, records into the host's encoder and
renders into host targets; cameras, lights and shadow maps come in as plain data. `./three` becomes a thin adapter
over those passes, and other engines (R6: babylon, pixi) can follow.

- Design first (K6.0): the pass graph, resource ownership, how the `vdb` module composes with renderer WGSL (one
  module per shader), a core-side shadow-map interface (the host gives the light's view-projection and depth; this
  also removes the class of three-specific shadow bugs K7.1 fixes), and the seam for host post-processing (MRT).
- Port pass by pass, each with a CPU twin and a pixel A/B against today's TSL path, in the K3 step order.
- The K3 parity captures (his renderer at `75e7345`) keep gating: the three adapter over core passes must still pass
  48/48.

**Exit gate:** a raw-WebGPU demo renders the WDAS cloud with 'frame' lighting, mesh shadows and emission without
importing three; `./three` draws through the core passes with the K3 parity suite and demo goldens unchanged; every
core pass has CPU-twin parity; `test/boundaries.test.ts` still holds.

## K7 — Upstream sync 🟨

**Status 2026-11-24:** K7.H (NanoVDB Half grids) and K7.C (v14 codecs and parser, native v14 fixtures committed) done.
Next: K7.1–K7.3 (his shadow fixes, post-processing/MRT, multi-scale scattering), K7.5, K7.6. Open: K7.4 (issue #1),
K7.C1, K7.H1, K7.7.

Since 0.1 OpenVDB shipped 13.1, 13.2 and 14.0, and mjurczyk/openvdb 0.27–0.29 (research 2026-11-24, TASKS K7). The
NanoVDB file format and PNanoVDB layout are unchanged, but NanoVDB 13.2 added Half grids, Blosc is on by default in
NanoVDB builds, and v14 tightened point-attribute checks. His releases fixed volume shadows and post-processing and
added `'multi-scale'` multiple scattering; his new defaults stay out for now (issue #1, V67).

**Exit gate:** files written by native OpenVDB/NanoVDB 14 tools (a Half `.nvdb`, a Blosc `.nvdb`, a HalfGrid `.vdb`)
read and render here with native parity; his post-0.26 fixes ported with tests; an upstream watch opens an issue
when OpenVDB, NanoVDB or mjurczyk/openvdb release.

## P1 — Pathtracer integration ⬜

**Status 2026-11-24:** planned from its code; starts when the pathtracer's volume phase closes. Our side first (P1.1–P1.6).

Owner-led on the pathtracer side (D19); our side builds what it needs in the core. Inputs read from pmndrs/pathtracer
on 2026-11-24 (`docs/internal/design/volumes-plan.md`, its V244/V308): its volume phase (V0–V6) is closing; it
renders grids from a dense texture or its own sparse brick atlas, with `global` and 8³ `brick` majorants (with
minima), one grid per kernel, compute only, at most 10 storage buffers per stage on Apple, and a ~50 KB kernel cap
for DXC. It imports only our `parseVdb` today, by path; 0.1.0 breaks nothing.

So NanoVDB has to earn its place in its kernel: several grids at one binding, direct `.nvdb`, Fp8/FpN compression,
multi-level sparsity. Build both (the rule) and A/B: NanoVDB in its kernel against our tools building its atlas.
Prerequisites on our side: the point-sample contract, a brick-majorant builder and the `majorant` profile (D22), and
the lean reader (K1.8), since our 19.8 KB `sample` module would push its media kernels past the DXC cap.

**Exit gate:** `vol-vdb-smoke` through `@pmndrs/vdb` agrees with the pbrt-v4 reference at the noise floor (4096 spp
relMSE ≤ 4.84e-5) with exactly one storage binding for grid data, behind a setting next to its dense path; the
majorant A/B recorded in DECISIONS.

## Later

- **Other adapters** (R6): babylon, pixi, on the K6 core passes.

## Research track (R)

Work that is not on the critical path. Each item ends in a written note in `research/`, a DECISIONS entry, or a
promotion into a K phase.

| ID | Item | Status | Feeds |
|---|---|---|---|
| R1 | Port plan for mjurczyk's renderer | ✅ 2026-11-18: [docs/dev/research/notes/R1-renderer-port.md](research/notes/R1-renderer-port.md), pinned at `75e7345` (v0.26.0) | K3 |
| R2 | WDAS perf on hardware | ✅ 2026-07-12: 103.96 fps at 1080p with `stepSize=4`, `shadowSteps=4`; table in `docs/archive/handoffs/PHASE-3.md` | K3 gate |
| R3 | `nanovdb_convert` parity on a native machine | ✅ 2026-07-12: strict FLOAT/FP8 parity on all four pinned sources (`pnpm parity:native`) | K4 |
| R4 | The companion service and its browser successors (D6; v1 phases 6b and 9) | ⬜ | later |
| R5 | WASM escalation: not triggered (D3). Keep watch only. | — | — |
| R6 | Other adapters: babylon, pixi (upscaler and bvh precedent) | ⬜ | after K5 |
| R7 | Bindless and texture-table grids, once a portable WebGPU feature exists | ⬜ | pathtracer |
| R8 | PicoVDB: re-evaluate on the D16 triggers | — | — |
| R9 | Retest the v1 sandbox workarounds on hardware | ✅ 2026-07-12: direct presentation, stock 3D texture and no-shim lanes pass (`pnpm hardware:perf`) | K2 |
| R10 | Bring mjurczyk into the project: outreach, the `.nvdb` loader PR offer (and a nudge to add the LICENSE file) | ⬜ owner | K3 |
