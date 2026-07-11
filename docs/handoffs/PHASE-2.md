# Handoff: Phase 2 — traversal core

Gate: **PASSED** — WGSL unit suite green: 18,000 random probes
(active/inactive/boundary, all 9 fixtures, Float+Fp8+FpN) plus 4,500
trilinear samples match the CPU reference with max |Δ| 6e-8 (tolerances
were 1e-5 / 1e-3). Full suite: 89 unit tests, 3 e2e specs, clean tsc.

## What exists now

| Artifact | Where | Proven by |
|---|---|---|
| Extended WGSL fork: audited baseline + Fp4/8/16/FpN decoders, typed dispatch (`readaccessor_get_value_float`), world↔index map, trilinear sampling (WebVDB extension), stats readers; HDDA + is_active were already complete | `packages/nanovdb-wgsl/vendor/pnanovdb.wgsl` (+ 6 diff-log rows in `VENDOR.md`) | compile-check script + parity harness |
| WGSL compile check (headless real device, fails on any diagnostic) | `scripts/check-wgsl-compile.mjs` | run in gate |
| GPU parity harness (raw WebGPU, no TSL) | `examples/src/harness/` + `e2e/gpu-parity.spec.ts` | the gate itself |
| CPU trilinear reference + deterministic splitmix64 probe/point generators (cross-language reproducible) | `packages/nanovdb-wgsl/src/cpu/sample-trilinear.ts`, `probe-coords.ts` | 12 tests incl. golden snapshots |
| Hardened loader (gridSize cross-check, bounded zlib, NanoVDB1 coverage) | `packages/nanovdb-wgsl/src/nvdb-file.ts` | 37 loader tests |

## Key facts for Phase 3 (cloud material)

- Line-by-line audit of the original port found **zero functional
  deviations** — two independent checks (demo 01 hand-port, harness
  re-transliteration) agree.
- Call surface for materials: bind the grid as
  `@group(G) @binding(B) var<storage, read> nanovdb_buffer : array<u32>;`,
  then `pnanovdb_make_buffer` → `grid_handle(0)` → `tree_get_root` →
  `readaccessor_init(&acc)` → `readaccessor_get_value_float(grid_type,
  buf, &acc, ijk)` / `pnanovdb_sample_trilinear_{float,fp8,fpn}(buf,
  &acc, xyz)` (index space). World↔index via
  `pnanovdb_grid_world_to_indexf` / `_index_to_worldf` (+ `_dirf`).
  HDDA: `pnanovdb_hdda_{init,step,update,ray_clip}` + `zero_crossing`.
  Stats: `pnanovdb_{root,upper,lower,leaf}_get_{min,max,ave,stddev}_float`.
- TSL binding (from Phase 1): `wgslFn` + `ptr<storage, array<u32>, read>`
  param works; three r185 takes an external GPUDevice.
- WebGPU trap: `bindGroupLayouts` is positional — a skipped `@group`
  index silently invalidates the dispatch (outputs stay zeroed). Wrap
  pipeline creation in `pushErrorScope("validation")`.
- SwiftShader timing: shader compile ~6 s once; per-fixture dispatch+
  readback 22–44 ms. Golden-image SSIM in CI is viable.

## Known debts (carried)

- `src/cpu/stride-tables.ts` reads JSON via `node:fs`, so the CPU
  reference isn't browser-importable; the harness carries its own
  transliteration (constants parsed from the WGSL text). Making the CPU
  module isomorphic would deduplicate ~2 files.
- FP4/FP16: decoders exist, no trilinear wrappers, no fixtures.
- 64-bit child offsets: fork inherits upstream's low-32-bit assumption
  (documented); revisit before multi-GiB grids.
- Legacy `NanoVDB0` magic ambiguity untested; ZIP fixtures synthetic;
  Docker bake image still unbuilt (no daemon in sandbox).
- WDAS cloud + EmberGen packs still undownloaded here (proxy-blocked in
  the authoring sandbox) — demo 02's WDAS gate needs a machine that can
  fetch them (`pnpm fixtures`), or the URLs verified by hand.

## Phase 3 entry points

- `packages/three-nanovdb`: implement `NanoVDBGrid` (grid image →
  StorageBufferAttribute + metadata → Box3/Matrix4), the
  `createVolumeRenderer` device-first bootstrap (D4; pattern proven in
  demo 01), and `NanoVDBVolumeMaterial` (fragment raymarch per SPEC
  §3.2: proxy-box entry, hdda_ray_clip, HG phase + sun shadow march).
- Demos 02 (cloud) + 03 (annotated gpu-read) with golden-image SSIM
  via Playwright screenshots (SwiftShader renders deterministically).
- Use baked primitive fixtures for CI; WDAS cloud is the hero asset on
  a network-open machine.
