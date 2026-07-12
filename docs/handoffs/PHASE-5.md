# Handoff: Phase 5 — CPU tools v1 (pure TS)

Gate: **PASSED in CI scope** — drag-drop a `.vdb` and render it
(demo 06 core) and demo 07 dense→TS-build→render round-trip both ship,
e2e-gated with byte-stable goldens. The `nanovdb_convert` byte/value
parity anchor is deferred to a native machine (below) — but the
strongest available in-sandbox evidence already exists: our Fp8
re-encode of the native float sphere matches the native fp8 fixture
**exactly** at all 73 sidecar coords, and FpN per-leaf bit-width
selection matches native's oracle leaf-for-leaf.

Suite at close: 206 unit / 10 e2e / clean tsc.

## vdb-web-tools v1 API (all pure TS, zero deps, zero wasm — D3)

| API | Status / evidence |
|---|---|
| `parseVdb` | 4 openvdb.org samples (v222, 5-4-3 float trees, zlib, half-float); activeVoxelCount matches file metadata exactly; own RFC1950/1951 inflate validated vs node:zlib |
| `buildFromDense` / `buildFromLeavesDetailed` | full-voxel round-trip via proven CPU reader; structural checker passes on native fixtures |
| `buildFromVdb` | leaf-streamed (7M-voxel teapot, no dense alloc); zero mismatches on all samples; uniform scale+translate Maps |
| `quantize` fp8/fpn | native-exact rounding (dither-off 0.5), per-leaf min/quantum, FpN width oracle match; active tiles expanded to constant leaves (documented) |
| `transform` | Map/worldBBox rewrite on a copy, tree bytes untouched; rotation/shear → clear throw + GPU-resample pointer (D6) |
| `inspect` | node counts + memory breakdown summing to mGridSize, validated vs sidecars |
| `writeNvdb` | NanoVDB2 segment files, loader round-trip byte-equal |
| `readNvdb` | intentional stub — `nanovdb-wgsl`'s NanoVDBFile covers reading |

## Native-machine checklist (the deferred anchor)

1. Build `docker/fixture-bake/` → `nanovdb_convert` available.
2. `nanovdb_convert` the four vdb-samples to float+fp8 `.nvdb`; diff
   value-wise (CPU readValue sweep) against `buildFromVdb`+`quantize`
   output. Expect value parity; byte parity will differ in three known
   ways: active-tile compression, CRC (we write disabled sentinel),
   trailing padding.
3. Known semantic edge: native writes root background ≠ near-field
   inactive-tile fill on some grids (observed 3 vs 0 on the sphere
   fixture); our leaf-only relayout uses the effective near-field value.

## Known debts

- `nanovdb-wgsl/src/cpu/*` still node-only (`node:fs` JSON load) —
  pages use the harness transliteration; making the CPU module
  isomorphic would deduplicate three copies now (harness, demo 04/06/07
  usage). Worth doing before the explorer grows.
- Explorer wireframes show real leaves only (uniform active tiles not
  expanded); slice view is CPU-only; multi-grid files show grid 0.
- Demo 06 histogram GPU path needs a device; CPU fallback samples 100k.

## What remains on PLAN after this phase

Phase 6 (WASM escalation) — not triggered; no demonstrated need.
Phase 6b (companion service) — untestable here (no Docker daemon).
Phase 7 (sequences) — needs EmberGen assets (network-open machine) +
the material grid-rebind API flagged in PHASE-3.md.
Phase 8 (explorer polish, docs site, npm dry-run) — explorer core done
early; polish items remain.
Native/hardware items — PHASE-3.md (WDAS perf), above checklist.
