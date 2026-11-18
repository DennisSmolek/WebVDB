# Handoff: Phase 5 — CPU tools v1 (pure TS)

Gate: **PASSED, including the native parity anchor** — drag-drop a `.vdb` and render it
(demo 06 core) and demo 07 dense→TS-build→render round-trip both ship,
e2e-gated with byte-stable goldens. The native `nanovdb_convert`
byte/value parity anchor now passes on all four samples in FLOAT and
FP8 modes.

Current suite: 249 unit / 11 E2E functional gates / clean tsc. On macOS,
10/11 also match the canonical Linux snapshots; Demo 06 has the documented
3% cross-OS image diff.

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

## Native parity result (2026-07-12)

Run `pnpm parity:native` after building `webvdb-fixture-bake`. The command
verifies raw source size/SHA-256, normalizes both conversion inputs, performs native and TypeScript
FLOAT/FP8 conversions, independently enumerates both NanoVDB trees, merges the
union of their active coordinates, and writes `fixtures/parity/report.json`.

Pinned source SHA-256 values recorded in that report:

- `sphere.vdb`: `27c266158458b52432795f6d8e72ac48524a65a8d2dc1721339ba287cdc2dba4`
- `cube.vdb`: `e5ed2748c07bc64aa38b6df6812008bed840d7734ce460ad447d0f86ef5e6da4`
- `smoke.vdb`: `09e5727dab14d1ef46532b9b731a8d2de4c1de333a97c443516fd8ae8ca8c03b`
- `utahteapot.vdb`: `3808c6f3a6261f8b46ea156aa808186f8f615d5e96ee8160ac59855de156f1c9`

| Sample | Mode | Union coordinates | Native-only | TS-only | Numeric failures | Topology | Result |
|---|---|---:|---:|---:|---:|---|---|
| sphere | FLOAT | 270,638 | 0 | 0 | 0 | exact | PASS |
| sphere | FP8 | 270,638 | 0 | 0 | 0 | exact | PASS |
| cube | FLOAT | 1,452,218 | 0 | 0 | 0 | exact | PASS |
| cube | FP8 | 1,452,218 | 0 | 0 | 0 | exact | PASS |
| smoke | FLOAT | 1,049,275 | 0 | 0 | 0 | exact | PASS |
| smoke | FP8 | 1,049,275 | 0 | 0 | 0 | exact | PASS |
| utahteapot | FLOAT | 6,960,047 | 0 | 0 | 0 | exact | PASS |
| utahteapot | FP8 | 6,960,047 | 0 | 0 | 0 | exact | PASS |

“Topology exact” means both outputs independently produced the same canonical
root keys/states, upper/lower node origins and child/value masks, leaf origins
and active masks, node counts, and active-voxel count. Reachability is checked
against every declared physical node, and normalized parity rejects any
remaining active root/internal tile. The active-coordinate merge therefore
detects native-only and TypeScript-only voxels rather than trusting the source
VDB as the coordinate oracle.

FLOAT values are exact at every coordinate in that union. FP8 compares the two
independently quantized images against a finite per-value bound: half the sum of
both encoded leaf quantums plus float32 roundoff. Every differing byte offset
is mapped through the proven node layout and masks before it can be classified.
The current report contains only checksum, derived-statistic, mask-proven
inactive internal/leaf payload, FP8 quantization-parameter, and bounded active
FP8-code fields. It records category counts, first offsets, and a SHA-256 of the
complete differing-offset sequence; unknown fields, structural differences,
or unbounded active-code differences fail.

This is intentionally not a claim that every inactive decoded value is equal.
The previous broad `near-field/root-background` label was removed: no inactive
value mismatch is relabeled semantically. The evidence is limited to exact
topology/all active coordinates plus byte-field classification where the masks
prove that a payload is inactive.

The corrected pinned `smoke.vdb` and `utahteapot.vdb` parse directly in the
pure-TypeScript fixture tests. The Docker `normalize_vdb` helper still writes a
common active-mask-only representation before both parity conversions so
compression/tile storage does not bias one implementation.

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
Phase 6b (companion service) — native Docker toolchain is validated; a hosted
service remains optional.
Phase 7 (sequences) — complete on hardware with a 100-frame EmberGen manifest
and the material grid-rebind API.
Phase 8 (explorer polish, docs site, npm dry-run) — explorer core done
early; polish items remain.
Native/hardware items — complete; see PHASE-3.md and the native result above.
