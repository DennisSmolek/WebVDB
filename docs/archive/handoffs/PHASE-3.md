# Handoff: Phase 3 — cloud material

Gate: **PASSED, including the 1080p hardware performance gate.**
Demos 02 + 03 ship and are e2e-gated: two-pose golden images byte-stable
across consecutive runs, demo 03 rated *publishable* by an independent
editorial review that checked every byte-offset claim against upstream.
Suite at close: 124 unit / 6 e2e / clean tsc.

## 1080p WDAS hardware gate (2026-07-12)

`pnpm hardware:perf` passed in stable Chrome 149 on the default Apple M5 Pro
GPU (`apple metal-3`) at an explicit 1920×1080 viewport. The selected fixture
was `wdas_cloud_quarter_fp8.nvdb`, rendered by the direct WebGPU canvas path.

| Preset | FPS | Frame | Shadow=0 frame | Shadow delta | Result |
|---|---:|---:|---:|---:|---|
| `stepSize=4`, `shadowSteps=4`, `sampleBudgetCap=8192` | 103.96 | 9.62 ms | 7.97 ms | 1.65 ms | PASS |

The original `stepSize=0.75`, `shadowSteps=12` preset measured 16.77 fps
(59.62 ms/frame), so it was not retained for the 1080p gate. A refreshed
`stepSize=3`, `shadowSteps=4` run measured 52.27 fps, so the default gate now
uses `stepSize=4`; its screenshot remains a coherent full cloud without
clipping. Evidence is in ignored `artifacts/hardware-report.json` and
`artifacts/hardware/demo-02-wdas-1080p.png`.

The gate now requires the published Demo 02 state to identify the WDAS FP8
fixture, `apple metal-3` adapter, direct presentation, both shims disabled,
1920×1080 resolution, and the exact three quality settings above. A 60 FPS
result alone is not sufficient.

Three r185 timestamp queries were enabled and Chrome granted the feature, but
the returned durations were larger than the queue-synchronized wall time by
orders of magnitude. Demo 02 now rejects those impossible values and records
`gpuTimestampValid=false`; the table therefore uses queue-synchronized frame
time and an alternating `shadowSteps=0` controlled delta rather than presenting
invalid GPU numbers. HDDA node-skipping remains the v1.1 quality/performance
lever.

## What exists

| Artifact | Where | Proven by |
|---|---|---|
| `NanoVDBGrid` (storage attr, bboxes, transforms, proxy geometry) + `createVolumeRenderer` (D4 device-first, pure-function limits math) | `packages/three-nanovdb/src/grid.ts`, `renderer.ts` | 20 tests; convention discovery: `worldBBox.max` = indexToWorld(indexBBox.max **+ one voxel**) |
| `NanoVDBVolumeMaterial` — fragment raymarch straight off the storage buffer: BackSide proxy box, library `Map` world→index, `hdda_ray_clip`, fixed-step march with per-type trilinear sampler chosen at build time, premultiplied front-to-back with early-out, HG phase + shadow march + ambient, live `uniform()` params, per-fragment sample budget (16384 default) | `src/material.ts`, `src/wgsl.ts` | 17 material/wgsl tests + demo-02 goldens |
| TSL×vendored-WGSL integration (headline): `setName("nvdbGrid")` on the storage node (WGSLNodeBuilder honors it — verified at three r185 source line 1326) + one-token rewrite of the library's single buffer-read site; vendored file untouched; guard test pins the occurrence count at 2 | `src/wgsl.ts` (`assembleVolumeWgsl`) | tests + verifier source-check |
| Demo 02 `cloud` (orbit, live params, `?test=1` deterministic mode, WDAS fetch-probe) | `examples/src/demos/02-cloud/` | `e2e/demo-02.spec.ts` two-pose goldens |
| Demo 03 `gpu-read` — 561-line single-file teaching artifact, 5 prose sections | `examples/src/demos/03-gpu-read/` | `e2e/demo-03.spec.ts` (8/8) + editorial review: publishable |

## Deferred by design (v1.1+, per SPEC)

- HDDA node-skipping in the march (fixed-step shipped); depth/scene
  compositing; blue-noise texture jitter (hash jitter shipped).
- Phase 7 flag from verification: the material bakes its `storage()`
  node at construction — sequence playback needs a grid-rebind API or a
  documented rebuild-per-frame pattern. Design Phase 7 with this in mind.

## Environment paths

Demo 02 now has explicit `presentation=direct|readback` and compatibility
switches. Stable Chrome hardware passed repeated direct-canvas startup,
resize, and rendering with both shims disabled. Canonical SwiftShader E2E
still selects readback and the shims explicitly; the package itself contains
none of these demo-scoped workarounds.

## Phase 4/5 entry points

- Phase 4 (compute utilities): `decodeToAtlas`/`gridStats`/
  `valueTransform` — reuse `assembleVolumeWgsl`'s include mechanism and
  demo 01's compute dispatch pattern; `NanoVDBGrid.storageAttribute` is
  the input surface.
- Phase 5 (pure-TS .vdb parser): fixtures identified and reachable
  in-sandbox — `mjurczyk/openvdb` (MIT) commits the classic openvdb.org
  sample `.vdb`s. The release corpus pins sphere/cube/smoke/utahteapot by
  commit, byte size, and SHA-256. Stanford bunny derivatives are excluded
  because their non-commercial/product-sale terms differ; see
  `fixtures/ATTRIBUTION.md`.
- `NanoVDBGrid.indexToWorld()` is axis-aligned-only (documented);
  rotated/sheared Maps need the loader to surface `GridData.mMap`.
