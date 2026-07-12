# Handoff: Phase 3 — cloud material

Gate: **PASSED in CI scope; hardware perf item deferred (see Residual).**
Demos 02 + 03 ship and are e2e-gated: two-pose golden images byte-stable
across consecutive runs, demo 03 rated *publishable* by an independent
editorial review that checked every byte-offset claim against upstream.
Suite at close: 124 unit / 6 e2e / clean tsc.

## Residual gate item (needs a real GPU + network)

PLAN's Phase 3 gate line "WDAS cloud ≥60 fps/1080p desktop + perf budget
table via GPU timestamps" cannot be executed in the authoring sandbox
(no WDAS download — proxy-blocked; SwiftShader fps is meaningless).
To finish it on a desktop machine:
1. `pnpm fixtures:wdas` (downloads the quarter cloud; verify the URL —
   unverified from the sandbox), bake fp8 via `nanovdb_convert --fp8`
   (Docker image in `docker/fixture-bake/`, also unbuilt here).
2. Open demo 02 — it fetch-probes `/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb`
   and prefers it over the primitive fallback.
3. Record fps at 1080p + a ms budget (march / shadow) via
   `trackTimestamp`/GPU timestamps; append the table here.

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

## Environment workarounds (demo-scoped, NOT in the package)

`examples/src/demos/02-cloud/main.ts` carries: r185 swizzle-string strip
shim; adapter keep-alive; offscreen RenderTarget → readback → 2D-canvas
blit (sandbox canvas presentation drops the Dawn instance under
SwiftShader). Re-test plain canvas presentation on real hardware — the
package itself has none of this.

## Phase 4/5 entry points

- Phase 4 (compute utilities): `decodeToAtlas`/`gridStats`/
  `valueTransform` — reuse `assembleVolumeWgsl`'s include mechanism and
  demo 01's compute dispatch pattern; `NanoVDBGrid.storageAttribute` is
  the input surface.
- Phase 5 (pure-TS .vdb parser): fixtures identified and reachable
  in-sandbox — `mjurczyk/openvdb` (MIT) commits the classic openvdb.org
  sample `.vdb`s (sphere 0.8 MB … bunny_cloud 80 MB); add a pinned-commit
  fetch script to the fixtures pipeline. CPU-only lane, fully parallel
  with Phase 4.
- `NanoVDBGrid.indexToWorld()` is axis-aligned-only (documented);
  rotated/sheared Maps need the loader to surface `GridData.mMap`.
