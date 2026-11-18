# On-Device Checklist

Completed on 2026-07-12 unless explicitly marked. Downloaded assets, generated
NanoVDBs, reports, and hardware screenshots remain local/ignored.

## Environment

- macOS 26.5.1 on a MacBook Pro with Apple M5 Pro (18 CPU / 20 GPU cores,
  64 GB), adapter reported by Chrome as `apple metal-3`.
- Stable Chrome 149.0.7827.201 at an explicit 1920×1080 viewport.
- Node 22.22.2, pnpm 10.33.0, Docker 29.4.1.

## Completed gates

- [x] `pnpm install --frozen-lockfile`, `pnpm typecheck`, `pnpm test`, and
  `pnpm build`. Current unit result: 249/249.
- [x] Downloaded WDAS quarter cloud, EmberGen Fire/Cloud packs, and pinned
  OpenVDB samples. The four VDB sources are fixed to one commit with exact
  byte-size/SHA-256 checks; stale local smoke/teapot revisions were replaced.
  MediaFire failures are isolated per pack. Archive paths and member types are
  checked before extraction; symlinks, hardlinks, and special files are
  rejected. The complete temporary tree is then recursively `lstat`-checked
  before promotion.
- [x] Built `webvdb-fixture-bake`; `nanovdb_convert --version` runs in the
  runtime image. Build parallelism is capped to avoid Docker Desktop memory
  exhaustion.
- [x] Re-baked all nine primitives twice into clean output directories and
  verified byte-identical NanoVDBs and sidecars.
- [x] Converted WDAS to
  `fixtures/wdas/wdas_cloud_quarter_fp8.nvdb` (28 MB).
- [x] Converted the EmberGen `Fire_01` `flames` grid to 100 validated FP8
  frames and generated `fixtures/embergen/Fire_01/manifest.json` plus
  `fixtures/embergen/index.json`. Source frame 0 is an empty grid that
  `nanovdb_convert` serializes inconsistently, so the playable sequence starts
  at frame 1 via `SEQUENCE_START_INDEX=1`. Baking now requires
  `SEQUENCE_TEMPORAL=1`, records `kind: "temporal-sequence"`, and rejects the
  Cloud pack's duplicate `_0000` variant indices.
- [x] Demo 05 loaded and preflighted all 100 frames, then hardware validation
  rendered frame progression `0…99→0`: 101 rebinds, one complete cycle, zero
  stalls, direct presentation, and both shims disabled.
- [x] `pnpm parity:native` passed all four pinned sources in FLOAT and FP8.
  It independently enumerates both outputs, compares the active-coordinate
  union and canonical root/internal/leaf topology, records each source hash,
  rejects unknown byte fields, and uses finite per-leaf FP8 bounds; see
  `docs/handoffs/PHASE-5.md`.
- [x] Current `pnpm hardware:perf` passed direct presentation, stock 3D
  texture, storage fallback, no-shim compatibility, the full EmberGen cycle,
  and the 103.96 FPS WDAS 1080p gate. Every lane validates fixture, adapter,
  presentation, mode, shims, resolution, and quality fields. See
  `docs/handoffs/PHASE-3.md` and
  `docs/handoffs/PHASE-4.md`.
- [x] Packed all three unscoped packages at 0.1.0, installed their real
  tarballs into a clean external consumer, verified default and
  `--conditions=development` imports resolve to packaged `dist/`, and passed
  `npm publish --dry-run`.
- [x] GitHub default branch is `main`; picovdb licensing question filed as
  `emcfarlane/picovdb#6`.
- [x] EmberGen/JangaFX CC0 terms and credit are preserved in tracked
  `fixtures/ATTRIBUTION.md`; binaries and bundled pack files remain ignored.
  Stanford bunny derivatives are separately documented and excluded.

## Verification note

The canonical screenshot lane remains Linux/SwiftShader. On this Mac, 10/11
E2E tests pass against those Linux snapshots; Demo 06 differs by 3% versus its
2% threshold due to cross-OS raster/text output. Functional assertions,
SwiftShader adapter assertions, and the `.vdb` route pass. No Mac-specific
goldens were retained. Run the canonical golden lane on Linux CI rather than
updating snapshots from this machine.

Three r185/Chrome exposes timestamp queries on this GPU, but its returned
durations were physically inconsistent with queue-synchronized wall time.
The benchmark rejects those values and records `gpuTimestampValid=false`;
this is the only measurement limitation. Actual package publication remains a
manual owner action and was intentionally not performed.
