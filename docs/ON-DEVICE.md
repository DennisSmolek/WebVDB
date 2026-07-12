# On-Device Checklist

Everything that could not be executed in the cloud sandbox (proxy-blocked
downloads, no Docker daemon, software-only GPU). Ordered so earlier items
unblock later ones. Each item says what to run, what "done" looks like,
and where to record results.

## 1. Fetch the hero assets (network)

```sh
pnpm fixtures            # WDAS cloud + EmberGen packs + vdb samples
```

- `scripts/fetch-wdas-cloud.mjs` and `fetch-embergen-pack.mjs` were
  written blind (their hosts are proxy-blocked in the sandbox) — if a
  URL moved, both scripts print manual fallbacks and take `WDAS_URL=` /
  `EMBERGEN_URLS=` overrides. Fix the default URL in the script when you
  learn the real one.
- Done when: `fixtures/wdas/wdas_cloud_quarter.vdb` exists and at least
  one EmberGen pack is unpacked under `fixtures/embergen/<pack>/`.

## 2. Build the fixture-bake / companion image (Docker)

```sh
docker build -t webvdb-fixture-bake docker/fixture-bake   # ~30-60 min (full OpenVDB build)
docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake   # re-bake primitives (should be byte-identical)
```

- The image is written but has never been built (no daemon in sandbox).
  Expect possible apt package-name drift in the runtime stage
  (`libboost-iostreams1.83.0` etc. are Ubuntu 24.04 names).
- Done when: `nanovdb_convert` runs via
  `docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake nanovdb_convert --version`.

## 3. Bake WDAS + EmberGen to .nvdb

```sh
docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake \
  nanovdb_convert --fp8 /out/wdas/wdas_cloud_quarter.vdb /out/wdas/wdas_cloud_quarter_fp8.nvdb
# EmberGen: convert a frame range the same way, then write a manifest:
#   fixtures/embergen/<pack>/manifest.json -> { "fps": 24, "frames": ["f0001.nvdb", ...] }
```

- Demo 02 auto-prefers the WDAS file; demo 05 plays the pack via
  `?src=embergen` (manifest format defined in docs/handoffs — Phase 7
  section / e2e demo-05 notes).

## 4. nanovdb_convert parity sweep (the deferred Phase 5 anchor)

Steps in `docs/handoffs/PHASE-5.md` §"Native-machine checklist".
Summary: convert the four `fixtures/vdb-samples/*.vdb` natively to
float+fp8, then value-sweep them against `buildFromVdb`/`quantize`
output through the CPU reader. Expected: value parity (sandbox evidence:
TS fp8 re-encode already matches the native fp8 fixture exactly at all
73 sidecar coords). Known byte-level diffs that are NOT failures:
active-tile compression, CRC (we write the disabled sentinel), trailing
padding, near-field background fill (documented anomaly).
Record results in PHASE-5.md.

## 5. WDAS perf gate — the one open PLAN Phase 3 item

Steps in `docs/handoffs/PHASE-3.md` §"Residual gate item". Summary:
demo 02 with the WDAS quarter fp8 cloud at 1080p on a desktop GPU;
target ≥60 fps; record an ms budget (march / shadow) — GPU timestamps
via three's `trackTimestamp` or browser profiling. Append the table to
PHASE-3.md. Tunables if under budget: stepSize, shadowSteps,
sampleBudgetCap, and HDDA node-skipping is the designed-but-unbuilt
lever (v1.1).

## 6. Real-hardware retests of sandbox-specific workarounds

All documented where they live; each may simply vanish on real hardware:
- Canvas presentation: demos render offscreen-RT→readback→2D-blit
  because the sandbox's SwiftShader drops the Dawn instance on canvas
  present (`examples/src/demos/02-cloud/main.ts` header). Try direct
  canvas presentation; if it works, simplify the demos.
- `Data3DTexture` + TSL `texture3D()`: rejected by the sandbox's Dawn
  (2D-view-vs-3D-texture validation). Demo 04 samples the atlas from a
  storage buffer instead. Retry the real `VolumeNodeMaterial` bridge
  (`examples/src/demos/04-atlas-fallback/main.ts` header).
- r185 swizzle-string shim + adapter keep-alive shim (same file) —
  check if still needed on current Chrome.
- Playwright goldens are SwiftShader-specific: on real hardware the
  screenshots WILL differ. Either keep CI on SwiftShader flags (the
  configured default) or maintain per-platform snapshots.

## 7. Odds and ends (owner: Dennis)

- Flip the GitHub default branch to `main` (Settings → Branches) if not
  done yet.
- File the picovdb license question upstream
  (github.com/emcfarlane/picovdb — PLAN §3 item 3; zero-cost, potential
  future format win).
- npm publish: packages pass `npm pack --dry-run`; actual publishing
  (org scope, 2FA, provenance) is a human step.
- EmberGen pack licensing: confirm the JangaFX free-pack terms cover
  redistribution in demos/CI if you ever commit any frames (fixtures
  are git-ignored today, so currently moot).
