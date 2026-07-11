# Fixtures

Binary volume assets are **never committed** (see the repo `.gitignore`);
every machine fetches them with the scripts below. Only this README and
JSON sidecars (ground-truth samples, tree stats — produced by the bake
step) are tracked in git.

## Fetching

```sh
pnpm fixtures            # everything below
pnpm fixtures:wdas       # WDAS cloud, quarter res → fixtures/wdas/
pnpm fixtures:embergen   # EmberGen free packs     → fixtures/embergen/
```

| Corpus | Source | License | Role |
|---|---|---|---|
| WDAS cloud (quarter; other variants via `WDAS_VARIANTS=`) | [disneyanimation.com/data-sets](https://disneyanimation.com/data-sets/) | CC-BY-SA 3.0, © Walt Disney Animation Studios | Stretch fixture: demo 02 hero cloud, perf gates (D4) |
| EmberGen free VDB animation packs | [jangafx.com — free VDB animations](https://jangafx.com/software/embergen/download/free-vdb-animations) | Free packs from JangaFX — check page for current terms | Sequence playback corpus (Phase 7), typical single-frame sizes |
| Procedural primitives (`sphere/torus/box` fog volumes × Float/Fp8/FpN) | `pnpm fixtures:bake`, see below | Apache-2.0 (ours) | Exact-ground-truth unit fixtures for the traversal suite |

Both fetch scripts are idempotent (skip existing files), print a manual
fallback if the upstream URL moves, and take URL overrides via
`WDAS_URL=` / `EMBERGEN_URLS=`.

## Baking `.nvdb` + sidecars

The traversal test suite consumes `.nvdb` files plus JSON sidecars of
sampled ground-truth values (docs/SPEC.md §6):

```sh
pnpm fixtures:bake
```

bakes fog-volume sphere/torus/box in Float/Fp8/FpN (nine grids) into
`fixtures/primitives/` via `docker/fixture-bake/bake_primitives.cpp`,
pinned to the same OpenVDB commit as the vendored `PNanoVDB.h`
(`a532de55`). The `.nvdb` files are git-ignored; the `.sidecar.json`
files (tree stats + 73 deterministic value probes each) **are committed**
and guarded by `packages/nanovdb-wgsl/test/sidecars.test.ts`.

The script prefers the reproducible Docker route
(`docker/fixture-bake/Dockerfile` — a full OpenVDB+NanoVDB tools build
that per D6 also seeds the interim companion service) and falls back to a
local `g++` build against a sparse clone of the pinned headers (NanoVDB
is header-only). Force either with `BAKE_MODE=docker|local`.

Converting downloaded `.vdb` assets needs `nanovdb_convert` from the
Docker image (or a native install):

```sh
docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake \
  nanovdb_convert --fp8 /out/wdas/wdas_cloud_quarter.vdb /out/wdas/wdas_cloud_quarter_fp8.nvdb
```

Layout convention:

```
fixtures/
  downloads/    # raw archives (git-ignored)
  wdas/         # wdas_cloud_quarter.vdb → *.nvdb + *.sidecar.json
  embergen/     # <pack>/frame_####.vdb → *.nvdb sequences
  primitives/   # baked sphere/torus/box .nvdb + sidecars
```
