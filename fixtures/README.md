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
| Procedural primitives (`sphere/torus/box` fog volumes) | baked locally, see below | Apache-2.0 (ours) | Exact-ground-truth unit fixtures for the traversal suite |

Both fetch scripts are idempotent (skip existing files), print a manual
fallback if the upstream URL moves, and take URL overrides via
`WDAS_URL=` / `EMBERGEN_URLS=`.

## Baking `.nvdb` + sidecars (Phase 0/1 follow-up)

The traversal test suite consumes `.nvdb` files plus JSON sidecars of
sampled ground-truth values (docs/SPEC.md §6). Baking uses the native
NanoVDB tools (`nanovdb_convert`, plus a tiny C++ program around
`createFogVolumeSphere/Torus/Box`) inside a Docker image so it's
reproducible — the same image later grows into the interim companion
service (docs/DECISIONS.md D6). Typical conversion:

```sh
nanovdb_convert --fp8 fixtures/wdas/wdas_cloud_quarter.vdb fixtures/wdas/wdas_cloud_quarter_fp8.nvdb
```

Layout convention:

```
fixtures/
  downloads/    # raw archives (git-ignored)
  wdas/         # wdas_cloud_quarter.vdb → *.nvdb + *.sidecar.json
  embergen/     # <pack>/frame_####.vdb → *.nvdb sequences
  primitives/   # baked sphere/torus/box .nvdb + sidecars
```
