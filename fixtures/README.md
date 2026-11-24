# Fixtures

Binary volume assets are **never committed** (see the repo `.gitignore`);
every machine fetches them with the scripts below. Only documentation and
JSON sidecars (ground-truth samples, tree stats — produced by the bake
step) are tracked in git.

## Fetching

```sh
pnpm fixtures              # everything below
pnpm fixtures:wdas         # WDAS cloud, quarter res → fixtures/wdas/
pnpm fixtures:embergen     # EmberGen free packs     → fixtures/embergen/
pnpm fixtures:vdb-samples  # classic openvdb.org samples → fixtures/vdb-samples/
pnpm fixtures:cie          # (not in `pnpm fixtures`) CIE 1931 2° CMF table, test-only → fixtures/cie/
```

| Corpus | Source | License | Role |
|---|---|---|---|
| WDAS cloud (quarter; other variants via `WDAS_VARIANTS=`) | [disneyanimation.com/data-sets](https://disneyanimation.com/data-sets/) | CC-BY-SA 3.0, © Walt Disney Animation Studios | Stretch fixture: demo 02 hero cloud, perf gates (D4) |
| JangaFX EmberGen free VDB packs | [jangafx.com — free VDB downloads](https://jangafx.com/software/embergen/download/free-vdb-animations) | CC0 1.0; tracked terms and credit in [ATTRIBUTION.md](./ATTRIBUTION.md) | Fire temporal playback corpus; Cloud files are independent variants, not animation |
| Procedural primitives (`sphere/torus/box` fog volumes × Float/Fp8/FpN) | `pnpm fixtures:bake`, see below | Apache-2.0 (ours) | Exact-ground-truth unit fixtures for the traversal suite |
| Half primitives + edge cases (`*_half`, `half/half_edge`) | `pnpm fixtures:bake-half`, see below | Apache-2.0 (ours) | Native NanoVDB v14 Half grids: the reference for our Half reader and writer (K7.H) |
| CIE 1931 2° colour matching functions (`CIE_xyz_1931_2deg.csv`, MD5 `17cca777db64b17170f06f67ce9d3ab7`) | [files.cie.co.at](https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv) | CC BY-SA 4.0 (CIE, DOI 10.25039/CIE.DS.xvudnb9b) | Test-only (D33): measures the package's analytic CMF fit (`packages/vdb/test/tools/cmf.test.ts`) and lets the K3 harness render the explosion parity scenes with his exact colours; never shipped. Absent: those tests skip and those scenes report INFO; `WEBVDB_REQUIRE_FIXTURES` does not require it |
| Classic openvdb.org sample models (`sphere`, `cube`, `smoke`, `utahteapot`) | Pinned commit of [mjurczyk/openvdb](https://github.com/mjurczyk/openvdb) | Assets: CC BY-SA 4.0; hosting repo: MIT | Phase 5 parser/parity corpus; exact sizes and SHA-256 values are enforced |

Stanford `bunny.vdb`/`bunny_cloud.vdb` derivatives have separate
non-commercial/product-sale restrictions. They are deliberately excluded from
fetching, parity, and packages; see [ATTRIBUTION.md](./ATTRIBUTION.md).

The fetch scripts are idempotent, print a manual fallback if the upstream URL
moves, and take URL overrides via `WDAS_URL=` / `EMBERGEN_URLS=`. The four VDB
sample files are immutable: existing files are size/hash-validated and replaced
when they do not match the pinned revision.

## Baking `.nvdb` + sidecars

The traversal test suite consumes `.nvdb` files plus JSON sidecars of
sampled ground-truth values (docs/archive/v1/SPEC.md §6):

```sh
pnpm fixtures:bake
pnpm fixtures:bake-sequence fixtures/embergen/<pack>
```

bakes fog-volume sphere/torus/box in Float/Fp8/FpN (nine grids) into
`fixtures/primitives/` via `docker/fixture-bake/bake_primitives.cpp`,
pinned to the same OpenVDB commit as the vendored `PNanoVDB.h`
(`a532de55`). The `.nvdb` files are git-ignored; the `.sidecar.json`
files (tree stats + 73 deterministic value probes each) **are committed**
and guarded by `packages/vdb/test/tools/sidecars.test.ts`.

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

`fixtures:bake-sequence` requires `SEQUENCE_TEMPORAL=1`, verifies distinct
monotonically increasing frame numbers, converts every source frame deterministically,
validates each generated NanoVDB, then atomically writes the pack's
`manifest.json` and the EmberGen manifest index. Multi-grid packs can select
one grid with `SEQUENCE_GRID=<name>`; use
`SEQUENCE_START_INDEX=<non-negative index>` to omit an intentionally empty
leading frame. The validated Fire pack command is:

```sh
SEQUENCE_TEMPORAL=1 SEQUENCE_GRID=flames SEQUENCE_START_INDEX=1 \
  pnpm fixtures:bake-sequence fixtures/embergen/Fire_01
```

## Committed native v14 fixtures

A few files written by native OpenVDB / NanoVDB v14.0.0 and NVIDIA Warp are small enough to commit (about 100 KiB
together): `packages/vdb/test/tools/fixtures/native-v14/` (BLOSC and BloscLZ `.nvdb`, HalfGrid `.vdb`; K7.C). Its
README holds the generator programs and the commands that rebuild them; nothing there needs Docker or a fetch.

### Half grids (K7.H)

```sh
pnpm fixtures:bake          # first: the float primitives are the source
pnpm fixtures:bake-half
```

NanoVDB's Half grid type (`GridType::Half`, raw IEEE binary16) was a placeholder until NanoVDB 13.2 (OpenVDB PR
2335), and `nanovdb_convert` has no Half option, so `scripts/bake-half-fixtures.mjs` compiles
`docker/fixture-bake/bake_half.cpp` against the header-only NanoVDB of OpenVDB **v14.0.0** (source tarball fetched
once into `fixtures/downloads/`; or `NANOVDB_V14_INCLUDE=<dir>`) with any C++17 compiler. It writes
`createNanoGrid<FloatGrid, nanovdb::Half>` of the three float primitives as `primitives/*_half.nvdb` (+ sidecars:
the primitive schema with each probe's binary16 `bits`, the root stats and an FNV-1a hash of the leaf value tables),
and `half/half_edge.nvdb` (+ sidecar), a small grid of rounding ties, subnormals, overflow to infinity and -0 with
every voxel's input and native bits and every node's stats as raw bits. The `.nvdb` files are git-ignored; the
sidecars are committed and checked by `packages/vdb/test/tools/half-native.test.ts` (our writer against them) and
`e2e/core-parity.spec.ts` (`sphere_fog_half`). Re-bakes are deterministic.

Layout convention:

```
fixtures/
  downloads/    # raw archives (git-ignored)
  wdas/         # wdas_cloud_quarter.vdb → *.nvdb + *.sidecar.json
  embergen/     # <pack>/frame_####.vdb → *.nvdb sequences
  primitives/   # baked sphere/torus/box .nvdb + sidecars (float/fp8/fpn; half from NanoVDB v14)
  half/         # half_edge.nvdb + sidecar (Half edge cases, NanoVDB v14)
  vdb-samples/  # sphere.vdb, cube.vdb, smoke.vdb, utahteapot.vdb
```
