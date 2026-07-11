# STATUS

- **Current phase:** 0 — scaffold + fixtures (docs/PLAN.md) — **complete**
- **Gate:** `pnpm test` green on stubs — **passing** (unit suites incl. vendor/stride/sidecar guards + Playwright smoke + tsc)
- **Done:** pnpm workspace (nanovdb-wgsl / three-nanovdb / vdb-web-tools / examples); vendored `pnanovdb.wgsl` fork @ `265e8d82` + reference `PNanoVDB.h` @ openvdb `a532de55` (ABI 32.9.1) with NOTICE/VENDOR.md; stride tables extracted to `vendor/stride-tables.json` (regen script + sync test); WDAS/EmberGen fetch scripts; primitive fixtures baked (9 grids × `.nvdb` git-ignored, sidecars committed); fixture-bake Dockerfile (also the D6 companion-service seed); study docs migrated to `docs/`
- **Caveats:** Docker image is written but unbuilt (no daemon in the authoring sandbox — `pnpm fixtures:bake` verified via the local-g++ path); WDAS/EmberGen download URLs unverified from sandbox (scripts have manual fallbacks)
- **Blockers:** none
- **Next entry point:** Phase 1 — `NanoVDBFile` TS loader + the `wgslFn` storage-binding spike (docs/PLAN.md Phase 1); fixtures for it are already in `fixtures/primitives/`
