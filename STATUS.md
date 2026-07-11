# STATUS

- **Current phase:** 0 — scaffold + fixtures (docs/PLAN.md)
- **Gate:** `pnpm test` green on stubs — **passing** (8 unit tests + Playwright smoke + tsc)
- **Done:** pnpm workspace (nanovdb-wgsl / three-nanovdb / vdb-web-tools / examples); vendored `pnanovdb.wgsl` fork @ `265e8d82` with NOTICE/VENDOR.md; WDAS + EmberGen fetch scripts (fixtures git-ignored); Vite/Vitest/Playwright wiring; study docs migrated to `docs/`
- **Remaining in Phase 0:** fixture bake image (`nanovdb_convert` + primitive-bake C++ → `.nvdb` + JSON sidecars); extract upstream stride tables to JSON
- **Blockers:** none
- **Next entry point:** finish the bake pipeline, then Phase 1 (`NanoVDBFile` loader + `wgslFn` binding spike — docs/PLAN.md Phase 1)
