# STATUS

- **Current phase:** 3 — cloud material; core + material + demo 02 LANDED, remaining: demo 03 (annotated gpu-read, T2 from the material), Phase 3 verification pass, docs/handoffs/PHASE-3.md
- **Suite:** 119 unit / 5 e2e (incl. two-pose demo-02 goldens, stable across runs) / clean tsc — all pushed
- **Phase 3 headline:** vendored WGSL library binds into TSL via storage-node setName + one-token buffer rewrite (assembleVolumeWgsl in three-nanovdb/src/wgsl.ts); raymarch material ships fixed-step + HG phase + shadow march; HDDA skipping & depth compositing deferred to v1.1
- **Sandbox quirks (documented in demo 02):** canvas presentation drops the Dawn instance under SwiftShader — demo renders offscreen RT → readback → 2D blit; r185 swizzle-string shim at app edge
- **Blockers:** none for CI path; WDAS/EmberGen fetch needs a network-open machine; .vdb parser fixtures for Phase 5 identified (mjurczyk/openvdb sample assets via pinned raw.githubusercontent — reachable in-sandbox)
- **Read first on resume:** this file, then docs/handoffs/PHASE-2.md (PHASE-3.md pending)
