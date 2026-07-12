# STATUS

- **Current phase:** 3 closed (CI scope) → next: Phase 4 (compute utilities) ∥ Phase 5 (pure-TS .vdb parser); docs/handoffs/PHASE-3.md has the one deferred hardware item
- **Suite:** 124 unit / 6 e2e (demo-01, demo-02 two-pose goldens, demo-03, gpu-parity, smoke) / clean tsc — all pushed
- **Shipped through Phase 3:** loader (+hardening), CPU reference (value+trilinear), extended WGSL fork (audited, decoders/map/trilinear/HDDA/stats), GPU parity harness (22.5k evals, 0 fail), NanoVDBGrid + createVolumeRenderer, NanoVDBVolumeMaterial (sample-budgeted), demos 01–03
- **Deferred/open:** WDAS 60fps + perf-budget table (real GPU + network; steps in PHASE-3.md); Docker bake image unbuilt here; HDDA march skipping + depth compositing (v1.1); material grid-rebind API (Phase 7 design input)
- **Blockers:** none for Phases 4/5
- **Read first on resume:** this file → docs/handoffs/PHASE-3.md
