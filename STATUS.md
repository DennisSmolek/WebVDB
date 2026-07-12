# STATUS

- **Phases 0–5 complete (CI scope)** — see docs/handoffs/PHASE-{1..5}.md. Demos 01–04, 06, 07 shipping and e2e-gated.
- **Suite:** 206 unit / 10 e2e (goldens byte-stable) / clean tsc — pushed to `claude/vdb-docs-phase-0-setup-pviw1b` AND `main` (flip default branch to main in GitHub settings)
- **Browser-first milestone (D6):** parse .vdb → build → quantize fp8/fpn → write .nvdb → render, all pure TS + WGSL, no server. TS fp8 encode is native-exact at every tested coord.
- **Needs a native/hardware machine:** WDAS 60fps + perf table (PHASE-3.md); nanovdb_convert value-parity sweep (PHASE-5.md checklist); Docker bake/companion image; Data3DTexture + canvas-present retests (PHASE-4.md)
- **Next candidates:** Phase 7 sequences (needs EmberGen download + material grid-rebind API); Phase 8 polish (docs site, npm dry-run, explorer extras); isomorphic CPU-reference refactor (top code-health debt); Phase 6b service (Docker)
- **Blockers:** none in-sandbox; remaining phases need assets/hardware this environment can't reach
- **Read first on resume:** this file → docs/handoffs/PHASE-5.md
