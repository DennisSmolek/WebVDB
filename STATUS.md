# STATUS

- **v1 complete**, tagged `v1-final`. On-device gates passed 2026-07-12. Never published to npm; the v1 tarballs in `artifacts/npm/` are historical.
- **Plan v2** is in [PHASES.md](./PHASES.md): one package `@pmndrs/vdb`, port-and-extend renderer, pathtracer after 0.1.
- **K0 done:** the three v1 packages are now one package, `@pmndrs/vdb` (`packages/vdb`; subpaths `.`, `./tools`, `./three`, `./pnanovdb.wgsl`). 255 unit tests pass. Packed-consumer check: `pnpm verify:packed-subpaths`.
- **K1 in progress:** core (`packages/vdb/src/core`).
- **Read first on resume:** CLAUDE.md → PHASES.md → TASKS.md → docs/handoffs/
