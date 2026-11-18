# STATUS

- **Plan v2 is built** ([PHASES.md](./PHASES.md)): K0 to K4 done; K5 (release 0.1) is prepared. One package, `@pmndrs/vdb` (`packages/vdb`; subpaths `.`, `./tools`, `./three`, `./pnanovdb.wgsl`). On-device gates passed 2026-07-12 and the 0.1.0 on-device release gates (native parity, hardware checks, packaging) are in.
- **Tests:** 1484 unit tests pass (15 skip without fixtures), `tsc --build` clean, `pnpm docs:check`, `pnpm wgsl:size` (204 variants) and `pnpm verify:packed-subpaths` pass. A packed consumer type-checks with `skipLibCheck: false`.
- **Release:** `packages/vdb/package.json` stays at `0.0.0`. The owner publishes 0.1.0 by hand, then sets up npm trusted publishing: see [RELEASING.md](./RELEASING.md). Nothing has been published, tagged or pushed.
- **Open:** [TASKS.md](./TASKS.md): K5.3 (cloud showcase, owner), K5.7 (post-release follow-ups), K4.22, then P1 (pathtracer) after 0.1.
- **Never published:** the v1 packages (`webvdb` and siblings); `v1-final` is the tag.
- **Read first on resume:** CLAUDE.md, PHASES.md, TASKS.md, docs/handoffs/
