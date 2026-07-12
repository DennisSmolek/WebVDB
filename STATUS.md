# STATUS

- **Sandbox-scope work COMPLETE: Phases 0–7 + Phase 8's packaging slice.** All remaining items need Dennis's machine — see **docs/ON-DEVICE.md** (dependency-ordered checklist).
- **Suite:** 234 unit / 11 e2e (demos 01–07 + gpu-parity + smoke; goldens byte-stable) / clean tsc / `pnpm build` + `npm pack --dry-run` green for all three packages — pushed to `claude/vdb-docs-phase-0-setup-pviw1b` and `main`
- **Shipped:** `.nvdb` loader (hardened) + CPU reference (isomorphic, single source of truth); extended audited WGSL fork (decoders/map/trilinear/HDDA/stats) with GPU parity proven at 22.5k evaluations; `NanoVDBGrid`/`createVolumeRenderer`/`NanoVDBVolumeMaterial` (sample-budgeted, grid-rebind + `maxGridBytes`); compute utilities (stats/transform/atlas); pure-TS `.vdb` parser + NanoVDB serializer + native-exact Fp8/FpN quantization + transform/inspect/writeNvdb; `NanoVDBSequence` player; demos 01–07; package READMEs + publishable exports maps
- **Key facts for future sessions:** TS fp8 encode is native-exact at every tested coord; r185 storage buffers are fixed-size (rebind via maxGridBytes padding); `"development"` export condition resolves workspace source (customConditions in tsconfig.base); sandbox Dawn quirks documented in PHASE-4.md/ON-DEVICE.md §6
- **Blockers:** none — sandbox work is done; next actions are all in docs/ON-DEVICE.md
- **Read first on resume:** this file → docs/ON-DEVICE.md → docs/handoffs/PHASE-5.md
