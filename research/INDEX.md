# Research index

Research notes for `@pmndrs/vdb`. Track items R1–R10 are defined in [PHASES.md](../PHASES.md). Notes go in
`research/notes/`, named `R<n>-<topic>.md`.

## Existing studies (v1)

| Note | What it settled |
|---|---|
| [docs/FEASIBILITY.md](../docs/FEASIBILITY.md) | The original feasibility study: WGSL NanoVDB is viable, the TSL binding risk, and the CPU-half options |
| [docs/spikes/PICOVDB.md](../docs/spikes/PICOVDB.md) | PicoVDB: not adopted now (D16). Its fog f32 is 2.2× our Fp8, and it has two read bugs on fog grids |
| mjurczyk/openvdb comparison (doc: https://claude.ai/artifact/FV7TNDJVmhuLxynaPf8X9o) | `.wvdb` is small because of resampling. At equal resolution our FpN is smaller (474 vs 790 KiB on smoke); at full resolution ours is 643 KiB at ~1% error against his ~17%. Basis for D14, D17 and D18. Read against his v0.25.0; he shipped 0.26.0 on 2026-11-18 |

## Open

| ID | Question | Note |
|---|---|---|
| R1 | Port plan for mjurczyk's renderer: module map, technique costs, port order (D18) | `notes/R1-renderer-port.md`, not written yet |
| ~~R2~~ | WDAS frame rate on hardware | Done 2026-07-12: 103.96 fps at 1080p (`stepSize=4`, `shadowSteps=4`); quality preset 16.8 fps. `docs/handoffs/PHASE-3.md` |
