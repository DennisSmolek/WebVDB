# Research index

Research notes for KumoVDB. Track items R1–R10 are defined in [PHASES.md](../PHASES.md). Notes go in
`research/notes/`, named `R<n>-<topic>.md`.

## Existing studies (v1)

| Note | What it settled |
|---|---|
| [docs/FEASIBILITY.md](../docs/FEASIBILITY.md) | The original feasibility study: WGSL NanoVDB is viable, the TSL binding risk, and the CPU-half options |
| [docs/spikes/PICOVDB.md](../docs/spikes/PICOVDB.md) | PicoVDB: not adopted now (D16). Its fog f32 is 2.2× our Fp8, and it has two read bugs on fog grids |
| mjurczyk/openvdb comparison (doc: https://claude.ai/artifact/FV7TNDJVmhuLxynaPf8X9o) | `.wvdb` is small because of resampling. At equal resolution our FpN is smaller (474 vs 790 KiB on smoke); at full resolution ours is 643 KiB at ~1% error against his ~17%. Basis for D7, D12 and D14 |

## Open

| ID | Question | Note |
|---|---|---|
| R1 | How does mjurczyk's `FrameRenderer` work (froxels, scattering, shadows, emission)? | `notes/R1-frame-renderer.md`, not written yet |
| R2 | WDAS frame rate and per-pass times on hardware | The pathtracer's research notes record 103.96 fps at 1080p; the run is not in this repo |
