# VDB-on-the-Web Study

> **Note:** These are the founding documents of **WebVDB** — this repository.
> The study was drafted in a working-session branch of an unrelated repo
> (`SebH-TSL-Sky` · `claude/webgpu-vdb-feasibility-kvwq2x`) and migrated
> here as `docs/` at Phase 0 kickoff, per decision D1. The build proceeds
> in this repo.

An investigation into loading and rendering OpenVDB/NanoVDB volumes on the
web: WebGPU + Three.js (TSL) for GPU-side rendering, WASM for CPU-side file
I/O and grid operations.

| Doc | Contents |
|---|---|
| [DECISIONS.md](./DECISIONS.md) | Resolved review decisions (repo, WGSL base, TS-first CPU path, targets, names) |
| [FEASIBILITY.md](./FEASIBILITY.md) | Research findings, prior art, platform constraints, verdict, risks |
| [SPEC.md](./SPEC.md) | What we will build: architecture, components, formats, APIs |
| [PLAN.md](./PLAN.md) | Phased build plan with sub-agent (Haiku/Sonnet/Opus) assignments and token/compaction strategy |

Research date: 2026-07-11. Verified against OpenVDB `master`
(NanoVDB ABI 32.9.1) and three.js r178+ TSL/WebGPURenderer.
