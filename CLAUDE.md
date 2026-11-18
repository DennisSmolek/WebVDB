# @pmndrs/vdb: rules for every agent

`@pmndrs/vdb` is sparse NanoVDB volumes on WebGPU, part of the pmndrs rendering initiative. An engine-free
WebGPU core comes first; three.js and other engines are thin adapters (D8). The renderer is ported from
mjurczyk's three-openvdb and re-based on the sparse core (D18). The repo is still named WebVDB until it moves to
`pmndrs/vdb` (D17, K0.7). The pathtracer integration comes after our 0.1 release (D19).

## Read first, write last

Read these before starting:
- [PHASES.md](PHASES.md): the plan, phases and exit gates.
- [TASKS.md](TASKS.md): what's open and who owns it.
- [DECISIONS.md](DECISIONS.md): what's settled; don't re-litigate.
- [DIVERGES.md](DIVERGES.md): where we split from a source, and why.
- Background: [spec/SPEC-draft.md](spec/SPEC-draft.md), [research/INDEX.md](research/INDEX.md), and the v1
  history in [docs/handoffs/](docs/handoffs/).

When you finish, update TASKS.md (mark done, add follow-ups) and add anything that would have saved you time to
**Traps** below. Record every departure from a source (`PNanoVDB.h`, NanoVDB, OpenVDB, mjurczyk/openvdb, a paper)
in DIVERGES.md with the reason.

## Working rules

- **Workspace resolution:** published exports point at `dist/` only; in-repo imports reach source through
  tsconfig `paths` and Vite/Vitest aliases. Never add a `development` export condition.
- **The core is engine-free.** `packages/vdb/src/core` and `src/tools` never import `three` or any engine, and
  `tools` never imports `core` (D20; enforced by `test/boundaries.test.ts`). The core takes the host's
  `GPUDevice`, records into the host's encoder, and frees only what it allocated.
- **One storage buffer per kernel** at a host-chosen slot; generated WGSL with the `vdb` prefix; features
  compile out (D10). Grow the WGSL size table (K1.5) only on purpose.
- **Correctness before timing.** Every WGSL entry has a CPU twin in `tools`; parity tests come first.
- **Build both, then decide.** Competing approaches go behind a setting and get an A/B. DECISIONS changes only
  after the A/B.
- **No timings from a loaded machine.** Name the device, browser and conditions; compare paired runs on one
  machine.
- **Questions for the owner** come with a suggestion, pros and cons, and the test that would settle them.
- **Consumer repos** (the owner's cloud system, `pmndrs/pathtracer`): follow their own CLAUDE.md. Read-only
  unless the task says otherwise. Don't change code here to suit the pathtracer before 0.1 (D19).

## Licences

- Our code is MIT, Copyright (c) 2026 Dennis Smolek (D13). Root and `packages/vdb` `LICENSE` are MIT.
- The vendored `pnanovdb.wgsl` (emcfarlane, Apache-2.0) and `PNanoVDB.h` (OpenVDB, Apache-2.0) keep their
  headers and NOTICE (D2).
- mjurczyk/openvdb is MIT (the repo has no LICENSE file yet; its npm packages declare MIT). Port with a header
  naming "mjurczyk/openvdb contributors (MIT)", a THIRD_PARTY entry and a DIVERGES row.
  Don't run its code in the cloud sandbox; read it.
- Test assets are git-ignored and fetched by script. The openvdb.org samples are CC BY-SA 4.0.

## Agent tiers (v1 practice)

T3 (Opus) designs and takes risky work. T2 (Sonnet) implements from a written brief. T1 (Haiku) does mechanical
work. A separate agent verifies each phase gate. Briefs are self-contained: goal, paths, acceptance commands,
and what not to touch.

## Traps

From v1; details in docs/handoffs/.
- WebGPU in headless Chromium needs a secure context: serve over `http://localhost`, never `about:blank`.
  SwiftShader flags are in `playwright.config.ts`.
- `bindGroupLayouts` is positional. A skipped `@group` index silently invalidates the dispatch and leaves outputs
  zeroed; wrap pipeline creation in `pushErrorScope('validation')`.
- `active` is a reserved WGSL word, and so is `set`.
- Back-ticks inside WGSL comments break the enclosing JS template literal.
- three's storage attributes are fixed-size GPUBuffers (r185–r187). The K2 adapter sidesteps it: `VDBGridNode`
  binds the pool's own buffer and rebinds after a growth (`grids.update()` after the pool's update).
- three r185 `NodeMaterial` lighting is skipped when the scene has no light.
- NanoVDB `worldBBox.max` is one voxel beyond `indexToWorld(indexBBox.max)`.
- Native fog grids can store root background ≠ near-field inactive tile values (3 vs 0 on the sphere fixture).
- The `.nvdb` grids under `fixtures/` are git-ignored. Run `pnpm fixtures:bake` before fixture tests;
  sidecars are committed.
- Fp leaf stats are u16 codes; never read them as f32. Fp4/Fp8/Fp16/FpN leaves keep min/max/ave/dev as uint16
  at bytes 88/90/92/94 (`code * quantum + minimum`; dev is `code * quantum`); 90 isn't word-aligned. Root, upper
  and lower stats stay f32. Use the fork's `*_get_*_float` readers or `tools` `nodeStats`.
- One VDB module per shader: generated symbols are declared once, so two modules (two profiles or two pools) in
  one shader redeclare everything. Generate the largest profile you need; several grids share one module by id.
- A `WebGPURenderer` built without `{ device }` destroys its own device in `renderer.dispose()`, and every
  pool buffer on it. Create the device yourself and pass it when a pool must outlive the renderer.
- `@types/three` r187's `Backend` doesn't declare `get`: type renderers structurally as `{ backend: object }`
  and check the internals at runtime (`three/device.ts`), or a real `WebGPURenderer` won't type-check.
- Node stats bound ACTIVE values only. A trilinear sample reads one voxel past its cell, so the cell's own max
  doesn't bound it; use the 2x2x2 support: `helpers: ['cellBoundTrilinear']` gives `vdbCellBoundTrilinear`
  (twin in `tools/cpu/march.ts`).
- Build the bind group AFTER `pool.update(encoder)` in the same frame. A growth replaces `pool.buffer`; a bind
  group made before it binds the old buffer, which stays alive (retired), so there is no validation error, only
  a frame that misses the new grids.
- `vdbModuleEntries` defaults `visibility` to `COMPUTE | FRAGMENT` (K1.7a). A vertex-stage read must pass
  `visibility` explicitly.
- Opt-in WGSL helpers (`helpers: [...]`) are not emitted unless listed, so default module bytes (and the size
  snapshot) don't move when one is added. A new API function that every module needs is a snapshot change.
- A WebGPU canvas context fails on headless SwiftShader ("A valid external Instance reference no longer
  exists"). Raw-WebGPU demos need a readback presentation for e2e (demo 08 `?presentation=readback`).
- `VDBVolumeMaterial` needs a `VDBGridNode` made with `helpers: ['rayClipSupport']` (and profile `sample`+). The
  helper is opt-in so module bytes don't move; the material throws `VDB_BAD_OPTION` naming the fix.
- Clipping a march to the trilinear support box (`vdbRayClipSupport`) moves the ray entry, and a fixed-step
  lattice anchored there resamples the whole volume (pixels move everywhere, ~20 px on demo 02 pose 1).
  `VDBVolumeMaterial` anchors the lattice at the voxel-box entry (`vdbRayClip`) and steps back over the shell.
- `pool.replace` holds the frame array by reference until the update that writes it, and until `settled()`
  while a growth is unconfirmed. A sequence loader that recycles frame buffers must wait for `settled()`.
