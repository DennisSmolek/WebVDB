# STATUS

- **0.1.0 release gates complete on device.** The four source VDB revisions are
  size/SHA-256 pinned. Strict native parity independently compares both
  NanoVDB trees, their active-coordinate union, and canonical topology with no
  unknown byte fields. Demo 05 validates a full 100-frame cycle, and
  packed-package consumers pass.
- **Verification:** 249/249 unit, clean TypeScript, package build green,
  strict FLOAT/FP8 native parity green, hardware smoke green, and all three
  `npm publish --dry-run` commands green. Mac execution passes 10/11 canonical
  Linux/SwiftShader E2E screenshots; Demo 06 has a 3% cross-OS visual diff
  against the 2% threshold, while its functional assertions pass.
- **Release artifacts:** `artifacts/npm/{nanovdb-wgsl,three-nanovdb,vdb-web-tools}-0.1.0.tgz`.
  They install together in a clean external consumer. Default and
  `--conditions=development` imports both resolve to packaged `dist/`; workspace
  source resolution is provided only by local TypeScript/Vite/Vitest aliases.
- **Deferred to v1.1:** HDDA march node-skipping and optional companion-service
  hosting. Three/Chrome GPU timestamp values were invalid on this adapter, so
  the recorded WDAS gate uses queue-synchronized frame time and a controlled
  shadow delta.
- **Hardware evidence:** each Demo 02, Demo 04, and performance lane validates
  fixture, adapter, presentation/texture mode, shims, resolution, and quality
  state. The current WDAS 1080p result is 103.96 FPS.
- **Manual action:** publish packages when registry ownership, 2FA, and
  provenance are ready. No package has been published yet.
- **Read first on resume:** this file → `docs/ON-DEVICE.md` →
  `docs/handoffs/PHASE-3.md` / `PHASE-5.md`.
