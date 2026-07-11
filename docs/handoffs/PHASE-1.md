# Handoff: Phase 1 — `.nvdb` loader + first GPU read

Gate: **PASSED** — demo 01 `hello-nvdb` GPU compute probe reproduces all
73 native-accessor sidecar samples (|Δ| ≤ 1e-5, active flags exact) on
headless SwiftShader; full suite 67 unit tests + 2 e2e + clean tsc.

## What exists now

| Artifact | Where | Proven by |
|---|---|---|
| `NanoVDBFile` loader — segment (`NanoVDB0/2`) + raw-grid (`NanoVDB1`) formats, codec NONE + ZIP (u64 size prefix + zlib, fflate), alignment-aware zero-copy `gridImage()` | `packages/nanovdb-wgsl/src/nvdb-file.ts` | `test/nvdb-file.test.ts`: 9/9 fixtures vs sidecars, synthetic ZIP round-trip byte-identical, error paths |
| CPU reference traversal — root→upper→lower→leaf for Float/Fp8/FpN, offsets from `stride-tables.json` at runtime | `packages/nanovdb-wgsl/src/cpu/read-value.ts` | `test/cpu-read-value.test.ts`: 657/657 sidecar probes |
| Generated WGSL constants (162 defines + FLOAT/FP8/FPN blocks + 6 aux accessors) | `packages/nanovdb-wgsl/src/wgsl/pnanovdb-constants.generated.wgsl` | sync-guard regen test |
| Demo 01 — device-first bootstrap, storage upload, 73-thread compute probe, on-page results + `window.__DEMO01__` | `examples/src/demos/01-hello-nvdb/` | `e2e/demo-01.spec.ts` |

## Key decisions + why (spike findings — the Phase 1 deliverable)

1. **Feasibility risk #1 is RETIRED.** `wgslFn` with a
   `ptr<storage, array<u32>, read>` parameter binds directly from
   `storage(attr, 'uint', n).toReadOnly()`. Mechanism: TSL wraps the
   buffer as `struct { value: array<u32> }` and, for pointer-typed
   params, emits `&NodeBuffer.value` — exactly the right type and access
   mode. Dawn/SwiftShader accept storage-address-space pointers to user
   functions (`unrestricted_pointer_parameters`). **No fallback to
   pure-TSL authoring is needed; Phase 2 builds on this pattern.**
2. **D4 device-first pattern confirmed in code:** three r185
   `WebGPUBackend.init()` uses `parameters.device` when provided;
   `new WebGPURenderer({ device })` with our raised
   `maxStorageBufferBindingSize`/`maxBufferSize` works end to end.
3. **CI GPU testing is real:** headless Chromium + `--enable-unsafe-webgpu
   --enable-features=Vulkan --enable-unsafe-swiftshader` yields a
   software-Vulkan adapter (1 GiB storage binding). WebGPU is
   secure-context-only — navigate to the served page before probing.
   Compute+readback for the 73-probe dispatch: ~96 ms (compile-dominated).
4. WGSL authoring traps hit (for Phase 2): `active` is a reserved WGSL
   keyword; back-ticks inside WGSL comments break the enclosing JS
   template literal; `.wgsl` files clash with Vite's asset handling —
   shader source ships as TS string exports (`traversal.ts`).
5. The hand-written WGSL subset matched the CPU reference on its first
   syntactically-valid compile — zero layout divergences to audit.

## Verification (fresh-eyes agent, adversarial pass)

Independent gate re-run: 67/67 unit, clean tsc, 2/2 e2e — **verdict:
gate passes**. Cross-checks that came back clean: coord_to_key bit
packing (CPU BigInt and WGSL split-32), root-tile scan termination, FpN
bits-per-value derivation, leaf NEG offsets, every WGSL inlined offset
vs the generated constants, D2–D5 compliance, loader truncation
bounds-checking. Findings carried as debt (below): silent
`FileMetaData.gridSize` vs internal `GridData.mGridSize` mismatch
(truncated image, no error — silently wrong voxels downstream);
unbounded `unzlibSync` (zip-bomb shape — pass `{out}` sized to
gridSize); raw-grid-buffer (`NanoVDB1`) path has zero test coverage;
WGSL demo reads only the low 32 bits of three 64-bit child offsets
(fine at fixture scale, a CPU/GPU divergence risk on >4 GiB grids).

## Known debts

- ZIP fixtures are synthetic (test-built); no real `nanovdb_convert --zip`
  output exercised yet (needs the Docker image built — no daemon in the
  authoring sandbox).
- CPU package carries a narrow ambient `node:fs` type shim
  (`src/cpu/node-fs-ambient.d.ts`) instead of `@types/node` — revisit.
- The cpu test suite slices `.nvdb` with local scaffolding instead of
  `NanoVDBFile` (written concurrently); swap in Phase 2.
- Demo 01 is FLOAT-only by design; Fp8/FpN GPU decode lands with the
  full Phase 2 port.
- picovdb license question (PLAN §3 item 3) not yet filed upstream.

## Phase 2 entry points

- Port target: `packages/nanovdb-wgsl/vendor/pnanovdb.wgsl` audited
  line-by-line against `vendor/upstream/PNanoVDB.h` (both pinned; see
  `vendor/VENDOR.md`), extended with Fp4/8/16/FpN decoders, HDDA,
  readaccessor, stats readers, world↔index map.
- Baked constants: import/generate from
  `src/wgsl/pnanovdb-constants.generated.wgsl` (regen:
  `node scripts/gen-wgsl-constants.mjs`).
- Test harness: replay ALL sidecar samples (657) on GPU across
  Float/Fp8/FpN the way `e2e/demo-01.spec.ts` does for FLOAT — the CPU
  reference (`src/cpu/read-value.ts`) is the arbiter when GPU and sidecar
  disagree.
- Binding pattern: copy demo 01's `wgslFn`/storage incantation verbatim
  (`examples/src/demos/01-hello-nvdb/main.ts`).
