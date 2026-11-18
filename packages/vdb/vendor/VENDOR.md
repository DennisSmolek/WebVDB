# Vendored: `pnanovdb.wgsl` (fork) + `upstream/PNanoVDB.h` (reference)

Per [D2](../../../docs/DECISIONS.md): we adopt the Apache-2.0 WGSL port of
PNanoVDB and **vendor it as a fork from day one** — pinned commit, NOTICE
preserved, our fixes applied in-tree with a diff log — rather than depending
on the upstream repo. We are prepared to diverge permanently and happy to
upstream fixes if the author engages.

## Pin

| | |
|---|---|
| Upstream | <https://github.com/emcfarlane/webgpu-nanovdb> |
| File | `pnanovdb.wgsl` (repo root) |
| Pinned commit | `265e8d825e4e4ab8752196a28cccad592d9b4262` |
| Vendored | 2026-07-11 |
| License | Apache-2.0 (see `LICENSE`; attribution in `NOTICE`) |
| SHA-256 of vendored file | `76021f6a76256cd009f22395af4430d9dbf538e4eb3b7eb32d62b6095054d7e6` |

The SHA-256 above is of the file **as vendored**. It started byte-identical
to upstream at the pinned commit; the Phase 2 additions logged below (all
appended after the upstream body — no upstream lines were modified) have
since advanced it. It is asserted by `test/vendor.test.ts` so accidental
edits fail CI; deliberate edits update the hash *and* the diff log below in
the same commit.

## Upstream ABI reference: `upstream/PNanoVDB.h`

The port tracks `PNanoVDB.h` in `PNANOVDB_ADDRESS_32` mode. The reference
header itself is vendored (read-only, never edited) so the Phase 2
line-by-line audit (see `docs/PLAN.md`) — Fp4/Fp8/Fp16/FpN decoders, the
HDDA family, readaccessor completeness, stats readers, world↔index map —
diffs against a fixed target:

| | |
|---|---|
| Upstream | <https://github.com/AcademySoftwareFoundation/openvdb> |
| File | `nanovdb/nanovdb/PNanoVDB.h` |
| Pinned commit | `a532de5526ef791280b6483a872336a811a68542` |
| ABI | 32.9.1 (`PNANOVDB_MAJOR/MINOR/PATCH_VERSION_NUMBER`) |
| Vendored | 2026-07-11 |
| License | Apache-2.0 (SPDX header in file) |
| SHA-256 | `083a61491472bd2b008d5c3406aea0b1ac047c031b251047a5e38dac29f17410` |

## Extracted stride tables: `stride-tables.json`

Generated from the vendored header by `scripts/extract-stride-tables.mjs`
(integer defines, grid-type ids, the `pnanovdb_grid_type_constants` table,
and the auxiliary per-type stride arrays). Committed so Phase 2's WGSL
baked-const codegen has a reviewable input; `test/stride-tables.test.ts`
fails if it drifts out of sync with the header. Regenerate — never
hand-edit — after bumping the header.

## Local diff log

Every in-tree change to `pnanovdb.wgsl` gets one row here, newest first.

| Date | Change | Reason | Upstreamed? |
|---|---|---|---|
| 2026-07-11 | Phase 2 audit: line-by-line diff of the existing port against `upstream/PNanoVDB.h` (constants table, coord_to_key, readaccessor cache, child getters, HDDA, is_active). No functional deviations found; no edits to upstream lines. | Establish audited baseline before extending. | N/A (no change) |
| 2026-07-11 | Add stats readers: `{root,lower,leaf}_get_{min,max,ave,stddev}_address` + `root_get_background_address` (upstream parity; `upper_*` already present), plus `{root,upper,lower,leaf}_get_{min,max,ave,stddev}_float` f32 convenience readers. | Renderer needs node min/max for shading/empty-space skipping. | Address getters yes; `_float` readers are WebVDB. |
| 2026-07-11 | Add trilinear sampling (WebVDB extension): `pnanovdb_sample_trilinear_{float,fp8,fpn}` + shared `_typed` impl (8 accessor taps, lerp, background outside). | Continuous index-space sampling has no PNanoVDB.h counterpart (NanoVDB SampleFromVoxels is C++ only). | No (WebVDB). |
| 2026-07-11 | Add world↔index map: `map_get_vecf`, `map_apply{,_inverse,_jacobi,_inverse_jacobi}`, `grid_{world_to_index,index_to_world}{,_dir}f`. Reuses the fork's mat3x3f getter via WGSL vector*matrix (`src * M`) since the vendored getter loads PNanoVDB row-major matf into mat columns. | Ray/gradient transforms between world and index space. | Yes (transliteration). |
| 2026-07-11 | Add per-grid-type value dispatch (WebVDB extension): `read_float_typed` + `readaccessor_get_value_float` — single runtime `grid_type` switch feeding the FP decoders; the public sample API specializes as thin wrappers over it. | Decode FLOAT/Fp8/Fpn voxels to f32 in one accessor call. | No (WebVDB; wraps upstream). |
| 2026-07-11 | Add Fp4/Fp8/Fp16/FpN leaf decoders: `leaf_fp_read_float` (shared) + `leaf_fp{4,8,16,n}_read_float` + `root_fp{4,8,16,n}_read_float` (level-aware), and helpers `address_offset_neg`, `uint32_to_float`. Transliterated from PNanoVDB.h §"Leaf FP Types"; cross-checked against `src/cpu/read-value.ts` (657/657 validated). | Quantized grid support required by the renderer. | Yes (transliteration). |

## Generation transforms

`core/wgsl/generate.ts` builds each WGSL module from a declaration table that
`scripts/extract-wgsl-chunks.mjs` extracts from the **unmodified** vendored
file (`src/core/wgsl/pnanovdb.chunks.generated.ts`; sync test
`test/core/wgsl-chunks.test.ts`). These transforms are applied to generated
output only: they are not edits to `pnanovdb.wgsl`, its SHA-256 or the diff log
above. One row per transform (design note `docs/design/K1-core.md` §3.3).

| Since | Declaration | Becomes | Why |
|---|---|---|---|
| 2026-11-18 (K1.4a) | every declaration | dropped unless reachable from the profile's API layer (dependency closure); comments removed at extraction | Unused features compile out (D10); the closure is the profile. |
| 2026-11-18 (K1.4a) | `pnanovdb_buf_read_uint32` | `return vdb_load((buf.byte_offset >> 2u) + (byte_offset >> 2u));` (same arithmetic as `/ 4u`) | The module's one buffer access site; `vdb_load` reads the `slot` binding or the host's `buffer` expression (§2, §3.1). |
| 2026-11-18 (K1.4a) | `pnanovdb_grid_type_constants` (27-row const array) and its `[grid_type]` call sites | `fn vdb_gtc(grid_type: u32)` returning only the compiled types' rows; fields no emitted declaration reads are written `0u`, and types whose rows are then equal share one branch (one distinct row: a constant, no branch). Call sites become `vdb_gtc(grid_type)`. | The array is 5.3 KB minified and dynamically indexed; FXC/DXC materialize dynamically indexed const arrays. Row values are copied from the vendored table at generation. |
| 2026-11-18 (K1.4a) | `pnanovdb_read_float_typed` | only the compiled Fp types' branches; Float is the fallthrough, or, with no Float, the last compiled Fp type is | Decoders of types not compiled drop out of the closure. Reads of other types never get here (`VdbAccessor.ok`). |
| 2026-11-18 (K1.4a) | `pnanovdb_sample_trilinear_{float,fp8,fpn}` | not emitted (unreachable) | The API layer's `vdbSample` calls `pnanovdb_sample_trilinear_typed` with the grid's runtime type. |
| 2026-11-18 (K1.4a) | every `pnanovdb_` / `PNANOVDB_` identifier | `vdb_` / `VDB_` | One prefix for the whole module (D10, D17). |
