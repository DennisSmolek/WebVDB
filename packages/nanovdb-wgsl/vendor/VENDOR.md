# Vendored fork: `pnanovdb.wgsl`

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
| SHA-256 of vendored file | `34e95eced0f03133c3b90fea4bfc8b517c228fb9525eb397763277379e6d772c` |

The SHA-256 above is of the file **as vendored** (identical to upstream at
the pinned commit). It is asserted by `test/vendor.test.ts` so accidental
edits fail CI; deliberate edits update the hash *and* the diff log below in
the same commit.

## Upstream ABI reference

The port tracks `PNanoVDB.h` in `PNANOVDB_ADDRESS_32` mode (NanoVDB
ABI 32.x). The Phase 2 line-by-line audit (see `docs/PLAN.md`) will extend
it with Fp4/Fp8/Fp16/FpN decoders, the HDDA family, readaccessor
completeness, stats readers, and world↔index map helpers.

## Local diff log

Every in-tree change to `pnanovdb.wgsl` gets one row here, newest first.

| Date | Change | Reason | Upstreamed? |
|---|---|---|---|
| — | *(none yet — file is byte-identical to upstream at the pinned commit)* | | |
