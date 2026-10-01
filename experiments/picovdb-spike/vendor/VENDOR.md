# Vendored: `picovdb.wgsl` + `picovdb.ts` (spike-only, unmodified)

**Evaluation spike, not a shipping dependency.** These files are vendored
under `experiments/picovdb-spike/` only, to produce evidence for a pending
owner decision (proposed D7, see [`docs/spikes/PICOVDB.md`](../../../docs/spikes/PICOVDB.md)).
Nothing under `packages/` imports them. If the owner adopts PicoVDB, the
vendoring moves into a package and follows [D2](../../../docs/DECISIONS.md):
pinned commit, NOTICE preserved, our fixes applied in-tree with a diff log.
We would be prepared to diverge permanently, and would upstream fixes if the
author engages.

## Pin

| | |
|---|---|
| Upstream | <https://github.com/emcfarlane/picovdb> |
| Files | `wgsl/picovdb.wgsl`, `ts/picovdb.ts` (plus repo-root `LICENSE`) |
| Pinned commit | `f5d22e5602a7fc7f206d33ec35328f58e8667c83` (2026-08-24, "Add GPU compute pipeline for grid editing (#11)") |
| Vendored | 2026-10-01 |
| License | Apache-2.0 (see `LICENSE`, added upstream 2026-07-15 in `0948edb`; attribution in `NOTICE`) |
| SHA-256 of `picovdb.wgsl` | `78f31ed0ef15f855a13e2a099505585665cfa47c0711c727989f6a633dcfbde0` |
| SHA-256 of `picovdb.ts` | `c736e64f27e71cc157c03bf2c7c4ba4820b514dce1a85dac5ea0df7cbbee738e` |
| SHA-256 of `LICENSE` | `6076ff3d4be9f35cc29763cb350b1008efb58dfa826fca6dd3013376977cabfb` |

The SHA-256 values are of the files **as vendored**. They are byte-identical
to upstream at the pinned commit. `test/vendor.test.ts` asserts them, so an
accidental edit fails the spike suite. A deliberate edit must update the hash
*and* the diff log below in the same change.

Not vendored: `ts/model.ts` (GPU CSG). It transitively imports 14 modules
under `ts/gpu/` and 11 WGSL kernels, and the spike only assesses it (see the report's
D6/CSG section). The native converter `src/main.zig` + `src/picovdb.zig` is
not vendored either. It was ported to TypeScript (`src/convert.ts`) and used
as a byte-level oracle from a scratch build (see the report's reproduction
section).

## Local diff log

Every in-tree change to a vendored file gets one row here, newest first.

| Date | Change | Reason | Upstreamed? |
|---|---|---|---|
| | *(none — files are verbatim)* | | |
