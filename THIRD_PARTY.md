# Third-party code

Our code is MIT (D13); the third-party files below keep their own licences.

Code and data in this repo that we did not write, with its licence and where it lives. Keep this list in step
with each port, vendor drop or new test fixture (see CLAUDE.md, "Licences").

| Component | Licence | Where it lives | Notes |
|---|---|---|---|
| emcfarlane `pnanovdb.wgsl` (pinned @265e8d8) | Apache-2.0 | `packages/vdb/vendor/pnanovdb.wgsl`, `vendor/LICENSE`, `vendor/NOTICE`, `vendor/VENDOR.md` | Vendored and extended in place. Upstream lines are untouched; the diff log is in `VENDOR.md`. The generator reads chunks from this file (`core/wgsl/pnanovdb.chunks.generated.ts`). |
| OpenVDB `PNanoVDB.h` | Apache-2.0 | `packages/vdb/vendor/upstream/PNanoVDB.h` | Reference header for the vendored WGSL. Keeps its licence header. |
| mjurczyk/openvdb (pinned `75e7345`, v0.26.0) | MIT | `packages/vdb/src/tools/wvdb/format.ts`, `read.ts`, `write.ts`, `bake.ts` | Ported logic for the `.wvdb` reader, writer and baker. Each file carries the header "Portions derived from mjurczyk/openvdb (MIT), commit 75e7345". `src/tools/vdb/*` cites his reader as a cross-check only; nothing is ported from it. |
| `wgslNameProblems` (name checker) | MIT, Copyright (c) 2026 Dennis Smolek (`@pmndrs/bvh` LICENSE) | `packages/vdb/test/core/wgsl-names.ts`; the reserved-word list in `packages/vdb/src/core/wgsl/reserved.ts` | Ported from `@pmndrs/bvh` `src/wgsl-names.ts`. |
| `.wvdb` test fixtures (JangaFX) | CC0 1.0 | `packages/vdb/test/tools/fixtures/wvdb/` (`ground_explosion_0010.wvdb`, `ground_explosion_0199.wvdb`, `cloud_04_variant_0000.wvdb`) | Byte copies from mjurczyk/openvdb, written by his `wvdb` CLI. The fixtures README records the source paths. |
| `.wvdb` smoke digest | Derived from CC BY-SA 4.0 sample | `packages/vdb/test/tools/fixtures/wvdb/smoke.wvdb.digest.json` | Header and inflated-chunk SHA-256 sums only. The source file is not committed. `smoke.vdb` is an openvdb.org sample (CC BY-SA 4.0), fetched by `pnpm fixtures:vdb-samples`. |
| fflate | MIT | runtime dependency (`packages/vdb/package.json`) | Inflate and deflate in `tools`. |

## mjurczyk/openvdb: MIT notice

mjurczyk/openvdb has no `LICENSE` file in its repository. Its npm packages declare MIT, and the owner has
confirmed the licence. The notice below applies to the code we ported, as MIT requires.

```
MIT License

Copyright (c) mjurczyk/openvdb contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

Divergences from his code are recorded in [DIVERGES.md](DIVERGES.md) (V10, V16 to V19).

- `packages/vdb/src/tools/vdb/points.ts` — portions derived from mjurczyk/openvdb (`src/openvdb/core/PointDataBuffers.js`, `src/openvdb/utils/points.js`), commit 75e7345, MIT, © mjurczyk/openvdb contributors. File header: "Portions derived from mjurczyk/openvdb (MIT), commit 75e7345".
