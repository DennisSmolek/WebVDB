# `.wvdb` test fixtures (K4.3)

Reference files for the `.wvdb` reader/writer tests (`test/tools/wvdb-*.test.ts`). Each `.wvdb` here is a byte
copy from [mjurczyk/openvdb](https://github.com/mjurczyk/openvdb) at commit `75e7345` (v0.26.0), written by his
`wvdb` CLI. Each is under 200 KB.

| File | Bytes | Source in his repo | Content licence |
|---|---:|---|---|
| `ground_explosion_0010.wvdb` | 93,234 | `src/docs/examples/public/assets/ground_explosion/` | CC0 (Ground Explosion, JangaFX EmberGen) |
| `ground_explosion_0199.wvdb` | 1,199 | same folder (an all-empty frame) | CC0 (as above) |
| `cloud_04_variant_0000.wvdb` | 86,879 | `src/docs/examples/public/assets/CloudPackVDB_wvdb/` | CC0 (Cloud Pack, JangaFX EmberGen) |
| `smoke.wvdb.digest.json` | — | digest of `src/docs/examples/public/assets/smoke.wvdb` (808,752 bytes, not committed) | derived from `smoke.vdb`, an openvdb.org sample (CC BY-SA 4.0) |

The JangaFX sets are CC0 1.0 (Public Domain); his repo carries their `LICENSE.txt` / `README.txt` next to the
files. The `.wvdb` encoding is his MIT-licensed tool's output. The digest records only the header (chunk offsets
zeroed) and SHA-256 sums of the inflated chunks, so the smoke re-bake test can run from our own
`fixtures/vdb-samples/smoke.vdb` (`pnpm fixtures:vdb-samples`) without committing his larger file.

Tests that need his other committed files (`venusstatue.wvdb`, `utahteapot.wvdb`, `sphere_points.wvdb`,
`smoke.wvdb`, `bunny_cloud.wvdb`) read them from a checkout named by `MJURCZYK_OPENVDB_DIR` (the repo root) and
skip when it is unset:

```sh
MJURCZYK_OPENVDB_DIR=/path/to/openvdb pnpm test
```
