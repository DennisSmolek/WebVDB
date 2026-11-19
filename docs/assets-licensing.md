# Gallery assets: licence review

The examples gallery (demos 10–16) uses assets from mjurczyk/openvdb's examples site (pinned 75e7345,
`src/docs/examples/public/assets/`). His repo is MIT for code, but these assets carry no attribution. They live in the
git-ignored `fixtures/gallery/` for local development only and must not be committed or deployed until each row has a
confirmed source and licence. If a source can't be found, swap the asset for a known-licence one.

Owner action: fill in **Source** and **Licence**, then set **Status** to `ok` (commit allowed), `swap` (replace it) or
`drop`.

| Asset | Size | Likely origin (unverified guess) | Source | Licence | Status |
|---|---|---|---|---|---|
| `sky-midday-HDR.jpg` (+ `-1K`) | 576 KB | HDRI site (Poly Haven?), UltraHDR JPEG | | | pending |
| `sky-cloudy-HDR.jpg` (+ `-1K`) | 540 KB | HDRI site (Poly Haven?) | | | pending |
| `sky-dusk-HDR.jpg` (+ `-1K`) | 444 KB | HDRI site (Poly Haven?) | | | pending |
| `sky-night-HDR.jpg` (+ `-1K`) | 568 KB | HDRI site (Poly Haven?) | | | pending |
| `shore-HDR.jpg` (+ `-1K`) | 2.2 MB | HDRI site (Poly Haven?) | | | pending |
| `desert-HDR.jpg` (+ `-1K`) | 1.1 MB | HDRI site (Poly Haven?) | | | pending |
| `grassland-HDR.jpg` (+ `-1K`) | 1.7 MB | HDRI site (Poly Haven?) | | | pending |
| `townsquare-HDR.jpg` (+ `-1K`) | 2.2 MB | HDRI site (Poly Haven?) | | | pending |
| `uv-1-HDR.jpg` (+ `-1K`), `uv-1-4K.jpg`, `uv-1-8K.jpg` | 0.3–3.3 MB | UV checker / test texture | | | pending |
| `damaged_helmet.glb` | 3.9 MB | Khronos glTF sample "DamagedHelmet" (check its licence and credit line) | | | pending |
| `car.glb` | 5.4 MB | unknown | | | pending |
| `lighthouse.glb` | 2.4 MB | unknown | | | pending |
| `shuttle.glb` | 3.7 MB | unknown | | | pending |
| `airbus.glb` | 1.3 MB | unknown | | | pending |
| `depth-testing.glb` | 924 KB | unknown (possibly his own test scene) | | | pending |
| `boat_points.wvdb` | 17 MB | unknown point simulation | | | pending |
| `CloudPackVDB_wvdb/` | 1.0 MB | JangaFX cloud pack | JangaFX (EmberGen), LICENSE.txt in the folder | CC0 1.0 | ok |

Already handled elsewhere (not part of this review):

- `ground_explosion/` — JangaFX EmberGen, CC0, with LICENSE.txt (`fixtures/k3/`, fetched locally).
- `smoke.vdb`, `sphere.vdb`, `cube.vdb`, `utahteapot.vdb`, `bunny_cloud.vdb`, `venusstatue.vdb`, `sphere_points.vdb` (and, for gallery
  demos 13–14, `bunny.vdb`, `icosahedron.vdb`, `waterfall_points.vdb`) —
  openvdb.org samples, CC BY-SA 4.0, fetched by `pnpm fixtures:vdb-samples` (not committed; Stanford-bunny derivatives
  excluded from CI).
- Their `.wvdb` bakes are his derivatives of those samples (same licence as the source).
