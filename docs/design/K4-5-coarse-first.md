# K4.5: coarse-first `.nvdb` delivery (design note)

Status: design only, 2026-11-18. Numbers are measured with K4.4's `resampleGrid` + `quantize` + gzip
(`scripts/measure-coarse-first.mjs`), except where a row says "estimate".

**Goal (PHASES K4 exit gate):** the first paint of a streamed `.nvdb` is no larger than mjurczyk's smoke preview.
His `smoke.wvdb` needs **14.8 KiB** before LOD0 (32³) can be drawn (`wvdbPrefixLength(info, 0)` on our
byte-exact re-bake of his file; "14 KiB" in research/INDEX is the LOD0 chunk without the header).

**Recommendation: (a), a low-resolution preview grid first in the same multi-grid `.nvdb`, streamed with HTTP
compression.** On smoke the preview is drawable after **11.9 KiB** (box filter, FpN, `resolution: 48`), at the
same error as his LOD0 (47.6% vs 47.3%), and the whole file is 654 KiB against his 791 KiB, ending at 1.2% error
instead of 17%. The test that settles it is in [The test that settles it](#the-test-that-settles-it).

## How the numbers are taken

- **Sizes** are gzip level 9 (Node zlib) of the `.nvdb` file, standing in for HTTP compression, as in the
  comparison report. `.wvdb` is already deflated and is counted as is.
- **Bytes to first paint for a single file** is a *stream offset*: the number of gzip bytes of the whole file
  a streaming decoder must receive before the first grid (file header, its metadata and its image) is fully
  decoded. Measured by pushing the gzip stream through fflate's `Gunzip` in 64-byte steps. This is what a
  `fetch()` with `Content-Encoding: gzip` delivers to a stream parser.
- **Error** is `resampleError`: mean |preview − source| over every 7th active source voxel (≈150k on smoke), as a
  percentage of the mean, with the preview read back by `sampleTrilinear` at each source voxel centre (his
  `.wvdb` LODs are read back the same way, as a clamped texture). It is the comparison report's metric; the
  full-resolution FpN grid scores 1.21% on smoke (the report says 1.2%), which checks the harness.
- **Mass ratio** is Σ value × voxel volume of the preview over the source's (leaves and active tiles). A fog
  renderer's optical depth is proportional to it, so it is the first-order brightness/opacity error of a
  preview.
- Machine: owner's Mac, Node 22. Sizes are deterministic; no timings are claimed.

## Smoke (`smoke.vdb`, 111 × 222 × 112, 1.05 M active, 3,117 leaves)

Baselines:

| Encoding | Bytes before it can draw | Whole file | Error |
|---|---|---|---|
| His `.wvdb` LOD0 (32³) | 14.8 KiB | 791.1 KiB | 47.3% |
| His `.wvdb` LOD1 (64³) | 116.8 KiB | | 32.1% |
| His `.wvdb` LOD2 (128³) | 791.1 KiB | | 17.0% |
| Our full-resolution FpN `.nvdb` (no preview) | 642.8 KiB | 642.8 KiB | 1.21% |
| Our FpN resampled to his 32³ lattice (box) | 12.5 KiB | | 47.3% |
| Our FpN resampled to his 128³ lattice (point / box) | 456.2 / 477.2 KiB | | 17.3% / 18.3% |

Option (a): preview grid first, then the full FpN grid, in one file (uniform voxels, `resolution` = voxels on
the longest side; FpN; the full grid after it is the 642.8 KiB FpN grid):

| `resolution` | filter | dims | preview gz | **(a) first paint** | (a) whole file | error | mass ratio |
|---|---|---|---|---|---|---|---|
| 24 | box | 12×24×13 | 2.6 KiB | 3.0 KiB | 645.6 KiB | 61.4% | 0.998 |
| 32 | point | 16×32×17 | 4.4 KiB | 4.8 KiB | 647.3 KiB | 57.7% | 1.069 |
| 32 | box | 16×32×17 | 4.3 KiB | 4.6 KiB | 647.2 KiB | 55.9% | 0.998 |
| 48 | point | 24×48×25 | 11.4 KiB | 11.7 KiB | 654.2 KiB | 48.0% | 1.000 |
| **48** | **box** | 24×48×25 | 11.6 KiB | **11.9 KiB** | 654.4 KiB | 47.6% | 0.998 |
| 48 | trilinear | 24×48×25 | 10.9 KiB | 11.3 KiB | 653.9 KiB | 46.1% | 1.010 |
| 64 | box | 32×64×33 | 21.9 KiB | 22.2 KiB | 664.9 KiB | 41.2% | 0.998 |

`resolution: 48` (28,800 voxels in the box) is about his LOD0's voxel count (32,768) and fits the gate with
2.9 KiB to spare. Fp8 previews are 0–50% larger than FpN at the same resolution (34% at `resolution: 48`, box); use FpN. Raw (uncompressed)
sizes are in [Transfer compression is required](#transfer-compression-is-required).

Option (c) (render the topology): the prefix before the first leaf is 331.2 KiB raw, **10.2 KiB** in the gzip
stream (the 270 KB upper node is nearly all zeros). What it can draw is a blocky silhouette: the lower-node child
masks say which 8³ blocks have data, and the only values are 2 lower-node averages. That proxy scores **123%**
error on smoke, worse than drawing nothing (100%).

## WDAS quarter cloud (`wdas_cloud_quarter.vdb`, 24.1 M active, 41,312 leaves, 16,309 lower tiles)

Full-resolution FpN: 12,689.5 KiB gz, 0.17% error (Fp8 13,914 KiB). WDAS has no `.wvdb` baseline and the gate
is defined on smoke only; this is the "large asset" check. FpN previews (Fp8 is 0–4% larger here):

| `resolution` | filter | dims | preview raw | preview gz | **(a) first paint** | error | mass ratio |
|---|---|---|---|---|---|---|---|
| 16 | point | 13×9×16 | 300.7 KiB | 1.4 KiB | 1.5 KiB | 46.4% | 0.936 |
| 16 | box | 13×9×16 | 301.4 KiB | 2.0 KiB | 2.1 KiB | 43.6% | 1.000 |
| 32 | point | 26×18×32 | 312.3 KiB | 4.9 KiB | 5.1 KiB | 36.2% | 1.010 |
| 32 | box | 26×18×32 | 313.4 KiB | 7.0 KiB | 7.3 KiB | 35.1% | 1.000 |
| 48 | point | 39×27×48 | 335.5 KiB | 12.6 KiB | **13.3 KiB** | 31.9% | 0.995 |
| 48 | box | 39×27×48 | 336.6 KiB | 17.1 KiB | 17.2 KiB | 31.4% | 1.000 |
| 64 | box | 52×36×64 | 373.3 KiB | 33.6 KiB | 34.4 KiB | 28.4% | 1.000 |
| 128 | point | 104×71×128 | 694.3 KiB | 162.6 KiB | 163.8 KiB | 19.4% | 1.000 |
| 128 | box | 104×71×128 | 706.2 KiB | 194.3 KiB | 195.3 KiB | 20.4% | 1.000 |

The preview adds 0.1% (48) to 1.5% (128) to the 12.7 MiB whole file. Under 14 KiB WDAS gets `resolution` 48
with point (13.3 KiB, 31.9%) or about 40 with box. The box filter keeps more active voxels at the soft cloud
edge, so it costs more bytes per resolution here than on smoke; at equal bytes the two filters are within about a
point of error. Option (c): the topology prefix is 3,833 KiB raw and **130.6 KiB** in the gzip stream (8 upper
nodes); its proxy scores 41.1% on leaf voxels (the 16,309 lower tiles are exact in the prefix), i.e. worse than
the 2.1 KiB box preview at `resolution` 16.

## The four options

### (a) Preview grid(s) in the same `.nvdb`, coarsest first

A NanoVDB file is a sequence of grid segments (`FileHeader` + per grid `FileMetaData` + name + image). Write
`[preview, full]` (or a chain, `[48, 128, full]`), coarse first. The preview has the full grid's world box
(`resampleGrid` keeps the box and writes the Map for the new voxel size), so both draw in the same place.

- **Bytes to first paint:** 11.9 KiB on smoke (above). The preview costs +1.8% on the whole file.
- **Complexity:** low. The bake is `resampleGrid(grid, { resolution, quantize: 'fpn' })` + `writeNvdb([preview,
  full])` (exists). The reader needs a streaming parser: a `NvdbStreamReader` that takes pushed bytes and emits
  each grid image as soon as its `FileMetaData.gridSize` bytes have arrived (the framing is fixed-size headers
  plus a known `gridSize`, so this is ~100 lines next to `NanoVDBFile`, mirroring `WvdbStreamReader`). No range
  requests are needed: a plain `fetch()` body stream with `Content-Encoding: gzip` (or `br`) arrives in order.
- **Native readers:** the file is valid NanoVDB; multi-grid files are the format's normal case
  (`nanovdb::io::readGrids`, `nanovdb_print` lists every grid, `nanovdb_convert`/`nanovdb_viewer` read them). The
  risk is readers that take **grid 0** by default (`nanovdb::io::readGrid(file)` with no name or index reads
  grid 0): they get the preview. Mitigation: the full grid keeps the source name (`density`) and previews get a
  suffixed name (`density.lod48` style), so readers that select by name get the full grid. As far as we know,
  Houdini and Blender import `.vdb`, not `.nvdb`, so the question is mostly about NanoVDB's own tools and
  `.nvdb` consumers such as EmberGen. None of this is **verified on native tools** yet; see the settling test.
- **K1 pool / K3 renderer:** `id = pool.add(preview, { capacity: fullImageBytes })` as soon as the preview parses;
  when the full grid arrives, `pool.replace(id, full)`. The id, the bind-group slot and the instance transform
  stay; the grid record's affine is recomputed from the new header, so the world placement is unchanged.
  Reserving `capacity` up front makes the swap in place (the full size is in the full grid's `FileMetaData`,
  which arrives right after the preview). `replace` drops a brick majorant, so set it again after the swap. The
  renderer needs nothing new; the march step may be tied to the voxel size, so re-derive it after the swap.

### (b) A separate preview file

`smoke.preview.nvdb` (single grid) next to `smoke.nvdb`.

- **Bytes to first paint:** the preview file alone, 11.6 KiB gz at `resolution: 48` (box, FpN), plus one extra
  request's headers. Fetched first, the full file starts one round trip later; fetched in parallel, the two
  responses share the link, so the preview can take up to about twice as long to arrive as in (a), depending on
  HTTP/2 prioritisation.
- **Complexity:** lowest (no streaming parser; `NanoVDBFile.fromURL` twice). The app must know or derive the
  second URL, and caches and CDNs keep two objects in step.
- **Native readers:** no risk; both files are ordinary single-grid `.nvdb`.
- **Pool/renderer:** the same `add` then `replace` as (a).

### (c) NanoVDB-native LOD from upper/lower node stats

Render node averages as a coarse proxy without extra data: the file's breadth-first layout puts GridData, the
root, the upper nodes and the lower nodes before the leaves.

- **Bytes to first paint:** smoke 10.2 KiB in the gzip stream (331.2 KiB raw); WDAS 130.6 KiB (3,833 KiB raw).
  Leaf-level averages are not in that prefix: they sit in each leaf's header (16 bytes per leaf, interleaved
  with the leaf data), so they cost the whole leaf array.
- **Quality:** what the prefix can draw is lower-node resolution (128 voxels) values on an 8-voxel occupancy
  mask (plus lower-level active tiles, which are exact). Error 123% on smoke, 41.1% on WDAS leaves. Unusable as
  a first paint on wispy sources; acceptable only as a bounding silhouette.
- **Complexity:** a streaming parser (as in (a)) plus a proxy renderer path (a special grid built from the
  topology, or a shader that stops at lower nodes). No bake.
- **Native readers:** no file change, so full compatibility.
- **Pool/renderer:** a synthetic proxy image to `add`, then `replace`, or a renderer mode; both are new code.

### (d) Sidecar index for range reads of some leaves

A small JSON or binary index (leaf byte offsets per brick region) so a client range-reads the topology and then
only some leaves (the ones in view, or every k-th).

- **Bytes to first paint:** range responses are usually not compressed by static hosts and CDNs (a byte range
  addresses the stored bytes), so the topology alone is 331 KiB raw on smoke and 3.8 MiB on WDAS before
  any leaf. A subset of leaves draws holes, not a coarse field. Fails the 14 KiB gate by >20×.
- **Complexity:** highest: a sidecar format, a range scheduler, partial-grid images (a grid missing leaves is not
  a valid NanoVDB tree for the pool without patching child offsets).
- **Native readers:** unaffected (the `.nvdb` is unchanged), but the sidecar is ours alone.
- **Pool/renderer:** needs partial-grid upload or per-leaf `writeBuffer`s; this is the streaming-residency
  problem (out of scope before 0.1).

## Comparison

| Option | Smoke first paint | WDAS first paint | Preview error (smoke) | Extra bytes on the whole download | Native compatibility | New code |
|---|---|---|---|---|---|---|
| (a) preview grid first, one file | **11.9 KiB** (`resolution` 48, box, FpN) | 13.3 KiB (48, point) / 7.3 KiB (32, box) | 47.6% | +1.8% (smoke) | Valid NanoVDB; grid-0 readers see the preview (verify) | stream parser (~100 lines) |
| (b) separate preview file | 11.6 KiB + 1 request | 12.6 KiB (48, point) + 1 request | 47.6% | +1.8% + headers | Full | none in the reader |
| (c) topology as proxy | 10.2 KiB (gzip stream), 331 KiB raw | 130.6 KiB | 123% | 0 | Full | stream parser + proxy path |
| (d) sidecar + leaf ranges | ≥ 331 KiB raw | ≥ 3.8 MiB raw | holes | sidecar | Full (sidecar is ours) | sidecar format, range scheduler, partial grids |

## Transfer compression is required

A NanoVDB upper node is 270,400 bytes (a 32,768-slot table), mostly zeros for sparse data, so even a 24×48×25
preview is 320.9 KiB uncompressed (smoke, FpN; 299 KiB even at 8×16×9). Every first-paint number for (a) and (b) assumes the response is
compressed: `Content-Encoding: gzip`/`br` from the host, or a pre-compressed `.nvdb.gz` served with that
header. Two ways to remove the dependency, both follow-ups:

1. `writeNvdb` with NanoVDB's own `Codec.ZIP` per grid segment (our reader already decodes ZIP; native tools
   read it when built with zlib). The preview segment is then ~gzip size inside the file, whatever the host does.
2. A smaller upper node is not possible in NanoVDB (the 5-4-3 tree is fixed).

## Recommendation

Choose **(a)**. It meets the gate on smoke with margin (11.9 KiB vs 14.8 KiB, same preview error), keeps one URL
and one cache entry, needs no range requests, uses only `resampleGrid` + `writeNvdb` on the bake side and a small
stream parser on the read side, and lands on the pool through the `add`/`replace` path that already exists.
(b) is the fallback if the native check below finds a grid-0 reader that matters; it uses the same bake. (c) and
(d) do not produce a usable first paint at this size.

Defaults to bake: `resampleGrid(grid, { resolution: 48, filter: 'box', quantize: 'fpn' })`, preview named
`<name>.lod48`, written before the full grid. A three-level chain (`48` → `128` → full) is the same mechanism if
a middle step is wanted (smoke: 11.9 KiB → +136 KiB → +643 KiB).

### The test that settles it

1. **Native compatibility (owner machine, Docker image of `pnpm parity:native`).** Write
   `[smoke.lod48, density]` with `writeNvdb`; run `nanovdb_print` (both grids listed, sizes and bboxes match our
   `FileMetaData`), `nanovdb_convert` back to `.vdb` (both grids convert), and a 10-line C++ check that
   `io::readGrid(file, "density")` returns a grid whose values equal the single-grid file's. Also load the file
   in EmberGen (or any `.nvdb` consumer the owner uses) and record which grid it shows. Pass: every tool reads
   the file, and name-based reads return the full grid. If a consumer that matters shows grid 0, switch to (b).
2. **Bytes to first paint in a browser (e2e).** Serve the file gzip-encoded from the Vite server, read
   `fetch().body` with the stream parser, and record the compressed bytes received (`PerformanceResourceTiming`
   `encodedBodySize` at the moment the preview grid is emitted, or a counting `TransformStream` on a raw
   `.nvdb.gz` + `DecompressionStream`). Pass: ≤ 14.8 KiB on smoke, then `pool.replace` swaps in the full grid and
   the frame matches the full-grid golden.
3. **Preview filter A/B (CLAUDE.md "build both, then decide").** Render the `resolution: 48` preview with
   `'box'` and `'point'` against the full-resolution render (PSNR), and the same under a half-voxel translation
   of the box (temporal stability). `'box'` stays the default unless `'point'` wins both.
