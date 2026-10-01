# Spike: PicoVDB evaluation (T3)

*Evaluation spike, 2026-10-01. Pinned upstream: [emcfarlane/picovdb](https://github.com/emcfarlane/picovdb)
@ `f5d22e5602a7fc7f206d33ec35328f58e8667c83` (2026-08-24), Apache-2.0 since
2026-07-15. Code: [`experiments/picovdb-spike/`](../../experiments/picovdb-spike/). Nothing
under `packages/` was touched. This document is evidence for a decision the
owner makes. It decides nothing.*

## TL;DR — recommendation: **wait** (do not adopt now; re-evaluate on named triggers)

PicoVDB works, and porting its converter to TypeScript is straightforward: our
TS converter is **byte-identical to the native Zig converter on all 10 fixture
outputs**, and the vendored WGSL compiles clean and reads correctly on a real
WebGPU device. But on the decisive question for our main goal (fog clouds),
**PicoVDB float loses to NanoVDB Fp8: the realistic fog sample (smoke) is 218%
of Fp8 raw and 219% gzipped, and 346% of FpN.** PicoVDB has no quantized fog
encoding. Its `u8` mode is SDF-only, and fog + u8 is broken upstream. Its wins
are on level sets (teapot u8 is 54% of Fp8) and on tiny grids where NanoVDB's
mostly empty upper nodes dominate. The spike also found **two upstream
read-correctness defects that hit fog specifically**:
1. The converter mis-indexes active internal tiles, giving wrong density on up to 26% of probes in our fog primitives.
2. The WGSL root search can match the even-count padding root.

On top of that, the format carries **no world transform**, and the on-disk
layout has changed ≥4 times under a never-bumped `version` field. The GPU CSG
(`model.ts`) is SDF-only, index-space, and needs >8 storage buffers, so it
cannot retire the fog side of the D6 "CSG / composites" endpoint. Suggested
course: keep NanoVDB Float/Fp8/FpN as the only shipping GPU format, send the
fixes and the requests listed below upstream (the owner has a line to the
author), and keep this spike as the re-evaluation harness. **Adopt as an
optional second format** only if level sets or SDF CSG become a priority, or
if upstream ships quantized fog that beats Fp8 in `scripts/sizes.mjs`.

## 1. Measured: size

`node experiments/picovdb-spike/scripts/sizes.mjs`. Every PicoVDB number is
the output of the TS converter, which is byte-identical to native (§3). "Fp8
(TS)" and "FpN (TS)" are `vdb-web-tools` `quantize` of the same FLOAT grid. Our
quantizer expands active tiles into leaves, which is why it is somewhat larger
than the native bake on the primitives. The vdb-samples go `.vdb` →
`parseVdb` → `buildFromVdb` (leaf-only, no internal tiles) → `writeNvdb`.

| Fixture | Class | Active voxels | Nodes U/L/Leaf | NanoVDB float | NanoVDB Fp8 (TS) | Fp8 (native bake) | NanoVDB FpN (TS) | **PicoVDB f32** | PicoVDB u8 | Pico f32 ÷ Fp8 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| primitives/sphere_fog | fog | 523,305 | 8/8/788 | 4028 KiB | 3199 KiB | 2846 KiB | 2905 KiB | **1127 KiB** | n/a | 35% |
| primitives/torus_fog | fog | 738,028 | 8/8/1367 | 5240 KiB | 3587 KiB | 3190 KiB | 3245 KiB | **1936 KiB** | n/a | 54% |
| primitives/box_fog | fog | 202,581 | 8/8/336 | 3082 KiB | 2692 KiB | 2578 KiB | 2488 KiB | **585 KiB** | n/a | 22% |
| vdb-samples/sphere | sdf | 270,638 | 8/8/1451 | 5416 KiB | 3240 KiB | — | 2856 KiB | **1460 KiB** | 667 KiB | 45% |
| vdb-samples/cube | sdf | 1,452,218 | 8/8/6812 | 16641 KiB | 6423 KiB | — | 4324 KiB | **7165 KiB** | 2910 KiB | 112% |
| **vdb-samples/smoke** | **fog** | 1,049,275 | 1/2/3117 | 6857 KiB | **2182 KiB** | — | **1372 KiB** | **4747 KiB** | n/a | **218%** |
| vdb-samples/utahteapot | sdf | 6,960,047 | 8/104/35099 | 79041 KiB | 26392 KiB | — | 22731 KiB | **34571 KiB** | 14180 KiB | 131% |

Gzipped (level 6), i.e. bytes over the wire (upstream ships `.pvdb.gz`):

| Fixture | NanoVDB float | NanoVDB Fp8 | NanoVDB FpN | **PicoVDB f32** | PicoVDB u8 | Pico f32 ÷ Fp8 |
|---|--:|--:|--:|--:|--:|--:|
| primitives/sphere_fog | 229 KiB | 137 KiB | 127 KiB | **180 KiB** | — | 132% |
| primitives/torus_fog | 669 KiB | 280 KiB | 274 KiB | **589 KiB** | — | 210% |
| primitives/box_fog | 12 KiB | 10 KiB | 8 KiB | **7 KiB** | — | 71% |
| vdb-samples/sphere | 609 KiB | 346 KiB | 153 KiB | **491 KiB** | 286 KiB | 142% |
| vdb-samples/cube | 142 KiB | 73 KiB | 57 KiB | **88 KiB** | 61 KiB | 121% |
| **vdb-samples/smoke** | 2694 KiB | 1128 KiB | 647 KiB | **2472 KiB** | — | **219%** |
| vdb-samples/utahteapot | 17885 KiB | 7715 KiB | 7136 KiB | **16621 KiB** | 6715 KiB | 215% |

Where the bytes go (KiB, raw): NanoVDB Fp8 vs PicoVDB f32:

| Fixture | Fp8 upper | Fp8 lower | Fp8 leaf | Pico upper | Pico lower | Pico leaf masks | Pico values |
|---|--:|--:|--:|--:|--:|--:|--:|
| primitives/sphere_fog | 2113 | 265 | 821 | 96 | 12 | 160 | 859 |
| vdb-samples/smoke | 264 | 66 | 1851 | 12 | 3 | 633 | 4099 |
| vdb-samples/utahteapot | 2113 | 3439 | 20840 | 96 | 158 | 7129 | 27188 |

**Reading the tables.**
- **Fog verdict: no, PicoVDB float does not beat NanoVDB Fp8.** It loses
  on the only realistic fog asset (smoke: 2.2× Fp8, 3.5× FpN, raw and
  gzipped).
- The primitive "wins" (22–54%) are an artifact of tiny grids straddling the
  origin. They have 8 root tiles, so 8 near-empty NanoVDB upper nodes at
  264 KiB each (2.1 MiB of a 3.2 MiB Fp8 grid), which PicoVDB's rank-compressed
  masks shrink to 12 KiB each. Those empty tables gzip to almost nothing, so
  the advantage mostly disappears on the wire (sphere: 132%).
- On dense data the cost is values: PicoVDB stores 4 B per active voxel against
  Fp8's 1 B per leaf slot. Per dense leaf that is ≈2.2 KiB vs ≈0.6 KiB.
- Upstream's headline ("50%+ smaller than NanoVDB") reproduces **against
  NanoVDB float**: PicoVDB is 27–69% of float here (SDF sphere 27%, teapot 44%,
  smoke 69%).
- **Quantization behaviour.** PicoVDB has `GRID_TYPE_SDF_UINT8` only: a fixed
  [−3, +3] voxel band (`(v/3+1)·127.5`), meaningless for density. For fog,
  `picovdb convert --type u8` writes level-set u8 bytes but tags the grid
  `FOG_FLOAT`, so every read is garbage. Our port refuses that combination.
  *Estimate, not a measurement:* a hypothetical fog u8 encoding would put
  smoke at ≈1673 KiB (77% of Fp8, 122% of FpN). That is still not a clear win
  over FpN, and its value range/precision design would be new.

## 2. Measured: correctness / parity

### 2a. GPU parity

`node experiments/picovdb-spike/scripts/gpu-parity.mjs`. Setup: headless
Chromium + SwiftShader Vulkan, localhost page, and **vendored `picovdb.wgsl`
verbatim** with a harness binding the six buffers plus probes and output.
Probes: 20,000 `probeCoords` coordinates per case (seed `0x7069636f`, bbox
dilated by 4), covering active, inactive and outside space. Expected values
come from CPU `readValue` on the **source NanoVDB grid**, mapped through
PicoVDB's value rules (fog = raw; SDF = value / voxelSize; inactive → background
or inside). Two dispatch modes: a fresh accessor per probe, and one accessor
walking 32 probes so the cached paths run. Both modes gave identical counts.
Compile: 0 errors, 0 warnings, pipeline validation clean.

| Fixture | Variant | Failures / total | of which active | Max \|Δ\| | CPU mirror vs GPU |
|---|---|--:|--:|--:|--:|
| primitives/sphere_fog | f32, **native layout** | **4081 / 20000** | 4081 / 8071 | **1.0** | bit-exact 20000 |
| primitives/sphere_fog | f32, fixed layout | 0 / 20000 | 0 | 0 | bit-exact 20000 |
| primitives/torus_fog | f32, **native layout** | **3239 / 20000** | 3239 / 7891 | **1.0** | bit-exact 20000 |
| primitives/torus_fog | f32, fixed layout | 0 / 20000 | 0 | 0 | bit-exact 20000 |
| primitives/box_fog | f32, **native layout** | **5236 / 20000** | 5236 / 13453 | **1.0** | bit-exact 20000 |
| primitives/box_fog | f32, fixed layout | 0 / 20000 | 0 | 0 | bit-exact 20000 |
| vdb-samples/sphere | f32 | 0 / 20000 | 0 | 0 | bit-exact 20000 |
| vdb-samples/sphere | u8 | 0 / 20000 | 0 | 1.17e-2 (≤ ½ step) | 18573 exact, rest ≤ 6e-7 |
| vdb-samples/cube | f32 | 0 / 20000 | 0 | 0 | bit-exact 20000 |
| vdb-samples/cube | u8 | 0 / 20000 | 0 | 1.18e-2 (≤ ½ step) | 18134 exact, rest ≤ 5e-7 |
| vdb-samples/smoke | f32 | 0 / 20000 | 0 | 0 | bit-exact 20000 |
| vdb-samples/utahteapot | f32 | 0 / 20000 | 0 | 0 | bit-exact 20000 |
| vdb-samples/utahteapot | u8 | 0 / 20000 | 0 | 1.17e-2 (≤ ½ step) | 19715 exact, rest ≤ 6e-7 |
| hazard/padding-root (§2c) | f32 | 0 / 4096 values, **2048 / 2048 mis-routed lookups** | — | 0 | 4066 exact; 30 differ by up to 1.0 |

The script exits 0: every gated case passes. The native-layout primitive rows
and the hazard row are ungated on purpose, because they document upstream
defects. The u8 mirror tolerance (absolute 1e-6) covers WGSL's 2.5-ULP
`unpack4x8unorm` division and optional `fma` fusion.

**Bindings.** PicoVDB needs **6 storage buffers**
(grids/roots/uppers/lowers/leaves/buffer). `maxStorageBuffersPerShaderStage`
defaults to **8**, and SwiftShader reports 10. This harness used exactly 8 (6
plus probe in/out). In a fragment shader, 6 leaves 2 for everything else the
material binds. Two grids at once (sequence cross-fade, multi-grid) would need
12 > 8, which means raised limits or a single-buffer repack (§4).

### 2b. CPU round-trip (oracle (a), every active voxel)

The round-trip goes through `src/reader.ts`, a function-by-function mirror of
the WGSL accessor (upstream's `ts/picovdb.ts` has **no** CPU value reader;
its `getGridFloat` is a commented-out TODO). Results are compared against
`readValue` (`pnpm vitest run --root experiments/picovdb-spike`, 17 tests, ~30 s):

| Fixture | Coords checked | Native layout | Fixed layout |
|---|--:|--:|--:|
| primitives/sphere_fog | 219,177 leaf voxels + 594 active lower tiles × 9 samples | leaf voxels exact; **574/594 tiles mis-indexed** (4599 tile samples wrong) | 0 mismatches |
| primitives/torus_fog | 396,012 + 668 tiles × 9 | leaf exact; **652/668 tiles mis-indexed** | 0 |
| primitives/box_fog | 104,277 + 192 tiles × 9 | leaf exact; **180/192 tiles mis-indexed** | 0 |
| vdb-samples/sphere (f32 / u8) | 270,638 | 0 / 0 (u8 max \|Δ\| 0.01176) | — |
| vdb-samples/cube (f32 / u8) | 1,452,218 | 0 / 0 | — |
| vdb-samples/smoke | 1,049,275 | 0 | — |
| vdb-samples/utahteapot (f32 / u8) | 6,960,047 | 0 / 0 | — |

Random probes (inactive and background coverage) match the mapped source on
every fixture. **Lossiness by design:** PicoVDB keeps no inactive values, only
"outside" = root background and "inside" (fog 1.0, SDF −background). For the
baked fog primitives the root background is 3 while inactive voxels hold 0
(the PHASE-5 "background ≠ near-field fill" edge), so **PicoVDB reads density 3
in all empty space** where NanoVDB reads 0. A productised converter would need
a fog background policy (force 0). The smoke sample (background 0) is
lossless.

### 2c. Upstream defects found

1. **Converter: active internal tiles mis-indexed** (`main.zig`
   `convertUpperNodesFromHandle` / `convertLowerNodesFromHandle`). A node's tile
   values are appended word by word, *interleaved* with its children's data,
   but the reader's rank query assumes they are contiguous from
   `base_active_index`. Every active tile in a word after a child reads another
   node's value: fog density error up to 1.0 on 16–26% of random probes. Level
   sets are unaffected, because their internal tiles are inactive. That is
   probably why upstream's tests (level-set `data/sphere.nvdb` only) never
   caught it. **Fix, format-compatible** (reader unchanged): write the node's
   tile values first, then recurse. This is `layout: "fixed"` in
   `src/convert.ts`, with 0 mismatches CPU and GPU. Converters from real cloud
   assets that prune constant interiors will hit this.
2. **WGSL root search reads the padding root.** `picovdbReadAccessorFindUpperIndex`
   bounds the last grid by `arrayLength(&picovdb_roots)`. The upstream loader's
   `rootsBuffer` includes the zero-key padding root added for even counts. On
   an odd-root grid without a real root at key (0,0), every lookup in
   [0,4096)³ matches it and indexes `upper[upperCount]`, out of bounds. Demonstrated:
   2048/2048 such lookups mis-routed on GPU. Values were correct only because
   robust buffer access returned zeros, and the CPU mirror got garbage on 30.
   Fix: bound by the header's upper count, or bind the unpadded slice. (The Zig
   CPU reader uses the unpadded count, so it is correct.)
3. **Fog + `--type u8`** is accepted and produces a broken file (§1).
4. **Fp8/FpN NanoVDB input** is accepted and read with FLOAT strides, producing
   garbage. Our port throws.
5. The **`version` header field** is never bumped (0 since 2026-01-25),
   although the layout changed after that (§6).

## 3. Converter status and oracle achieved

**`src/convert.ts`: a pure-TypeScript port (557 lines incl. docs) of `main.zig`'s
conversion path and `picovdb.zig`'s encoder.** It consumes a FLOAT NanoVDB
grid image (from `NanoVDBFile.gridImage`, or our `buildFromVdb`) and emits
`.pvdb` bytes. Supported grid types (confirmed from source): `SDF_FLOAT` (1),
`SDF_UINT8` (2), `FOG_FLOAT` (3). Fog is selected by NanoVDB class
FogVolume (2), and every other class is treated as SDF. SDF values are divided
by voxel size (index-space distance), with surface bits set from a +x/+y/+z
7-neighbour sign test. The port also supports multi-grid files.

**Oracles achieved: both.**
- **(b) Byte-for-byte vs the native converter: 10/10 outputs identical**
  (3 fog primitives f32; sphere/cube/teapot f32 + u8; smoke f32). The goldens
  are SHA-256 hashes recorded from a native build of the pinned commit, stored
  in `test/native-golden.json` and asserted on every run, so CI needs no Zig.
  The toolchain was obtained cheaply: ziglang.org and GitHub archive tarballs
  are proxy-blocked here, but Zig 0.16.0 is on **PyPI** (`ziglang` wheel), and
  OpenVDB v13.0.0's headers came from a sparse git clone, wired in as a local
  `.path` dependency in a scratch copy of `build.zig.zon`.
- **(a) Value round-trip vs `readValue`**: every active voxel (§2b).
- Upstream's own `PicoVDBFile` loader parses every output, with section sizes,
  grid record, bbox and voxel count consistent with ours.

Conversion speed in Node (TS vs native ReleaseSafe): smoke 43 ms vs 59 ms;
SDF sphere 234 ms vs 41 ms (the neighbour sign test calls `readValue`
uncached); teapot 3.5 s vs 1.2 s. Fine for a browser-side converter (D3/D6).
A productised version would add the defect-1 fix, a fog background policy, a
transform sidecar, and an accessor cache for the SDF surface pass.

## 4. Integration shape (assessed, not built)

Today `NanoVDBVolumeMaterial` binds **one** TSL storage node,
`storage(attr, "uint", n).setName("nvdbGrid")` (emitted as `nvdbGrid.value`).
It applies a **one-token rewrite** (`nanovdb_buffer` → `nvdbGrid.value`) to
the vendored library and passes the node as a `ptr<storage, array<u32>, read>`
`wgslFn` param so TSL emits and binds it. PicoVDB declares **six typed**
storage arrays (`array<PicoVDBUpper>` etc. with nested fixed arrays of
structs). TSL's `storage()` is typed by TSL element types, not arbitrary WGSL
structs. Whether three r185's struct nodes can express
`array<PicoVDBNodeElement,1024>` members is **unverified and doubtful**. So
a one-token rewrite cannot work.

- **Option A: six `uint` storage nodes + rewrite every struct-field access.**
  This means about 25 access sites in the accessor/stencil code (e.g.
  `picovdb_uppers[i].elements[w].stateMask` → index arithmetic). It is a real
  fork of ~250 lines, it uses 6 of 8 fragment storage slots, and two grids
  would not fit. Not recommended.
- **Option B (recommended if adopted): single-buffer repack.** No repacking
  is actually needed, because the `.pvdb` file is already one contiguous,
  16-byte-aligned blob with section offsets derivable from its header. Bind the
  whole file as one `array<u32>` (the existing `setName` + ptr-param mechanism,
  unchanged). Replace the accessor's six globals with ~6 small load helpers
  (`pico_upper_elem(i, w)` …) reading at header-derived offsets. The HDDA,
  stencil math and analytic voxel intersector are pure functions and are reused
  verbatim. That costs 1 storage slot, the same as today.
- **Gaps either way.**
  - PicoVDB has **no Map/voxel size/world bbox**. The material's
    `pnanovdb_grid_world_to_indexf` calls need uniforms fed from a sidecar or a
    format extension.
  - It has **no per-node min/max stats**, so no value-range empty-space
    skipping (topology skipping via level dims is available).
  - Its HDDA is a level-set zero-crossing tracer. A fog march would be ours,
    built on `picovdbSampleTrilinear`.

**Effort estimate for an experimental `PicoVDBVolumeMaterial` sibling
(Option B):**

| Task | Days |
|---|--:|
| WGSL adapter fork + re-run this spike's GPU parity harness | 1–2 |
| `PicoVDBGrid` wrapper (load, transform sidecar) | 0.5–1 |
| Material sibling (fog march via `picovdbSampleTrilinear`, transform uniforms) | 1–2 |
| Golden-image e2e + perf bench | 1–2 |
| Productise the converter into `vdb-web-tools` (defect-1 fix, fog background policy, transform) | 1–2 |
| **Total** | **≈4.5–9 dev-days** |

Not included: upstreaming, sequence support, or a `.pvdb` → `.nvdb` path.

## 5. D6 relevance: `model.ts` GPU CSG

Assessment only; not executed here. Its tests are Deno + WebGPU, and the
module pulls in 14 `ts/gpu/*` modules and 11 WGSL kernels. The D6 table's
interim server op is "CSG / composites".

- **What it does well.** It offers csg.js-style `union` / `subtract` /
  `intersect` / `offset(n)` (redistanced from the marching-cubes surface) /
  `translate` on GPU-resident narrow-band SDF grids. It can stamp WGSL shape
  functions directly and load meshes (STL). This is genuinely the browser-native
  level-set CSG that D6 wants to stop sending to OpenVDB.
- **Limits.**
  - **SDF only**: `Loader.load` throws `grid type … is not an SDF` for fog,
    so fog composites (max/add/mask of densities), our main use, are not
    covered.
  - **Index space only**: there is no transform, `translate` takes whole
    voxels, and there is no resample/rotate/scale. It therefore does not cover
    D6's "general case rides the GPU-resample path".
  - **Extent**: at most 1024 leaves (8192 voxels) per axis per solid or
    operand pair.
  - **Fixed half-width** per `Space`: values are rescaled on load, and inactive
    values are dropped.
  - **Device limits**: kernels "bind up to ten storage buffers", so a device
    must be created with raised `maxStorageBuffersPerShaderStage`. That is
    compatible with our D4 device-first creation, but above the WebGPU default.
  - **Output** is a `.pvdb` or a GPU picovdb tree. To feed our NanoVDB pipeline
    we would need a `.pvdb` → NanoVDB rebuild (feasible via
    `buildFromLeavesDetailed`, ~1 day, not built).
- **Verdict.** It could plausibly retire the **level-set half** of "CSG /
  composites" once SDFs matter. It does not retire the fog half, and it does
  not replace the GPU-resample successor. If level sets become a priority, it
  is the strongest available candidate (it would need its own spike: run the
  Deno tests under our harness, plus a round-trip through `.nvdb`).

## 6. Risks

- **Format churn.** The README warns the format and API may change. History
  bears it out: the on-disk layout changed at least on 2026-01-25 (index-space
  values), 02-22 (packed node indices, new node struct), 02-23 (surface bit /
  state-mask semantics) and 03-01 (u8 + fog types). Meanwhile `header.version`
  went 1 → 0 and has stayed 0. **Files carry no usable format version.**
- **Activity.** One author, 31 commits. The cadence is bursty: Dec 2025 (2),
  Jan (5), Feb (13), Mar (5), **Apr–Jun: 0**, Jul (4), Aug (2). One tag
  (`0.0.1`, 2026-03-01). Upstream CI (added 2026-08-24) runs `deno lint`,
  `deno check`, and GPU tests that skip when no adapter is present. The Zig
  converter tests (`zig build test`) are not run in CI.
- **API stability.** Recent work is in the C ABI and GPU editing (`model.ts`,
  `ts/gpu/*`), not the read path. `picovdb.wgsl` itself has been stable since
  07-17. The TS loader is a thin section slicer with no value reader.
- **Correctness.** Five defects were found in a few hours (§2c), two of which
  silently corrupt fog reads.
- **What pinning + a D2-style vendor-fork buys.** Immunity to format churn
  (we would own the bytes, as this spike's verbatim vendor + SHA guard does
  already), the freedom to carry fixes 1 and 2 in-tree with a diff log, and a
  TS converter (D3-compliant, no Zig in our toolchain) whose byte-parity
  goldens tell us exactly when upstream's output changes. **What it does not
  buy:** a quantized fog encoding, a transform in the format, or a second
  maintainer. Owning a fork means owning a format nobody else reads. Any
  `.pvdb` we write is only as portable as our fork.
- **Upstream asks** (cheap, given the owner's contact): fixes 1 and 2 with
  repro (this spike's tests); a real `version`; a Map/voxel size in
  `PicoVDBGrid`; quantized fog; and a CPU value reader in the TS loader.

## 7. Proposed D7 (DRAFT — requires owner approval; not in DECISIONS.md)

> ## D7 — PicoVDB: not adopted now; re-evaluate on triggers
> NanoVDB Float/Fp8/FpN remains WebVDB's only shipping GPU format, and
> PicoVDB is **not** adopted as a format or as a CSG engine at this time
> (evidence: [spikes/PICOVDB.md](./spikes/PICOVDB.md)). On our fog corpus
> PicoVDB float is 2.2× NanoVDB Fp8, it has no quantized fog encoding, no
> world transform and no format version, and its converter mis-reads fog
> grids with active internal tiles. The spike stays in
> `experiments/picovdb-spike/` (pinned `f5d22e5`, vendored verbatim,
> Apache-2.0) as the re-evaluation harness, outside the workspace build.
> We offer upstream the two read-correctness fixes and the format requests.
> **Re-evaluate when any of these fire:** (a) upstream ships a quantized fog
> encoding that beats our Fp8 on the smoke sample in the spike's
> `sizes.mjs`; (b) level-set rendering or SDF CSG becomes a roadmap
> priority, where PicoVDB u8 (≈54% of Fp8 on the teapot) and `model.ts`
> are the first candidates; (c) upstream versions the format and adds a
> transform. If adopted, it follows D2 (vendor-fork, pinned, diff log),
> uses the spike's TS converter (D3, with no Zig), and ships as an optional
> `PicoVDBVolumeMaterial` sibling binding the `.pvdb` as a single storage
> buffer.

## 8. Reproduction

```bash
# fixtures (present in this checkout; regenerate if missing)
pnpm fixtures:bake            # fixtures/primitives/*.nvdb (9 files)
pnpm fixtures:vdb-samples     # fixtures/vdb-samples/*.vdb (4 files)

# spike suite: vendor hashes, byte parity vs native goldens, round-trip (17 tests, ~30 s)
pnpm vitest run --root experiments/picovdb-spike
# GPU parity (headless Chromium/SwiftShader; ~30 s; exit 0 = all gated cases pass)
node experiments/picovdb-spike/scripts/gpu-parity.mjs [--count 20000] [--only smoke]
# size tables
node experiments/picovdb-spike/scripts/sizes.mjs
# typecheck of the browser-safe spike modules (not part of `pnpm typecheck`)
pnpm tsc -p experiments/picovdb-spike

# re-record native goldens (only when the pin moves); ziglang.org + GitHub tarballs are proxy-blocked here
python3 -m pip download ziglang==0.16.0 --no-deps --only-binary=:all: \
  --platform manylinux_2_12_x86_64 -d zigwheel
python3 -m zipfile -e zigwheel/ziglang-0.16.0-*.whl zig && chmod +x zig/ziglang/zig
git clone https://github.com/emcfarlane/picovdb && git -C picovdb checkout f5d22e5602a7fc7f206d33ec35328f58e8667c83
git clone -q --depth 1 --branch v13.0.0 --filter=blob:none --sparse \
  https://github.com/AcademySoftwareFoundation/openvdb && git -C openvdb sparse-checkout set nanovdb/nanovdb
# point the openvdb dependency at the local clone instead of the archive URL:
python3 - <<'EOF'
import re; p='picovdb/build.zig.zon'; s=open(p).read()
open(p,'w').write(re.sub(r'\.openvdb = \.\{[^}]*\}', '.openvdb = .{ .path = "../openvdb" }', s, flags=re.S))
EOF
(cd picovdb && ../zig/ziglang/zig build -Doptimize=ReleaseSafe)
PICOVDB_NATIVE=$PWD/picovdb/zig-out/bin/picovdb node experiments/picovdb-spike/scripts/native-oracle.mjs
```

The root suite is unaffected: `pnpm vitest run` (234/234; `projects:
["packages/*"]` never sees `experiments/`) and `pnpm typecheck` (the spike is
not in the root `tsconfig.json` references) both stay green.
