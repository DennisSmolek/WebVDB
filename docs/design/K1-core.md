# K1 design note: the engine-free WebGPU core

Status: proposed, 2026-11-18 (task K1.2; inputs for K1.1 and K1.4). Owner confirms the open questions in §9.
Code is named by its post-K0 home (`core/`, `tools/`, `three/` under `packages/vdb/src/`). Current paths are
cited only where this note was measured from them.

Read with: [PHASES.md](../../PHASES.md) K1, [DECISIONS.md](../../DECISIONS.md) D2, D8, D10, D11, D17–D20,
[spec/SPEC-draft.md](../../spec/SPEC-draft.md) §2–3. Precedents: `@pmndrs/bvh` (`traceModuleWgsl`, `traceSlot`,
`GPU_LAYOUT_VERSION`, `GrowableStorage`, `src/three/trace-nodes.ts`), `@pmndrs/downsampler` (`requiredLimits`,
"checked, never destroyed"), the pathtracer's `docs/design/volumes.md`, and the cloud system's
`work/specs/P8-hero-vdb.md`.

## 0. Decisions at a glance

| # | Decision | Rejected |
|---|---|---|
| 1 | One `GPUBuffer`: 32 B header, fixed-stride 160 B grid records, then sub-allocated ranges (images, brick majorants), all 32 B aligned. | Images packed back to back with no table (no per-grid transform); one buffer per grid (breaks D10). |
| 2 | Images are addressed **relative to their own start** through the fork's existing `pnanovdb_buf_t.byte_offset`. The vendored traversal needs **no** addressing edits. | Absolute addresses (would rewrite every child-offset read). |
| 3 | Growth and compaction happen only in `update(encoder)`, by a new buffer plus copies of resident ranges; `version` bumps then. In-place writes never bump `version`; `revision` counts content changes. | Re-layout on every `add` (rebinds every frame during loading); a fixed pool (v1's `maxGridBytes` trap). |
| 4 | Per-grid world↔index affines live in the record (both directions, f32, composed in f64 on the CPU). `setTransform` rewrites 96 bytes; the image is never touched. | Reading the NanoVDB map in the shader (no instancing; two extra matrix reads per call). |
| 5 | The generator works on a **declaration table extracted at build time** from the unmodified vendored file and takes the dependency closure of each profile's roots. | Hand-splitting `pnanovdb.wgsl` into chunk files (breaks the SHA check and the D2 diff log). |
| 6 | Generation-time transforms (logged in `VENDOR.md`, not edits): buffer read via `vdb_load`, grid-type table specialized to the compiled types, typed decode limited to them, `pnanovdb_`→`vdb_` rename. One real edit: the Fp leaf-stats readers (§5.3). | Regex-rewriting the vendored text at runtime (v1's `rewriteBufferGlobal`). |
| 7 | Two binding modes: `slot: { group, firstBinding }` declares the binding; `buffer: '<expr>'` declares nothing and reads a host-declared `array<u32>` (engine adapters). Both emit exactly one buffer access site. | `slot` only: three's TSL assigns its own `@group/@binding`, so it can't honour a slot (§6). |
| 8 | Majorant data: a brick grid in the pool (default) or NanoVDB node stats via HDDA, behind `majorant: 'brick' \| 'nodeMax' \| 'global'`. Both take the same iterator API. | Separate majorant binding (breaks D10); a 3D texture (needs a second binding, and the cloud system's measurement says it isn't the bottleneck). |
| 9 | **The provisional `read`/Float ≤ 4 KB gate cannot be met by the vendored accessor**: measured 16–17 KB minified. K1 gates on measured budgets (§3.6); a 4 KB target moves to an optional `reader: 'lean'` (new K1.8). | Keeping the 4 KB gate on the vendored path (forces a rewrite of the audited traversal inside K1). |

## 1. The pool buffer

### 1.1 Header (32 bytes, words)

| Word | Field | Value |
|---|---|---|
| 0 | magic | `0x50424456` (`"VDBP"`, little-endian) |
| 1 | layoutVersion | `1` (`VDB_GPU_LAYOUT_VERSION`) |
| 2 | gridCapacity | records in the table |
| 3 | gridCount | highest live id + 1 (`vdbGridCount()`) |
| 4 | tableOffset | `32` (fixed in layout 1; written for readers that don't know the version) |
| 5 | recordBytes | `160` |
| 6 | usedBytes | allocation high-water mark (diagnostics) |
| 7 | reserved | `0` |

The WGSL bakes `VDB_POOL_TABLE_OFFSET = 32u` and `VDB_POOL_RECORD_WORDS = 40u`, so a lookup is one multiply-add.
The header exists for CPU readers and readback debugging; the hot path reads only word 3.

### 1.2 Grid record (160 bytes, 40 words, at `32 + id * 160`)

| Words | Field | Notes |
|---|---|---|
| 0 | imageOffset | bytes from the start of the pool; multiple of 32 |
| 1 | imageBytes | NanoVDB `GridData.mGridSize` (low word; high word must be 0) |
| 2 | gridType | NanoVDB `GridType` id (1 Float, 13 Fp4, 14 Fp8, 15 Fp16, 16 FpN) |
| 3 | flags | b0 live · b1 hasStats (`HAS_MIN_MAX`) · b2 hasBrickMajorant · b3 instanceTransform · b8–15 gridClass · b16–23 brick log2 |
| 4–15 | worldToIndex | 3×4 row-major f32 (`idx = M · [p, 1]`) |
| 16–27 | indexToWorld | 3×4 row-major f32 |
| 28–30 | indexBBoxMin | i32 ×3, root active bbox |
| 31 | background | f32 bits, root background |
| 32–34 | indexBBoxMax | i32 ×3, inclusive |
| 35 | valueMax | f32: `max(root max stat, background)` or the caller's bound; the `global` majorant |
| 36 | majorantOffset | bytes; 0 = none |
| 37–39 | majorantDims | u32 ×3, brick-grid cells |

Why these fields: everything the hot path needs before touching an image (type, base, transform, bounds for
ray clipping and `outside: 'zero'`, a global bound) is in one 160 B record, which a workgroup reads through the
cache once. Both affine directions are stored so no shader inverts a matrix. Records are 16 B multiples (`vec4`
loads stay possible). A dead record has flags b0 = 0; its id is reused.

Rejected: a variable-length record or per-grid user words. A host keeps its own per-grid data (σ, scale,
emission) in its own uniform indexed by the same id; adding fields later is a layout bump, as in bvh.

### 1.3 Images and other ranges

- Images are byte-identical NanoVDB grid buffers (the `tools` contract), placed at **32-byte** alignment, the
  NanoVDB data alignment, so any slice of the pool is a valid NanoVDB grid for readback and CPU tools. The spec
  sketch said 16 B; 32 B costs at most 16 B per grid more.
- A brick-majorant grid (§5.1) is its own range, referenced by `majorantOffset`.
- The table sits at the front with `gridCapacity` records (default 16 = 2.5 KiB); images start at
  `align32(32 + capacity * 160)`.

### 1.4 add, remove, replace, update(encoder)

Model: CPU calls stage changes; `update(encoder)` applies them. This is bvh's `SceneTrees.update` contract.

- `add(image, opts)`: parses and validates the header on the CPU (magic, ABI major 32, type, size, flags, map,
  root bbox, background, root max), allocates a range (first-fit, 32 B units, `opts.capacity` may reserve more
  than the image for sequences), fills the record, and returns the id **synchronously**. Nothing reaches the
  GPU yet.
- `replace(id, image)`: in place if it fits the range (no `version` change; this is the sequence-player path),
  else free + allocate. `remove(id)`: frees the range and clears the record. A freed range may be reused in the
  same update: WebGPU orders queue writes after previously submitted reads.
- `setTransform(id, objectToWorld | null)`: record words 4–27 only.
- `update(encoder)`:
  1. Destroy buffers retired by the previous update (their copies have been submitted since).
  2. If the staged state does not fit (bytes or table capacity), **grow and compact**: create a buffer of
     `max(needed, 2 × current)` clamped to the limit (§1.5), record `copyBufferToBuffer` for every range that
     was **resident before this update** to its new, packed offset, retire the old buffer, `version++`. Growth is
     planned and checked before anything changes; on `VDB_POOL_LIMIT` it throws with the pool unchanged.
  3. `queue.writeBuffer` the new and replaced images, the dirty records and the header. Queue writes land before
     the encoder runs, so step 2's copies only cover ranges that were already resident and never ones being
     written now (the ordering rule bvh's `GrowableStorage` documents).
- Fragmentation policy: never move a range except during a growth or when an allocation fails although free
  bytes suffice; then compact into a buffer of the same size (still a `version` bump). Images can move freely
  because only the table points at them, so compaction is a copy plus record rewrites.
- `flush()`: convenience that makes its own encoder, calls `update`, submits (upscaler precedent).

Rejected: sub-allocating inside a never-moving buffer (fragmentation with no recovery), and growth in `add()`
(it needs an encoder, and would bump `version` several times while a sequence loads).

### 1.5 Sizes and limits

- Pool bytes ≤ `min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize, 2^32 − 4)`. Defaults
  are 128 MiB and 256 MiB; `requiredLimits(adapter)` asks for the adapter's maxima of both.
- One image ≤ 4 GiB − 4 (record word 1, and the fork's u32 byte addresses). In practice the binding limit binds
  first: WDAS quarter Fp8 is 29 MB; a full-resolution Float hero can exceed 128 MiB, which is why
  `requiredLimits` exists.
- Default `initialBytes` 1 MiB, grown to fit the first `update`.

### 1.6 Transforms

`setTransform(id, objectToWorld)` takes a 4×4 column-major matrix (WebGPU and three convention). The pool
composes `worldToIndex = map⁻¹ ∘ objectToWorld⁻¹` and `indexToWorld` in **f64** from the image's double map
(`matd`/`invmatd`/`vecd`), then rounds to f32. `null` restores the file's map. Index-space rays keep `t`:
`vdbWorldToIndexDir` is not normalized, so a world-space `t` is valid in index space (the majorant iterator
relies on this). The cloud system's movable heroes (≤ 4) and the pathtracer's instanced grids are both this
call.

### 1.7 Where majorant data lives

`brick`: a range in the pool (§5.1), offset and dims in the record. `nodeMax`: inside the image (NanoVDB node
stats; needs `HAS_MIN_MAX`, record b1). `global`: record word 35. No extra binding in any case.

### 1.8 Layout version

`VDB_GPU_LAYOUT_VERSION` (core export) and header word 1. Any change to a header word, record word or meaning
bumps it and is a breaking change (bvh's rule). The module generator bakes the version it targets;
`checkModule(pool, options)` (JS) throws `VDB_LAYOUT_MISMATCH` if they differ.

## 2. Addressing

The fork's reads all go through one function (measured in the current
`packages/nanovdb-wgsl/vendor/pnanovdb.wgsl`, lines 37–40):

```wgsl
fn pnanovdb_buf_read_uint32(buf: pnanovdb_buf_t, byte_offset: u32) -> u32 {
    let data = nanovdb_buffer[(buf.byte_offset / 4u) + (byte_offset / 4u)];
    return data;
}
```

`pnanovdb_buf_t.byte_offset` is already a per-buffer base. So "image base from the grid table" is:
`vdbAccessorInit(g)` reads record word 0 and builds `vdb_make_buffer(imageOffset, imageBytes / 4)`; every
address inside the traversal stays image-relative and unchanged. The only substitution is the declaration of
`pnanovdb_buf_read_uint32`, which becomes `return vdb_load((buf.byte_offset >> 2u) + (byte_offset >> 2u));`
with `vdb_load` the module's single buffer access (§3.4). Zero lines of the upstream body change.

Null handles stay sound: the accessor cache and `root_find_tile` use address 0 as "none". Relative address 0 is
the grid header, never a node, so the rule holds for every grid in the pool.

**DIVERGES V2 (32-bit addresses) does not bite.** The fork reads the low word of 64-bit child offsets and keeps
u32 byte addresses. Image-relative addresses are < 4 GiB because the binding is (§1.5); the absolute word index
is computed after dividing each part by 4, so it can't overflow either. The pool enforces record word 1's high
word = 0 at `add`. Revisit only if `maxStorageBufferBindingSize` passes 4 GiB on some adapter.

## 3. Module generator

### 3.1 API

```ts
vdbModuleWgsl({
  slot: { group: 2, firstBinding: 20 },   // declares @group(2) @binding(20) var<storage, read> vdbPool: array<u32>;
  // or buffer: 'vdbPool.value',          // engine adapters: no declaration; reads <buffer>[i]
  types: ['float', 'fp8'],                // 'float' | 'fp4' | 'fp8' | 'fp16' | 'fpn'; others compile out
  profile: 'sample',                      // 'read' | 'sample' | 'march' | 'majorant'
  sampler: 'trilinear',                   // sample+: 'trilinear' | 'nearest'
  isActive: true,                         // read+: false drops vdbIsActive (−3.4 KB)
  outside: 'background',                  // sample+: 'background' (NanoVDB) | 'zero' (bbox test, pathtracer convention)
  majorant: 'brick',                      // majorant: 'brick' | 'nodeMax' | 'global'
  helpers: ['cellBoundTrilinear'],        // K1.7a, opt-in, not emitted unless listed (see below)
}): string;                               // deterministic: same options, same bytes (shader caches hit)

vdbModuleEntries(slot | { slot, visibility }): GPUBindGroupLayoutEntry[]; // one read-only-storage entry at firstBinding;
                                                                         // visibility defaults to COMPUTE | FRAGMENT (K1.7a)
pool.bindGroupEntry(slot): GPUBindGroupEntry;                         // { binding, resource: { buffer: pool.buffer } }
checkModule(pool, options | { types, layoutVersion? }): void;       // K1.7a: type/layout-only form for adapters
```

K1.7a (first consumer, demo 08): the options type is the union `VDBSlotModuleOptions | VDBBufferModuleOptions`,
so a host that annotates `VDBSlotModuleOptions` reads `options.slot` without `!`, and the same `VDBSlot` goes to
all three calls. Opt-in `helpers` (each emitted only when listed, so default modules keep their bytes):
`cellBoundTrilinear` (`march`: `vdbCellBoundTrilinear(acc, ijk, dim)`, max of `vdbNodeMax` over the 2x2x2 cell
block, the trilinear-safe skip bound), `hddaCellEnd` (`march`: `vdbHddaCellEnd(&h)`, the current cell's exit `t`)
and `rayClipSupport` (`sample`+: `vdbRayClipSupport(g, o, d, &t)`, clips to `[min - 1, max + 1]` for trilinear,
`[min - 0.5, max + 0.5]` for nearest; `vdbRayClip` keeps the voxel-cell box `[min, max + 1]`). CPU twins in
`tools`: `cellBoundTrilinear`, `hddaCellEnd`, `rayClipSupport`. A module must be generated for every type the host
may add (`VDB_ALL_SCALAR_TYPES` for all of them), or regenerated with its pipeline rebuilt; `checkModule` says so.

`firstBinding` keeps bvh's name; layout 1 uses one binding (D10), and a later layout could add one without
renaming. Exactly one of `slot` / `buffer` is required (`VDB_BAD_OPTION` otherwise, bvh's positional-call guard
included). `fp4`/`fp16` compile from the existing decoders but have no parity fixtures yet; the K1 gate covers
Float, Fp8, FpN.

### 3.2 Profiles (each includes the previous one; the generator takes the closure)

| Profile | Public WGSL (all `vdb`/`Vdb`) | Vendored roots it pulls in |
|---|---|---|
| `read` | `vdbGridCount()`, `vdbGridType(g)`, `VdbAccessor`, `vdbAccessorInit(g) -> VdbAccessor`, `vdbReadValue(acc, ijk) -> f32`, `vdbIsActive(acc, ijk) -> bool`, `vdbIndexMin(g)`, `vdbIndexMax(g)` | `readaccessor_init`, `readaccessor_get_value_float`, `readaccessor_is_active`, `grid_get_tree`, `tree_get_root` |
| `sample` | + `vdbWorldToIndex(g, p)`, `vdbWorldToIndexDir(g, d)`, `vdbIndexToWorld`, `vdbIndexToWorldDir`, `vdbSample(acc, xyzIndex) -> f32`, `vdbRayClip(g, o, d, t: ptr<function, vec2f>) -> bool` | `sample_trilinear_typed`, `hdda_ray_clip` (transforms come from the record, not `map_apply`) |
| `march` | + `VdbHdda`, `vdbHddaInit/Update/Step`, `vdbGetDim(acc, ijk) -> u32`, `vdbNodeMax(acc, ijk, dim) -> f32` | `hdda_*`, `readaccessor_get_dim`, `{root,upper,lower,leaf}_get_max_float` (fixed, §5.3) |
| `majorant` | + `VdbMajorantIter`, `vdbMajorantInit(g, oIdx, dIdx, tMin, tMax)`, `vdbMajorantNext(it) -> vec3f` (t0, t1, density bound; `t1 <= t0` ends) | `brick`/`global`: `sample` + ~1.8 KB of our DDA; `nodeMax`: `march` + ~2.5 KB walk |

`VdbAccessor` is `{ buf, gridType, ok, inner: vdb_readaccessor_t }`. `ok` is false for a dead id or a grid type
the module wasn't compiled for; reads then return 0. One uniform branch per read is the price of never decoding
a grid with the wrong decoder; `checkModule(pool, options)` also throws `VDB_TYPE_NOT_COMPILED` on the JS side.
Rays are transformed once to index space by the caller (as v1's march does); `vdbWorldToIndex` costs 12 record
loads, so it isn't a per-sample call.

Level-set zero crossing (`hdda_zero_crossing`), gradients and stencils stay out of K1: no consumer needs them
before K4's level-set work, and they would be one more option, not a profile.

### 3.3 How the vendored file becomes chunks (D2 kept)

- `scripts/extract-wgsl-chunks` parses the **unmodified** vendored file into top-level declarations (fn, struct,
  const, alias) with their dependency names, and writes `core/wgsl/pnanovdb.chunks.generated.ts`, headed by the
  file's Apache-2.0 header and the pinned commit. A sync test fails if the table drifts from the file (the
  `stride-tables.json` pattern). The vendored file, its SHA test, NOTICE and diff log stay as they are.
- The generator: roots for the profile and options → closure over the table → apply overrides → rename →
  append our API layer (`core/wgsl/api.ts`, MIT) → prepend one line:
  `// SPDX-License-Identifier: Apache-2.0 AND MIT. Derived from PNanoVDB.h (OpenVDB) via emcfarlane/webgpu-nanovdb; see NOTICE.`
  The full Apache header travels in the package (NOTICE, the chunk file). Emission order is the file order, so
  output is stable.
- Overrides (whole declarations replaced at generation; each gets a "generation transform" row in `VENDOR.md`):

| Declaration | Becomes | Why |
|---|---|---|
| `pnanovdb_buf_read_uint32` | reads `vdb_load(...)` | the one buffer site (§2) |
| `pnanovdb_grid_type_constants` (+ `_t`) | `fn vdb_gtc(t: u32) -> vdb_grid_type_constants_t` returning the compiled types' rows (one type: a constant, no branch) | the 27-row table is 5.3 KB minified and dynamically indexed; FXC/DXC materialize dynamically indexed const arrays. The `[grid_type]` call sites are rewritten to `vdb_gtc(grid_type)`. |
| `pnanovdb_read_float_typed` | only the compiled types' branches | compile-out |
| `pnanovdb_sample_trilinear_{float,fp8,fpn}` | dropped; `vdbSample` calls `_typed` | API layer owns the entry |
| every `pnanovdb_` / `PNANOVDB_` | `vdb_` / `VDB_` | prefix rule (D10, D17); −2.7 KB on `read` |

Rejected for the type table: keeping the array and padding unused rows with zero values (keeps call sites, but
costs 0.9 KB and keeps the dynamic indexing).

### 3.4 Names and collisions

- Public functions `vdbCamelCase`; public structs `VdbPascalCase` (bvh's `BvhRay`); internals `vdb_snake_case`;
  constants `VDB_UPPER`; the binding `vdbPool`. Hosts reserve the prefixes `vdb`, `Vdb`, `VDB_`.
- One VDB module per shader. Two modules (two profiles, or two pools) in one shader redeclare every symbol; pick
  the largest profile you need. Several grids share one module and one pool by id; that is the point of D10.
- Port bvh's `wgslNameProblems` test (reserved words such as `active`, `set`, `target`; a function named like a
  variable) and run it over every generated variant.
- No back-ticks in generated WGSL comments (Traps: they break the host's template literal).

### 3.5 Measured sizes (input to K1.1)

Measured 2026-11-18 from the current vendored file (72,761 bytes, 293 declarations) with a closure script over
the declaration graph. "min" = comments stripped, whitespace collapsed, no identifier mangling; "gz" = gzip of
min. Specialization = §3.3 overrides applied (type table, compile-out, rename). The API layer (~0.6–1.0 KB) is an
estimate until K1.5 measures real output.

| Profile | Unspecialized closure, min | Specialized, Float | Fp8 | FpN | Float+Fp8+FpN | Float gz |
|---|---|---|---|---|---|---|
| `read` | 25.9 KB | 16.2 KB | 17.5 | 17.7 | 18.4 | 2.7 KB |
| `read`, `isActive: false` | — | 12.8 KB | 14.1 | 14.4 | 15.0 | — |
| `sample` | 30.4 KB | 20.2 KB | 21.5 | 21.7 | 22.4 | 3.4 KB |
| `march` | 39.3 KB | 27.0 KB | 28.2 | 28.5 | 29.2 | 4.4 KB |
| `majorant` (`brick`) | — | ≈ 22.0 KB (sample + DDA) | ≈ 23.3 | ≈ 23.5 | ≈ 24.2 | — |
| whole file | 53.8 KB (raw 72.8 KB) | | | | | 12.3 KB |

Where `read`/Float goes (specialized): node getters and handle structs 6.4 KB, accessor cache 3.0 KB,
`isActive` family 2.4 KB, root tile search and descent 3.5 KB, constants 0.6 KB. The vendored accessor is
built from many small typed getters; specialization can't shrink that below ~12 KB.

Cross-check: the cloud system measured "12 KB gzip" for the tree code. That is the whole file (12.3 KB gz here),
not a profile.

### 3.6 Proposed K1.1 budgets (min bytes; CI fails on any growth not committed to the snapshot)

| Profile (vendored accessor) | Float | + one Fp type | Float+Fp8+FpN |
|---|---|---|---|
| `read` | 17.5 KB | 18.5 KB | 19.5 KB |
| `read`, `isActive: false` | 14 KB | 15 KB | 16 KB |
| `sample` | 21.5 KB | 23 KB | 24 KB |
| `march` | 29 KB | 30 KB | 31 KB |
| `majorant` `brick` / `nodeMax` | 23.5 KB / 32 KB | 25 / 33.5 | 26 / 34 |
| `read`, `reader: 'lean'` (K1.8) | **4 KB** | 5 KB | 6.5 KB |

Compile time is measured, not gated in CI: `createComputePipelineAsync` for a one-entry kernel per variant,
median of 9, fresh shader text per run (a nonce constant) so Dawn's cache misses. **Reference device: the RTX
3060 on Windows Chrome (D3D12, DXC)**, the pathtracer's slow path; M5 Pro Chrome (Metal) as second column.
Quiet machine (D11).

What this means for consumers: the pathtracer's media kernels are 43.7–47.9 KB against a ~50 KB DXC class, and
its wavefront intersect kernel is 19.9 KB. A 21.5 KB `sample` module doubles that kernel. The lean reader is the
pathtracer-facing answer; it does not block 0.1 (D19).

## 4. JS API (`core`)

```ts
class VDBGridPool {
  static create(device: GPUDevice, options?: VDBGridPoolOptions): Promise<VDBGridPool>;
  static isSupported(device: GPUDevice): boolean;
  static unsupportedReason(device: GPUDevice): string | null;   // downsampler style
  static requiredLimits(adapter: GPUAdapter): Record<string, number>; // maxStorageBufferBindingSize, maxBufferSize

  readonly device: GPUDevice;
  readonly buffer: GPUBuffer;        // replaced only by update(); then version changes
  readonly version: number;          // +1 when `buffer` is a new object: rebuild bind groups
  readonly revision: number;         // +1 on any content change (add, replace, remove, setTransform, setMajorant)
  readonly layoutVersion: number;    // VDB_GPU_LAYOUT_VERSION
  readonly byteLength: number; readonly usedBytes: number; readonly retiredBytes: number;
  readonly types: ReadonlySet<VDBGridTypeName>;                    // types of live grids, for vdbModuleWgsl

  add(image: Uint32Array, options?: { capacity?: number; transform?: Float32Array | number[] | null;
       majorant?: VDBBrickMajorant; valueMax?: number; label?: string }): number;   // grid id
  replace(id: number, image: Uint32Array): void;
  remove(id: number): void;
  setTransform(id: number, objectToWorld: Float32Array | number[] | null): void;
  setMajorant(id: number, majorant: VDBBrickMajorant | null): void;
  info(id: number): VDBGridInfo;      // type, class, name, index/world bbox, offsets, valueMax, hasStats
  update(encoder: GPUCommandEncoder): void;
  flush(): void;                      // own encoder + submit
  bindGroupEntry(slot: VDBSlot): GPUBindGroupEntry;
  dispose(): void;
}
```

Options: `initialBytes` (1 MiB), `maxGrids` (16, table grows by doubling through the same growth path),
`maxBytes` (default: the device limit), `label`, `usage` (extra flags ORed into the always-set
`STORAGE | COPY_DST | COPY_SRC`), `buffer` (a host `GPUBuffer`: checked for `STORAGE | COPY_DST` and size,
**never destroyed**, fixed capacity, so growth throws `VDB_POOL_FULL`).

What it allocates and frees: one `GPUBuffer` (`STORAGE | COPY_DST | COPY_SRC`); during a growth, the new buffer;
retired buffers are destroyed at the next `update` or at `dispose`. `dispose` destroys the current and retired
buffers it created, never the device, never a host buffer. The CPU keeps no image copies (uploads go straight to
`writeBuffer`), only the records.

`VDBBrickMajorant` (`{ brickLog2, dims, data: Float32Array }`) comes from `tools` (`vdbBrickMajorant(image,
{ brickSize: 8 })`, §5.1), so `core` doesn't depend on `tools`' decoders. K3.2 adds a GPU builder in `core`.

Errors: `class VDBError extends Error { code }`; every message says what to do next.

| Code | When |
|---|---|
| `VDB_UNSUPPORTED_DEVICE` | no device, lost device, or a missing limit (`unsupportedReason` says which) |
| `VDB_BAD_IMAGE` | bad magic, ABI major ≠ 32, size mismatch, size ≥ 4 GiB, not 4-byte sized |
| `VDB_UNSUPPORTED_GRID_TYPE` | type outside Float/Fp4/Fp8/Fp16/FpN |
| `VDB_POOL_LIMIT` | growth would pass the device limit; names both numbers and `requiredLimits` |
| `VDB_POOL_FULL` | host-supplied buffer is full |
| `VDB_UNKNOWN_GRID` | dead or out-of-range id |
| `VDB_NO_STATS` | `nodeMax` requested for a grid without `HAS_MIN_MAX` (`tools` can add stats) |
| `VDB_TYPE_NOT_COMPILED` / `VDB_LAYOUT_MISMATCH` | `checkModule(pool, options)` |
| `VDB_BAD_OPTION` | invalid generator or pool options |
| `VDB_DISPOSED` | any call after `dispose` |
| `VDB_ENGINE_MISMATCH` | an engine adapter finds the engine's internals don't behave as it expects (K1.7a, for K2) |

## 5. Majorants

Delta and ratio tracking need, per ray segment, a bound `μ̄ ≥ density(x)` for every sample the integrator will
take in it. The iterator contract (both candidates):

```wgsl
var it = vdbMajorantInit(g, oIdx, dIdx, tMin, tMax);   // index-space ray, unnormalized dir, world t
loop { let s = vdbMajorantNext(&it); if (s.y <= s.x) { break; } /* s.z bounds density on [s.x, s.y) */ }
```

It mirrors the pathtracer's `ptDdaInit`/`ptDdaNext`, so its integrator ports as is. The spec sketch's single
`vdbMajorantSegment(...)` can't return a sequence; rejected.

**The bound is over the *sampled* field.** Trilinear at `p` reads voxels `floor(p)` and `floor(p) + 1`, so a cell
must bound every voxel within one voxel of it, inactive voxels, tiles and the background included.

### 5.1 `brick` (default candidate)

- Data: a dense f32 grid in the pool, x-fastest, one cell per `B³` index voxels (`B = 8`, the pathtracer's
  size), origin `indexBBoxMin − 1`, dims `ceil((max − min + 3) / B)`. Cell value = max of every value that can
  reach a sample in the cell: leaf voxels (active or not), tile values, and the background for cells touching
  space outside the tree.
- Built on the CPU by `tools.vdbBrickMajorant` (K1.4b; the CPU twin) and on the GPU by `core` in K3.2.
- Cost: `dims³ × 4` B; a 512³ grid is 65³ × 4 ≈ 1.1 MB. DDA ≈ 1.8 KB WGSL, no tree access in the walk.
- Fallback: a grid without a brick range walks as one `global` segment (record word 35). Correct, just slower.

### 5.2 `nodeMax` (A/B candidate, K3.2)

- Data: none beyond the image; needs `HAS_MIN_MAX` (record b1; `add` throws `VDB_NO_STATS` otherwise).
- Walk: HDDA at leaf dim (8), widening to 128/4096 over tiles and empty space via `vdbGetDim`. A leaf cell's bound
  is the max of the leaf-max stats of the up to 8 leaves its dilated support touches; a tile cell's bound is the
  tile value; empty space is the background.
- Caveats, each a test: NanoVDB stats cover **active** voxels only, so the bound assumes inactive leaf values ≤
  the leaf max (true for fog from `nanovdb_convert` and our serializer; the CPU twin checks it per fixture, and
  DIVERGES V5's sphere fixture with background 3 is in the set). The 8-leaf support costs up to 8 accessor
  descents per cell; that is what the A/B measures.

### 5.3 Fork bug found: Fp leaf stats

`pnanovdb_leaf_get_{min,max,ave,stddev}_float` read the stats slot as f32 for every type. For Fp4/Fp8/Fp16/FpN
leaves those slots are **u16 codes** at byte offsets 88/90/92/94 (`tools` `bytes.ts`, verified against native
fixtures), decoded as `code × quantum + minimum`; offset 90 isn't word-aligned, so the f32 read returns the
packed min|max pair. Upper/lower/root stats are f32 for all these types and read correctly. v1 never called
the leaf readers (it used the root max only), so no test caught it. K1.4b fixes it in the fork's WebVDB section
(diff log row, SHA update) with a CPU twin and parity. `nodeMax` depends on it; `brick` doesn't.

### 5.4 Tests (D11)

The CPU property test is the primary check: for every fixture × type × candidate, 1,000 rays × 256 jittered
samples, `sampleTrilinear(p) ≤ segment bound + 1e-6` (vitest, no GPU). GPU parity: segments from the generated
module equal the CPU twin's (`|Δt| ≤ 1e-5 · t`, bound exact).

## 6. three adapter contract (K2)

What K2.2 builds on this contract, following bvh's `src/three/trace-nodes.ts` (three r187):

```ts
const grids = new VDBGridNode(renderer, pool, { types: [...pool.types], profile: 'sample' });
// grids.storage = storage(standIn, 'uint', 1).toReadOnly().setName('vdbPool')
// grids.module  = a CodeNode subclass holding grids.storage as a node property; generate(builder):
//                   const ref = this.storage.build(builder, 'property');          // 'vdbPool.value'
//                   return vdbModuleWgsl({ ...opts, buffer: ref });               // memoized per ref
const fn = wgslFn(`fn shade(p: vec3f) -> f32 { var a = vdbAccessorInit(0u); return vdbSample(&a, p); }`, [grids.module]);
grids.update();   // each frame, after pool.update(encoder) was submitted: if pool.version changed, bind a new stand-in
```

- **Binding:** the stand-in `StorageBufferAttribute`'s backend record gets `buffer = pool.buffer` before three
  first sees it (`backend.get(attr).buffer`; three r185's `createAttribute` skips creation when it is set,
  checked). three emits `struct vdbPoolStruct { value: array<u32> }` at a binding it chooses; the generator's
  `buffer` mode reads `vdbPool.value[i]` at its one site. `setName` is public API and keeps shader text
  identical across loads; the v1 parts that go away are the regex over the vendored file and the
  `ptr<storage>` parameter that forced emission (the CodeNode's node property does that).
- **Rebind:** `update()` compares `pool.version`; on change it makes a new stand-in bound to the new buffer and
  sets the storage node's value. Nothing recompiles. `revision` changes need no three action.
- **Disposal:** three destroys an attribute's buffer on `dispose` and in `renderer.dispose()`
  (`destroyAttribute`, which calls `data.buffer.destroy()` in r185). Wrap `backend.destroyAttribute` once per
  backend so a stand-in gets a throwaway buffer first, as bvh does. The pool stays the owner.

three limitations found (none breaks the one-binding contract):

1. TSL owns `@group/@binding`; the host can't choose a slot. Hence the `buffer` mode. Still one binding.
2. Binding a foreign `GPUBuffer` uses backend internals (`backend.get`, `destroyAttribute`), not public API. Pin
   the three peer range and keep a compatibility check that fails loudly (bvh's `checkStandIn`).
3. **Version skew:** bvh pins three `>=0.187 <0.188`; WebVDB v1 is on r185. A host app has one three. K2 should
   move to bvh's range, or both packages will be unusable together.
4. Compatibility-mode devices may allow 0 storage buffers in the fragment stage, so a fragment-stage volume
   material needs a core device; `unsupportedReason` reports it.
5. `wgslFn` parses only the first `fn`; helpers belong in the module CodeNode (v1 learned this).

## 7. Parity harness (K1.6) and raw-WebGPU example (K1.7)

**Harness.** Replace the hand footer (`examples/src/harness/shader-footer.ts`, which declares `nanovdb_buffer`
and calls `pnanovdb_*`) with generated modules plus small entry points that call only the public API:

- One pool holds **all nine fixtures** (box, sphere, torus × Float, Fp8, FpN), built with a tiny
  `initialBytes` so it grows and compacts twice. Every grid then sits at a non-zero offset, which v1 never
  tested.
- Matrix: profile × type set {`float`}, {`fp8`}, {`fpn`}, {all three}, plus `isActive: false`,
  `sampler: 'nearest'`, `outside: 'zero'`, `majorant: 'brick' | 'global'`. Each variant compiles under
  `pushErrorScope('validation')`.
- Checks per fixture: `read`, 2,000 probes, `|Δ| ≤ 1e-5`, active exact (v1's bounds); `sample`, 500 trilinear
  points `|Δ| ≤ 1e-3`, and world↔index against the f64 CPU map with an instance transform set; `march`, 200 rays,
  HDDA visit sequences equal to the CPU twin (voxel exact, `t` within 1e-5 relative), `vdbGetDim` exact,
  `vdbNodeMax` exact; `majorant`, §5.4. Negative cases: a Float-only module reading an Fp8 grid returns 0 with
  `ok == false`; a removed id returns 0.
- CPU twins live in `tools` (existing `readValue`, `sampleTrilinear`; new `hddaWalk`, `getDim`, `nodeStats`,
  `vdbBrickMajorant`, `majorantWalk`). Results publish to `window.__GPU_PARITY__` as today; the e2e spec asserts
  every row.

**Raw-WebGPU example** (`examples/webgpu/smoke`, upscaler's `examples/webgpu/hello` precedent): the host creates
the device with `VDBGridPool.requiredLimits(adapter)`, loads the smoke `.nvdb` through `tools`, adds it, and
draws a full-screen triangle whose fragment shader is `vdbModuleWgsl({ slot: { group: 0, firstBinding: 1 },
types, profile: 'sample' })` plus a ~60-line index-space march (not the K3 renderer). Per frame: one encoder,
`pool.update(encoder)`, the render pass, submit; it rebuilds its bind group when `pool.version` changes. A button
adds a second grid at runtime (forces growth) and animates `setTransform` to show that a transform moves a grid
without touching its image. It is the reference for the engine contract.

## 8. Tasks K1.3–K1.8

Order: K1.3a → (K1.3b ∥ K1.4a) → K1.4b → K1.5 ∥ K1.6 → K1.7. K1.8 can start after K1.4a and does not gate K1.

| ID | Tier | Scope (files under `packages/vdb/src/`) | Acceptance |
|---|---|---|---|
| K1.3a | T2 | `core/pool/layout.ts`: header and record constants, `encodeRecord`/`decodeRecord`, `VDB_GPU_LAYOUT_VERSION`; `core/errors.ts` | Round-trip test; golden words for one fixture's record; WGSL constants generated from the same file (test compares) |
| K1.3b | T2 | `core/pool/{pool,allocator,image-header,limits}.ts`: §1 and §4 | Vitest against a recording fake device: add/replace/remove/setTransform staging; growth records copies of resident ranges only, bumps `version` once, retires and later destroys the old buffer; compaction on failed first-fit; `VDB_POOL_LIMIT` leaves the pool unchanged; `dispose` destroys exactly its own buffers and never a host buffer; 32 B alignment; f64 transform composition vs a reference within 1e-6; `revision` vs `version` semantics. Browser test: nine fixtures in a pool started at 4 KiB, read back, every image range byte-identical to its file |
| K1.4a | T3 | `scripts/extract-wgsl-chunks`, `core/wgsl/pnanovdb.chunks.generated.ts`, `core/wgsl/generate.ts`, `core/wgsl/api.ts` (`read`, `sample`), `vdbModuleEntries`, `checkModule`; `VENDOR.md` generation-transform rows | Chunk sync test; deterministic output; `wgslNameProblems` empty for every variant; output contains no `pnanovdb`, no `nanovdb_buffer`, exactly one buffer access; `slot` vs `buffer` modes differ only in the declaration and that site; a browser page compiles every variant with no errors |
| K1.4b | T3 | `march` and `majorant` (`brick`, `global`); the Fp leaf-stats fix in the fork (diff log + SHA); `tools`: `hddaWalk`, `getDim`, `nodeStats`, `vdbBrickMajorant`, `majorantWalk` | CPU majorant property test (§5.4) on all nine fixtures; Fp leaf stats match `tools` `stats.ts` on Fp8/FpN fixtures; `nodeMax` interface stubbed with `VDB_BAD_OPTION` "lands in K3.2" until then |
| K1.5 | T1 | `scripts/wgsl-size` and `core/wgsl/size-snapshot.json`: every profile × type set × option, min and gz bytes | Snapshot committed; a vitest fails when a variant's bytes differ from the snapshot or pass the §3.6 budget; compile-time column filled by the owner on the reference device (not in CI) |
| K1.6 | T2 | Harness on generated modules (§7), `e2e/gpu-parity.spec.ts` | Green on SwiftShader for every row of the §7 matrix; then a fresh-eyes verification agent re-derives one fixture's numbers from the CPU twins (D11) |
| K1.7 | T2 | `examples/webgpu/smoke` + `e2e/webgpu-smoke.spec.ts` | Golden image within the demo threshold; the second-grid button bumps `version` and still renders; an import-graph test shows no `three` in the example; the page prints its WGSL bytes |
| K1.8 | T3 | `reader: 'lean'` (optional): a compact single-type descent (no handle structs, literal offsets, leaf-only cache), derived from `PNanoVDB.h` with attribution | Same parity rows as `read` and `sample`; ≤ 4 KB min for `read`/Float; an A/B of march time vs the vendored accessor on the reference device, recorded before any default changes (D11) |

## 9. Open questions

1. **The 4 KB `read` gate.** Measured 16.2 KB for `read`/Float after every specialization (§3.5).
   *Recommendation:* gate K1 on the §3.6 vendored budgets and move the 4 KB target to K1.8 (`reader: 'lean'`),
   off the critical path. *Pros:* the audited traversal ships unchanged; the cloud system doesn't march the tree
   on the GPU at all (its P8 spec decodes leaves into a brick pool), and the pathtracer comes after 0.1 (D19).
   *Cons:* 0.1 ships without the pathtracer-sized reader. *Settled by:* K1.5's table plus K1.8's A/B. If the lean
   reader matches parity and isn't slower in the march, it becomes the default for `read`/`sample`.
2. **`brick` or `nodeMax` as the default majorant.** *Recommendation:* `brick` now; A/B in K3.2. `brick` is
   tighter, needs no stats, and doesn't depend on the §5.3 fix; `nodeMax` needs no memory or build step.
   *Settled by:* K3.2, null collisions per ray and ms per frame on WDAS quarter and the smoke, same seed.
3. **Generated-WGSL licence banner.** A one-line SPDX comment (§3.3) plus NOTICE in the package. *Owner confirms*
   with D13. No test settles it.
4. **three peer range** (§6 item 3). *Recommendation:* K2 targets bvh's r187 range. *Settled by:* K2.2's
   stand-in compatibility test passing on that release.
5. **Trilinear cost.** The cloud system measured the 8-tap accessor trilinear at 2.2–2.7× a hardware fetch.
   *Recommendation:* a same-leaf fast path in `vdbSample` (decode 8 taps from one leaf when they share it, about
   2/3 of interior samples) as a K3 perf item behind the existing parity rows, not a K1 change.

## 10. Records to add when K1 lands

- `VENDOR.md`: one row per generation transform (§3.3) and the Fp leaf-stats fix (§5.3).
- `DIVERGES.md`: grid-type table specialization and the prefix rename (vs. emcfarlane's file); the record's f32
  affines composed in f64 (vs. NanoVDB's in-image map); `outside: 'zero'` (vs. NanoVDB background).
- `CLAUDE.md` Traps: "Fp leaf stats are u16 codes; never read them as f32"; "one VDB module per shader".
