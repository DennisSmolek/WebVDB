/**
 * Compute toolset (SPEC §3.3, docs/PLAN.md Phase 4): three raw-WebGPU compute
 * passes over a `NanoVDBGrid` — `gridStats`, `valueTransform`,
 * `decodeToAtlas`. Deliberately raw WebGPU, NOT TSL compute (the brief's own
 * words): `NanoVDBVolumeMaterial` (`material.ts`/`wgsl.ts`) is a `NodeMaterial`
 * fragment shader, so it needs TSL to own bind-group layout and buffer
 * naming; a standalone compute utility owns its own pipeline end to end, so
 * it can bind the vendored library's `nanovdb_buffer` global directly — the
 * same pattern `examples/src/harness/main.ts` (Phase 2) already proved.
 *
 * ## Shared shader assembly
 *
 * All three utilities compile ONE shader module: the vendored
 * `pnanovdb.wgsl` text (unmodified — no `rewriteBufferGlobal`; raw WebGPU
 * just declares `nanovdb_buffer` under the name the library already expects,
 * exactly like the harness footer) plus `COMPUTE_FOOTER_WGSL` below, which
 * adds four `@compute` entry points. `nanovdb_buffer` is declared
 * `read_write` (not `read-only-storage`) even though `gridStats`/
 * `decodeToAtlas` never write it — a single access mode lets one shader
 * module serve all utilities, including `valueTransform`'s in-place mutator,
 * without a second compiled variant. `assertHasBufferGlobal` (shared with
 * `wgsl.ts` — see its doc comment) guards the same "did the caller actually
 * pass the real vendored source" mistake `assembleVolumeWgsl` guards.
 *
 * Compiled modules are cached per `(device, assembled source text)` so
 * calling multiple utilities against the same device/source (as demo 04
 * does) only compiles once.
 *
 * ## Dispatch shape: always 3D
 *
 * Every entry point dispatches `@workgroup_size(4, 4, 4)` (64 threads/
 * workgroup, matching the harness's 1D workgroup size of 64) over a 3D grid
 * sized to the index-space region being processed (the grid's dense index
 * bbox for `gridStats`/`valueTransform`, the output atlas `dims` for
 * `decodeToAtlas`), bounds-checked per-thread. A 1D dispatch would need
 * `ceil(voxelCount / 64)` workgroups in a single dimension, which risks
 * exceeding `maxComputeWorkgroupsPerDimension` (typically 65535) for large
 * grids; splitting the same total thread count across x/y/z keeps every
 * per-axis workgroup count small regardless of overall voxel count.
 *
 * ## Known v1 limitation: dense bbox, not sparse leaf enumeration
 *
 * `gridStats` and `valueTransform` dispatch one thread per voxel in the
 * grid's FULL index bbox (`pnanovdb_root_get_bbox_{min,max}`), calling the
 * accessor per-thread — correct or a sparse grid with a huge bbox but few
 * active voxels wastes threads on background reads. Fine for this project's
 * fixture-scale grids (SPEC §6: primitives + WDAS/EmberGen, all ≤ low
 * millions of bbox voxels); a real out-of-core-scale grid would want to walk
 * the tree's leaf-node array directly instead (a documented v2 improvement,
 * not attempted here).
 */
import type { NanoVDBGrid } from "./grid.js";
import { assertHasBufferGlobal, GRID_TYPE_FLOAT } from "./wgsl.js";

// ---------------------------------------------------------------------------
// Shared WGSL footer
// ---------------------------------------------------------------------------

/** Threads per workgroup per axis (4*4*4 = 64, matching the harness's 1D 64). */
const WORKGROUP_AXIS = 4;

/**
 * `array<u32>` param layout shared by every entry point (group 0, binding 1):
 *   [0] gridTypeId
 *   [1..3] bbox/region min (i32, index space) — ijk origin for gridStats/
 *          valueTransform, unused (0) for decodeToAtlas's own bbox-relative math
 *   [4..6] region size in voxels (u32-valued, stored as i32) — the dense
 *          dispatch extent for gridStats/valueTransform, the SOURCE bbox size
 *          for decodeToAtlas (its OUTPUT dims are separate, see below)
 *   [7..9] decodeToAtlas output dims (unused by the other entry points)
 *   [10]   decodeToAtlas filter mode: 0 = nearest, 1 = trilinear
 *   [11]   histogram bin count (gridStats only)
 */
const PARAMS_I32_LENGTH = 12;

/** `array<f32>` params (group 0, binding 2): [0] rangeLo [1] rangeHi. */
const PARAMS_F32_LENGTH = 2;

/**
 * The library's own module header says a consumer must declare
 * `nanovdb_buffer`; the shared footer below is that declaration plus our
 * five (four static + one for `valueTransform`, appended separately since it
 * needs the caller's WGSL body inlined — see `buildValueTransformSource`)
 * compute entry points.
 */
const COMPUTE_FOOTER_WGSL = /* wgsl */ `
// ===========================================================================
// three-nanovdb compute-utilities footer (compute.ts). Raw WebGPU, no TSL.
// ===========================================================================

@group(0) @binding(0) var<storage, read_write> nanovdb_buffer : array<u32>;
@group(0) @binding(1) var<storage, read> nvdbx_params_i32 : array<i32>;
@group(0) @binding(2) var<storage, read> nvdbx_params_f32 : array<f32>;

fn nvdbx_grid_type() -> u32 { return u32(nvdbx_params_i32[0]); }
fn nvdbx_region_min() -> vec3<i32> {
  return vec3<i32>(nvdbx_params_i32[1], nvdbx_params_i32[2], nvdbx_params_i32[3]);
}
fn nvdbx_region_size() -> vec3<u32> {
  return vec3<u32>(u32(nvdbx_params_i32[4]), u32(nvdbx_params_i32[5]), u32(nvdbx_params_i32[6]));
}
fn nvdbx_root() -> pnanovdb_root_handle_t {
  let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
  let grid = pnanovdb_grid_handle_t(0u);
  let tree = pnanovdb_grid_get_tree(grid);
  return pnanovdb_tree_get_root(buf, tree);
}

// --- Pass 0: root min/max (one thread). Decides gridStats's atomic
// fast-path vs. dump-and-CPU-reduce fallback, and gives the histogram bin
// range / decodeToAtlas's default normalize range — see compute.ts. ---
@group(1) @binding(0) var<storage, read_write> nvdbx_minmax_out : array<f32>; // [min, max]

@compute @workgroup_size(1)
fn nvdbx_root_minmax() {
  let grid_type = nvdbx_grid_type();
  let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
  let root = nvdbx_root();
  nvdbx_minmax_out[0] = pnanovdb_root_get_min_float(grid_type, buf, root);
  nvdbx_minmax_out[1] = pnanovdb_root_get_max_float(grid_type, buf, root);
}

// --- gridStats fast path (root min >= 0 — see compute.ts's bitcast-ordering
// doc). One dense-bbox dispatch; atomics accumulate min/max/count/sum/hist. ---
@group(1) @binding(0) var<storage, read_write> nvdbx_stats_i32_out : array<atomic<i32>>; // [0]=min [1]=max
// [0]=activeCount [1]=sumLo [2]=sumHi [3..]=histogram[bins]
@group(1) @binding(1) var<storage, read_write> nvdbx_stats_u32_out : array<atomic<u32>>;

// Fixed-point scale for the 64-bit (sumHi:sumLo) sum accumulator: WGSL has no
// atomic<f32> add, so each active voxel's value is normalized into
// [0, NVDBX_SUM_SCALE] against the root [rangeLo, rangeHi] stat range, then
// added into a manually carried 64-bit integer (see the carry check below).
const NVDBX_SUM_SCALE : f32 = 1048576.0; // 2^20 — ~20 bits of fractional precision

@compute @workgroup_size(${WORKGROUP_AXIS}, ${WORKGROUP_AXIS}, ${WORKGROUP_AXIS})
fn nvdbx_stats_fast(@builtin(global_invocation_id) gid : vec3<u32>) {
  let size = nvdbx_region_size();
  if (gid.x >= size.x || gid.y >= size.y || gid.z >= size.z) { return; }
  let grid_type = nvdbx_grid_type();
  let ijk = nvdbx_region_min() + vec3<i32>(gid);
  let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
  let root = nvdbx_root();
  var acc : pnanovdb_readaccessor_t;
  pnanovdb_readaccessor_init(&acc, root);
  if (!pnanovdb_readaccessor_is_active(grid_type, buf, &acc, ijk)) { return; }
  let value = pnanovdb_readaccessor_get_value_float(grid_type, buf, &acc, ijk);

  // Non-negative-float bitcast-to-i32 preserves numeric order: for v >= 0 the
  // IEEE-754 sign bit is 0, so reinterpreting the bit pattern as a signed i32
  // yields the same relative ordering as the float itself — the standard
  // atomicMin/atomicMax-on-floats trick, valid ONLY because the caller
  // checked the root min stat is >= 0 before ever dispatching this entry
  // point (see gridStats() in compute.ts).
  let bits = bitcast<i32>(value);
  atomicMin(&nvdbx_stats_i32_out[0], bits);
  atomicMax(&nvdbx_stats_i32_out[1], bits);
  atomicAdd(&nvdbx_stats_u32_out[0], 1u);

  let range_lo = nvdbx_params_f32[0];
  let range_hi = nvdbx_params_f32[1];
  let span = max(range_hi - range_lo, 1.0e-20);
  let norm = clamp((value - range_lo) / span, 0.0, 1.0);

  // 64-bit carry-checked add: atomicAdd returns the PRE-add value, so if
  // old_lo would overflow past 0xFFFFFFFF once "scaled" is added, bump the
  // high word by exactly one.
  let scaled = u32(norm * NVDBX_SUM_SCALE);
  let old_lo = atomicAdd(&nvdbx_stats_u32_out[1], scaled);
  if (old_lo > (0xFFFFFFFFu - scaled)) {
    atomicAdd(&nvdbx_stats_u32_out[2], 1u);
  }

  // Histogram over [rangeLo, rangeHi] (the root's own reported stats range,
  // not the exact per-voxel scan min/max — see compute.ts's gridStats doc).
  let bins = u32(nvdbx_params_i32[11]);
  let bin = min(u32(norm * f32(bins)), bins - 1u);
  atomicAdd(&nvdbx_stats_u32_out[3u + bin], 1u);
}

// --- gridStats fallback path (root min < 0 — the fast path's bitcast trick
// doesn't hold): dump (value, active) for every dense-bbox voxel; compute.ts
// reduces min/max/mean/histogram/count on the CPU after one readback ("the
// two-pass" the design brief calls for — a GPU dump pass + a CPU reduce
// pass, rather than a second GPU pass). ---
@group(1) @binding(0) var<storage, read_write> nvdbx_dump_value_out : array<f32>;
@group(1) @binding(1) var<storage, read_write> nvdbx_dump_active_out : array<u32>;

@compute @workgroup_size(${WORKGROUP_AXIS}, ${WORKGROUP_AXIS}, ${WORKGROUP_AXIS})
fn nvdbx_stats_dump(@builtin(global_invocation_id) gid : vec3<u32>) {
  let size = nvdbx_region_size();
  if (gid.x >= size.x || gid.y >= size.y || gid.z >= size.z) { return; }
  let idx = gid.x + gid.y * size.x + gid.z * size.x * size.y;
  let grid_type = nvdbx_grid_type();
  let ijk = nvdbx_region_min() + vec3<i32>(gid);
  let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
  let root = nvdbx_root();
  var acc : pnanovdb_readaccessor_t;
  pnanovdb_readaccessor_init(&acc, root);
  let is_act = pnanovdb_readaccessor_is_active(grid_type, buf, &acc, ijk);
  let value = pnanovdb_readaccessor_get_value_float(grid_type, buf, &acc, ijk);
  nvdbx_dump_value_out[idx] = value;
  nvdbx_dump_active_out[idx] = select(0u, 1u, is_act);
}

// --- decodeToAtlas: one thread per OUTPUT voxel (dims from params_i32[7..9]),
// sampling the SOURCE region (params_i32[1..6]) at a cell-centered continuous
// index-space position. Always writes raw f32 (no normalize/quantize on the
// GPU) — compute.ts does that in JS after readback, so the shader stays
// filter-mode-only and format-agnostic. ---
@group(1) @binding(0) var<storage, read_write> nvdbx_atlas_out : array<f32>;

@compute @workgroup_size(${WORKGROUP_AXIS}, ${WORKGROUP_AXIS}, ${WORKGROUP_AXIS})
fn nvdbx_atlas_decode(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = vec3<u32>(u32(nvdbx_params_i32[7]), u32(nvdbx_params_i32[8]), u32(nvdbx_params_i32[9]));
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
  let grid_type = nvdbx_grid_type();
  let region_min = nvdbx_region_min();
  let region_size = nvdbx_region_size();
  let filter_mode = u32(nvdbx_params_i32[10]);

  // Cell-centered sample position: (i + 0.5) / dims maps the output voxel to
  // a fraction of the source region, then scaled back into index space.
  let t = (vec3<f32>(gid) + vec3<f32>(0.5, 0.5, 0.5)) / vec3<f32>(dims);
  let pos = vec3<f32>(region_min) + t * vec3<f32>(region_size);

  let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
  let root = nvdbx_root();
  var acc : pnanovdb_readaccessor_t;
  pnanovdb_readaccessor_init(&acc, root);

  var value : f32;
  if (filter_mode == 1u) {
    value = pnanovdb_sample_trilinear_typed(grid_type, buf, &acc, pos);
  } else {
    let ijk = vec3<i32>(floor(pos + vec3<f32>(0.5, 0.5, 0.5)));
    value = pnanovdb_readaccessor_get_value_float(grid_type, buf, &acc, ijk);
  }

  let idx = gid.x + gid.y * dims.x + gid.z * dims.x * dims.y;
  nvdbx_atlas_out[idx] = value;
}
`;

/**
 * `valueTransform`'s entry point is built separately because the caller's
 * WGSL body has to be inlined at shader-build time (WGSL has no runtime
 * function parameter). It reads the OLD value via the library's own
 * `pnanovdb_read_float` at the address `pnanovdb_readaccessor_get_value_
 * address_and_level` resolves, then writes the transformed value back into
 * the SAME `nanovdb_buffer` binding directly (bypassing the library, which is
 * read-only by design) — see `valueTransform`'s doc comment in this file for
 * why only `level == 0u` (an individually-addressable LEAF voxel, not a
 * shared upper/lower/root tile constant) is mutated.
 */
function buildValueTransformEntry(transformBody: string): string {
  return /* wgsl */ `
@compute @workgroup_size(${WORKGROUP_AXIS}, ${WORKGROUP_AXIS}, ${WORKGROUP_AXIS})
fn nvdbx_value_transform(@builtin(global_invocation_id) gid : vec3<u32>) {
  let size = nvdbx_region_size();
  if (gid.x >= size.x || gid.y >= size.y || gid.z >= size.z) { return; }
  let grid_type = nvdbx_grid_type();
  let ijk = nvdbx_region_min() + vec3<i32>(gid);
  let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
  let root = nvdbx_root();
  var acc : pnanovdb_readaccessor_t;
  pnanovdb_readaccessor_init(&acc, root);
  let resolved = pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, &acc, ijk);
  if (resolved.level != 0u) {
    return; // shared tile/background value, not an individually-stored leaf voxel
  }
  let old_value = pnanovdb_read_float(buf, resolved.address);
  let new_value = nvdbx_transform(old_value);
  nanovdb_buffer[resolved.address >> 2u] = bitcast<u32>(new_value);
}

fn nvdbx_transform(v : f32) -> f32 {
  ${transformBody}
}
`;
}

/** Assembles the static (gridStats/decodeToAtlas) module source. Pure, testable. */
export function buildComputeShaderSource(pnanovdbSource: string): string {
  assertHasBufferGlobal(pnanovdbSource);
  return `${pnanovdbSource}\n${COMPUTE_FOOTER_WGSL}`;
}

/** Assembles the `valueTransform` module source (library + footer + inlined transform fn). */
export function buildValueTransformShaderSource(pnanovdbSource: string, transformBody: string): string {
  assertHasBufferGlobal(pnanovdbSource);
  if (!transformBody || typeof transformBody !== "string" || transformBody.trim().length === 0) {
    throw new Error("buildValueTransformShaderSource: `transformBody` must be a non-empty WGSL function body.");
  }
  return `${pnanovdbSource}\n${COMPUTE_FOOTER_WGSL}\n${buildValueTransformEntry(transformBody)}`;
}

/**
 * Named presets for `valueTransform`'s `wgslExprOrPreset` argument — small,
 * common per-voxel edits spelled out as WGSL bodies so callers don't have to
 * hand-write WGSL for the obvious cases. Anything else is treated as a raw
 * WGSL function body (e.g. `"return v * 2.0;"`).
 */
export const VALUE_TRANSFORM_PRESETS: Readonly<Record<string, string>> = {
  double: "return v * 2.0;",
  half: "return v * 0.5;",
  negate: "return -v;",
  clamp01: "return clamp(v, 0.0, 1.0);",
};

/** Resolves `wgslExprOrPreset` to a WGSL function body: a known preset name, or the string verbatim. */
export function resolveTransformBody(wgslExprOrPreset: string): string {
  return VALUE_TRANSFORM_PRESETS[wgslExprOrPreset] ?? wgslExprOrPreset;
}

// ---------------------------------------------------------------------------
// Pure helpers (index-bbox / atlas-dims math) — exported for unit testing.
// ---------------------------------------------------------------------------

export interface IndexBBox {
  min: readonly [number, number, number];
  max: readonly [number, number, number];
}

/** Per-axis voxel count of an inclusive-max index bbox (`max - min + 1`). */
export function bboxSize(bbox: IndexBBox): [number, number, number] {
  return [bbox.max[0] - bbox.min[0] + 1, bbox.max[1] - bbox.min[1] + 1, bbox.max[2] - bbox.min[2] + 1];
}

/**
 * `decodeToAtlas` output resolution: each axis of the source index bbox,
 * clamped to `maxDim` (SPEC §3.3 / the mini-design: "clamped to opts.maxDim,
 * default 256" — a literal per-axis clamp, not a proportional downscale).
 */
export function computeAtlasDims(bbox: IndexBBox, maxDim: number): [number, number, number] {
  const [sx, sy, sz] = bboxSize(bbox);
  return [Math.min(sx, maxDim), Math.min(sy, maxDim), Math.min(sz, maxDim)];
}

function assertPositiveInt(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`compute.ts: \`${name}\` must be an integer >= 1 (got ${JSON.stringify(value)}).`);
  }
}

// ---------------------------------------------------------------------------
// GPU plumbing shared by all three utilities.
// ---------------------------------------------------------------------------

const moduleCache = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();

function getOrCreateModule(device: GPUDevice, source: string): GPUShaderModule {
  let bySource = moduleCache.get(device);
  if (!bySource) {
    bySource = new Map();
    moduleCache.set(device, bySource);
  }
  let module = bySource.get(source);
  if (!module) {
    module = device.createShaderModule({ code: source });
    bySource.set(source, module);
  }
  return module;
}

function alignTo4(n: number): number {
  return Math.ceil(n / 4) * 4;
}

function createStorageBuffer(device: GPUDevice, byteLength: number, extraUsage = 0): GPUBuffer {
  return device.createBuffer({
    size: Math.max(4, alignTo4(byteLength)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
  });
}

function writeArray(device: GPUDevice, buffer: GPUBuffer, data: Uint32Array | Int32Array | Float32Array): void {
  device.queue.writeBuffer(buffer, 0, data.slice());
}

async function readback(device: GPUDevice, src: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
  const dst = device.createBuffer({
    size: Math.max(4, alignTo4(byteLength)),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, dst, 0, dst.size);
  device.queue.submit([encoder.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const copy = dst.getMappedRange().slice(0);
  dst.unmap();
  dst.destroy();
  return copy;
}

/** Group(0) layout shared by every pipeline (nanovdb_buffer + the two params arrays). */
function createGroup0Layout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
}

function makeParamsI32(overrides: Partial<Record<number, number>>): Int32Array {
  const arr = new Int32Array(PARAMS_I32_LENGTH);
  for (const [i, v] of Object.entries(overrides)) arr[Number(i)] = v ?? 0;
  return arr;
}

function makeParamsF32(rangeLo: number, rangeHi: number): Float32Array {
  const arr = new Float32Array(PARAMS_F32_LENGTH);
  arr[0] = rangeLo;
  arr[1] = rangeHi;
  return arr;
}

interface Group0Buffers {
  gridBuffer: GPUBuffer;
  paramsI32Buffer: GPUBuffer;
  paramsF32Buffer: GPUBuffer;
  group0Layout: GPUBindGroupLayout;
  group0: GPUBindGroup;
}

function setUpGroup0(device: GPUDevice, image: Uint32Array, paramsI32: Int32Array, paramsF32: Float32Array): Group0Buffers {
  const group0Layout = createGroup0Layout(device);
  const gridBuffer = createStorageBuffer(device, image.byteLength);
  writeArray(device, gridBuffer, image);
  const paramsI32Buffer = createStorageBuffer(device, paramsI32.byteLength);
  writeArray(device, paramsI32Buffer, paramsI32);
  const paramsF32Buffer = createStorageBuffer(device, paramsF32.byteLength);
  writeArray(device, paramsF32Buffer, paramsF32);
  const group0 = device.createBindGroup({
    layout: group0Layout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: paramsI32Buffer } },
      { binding: 2, resource: { buffer: paramsF32Buffer } },
    ],
  });
  return { gridBuffer, paramsI32Buffer, paramsF32Buffer, group0Layout, group0 };
}

function destroyGroup0(g: Group0Buffers): void {
  g.gridBuffer.destroy();
  g.paramsI32Buffer.destroy();
  g.paramsF32Buffer.destroy();
}

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

function dispatchDims3(size: readonly [number, number, number]): [number, number, number] {
  return [ceilDiv(size[0], WORKGROUP_AXIS), ceilDiv(size[1], WORKGROUP_AXIS), ceilDiv(size[2], WORKGROUP_AXIS)];
}

/** Runs `nvdbx_root_minmax` (one thread) and returns `[min, max]`. */
async function queryRootMinMax(
  device: GPUDevice,
  module: GPUShaderModule,
  group0Layout: GPUBindGroupLayout,
  group0: GPUBindGroup,
): Promise<[number, number]> {
  const group1Layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
  });
  const outBuffer = createStorageBuffer(device, 8);
  const group1 = device.createBindGroup({ layout: group1Layout, entries: [{ binding: 0, resource: { buffer: outBuffer } }] });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [group0Layout, group1Layout] }),
    compute: { module, entryPoint: "nvdbx_root_minmax" },
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group0);
  pass.setBindGroup(1, group1);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const bytes = await readback(device, outBuffer, 8);
  outBuffer.destroy();
  const f32 = new Float32Array(bytes);
  return [f32[0]!, f32[1]!];
}

// ---------------------------------------------------------------------------
// gridStats
// ---------------------------------------------------------------------------

export interface GridStatsOptions {
  /** Number of histogram bins. Default 256. */
  histogramBins?: number;
}

export interface GridStatsResult {
  min: number;
  max: number;
  mean: number;
  /** `histogramBins`-length (default 256) bin counts over `[rangeLo, rangeHi]` — see the module doc. */
  histogram: Uint32Array;
  activeVoxelCount: number;
  /**
   * `true` if the single-pass GPU-atomics path ran (root min >= 0); `false`
   * if the grid has negative values and the dump-and-CPU-reduce fallback ran
   * instead (see the module doc's "atomics design" discussion).
   */
  usedAtomicFastPath: boolean;
}

const DEFAULT_HISTOGRAM_BINS = 256;

/**
 * `gridStats(device, grid, pnanovdbSource, opts?)` — histogram/min-max/mean/
 * active-voxel-count readback over the grid's full index bbox (SPEC §3.3).
 *
 * ## Deviation from the mini-design's literal signature
 *
 * The design sketch was `gridStats(device, grid, opts?)`. This package (see
 * `material.ts`'s `pnanovdbSource` parameter) deliberately never does a
 * bundler-specific `?raw` import itself, so — matching `NanoVDBVolumeMaterial`
 * exactly — the vendored WGSL text is a required third parameter here instead
 * of living inside `opts`. Same reasoning, same shape as the existing
 * Phase 3 API.
 *
 * ## Atomics design (see also the WGSL footer's inline comments)
 *
 * One dense-bbox dispatch (`nvdbx_stats_fast`) accumulates via atomics:
 * `atomicMin`/`atomicMax` on the value's `bitcast<i32>` bit pattern (valid
 * because non-negative IEEE-754 floats preserve their ordering when
 * reinterpreted as signed 32-bit integers — the standard trick, and ONLY
 * valid for non-negative values); a plain `atomicAdd` for the active-voxel
 * count; a hand-rolled 64-bit (`sumHi`:`sumLo`) fixed-point accumulator for
 * the sum (WGSL has no `atomic<f32>` add); and `atomicAdd` histogram bins
 * over `[rootMin, rootMax]` (the grid's own reported stats range, not the
 * exact per-voxel scan extrema — the `min`/`max` fields ARE the exact scan
 * extrema, so the two can disagree by a hair if quantization decode ever
 * excurses past the stored root bounds; documented, not "fixed", since it
 * never happens on this project's fixtures).
 *
 * Whether the fast path is even safe is decided by a cheap 1-thread pass
 * (`nvdbx_root_minmax`) reading the grid's own min/max stats FIRST. If the
 * root min is negative, `gridStats` falls back to a dump-and-CPU-reduce pass
 * (`nvdbx_stats_dump`): every dense-bbox voxel's `(value, active)` is written
 * out, then min/max/mean/histogram/count are computed in JS after one
 * readback — a "two-pass" in the sense the mini-design asked for (a GPU pass
 * + a CPU reduce), just without a second GPU atomics pass.
 */
export async function gridStats(
  device: GPUDevice,
  grid: NanoVDBGrid,
  pnanovdbSource: string,
  opts: GridStatsOptions = {},
): Promise<GridStatsResult> {
  const histogramBins = opts.histogramBins ?? DEFAULT_HISTOGRAM_BINS;
  assertPositiveInt("histogramBins", histogramBins);

  const source = buildComputeShaderSource(pnanovdbSource);
  const module = getOrCreateModule(device, source);

  const bbox = grid.metadata.indexBBox;
  const [minX, minY, minZ] = bbox.min;
  const size = bboxSize(bbox);

  const paramsI32 = makeParamsI32({ 0: grid.gridTypeId, 1: minX, 2: minY, 3: minZ, 4: size[0], 5: size[1], 6: size[2], 11: histogramBins });
  const g0 = setUpGroup0(device, grid.image, paramsI32, makeParamsF32(0, 0));

  const [rangeLo, rangeHi] = await queryRootMinMax(device, module, g0.group0Layout, g0.group0);
  // Re-upload params_f32 now that the real range is known (the fast/dump
  // entry points both read it).
  writeArray(device, g0.paramsF32Buffer, makeParamsF32(rangeLo, rangeHi));

  const [dx, dy, dz] = dispatchDims3(size);
  const usedAtomicFastPath = rangeLo >= 0;

  if (usedAtomicFastPath) {
    const group1Layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const i32Out = createStorageBuffer(device, 2 * 4);
    const u32Out = createStorageBuffer(device, (3 + histogramBins) * 4);
    // Sentinels: min-accumulator starts at i32 max (any real non-negative
    // float's bit pattern is smaller), max-accumulator starts at 0 (the bit
    // pattern of +0.0, the smallest non-negative bit pattern).
    writeArray(device, i32Out, new Int32Array([0x7fffffff, 0]));
    writeArray(device, u32Out, new Uint32Array(3 + histogramBins));
    const group1 = device.createBindGroup({
      layout: group1Layout,
      entries: [
        { binding: 0, resource: { buffer: i32Out } },
        { binding: 1, resource: { buffer: u32Out } },
      ],
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [g0.group0Layout, group1Layout] }),
      compute: { module, entryPoint: "nvdbx_stats_fast" },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, g0.group0);
    pass.setBindGroup(1, group1);
    pass.dispatchWorkgroups(dx, dy, dz);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const i32Bytes = await readback(device, i32Out, 2 * 4);
    const u32Bytes = await readback(device, u32Out, (3 + histogramBins) * 4);
    i32Out.destroy();
    u32Out.destroy();
    destroyGroup0(g0);

    const minMaxI32 = new Int32Array(i32Bytes);
    const u32 = new Uint32Array(u32Bytes);
    const activeVoxelCount = u32[0]!;
    const sumLo = u32[1]!;
    const sumHi = u32[2]!;
    const histogram = u32.slice(3, 3 + histogramBins);

    if (activeVoxelCount === 0) {
      return { min: 0, max: 0, mean: 0, histogram, activeVoxelCount: 0, usedAtomicFastPath };
    }

    const f32 = new Float32Array(minMaxI32.buffer);
    const min = f32[0]!;
    const max = f32[1]!;
    const totalScaled = BigInt(sumHi) * (1n << 32n) + BigInt(sumLo);
    const meanNorm = Number(totalScaled) / (1048576 /* NVDBX_SUM_SCALE */ * activeVoxelCount);
    const mean = rangeLo + meanNorm * (rangeHi - rangeLo);

    return { min, max, mean, histogram, activeVoxelCount, usedAtomicFastPath };
  }

  // Fallback: dump every dense-bbox voxel, reduce in JS.
  const voxelCount = size[0] * size[1] * size[2];
  const group1Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const valueOut = createStorageBuffer(device, voxelCount * 4);
  const activeOut = createStorageBuffer(device, voxelCount * 4);
  const group1 = device.createBindGroup({
    layout: group1Layout,
    entries: [
      { binding: 0, resource: { buffer: valueOut } },
      { binding: 1, resource: { buffer: activeOut } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [g0.group0Layout, group1Layout] }),
    compute: { module, entryPoint: "nvdbx_stats_dump" },
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, g0.group0);
  pass.setBindGroup(1, group1);
  pass.dispatchWorkgroups(dx, dy, dz);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const valueBytes = await readback(device, valueOut, voxelCount * 4);
  const activeBytes = await readback(device, activeOut, voxelCount * 4);
  valueOut.destroy();
  activeOut.destroy();
  destroyGroup0(g0);

  const values = new Float32Array(valueBytes);
  const actives = new Uint32Array(activeBytes);
  const histogram = new Uint32Array(histogramBins);
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let activeVoxelCount = 0;
  const span = Math.max(rangeHi - rangeLo, 1e-20);
  for (let i = 0; i < voxelCount; i++) {
    if (!actives[i]) continue;
    const v = values[i]!;
    activeVoxelCount++;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
    const norm = Math.min(Math.max((v - rangeLo) / span, 0), 1);
    const bin = Math.min(Math.floor(norm * histogramBins), histogramBins - 1);
    histogram[bin]!++;
  }
  if (activeVoxelCount === 0) {
    return { min: 0, max: 0, mean: 0, histogram, activeVoxelCount: 0, usedAtomicFastPath };
  }
  return { min, max, mean: sum / activeVoxelCount, histogram, activeVoxelCount, usedAtomicFastPath };
}

// ---------------------------------------------------------------------------
// valueTransform
// ---------------------------------------------------------------------------

export interface ValueTransformResult {
  /** The mutated grid image — a fresh `Uint32Array`, NOT the input grid's own image (see the doc below). */
  image: Uint32Array;
}

/**
 * `valueTransform(device, grid, pnanovdbSource, wgslExprOrPreset)` — in-place
 * per-voxel value edit (SPEC §3.3; Float grids only in v1 — the address math
 * assumes a 4-byte, non-quantized value stride).
 *
 * `wgslExprOrPreset` is either a key of `VALUE_TRANSFORM_PRESETS` (e.g.
 * `"double"`) or a raw WGSL function body for `fn transform(v: f32) -> f32`,
 * e.g. `"return v * 2.0;"`.
 *
 * ## Why this returns a NEW `Uint32Array`, not a mutated `grid`
 *
 * `NanoVDBGrid` is immutable by design (SPEC §3.1) — its cached
 * `storageAttribute` and any GPU state derived from it must not go stale
 * silently. This function copies `grid.image` to a device buffer, mutates
 * that copy on the GPU, and reads the result back into a brand-new
 * `Uint32Array`; the caller builds a new `NanoVDBGrid` from it (`new
 * NanoVDBGrid({ image, metadata: grid.metadata })`) if they want to render/
 * inspect the transformed grid.
 *
 * **Known limitation**: the returned image's checksum/min/max/ave/stddev
 * stats baked into the grid (`RootData`/node-level stats read by
 * `pnanovdb_*_get_min/max/ave/stddev_float`) are now STALE — this function
 * only rewrites leaf VALUE bytes, not the stats blocks. A grid built from
 * the returned image will report its ORIGINAL min/max/etc. via the stats
 * readers (and therefore also to `gridStats`'s root-range-derived histogram
 * bins/fast-path decision) even though the actual values changed. Recomputing
 * stats correctly requires walking back up the tree re-deriving each node's
 * min/max/ave/stddev from its children — not attempted here; v1 documents
 * the staleness rather than fixing it.
 *
 * Only individually-addressable LEAF voxels are touched (`level == 0u` from
 * `pnanovdb_readaccessor_get_value_address_and_level`) — a shared upper/
 * lower/root tile constant or the background value is a single value shared
 * by many voxels, not a per-voxel slot, so it is intentionally left alone.
 */
export async function valueTransform(
  device: GPUDevice,
  grid: NanoVDBGrid,
  pnanovdbSource: string,
  wgslExprOrPreset: string,
): Promise<ValueTransformResult> {
  if (grid.gridTypeId !== GRID_TYPE_FLOAT) {
    throw new Error(
      `valueTransform: grid type id ${grid.gridTypeId} is not supported — v1 only mutates FLOAT (id ` +
        `${GRID_TYPE_FLOAT}) grids (the address math assumes a 4-byte, non-quantized value stride).`,
    );
  }
  const transformBody = resolveTransformBody(wgslExprOrPreset);
  const source = buildValueTransformShaderSource(pnanovdbSource, transformBody);
  // Distinct transform bodies need distinct compiled modules (the body is
  // inlined into the shader text), so this is cached by the FULL assembled
  // source text, same cache the static entry points share.
  const module = getOrCreateModule(device, source);

  const bbox = grid.metadata.indexBBox;
  const [minX, minY, minZ] = bbox.min;
  const size = bboxSize(bbox);
  const paramsI32 = makeParamsI32({ 0: grid.gridTypeId, 1: minX, 2: minY, 3: minZ, 4: size[0], 5: size[1], 6: size[2] });
  const g0 = setUpGroup0(device, grid.image, paramsI32, makeParamsF32(0, 0));

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [g0.group0Layout] }),
    compute: { module, entryPoint: "nvdbx_value_transform" },
  });
  const [dx, dy, dz] = dispatchDims3(size);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, g0.group0);
  pass.dispatchWorkgroups(dx, dy, dz);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const bytes = await readback(device, g0.gridBuffer, grid.image.byteLength);
  destroyGroup0(g0);
  return { image: new Uint32Array(bytes) };
}

// ---------------------------------------------------------------------------
// decodeToAtlas
// ---------------------------------------------------------------------------

export type AtlasFilter = "nearest" | "trilinear";
export type AtlasFormat = "uint8" | "float32";

export interface DecodeToAtlasOptions {
  /** Per-axis output resolution cap (a literal clamp, not a proportional downscale). Default 256. */
  maxDim?: number;
  /** Sampling mode. Default `"trilinear"`. */
  filter?: AtlasFilter;
  /** Output element type. Default `"uint8"`. */
  format?: AtlasFormat;
  /** Map `[min, max] -> [0, 1]` before quantizing/storing. Default `true`. */
  normalize?: boolean;
  /** Explicit normalize range; if omitted and `normalize`, the grid's own root min/max stats are queried. */
  range?: { min: number; max: number };
}

export interface DecodeToAtlasResult {
  data: Uint8Array | Float32Array;
  dims: [number, number, number];
  /** The `[min, max]` actually used for normalization (echoed back so the caller can un-normalize later). */
  range: { min: number; max: number };
}

const DEFAULT_ATLAS_MAX_DIM = 256;

function assertValidAtlasOptions(opts: DecodeToAtlasOptions): void {
  const maxDim = opts.maxDim ?? DEFAULT_ATLAS_MAX_DIM;
  assertPositiveInt("maxDim", maxDim);
  if (opts.filter !== undefined && opts.filter !== "nearest" && opts.filter !== "trilinear") {
    throw new Error(`decodeToAtlas: \`filter\` must be "nearest" or "trilinear" (got ${JSON.stringify(opts.filter)}).`);
  }
  if (opts.format !== undefined && opts.format !== "uint8" && opts.format !== "float32") {
    throw new Error(`decodeToAtlas: \`format\` must be "uint8" or "float32" (got ${JSON.stringify(opts.format)}).`);
  }
}

/**
 * `decodeToAtlas(device, grid, pnanovdbSource, opts?)` — dense decode of the
 * grid's index bbox into a flat buffer suitable for `THREE.Data3DTexture`
 * (SPEC §3.3; demo 04's mobile/compat fallback path).
 *
 * One thread per OUTPUT voxel (dims from `computeAtlasDims`, i.e. the source
 * bbox per axis clamped to `opts.maxDim`), sampling the source at a
 * cell-centered continuous index-space position with either the readaccessor's
 * exact value (`filter: "nearest"`) or `pnanovdb_sample_trilinear_typed`
 * (`filter: "trilinear"`, the default). The GPU pass always produces raw
 * `f32`; normalize/quantize-to-`uint8` happens in JS after readback — kept
 * deliberately a buffer readback (not a `textureStore`/`GPUTexture` write),
 * so the same function is usable/testable outside a render context and the
 * caller controls exactly how the bytes reach a `Data3DTexture`. A
 * direct-to-texture variant (writing straight into a `GPUTexture` via
 * `textureStore` in a 3D-dispatch compute shader) is a documented future
 * optimization, not attempted here.
 */
export async function decodeToAtlas(
  device: GPUDevice,
  grid: NanoVDBGrid,
  pnanovdbSource: string,
  opts: DecodeToAtlasOptions = {},
): Promise<DecodeToAtlasResult> {
  assertValidAtlasOptions(opts);
  const maxDim = opts.maxDim ?? DEFAULT_ATLAS_MAX_DIM;
  const filter = opts.filter ?? "trilinear";
  const format = opts.format ?? "uint8";
  const normalize = opts.normalize ?? true;

  const source = buildComputeShaderSource(pnanovdbSource);
  const module = getOrCreateModule(device, source);

  const bbox = grid.metadata.indexBBox;
  const [minX, minY, minZ] = bbox.min;
  const regionSize = bboxSize(bbox);
  const dims = computeAtlasDims(bbox, maxDim);

  const paramsI32 = makeParamsI32({
    0: grid.gridTypeId,
    1: minX,
    2: minY,
    3: minZ,
    4: regionSize[0],
    5: regionSize[1],
    6: regionSize[2],
    7: dims[0],
    8: dims[1],
    9: dims[2],
    10: filter === "trilinear" ? 1 : 0,
  });
  const g0 = setUpGroup0(device, grid.image, paramsI32, makeParamsF32(0, 0));

  let range = opts.range;
  if (normalize && !range) {
    const [rangeLo, rangeHi] = await queryRootMinMax(device, module, g0.group0Layout, g0.group0);
    range = { min: rangeLo, max: rangeHi };
  }

  const voxelCount = dims[0] * dims[1] * dims[2];
  const group1Layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
  });
  const atlasOut = createStorageBuffer(device, voxelCount * 4);
  const group1 = device.createBindGroup({ layout: group1Layout, entries: [{ binding: 0, resource: { buffer: atlasOut } }] });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [g0.group0Layout, group1Layout] }),
    compute: { module, entryPoint: "nvdbx_atlas_decode" },
  });
  const [dx, dy, dz] = dispatchDims3(dims);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, g0.group0);
  pass.setBindGroup(1, group1);
  pass.dispatchWorkgroups(dx, dy, dz);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const bytes = await readback(device, atlasOut, voxelCount * 4);
  atlasOut.destroy();
  destroyGroup0(g0);
  const raw = new Float32Array(bytes);

  const finalRange = range ?? { min: 0, max: 1 };
  let data: Uint8Array | Float32Array;
  if (format === "uint8") {
    const out = new Uint8Array(voxelCount);
    if (normalize) {
      const span = Math.max(finalRange.max - finalRange.min, 1e-20);
      for (let i = 0; i < voxelCount; i++) {
        const norm = Math.min(Math.max((raw[i]! - finalRange.min) / span, 0), 1);
        out[i] = Math.round(norm * 255);
      }
    } else {
      for (let i = 0; i < voxelCount; i++) out[i] = Math.min(Math.max(Math.round(raw[i]!), 0), 255);
    }
    data = out;
  } else if (normalize) {
    const span = Math.max(finalRange.max - finalRange.min, 1e-20);
    const out = new Float32Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) out[i] = Math.min(Math.max((raw[i]! - finalRange.min) / span, 0), 1);
    data = out;
  } else {
    data = raw;
  }

  return { data, dims, range: finalRange };
}
