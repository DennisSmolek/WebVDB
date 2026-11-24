/**
 * `gridStats`: min, max, mean, active-voxel count and a histogram of one pool
 * grid, as compute passes recorded into the host's encoder (K2.3, ported from
 * v1 `three/compute.ts`). CPU twin: `vdbGridStatsCpu` in `@pmndrs/vdb/tools`.
 *
 * One thread per voxel of the grid's dense index bbox reads `vdbIsActive` and
 * `vdbReadValue`; inactive voxels are skipped. The histogram spans the root
 * statistics `[rootMin, rootMax]` from `pool.info(id)` (v1 read the same f32s
 * on the GPU in a separate pass first).
 *
 * Two paths, as in v1:
 * - root min ≥ 0 (fog volumes): one pass of atomics. `atomicMin/Max` on the
 *   value's bit pattern as i32 (order-preserving for non-negative floats),
 *   a count, a 64-bit fixed-point sum of `u32(norm * 2^20)` (no f32 atomics
 *   in WGSL) and histogram bins. Readback: 3 + bins words + 2.
 * - root min < 0: every voxel's (value, active) is written out and reduced on
 *   the CPU after readback (8 B per bbox voxel).
 *
 * Limitations kept from v1: dense bbox dispatch (a sparse grid with a large
 * bbox wastes threads); the range comes from the image's stats, so a grid
 * without stats (or with stale stats after a value transform) bins against
 * whatever they hold; and a negative value in a grid whose root min is ≥ 0
 * (stale stats) breaks the atomics path's min/max.
 */
import type { VDBGridPool } from "../pool/pool.js";
import { VDBError } from "../errors.js";
import {
  checkPositiveInt,
  checkTarget,
  ComputeJob,
  createInitialized,
  hostEncoder,
  kernelPipeline,
  packParams,
  USAGE,
  workgroups,
  WORKGROUP_AXIS,
  type KernelSource,
  type VDBReadback,
} from "./shared.js";
import { bboxSize, GRID_STATS_SUM_SCALE } from "../../tools/cpu/compute.js";

const W = WORKGROUP_AXIS;

const STATS_FAST: KernelSource = {
  entryPoint: "csStatsFast",
  outputs: 2,
  footer: /* wgsl */ `
@group(1) @binding(0) var<storage, read_write> csStatsI32: array<atomic<i32>>; // [0] min bits, [1] max bits
@group(1) @binding(1) var<storage, read_write> csStatsU32: array<atomic<u32>>; // [0] count, [1] sum lo, [2] sum hi, [3..] bins

const CS_SUM_SCALE: f32 = ${GRID_STATS_SUM_SCALE}.0;

@compute @workgroup_size(${W}, ${W}, ${W})
fn csStatsFast(@builtin(global_invocation_id) gid: vec3u) {
    if any(gid >= csParams.regionSize) {
        return;
    }
    let ijk = csParams.regionMin + vec3i(gid);
    var acc = vdbAccessorInit(csParams.grid);
    if !vdbIsActive(&acc, ijk) {
        return;
    }
    let value = vdbReadValue(&acc, ijk);
    // Non-negative floats keep their order as i32 bit patterns (root min >= 0 was checked on the CPU).
    let bits = bitcast<i32>(value);
    atomicMin(&csStatsI32[0], bits);
    atomicMax(&csStatsI32[1], bits);
    atomicAdd(&csStatsU32[0], 1u);
    let span = max(csParams.range.y - csParams.range.x, 1.0e-20);
    let norm = clamp((value - csParams.range.x) / span, 0.0, 1.0);
    // 64-bit sum: atomicAdd returns the old low word; carry when it wraps.
    let scaled = u32(norm * CS_SUM_SCALE);
    let oldLo = atomicAdd(&csStatsU32[1], scaled);
    if oldLo > (0xFFFFFFFFu - scaled) {
        atomicAdd(&csStatsU32[2], 1u);
    }
    let bins = csParams.bins;
    let bin = min(u32(norm * f32(bins)), bins - 1u);
    atomicAdd(&csStatsU32[3u + bin], 1u);
}
`,
};

const STATS_DUMP: KernelSource = {
  entryPoint: "csStatsDump",
  outputs: 2,
  footer: /* wgsl */ `
@group(1) @binding(0) var<storage, read_write> csDumpValue: array<f32>;
@group(1) @binding(1) var<storage, read_write> csDumpIsOn: array<u32>;

@compute @workgroup_size(${W}, ${W}, ${W})
fn csStatsDump(@builtin(global_invocation_id) gid: vec3u) {
    let size = csParams.regionSize;
    if any(gid >= size) {
        return;
    }
    let idx = gid.x + gid.y * size.x + gid.z * size.x * size.y;
    let ijk = csParams.regionMin + vec3i(gid);
    var acc = vdbAccessorInit(csParams.grid);
    let on = vdbIsActive(&acc, ijk);
    csDumpValue[idx] = vdbReadValue(&acc, ijk);
    csDumpIsOn[idx] = select(0u, 1u, on);
}
`,
};

/** Options of {@link vdbGridStatsGpu}. */
export interface VDBGridStatsOptions {
  /** Histogram bins over `[rootMin, rootMax]`. Default 256. */
  histogramBins?: number;
  /**
   * Record into this encoder. The host submits it, then calls
   * `job.submitted().read()`; `read()` before `submitted()` rejects
   * (`VDB_BAD_OPTION`). Omitted: the tool creates and submits its own encoder.
   */
  encoder?: GPUCommandEncoder;
}

/** What {@link vdbGridStatsGpu} reads back (v1's `GridStatsResult` plus the range). */
export interface VDBGridStats {
  /** Exact extremes over the active voxels (0 when there are none). */
  min: number;
  /** Exact maximum over the active voxels. */
  max: number;
  /** Mean over the active voxels: fixed-point (2^-20 of the range) on the atomics path, f64 on the dump path. */
  mean: number;
  /** `histogramBins` counts over `range`, clamped to the end bins. */
  histogram: Uint32Array;
  /** Number of active voxels counted. */
  activeVoxelCount: number;
  /** True when the atomics path ran (root min ≥ 0). */
  usedAtomicFastPath: boolean;
  /** The histogram range: the grid's root min/max statistics. */
  range: { min: number; max: number };
}

const DEFAULT_BINS = 256;
const WHERE = "gridStats";

/**
 * Records the statistics passes for grid `gridId` of `pool` and returns a
 * {@link VDBReadback}: `await gridStats(device, pool, id).read()`, or with a
 * host encoder `submit`, then `await job.submitted().read()`.
 *
 * The grid must be resident (`pool.update(encoder)` / `pool.flush()` ran
 * after its `add`), and the pool must live on `device`. The tool allocates
 * its own parameter, output and readback buffers and frees them in `read()`
 * or `dispose()`; it never touches the pool's buffer beyond binding it.
 *
 * @param device The host's `GPUDevice`; the pool must live on it.
 * @param pool The pool that holds the grid.
 * @param gridId The grid's pool id; it must be resident.
 * @param options Histogram bins and an optional host encoder ({@link VDBGridStatsOptions}).
 *
 * @throws VDBError `VDB_BAD_OPTION` for bad options, a pool on another
 *   device, a non-resident grid, or (root min < 0, the dump path) a bbox whose
 *   per-voxel outputs (`voxels * 4` bytes each, `voxels * 8` to read back)
 *   exceed `device.limits.maxStorageBufferBindingSize` / `maxBufferSize`.
 */
export function vdbGridStatsGpu(device: GPUDevice, pool: VDBGridPool, gridId: number, options: VDBGridStatsOptions = {}): VDBReadback<VDBGridStats> {
  const bins = options.histogramBins ?? DEFAULT_BINS;
  checkPositiveInt(WHERE, "histogramBins", bins);
  const info = checkTarget(WHERE, device, pool, gridId);
  const range = { min: info.rootMin, max: info.rootMax };
  const regionMin = info.indexBBox.min;
  const size = bboxSize(info.indexBBox);
  const fast = range.min >= 0;
  if (!fast) checkDumpSize(device, gridId, size);
  const kernel = fast ? STATS_FAST : STATS_DUMP;
  const cached = kernelPipeline(device, kernel, info.type);

  const params = createInitialized(device, "vdb gridStats params", USAGE.UNIFORM, new Uint8Array(packParams({ grid: gridId, bins, regionMin, regionSize: size, range })));
  let outA: GPUBuffer;
  let outB: GPUBuffer;
  const outUsage = USAGE.STORAGE | USAGE.COPY_SRC;
  if (fast) {
    // min starts at i32 max (above every non-negative float's bits), max at +0.0.
    outA = createInitialized(device, "vdb gridStats min/max", outUsage, new Int32Array([0x7fffffff, 0]));
    outB = createInitialized(device, "vdb gridStats sums", outUsage, new Uint32Array(3 + bins));
  } else {
    const voxels = size[0] * size[1] * size[2];
    outA = device.createBuffer({ label: "vdb gridStats values", size: voxels * 4, usage: outUsage });
    outB = device.createBuffer({ label: "vdb gridStats active", size: voxels * 4, usage: outUsage });
  }
  const staging = device.createBuffer({ label: "vdb gridStats readback", size: outA.size + outB.size, usage: USAGE.MAP_READ | USAGE.COPY_DST });

  const group0 = device.createBindGroup({ layout: cached.group0, entries: [pool.bindGroupEntry({ group: 0, firstBinding: 0 }), { binding: 1, resource: { buffer: params } }] });
  const group1 = device.createBindGroup({ layout: cached.group1, entries: [{ binding: 0, resource: { buffer: outA } }, { binding: 1, resource: { buffer: outB } }] });

  const host = hostEncoder(device, options.encoder, "vdb gridStats");
  const pass = host.encoder.beginComputePass({ label: `vdb gridStats (${fast ? "atomics" : "dump"})` });
  pass.setPipeline(cached.pipeline);
  pass.setBindGroup(0, group0);
  pass.setBindGroup(1, group1);
  pass.dispatchWorkgroups(...workgroups(size));
  pass.end();
  host.encoder.copyBufferToBuffer(outA, 0, staging, 0, outA.size);
  host.encoder.copyBufferToBuffer(outB, 0, staging, outA.size, outB.size);
  host.submit();

  const splitAt = outA.size;
  return new ComputeJob(
    WHERE,
    [params, outA, outB],
    staging,
    cached.error,
    (bytes) => (fast ? reduceFast(bytes, bins, range) : reduceDump(bytes, splitAt, bins, range)),
    host.selfSubmitted,
  );
}

/**
 * The dump path writes 4 B of value and 4 B of active flag per bbox voxel and
 * reads both back (8 B per voxel); refuse a bbox the device can't hold before
 * any buffer is made.
 */
function checkDumpSize(device: GPUDevice, gridId: number, size: readonly [number, number, number]): void {
  const voxels = size[0] * size[1] * size[2];
  const each = voxels * 4;
  const limits = device.limits as Partial<GPUSupportedLimits> | undefined;
  const binding = limits?.maxStorageBufferBindingSize ?? Infinity;
  const total = limits?.maxBufferSize ?? Infinity;
  if (each > binding || each * 2 > total) {
    throw new VDBError(
      "VDB_BAD_OPTION",
      `${WHERE}: grid ${gridId} has a negative root min, so its stats take the dump path (8 B per bbox voxel): ` +
        `${size.join(" x ")} = ${voxels} voxels needs ${each} B per output binding and ${each * 2} B of readback, over this device's ` +
        `maxStorageBufferBindingSize ${binding} / maxBufferSize ${total}. Raise the limits (VDBGridPool.requiredLimits(adapter)) ` +
        "or compute the stats on the CPU (vdbGridStatsCpu in @pmndrs/vdb/tools).",
    );
  }
}

function reduceFast(bytes: ArrayBuffer, bins: number, range: { min: number; max: number }): VDBGridStats {
  const u32 = new Uint32Array(bytes, 8, 3 + bins);
  const count = u32[0]!;
  const histogram = u32.slice(3, 3 + bins);
  if (count === 0) return { min: 0, max: 0, mean: 0, histogram, activeVoxelCount: 0, usedAtomicFastPath: true, range };
  const minMax = new Float32Array(bytes, 0, 2);
  const total = BigInt(u32[2]!) * 4294967296n + BigInt(u32[1]!);
  const meanNorm = Number(total) / (GRID_STATS_SUM_SCALE * count);
  return {
    min: minMax[0]!,
    max: minMax[1]!,
    mean: range.min + meanNorm * (range.max - range.min),
    histogram,
    activeVoxelCount: count,
    usedAtomicFastPath: true,
    range,
  };
}

function reduceDump(bytes: ArrayBuffer, splitAt: number, bins: number, range: { min: number; max: number }): VDBGridStats {
  const values = new Float32Array(bytes, 0, splitAt / 4);
  const on = new Uint32Array(bytes, splitAt, splitAt / 4);
  const histogram = new Uint32Array(bins);
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  const span = Math.max(range.max - range.min, 1e-20);
  for (let i = 0; i < values.length; i++) {
    if (!on[i]) continue;
    const v = values[i]!;
    count++;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
    const norm = Math.min(Math.max((v - range.min) / span, 0), 1);
    histogram[Math.min(Math.floor(norm * bins), bins - 1)]!++;
  }
  if (count === 0) return { min: 0, max: 0, mean: 0, histogram, activeVoxelCount: 0, usedAtomicFastPath: false, range };
  return { min, max, mean: sum / count, histogram, activeVoxelCount: count, usedAtomicFastPath: false, range };
}
