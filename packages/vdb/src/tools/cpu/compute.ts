/**
 * CPU twins (D11) of the core compute tools (K2.3, `core/compute/`):
 * `vdbGridStatsCpu` (twin of `gridStats`), `vdbDecodeToAtlasCpu` (twin of
 * `decodeToAtlas`) and `transformLeafValues`, the CPU value transform that
 * `vdbApplyValueTransform` writes back with `pool.replace` (it is also the twin of
 * v1's GPU `valueTransform`, which the pool's read-only binding retired).
 *
 * The shared pure pieces live here so the core and its twins cannot drift:
 * `atlasDims` (output resolution), `atlasFromRaw` (normalize / quantize after
 * readback) and `rootStats` (the range both tools use).
 *
 * Float arithmetic: each WGSL f32 operation is rounded with `Math.fround` in
 * the WGSL's order. WGSL allows a 2.5 ULP division and fused multiply-adds,
 * so GPU parity compares with a tolerance and reports the bit-exact count.
 */
import { VDBToolsError } from "../errors.js";
import { indexBBox, isActive, vdbReadValueF32, treeNodes, type Vec3 } from "./march.js";
import { defineNumber, gridTypeConstantsFor, gridTypeConstantsForId, gridTypeIds } from "./stride-tables.js";

const f = Math.fround;

const GRID_OFF_GRID_TYPE = defineNumber("PNANOVDB_GRID_OFF_GRID_TYPE");
const GRID_SIZE = defineNumber("PNANOVDB_GRID_SIZE");
const TREE_OFF_NODE_OFFSET_ROOT = defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT");

/** Fixed-point scale of the GPU fast path's 64-bit sum (2^20). */
export const GRID_STATS_SUM_SCALE = 1048576;

/** Index bbox, inclusive on both ends. */
export interface IndexBBox {
  min: readonly [number, number, number];
  max: readonly [number, number, number];
}

/** Per-axis voxel count of an inclusive-max index bbox (`max - min + 1`). */
export function bboxSize(bbox: IndexBBox): [number, number, number] {
  return [bbox.max[0] - bbox.min[0] + 1, bbox.max[1] - bbox.min[1] + 1, bbox.max[2] - bbox.min[2] + 1];
}

/**
 * `decodeToAtlas` output resolution: each axis of the index bbox clamped to
 * `maxDim` (a literal per-axis clamp, not a proportional downscale; v1).
 */
export function atlasDims(bbox: IndexBBox, maxDim: number): [number, number, number] {
  const [sx, sy, sz] = bboxSize(bbox);
  return [Math.min(sx, maxDim), Math.min(sy, maxDim), Math.min(sz, maxDim)];
}

/** The root node's stored min/max statistics (f32), meaningful when the grid has stats. */
export function rootStats(gridWords: Uint32Array): { min: number; max: number } {
  const gt = gridTypeConstantsForId(gridWords[GRID_OFF_GRID_TYPE >>> 2]!);
  const root = GRID_SIZE + gridWords[(GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT) >>> 2]!;
  const f32 = new Float32Array(gridWords.buffer, gridWords.byteOffset, gridWords.length);
  return { min: f32[(root + gt.root_off_min) >>> 2]!, max: f32[(root + gt.root_off_max) >>> 2]! };
}

// ---------------------------------------------------------------------------
// gridStats

/** Options of {@link vdbGridStatsCpu}. */
export interface VDBGridStatsCpuOptions {
  /** Histogram bins (default 256). */
  histogramBins?: number;
}

/** Result of {@link vdbGridStatsCpu}, the twin of the GPU `gridStats` result. */
export interface VDBGridStatsCpuResult {
  /** Smallest active value. */
  min: number;
  /** Largest active value. */
  max: number;
  /** Mean of the active values. */
  mean: number;
  /** Counts of active voxels per bin over {@link range}. */
  histogram: Uint32Array;
  /** Number of active voxels in the index bbox. */
  activeVoxelCount: number;
  /** The path the GPU takes for this grid: `true` when the root min is ≥ 0 (atomics). */
  usedAtomicFastPath: boolean;
  /** The histogram range: the root min/max statistics. */
  range: { min: number; max: number };
}

/**
 * Twin of `gridStats`: min, max, mean and a histogram over the ACTIVE voxels of
 * the dense index bbox, read as the GPU reads them (`vdbIsActive`,
 * `vdbReadValue`). The histogram spans the root statistics `[rootMin,
 * rootMax]`.
 *
 * Both GPU paths are reproduced: the atomics path (root min ≥ 0) normalizes in
 * f32 and sums `u32(norm * 2^20)` in a 64-bit integer; the dump path (root min
 * < 0) reduces f32 values in f64 on the CPU. The f32 normalization divides, so
 * a GPU with an inexact division may put a voxel in the neighbouring bin or
 * shift the mean by a fixed-point step.
 *
 * @param gridWords The grid image (little-endian u32 words, `GridData` at word 0).
 * @param options Histogram bin count.
 */
export function vdbGridStatsCpu(gridWords: Uint32Array, options: VDBGridStatsCpuOptions = {}): VDBGridStatsCpuResult {
  const bins = options.histogramBins ?? 256;
  const range = rootStats(gridWords);
  const fast = range.min >= 0;
  const { min: lo, max: hi } = indexBBox(gridWords);
  const histogram = new Uint32Array(bins);
  let min = Infinity;
  let max = -Infinity;
  let count = 0;
  let sumFixed = 0n;
  let sum = 0;
  const spanF = f(Math.max(f(range.max - range.min), f(1e-20)));
  const spanD = Math.max(range.max - range.min, 1e-20);
  const ijk: Vec3 = [0, 0, 0];
  for (let z = lo[2]; z <= hi[2]; z++) {
    for (let y = lo[1]; y <= hi[1]; y++) {
      for (let x = lo[0]; x <= hi[0]; x++) {
        ijk[0] = x;
        ijk[1] = y;
        ijk[2] = z;
        if (!isActive(gridWords, ijk)) continue;
        const v = vdbReadValueF32(gridWords, ijk);
        count++;
        if (v < min) min = v;
        if (v > max) max = v;
        let norm: number;
        if (fast) {
          norm = Math.min(Math.max(f(f(v - range.min) / spanF), 0), 1);
          sumFixed += BigInt(Math.trunc(f(norm * GRID_STATS_SUM_SCALE)));
          histogram[Math.min(Math.trunc(f(norm * bins)), bins - 1)]!++;
        } else {
          sum += v;
          norm = Math.min(Math.max((v - range.min) / spanD, 0), 1);
          histogram[Math.min(Math.floor(norm * bins), bins - 1)]!++;
        }
      }
    }
  }
  if (count === 0) return { min: 0, max: 0, mean: 0, histogram, activeVoxelCount: 0, usedAtomicFastPath: fast, range };
  // The GPU's 64-bit sum wraps like a u64.
  sumFixed &= (1n << 64n) - 1n;
  const mean = fast
    ? range.min + (Number(sumFixed) / (GRID_STATS_SUM_SCALE * count)) * (range.max - range.min)
    : sum / count;
  return { min, max, mean, histogram, activeVoxelCount: count, usedAtomicFastPath: fast, range };
}

// ---------------------------------------------------------------------------
// decodeToAtlas

/** How an atlas samples the grid: the nearest voxel or trilinear interpolation. */
export type VDBAtlasFilter = "nearest" | "trilinear";
/** Texel format of a decoded atlas: 8-bit unsigned or 32-bit float. */
export type VDBAtlasFormat = "uint8" | "float32";

export interface AtlasFromRawOptions {
  format: VDBAtlasFormat;
  normalize: boolean;
  range: { min: number; max: number };
}

/**
 * Post-readback step of `decodeToAtlas` (v1, in f64): `uint8` maps `[min,
 * max]` to `[0, 255]` with `Math.round` (or clamps the raw value when not
 * normalizing); `float32` maps to `[0, 1]` or returns `raw` itself.
 */
export function atlasFromRaw(raw: Float32Array, o: AtlasFromRawOptions): Uint8Array | Float32Array {
  const n = raw.length;
  const span = Math.max(o.range.max - o.range.min, 1e-20);
  if (o.format === "uint8") {
    const out = new Uint8Array(n);
    if (o.normalize) {
      for (let i = 0; i < n; i++) out[i] = Math.round(Math.min(Math.max((raw[i]! - o.range.min) / span, 0), 1) * 255);
    } else {
      for (let i = 0; i < n; i++) out[i] = Math.min(Math.max(Math.round(raw[i]!), 0), 255);
    }
    return out;
  }
  if (!o.normalize) return raw;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.min(Math.max((raw[i]! - o.range.min) / span, 0), 1);
  return out;
}

/** Options of {@link vdbDecodeToAtlasCpu}. */
export interface VDBDecodeToAtlasCpuOptions {
  /** Longest atlas side in texels (default 256). */
  maxDim?: number;
  /** Sampling filter (default `'trilinear'`). */
  filter?: VDBAtlasFilter;
}

/**
 * Twin of the `decodeToAtlas` pass before normalization: one f32 per output
 * voxel (x fastest), sampled at index position `min + (i + 0.5) * size / dims
 * - 0.5` (cell `i` of `dims` over `[min, max]`, `size = max - min + 1`; voxel
 * centres at full resolution, K2.6) with `vdbReadValue(floor(p + 0.5))` (`'nearest'`) or
 * the vendored trilinear (`'trilinear'`, the default). Feed the result to
 * {@link atlasFromRaw} for the formatted atlas.
 *
 * @param gridWords The grid image (little-endian u32 words, `GridData` at word 0).
 * @param options Longest atlas side and filter.
 */
export function vdbDecodeToAtlasCpu(
  gridWords: Uint32Array,
  options: VDBDecodeToAtlasCpuOptions = {},
): { raw: Float32Array; dims: [number, number, number] } {
  const bbox = indexBBox(gridWords);
  const size = bboxSize(bbox);
  const dims = atlasDims(bbox, options.maxDim ?? 256);
  const trilinear = (options.filter ?? "trilinear") === "trilinear";
  const raw = new Float32Array(dims[0] * dims[1] * dims[2]);
  const pos: Vec3 = [0, 0, 0];
  let i = 0;
  for (let z = 0; z < dims[2]; z++) {
    for (let y = 0; y < dims[1]; y++) {
      for (let x = 0; x < dims[0]; x++) {
        const g: Vec3 = [x, y, z];
        for (let a = 0; a < 3; a++) pos[a] = f(bbox.min[a]! + f(f(f(f(g[a]! + 0.5) * size[a]!) / dims[a]!) - 0.5));
        raw[i++] = trilinear ? trilinearF32(gridWords, pos) : vdbReadValueF32(gridWords, [Math.floor(f(pos[0] + 0.5)), Math.floor(f(pos[1] + 0.5)), Math.floor(f(pos[2] + 0.5))]);
      }
    }
  }
  return { raw, dims };
}

/** The vendored `sample_trilinear_typed` in f32: `mix(a, b, t)` as `a + t * (b - a)`. */
function trilinearF32(gridWords: Uint32Array, xyz: Vec3): number {
  const b: Vec3 = [Math.floor(xyz[0]), Math.floor(xyz[1]), Math.floor(xyz[2])];
  const u = [f(xyz[0] - b[0]), f(xyz[1] - b[1]), f(xyz[2] - b[2])];
  const tap = (dx: number, dy: number, dz: number) => vdbReadValueF32(gridWords, [b[0] + dx, b[1] + dy, b[2] + dz]);
  const mix = (a: number, c: number, t: number) => f(a + f(t * f(c - a)));
  const x00 = mix(tap(0, 0, 0), tap(1, 0, 0), u[0]!);
  const x10 = mix(tap(0, 1, 0), tap(1, 1, 0), u[0]!);
  const x01 = mix(tap(0, 0, 1), tap(1, 0, 1), u[0]!);
  const x11 = mix(tap(0, 1, 1), tap(1, 1, 1), u[0]!);
  return mix(mix(x00, x10, u[1]!), mix(x01, x11, u[1]!), u[2]!);
}

// ---------------------------------------------------------------------------
// Value transform

/**
 * Returns a copy of a Float grid image with `fn` applied to every leaf voxel
 * value inside the index bbox (active or inactive), rounded to f32. Tile,
 * background and out-of-bbox leaf values are left alone, exactly as v1's GPU
 * `valueTransform` did (it visited the dense bbox and rewrote level-0
 * addresses only).
 *
 * The image's statistics (root/node min, max, average, deviation) are NOT
 * recomputed: they describe the source values. Tools that trust them
 * (`gridStats`' range, `vdbNodeMax` skipping) see stale bounds; rebuild the
 * grid (e.g. `vdbBuildFromDense`) when they matter.
 *
 * Float (GridType 1) only: quantized leaves would need re-encoding.
 */
export function transformLeafValues(gridWords: Uint32Array, fn: (value: number) => number): Uint32Array {
  if (typeof fn !== "function") throw new VDBToolsError("VDB_BAD_OPTION", "transformLeafValues: fn must be a function (value: number) => number");
  const gridType = gridWords[GRID_OFF_GRID_TYPE >>> 2];
  if (gridType !== gridTypeIds.FLOAT) {
    throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", 
      `transformLeafValues: grid type ${gridType} is not Float (${gridTypeIds.FLOAT}); quantized leaves would need re-encoding. ` +
        "Transform the Float source and quantize() afterwards.",
    );
  }
  const out = gridWords.slice();
  const outF32 = new Float32Array(out.buffer, out.byteOffset, out.length);
  const table = gridTypeConstantsFor("FLOAT").leaf_off_table;
  const { min, max } = indexBBox(gridWords);
  for (const node of treeNodes(gridWords)) {
    if (node.level !== 0) continue;
    const [ox, oy, oz] = node.origin;
    for (let n = 0; n < 512; n++) {
      const x = ox + (n >> 6);
      const y = oy + ((n >> 3) & 7);
      const z = oz + (n & 7);
      if (x < min[0] || y < min[1] || z < min[2] || x > max[0] || y > max[1] || z > max[2]) continue;
      const word = (node.address + table + 4 * n) >>> 2;
      outF32[word] = f(fn(outF32[word]!));
    }
  }
  return out;
}
