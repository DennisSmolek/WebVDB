// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameMipmaps.js L14–106 (level sizes, 2x2x2 box downsample with
//   clamped edge taps, the mip atlas layout), as float data.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: f32 data without the 8-bit rounding; the first level comes from a sparse grid through
// `vdbResampleGrid` (box); node-average sampling and the HDDA skip are ours (K3.2). Divergences:
// DIVERGES.md (K3.2 rows).

/**
 * CPU side of the sparse volume source's LOD strategies (K3.2, R1 §4.3) and the CPU twins of its
 * WGSL (`@pmndrs/vdb/three` `sparse-source.ts`):
 *
 * - `'coarse-pyramid'`: {@link vdbBoxPyramid} builds the dense coarse levels his
 *   `sampleExtinction(localPos, lod)` reads at lod ≥ 1. The base level L₀ = max(1, ⌈log2(maxTexels /
 *   pyramidMax)⌉) is a box resample of the sparse grid (`vdbResampleGrid`, filter `'box'`) to
 *   ⌈texels / 2^L₀⌉ voxels over the exact index box, so texel j is centred at uvw (j + 0.5) / dims as
 *   his mip levels are; the levels above it are his 2x2x2 box mip atlas, in float
 *   ({@link vdbBoxMipAtlasF32}).
 * - `'node-average'`: {@link vdbFrameNodeAverageSample} is the twin of the trilinear interpolation
 *   of `vdbNodeAverage` over the leaf (dim 8) or lower-node (dim 128) lattice.
 * - HDDA skip: {@link vdbFrameSkipEmpty} is the twin of the shadow march's empty-space query.
 *
 * Engine-free and core-free (D20): reads the NanoVDB image with the `tools/cpu` twins.
 */

import { VDBToolsError } from "../errors.js";
import { cellBoundTrilinear, hddaCellEnd, hddaInit, hddaStep, indexBBox, leafVoxels, nodeAverage, nodeMax, treeNodes, vdbReadValueF32, type Vec3 } from "../cpu/march.js";
import { halfToFloat } from "../vdb/half-float.js";
import { vdbResampleGrid } from "../nanovdb/resample.js";
import { GRID_OFF_MAP, MAP_OFF_MATD, MAP_OFF_VECD } from "../nanovdb/bytes.js";
import { vdbBoxPyramidLayout, vdbPyramidBaseLevel, vdbPyramidLevelSize as levelSize } from "./pyramid-layout.js";

export { vdbPyramidBaseLevel };

const f = Math.fround;

/** A float mip atlas in his layout: levels 1..n side by side on x. */
export interface VDBFloatMipAtlas {
  readonly data: Float32Array;
  readonly dimensions: [number, number, number];
  /** Levels in the atlas (above the base). */
  readonly levels: number;
}

/**
 * His box mip atlas (frameMipmaps.js `buildAtlas` / `packAtlas` / `downsample`) over float data:
 * levels 1..n with n = ⌈log2(max side)⌉, each the 2x2x2 mean of the level below with edge taps
 * clamped; level l at x = atlasWidth − atlasWidth / 2^(l−1). No 8-bit rounding. Null for a 1³ base.
 */
export function vdbBoxMipAtlasF32(data: Float32Array, dimensions: readonly number[]): VDBFloatMipAtlas | null {
  const [width, height, depth] = dimensions as Vec3;
  const count = Math.ceil(Math.log2(Math.max(width, height, depth, 1)));
  if (count < 1) return null;
  const levels: { data: Float32Array; width: number; height: number; depth: number }[] = [];
  let level = { data, width, height, depth };
  for (let index = 1; index <= count; index++) {
    const tw = levelSize(level.width, 1);
    const th = levelSize(level.height, 1);
    const td = levelSize(level.depth, 1);
    const out = new Float32Array(tw * th * td);
    const src = level;
    for (let z = 0; z < td; z++) {
      const z0 = Math.min(z * 2, src.depth - 1);
      const z1 = Math.min(z * 2 + 1, src.depth - 1);
      for (let y = 0; y < th; y++) {
        const y0 = Math.min(y * 2, src.height - 1);
        const y1 = Math.min(y * 2 + 1, src.height - 1);
        const r00 = (y0 + z0 * src.height) * src.width;
        const r10 = (y1 + z0 * src.height) * src.width;
        const r01 = (y0 + z1 * src.height) * src.width;
        const r11 = (y1 + z1 * src.height) * src.width;
        for (let x = 0; x < tw; x++) {
          const x0 = Math.min(x * 2, src.width - 1);
          const x1 = Math.min(x * 2 + 1, src.width - 1);
          const d = src.data;
          out[x + (y + z * th) * tw] =
            (d[x0 + r00]! + d[x1 + r00]! + d[x0 + r10]! + d[x1 + r10]! + d[x0 + r01]! + d[x1 + r01]! + d[x0 + r11]! + d[x1 + r11]!) / 8;
        }
      }
    }
    level = { data: out, width: tw, height: th, depth: td };
    levels.push(level);
  }
  const atlasWidth = 2 ** Math.max(Math.ceil(Math.log2(width)), levels.length);
  const atlasHeight = levelSize(height, 1);
  const atlasDepth = levelSize(depth, 1);
  const atlas = new Float32Array(atlasWidth * atlasHeight * atlasDepth);
  levels.forEach((l, levelIndex) => {
    const offset = atlasWidth - atlasWidth / 2 ** levelIndex;
    for (let z = 0; z < l.depth; z++) {
      for (let y = 0; y < l.height; y++) {
        const row = (y + z * atlasHeight) * atlasWidth + offset;
        atlas.set(l.data.subarray((y + z * l.height) * l.width, (y + z * l.height + 1) * l.width), row);
      }
    }
  });
  return { data: atlas, dimensions: [atlasWidth, atlasHeight, atlasDepth], levels: levels.length };
}

/** Options of {@link vdbBoxPyramid}. */
export interface VDBBoxPyramidOptions {
  /** Inclusive voxel index box the volume's unit box covers. Default: the grid's index bbox. */
  indexBox?: { min: Vec3; max: Vec3 };
  /** Texels on the longest side of the base level at most (R1: `pyramidMax`, default 128). */
  pyramidMax?: number;
  /** Base level override (default {@link vdbPyramidBaseLevel}). */
  level?: number;
}

/** The dense coarse levels of a sparse grid (see the file comment). */
export interface VDBBoxPyramid {
  /** L₀: the level of `data` in level-0 voxels (texel footprint 2^L₀). */
  readonly level: number;
  /** Texels of the base level, ⌈indexBox size / 2^L₀⌉. */
  readonly dimensions: [number, number, number];
  /** Base level, x fastest, the box mean of the sparse grid over each texel's cell. */
  readonly data: Float32Array;
  /** Levels L₀+1… in his atlas layout (null when the base is 1³). */
  readonly atlas: VDBFloatMipAtlas | null;
  /** Total levels the source then has: L₀ + atlas levels. */
  readonly lodLevels: number;
}

/**
 * The coarse pyramid of a sparse FLOAT/Fp grid image (CPU, `vdbResampleGrid` box). The grid's Map must
 * be axis-aligned (vdbResampleGrid's rule).
 */
export function vdbBoxPyramid(image: Uint32Array, o: VDBBoxPyramidOptions = {}): VDBBoxPyramid {
  const box = o.indexBox ?? indexBBox(image);
  const texels: Vec3 = [box.max[0] - box.min[0] + 1, box.max[1] - box.min[1] + 1, box.max[2] - box.min[2] + 1];
  if (texels.some((t) => !(t > 0))) throw new VDBToolsError("VDB_BAD_OPTION", "vdbBoxPyramid: empty index box");
  const layout = vdbBoxPyramidLayout(texels, { ...(o.pyramidMax !== undefined ? { pyramidMax: o.pyramidMax } : {}), ...(o.level !== undefined ? { level: o.level } : {}) });
  const level = layout.level;
  const dims: Vec3 = [...layout.dimensions];
  // The index cell box [min − ½, max + ½] in world (NanoVDB: world = matd · ijk + vecd).
  const view = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const mat = Array.from({ length: 9 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_MATD + i * 8, true));
  const vec = Array.from({ length: 3 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_VECD + i * 8, true));
  const world = (ijk: Vec3): Vec3 => [0, 1, 2].map((r) => mat[r * 3]! * ijk[0] + mat[r * 3 + 1]! * ijk[1] + mat[r * 3 + 2]! * ijk[2] + vec[r]!) as Vec3;
  const lo = world([box.min[0] - 0.5, box.min[1] - 0.5, box.min[2] - 0.5]);
  const hi = world([box.max[0] + 0.5, box.max[1] + 0.5, box.max[2] + 0.5]);
  const coarse = vdbResampleGrid(image, { filter: "box", dimensions: dims, bbox: [lo, hi] });
  // Dense readout of the resampled grid (index box [0, dims − 1]); voxels outside its leaves and tiles are background.
  const data = new Float32Array(dims[0] * dims[1] * dims[2]);
  const background = 0;
  if (background !== 0) data.fill(background);
  const put = (x: number, y: number, z: number, v: number) => {
    if (x < 0 || y < 0 || z < 0 || x >= dims[0] || y >= dims[1] || z >= dims[2]) return;
    data[x + (y + z * dims[1]) * dims[0]] = v;
  };
  for (const node of treeNodes(coarse.image)) {
    for (const t of node.activeTiles) {
      for (let z = 0; z < t.dim; z++) for (let y = 0; y < t.dim; y++) for (let x = 0; x < t.dim; x++) put(t.origin[0] + x, t.origin[1] + y, t.origin[2] + z, t.value);
    }
    if (node.level !== 0) continue;
    const { values } = leafVoxels(coarse.image, node.address);
    const [ox, oy, oz] = node.origin;
    for (let n = 0; n < 512; n++) put(ox + (n >> 6), oy + ((n >> 3) & 7), oz + (n & 7), values[n]!);
  }
  const atlas = vdbBoxMipAtlasF32(data, dims);
  return { level, dimensions: dims, data, atlas, lodLevels: level + (atlas?.levels ?? 0) };
}

/**
 * Twin of the sparse source's node-average tap (`vdbFrameNodeAverage` in `three/sparse-source.ts`):
 * trilinear interpolation of {@link nodeAverage} over the lattice of aligned `dim`-cells, whose
 * centres are at `dim·j + dim/2 − ½` in index space (voxel centres at integers). `p` is in index
 * space. f32 arithmetic in WGSL order.
 */
export function vdbFrameNodeAverageSample(image: Uint32Array, p: Vec3, dim: number): number {
  const fd = f(dim);
  const half = f(f(0.5 * fd) - 0.5);
  const q = p.map((v) => f(f(v - half) / fd)) as Vec3;
  const base = q.map(Math.floor) as Vec3;
  const w = q.map((v, i) => f(v - base[i]!)) as Vec3;
  const c: Vec3 = [base[0] * dim, base[1] * dim, base[2] * dim];
  const v: number[] = [];
  for (let k = 0; k < 8; k++) v.push(nodeAverage(image, [c[0] + (k & 1) * dim, c[1] + ((k >> 1) & 1) * dim, c[2] + ((k >> 2) & 1) * dim], dim));
  const mix = (a: number, b: number, t: number) => f(a + f(t * f(b - a)));
  const x00 = mix(v[0]!, v[1]!, w[0]);
  const x10 = mix(v[2]!, v[3]!, w[0]);
  const x01 = mix(v[4]!, v[5]!, w[0]);
  const x11 = mix(v[6]!, v[7]!, w[0]);
  return mix(mix(x00, x10, w[1]), mix(x01, x11, w[1]), w[2]);
}

/**
 * Twin of the shadow march's empty-space query (`vdbFrameSkipEmpty`, K3.2): an HDDA over 8³ cells
 * of the ray `o + d·t` (sample space: index space, the trilinear base voxel is `floor(p)`) from `t0`
 * to `t1`. Returns `[ts, te]`: every sample with `t < ts` reads only voxels whose bound is ≤ `cutoff`
 * (so it is skipped exactly), and the cell from `ts` to `te` must be sampled; `[t1, t1]` when the
 * rest is empty. The bound per cell is `cellBoundTrilinear` (the 2x2x2 block a level-0 trilinear tap
 * reads) or, with `dilate`, the max of `nodeMax` over the 3x3x3 cells around (taps of the coarse
 * levels that reach at most 8 voxels past the cell). After `maxCells` empty cells it returns
 * `[ts, ts]` (the caller asks again from there). `ts` steps back by `1e-3` voxel along the ray so a
 * sample on a cell face is never skipped by rounding.
 */
export function vdbFrameSkipEmpty(image: Uint32Array, o: Vec3, d: Vec3, t0: number, t1: number, cutoff: number, dilate: boolean, maxCells = 64): [number, number] {
  const h = hddaInit(o, t0, d, t1, 8);
  const back = f(1e-3 / Math.max(Math.hypot(d[0], d[1], d[2]), 1e-6));
  let ts = t0;
  for (let n = 0; n < maxCells; n++) {
    const te = hddaCellEnd(h);
    let b: number;
    if (!dilate) b = cellBoundTrilinear(image, h.voxel, 8);
    else {
      b = -Infinity;
      for (let k = 0; k < 27; k++) {
        const c: Vec3 = [h.voxel[0] + ((k % 3) - 1) * 8, h.voxel[1] + ((Math.floor(k / 3) % 3) - 1) * 8, h.voxel[2] + (Math.floor(k / 9) - 1) * 8];
        b = Math.max(b, nodeMax(image, c, 8));
      }
    }
    if (b > cutoff) return [Math.max(t0, f(ts - back)), te];
    ts = te;
    if (!hddaStep(h)) return [t1, t1];
  }
  return [Math.max(t0, f(ts - back)), Math.max(t0, f(ts - back))];
}

// ---------------------------------------------------------------------------------------------------------------
// K6.7: twins of the core sparse source (`core/source/sparse.ts`, `vdbSparseSourceGpu`): its `vdbSrc*` WGSL on one
// volume, the density stats it computes and the f16 pyramid it samples.
// ---------------------------------------------------------------------------------------------------------------

/** The non-zero density stats of an image (every leaf voxel, active or not, and active tiles); the source's `stats` rule (K3.2). */
export function vdbSparseDensityStats(image: Uint32Array, clamp01 = false): { meanNonZero: number; max: number; activeVoxels: number } {
  let sum = 0;
  let count = 0;
  let maxValue = 0;
  const add = (v: number, n: number) => {
    const x = clamp01 ? Math.min(Math.max(v, 0), 1) : v;
    if (x !== 0) {
      sum += x * n;
      count += n;
      maxValue = Math.max(maxValue, x);
    }
  };
  for (const node of treeNodes(image)) {
    for (const t of node.activeTiles) add(t.value, t.dim ** 3);
    if (node.level !== 0) continue;
    const { values } = leafVoxels(image, node.address);
    for (let n = 0; n < 512; n++) add(values[n]!, 1);
  }
  return { meanNonZero: count > 0 ? sum / count : 0, max: maxValue, activeVoxels: count };
}

/** f32 → binary16 bits as the GPU pyramid packs them (three's `DataUtils.toHalfFloat`: truncation; `pyrHalf`). */
export function vdbPyramidHalfBits(v: number): number {
  const buf = new DataView(new ArrayBuffer(4));
  buf.setFloat32(0, Math.min(Math.max(v, -65504), 65504), true);
  const bits = buf.getUint32(0, true);
  const e = ((bits >>> 23) & 0xff) - 127;
  let base: number;
  let shift: number;
  if (e < -27) [base, shift] = [0, 24];
  else if (e < -14) [base, shift] = [0x400 >>> (-e - 14), -e - 1];
  else if (e <= 15) [base, shift] = [(e + 15) << 10, 13];
  else if (e < 128) [base, shift] = [0x7c00, 24];
  else [base, shift] = [0x7c00, 13];
  if (bits & 0x80000000) base |= 0x8000;
  return base + ((bits & 0x7fffff) >>> shift);
}

/** A pyramid as the GPU holds it: every texel through binary16 ({@link vdbPyramidHalfBits}). */
export function vdbPyramidToHalf(p: VDBBoxPyramid): VDBBoxPyramid {
  const round = (data: Float32Array) => Float32Array.from(data, (v) => halfToFloat(vdbPyramidHalfBits(v)));
  return { ...p, data: round(p.data), atlas: p.atlas ? { ...p.atlas, data: round(p.atlas.data) } : null };
}

/** One channel of a {@link VDBSparseVolumeCpu}: a grid image and the inclusive index box the unit box covers. */
export interface VDBSparseChannelCpu {
  /** The grid image. */
  readonly image: Uint32Array;
  /** The inclusive voxel index box uvw [0, 1]³ maps onto (cell box `[min − ½, max + ½]`). */
  readonly box: { readonly min: Vec3; readonly max: Vec3 };
}

/** One volume of the core sparse source, on the CPU (the twin's input). */
export interface VDBSparseVolumeCpu {
  /** The source's LOD strategy. */
  readonly lod: "none" | "coarse-pyramid" | "node-average";
  /** The source's `normalize: 'clamp01'`. */
  readonly clamp01?: boolean;
  /** The density grid. */
  readonly density: VDBSparseChannelCpu;
  /** Optional mask grid (level 0). */
  readonly mask?: VDBSparseChannelCpu;
  /** Optional albedo: one grey grid or three (RGB), one box (level 0). */
  readonly albedo?: { readonly images: readonly Uint32Array[]; readonly box: { readonly min: Vec3; readonly max: Vec3 } };
  /** Optional emission grid (level 0). */
  readonly emission?: VDBSparseChannelCpu;
  /** `'coarse-pyramid'`: the pyramid as the GPU samples it (`vdbPyramidToHalf(vdbBoxPyramid(...))`). */
  readonly pyramid?: VDBBoxPyramid;
}

/** A twin sample: the value and the largest magnitude among the values it was blended from (for tolerances). */
export interface VDBSparseSampleCpu {
  /** The sampled value. */
  readonly value: number;
  /** The largest |tap| the value was interpolated from. */
  readonly scale: number;
  /** True when a pyramid (hardware-filtered r16float) tap contributed. */
  readonly filtered: boolean;
}

/** Levels the volume's density has above level 0 (`vdbSrcLodLevels`). */
export function vdbSparseLodLevelsCpu(vol: VDBSparseVolumeCpu): number {
  return vol.lod === "none" ? 0 : vol.lod === "node-average" ? 7 : (vol.pyramid?.lodLevels ?? 0);
}

/** uvw → sample index position, clamped to the voxels, in f32 as the WGSL (`vdbSrcSamplePos`). */
function samplePos(box: { readonly min: Vec3; readonly max: Vec3 }, uvw: readonly number[]): Vec3 {
  return [0, 1, 2].map((a) => {
    const lo = box.min[a]!;
    const size = box.max[a]! - lo + 1;
    const p = f(f(lo + f(f(uvw[a]!) * size)) - 0.5);
    return Math.min(Math.max(p, lo), lo + size - 1);
  }) as Vec3;
}

/** Level-0 trilinear (`vdbSample`) with the tap magnitude. */
function level0(image: Uint32Array, p: Vec3): { value: number; scale: number } {
  const b = p.map(Math.floor) as Vec3;
  const w = p.map((v, i) => v - b[i]!);
  const t: number[] = [];
  for (let k = 0; k < 8; k++) t.push(vdbReadValueF32(image, [b[0] + (k & 1), b[1] + ((k >> 1) & 1), b[2] + ((k >> 2) & 1)]));
  const lerp = (a: number, c: number, s: number) => a + (c - a) * s;
  const x00 = lerp(t[0]!, t[1]!, w[0]!);
  const x10 = lerp(t[2]!, t[3]!, w[0]!);
  const x01 = lerp(t[4]!, t[5]!, w[0]!);
  const x11 = lerp(t[6]!, t[7]!, w[0]!);
  return { value: lerp(lerp(x00, x10, w[1]!), lerp(x01, x11, w[1]!), w[2]!), scale: Math.max(...t.map(Math.abs)) };
}

/** Hardware linear filtering, clamp to edge, of `data` (`dims`) at texel coordinate `texel` (centres at i + ½). */
function linearTexel(data: Float32Array, dims: readonly number[], texel: readonly number[]): { value: number; scale: number } {
  const [W, H, D] = [dims[0]!, dims[1]!, dims[2]!];
  const fx = texel[0]! - 0.5;
  const fy = texel[1]! - 0.5;
  const fz = texel[2]! - 0.5;
  const [x0, y0, z0] = [Math.floor(fx), Math.floor(fy), Math.floor(fz)];
  const [tx, ty, tz] = [fx - x0, fy - y0, fz - z0];
  const cl = (i: number, n: number) => Math.min(Math.max(i, 0), n - 1);
  let scale = 0;
  const at = (x: number, y: number, z: number) => {
    const v = data[cl(x, W) + (cl(y, H) + cl(z, D) * H) * W]!;
    scale = Math.max(scale, Math.abs(v));
    return v;
  };
  const c00 = at(x0, y0, z0) * (1 - tx) + at(x0 + 1, y0, z0) * tx;
  const c10 = at(x0, y0 + 1, z0) * (1 - tx) + at(x0 + 1, y0 + 1, z0) * tx;
  const c01 = at(x0, y0, z0 + 1) * (1 - tx) + at(x0 + 1, y0, z0 + 1) * tx;
  const c11 = at(x0, y0 + 1, z0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1, z0 + 1) * tx;
  return { value: (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz, scale };
}

/** His `sampleMipmapped` over the pyramid (base + atlas), `lod` relative to the base (`vdbSrcPyrSample`). */
function pyramidSample(p: VDBBoxPyramid, uvw: readonly number[], lod: number): { value: number; scale: number } {
  const size0 = p.dimensions;
  const base = () => linearTexel(p.data, size0, size0.map((s, a) => Math.min(Math.max(uvw[a]! * s, 0.5), s - 0.5)));
  const levels = p.atlas?.levels ?? 0;
  const mip = (level: number) => {
    const atlas = p.atlas!;
    const size = size0.map((t) => Math.max(1, Math.ceil(t / 2 ** level)));
    const offset = atlas.dimensions[0] - atlas.dimensions[0] / 2 ** (level - 1);
    const texel = size.map((s, a) => Math.min(Math.max(uvw[a]! * s, 0.5), s - 0.5));
    return linearTexel(atlas.data, atlas.dimensions, [texel[0]! + offset, texel[1]!, texel[2]!]);
  };
  const level = Math.min(lod, levels);
  if (level <= 0) return base();
  const level0i = Math.floor(level);
  const low = level0i < 0.5 ? base() : mip(level0i);
  const high = mip(Math.min(level0i + 1, levels));
  const t = level - level0i;
  return { value: low.value * (1 - t) + high.value * t, scale: Math.max(low.scale, high.scale) };
}

/** Twin of `vdbSrcDensity(v, uvw, lod)` of the core sparse source (uvw already offset and wrapped by the caller). */
export function vdbSparseDensityCpu(vol: VDBSparseVolumeCpu, uvw: readonly number[], lod: number): VDBSparseSampleCpu {
  const { image, box } = vol.density;
  const p = samplePos(box, uvw);
  const out = (s: { value: number; scale: number }, filtered: boolean): VDBSparseSampleCpu => ({
    value: vol.clamp01 ? Math.min(Math.max(s.value, 0), 1) : s.value,
    scale: s.scale,
    filtered,
  });
  if (vol.lod === "none") return out(level0(image, p), false);
  const l = Math.min(Math.max(lod, 0), vdbSparseLodLevelsCpu(vol));
  if (vol.lod === "node-average") {
    if (l <= 0) return out(level0(image, p), false);
    const fine = l < 3;
    const a = fine ? level0(image, p) : { value: vdbFrameNodeAverageSample(image, p, 8), scale: 0 };
    if (l >= 7) {
      const v = vdbFrameNodeAverageSample(image, p, 128);
      return out({ value: v, scale: Math.abs(v) }, false);
    }
    const b = vdbFrameNodeAverageSample(image, p, fine ? 8 : 128);
    const t = fine ? l / 3 : (l - 3) / 4;
    return out({ value: a.value * (1 - t) + b * t, scale: Math.max(a.scale, Math.abs(a.value), Math.abs(b)) }, false);
  }
  const pyr = vol.pyramid;
  if (!pyr) throw new VDBToolsError("VDB_BAD_OPTION", "vdbSparseDensityCpu: lod 'coarse-pyramid' needs the volume's pyramid");
  const L0 = pyr.level;
  if (l <= 0) return out(level0(image, p), false);
  if (l < L0) {
    const a = level0(image, p);
    const b = pyramidSample(pyr, uvw, 0);
    const t = l / L0;
    return out({ value: a.value * (1 - t) + b.value * t, scale: Math.max(a.scale, b.scale) }, true);
  }
  return out(pyramidSample(pyr, uvw, l - L0), true);
}

/** Twin of `vdbSrcMask(v, uvw, lod)` (level 0; 1 without a mask). */
export function vdbSparseMaskCpu(vol: VDBSparseVolumeCpu, uvw: readonly number[]): VDBSparseSampleCpu {
  if (!vol.mask) return { value: 1, scale: 1, filtered: false };
  return { ...level0(vol.mask.image, samplePos(vol.mask.box, uvw)), filtered: false };
}

/** Twin of `vdbSrcAlbedo(v, uvw, lod)`: level 0, clamped to [0, 1]; (1, 1, 1) without albedo. */
export function vdbSparseAlbedoCpu(vol: VDBSparseVolumeCpu, uvw: readonly number[]): { value: Vec3; scale: number } {
  const a = vol.albedo;
  if (!a) return { value: [1, 1, 1], scale: 1 };
  const p = samplePos(a.box, uvw);
  const taps = (a.images.length === 3 ? a.images : [a.images[0]!, a.images[0]!, a.images[0]!]).map((img) => level0(img, p));
  return { value: taps.map((t) => Math.min(Math.max(t.value, 0), 1)) as Vec3, scale: Math.max(...taps.map((t) => t.scale)) };
}

/** Twin of `vdbSrcEmission(v, uvw, lod)` (level 0; 0 without emission). */
export function vdbSparseEmissionCpu(vol: VDBSparseVolumeCpu, uvw: readonly number[]): VDBSparseSampleCpu {
  if (!vol.emission) return { value: 0, scale: 0, filtered: false };
  return { ...level0(vol.emission.image, samplePos(vol.emission.box, uvw)), filtered: false };
}

/**
 * Twin of `vdbSrcSkipSpan(v, o, d, t0, t1, lod, cutoff)`: the HDDA query of a march `o + d·t` in uvw (the source
 * box [0, 1]³, no offset or wrap) whose samples are read at `lod`. Exact only where the bound covers every voxel a
 * sample reads (the K3.2 rules): level 0 (2x2x2 block), or the dilated 3x3x3 block for node averages up to level 3
 * and pyramids with L₀ ≤ 2 up to lod 2; off when `lod > 3` (the cost rule) or `skipEmpty` is off: `[t0, 1e30]`.
 * With a mask only a zero bound skips. Returns `[ts, te]` as {@link vdbFrameSkipEmpty}.
 */
export function vdbSparseSkipSpanCpu(
  vol: VDBSparseVolumeCpu,
  o: readonly number[],
  d: readonly number[],
  t0: number,
  t1: number,
  lod: number,
  cutoff: number,
  skipEmpty = true,
): [number, number] {
  let enabled = skipEmpty && lod <= 3;
  let dilate = false;
  const l = Math.min(Math.max(lod, 0), vdbSparseLodLevelsCpu(vol));
  if (vol.lod === "node-average" && l > 0) {
    dilate = true;
    if (l > 3) enabled = false;
  } else if (vol.lod === "coarse-pyramid" && l > 0) {
    dilate = true;
    if ((vol.pyramid?.level ?? 0) > 2 || l > 2) enabled = false;
  }
  if (!enabled) return [t0, 1e30];
  const { box, image } = vol.density;
  const o2 = [0, 1, 2].map((a) => f(f(box.min[a]! - 0.5) + f(f(o[a]!) * (box.max[a]! - box.min[a]! + 1)))) as Vec3;
  const d2 = [0, 1, 2].map((a) => f(f(d[a]!) * (box.max[a]! - box.min[a]! + 1))) as Vec3;
  const c = vol.mask ? 0 : Math.max(cutoff, 0);
  return vdbFrameSkipEmpty(image, o2, d2, t0, t1, c, dilate);
}
