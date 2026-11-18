// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameMipmaps.js L14–106 (level sizes, 2x2x2 box downsample with
//   clamped edge taps, the mip atlas layout), as float data.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: f32 data without the 8-bit rounding; the first level comes from a sparse grid through
// `resampleGrid` (box); node-average sampling and the HDDA skip are ours (K3.2). Divergences:
// DIVERGES.md (K3.2 rows).

/**
 * CPU side of the sparse volume source's LOD strategies (K3.2, R1 §4.3) and the CPU twins of its
 * WGSL (`@pmndrs/vdb/three` `sparse-source.ts`):
 *
 * - `'coarse-pyramid'`: {@link vdbBoxPyramid} builds the dense coarse levels his
 *   `sampleExtinction(localPos, lod)` reads at lod ≥ 1. The base level L₀ = max(1, ⌈log2(maxTexels /
 *   pyramidMax)⌉) is a box resample of the sparse grid (`resampleGrid`, filter `'box'`) to
 *   ⌈texels / 2^L₀⌉ voxels over the exact index box, so texel j is centred at uvw (j + 0.5) / dims as
 *   his mip levels are; the levels above it are his 2x2x2 box mip atlas, in float
 *   ({@link vdbBoxMipAtlasF32}).
 * - `'node-average'`: {@link vdbFrameNodeAverageSample} is the twin of the trilinear interpolation
 *   of `vdbNodeAverage` over the leaf (dim 8) or lower-node (dim 128) lattice.
 * - HDDA skip: {@link vdbFrameSkipEmpty} is the twin of the shadow march's empty-space query.
 *
 * Engine-free and core-free (D20): reads the NanoVDB image with the `tools/cpu` twins.
 */

import { cellBoundTrilinear, hddaCellEnd, hddaInit, hddaStep, indexBBox, leafVoxels, nodeAverage, nodeMax, treeNodes, type Vec3 } from "../cpu/march.js";
import { resampleGrid } from "../nanovdb/resample.js";
import { GRID_OFF_MAP, MAP_OFF_MATD, MAP_OFF_VECD } from "../nanovdb/bytes.js";

const f = Math.fround;

/** frameMipmaps.js L14 `getLevelSize`: ceil sizes. */
const levelSize = (size: number, level: number): number => Math.max(1, Math.ceil(size / 2 ** level));

/** The base level of the coarse pyramid: max(1, ⌈log2(maxTexels / pyramidMax)⌉) (R1 §4.3). */
export function vdbPyramidBaseLevel(texels: readonly number[], pyramidMax = 128): number {
  const maxTexels = Math.max(...texels, 1);
  return Math.max(1, Math.ceil(Math.log2(maxTexels / Math.max(1, pyramidMax))));
}

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
 * The coarse pyramid of a sparse FLOAT/Fp grid image (CPU, `resampleGrid` box). The grid's Map must
 * be axis-aligned (resampleGrid's rule).
 */
export function vdbBoxPyramid(image: Uint32Array, o: VDBBoxPyramidOptions = {}): VDBBoxPyramid {
  const box = o.indexBox ?? indexBBox(image);
  const texels: Vec3 = [box.max[0] - box.min[0] + 1, box.max[1] - box.min[1] + 1, box.max[2] - box.min[2] + 1];
  if (texels.some((t) => !(t > 0))) throw new Error("vdbBoxPyramid: empty index box");
  const level = o.level ?? vdbPyramidBaseLevel(texels, o.pyramidMax ?? 128);
  const dims: Vec3 = [levelSize(texels[0], level), levelSize(texels[1], level), levelSize(texels[2], level)];
  // The index cell box [min − ½, max + ½] in world (NanoVDB: world = matd · ijk + vecd).
  const view = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const mat = Array.from({ length: 9 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_MATD + i * 8, true));
  const vec = Array.from({ length: 3 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_VECD + i * 8, true));
  const world = (ijk: Vec3): Vec3 => [0, 1, 2].map((r) => mat[r * 3]! * ijk[0] + mat[r * 3 + 1]! * ijk[1] + mat[r * 3 + 2]! * ijk[2] + vec[r]!) as Vec3;
  const lo = world([box.min[0] - 0.5, box.min[1] - 0.5, box.min[2] - 0.5]);
  const hi = world([box.max[0] + 0.5, box.max[1] + 0.5, box.max[2] + 0.5]);
  const coarse = resampleGrid(image, { filter: "box", dimensions: dims, bbox: [lo, hi] });
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
