// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/wvdb/wvdbKernel.js (toByte, toTexels, writeBricks, background fill),
//   src/three-openvdb/wvdb/WVDBReader.js (getBackgroundTexel, getScale, baked density stats),
//   src/three-openvdb/wvdb/WVDBLoader.js (attach / updateTextures stats rules),
//   src/three-openvdb/frame/core/FrameVolume.js (getDensityStats, getMeanDensity, getDensityScale),
//   src/three-openvdb/frame/webgpu/frameMipmaps.js (box mip atlas: downsample, packAtlas),
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js (sampleMipAtlas, sampleMipmapped, getLod).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript, works from our decoded `.wvdb` levels (read.ts) instead of his worker kernel;
// Gaussian mips not ported. Divergences: DIVERGES.md V20–V22.

/**
 * The dense `.wvdb` volume source, CPU half (K3.0, R1 §4.2): the exact bytes
 * mjurczyk's renderer uploads for a `.wvdb` grid, built from our decoded LODs.
 *
 * His path is `WVDBLoader` → `WVDBReader` → worker `wvdbKernel` (8-bit texels:
 * `clamp(value / scale, 0, 1) * 255 | 0`, level sets 255 inside the band, vector
 * grids RGBA8 with alpha 255) → `new FrameVolume(grid.data, { dimensions, bbox })`
 * (an R8 `Data3DTexture`, linear filter, clamp to edge, no GPU mipmaps) plus a CPU
 * box mip atlas (`frameMipmaps.js`: levels 1..n side by side on x). The engine-free
 * GPU upload is `vdbDenseSource` in core; the three textures are `vdbDenseTextures`
 * in `./three`. Both upload exactly these bytes (`test/tools/wvdb-dense.test.ts`
 * checks them against a re-implementation of his upload path).
 *
 * Values are 8-bit and clamped to [0, 1] (R1 §4.3 `normalize: 'clamp01'`): a
 * density above `scale` saturates, as in his renderer.
 */

import type { WvdbBox, WvdbVec3 } from "./format.js";
import type { WvdbFile, WvdbVolumeGrid, WvdbVolumeLevel } from "./read.js";

/** One uploaded 3D texture: texels x fastest, then y, then z. */
export interface WvdbDenseTexture {
  /** 1 byte per texel (`r8unorm`) or 4 (`rgba8unorm`). */
  readonly data: Uint8Array<ArrayBuffer>;
  readonly dimensions: WvdbVec3;
  readonly format: "r8unorm" | "rgba8unorm";
  /** Bytes per texel (1 or 4). */
  readonly texelBytes: 1 | 4;
  /** CPU box mip atlas (levels 1..`levels` side by side on x), or null without mipmaps. */
  readonly mip: WvdbMipAtlas | null;
}

/** His `packAtlas` output (frameMipmaps.js L65–88). */
export interface WvdbMipAtlas {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly dimensions: WvdbVec3;
  /** Levels in the atlas (level 0 is the texture itself). */
  readonly levels: number;
}

/** Density statistics as his `getDensityStats` (FrameVolume.js L50–62): sum of non-zero texels / 255, and their count. */
export interface WvdbTexelStats {
  readonly sum: number;
  readonly count: number;
}

/** A `.wvdb` LOD as the dense textures his renderer uploads (R1 §4.2 `vdbDenseSource` input). */
export interface WvdbDenseVolume {
  readonly kind: "dense";
  /** The only dense normalization: 8-bit, `clamp(value / scale, 0, 1)` (R1 §4.3). */
  readonly normalize: "clamp01";
  readonly lod: number;
  /** Texels of the density texture; aux textures have the same size. */
  readonly dimensions: WvdbVec3;
  /** World box of the density grid (`grid.bbox`), or null for an empty grid. */
  readonly bbox: WvdbBox | null;
  readonly density: WvdbDenseTexture;
  readonly mask: WvdbDenseTexture | null;
  readonly albedo: WvdbDenseTexture | null;
  readonly emission: WvdbDenseTexture | null;
  /** The stats his volume ends up with: baked (header) when present and the scale is 1, else from the texels. */
  readonly densityStats: WvdbTexelStats;
  readonly densityStatsSource: "baked" | "texels";
  /** `getMeanDensity` of the one volume: `sum / count`, 0 when empty. */
  readonly meanDensity: number;
  /** Largest density texel / 255. */
  readonly maxDensity: number;
  /** Non-zero density texels. */
  readonly activeTexels: number;
  readonly gridNames: { readonly density: string; readonly mask?: string; readonly albedo?: string; readonly emission?: string };
}

export interface WvdbDenseOptions {
  /** Density grid. Default: the grid named `density`, else the first volume grid (his frame.js L519). */
  density?: string;
  /** Scalar grid read as a mask (multiplies density). */
  mask?: string;
  /** Vector grid read as base colour (RGBA8). */
  albedo?: string;
  /** Scalar grid read as emission (`value / scale`; use `scales` for his `emissiveScale`). */
  emission?: string;
  /** LOD to build. Default: the finest LOD the file has decoded. */
  lod?: number;
  /** His `WVDBLoader.setGridScales`: `{ gridName: scale }`, scalar values divide by it (default 1). */
  scales?: Readonly<Record<string, number>>;
  /** Build the CPU box mip atlas (his `FrameVolume` `mipmaps`, default true). */
  mipmaps?: boolean;
}

/** WVDBReader.js L128 / wvdbKernel.js L20: `(Math.min(Math.max(value, 0.0), 1.0) * 255.0) | 0`. */
export const wvdbToByte = (value: number): number => (Math.min(Math.max(value, 0.0), 1.0) * 255.0) | 0;

/**
 * The 8-bit texels his worker kernel produces for one decoded LOD (wvdbKernel.js
 * `toTexels` L24–58, `writeBricks` L89–119 over a background fill L255–256 with
 * `getBackgroundTexel` WVDBReader.js L131–137). Our decoded level already holds
 * the background where no brick is stored, so the per-voxel rule is applied to
 * the whole dense array. Scalar: 1 byte; vector: RGBA8 with alpha 255 (no scale).
 */
export function wvdbTexels(grid: WvdbVolumeGrid, level: WvdbVolumeLevel, scale = 1): Uint8Array<ArrayBuffer> {
  const [w, h, d] = level.dimensions;
  const plane = w * h * d;
  const values = level.values;
  if (level.channels === 3) {
    const out = new Uint8Array(plane * 4);
    for (let i = 0; i < plane; i++) {
      out[i * 4] = wvdbToByte(values[i]!);
      out[i * 4 + 1] = wvdbToByte(values[plane + i]!);
      out[i * 4 + 2] = wvdbToByte(values[2 * plane + i]!);
      out[i * 4 + 3] = 255;
    }
    return out;
  }
  const out = new Uint8Array(plane);
  if (grid.class === "level set") {
    const background = grid.background[0]!;
    for (let i = 0; i < plane; i++) out[i] = Math.abs(values[i]!) < background ? 255 : 0;
    return out;
  }
  for (let i = 0; i < plane; i++) out[i] = wvdbToByte(values[i]! / scale);
  return out;
}

/** FrameVolume.js L50–62 `getDensityStats`. */
export function wvdbTexelStats(data: Uint8Array): WvdbTexelStats {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i]! > 0) {
      sum += data[i]!;
      count++;
    }
  }
  return { sum: sum / 255.0, count };
}

/** frameMipmaps.js L14 `getLevelSize` (ceil sizes; the WebGL path floors, R1 §8 quality note 5). */
export const wvdbMipLevelSize = (size: number, level: number): number => Math.max(1, Math.ceil(size / 2 ** level));

/** frameMipmaps.js L16–17 `getMipLevelCount`. */
export const wvdbMipLevelCount = (width: number, height: number, depth: number): number =>
  Math.ceil(Math.log2(Math.max(width, height, depth, 1)));

/** frameMipmaps.js L20–62: 2x2x2 box mean, `Math.round(sum / 8)`, edge taps clamped. */
function downsample(source: Uint8Array<ArrayBuffer>, width: number, height: number, depth: number, channels: number) {
  const tw = wvdbMipLevelSize(width, 1);
  const th = wvdbMipLevelSize(height, 1);
  const td = wvdbMipLevelSize(depth, 1);
  const target = new Uint8Array(tw * th * td * channels);
  for (let z = 0; z < td; z++) {
    const z0 = Math.min(z * 2, depth - 1);
    const z1 = Math.min(z * 2 + 1, depth - 1);
    for (let y = 0; y < th; y++) {
      const y0 = Math.min(y * 2, height - 1);
      const y1 = Math.min(y * 2 + 1, height - 1);
      const row00 = (y0 + z0 * height) * width;
      const row10 = (y1 + z0 * height) * width;
      const row01 = (y0 + z1 * height) * width;
      const row11 = (y1 + z1 * height) * width;
      for (let x = 0; x < tw; x++) {
        const x0 = Math.min(x * 2, width - 1);
        const x1 = Math.min(x * 2 + 1, width - 1);
        const t = (x + (y + z * th) * tw) * channels;
        for (let c = 0; c < channels; c++) {
          const sum =
            source[(x0 + row00) * channels + c]! +
            source[(x1 + row00) * channels + c]! +
            source[(x0 + row10) * channels + c]! +
            source[(x1 + row10) * channels + c]! +
            source[(x0 + row01) * channels + c]! +
            source[(x1 + row01) * channels + c]! +
            source[(x0 + row11) * channels + c]! +
            source[(x1 + row11) * channels + c]!;
          target[t + c] = Math.round(sum / 8);
        }
      }
    }
  }
  return { data: target, width: tw, height: th, depth: td };
}

/**
 * frameMipmaps.js L90–106 `buildAtlas` + L65–88 `packAtlas`: levels 1..n (n =
 * `ceil(log2(max side))`), level l at x = atlasWidth − atlasWidth / 2^(l−1);
 * atlas width 2^max(ceil(log2 width), n), height and depth those of level 1.
 * Null when the texture has one texel on every side.
 */
export function wvdbMipAtlas(data: Uint8Array<ArrayBuffer>, dimensions: readonly number[], channels: 1 | 4): WvdbMipAtlas | null {
  const [width, height, depth] = dimensions as WvdbVec3;
  const count = wvdbMipLevelCount(width, height, depth);
  if (count < 1) return null;
  const levels: ReturnType<typeof downsample>[] = [];
  let level = { data, width, height, depth };
  for (let index = 1; index <= count; index++) {
    level = downsample(level.data, level.width, level.height, level.depth, channels);
    levels.push(level);
  }
  const atlasWidth = 2 ** Math.max(Math.ceil(Math.log2(width)), levels.length);
  const atlasHeight = wvdbMipLevelSize(height, 1);
  const atlasDepth = wvdbMipLevelSize(depth, 1);
  const atlas = new Uint8Array(atlasWidth * atlasHeight * atlasDepth * channels);
  levels.forEach((l, levelIndex) => {
    const offset = atlasWidth - atlasWidth / 2 ** levelIndex;
    for (let z = 0; z < l.depth; z++) {
      for (let y = 0; y < l.height; y++) {
        const sourceStart = (y + z * l.height) * l.width * channels;
        const targetStart = (offset + (y + z * atlasHeight) * atlasWidth) * channels;
        atlas.set(l.data.subarray(sourceStart, sourceStart + l.width * channels), targetStart);
      }
    }
  });
  return { data: atlas, dimensions: [atlasWidth, atlasHeight, atlasDepth], levels: levels.length };
}

function denseTexture(data: Uint8Array<ArrayBuffer>, dimensions: WvdbVec3, texelBytes: 1 | 4, mipmaps: boolean): WvdbDenseTexture {
  return {
    data,
    dimensions,
    format: texelBytes === 4 ? "rgba8unorm" : "r8unorm",
    texelBytes,
    mip: mipmaps ? wvdbMipAtlas(data, dimensions, texelBytes) : null,
  };
}

function volumeGrid(file: WvdbFile, name: string, role: string): WvdbVolumeGrid {
  const grid = file.grids.find((g) => g.name === name);
  if (!grid) throw new Error(`wvdbDenseVolume: no grid named "${name}" for ${role} (grids: ${file.grids.map((g) => g.name).join(", ")})`);
  if (grid.kind !== "volume") throw new Error(`wvdbDenseVolume: grid "${name}" (${role}) is a points grid; dense sources take volume grids`);
  return grid;
}

function levelOf(grid: WvdbVolumeGrid, lod: number): WvdbVolumeLevel {
  const level = grid.levels[lod];
  if (!level) {
    throw new Error(`wvdbDenseVolume: LOD ${lod} of "${grid.name}" is not decoded (${grid.levels.length} decoded); read the file with a larger maxLod`);
  }
  return level;
}

/**
 * Builds the dense textures his renderer would upload for one LOD of a decoded
 * `.wvdb` file: density (R8), optional mask (R8), albedo (RGBA8) and emission
 * (R8), each with its CPU box mip atlas, plus the density stats his volume uses
 * for `normalizeDensity` (`meanDensity`).
 */
export function wvdbDenseVolume(file: WvdbFile, o: WvdbDenseOptions = {}): WvdbDenseVolume {
  const volumeGrids = file.grids.filter((g): g is WvdbVolumeGrid => g.kind === "volume");
  const densityName = o.density ?? (file.grids.find((g) => g.name === "density" && g.kind === "volume") ?? volumeGrids[0])?.name;
  if (densityName === undefined) throw new Error("wvdbDenseVolume: the file has no volume grid");
  const density = volumeGrid(file, densityName, "density");
  if (density.channels !== 1) throw new Error(`wvdbDenseVolume: density grid "${densityName}" has 3 channels; pass a scalar grid`);
  const lod = o.lod ?? density.levels.length - 1;
  if (!Number.isInteger(lod) || lod < 0) throw new Error(`wvdbDenseVolume: lod must be a non-negative integer, got ${o.lod}`);
  const mipmaps = o.mipmaps ?? true;
  // WVDBReader.js L494: scales[name] || 1.0
  const scaleOf = (name: string) => o.scales?.[name] || 1.0;
  const level = levelOf(density, lod);
  const dimensions = [...level.dimensions] as WvdbVec3;
  const densityData = wvdbTexels(density, level, scaleOf(densityName));

  const aux = (name: string | undefined, role: "mask" | "albedo" | "emission"): WvdbDenseTexture | null => {
    if (name === undefined) return null;
    const grid = volumeGrid(file, name, role);
    const want = role === "albedo" ? 3 : 1;
    if (grid.channels !== want) throw new Error(`wvdbDenseVolume: ${role} grid "${name}" has ${grid.channels} channel(s), expected ${want}`);
    const auxLevel = levelOf(grid, lod);
    if (auxLevel.dimensions.some((v, a) => v !== dimensions[a])) {
      // FrameVolume.js L357–405: raw aux grids are uploaded at the density grid's dimensions.
      throw new Error(
        `wvdbDenseVolume: ${role} grid "${name}" LOD ${lod} is ${auxLevel.dimensions.join("x")}, density is ${dimensions.join("x")}; ` +
          "his renderer uploads aux grids at the density size, so they must match",
      );
    }
    const data = wvdbTexels(grid, auxLevel, scaleOf(name));
    return denseTexture(data, dimensions, role === "albedo" ? 4 : 1, mipmaps);
  };

  // WVDBReader.js L519–523 + WVDBLoader.js L147–152: baked stats when the header has
  // meanDensity and the scale is 1; else WVDBLoader.js L196–198 getDensityStats(data).
  const baked = typeof density.meanDensity === "number" && scaleOf(densityName) === 1.0;
  const densityStats: WvdbTexelStats = baked
    ? { sum: density.meanDensity! * density.nonZeroVoxels!, count: density.nonZeroVoxels! }
    : wvdbTexelStats(densityData);
  let max = 0;
  let active = 0;
  for (let i = 0; i < densityData.length; i++) {
    const v = densityData[i]!;
    if (v > max) max = v;
    if (v > 0) active++;
  }

  return {
    kind: "dense",
    normalize: "clamp01",
    lod,
    dimensions,
    bbox: density.bbox,
    density: denseTexture(densityData, dimensions, 1, mipmaps),
    mask: aux(o.mask, "mask"),
    albedo: aux(o.albedo, "albedo"),
    emission: aux(o.emission, "emission"),
    densityStats,
    densityStatsSource: baked ? "baked" : "texels",
    // FrameVolume.js L65–75 getMeanDensity (one volume)
    meanDensity: densityStats.count > 0 ? densityStats.sum / densityStats.count : 0.0,
    maxDensity: max / 255,
    activeTexels: active,
    gridNames: {
      density: densityName,
      ...(o.mask !== undefined ? { mask: o.mask } : {}),
      ...(o.albedo !== undefined ? { albedo: o.albedo } : {}),
      ...(o.emission !== undefined ? { emission: o.emission } : {}),
    },
  };
}

/** FrameVolume.js L561–568 `getDensityScale`: `densityScale / meanDensity` with `normalizeDensity`. */
export function wvdbDensityScale(volume: Pick<WvdbDenseVolume, "meanDensity">, densityScale = 1.0, normalizeDensity = true): number {
  return normalizeDensity && volume.meanDensity > 0.0 ? densityScale / volume.meanDensity : densityScale;
}

// ---------------------------------------------------------------------------
// CPU twin of the dense source's sampling (vdbSrcDensity & co. in core)
// ---------------------------------------------------------------------------

/** Linear filtering, clamp to edge, of one channel at uvw in [0, 1]³ (texel centres at (i + 0.5) / size). */
function linear(data: Uint8Array, dims: readonly number[], texelBytes: number, channel: number, u: number, v: number, w: number): number {
  // dims: the sampled texture's size (for the mip atlas, the whole atlas: its taps clamp at the atlas edge, not the level's).
  const [W, H, D] = dims as WvdbVec3;
  const fx = u * W - 0.5;
  const fy = v * H - 0.5;
  const fz = w * D - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const z0 = Math.floor(fz);
  const tx = fx - x0;
  const ty = fy - y0;
  const tz = fz - z0;
  const cl = (i: number, n: number) => Math.min(Math.max(i, 0), n - 1);
  const at = (x: number, y: number, z: number) => data[(cl(x, W) + (cl(y, H) + cl(z, D) * H) * W) * texelBytes + channel]! / 255;
  const c00 = at(x0, y0, z0) * (1 - tx) + at(x0 + 1, y0, z0) * tx;
  const c10 = at(x0, y0 + 1, z0) * (1 - tx) + at(x0 + 1, y0 + 1, z0) * tx;
  const c01 = at(x0, y0, z0 + 1) * (1 - tx) + at(x0 + 1, y0, z0 + 1) * tx;
  const c11 = at(x0, y0 + 1, z0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1, z0 + 1) * tx;
  const c0 = c00 * (1 - ty) + c10 * ty;
  const c1 = c01 * (1 - ty) + c11 * ty;
  return c0 * (1 - tz) + c1 * tz;
}

/** frameShaderChunks.js L838–846 `sampleMipAtlas` (level ≥ 1). */
function sampleMipAtlasCpu(tex: WvdbDenseTexture, level: number, uvw: readonly number[], channel: number): number {
  const atlas = tex.mip!;
  const [AW, AH, AD] = atlas.dimensions;
  const size = tex.dimensions.map((t) => Math.max(1, Math.ceil(t / 2 ** level)));
  const offset = AW - AW / 2 ** (level - 1);
  const texel = uvw.map((c, a) => Math.min(Math.max(c * size[a]!, 0.5), size[a]! - 0.5));
  return linear(atlas.data, [AW, AH, AD], tex.texelBytes, channel, (texel[0]! + offset) / AW, texel[1]! / AH, texel[2]! / AD);
}

/**
 * CPU twin of the dense source's read at `uvw` (already offset and wrapped) and
 * `lod` (frameShaderChunks.js L849–877 `sampleMipmapped` / `sample3D`): hardware
 * trilinear on the 8-bit texels (as normalized floats), and with mip levels a
 * linear blend of the two nearest levels from the atlas. Real GPUs filter with
 * limited weight precision (typically 8 fractional bits), so compare with a
 * tolerance of about 1/255 of the local value range.
 */
export function sampleWvdbDenseTexture(tex: WvdbDenseTexture, uvw: readonly number[], lod = 0, channel = 0): number {
  const levels = tex.mip?.levels ?? 0;
  const base = () => linear(tex.data, tex.dimensions, tex.texelBytes, channel, uvw[0]!, uvw[1]!, uvw[2]!);
  if (levels === 0) return base();
  const level = Math.min(lod, levels);
  if (level <= 0) return base();
  const level0 = Math.floor(level);
  const low = level0 < 0.5 ? base() : sampleMipAtlasCpu(tex, level0, uvw, channel);
  const high = sampleMipAtlasCpu(tex, Math.min(level0 + 1, levels), uvw, channel);
  return low + (high - low) * (level - level0);
}
