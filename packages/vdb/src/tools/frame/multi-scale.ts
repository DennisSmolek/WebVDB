// Portions derived from mjurczyk/openvdb (MIT), commit 4e2d46f ("Add opt-in multi-scale density lookups for
// indirect light", after v0.29):
//   src/three-openvdb/frame/webgpu/frameMultiScaleChunks.js L1–71 (all: frameMultiScaleLight, the Frisvad frame,
//   the 5-point bulk per level, the diffusion of the summed bulk);
//   src/three-openvdb/frame/webgpu/FrameGridMipMaterial.js (the binomial branch: weights 1 3 3 1 on 4 x 4 x 4 taps);
//   src/three-openvdb/frame/webgpu/frameNeuralChunks.js (createNeuralMipNodes: sampleLevel, unit);
//   scripts/neural-indirect/features.mjs (buildMips 'binomial', getMultiScaleIndirect).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free f64 CPU twins of the core WGSL (`core/compute/multi-scale.ts`) on 3D textures instead of
// his z-slice atlas; zero-padded binomial levels on an exactly halving layout by default (his clamped edges behind
// `edge: 'clamp'`); level k is addressed in base texels / 2^k. Divergences: DIVERGES.md (K7.3 row).

/**
 * `'multi-scale'` multiple scattering (K7.3; mjurczyk 4e2d46f after Disney "Deep Scattering", Kallweit et al.
 * 2017, §4.2): the scattered light of a light slot at a shadow-grid voxel comes from the density mip levels of the
 * grid toward the light. Level k (0 to levels − 1) covers the segment from h (2^k − 1) to h (2^(k+1) − 1) toward
 * the light (h: the grid's largest voxel side); the mean of 5 points (on the light axis and at ±h 2^k on two axes
 * at right angles to it) is the occluding bulk of that scale; the 1D diffusion approximation of the summed bulk
 * gives the scattered light (total − direct). The direct light stays the explicit shadow march.
 *
 * This module holds the CPU twins of the core GPU pieces (`@pmndrs/vdb` core internals):
 * - {@link vdbFrameMultiScaleLayout}: the mip chain sizes (shared with the GPU build);
 * - {@link vdbFrameMultiScaleMips}: the binomial mip chain (twin of `csMsLevel`), f64;
 * - {@link vdbFrameMultiScaleToHalf}: the f16 texels the GPU writes (twin of `csMsPack`);
 * - {@link vdbFrameTexture3DLinear}: the GPU's linear filter with clamp-to-edge;
 * - {@link vdbFrameMultiScaleSample} / {@link vdbFrameMultiScaleLight}: the WGSL lookup
 *   (`vdbFrameMultiScaleSample` / `vdbFrameMultiScaleLight`).
 *
 * Layout. The base (level 0) is the host's grid texture: `size` texels per axis, at `origin` in it. Levels 1 to
 * levels − 1 are the mips of one 3D texture (mip j = level j + 1). Their sizes halve exactly:
 * D_k = ⌈size / 2^(levels−1)⌉ · 2^(levels−1−k), so texel x of level k covers base texels [x 2^k, (x+1) 2^k) for
 * every size (no drift on odd sizes) and the zero padding past ⌈size / 2^k⌉ texels is part of the filter.
 *
 * Filter. Each level-k texel is Σ w_i w_j w_l · L_{k−1}(2x − 1 + i, …) / 512 with w = 1 3 3 1 (separable, x
 * inner, then y, then z), a binomial (Gaussian-like) stencil: level k is the base blurred with σ ≈ 2^(k−1) voxels.
 * `edge: 'zero'` (default) reads 0 outside the source level (the density outside the grid is 0, as the lookup
 * reads it), so each level keeps the mass of the one below (Σ L_k · 8 = Σ L_{k−1}) as long as nothing blurs past
 * the texture's faces; `'clamp'` is his clamp-to-edge (taps clamped
 * to the valid ⌈size / 2^(k−1)⌉ texels; padding texels repeat the last valid one), identical to his levels when
 * `size` is a multiple of 2^(levels−1).
 */

import { vdbFloatToHalf } from "../nanovdb/half.js";
import { VDBToolsError } from "../errors.js";
import type { Vec2, Vec3 } from "./index.js";

/** Most levels (base included) the layout takes: level 7 is 128 voxels. */
export const VDB_FRAME_MULTI_SCALE_MAX_LEVELS = 8;

/** His default `multiScaleLevels` (base included). */
export const VDB_FRAME_MULTI_SCALE_DEFAULT_LEVELS = 6;

/** Edge rule of the mip build: zero outside the grid (default), or his clamp-to-edge. */
export type VDBFrameMultiScaleEdge = "zero" | "clamp";

/** One mip level (1 to levels − 1). */
export interface VDBFrameMultiScaleMipLevel {
  /** The level k (1 = first mip). */
  level: number;
  /** Texels per axis of the level (exactly halving; mip level k − 1 of the mip texture). */
  dimensions: Vec3;
  /** Texels that cover the base (⌈size / 2^k⌉); the rest is padding. */
  extent: Vec3;
}

/** The sizes of the mip chain of a base grid. */
export interface VDBFrameMultiScaleLayout {
  /** Base texels per axis (the grid region the levels filter). */
  size: Vec3;
  /** Levels including the base (2 to {@link VDB_FRAME_MULTI_SCALE_MAX_LEVELS}). */
  levels: number;
  /** Levels 1 to levels − 1. */
  mips: VDBFrameMultiScaleMipLevel[];
}

const bad = (where: string, message: string): never => {
  throw new VDBToolsError("VDB_BAD_OPTION", `${where}: ${message}`);
};

/**
 * The mip chain sizes of a base grid of `size` texels: D_k = ⌈size / 2^(levels−1)⌉ · 2^(levels−1−k) per axis,
 * extent ⌈size / 2^k⌉.
 *
 * @param size Base texels per axis (positive integers).
 * @param levels Levels including the base (default 6, his `multiScaleLevels`).
 */
export function vdbFrameMultiScaleLayout(size: readonly number[], levels: number = VDB_FRAME_MULTI_SCALE_DEFAULT_LEVELS): VDBFrameMultiScaleLayout {
  const where = "vdbFrameMultiScaleLayout";
  if (size.length !== 3 || size.some((v) => !Number.isInteger(v) || v < 1)) bad(where, `size must be 3 positive integers, got ${JSON.stringify(size)}.`);
  if (!Number.isInteger(levels) || levels < 2 || levels > VDB_FRAME_MULTI_SCALE_MAX_LEVELS) {
    bad(where, `levels must be an integer from 2 to ${VDB_FRAME_MULTI_SCALE_MAX_LEVELS} (base included), got ${JSON.stringify(levels)}.`);
  }
  const s: Vec3 = [size[0]!, size[1]!, size[2]!];
  const top = s.map((v) => Math.ceil(v / 2 ** (levels - 1)));
  const mips: VDBFrameMultiScaleMipLevel[] = [];
  for (let k = 1; k < levels; k++) {
    mips.push({
      level: k,
      dimensions: [top[0]! * 2 ** (levels - 1 - k), top[1]! * 2 ** (levels - 1 - k), top[2]! * 2 ** (levels - 1 - k)],
      extent: [Math.ceil(s[0] / 2 ** k), Math.ceil(s[1] / 2 ** k), Math.ceil(s[2] / 2 ** k)],
    });
  }
  return { size: s, levels, mips };
}

/** A grid on the CPU in texture order: x fastest, then y, then z; `channels` components per texel, interleaved. */
export interface VDBFrameTexture3D {
  data: ArrayLike<number>;
  dimensions: Vec3;
  channels: number;
}

/** Options of {@link vdbFrameMultiScaleMips}. */
export interface VDBFrameMultiScaleMipOptions {
  /** Levels including the base (default 6). */
  levels?: number;
  /** Base texels per axis to filter (default: the source's dimensions minus `origin`). */
  size?: Vec3;
  /** First texel of the region in the source (default 0, 0, 0): one cascade of a stacked grid. */
  origin?: Vec3;
  /** Source components that become the level channels, in order (default [0, 1]: extinction, absorption). */
  channels?: readonly number[];
  /** Edge rule (default `'zero'`). */
  edge?: VDBFrameMultiScaleEdge;
}

/** The f64 mip chain: `levels[k − 1]` is level k (`dimensions` texels × `channels`, interleaved). */
export interface VDBFrameMultiScaleMips {
  layout: VDBFrameMultiScaleLayout;
  channels: number;
  levels: Float64Array[];
}

const weight = (i: number) => (i === 0 || i === 3 ? 1 : 3);

/**
 * The binomial mip chain of a base grid (CPU twin of the core `csMsLevel` kernel): level 1 from the source region,
 * level k from level k − 1, each texel the separable 1 3 3 1 sum over 4 × 4 × 4 taps at 2x − 1 … 2x + 2, / 512.
 *
 * @param source The base grid (texture order).
 * @param options Levels, region, channels and edge rule.
 */
export function vdbFrameMultiScaleMips(source: VDBFrameTexture3D, options: VDBFrameMultiScaleMipOptions = {}): VDBFrameMultiScaleMips {
  const where = "vdbFrameMultiScaleMips";
  const origin = options.origin ?? [0, 0, 0];
  const size = options.size ?? [source.dimensions[0] - origin[0], source.dimensions[1] - origin[1], source.dimensions[2] - origin[2]];
  if ([0, 1, 2].some((a) => origin[a]! < 0 || origin[a]! + size[a]! > source.dimensions[a]!)) {
    bad(where, `the region (origin ${JSON.stringify(origin)}, size ${JSON.stringify(size)}) is outside the source (${source.dimensions.join("x")}).`);
  }
  const select = options.channels ?? [0, 1];
  if (select.length < 1 || select.length > 4 || select.some((c) => !Number.isInteger(c) || c < 0 || c >= source.channels)) {
    bad(where, `channels must be 1 to 4 component indices below ${source.channels}, got ${JSON.stringify(select)}.`);
  }
  const edge = options.edge ?? "zero";
  if (edge !== "zero" && edge !== "clamp") bad(where, `edge must be 'zero' or 'clamp', got ${JSON.stringify(edge)}.`);
  const layout = vdbFrameMultiScaleLayout(size, options.levels);
  const C = select.length;
  const [W, H] = source.dimensions;
  const out: Float64Array[] = [];

  for (const mip of layout.mips) {
    const k = mip.level;
    const D = mip.dimensions;
    const fromBase = k === 1;
    const below = fromBase ? null : out[k - 2]!;
    const srcDims = fromBase ? size : layout.mips[k - 2]!.dimensions;
    const srcExtent = fromBase ? size : layout.mips[k - 2]!.extent;
    const tap = (p: Vec3, c: number): number => {
      let q: Vec3 = p;
      if (edge === "clamp") {
        q = [Math.min(Math.max(p[0], 0), srcExtent[0] - 1), Math.min(Math.max(p[1], 0), srcExtent[1] - 1), Math.min(Math.max(p[2], 0), srcExtent[2] - 1)];
      } else if (p[0] < 0 || p[1] < 0 || p[2] < 0 || p[0] >= srcDims[0] || p[1] >= srcDims[1] || p[2] >= srcDims[2]) {
        return 0;
      }
      if (fromBase) {
        const i = origin[0] + q[0] + (origin[1] + q[1] + (origin[2] + q[2]) * H) * W;
        return source.data[i * source.channels + select[c]!]!;
      }
      return below![(q[0] + (q[1] + q[2] * srcDims[1]) * srcDims[0]) * C + c]!;
    };
    const level = new Float64Array(D[0] * D[1] * D[2] * C);
    for (let z = 0; z < D[2]; z++) {
      for (let y = 0; y < D[1]; y++) {
        for (let x = 0; x < D[0]; x++) {
          const e: Vec3 = edge === "clamp" ? [Math.min(x, mip.extent[0] - 1), Math.min(y, mip.extent[1] - 1), Math.min(z, mip.extent[2] - 1)] : [x, y, z];
          const o = (x + (y + z * D[1]) * D[0]) * C;
          for (let c = 0; c < C; c++) {
            let sz = 0;
            for (let l = 0; l < 4; l++) {
              let sy = 0;
              for (let j = 0; j < 4; j++) {
                let sx = 0;
                for (let i = 0; i < 4; i++) sx += weight(i) * tap([e[0] * 2 - 1 + i, e[1] * 2 - 1 + j, e[2] * 2 - 1 + l], c);
                sy += weight(j) * sx;
              }
              sz += weight(l) * sy;
            }
            level[o + c] = sz / 512;
          }
        }
      }
    }
    out.push(level);
  }
  return { layout, channels: C, levels: out };
}

/**
 * The f16 texels of a level as the GPU writes them (twin of `csMsPack`): `texelChannels` components per texel
 * (1, 2 or 4: r16float, rg16float, rgba16float), the level's `channels` first and 0 after, each f32-rounded then
 * rounded to the nearest half, ties to even (`vdbFloatToHalf`).
 */
export function vdbFrameMultiScaleToHalf(level: ArrayLike<number>, channels: number, texelChannels: number): Uint16Array {
  const texels = level.length / channels;
  const out = new Uint16Array(texels * texelChannels);
  for (let t = 0; t < texels; t++) {
    for (let c = 0; c < texelChannels && c < channels; c++) out[t * texelChannels + c] = vdbFloatToHalf(level[t * channels + c]!);
  }
  return out;
}

/**
 * The GPU's linear filter with clamp-to-edge at a texel-space coordinate (texel centres at i + ½): the trilinear
 * blend of the 8 texels around it, f64 weights (SwiftShader filters in full precision; hardware may quantize the
 * weights to 8 bits). Writes `out` (one entry per channel) and returns it.
 */
export function vdbFrameTexture3DLinear(t: VDBFrameTexture3D, coord: Vec3, out: number[] = []): number[] {
  const { data, dimensions: d, channels } = t;
  const i0: number[] = [];
  const i1: number[] = [];
  const f: number[] = [];
  for (let a = 0; a < 3; a++) {
    const p = coord[a]! - 0.5;
    const fl = Math.floor(p);
    f[a] = p - fl;
    i0[a] = Math.min(d[a]! - 1, Math.max(0, fl));
    i1[a] = Math.min(d[a]! - 1, Math.max(0, fl + 1));
  }
  const at = (x: number, y: number, z: number, c: number) => data[(x + (y + z * d[1]) * d[0]) * channels + c]!;
  for (let c = 0; c < channels; c++) {
    const lerp = (a: number, b: number, w: number) => a * (1 - w) + b * w;
    const y0 = lerp(lerp(at(i0[0]!, i0[1]!, i0[2]!, c), at(i1[0]!, i0[1]!, i0[2]!, c), f[0]!), lerp(at(i0[0]!, i1[1]!, i0[2]!, c), at(i1[0]!, i1[1]!, i0[2]!, c), f[0]!), f[1]!);
    const y1 = lerp(lerp(at(i0[0]!, i0[1]!, i1[2]!, c), at(i1[0]!, i0[1]!, i1[2]!, c), f[0]!), lerp(at(i0[0]!, i1[1]!, i1[2]!, c), at(i1[0]!, i1[1]!, i1[2]!, c), f[0]!), f[1]!);
    out[c] = lerp(y0, y1, f[2]!);
  }
  out.length = channels;
  return out;
}

/** The lookup's grid: the WGSL `VdbFrameMultiScale` struct (core `vdbFrameMultiScaleParams`). */
export interface VDBFrameMultiScaleGrid {
  /** World box of the base region: `boxMin` to `boxMin + boxSize` (his last cascade: gridMin, gridSize). */
  boxMin: Vec3;
  boxSize: Vec3;
  /** h: the largest voxel side of the grid, world units (his `unit`). */
  unit: number;
  /** Levels read, base included (his `multiScaleLevels`; at most the chain's levels). */
  levels: number;
  /** 1 − g of the phase function (his `frameMultipleScattering`); 0 is single scattering (no scattered light). */
  multipleScattering: number;
}

/** The textures the lookup reads: the base region and the mip levels. */
export interface VDBFrameMultiScaleTextures {
  /** The host's grid texture (level 0). */
  base: VDBFrameTexture3D;
  /** First texel of the base region in `base` (default 0, 0, 0). */
  baseOrigin?: Vec3;
  /** Base region texels per axis (the layout's `size`). */
  baseSize: Vec3;
  /** Base components of extinction and absorption (default [0, 1]; one entry: no absorption). */
  baseChannels?: readonly number[];
  /** Levels 1 to levels − 1 (channel 0 extinction, channel 1 absorption when present). */
  mips: readonly VDBFrameTexture3D[];
}

/**
 * (extinction, absorption) of `level` at a world position (WGSL `vdbFrameMultiScaleSample`, his `sampleLevel`):
 * 0 outside the box; level 0 is the base region with its edges clamped (no bleed into a neighbouring cascade);
 * level k is the mip at base texel coordinate / 2^k, clamp-to-edge.
 */
export function vdbFrameMultiScaleSample(textures: VDBFrameMultiScaleTextures, grid: Pick<VDBFrameMultiScaleGrid, "boxMin" | "boxSize">, level: number, world: Vec3): Vec2 {
  const uvw = [0, 1, 2].map((a) => (world[a]! - grid.boxMin[a]!) / grid.boxSize[a]!);
  if (uvw.some((v) => v < 0 || v > 1)) return [0, 0];
  const size = textures.baseSize;
  const q: Vec3 = [uvw[0]! * size[0], uvw[1]! * size[1], uvw[2]! * size[2]];
  if (level === 0) {
    const o = textures.baseOrigin ?? [0, 0, 0];
    const ch = textures.baseChannels ?? [0, 1];
    const c: Vec3 = [o[0] + Math.min(Math.max(q[0], 0.5), size[0] - 0.5), o[1] + Math.min(Math.max(q[1], 0.5), size[1] - 0.5), o[2] + Math.min(Math.max(q[2], 0.5), size[2] - 0.5)];
    const s = vdbFrameTexture3DLinear(textures.base, c);
    return [s[ch[0]!]!, ch.length > 1 ? s[ch[1]!]! : 0];
  }
  const mip = textures.mips[level - 1]!;
  const scale = 2 ** level;
  const s = vdbFrameTexture3DLinear(mip, [q[0] / scale, q[1] / scale, q[2] / scale]);
  return [s[0]!, mip.channels > 1 ? s[1]! : 0];
}

/**
 * The scattered light (total − direct) of a light slot at a world position (WGSL `vdbFrameMultiScaleLight`, his
 * `frameMultiScaleLight`). `direction` is the unit direction to the light, `maxDistance` the distance to it
 * (1e20 for a directional light). The consumer adds it to the direct transmittance of its shadow march and
 * clamps the total to 1 (his transmittance pass).
 *
 * @param sampleLevel (extinction, absorption) of level k at a world position ({@link vdbFrameMultiScaleSample}).
 */
export function vdbFrameMultiScaleLight(
  sampleLevel: (level: number, world: Vec3) => Vec2,
  grid: Pick<VDBFrameMultiScaleGrid, "unit" | "levels" | "multipleScattering">,
  world: Vec3,
  direction: Vec3,
  maxDistance: number,
): number {
  const h = grid.unit;
  const g = grid.multipleScattering;
  const [dx, dy, dz] = direction;
  // Two unit vectors at right angles to the direction (Frisvad 2012), pole at −y (lights below are rare).
  const pole = dy < -0.9999;
  const a = 1 / Math.max(dy + 1, 1e-4);
  const b = -dx * dz * a;
  const u: Vec3 = pole ? [1, 0, 0] : [1 - dx * dx * a, -dx, b];
  const v: Vec3 = pole ? [0, 0, 1] : [b, -dz, 1 - dz * dz * a];
  const depths: Vec3 = [0, 0, 0];
  for (let k = 0; k < grid.levels; k++) {
    const scale = h * 2 ** k;
    const start = scale - h;
    const segment = Math.min(scale, maxDistance - start);
    if (!(segment > 0)) continue;
    const t = start + segment * 0.5;
    const center: Vec3 = [world[0] + dx * t, world[1] + dy * t, world[2] + dz * t];
    const at = (axis: Vec3, sign: number): Vec2 => sampleLevel(k, [center[0] + axis[0] * scale * sign, center[1] + axis[1] * scale * sign, center[2] + axis[2] * scale * sign]);
    const p = [sampleLevel(k, center), at(u, 1), at(u, -1), at(v, 1), at(v, -1)];
    const extinction = (p[0]![0] + p[1]![0] + p[2]![0] + p[3]![0] + p[4]![0]) * 0.2;
    const absorption = (p[0]![1] + p[1]![1] + p[2]![1] + p[3]![1] + p[4]![1]) * 0.2;
    // frameShadowStep of the bulk.
    const absorbed = Math.min(absorption, extinction);
    const reduced = (extinction - absorbed) * g;
    depths[0] += extinction * segment;
    depths[1] += Math.sqrt(absorbed * 3 * (absorbed + reduced)) * segment;
    depths[2] += reduced * segment;
  }
  // frameShadowDepth: 1D diffusion of the summed bulk.
  const total = g > 0 ? Math.min(depths[0], depths[1] + Math.log(depths[2] * 0.75 + 1)) : depths[0];
  return Math.max(Math.exp(-total) - Math.exp(-depths[0]), 0);
}
