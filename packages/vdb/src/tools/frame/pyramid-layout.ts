// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameMipmaps.js L14–106 (level sizes, the mip atlas layout).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: the layout alone (no data), shared by the CPU pyramid (`vdbBoxPyramid`) and the GPU build
// in the core (`createBoxPyramid`, K3.2a).

/**
 * The texel layout of the sparse source's coarse pyramid (K3.2 / K3.2a): the base level L₀ and its
 * size, and his box mip atlas above it (levels 1..n side by side on x). Pure arithmetic, so the CPU
 * build (`vdbBoxPyramid`) and the GPU build (core `createBoxPyramid`) agree on every size and offset.
 */

type Vec3 = [number, number, number];

/** frameMipmaps.js L14 `getLevelSize`: ceil sizes. */
export const vdbPyramidLevelSize = (size: number, level: number): number => Math.max(1, Math.ceil(size / 2 ** level));

/** The base level of the coarse pyramid: max(1, ⌈log2(maxTexels / pyramidMax)⌉) (R1 §4.3). */
export function vdbPyramidBaseLevel(texels: readonly number[], pyramidMax = 128): number {
  const maxTexels = Math.max(...texels, 1);
  return Math.max(1, Math.ceil(Math.log2(maxTexels / Math.max(1, pyramidMax))));
}

/** One level of the mip atlas: its x offset in the atlas and its size. */
export interface VDBPyramidAtlasLevel {
  readonly offset: number;
  readonly dimensions: Vec3;
}

/** The layout of {@link vdbBoxPyramidLayout}. */
export interface VDBBoxPyramidLayout {
  /** L₀: the base level in level-0 voxels (texel footprint 2^L₀). */
  readonly level: number;
  /** Voxels of the index box the pyramid covers. */
  readonly texels: Vec3;
  /** Texels of the base level, ⌈texels / 2^L₀⌉. */
  readonly dimensions: Vec3;
  /** His atlas of the levels above the base (null for a 1³ base). */
  readonly atlas: { readonly dimensions: Vec3; readonly levels: readonly VDBPyramidAtlasLevel[] } | null;
  /** Total levels the source has: L₀ + atlas levels. */
  readonly lodLevels: number;
}

/**
 * The coarse pyramid's layout for an index box of `texels` voxels: L₀ (default
 * {@link vdbPyramidBaseLevel}), the base size, and his atlas (frameMipmaps.js `buildAtlas` /
 * `packAtlas`): n = ⌈log2(max base side)⌉ levels, each the ceil half of the one below, level l at
 * x = atlasWidth − atlasWidth / 2^(l−1), atlasWidth = 2^max(⌈log2 width⌉, n), height and depth the
 * first level's.
 */
export function vdbBoxPyramidLayout(texels: readonly number[], o: { pyramidMax?: number; level?: number } = {}): VDBBoxPyramidLayout {
  const t: Vec3 = [texels[0]!, texels[1]!, texels[2]!];
  if (t.some((v) => !(v > 0) || !Number.isInteger(v))) throw new Error(`vdbBoxPyramidLayout: texels must be positive integers, got ${JSON.stringify(texels)}`);
  const level = o.level ?? vdbPyramidBaseLevel(t, o.pyramidMax ?? 128);
  if (!Number.isInteger(level) || level < 0) throw new Error(`vdbBoxPyramidLayout: level must be an integer >= 0, got ${level}`);
  const dims: Vec3 = [vdbPyramidLevelSize(t[0], level), vdbPyramidLevelSize(t[1], level), vdbPyramidLevelSize(t[2], level)];
  const [width, height, depth] = dims;
  const count = Math.ceil(Math.log2(Math.max(width, height, depth, 1)));
  if (count < 1) return { level, texels: t, dimensions: dims, atlas: null, lodLevels: level };
  const atlasWidth = 2 ** Math.max(Math.ceil(Math.log2(width)), count);
  const levels: VDBPyramidAtlasLevel[] = [];
  let size: Vec3 = dims;
  for (let index = 0; index < count; index++) {
    size = [vdbPyramidLevelSize(size[0], 1), vdbPyramidLevelSize(size[1], 1), vdbPyramidLevelSize(size[2], 1)];
    levels.push({ offset: atlasWidth - atlasWidth / 2 ** index, dimensions: size });
  }
  return {
    level,
    texels: t,
    dimensions: dims,
    atlas: { dimensions: [atlasWidth, vdbPyramidLevelSize(height, 1), vdbPyramidLevelSize(depth, 1)], levels },
    lodLevels: level + count,
  };
}
