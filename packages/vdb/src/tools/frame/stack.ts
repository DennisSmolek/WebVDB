// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L1061–1194 (createGridNodes: sampleTiles, sampleShadows),
//   FrameShadowDensityMaterial.js L111–155 (the shadow density pass at one voxel).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: stacked 3D textures instead of his 2D slice atlases (K6.2, design §3.2): the layout of the sub-volumes
// and the hardware trilinear sample of one, as f64 CPU twins of the core kernels. Divergences: DIVERGES.md.

/**
 * Stacked 3D textures of the core frame renderer (K6.2, design §3.2): several logical R³ grids (cascades × targets)
 * are sub-volumes of one 3D texture, tiled along z, then x, then y within the device's `maxTextureDimension3D`. A
 * read clamps the coordinate to the sub-volume's half-texel box, which equals clamp-to-edge per sub-volume and is his
 * `sampleTiles` (bilinear in a slice, linear between slices, both clamped). The core imports the layout; the
 * `k6-core` harness compares the kernels against these twins.
 */

import { vdbFrameFindCascade, vdbFrameGridMarch, vdbFrameGridUvw, vdbFrameIsotropicDiffusion, type VDBFrameGridLayout, type VDBFrameShadowLight } from "./grid.js";
import { vdbFrameNoise, vdbFrameShadowAbsorption, vdbFrameShadowDepth, vdbFrameShadowStep, vdbFrameShadowTransmittance, type Vec2, type Vec3, type Vec4 } from "./index.js";

/** Where the sub-volumes of a stack sit. */
export interface VDBFrameStackLayout {
  /** The 3D texture's size in texels. */
  readonly size: [number, number, number];
  /** Texel origin of each sub-volume, in order. */
  readonly origins: [number, number, number][];
}

/**
 * The layout of `count` cubes of `side` texels in one 3D texture: along z first (as many as fit in
 * `maxDimension`), then further columns along x, then rows along y. Throws when they do not fit.
 */
export function vdbFrameStackLayout(side: number, count: number, maxDimension = 2048): VDBFrameStackLayout {
  const perAxis = Math.floor(maxDimension / side);
  if (perAxis < 1 || count > perAxis * perAxis * perAxis) throw new RangeError(`vdbFrameStackLayout: ${count} sub-volumes of ${side}³ do not fit ${maxDimension}³`);
  const alongZ = Math.min(count, perAxis);
  const columns = Math.min(Math.ceil(count / alongZ), perAxis);
  const rows = Math.ceil(count / (alongZ * columns));
  const origins: [number, number, number][] = [];
  for (let i = 0; i < count; i++) {
    const z = i % alongZ;
    const x = Math.floor(i / alongZ) % columns;
    const y = Math.floor(i / (alongZ * columns));
    origins.push([x * side, y * side, z * side]);
  }
  return { size: [columns * side, rows * side, alongZ * side], origins };
}

/** A read-back 3D texture: `data` is `channels` values per texel, x fastest, then y, then z. */
export interface VDBFrameTexture3d {
  readonly data: ArrayLike<number>;
  readonly size: readonly [number, number, number];
  readonly channels: number;
}

/** Hardware trilinear filtering (texel centres at i + ½, clamp to edge) of a 3D texture at texel coordinates `p`. */
export function vdbFrameTrilinear(t: VDBFrameTexture3d, p: readonly number[]): number[] {
  const [W, H, D] = t.size;
  const f = [p[0]! - 0.5, p[1]! - 0.5, p[2]! - 0.5];
  const i0 = f.map(Math.floor);
  const w = f.map((v, a) => v - i0[a]!);
  const cl = (i: number, n: number) => Math.min(Math.max(i, 0), n - 1);
  const out = new Array<number>(t.channels).fill(0);
  for (let k = 0; k < 8; k++) {
    const dx = k & 1;
    const dy = (k >> 1) & 1;
    const dz = (k >> 2) & 1;
    const weight = (dx ? w[0]! : 1 - w[0]!) * (dy ? w[1]! : 1 - w[1]!) * (dz ? w[2]! : 1 - w[2]!);
    if (weight === 0) continue;
    const at = (cl(i0[0]! + dx, W) + (cl(i0[1]! + dy, H) + cl(i0[2]! + dz, D) * H) * W) * t.channels;
    for (let c = 0; c < t.channels; c++) out[c]! += t.data[at + c]! * weight;
  }
  return out;
}

/** `vdbFrameStackSample`: the sub-volume at `origin` (side `side`) at `uvw`, clamped to its half-texel box. */
export function vdbFrameStackSample(t: VDBFrameTexture3d, origin: readonly number[], side: number, uvw: readonly number[]): number[] {
  return vdbFrameTrilinear(
    t,
    [0, 1, 2].map((a) => origin[a]! + Math.min(Math.max(uvw[a]! * side, 0.5), side - 0.5)),
  );
}

/** His grid atlas pixel of a voxel (the jitter noise coordinate): tile `cascade · R + z` of `columns` columns. */
export function vdbFrameGridFragCoord(resolution: number, columns: number, cascade: number, cell: Vec3): Vec2 {
  const tile = cascade * resolution + cell[2];
  return [(tile % columns) * resolution + cell[0] + 0.5, Math.floor(tile / columns) * resolution + cell[1] + 0.5];
}

/** The world position of grid voxel `cell` of `cascade` (his getGridCell + gridMin + uvw · gridSize). */
export function vdbFrameGridVoxelWorld(grid: VDBFrameGridLayout, cascade: number, cell: Vec3): Vec3 {
  const m = grid.gridMin[cascade]!;
  const s = grid.gridSize[cascade]!;
  const r = grid.resolution;
  return [m[0] + ((cell[0] + 0.5) / r) * s[0], m[1] + ((cell[1] + 0.5) / r) * s[1], m[2] + ((cell[2] + 0.5) / r) * s[2]];
}

/** One shadow caster as the shadow density pass reads it. */
export interface VDBFrameGridCaster {
  /** World → the local unit box (column-major). */
  readonly worldToLocal: ArrayLike<number>;
  /** The source's largest texel count (getLod) and the volume's max world scale. */
  readonly maxTexels: number;
  readonly worldScale: number;
  /** The volume reads mip levels (else level 0). */
  readonly mipmaps: boolean;
  readonly absorbance: number;
  /** sampleExtinction (extinction per world unit) and sampleAlbedo at a local position and lod. */
  extinctionAt(local: Vec3, lod: number): number;
  albedoAt(local: Vec3, lod: number): Vec3;
}

/**
 * The shadow density pass at one voxel (`vdbFrameGridDensityCs`): the summed (extinction, absorption) of the casters
 * whose unit box holds the voxel centre, each read at the voxel's lod (`voxelSize` = the cascade's largest voxel side).
 */
export function vdbFrameGridDensityVoxel(world: Vec3, voxelSize: number, casters: readonly VDBFrameGridCaster[]): Vec2 {
  let extinction = 0;
  let absorption = 0;
  for (const c of casters) {
    const m = c.worldToLocal;
    const local = [0, 1, 2].map((r) => m[r]! * world[0] + m[4 + r]! * world[1] + m[8 + r]! * world[2] + m[12 + r]!) as Vec3;
    if (Math.max(Math.abs(local[0]), Math.abs(local[1]), Math.abs(local[2])) > 0.5) continue;
    const lod = Math.max(0, Math.log2((voxelSize / c.worldScale) * c.maxTexels));
    const readLod = c.mipmaps ? lod : 0;
    const e = c.extinctionAt(local, readLod) * c.absorbance;
    if (!(e > 0)) continue;
    extinction += e;
    absorption += vdbFrameShadowAbsorption(e, c.albedoAt(local, readLod));
  }
  return [extinction, absorption];
}

/**
 * The transmittance pass (grey, octaves / two-sided / neural off) at voxel `cell` of `cascade`, through a grid sampled
 * by `gridAt(cascade, uvw)` ((extinction, absorption) of that cascade): the 4 targets of `vdbFrameTransmittanceCs`.
 * `lights` are the slots (≤ 6), `jitter` the frame's (< 0: off); the noise reads his grid atlas pixel.
 */
export function vdbFrameTransmittanceVoxel3d(
  gridAt: (cascade: number, uvw: Vec3) => Vec2,
  grid: VDBFrameGridLayout,
  cascade: number,
  cell: Vec3,
  lights: readonly VDBFrameShadowLight[],
  skyDirection: Vec3,
  multipleScattering: number,
  jitter: number,
): [Vec4, Vec4, Vec4, Vec4] {
  const out: [Vec4, Vec4, Vec4, Vec4] = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ];
  const world = vdbFrameGridVoxelWorld(grid, cascade, cell);
  const voxel = grid.gridVoxel[cascade]!;
  const extinctionAt = (p: Vec3): Vec2 => {
    const c = vdbFrameFindCascade(grid, p, 0);
    return c < 0 ? [0, 0] : gridAt(c, vdbFrameGridUvw(grid, c, p));
  };
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    near += extinctionAt([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]])[0];
  }
  if (!(near > 0)) return out;
  const noise = vdbFrameNoise(jitter, vdbFrameGridFragCoord(grid.resolution, grid.columns, cascade, cell));
  const n = (v: Vec3): Vec3 => {
    const l = Math.hypot(v[0], v[1], v[2]);
    return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
  };
  const sky = n(skyDirection);
  const slot = (i: number): Vec4 => {
    if (i >= lights.length) return [1, 1, 0, 0];
    const l = lights[i]!;
    const isDirection = l[3] < 1.5;
    const toLight: Vec3 = [l[0] - world[0], l[1] - world[1], l[2] - world[2]];
    return vdbFrameGridMarch(extinctionAt, grid, world, isDirection ? n([l[0], l[1], l[2]]) : n(toLight), isDirection ? 1e20 : Math.hypot(...toLight), noise, multipleScattering);
  };
  const ls = [0, 1, 2, 3, 4, 5].map(slot);
  const up = vdbFrameGridMarch(extinctionAt, grid, world, sky, 1e20, noise, multipleScattering);
  const down = vdbFrameGridMarch(extinctionAt, grid, world, [-sky[0], -sky[1], -sky[2]], 1e20, noise, multipleScattering);
  const skyTotal = (f: Vec4, b: Vec4) => (multipleScattering > 0 ? vdbFrameIsotropicDiffusion(f[2], b[2], f[3]) : f[1]);
  out[0] = [skyTotal(up, down), skyTotal(down, up), ls[0]![1], ls[1]![1]];
  out[1] = [ls[2]![1], ls[3]![1], ls[4]![1], ls[5]![1]];
  out[2] = [ls[0]![0], ls[1]![0], ls[2]![0], ls[3]![0]];
  out[3] = [ls[4]![0], ls[5]![0], 1, 1];
  return out;
}

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

/**
 * sampleShadows on a stacked transmittance texture: the 4 targets at a world position (`targetAt(cascade, k, uvw)`),
 * 1 outside the grid, blended toward the next cascade within 10% of a cascade's sides.
 */
export function vdbFrameStackShadows(targetAt: (cascade: number, k: number, uvw: Vec3) => number[], grid: VDBFrameGridLayout, world: Vec3): number[][] {
  const out = [0, 1, 2, 3].map(() => [1, 1, 1, 1]);
  const cascade = vdbFrameFindCascade(grid, world, 0);
  if (cascade < 0) return out;
  const uvw = vdbFrameGridUvw(grid, cascade, world);
  for (let k = 0; k < 4; k++) out[k] = targetAt(cascade, k, uvw);
  const side = Math.min(uvw[0], uvw[1], uvw[2], 1 - uvw[0], 1 - uvw[1], 1 - uvw[2]);
  const blend = 1 - smoothstep(0, 0.1, side);
  if (blend > 0) {
    const next = vdbFrameFindCascade(grid, world, cascade + 1);
    if (next >= 0) {
      const nextUvw = vdbFrameGridUvw(grid, next, world);
      for (let k = 0; k < 4; k++) {
        const v = targetAt(next, k, nextUvw);
        out[k] = out[k]!.map((x, c) => x + (v[c]! - x) * blend);
      }
    }
  }
  return out;
}

/**
 * The environment grid pass (grey, octaves off; K6.2b) at voxel `cell` of the last cascade, through a grid sampled by
 * `gridAt(cascade, uvw)`: `unshadowed` (the colour sum of `lights`, ≥ 0) where no extinction is within half a voxel,
 * else Σ colour × total of one-voxel-step marches that stay in the last cascade (with diffusion the two-sided total of
 * the march and the opposite one), ≥ 0. His FrameEnvironmentGridMaterial on a 3D grid.
 */
export function vdbFrameEnvironmentGridVoxel3d(
  gridAt: (cascade: number, uvw: Vec3) => Vec2,
  grid: VDBFrameGridLayout,
  cell: Vec3,
  lights: readonly { direction: Vec3; color: Vec3 }[],
  multipleScattering: number,
): Vec4 {
  const cascade = grid.count - 1;
  const world = vdbFrameGridVoxelWorld(grid, cascade, cell);
  const voxel = grid.gridVoxel[cascade]!;
  const uvwOf = (p: Vec3) => vdbFrameGridUvw(grid, cascade, p);
  const unshadowed: Vec3 = [0, 0, 0];
  for (const l of lights) for (let c = 0; c < 3; c++) unshadowed[c] = unshadowed[c]! + l.color[c]!;
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    near += gridAt(cascade, uvwOf([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]]))[0];
  }
  if (!(near > 0)) return [Math.max(unshadowed[0], 0), Math.max(unshadowed[1], 0), Math.max(unshadowed[2], 0), 1];
  const marchDepths = (dir: Vec3): Vec3 => {
    const step = 1 / Math.max(Math.abs(dir[0]) / voxel[0], Math.abs(dir[1]) / voxel[1], Math.abs(dir[2]) / voxel[2]);
    const probe: Vec3 = [world[0] + dir[0] * step * 0.5, world[1] + dir[1] * step * 0.5, world[2] + dir[2] * step * 0.5];
    const depths: Vec3 = [0, 0, 0];
    for (let i = 0; i < 1024; i++) {
      const uvw = uvwOf(probe);
      if (uvw.some((v) => v < 0 || v > 1) || vdbFrameShadowDepth(depths, multipleScattering) > 6) break;
      const s = gridAt(cascade, uvw);
      const d = vdbFrameShadowStep(s[0], s[1], multipleScattering);
      for (let c = 0; c < 3; c++) {
        depths[c] = depths[c]! + d[c]! * step;
        probe[c] = probe[c]! + dir[c]! * step;
      }
    }
    return depths;
  };
  const irradiance: Vec3 = [0, 0, 0];
  for (const l of lights) {
    const depths = marchDepths(l.direction);
    const total =
      multipleScattering > 0
        ? vdbFrameIsotropicDiffusion(depths[2], marchDepths([-l.direction[0], -l.direction[1], -l.direction[2]])[2], depths[1])
        : vdbFrameShadowTransmittance(depths, multipleScattering)[1];
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + l.color[c]! * total;
  }
  return [Math.max(irradiance[0], 0), Math.max(irradiance[1], 0), Math.max(irradiance[2], 0), 1];
}
