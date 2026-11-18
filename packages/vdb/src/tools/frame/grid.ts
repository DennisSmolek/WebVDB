// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameGrid.js L12–223 (signature helpers, updateGridCascades,
//   getGridRange, getCascadeCasters, updateGridSignatures);
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L403–414 (frameIsotropicDiffusion),
//   L1061–1194 (createGridNodes: gridUvw, insideGrid, findCascade, sampleTiles, sampleShadows,
//   getGridCell);
//   src/three-openvdb/frame/webgpu/FrameTransmittanceMaterial.js L129–190 (gridExtinction, march,
//   skyTotal), L237–269 (near-extinction skip, slot packing);
//   src/three-openvdb/frame/webgpu/FrameTemporalMaterial.js L119–157 (reprojection, motion blend);
//   src/three-openvdb/frame/webgpu/FrameDensityMaterial.js L172–357 (the grey density pass),
//   FrameLightingMaterial.js L296–436 (the grey lighting pass), FrameEnvironmentGridMaterial.js L92–281
//   (the grey environment grid pass), as CPU twins.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: engine-free (plain `{ x, y, z }` vectors and boxes instead of three's `Box3`/`Vector3`,
// so `tools` stays free of three; the three adapter passes its own objects, which fit); the shader
// maths as f64 CPU twins (test oracles of the `./three` 'frame' lighting passes); no octaves,
// two-sided, neural or spectral terms (K3.4/K3.5). Divergences: DIVERGES.md (K3.3 rows).

/**
 * The world shadow grid of mjurczyk's `'frame'` lighting (K3.3, R1 §6 Step 3), engine-free: the
 * cascade layout (power-of-two voxels, snapped bounds, the last cascade fits every caster), each
 * caster's voxel range in a cascade, the change signatures that skip redraws, and CPU twins of the
 * grid shader maths (atlas cell, cascade search, slice-atlas sampling, the transmittance march,
 * isotropic diffusion, the temporal reprojection). `@pmndrs/vdb/three` `frame/grid.ts` and the
 * 'frame' passes use these; the e2e harness checks the GPU passes against them.
 */

import {
  vdbFrameDistanceFade,
  vdbFrameFroxel,
  vdbFrameNoise,
  vdbFramePhaseWeight,
  vdbFrameShadowDepth,
  vdbFrameShadowStep,
  vdbFrameShadowTransmittance,
  vdbFrameViewPosition,
  type Vec2,
  type Vec3,
  type Vec4,
} from "./index.js";

/** A writable 3-vector (three's `Vector3` fits). */
export interface VDBVec3Like {
  x: number;
  y: number;
  z: number;
}

/** An axis-aligned box (three's `Box3` fits). */
export interface VDBBox3Like {
  min: VDBVec3Like;
  max: VDBVec3Like;
}

const AXES = ["x", "y", "z"] as const;

/** His `pushVector`: x, y (, z (, w)) of a vector into a signature. */
export function vdbSignaturePushVector(values: number[], v: { x: number; y: number; z?: number; w?: number }): void {
  values.push(v.x, v.y);
  if (v.z !== undefined) values.push(v.z);
  if (v.w !== undefined) values.push(v.w);
}

/** His `pushElements`. */
export function vdbSignaturePushElements(values: number[], elements: ArrayLike<number>): void {
  for (let i = 0; i < elements.length; i++) values.push(elements[i]!);
}

/** His `sameValues`: signatures are flat arrays compared value by value. */
export function vdbSameValues(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

const box = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
const casterBox = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };

const makeEmpty = (b: VDBBox3Like) => {
  b.min.x = b.min.y = b.min.z = Infinity;
  b.max.x = b.max.y = b.max.z = -Infinity;
};
const isEmpty = (b: VDBBox3Like) => b.max.x < b.min.x || b.max.y < b.min.y || b.max.z < b.min.z;
const copyBox = (to: VDBBox3Like, from: VDBBox3Like) => {
  for (const a of AXES) {
    to.min[a] = from.min[a];
    to.max[a] = from.max[a];
  }
};

/**
 * His `updateGridCascades`: the last cascade fits all caster boxes (then far volumes get coarse
 * shadows, not none); finer cascades are around the camera within `shadowDistance`, each half the
 * size of the next; voxels are powers of two and bounds snap to them (shadows stay still while the
 * camera moves). A fine cascade equal to the casters' box is skipped (the last one covers it).
 * Writes `gridMin`, `gridSize`, `gridVoxel` (one entry per cascade); returns the count (0: no casters).
 */
export function vdbFrameGridCascades(
  casterBoxes: readonly VDBBox3Like[],
  cameraPosition: VDBVec3Like,
  shadowDistance: number,
  resolution: number,
  cascades: number,
  gridMin: VDBVec3Like[],
  gridSize: VDBVec3Like[],
  gridVoxel: VDBVec3Like[],
): number {
  makeEmpty(casterBox);
  for (const b of casterBoxes) {
    for (const a of AXES) {
      casterBox.min[a] = Math.min(casterBox.min[a], b.min[a]);
      casterBox.max[a] = Math.max(casterBox.max[a], b.max[a]);
    }
  }
  if (isEmpty(casterBox)) return 0;

  const fineHalfSize = Math.min(
    shadowDistance,
    0.5 * Math.max(casterBox.max.x - casterBox.min.x, casterBox.max.y - casterBox.min.y, casterBox.max.z - casterBox.min.z),
  );

  let count = 0;
  for (let cascade = 0; cascade < cascades; cascade++) {
    if (cascade === cascades - 1) {
      copyBox(box, casterBox);
    } else {
      const halfSize = fineHalfSize / Math.pow(2.0, cascades - 1 - cascade);
      for (const a of AXES) {
        // three Box3.intersect: max of the mins, min of the maxes, then empty when inverted.
        box.min[a] = Math.max(cameraPosition[a] - halfSize, casterBox.min[a]);
        box.max[a] = Math.min(cameraPosition[a] + halfSize, casterBox.max[a]);
      }
      if (isEmpty(box)) makeEmpty(box);
    }

    const equalsCasters = AXES.every((a) => box.min[a] === casterBox.min[a] && box.max[a] === casterBox.max[a]);
    if (isEmpty(box) || (cascade < cascades - 1 && equalsCasters)) continue;

    for (const a of AXES) {
      const extent = Math.max(box.max[a] - box.min[a], 1e-6);
      const voxel = Math.pow(2.0, Math.ceil(Math.log2(extent / (resolution - 1))));
      gridVoxel[count]![a] = voxel;
      gridMin[count]![a] = Math.floor(box.min[a] / voxel) * voxel;
      gridSize[count]![a] = voxel * resolution;
    }
    count++;
  }
  return count;
}

const toCell = (value: number, min: number, voxel: number, resolution: number, round: (x: number) => number) =>
  Math.min(resolution, Math.max(0, round((value - min) / voxel)));

/**
 * His `getGridRange`: the voxel rectangle (x0, y0, x1, y1) and the atlas slices (first, count) of a
 * world box in one cascade. Writes `target`; returns false when the box is out of the cascade.
 */
export function vdbFrameGridRange(
  worldBox: VDBBox3Like,
  cascade: number,
  resolution: number,
  gridMin: readonly VDBVec3Like[],
  gridVoxel: readonly VDBVec3Like[],
  target: { rect: { x: number; y: number; z: number; w: number } | Vec4; first: number; count: number },
): boolean {
  const origin = gridMin[cascade]!;
  const voxel = gridVoxel[cascade]!;
  const { min, max } = worldBox;
  const x0 = toCell(min.x, origin.x, voxel.x, resolution, Math.floor);
  const y0 = toCell(min.y, origin.y, voxel.y, resolution, Math.floor);
  const z0 = toCell(min.z, origin.z, voxel.z, resolution, Math.floor);
  const x1 = toCell(max.x, origin.x, voxel.x, resolution, Math.ceil);
  const y1 = toCell(max.y, origin.y, voxel.y, resolution, Math.ceil);
  const z1 = toCell(max.z, origin.z, voxel.z, resolution, Math.ceil);
  if (x1 <= x0 || y1 <= y0 || z1 <= z0) return false;
  if (Array.isArray(target.rect)) {
    target.rect[0] = x0;
    target.rect[1] = y0;
    target.rect[2] = x1;
    target.rect[3] = y1;
  } else {
    const r = target.rect as { x: number; y: number; z: number; w: number };
    r.x = x0;
    r.y = y0;
    r.z = x1;
    r.w = y1;
  }
  target.first = cascade * resolution + z0;
  target.count = z1 - z0;
  return true;
}

/** His `getCascadeCasters`: one item per caster per cascade from `firstCascade`; `pool` is reused. */
export function vdbFrameCascadeCasters<T>(
  casters: readonly T[],
  firstCascade: number,
  count: number,
  pool: { items: { caster: T; cascade: number }[]; entries: { caster: T; cascade: number }[] },
): { caster: T; cascade: number }[] {
  const { items, entries } = pool;
  items.length = 0;
  for (let cascade = firstCascade; cascade < count; cascade++) {
    for (const caster of casters) {
      if (items.length === entries.length) entries.push({ caster, cascade });
      const entry = entries[items.length]!;
      entry.caster = caster;
      entry.cascade = cascade;
      items.push(entry);
    }
  }
  return items;
}

/** The signature state of {@link vdbFrameUpdateGridSignatures} (his renderer fields of the same names). */
export interface VDBFrameGridSignatureState {
  gridSignature: number[];
  nextGridSignature: number[];
  castSignature: number[];
  nextCastSignature: number[];
  casterSignature: number[];
  gridVersion: number;
  castVersion: number;
}

/** A fresh {@link VDBFrameGridSignatureState}. */
export const vdbFrameGridSignatureState = (): VDBFrameGridSignatureState => ({
  gridSignature: [],
  nextGridSignature: [],
  castSignature: [],
  nextCastSignature: [],
  casterSignature: [],
  gridVersion: 0,
  castVersion: 0,
});

/**
 * His `updateGridSignatures`: `gridVersion` changes when the cascades, a caster transform, a caster's
 * data or a caster's shadow props change; `castVersion` only for the last cascade. `pushCaster(values, caster)`
 * adds what the shadow density pass reads of one caster (transform, source identity and version, props).
 * Returns true when `castVersion` changed. Arrays are swapped, never reallocated.
 */
export function vdbFrameUpdateGridSignatures<T>(
  state: VDBFrameGridSignatureState,
  casters: readonly T[],
  count: number,
  resolution: number,
  spectral: boolean,
  gridMin: readonly VDBVec3Like[],
  gridSize: readonly VDBVec3Like[],
  pushCaster: (values: number[], caster: T) => void,
): boolean {
  const casterValues = state.casterSignature;
  const castValues = state.nextCastSignature;
  const gridValues = state.nextGridSignature;

  casterValues.length = 0;
  for (const caster of casters) pushCaster(casterValues, caster);

  // NOTE The cascade count changes the atlas layout, so a new count needs a new draw
  castValues.length = 0;
  castValues.push(casters.length, count, resolution, spectral ? 1 : 0);
  if (count > 0) {
    vdbSignaturePushVector(castValues, gridMin[count - 1]!);
    vdbSignaturePushVector(castValues, gridSize[count - 1]!);
  }
  vdbSignaturePushElements(castValues, casterValues);

  gridValues.length = 0;
  gridValues.push(count);
  for (let i = 0; i < count; i++) {
    vdbSignaturePushVector(gridValues, gridMin[i]!);
    vdbSignaturePushVector(gridValues, gridSize[i]!);
  }
  vdbSignaturePushElements(gridValues, castValues);

  if (!vdbSameValues(gridValues, state.gridSignature)) {
    state.gridVersion++;
    state.nextGridSignature = state.gridSignature;
    state.gridSignature = gridValues;
  }

  const castChanged = !vdbSameValues(castValues, state.castSignature);
  if (castChanged) {
    state.castVersion++;
    state.nextCastSignature = state.castSignature;
    state.castSignature = castValues;
  }
  return castChanged;
}

// ── CPU twins of the grid shader chunks ────────────────────────────────────────────────────────────

/** The grid layout the shaders read (`createGridUniforms`), as plain arrays. */
export interface VDBFrameGridLayout {
  /** Cascades in use (`frameGridCount`). */
  count: number;
  /** Voxels per side (`frameGridResolution`). */
  resolution: number;
  /** Tile columns of the atlas (`frameGridColumns`). */
  columns: number;
  /** Atlas size in texels (`frameGridAtlasSize`). */
  atlasSize: Vec2;
  gridMin: Vec3[];
  gridSize: Vec3[];
  gridVoxel: Vec3[];
}

/** gridUvw: world position → [0, 1]³ of a cascade. */
export function vdbFrameGridUvw(grid: VDBFrameGridLayout, cascade: number, world: Vec3): Vec3 {
  const m = grid.gridMin[cascade]!;
  const s = grid.gridSize[cascade]!;
  return [(world[0] - m[0]) / s[0], (world[1] - m[1]) / s[1], (world[2] - m[2]) / s[2]];
}

/** insideGrid: half a voxel inside the cascade (the linear filter then never reads the clamped edge). */
export function vdbFrameInsideGrid(resolution: number, uvw: Vec3): boolean {
  const inset = 0.5 / resolution;
  return Math.min(uvw[0], uvw[1], uvw[2]) >= inset && Math.max(uvw[0], uvw[1], uvw[2]) <= 1 - inset;
}

/** findCascade: the finest cascade ≥ `firstCascade` that contains the position, or −1. */
export function vdbFrameFindCascade(grid: VDBFrameGridLayout, world: Vec3, firstCascade = 0, maxCascades = 4): number {
  let cascade = -1;
  for (let i = maxCascades - 1; i >= 0; i--) {
    if (i < grid.count && i >= firstCascade && vdbFrameInsideGrid(grid.resolution, vdbFrameGridUvw(grid, i, world))) cascade = i;
  }
  return cascade;
}

/** getGridCell: cascade and grid uvw of an atlas pixel (bottom-left `fragCoord`, a pixel centre). */
export function vdbFrameGridCell(resolution: number, columns: number, fragCoord: Vec2): { cascade: number; gridUvw: Vec3 } {
  const tile: Vec2 = [Math.floor(fragCoord[0] / resolution), Math.floor(fragCoord[1] / resolution)];
  const tileIndex = tile[1] * columns + tile[0];
  const cascade = Math.floor(tileIndex / resolution);
  const z = tileIndex - cascade * resolution;
  return {
    cascade,
    gridUvw: [(fragCoord[0] - tile[0] * resolution) / resolution, (fragCoord[1] - tile[1] * resolution) / resolution, (z + 0.5) / resolution],
  };
}

/**
 * A slice atlas on the CPU: `data` is RGBA (or `channels`) floats, rows **bottom first** (the shaders'
 * `frameFragCoord` convention), `width` × `height` texels. Sampling is the hardware's linear filter
 * with clamp-to-edge, as the render targets are configured (`LinearFilter`).
 */
export interface VDBFrameAtlas {
  data: ArrayLike<number>;
  width: number;
  height: number;
  channels: number;
}

/** Hardware bilinear (texel centres at i + ½, clamp to edge) of a bottom-first atlas at texel coordinates. */
export function vdbFrameAtlasBilinear(atlas: VDBFrameAtlas, x: number, y: number, out: number[] = []): number[] {
  const { data, width, height, channels } = atlas;
  const fx = x - 0.5;
  const fy = y - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const ax = fx - x0;
  const ay = fy - y0;
  const cx = (i: number) => Math.min(width - 1, Math.max(0, i));
  const cy = (i: number) => Math.min(height - 1, Math.max(0, i));
  const xa = cx(x0);
  const xb = cx(x0 + 1);
  const ya = cy(y0);
  const yb = cy(y0 + 1);
  for (let c = 0; c < channels; c++) {
    const t00 = data[(ya * width + xa) * channels + c]!;
    const t10 = data[(ya * width + xb) * channels + c]!;
    const t01 = data[(yb * width + xa) * channels + c]!;
    const t11 = data[(yb * width + xb) * channels + c]!;
    out[c] = (t00 * (1 - ax) + t10 * ax) * (1 - ay) + (t01 * (1 - ax) + t11 * ax) * ay;
  }
  out.length = channels;
  return out;
}

/** sampleTiles: hardware bilinear in a slice, linear between slices (z tiles from `tileOffset`). */
export function vdbFrameSampleTiles(atlas: VDBFrameAtlas, resolution: number, tileOffset: number, columns: number, uvw: Vec3): number[] {
  const z = uvw[2] * resolution - 0.5;
  const z0 = Math.min(Math.max(Math.floor(z), 0), resolution - 1);
  const z1 = Math.min(z0 + 1, resolution - 1);
  const blend = Math.min(Math.max(z - z0, 0), 1);
  const xy: Vec2 = [Math.min(Math.max(uvw[0] * resolution, 0.5), resolution - 0.5), Math.min(Math.max(uvw[1] * resolution, 0.5), resolution - 0.5)];
  const tileTexel = (index: number): Vec2 => [(index % columns) * resolution + xy[0], Math.floor(index / columns) * resolution + xy[1]];
  const [ax, ay] = tileTexel(tileOffset + z0);
  const [bx, by] = tileTexel(tileOffset + z1);
  const a = vdbFrameAtlasBilinear(atlas, ax, ay);
  const b = vdbFrameAtlasBilinear(atlas, bx, by);
  return a.map((v, c) => v + (b[c]! - v) * blend);
}

/** sampleGrid: {@link vdbFrameSampleTiles} of one cascade of the grid atlas. */
export function vdbFrameSampleGrid(atlas: VDBFrameAtlas, grid: VDBFrameGridLayout, cascade: number, uvw: Vec3): number[] {
  return vdbFrameSampleTiles(atlas, grid.resolution, cascade * grid.resolution, grid.columns, uvw);
}

/** gridExtinction of the transmittance pass: (extinction, absorption) at a world position, 0 outside the grid. */
export function vdbFrameGridExtinction(atlas: VDBFrameAtlas, grid: VDBFrameGridLayout, world: Vec3): Vec2 {
  const found = vdbFrameFindCascade(grid, world, 0);
  if (found < 0) return [0, 0];
  const s = vdbFrameSampleGrid(atlas, grid, found, vdbFrameGridUvw(grid, found, world));
  return [s[0]!, s[1]!];
}

/**
 * The transmittance pass `march` (grey, octaves off): one voxel of the finest cascade at the probe per
 * step along the direction's fastest axis, coarser cascades after leaving a fine one, stop outside the
 * grid, at `maxDistance` or past optical depth 6. Each step samples `jitter` of the way along it.
 * Returns (direct, total, reduced depth, effective depth).
 */
export function vdbFrameGridMarch(
  extinctionAt: (world: Vec3) => Vec2,
  grid: VDBFrameGridLayout,
  origin: Vec3,
  direction: Vec3,
  maxDistance: number,
  jitter: number,
  multipleScattering: number,
): Vec4 {
  const depths: Vec3 = [0, 0, 0];
  let traveled = 0;
  const probe: Vec3 = [...origin];
  for (let i = 0; i < 1024; i++) {
    const cascade = vdbFrameFindCascade(grid, probe, 0);
    if (cascade < 0 || traveled >= maxDistance || vdbFrameShadowDepth(depths, multipleScattering) > 6) break;
    const voxel = grid.gridVoxel[cascade]!;
    const perUnit = Math.max(Math.abs(direction[0]) / voxel[0], Math.abs(direction[1]) / voxel[1], Math.abs(direction[2]) / voxel[2]);
    const step = Math.min(1 / perUnit, maxDistance - traveled);
    const s = extinctionAt([probe[0] + direction[0] * step * jitter, probe[1] + direction[1] * step * jitter, probe[2] + direction[2] * step * jitter]);
    const d = vdbFrameShadowStep(s[0], s[1], multipleScattering);
    depths[0] += d[0] * step;
    depths[1] += d[1] * step;
    depths[2] += d[2] * step;
    for (let c = 0; c < 3; c++) probe[c] = probe[c]! + direction[c]! * step;
    traveled += step;
  }
  const t = vdbFrameShadowTransmittance(depths, multipleScattering);
  return [t[0], t[1], depths[2], depths[1]];
}

/**
 * frameIsotropicDiffusion: total transmittance of isotropic light (sky, environment) with diffusion,
 * the two-sided slab 2 (back + z₀) / (front + back + 2 z₀) of the reduced depths (z₀ = ⅔), times
 * exp(−effective front). Opposite directions sum to 2 in a white medium (white furnace).
 */
export function vdbFrameIsotropicDiffusion(frontReduced: number, backReduced: number, frontEffective: number): number {
  const z0 = 2 / 3;
  return ((backReduced + z0) * 2) / (frontReduced + backReduced + 2 * z0) * Math.exp(-frontEffective);
}

/** The shader's `normalize`. */
const normalize3 = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
};

/** One grid light slot of the transmittance pass: position xyz, w < 1.5 a direction to the light, else a position. */
export type VDBFrameShadowLight = Vec4;

/**
 * The transmittance pass (grey, no neural/two-sided/octaves) at one grid voxel: the 4 MRT outputs
 * (vec4 each) his `colorNode` writes, for `lights` (≤ 6 slots) and the sky direction. `fragCoord`
 * is the voxel's atlas pixel (for the jitter noise; `jitter` < 0 is off). Voxels with no extinction
 * within half a voxel keep 1.0 everywhere, as his pass.
 */
export function vdbFrameTransmittanceVoxel(
  atlas: VDBFrameAtlas,
  grid: VDBFrameGridLayout,
  fragCoord: Vec2,
  lights: readonly VDBFrameShadowLight[],
  skyDirection: Vec3,
  multipleScattering: number,
  jitterValue: number,
): [Vec4, Vec4, Vec4, Vec4] {
  const out: [Vec4, Vec4, Vec4, Vec4] = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ];
  const cell = vdbFrameGridCell(grid.resolution, grid.columns, fragCoord);
  if (cell.cascade >= grid.count) return out;
  const voxel = grid.gridVoxel[cell.cascade]!;
  const m = grid.gridMin[cell.cascade]!;
  const s = grid.gridSize[cell.cascade]!;
  const world: Vec3 = [m[0] + cell.gridUvw[0] * s[0], m[1] + cell.gridUvw[1] * s[1], m[2] + cell.gridUvw[2] * s[2]];
  const extinctionAt = (p: Vec3) => vdbFrameGridExtinction(atlas, grid, p);
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    near += extinctionAt([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]])[0];
  }
  if (!(near > 0)) return out;
  const jitter = vdbFrameNoise(jitterValue, fragCoord);
  const sky = normalize3(skyDirection);
  const slot = (i: number): Vec4 => {
    if (i >= lights.length) return [1, 1, 0, 0];
    const l = lights[i]!;
    const isDirection = l[3] < 1.5;
    const toLight: Vec3 = [l[0] - world[0], l[1] - world[1], l[2] - world[2]];
    const dir = isDirection ? normalize3([l[0], l[1], l[2]]) : normalize3(toLight);
    return vdbFrameGridMarch(extinctionAt, grid, world, dir, isDirection ? 1e20 : Math.hypot(...toLight), jitter, multipleScattering);
  };
  const ls = [0, 1, 2, 3, 4, 5].map(slot);
  const up = vdbFrameGridMarch(extinctionAt, grid, world, sky, 1e20, jitter, multipleScattering);
  const down = vdbFrameGridMarch(extinctionAt, grid, world, [-sky[0], -sky[1], -sky[2]], 1e20, jitter, multipleScattering);
  const skyTotal = (f: Vec4, b: Vec4) => (multipleScattering > 0 ? vdbFrameIsotropicDiffusion(f[2], b[2], f[3]) : f[1]);
  out[0] = [skyTotal(up, down), skyTotal(down, up), ls[0]![1], ls[1]![1]];
  out[1] = [ls[2]![1], ls[3]![1], ls[4]![1], ls[5]![1]];
  out[2] = [ls[0]![0], ls[1]![0], ls[2]![0], ls[3]![0]];
  out[3] = [ls[4]![0], ls[5]![0], 1, 1];
  return out;
}

/**
 * sampleShadows: the slot atlases at a world position, 1 outside the grid, blended toward the next
 * cascade within 10% of a cascade's sides (no visible cascade borders).
 */
export function vdbFrameSampleShadows(atlases: readonly VDBFrameAtlas[], grid: VDBFrameGridLayout, world: Vec3): number[][] {
  const out = atlases.map(() => [1, 1, 1, 1]);
  const cascade = vdbFrameFindCascade(grid, world, 0);
  if (cascade < 0) return out;
  const uvw = vdbFrameGridUvw(grid, cascade, world);
  atlases.forEach((a, i) => {
    out[i] = vdbFrameSampleGrid(a, grid, cascade, uvw);
  });
  const side = Math.min(uvw[0], uvw[1], uvw[2], 1 - uvw[0], 1 - uvw[1], 1 - uvw[2]);
  const blend = 1 - smoothstep(0, 0.1, side);
  if (blend > 0) {
    const next = vdbFrameFindCascade(grid, world, cascade + 1);
    if (next >= 0) {
      const nextUvw = vdbFrameGridUvw(grid, next, world);
      atlases.forEach((a, i) => {
        const v = vdbFrameSampleGrid(a, grid, next, nextUvw);
        out[i] = out[i]!.map((x, c) => x + (v[c]! - x) * blend);
      });
    }
  }
  return out;
}

/** The inputs of {@link vdbFrameDensityFroxel}: one volume's uniforms and the renderer's, as the density pass reads them. */
export interface VDBFrameDensityInputs {
  /** Froxel grid (width, height, slices), atlas tile columns, the log slice range and the inverse projection. */
  froxelGrid: Vec3;
  columns: number;
  range: Vec2;
  projectionInverse: ArrayLike<number>;
  cameraWorld: ArrayLike<number>;
  /** The volume's view → local unit-box matrix (column-major). */
  viewToLocal: ArrayLike<number>;
  /** Largest texel count of the source (getLod) and its density mip levels (0: no mipmaps). */
  maxTexels: number;
  mipLevels: number;
  worldScale: number;
  subSteps: number;
  /** `frameJitter` (< 0: off). */
  jitter: number;
  fadeRadius: number;
  opacity: number;
  baseColor: Vec3;
  scatterColor: Vec3;
  /** The source's `sampleExtinction(local, lod)` (extinction per world unit). */
  extinctionAt: (local: Vec3, lod: number) => number;
  /** The 4 transmittance atlases and their layout, or null without shadows. */
  shadows: { atlases: readonly VDBFrameAtlas[]; grid: VDBFrameGridLayout } | null;
  /** Scene lights with a shadow slot, their positions (slot order) and the renderer phase. */
  sceneLightCount: number;
  lightPositions: readonly Vec4[];
  phase: Vec4;
  /** Environment mode 'grid': the environment grid atlas (last cascade), its tile columns and the unshadowed light. */
  environmentGrid?: { atlas: VDBFrameAtlas; columns: number; unshadowed: Vec3 } | null;
}

/**
 * The density pass of 'frame' lighting at one froxel of one volume (FrameDensityMaterial `buildNodes`,
 * grey, octaves off, no environment term): the slice's sub-step pre-integration through the source,
 * the visible position, and the shadow slots there. Returns the 4 MRT outputs (before additive
 * blending), or null when the froxel is discarded. `fragCoord` is the atlas pixel (bottom-left).
 */
export function vdbFrameDensityFroxel(fragCoord: Vec2, p: VDBFrameDensityInputs): [Vec4, Vec4, Vec4, Vec4] | null {
  const froxel = vdbFrameFroxel(p.froxelGrid, p.columns, p.range, p.projectionInverse, fragCoord);
  const mul = (m: ArrayLike<number>, v: Vec3): Vec3 => [0, 1, 2].map((r) => m[r]! * v[0] + m[4 + r]! * v[1] + m[8 + r]! * v[2] + m[12 + r]!) as Vec3;
  const { viewNear, viewFar } = froxel;
  const localNear = mul(p.viewToLocal, viewNear);
  const localFar = mul(p.viewToLocal, viewFar);
  // boxSegment of the unit box
  const dir: Vec3 = [localFar[0] - localNear[0], localFar[1] - localNear[1], localFar[2] - localNear[2]];
  let t0 = 0;
  let t1 = 1;
  {
    let lo = -Infinity;
    let hi = Infinity;
    for (let a = 0; a < 3; a++) {
      const inv = 1 / dir[a]!;
      const ta = (-0.5 - localNear[a]!) * inv;
      const tb = (0.5 - localNear[a]!) * inv;
      lo = Math.max(lo, Math.min(ta, tb));
      hi = Math.min(hi, Math.max(ta, tb));
    }
    t0 = Math.max(0, lo);
    t1 = Math.min(1, hi);
  }
  if (t0 >= t1) return null;
  const segment = t1 - t0;
  const getLod = (size: number) => Math.max(0, Math.log2(size * p.maxTexels));
  const sliceMiddle = (froxel.sliceNear + froxel.sliceFar) * 0.5;
  const a = vdbFrameViewPosition(p.projectionInverse, [froxel.uv[0] + 1 / p.froxelGrid[0], froxel.uv[1]], sliceMiddle);
  const b = vdbFrameViewPosition(p.projectionInverse, froxel.uv, sliceMiddle);
  const widthLod = getLod(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / p.worldScale);
  const sampledLod = p.mipLevels > 0 ? Math.min(widthLod, p.mipLevels) : 0;
  const boxLength = Math.hypot(...dir) * segment;
  const steps = Math.trunc(Math.min(Math.max(Math.ceil(Math.pow(2, getLod(boxLength) - sampledLod) * 2), 1), p.subSteps));
  const subStepLength = (Math.hypot(viewFar[0] - viewNear[0], viewFar[1] - viewNear[1], viewFar[2] - viewNear[2]) * segment) / steps;
  const lod = Math.max(widthLod, getLod(subStepLength / p.worldScale));
  const jitter = vdbFrameNoise(p.jitter, fragCoord);
  const lerp3 = (x: Vec3, y: Vec3, t: number): Vec3 => [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t];
  let extinctionSum = 0;
  let transmittance = 1;
  let visibleWeight = 0;
  let visibleTSum = 0;
  for (let i = 0; i < steps; i++) {
    const t = t0 + segment * ((i + jitter) / steps);
    const e = p.extinctionAt(lerp3(localNear, localFar, t), lod) * p.opacity * vdbFrameDistanceFade(p.fadeRadius, lerp3(viewNear, viewFar, t));
    const stepT = Math.exp(-e * subStepLength);
    const w = transmittance * (1 - stepT);
    visibleWeight += w;
    visibleTSum += w * t;
    extinctionSum += e;
    transmittance *= stepT;
  }
  const extinction = (extinctionSum / steps) * segment;
  if (!(extinction > 0)) return null;
  const visibleT = visibleWeight > 0 ? visibleTSum / visibleWeight : 0.5;
  const visibleWorld = mul(p.cameraWorld, lerp3(viewNear, viewFar, visibleT));
  const sh = p.shadows ? vdbFrameSampleShadows(p.shadows.atlases, p.shadows.grid, visibleWorld) : [1, 2, 3, 4].map(() => [1, 1, 1, 1]);
  const shadow0 = [...sh[0]!];
  const shadow1 = [...sh[1]!];
  const directs = [sh[2]![0]!, sh[2]![1]!, sh[2]![2]!, sh[2]![3]!, sh[3]![0]!, sh[3]![1]!];
  const m = p.cameraWorld;
  const ray = normalize3([0, 1, 2].map((r) => m[r]! * (viewFar[0] - viewNear[0]) + m[4 + r]! * (viewFar[1] - viewNear[1]) + m[8 + r]! * (viewFar[2] - viewNear[2])) as Vec3);
  const slotRef: [number[], number][] = [
    [shadow0, 2],
    [shadow0, 3],
    [shadow1, 0],
    [shadow1, 1],
    [shadow1, 2],
    [shadow1, 3],
  ];
  slotRef.forEach(([s, c], index) => {
    if (index < p.sceneLightCount) {
      const l = p.lightPositions[index]!;
      const lightDir = l[3] < 1.5 ? ([l[0], l[1], l[2]] as Vec3) : normalize3([l[0] - visibleWorld[0], l[1] - visibleWorld[1], l[2] - visibleWorld[2]]);
      const direct = directs[index]!;
      const phase = vdbFramePhaseWeight(p.phase, ray, lightDir);
      s[c] = phase * direct + (s[c]! - direct);
    }
  });
  const ext = (v: number[]) => v.map((x) => Math.min(x * extinction, 6.0e4)) as Vec4;
  const emission: Vec3 = [...p.scatterColor];
  if (p.environmentGrid && p.shadows) {
    // sampleEnvironmentGrid: the last cascade, its own atlas; the unshadowed light outside it
    const g = p.shadows.grid;
    const uvw = vdbFrameGridUvw(g, g.count - 1, visibleWorld);
    const env = vdbFrameInsideGrid(g.resolution, uvw) ? vdbFrameSampleTiles(p.environmentGrid.atlas, g.resolution, 0, p.environmentGrid.columns, uvw) : p.environmentGrid.unshadowed;
    for (let c = 0; c < 3; c++) emission[c] = emission[c]! + (p.baseColor[c]! * env[c]!) / Math.PI;
  }
  return [
    [p.baseColor[0] * extinction, p.baseColor[1] * extinction, p.baseColor[2] * extinction, extinction],
    [emission[0] * extinction, emission[1] * extinction, emission[2] * extinction, visibleT * extinction],
    ext(shadow0),
    ext(shadow1),
  ];
}

/**
 * The environment grid pass (FrameEnvironmentGridMaterial, grey, octaves off) at one voxel of the last
 * cascade: `fragCoord` is the pixel of the environment grid atlas (`envColumns` tile columns), `lights`
 * the environment lights without a shadow slot, `unshadowed` their colour sum (the value outside the
 * casters). One-voxel steps that stay in the last cascade; with diffusion the two-sided total of the
 * march and the opposite march. Returns the RGBA output.
 */
export function vdbFrameEnvironmentGridVoxel(
  atlas: VDBFrameAtlas,
  grid: VDBFrameGridLayout,
  envColumns: number,
  fragCoord: Vec2,
  lights: readonly { direction: Vec3; color: Vec3 }[],
  unshadowed: Vec3,
  multipleScattering: number,
): Vec4 {
  const cascade = grid.count - 1;
  const res = grid.resolution;
  const tile: Vec2 = [Math.floor(fragCoord[0] / res), Math.floor(fragCoord[1] / res)];
  const z = tile[1] * envColumns + tile[0];
  const uvw0: Vec3 = [(fragCoord[0] - tile[0] * res) / res, (fragCoord[1] - tile[1] * res) / res, (z + 0.5) / res];
  const voxel = grid.gridVoxel[cascade]!;
  const m = grid.gridMin[cascade]!;
  const s = grid.gridSize[cascade]!;
  const world: Vec3 = [m[0] + uvw0[0] * s[0], m[1] + uvw0[1] * s[1], m[2] + uvw0[2] * s[2]];
  const uvwOf = (p: Vec3): Vec3 => [(p[0] - m[0]) / s[0], (p[1] - m[1]) / s[1], (p[2] - m[2]) / s[2]];
  const sample = (p: Vec3) => vdbFrameSampleGrid(atlas, grid, cascade, uvwOf(p));
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    near += sample([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]])[0]!;
  }
  if (!(z < res && near > 0)) return [unshadowed[0], unshadowed[1], unshadowed[2], 1];
  const marchDepths = (dir: Vec3): Vec3 => {
    const perUnit = Math.max(Math.abs(dir[0]) / voxel[0], Math.abs(dir[1]) / voxel[1], Math.abs(dir[2]) / voxel[2]);
    const step = 1 / perUnit;
    const probe: Vec3 = [world[0] + dir[0] * step * 0.5, world[1] + dir[1] * step * 0.5, world[2] + dir[2] * step * 0.5];
    const depths: Vec3 = [0, 0, 0];
    for (let i = 0; i < 1024; i++) {
      const uvw = uvwOf(probe);
      if (uvw.some((v) => v < 0 || v > 1) || vdbFrameShadowDepth(depths, multipleScattering) > 6) break;
      const v = vdbFrameSampleGrid(atlas, grid, cascade, uvw);
      const d = vdbFrameShadowStep(v[0]!, v[1]!, multipleScattering);
      for (let c = 0; c < 3; c++) {
        depths[c] = depths[c]! + d[c]! * step;
        probe[c] = probe[c]! + dir[c]! * step;
      }
    }
    return depths;
  };
  const irradiance: Vec3 = [0, 0, 0];
  for (const light of lights) {
    const depths = marchDepths(light.direction);
    const total =
      multipleScattering > 0
        ? vdbFrameIsotropicDiffusion(depths[2], marchDepths([-light.direction[0], -light.direction[1], -light.direction[2]])[2], depths[1])
        : vdbFrameShadowTransmittance(depths, multipleScattering)[1];
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + light.color[c]! * total;
  }
  return [Math.max(irradiance[0], 0), Math.max(irradiance[1], 0), Math.max(irradiance[2], 0), 1];
}

/**
 * The temporal pass reprojection of a froxel centre (FrameTemporalMaterial L123–157): the previous
 * frame's uv and fractional slice of a world position, and the history blend (the motion of a
 * quarter cell or more uses only the current froxel). Returns null when the point is outside the
 * previous froxel grid (or behind its camera).
 */
export function vdbFrameReproject(
  world: Vec3,
  previousView: ArrayLike<number>,
  previousProjection: ArrayLike<number>,
  previousRange: Vec2,
  grid: Vec3,
  current: { uv: Vec2; slice: number },
  temporalBlend: number,
): { uv: Vec2; slice: number; blend: number } | null {
  const mul = (m: ArrayLike<number>, v: Vec4): Vec4 => {
    const o: Vec4 = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) o[r] = m[r]! * v[0] + m[4 + r]! * v[1] + m[8 + r]! * v[2] + m[12 + r]! * v[3];
    return o;
  };
  const view = mul(previousView, [world[0], world[1], world[2], 1]);
  const clip = mul(previousProjection, view);
  if (!(clip[3] > 0)) return null;
  const uv: Vec2 = [(clip[0] / clip[3]) * 0.5 + 0.5, (clip[1] / clip[3]) * 0.5 + 0.5];
  const t = Math.log(Math.max(-view[2], 1e-6) / previousRange[0]) / Math.log(previousRange[1] / previousRange[0]);
  const slice = t * grid[2] - 0.5;
  const inside = Math.min(uv[0], uv[1]) >= 0 && Math.max(uv[0], uv[1]) <= 1 && slice >= -0.5 && slice <= grid[2] - 0.5;
  if (!inside) return null;
  const motion = Math.hypot((uv[0] - current.uv[0]) * grid[0], (uv[1] - current.uv[1]) * grid[1], slice - current.slice);
  const blend = temporalBlend + (1 - temporalBlend) * Math.min(Math.max(motion * 4, 0), 1);
  return { uv, slice, blend };
}

/** The lights {@link vdbFrameLightingFroxel} reads (the renderer's light uniforms; light probe off, grey phase). */
export interface VDBFrameLightingInputs {
  ambient: Vec3;
  hemiSky: Vec3;
  hemiGround: Vec3;
  /** Scene lights in slot order: position (w 1 directional, 2 point, 3 spot), colour × intensity; spot (xyz, cos) and penumbra cos; falloff (distance, decay). */
  lights: { position: Vec4; color: Vec3; spot?: Vec4; penumbra?: number; falloff?: Vec2 }[];
  shadowLightCount: number;
  /** The renderer phase (forward g, back g, back weight, Mie mode). */
  phase: Vec4;
  /** Environment lights with a shadow slot (colours), from `environmentShadowSlot`. */
  environmentColors: Vec3[];
  environmentShadowSlot: number;
}

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

/** three's `getDistanceAttenuation` (Frostbite window). */
const distanceAttenuation = (d: number, cutoff: number, decay: number) => {
  const falloff = 1 / Math.max(Math.pow(d, decay), 0.01);
  return cutoff > 0 ? falloff * Math.pow(Math.min(Math.max(1 - Math.pow(d / cutoff, 4), 0), 1), 2) : falloff;
};

/**
 * The lighting pass of 'frame' lighting at one froxel (FrameLightingMaterial `createFragmentNode`,
 * grey, octaves off, light probe off, one phase for all channels): from the 4 density targets
 * (albedo·σ, σ), (emission·σ, t·σ), (sky up, sky down, slot 0, slot 1)·σ, (slots 2–5)·σ, the froxel's
 * view segment and the camera matrix (column-major). Returns the premultiplied (radiance·α, α), or
 * zeros when σ ≤ 0 (the pass discards; the target is cleared).
 */
export function vdbFrameLightingFroxel(
  density: readonly [Vec4, Vec4, Vec4, Vec4],
  viewNear: Vec3,
  viewFar: Vec3,
  cameraWorld: ArrayLike<number>,
  inputs: VDBFrameLightingInputs,
): Vec4 {
  const [albedoT, emissionT, shadow0T, shadow1T] = density;
  const extinction = albedoT[3];
  if (!(extinction > 0)) return [0, 0, 0, 0];
  const m = cameraWorld;
  const visibleT = Math.min(Math.max(emissionT[3] / extinction, 0), 1);
  const view: Vec3 = [viewNear[0] + (viewFar[0] - viewNear[0]) * visibleT, viewNear[1] + (viewFar[1] - viewNear[1]) * visibleT, viewNear[2] + (viewFar[2] - viewNear[2]) * visibleT];
  const world: Vec3 = [0, 1, 2].map((r) => m[r]! * view[0] + m[4 + r]! * view[1] + m[8 + r]! * view[2] + m[12 + r]!) as Vec3;
  const seg: Vec3 = [viewFar[0] - viewNear[0], viewFar[1] - viewNear[1], viewFar[2] - viewNear[2]];
  const ray = normalize3([0, 1, 2].map((r) => m[r]! * seg[0] + m[4 + r]! * seg[1] + m[8 + r]! * seg[2]) as Vec3);
  const s0 = shadow0T.map((v) => v / extinction);
  const s1 = shadow1T.map((v) => v / extinction);
  const slots = [s0[2]!, s0[3]!, s1[0]!, s1[1]!, s1[2]!, s1[3]!];
  const skyUp = s0[0]! * 0.5;
  const skyDown = s0[1]! * 0.5;
  const irradiance: Vec3 = [0, 1, 2].map((c) => inputs.ambient[c]! + inputs.hemiSky[c]! * skyUp + inputs.hemiGround[c]! * skyDown) as Vec3;
  const hg = (cos: number, g: number) => (1 - g * g) / Math.pow(Math.max(1 + g * g - 2 * g * cos, 1e-4), 1.5);
  inputs.lights.forEach((light, i) => {
    const p = light.position;
    const toLight: Vec3 = [p[0] - world[0], p[1] - world[1], p[2] - world[2]];
    const lightDistance = Math.hypot(...toLight);
    const dir = p[3] < 1.5 ? ([p[0], p[1], p[2]] as Vec3) : normalize3(toLight);
    const spot = p[3] > 2.5 && light.spot ? smoothstep(light.spot[3], light.penumbra ?? 1, normalize3(toLight)[0] * light.spot[0] + normalize3(toLight)[1] * light.spot[1] + normalize3(toLight)[2] * light.spot[2]) : 1;
    const dist = p[3] > 1.5 ? distanceAttenuation(lightDistance, light.falloff?.[0] ?? 0, light.falloff?.[1] ?? 0) : 1;
    const cos = ray[0] * dir[0] + ray[1] * dir[1] + ray[2] * dir[2];
    const a = hg(cos, inputs.phase[0]);
    const unshadowed = a + (hg(cos, inputs.phase[1]) - a) * inputs.phase[2];
    const factor = i < inputs.shadowLightCount ? (i < 6 ? slots[i]! : 1) : unshadowed;
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + light.color[c]! * factor * spot * dist;
  });
  inputs.environmentColors.forEach((color, i) => {
    const slot = inputs.environmentShadowSlot + i;
    const shadow = slot < 6 ? slots[slot]! : 1;
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + color[c]! * shadow;
  });
  const alpha = 1 - Math.exp(-extinction * Math.hypot(...seg));
  const out: Vec4 = [0, 0, 0, alpha];
  for (let c = 0; c < 3; c++) {
    const radiance = (albedoT[c]! / extinction) * irradiance[c]! * (1 / Math.PI) + emissionT[c]! / extinction;
    out[c] = radiance * alpha;
  }
  return out;
}
