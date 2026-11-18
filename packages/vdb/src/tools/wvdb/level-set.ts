/**
 * level-set.ts — mjurczyk's level-set bake sampler (`sampleLevelSet`), an
 * option of the `.wvdb` bake (`levelSetSampler: 'surface-mean'`, K4.13).
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345:
 * `src/openvdb/utils/levelSet.js` (`sampleLevelSet`, `getVoxelDistance`,
 * `getTrilinearDistance`) and `src/wvdb/bake.js` (`sampleLevelSetGrid`, the
 * band limit). His arithmetic order is kept so the output is bit-identical.
 *
 * What it does (his notes, restated): a level set's narrow band is often
 * thinner than a texel, so a point sample is only inside or outside and the
 * surface gets steps. When texels are larger than a voxel, a texel whose eight
 * corners do not all have the same sign (it crosses the surface) is the mean
 * of 4x4x4 sub-samples inside it; every other texel is a trilinear sample at
 * its centre (only near the surface; far voxels keep the nearest value). A
 * voxel's distance is its value clamped to `±background` when active, and
 * `±background` by sign when inactive. Finally, values inside the band are
 * clamped one quantization step inside it (`background - 2·background /
 * (2^bits - 1)`) so they never decode as outside.
 *
 * The spec text (§8) instead point-samples and encodes by the source's active
 * state; that stays the default (`levelSetSampler: 'spec'`, DIVERGES V17).
 */

import { VDBToolsError } from "../errors.js";
import type { WvdbBits, WvdbBox, WvdbVec3 } from "./format.js";

/** Index-space access to a sample source (axis-aligned or affine map). */
export interface WvdbIndexAccess {
  /** World position → fractional index coordinate (the grid's inverse map), into `out`. */
  toIndex(world: WvdbVec3, out: WvdbVec3): WvdbVec3;
  /** A world-space vector → index units (the inverse Jacobian), into `out`. */
  toIndexVector(vector: WvdbVec3, out: WvdbVec3): WvdbVec3;
  /** Value and active state of an integer voxel (inactive tile values included). */
  voxel(ijk: [number, number, number]): { value: number; active: boolean };
}

/** The part of a sample source this sampler reads. */
export interface WvdbLevelSetSource {
  name: string;
  background: number;
  bbox: WvdbBox | null;
  index?: WvdbIndexAccess;
}

/** Sub-samples on each axis for texels that cross the surface (his `filterSamples`). */
const FILTER_SAMPLES = 4;

/**
 * His `sampleLevelSetGrid`: samples a level set over `bbox` at `dimensions`
 * (x fastest) and applies the band limit for `bits`.
 */
export function sampleWvdbLevelSetSurfaceMean(
  source: WvdbLevelSetSource,
  bbox: WvdbBox | null,
  dimensions: WvdbVec3,
  bits: WvdbBits = 8,
): Float32Array {
  const [width, height, depth] = dimensions;
  const values = new Float32Array(width * height * depth);
  const background = source.background;
  if (!source.bbox || !bbox) return values.fill(background);
  const access = source.index;
  if (!access) {
    throw new VDBToolsError("VDB_BAD_OPTION", `levelSetSampler 'surface-mean': source "${source.name}" has no index access (use wvdbSourceFromVdbGrid or wvdbSourceFromNvdb)`);
  }
  const min = bbox[0];
  const size: WvdbVec3 = [bbox[1][0] - min[0], bbox[1][1] - min[1], bbox[1][2] - min[2]];

  const world: WvdbVec3 = [0, 0, 0];
  const index: WvdbVec3 = [0, 0, 0];
  const rounded: [number, number, number] = [0, 0, 0];
  const corner: WvdbVec3 = [0, 0, 0];

  const toIndex = (x: number, y: number, z: number): WvdbVec3 => {
    world[0] = min[0] + (x / width) * size[0];
    world[1] = min[1] + (y / height) * size[1];
    world[2] = min[2] + (z / depth) * size[2];
    return access.toIndex(world, index);
  };

  // His grid.getValue / isActive round the position (Vector3.round: Math.round per axis).
  const lookup = (p: WvdbVec3): { value: number; active: boolean } => {
    rounded[0] = Math.round(p[0]);
    rounded[1] = Math.round(p[1]);
    rounded[2] = Math.round(p[2]);
    return access.voxel(rounded);
  };

  // getVoxelDistance: inactive voxels are only outside (+b) or inside (-b); values clamp to the band.
  const voxelDistance = (p: WvdbVec3): number => {
    const { value, active } = lookup(p);
    if (!Number.isFinite(value)) return background;
    if (!active) return value < 0.0 ? -background : background;
    return Math.min(Math.max(value, -background), background);
  };

  // getTrilinearDistance: trilinear only near the surface; far voxels keep the nearest value.
  const trilinearDistance = (p: WvdbVec3): number => {
    const nearest = voxelDistance(p);
    if (Math.abs(nearest) >= background) return nearest;
    const x0 = Math.floor(p[0]);
    const y0 = Math.floor(p[1]);
    const z0 = Math.floor(p[2]);
    const wx = p[0] - x0;
    const wy = p[1] - y0;
    const wz = p[2] - z0;
    let value = 0.0;
    for (let i = 0; i < 8; i++) {
      const dx = i & 1;
      const dy = (i >> 1) & 1;
      const dz = (i >> 2) & 1;
      const weight = (dx ? wx : 1.0 - wx) * (dy ? wy : 1.0 - wy) * (dz ? wz : 1.0 - wz);
      if (weight > 0.0) {
        corner[0] = x0 + dx;
        corner[1] = y0 + dy;
        corner[2] = z0 + dz;
        value += weight * voxelDistance(corner);
      }
    }
    return value;
  };

  // Texel size in voxels.
  const texel = access.toIndexVector([size[0] / width, size[1] / height, size[2] / depth], [0, 0, 0]);
  const filter = Math.max(Math.abs(texel[0]), Math.abs(texel[1]), Math.abs(texel[2])) > 1.0;

  // Signs at texel corners, one corner slice for each side of the texel row.
  const cornersX = width + 1;
  const cornerSlice = cornersX * (height + 1);
  let lowerSigns = new Uint8Array(filter ? cornerSlice : 0);
  let upperSigns = new Uint8Array(filter ? cornerSlice : 0);

  const fillSigns = (signs: Uint8Array, z: number): void => {
    for (let y = 0; y <= height; y++) {
      for (let x = 0; x <= width; x++) {
        signs[x + y * cornersX] = lookup(toIndex(x, y, z)).value < 0.0 ? 1 : 0;
      }
    }
  };

  const crossesSurface = (x: number, y: number): boolean => {
    const first = lowerSigns[x + y * cornersX];
    for (let i = 1; i < 8; i++) {
      const signs = (i >> 2) & 1 ? upperSigns : lowerSigns;
      if (signs[x + (i & 1) + (y + ((i >> 1) & 1)) * cornersX] !== first) return true;
    }
    return false;
  };

  if (filter) fillSigns(lowerSigns, 0);
  const subSamples = FILTER_SAMPLES * FILTER_SAMPLES * FILTER_SAMPLES;
  let offset = 0;
  for (let z = 0; z < depth; z++) {
    if (filter) fillSigns(upperSigns, z + 1);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++, offset++) {
        if (filter && crossesSurface(x, y)) {
          let sum = 0.0;
          for (let i = 0; i < subSamples; i++) {
            const sx = (i % FILTER_SAMPLES) + 0.5;
            const sy = (Math.floor(i / FILTER_SAMPLES) % FILTER_SAMPLES) + 0.5;
            const sz = Math.floor(i / (FILTER_SAMPLES * FILTER_SAMPLES)) + 0.5;
            sum += voxelDistance(toIndex(x + sx / FILTER_SAMPLES, y + sy / FILTER_SAMPLES, z + sz / FILTER_SAMPLES));
          }
          values[offset] = sum / subSamples;
        } else {
          values[offset] = trilinearDistance(toIndex(x + 0.5, y + 0.5, z + 0.5));
        }
      }
    }
    [lowerSigns, upperSigns] = [upperSigns, lowerSigns];
  }

  // Values in the band stay one quantization step inside it, so they never decode as outside.
  const bandLimit = background - (2.0 * background) / ((1 << bits) - 1);
  for (let i = 0; i < values.length; i++) {
    if (Math.abs(values[i]!) < background) values[i] = Math.min(Math.max(values[i]!, -bandLimit), bandLimit);
  }
  return values;
}
