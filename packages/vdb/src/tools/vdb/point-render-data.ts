/**
 * The render-ready point list of a point data grid (K3.7): world positions,
 * optional `Cd` colours, a subsample stride above `maxPoints`, and the mean
 * point spacing that sizes the sprites. The input `VDBPoints` (`./three`)
 * draws.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/openvdb/utils/points.js` `createPointData`: the stride rule, the count,
 * the spacing formula). Copyright (c) mjurczyk/openvdb contributors, MIT (see
 * THIRD_PARTY_NOTICES.md). Changes: built on our parsed `VdbPointDataGrid.points()`
 * (K4.2) in one pass instead of a sliced generator; `Cd` of any decoded
 * 3-component real attribute (DIVERGES.md).
 */
import { VDBToolsError } from "../errors.js";
import type { VdbPointDataGrid } from "./types.js";

/** What `VDBPoints` draws: world positions (xyz per point), optional linear RGB, the mean spacing. */
export interface VDBPointRenderData {
  /** Points in the arrays. */
  count: number;
  /** Points in the grid before the stride. */
  totalCount: number;
  /** Every `stride`-th point (in the grid's point order) is kept; 1 below `maxPoints`. */
  stride: number;
  /** Mean distance between the kept points in world units: `voxelSize / cbrt(points per active voxel) * cbrt(stride)`. */
  spacing: number;
  /** World positions, `count * 3`. */
  positions: Float32Array;
  /** `Cd` per point (`count * 3`), or null when the grid has no 3-component real `Cd`. */
  colors: Float32Array | null;
}

/** Options of {@link vdbPointRenderData}. */
export interface VDBPointRenderDataOptions {
  /** Above this many points every `ceil(total / maxPoints)`-th point is kept (his default 4,000,000). */
  maxPoints?: number;
  /** Colour attribute name. Default `'Cd'`. */
  color?: string;
}

/** His `defaultMaxPoints`. */
export const VDB_POINTS_DEFAULT_MAX = 4_000_000;

/**
 * Builds the {@link VDBPointRenderData} of a parsed point data grid
 * (`parseVdb(...).grids` entry with `valueType: 'ptdataidx32'`). Decodes the
 * grid's points (`grid.points()`, cached by the grid).
 */
export function vdbPointRenderData(grid: VdbPointDataGrid, options: VDBPointRenderDataOptions = {}): VDBPointRenderData {
  if (grid?.valueType !== "ptdataidx32" || typeof grid.points !== "function") {
    throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", "vdbPointRenderData: grid must be a parsed point data grid (valueType 'ptdataidx32').");
  }
  const maxPoints = options.maxPoints ?? VDB_POINTS_DEFAULT_MAX;
  if (!(maxPoints >= 1)) throw new VDBToolsError("VDB_BAD_OPTION", `vdbPointRenderData: maxPoints must be >= 1, got ${maxPoints}.`);
  const list = grid.points();
  const totalCount = list.count;
  const stride = Math.max(1, Math.ceil(totalCount / maxPoints));
  const count = totalCount > 0 ? Math.floor((totalCount - 1) / stride) + 1 : 0;
  const positions = new Float32Array(count * 3);
  const name = options.color ?? "Cd";
  const cd = list.attributes[name];
  const shape = list.attributeShapes[name];
  const hasColors = cd !== undefined && shape?.components === 3 && shape.stride === 1 && (cd instanceof Float32Array || cd instanceof Float64Array);
  const colors = hasColors ? new Float32Array(count * 3) : null;
  for (let i = 0, j = 0; i < totalCount; i += stride, j++) {
    positions[j * 3] = list.positions[i * 3]!;
    positions[j * 3 + 1] = list.positions[i * 3 + 1]!;
    positions[j * 3 + 2] = list.positions[i * 3 + 2]!;
    if (colors) {
      const c = cd as Float32Array | Float64Array;
      colors[j * 3] = c[i * 3]!;
      colors[j * 3 + 1] = c[i * 3 + 1]!;
      colors[j * 3 + 2] = c[i * 3 + 2]!;
    }
  }
  const [sx, sy, sz] = grid.transform.voxelSize;
  const voxelSize = Math.cbrt(Math.abs(sx * sy * sz));
  const activeVoxels = Number(grid.activeVoxelCount);
  const pointsPerVoxel = activeVoxels > 0 ? totalCount / activeVoxels : 1;
  const spacing = (voxelSize / Math.cbrt(pointsPerVoxel)) * Math.cbrt(stride);
  return { count, totalCount, stride, spacing, positions, colors };
}
