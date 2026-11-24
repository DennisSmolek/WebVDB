/**
 * Pure-TypeScript CPU reference for continuous index-space trilinear
 * sampling of a NanoVDB grid image. Built directly on top of `readValue`
 * (`./read-value.ts`) — it does no traversal of its own, it just taps eight
 * neighboring integer voxels and blends them.
 *
 * This exists for the same reason `read-value.ts` does: to de-risk the WGSL
 * sampling math ahead of the Phase 2 GPU test suite. This module and the
 * eventual `pnanovdb_sample_trilinear` in `pnanovdb.wgsl` must agree, point
 * for point.
 *
 * ---------------------------------------------------------------------------
 * Convention (must be copied verbatim by the WGSL port):
 * ---------------------------------------------------------------------------
 * Voxels are centered at integer index coordinates — there is no half-voxel
 * offset between a voxel's `(i, j, k)` and its sample location. This is the
 * same convention used by the vendored NanoVDB reference implementation
 * (see `fixtures/downloads/openvdb-sparse/nanovdb/nanovdb/math/SampleFromVoxels.h`,
 * the `Floor()` helper and `TrilinearSampler`): given a continuous index-space
 * position `xyz`, the base voxel is `base = floor(xyz)` component-wise, and
 * the fractional remainder is `f = xyz - base`, with each component of `f` in
 * `[0, 1)`.
 *
 * The eight taps are `vdbReadValueCpu(base + [dx, dy, dz])` for `dx, dy, dz` each
 * in `{0, 1}` — i.e. the voxel at `base` and its unit-offset neighbors toward
 * `+x, +y, +z`. Each tap contributes its stored/background VALUE regardless
 * of active state: an inactive (background) voxel still participates in the
 * blend using its background value, exactly like a GPU texture sampler would
 * (it has no notion of "active" — it just reads whatever is in the texel).
 * This is a deliberate divergence from `readValue`'s `{ value, active }`
 * pair: `vdbSampleTrilinear` only ever returns a value.
 *
 * Interpolation order (numerically order-independent, but WGSL should match
 * this exact structure so intermediate rounding lines up under float32):
 * lerp along x first (four edge lerps), then y (two face lerps), then z
 * (the final lerp):
 *
 *   c00 = lerp(v(0,0,0), v(1,0,0), fx)   // along x, at y=0, z=0
 *   c10 = lerp(v(0,1,0), v(1,1,0), fx)   // along x, at y=1, z=0
 *   c01 = lerp(v(0,0,1), v(1,0,1), fx)   // along x, at y=0, z=1
 *   c11 = lerp(v(0,1,1), v(1,1,1), fx)   // along x, at y=1, z=1
 *   c0  = lerp(c00, c10, fy)             // along y, at z=0
 *   c1  = lerp(c01, c11, fy)             // along y, at z=1
 *   result = lerp(c0, c1, fz)            // along z
 *
 * where `lerp(a, b, t) = a + (b - a) * t`.
 */
import { VDBToolsError } from "../errors.js";
import { indexBBox } from "./march.js";
import { vdbReadValueCpu, vdbReadValueVec3 } from "./read-value.js";
import { defineNumber } from "./stride-tables.js";

type Coord = readonly [number, number, number];

/**
 * CPU twins of the generator's `outside` option (K1-core §3.1).
 * - `'background'` (default): read the tree everywhere (NanoVDB).
 * - `'zero'`: sample the field that equals the tree inside the index bbox
 *   `[min, max]` (voxels, inclusive, the root bbox = the pool record's
 *   `indexBBoxMin/Max`) and 0 outside it. Trilinear: each tap outside the box
 *   reads 0, so the sample is 0 unless `xyz` is in `[min - 1, max + 1)`.
 *   Nearest: 0 unless `floor(xyz + 0.5)` is in the box, i.e. `xyz` in
 *   `[min - 0.5, max + 0.5)`.
 */
export interface VDBSampleCpuOptions {
  /** What lies outside the root index bbox: the tree's background (default) or zero. */
  outside?: "background" | "zero";
}

/** Root index bbox of any grid type (the root block's bbox is type-independent). */
function rootIndexBBox(gridWords: Uint32Array): { min: Coord; max: Coord } {
  const gridSize = defineNumber("PNANOVDB_GRID_SIZE");
  const root = gridSize + gridWords[(gridSize + defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT")) >>> 2]!;
  const at = (name: string): Coord => {
    const w = (root + defineNumber(name)) >>> 2;
    return [gridWords[w]! | 0, gridWords[w + 1]! | 0, gridWords[w + 2]! | 0];
  };
  return { min: at("PNANOVDB_ROOT_OFF_BBOX_MIN"), max: at("PNANOVDB_ROOT_OFF_BBOX_MAX") };
}

function zeroOutside(
  gridWords: Uint32Array,
  options: VDBSampleCpuOptions | undefined,
  bbox: (words: Uint32Array) => { min: Coord; max: Coord } = indexBBox,
): ((c: Coord) => boolean) | null {
  if ((options?.outside ?? "background") === "background") return null;
  if (options?.outside !== "zero") throw new VDBToolsError("VDB_BAD_OPTION", `sample: outside must be 'background' or 'zero', got ${String(options?.outside)}`);
  const { min, max } = bbox(gridWords);
  return (c) => c[0] < min[0] || c[1] < min[1] || c[2] < min[2] || c[0] > max[0] || c[1] > max[1] || c[2] > max[2];
}

/**
 * Nearest-voxel sample (twin of `vdbSample` with `sampler: 'nearest'`):
 * the voxel `floor(xyz + 0.5)` (NanoVDB `SampleFromVoxels` order 0 rounds).
 *
 * @param gridWords The grid image (little-endian u32 words, `GridData` at word 0).
 * @param xyz Position in index space.
 * @param options The `outside` mode.
 */
export function vdbSampleNearest(gridWords: Uint32Array, xyz: [number, number, number], options?: VDBSampleCpuOptions): number {
  const ijk: [number, number, number] = [Math.floor(xyz[0] + 0.5), Math.floor(xyz[1] + 0.5), Math.floor(xyz[2] + 0.5)];
  const outside = zeroOutside(gridWords, options);
  if (outside?.(ijk)) return 0;
  return vdbReadValueCpu(gridWords, ijk).value;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Continuous index-space trilinear sample. `xyz` is in the same index space
 * as the integer coords passed to `readValue` (i.e. NOT world space — no
 * voxel-size or grid-transform scaling is applied here).
 *
 * @param gridWords The grid image (little-endian u32 words, `GridData` at word 0).
 * @param xyz Position in index space.
 * @param options The `outside` mode.
 */
export function vdbSampleTrilinear(gridWords: Uint32Array, xyz: [number, number, number], options?: VDBSampleCpuOptions): number {
  const outside = zeroOutside(gridWords, options);
  const bx = Math.floor(xyz[0]);
  const by = Math.floor(xyz[1]);
  const bz = Math.floor(xyz[2]);
  const fx = xyz[0] - bx;
  const fy = xyz[1] - by;
  const fz = xyz[2] - bz;

  const tap = (dx: 0 | 1, dy: 0 | 1, dz: 0 | 1): number => {
    const coord: Coord = [bx + dx, by + dy, bz + dz];
    if (outside?.(coord)) return 0;
    return vdbReadValueCpu(gridWords, coord as [number, number, number]).value;
  };

  const v000 = tap(0, 0, 0);
  const v100 = tap(1, 0, 0);
  const v010 = tap(0, 1, 0);
  const v110 = tap(1, 1, 0);
  const v001 = tap(0, 0, 1);
  const v101 = tap(1, 0, 1);
  const v011 = tap(0, 1, 1);
  const v111 = tap(1, 1, 1);

  const c00 = lerp(v000, v100, fx);
  const c10 = lerp(v010, v110, fx);
  const c01 = lerp(v001, v101, fx);
  const c11 = lerp(v011, v111, fx);

  const c0 = lerp(c00, c10, fy);
  const c1 = lerp(c01, c11, fy);

  return lerp(c0, c1, fz);
}

/**
 * Componentwise trilinear sample of a VEC3F grid image: the same eight taps,
 * lerp order and `outside` handling as {@link vdbSampleTrilinear}, applied to x,
 * y and z independently (NanoVDB `TrilinearSampler` on a `Vec3f` grid).
 *
 * @param gridWords The VEC3F grid image (little-endian u32 words, `GridData` at word 0).
 * @param xyz Position in index space.
 * @param options The `outside` mode.
 */
export function vdbSampleTrilinearVec3(
  gridWords: Uint32Array,
  xyz: [number, number, number],
  options?: VDBSampleCpuOptions,
): [number, number, number] {
  const outside = zeroOutside(gridWords, options, rootIndexBBox);
  const bx = Math.floor(xyz[0]);
  const by = Math.floor(xyz[1]);
  const bz = Math.floor(xyz[2]);
  const fx = xyz[0] - bx;
  const fy = xyz[1] - by;
  const fz = xyz[2] - bz;

  const tap = (dx: 0 | 1, dy: 0 | 1, dz: 0 | 1): [number, number, number] => {
    const coord: [number, number, number] = [bx + dx, by + dy, bz + dz];
    if (outside?.(coord)) return [0, 0, 0];
    return vdbReadValueVec3(gridWords, coord).value;
  };

  const v000 = tap(0, 0, 0);
  const v100 = tap(1, 0, 0);
  const v010 = tap(0, 1, 0);
  const v110 = tap(1, 1, 0);
  const v001 = tap(0, 0, 1);
  const v101 = tap(1, 0, 1);
  const v011 = tap(0, 1, 1);
  const v111 = tap(1, 1, 1);

  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const c00 = lerp(v000[c]!, v100[c]!, fx);
    const c10 = lerp(v010[c]!, v110[c]!, fx);
    const c01 = lerp(v001[c]!, v101[c]!, fx);
    const c11 = lerp(v011[c]!, v111[c]!, fx);
    out[c] = lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz);
  }
  return out;
}
