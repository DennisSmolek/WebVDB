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
 * The eight taps are `readValue(base + [dx, dy, dz])` for `dx, dy, dz` each
 * in `{0, 1}` — i.e. the voxel at `base` and its unit-offset neighbors toward
 * `+x, +y, +z`. Each tap contributes its stored/background VALUE regardless
 * of active state: an inactive (background) voxel still participates in the
 * blend using its background value, exactly like a GPU texture sampler would
 * (it has no notion of "active" — it just reads whatever is in the texel).
 * This is a deliberate divergence from `readValue`'s `{ value, active }`
 * pair: `sampleTrilinear` only ever returns a value.
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
import { readValue } from "./read-value.js";

type Coord = readonly [number, number, number];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Continuous index-space trilinear sample. `xyz` is in the same index space
 * as the integer coords passed to `readValue` (i.e. NOT world space — no
 * voxel-size or grid-transform scaling is applied here).
 */
export function sampleTrilinear(gridWords: Uint32Array, xyz: [number, number, number]): number {
  const bx = Math.floor(xyz[0]);
  const by = Math.floor(xyz[1]);
  const bz = Math.floor(xyz[2]);
  const fx = xyz[0] - bx;
  const fy = xyz[1] - by;
  const fz = xyz[2] - bz;

  const tap = (dx: 0 | 1, dy: 0 | 1, dz: 0 | 1): number => {
    const coord: Coord = [bx + dx, by + dy, bz + dz];
    return readValue(gridWords, coord as [number, number, number]).value;
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
