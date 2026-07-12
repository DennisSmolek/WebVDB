/**
 * build-from-vdb.ts — bridges the `.vdb` parser to the NanoVDB serializer.
 *
 * A parsed {@link VdbGrid} exposes `iterLeaves()` (each leaf: 8-aligned origin,
 * 512 values, 512-bit value mask) and an index->world transform. `buildFromVdb`
 * streams those leaves straight into the leaf-iterator build path
 * ({@link buildFromLeavesDetailed}) with the FLOAT codec — no dense array is
 * materialised, so a 7M-active-voxel teapot in a large bbox stays sparse.
 *
 * The `.vdb` transform is carried into the NanoVDB Map. v1 supports uniform
 * scale + translate only (which is what all four openvdb.org samples use); a
 * non-uniform scale, rotation or shear throws clearly (GPU resampling is the
 * roadmap for those).
 */

import type { VdbGrid, VdbTransformInfo } from "../vdb/types.js";
import { FLOAT_LEAF_CODEC } from "./leaf-codec.js";
import {
  buildFromLeavesDetailed,
  type BuildFromLeavesOptions,
  type BuiltGrid,
} from "./serialize.js";

export interface BuildFromVdbOptions {
  /** Grid name for the output image. Default: the source grid's name (or "grid"). */
  gridName?: string;
  /** Grid class. Default "FogVolume". */
  gridClass?: "FogVolume" | "Unknown";
}

/** Builds a FLOAT NanoVDB grid image from a parsed `.vdb` grid. */
export function buildFromVdb(grid: VdbGrid, opts: BuildFromVdbOptions = {}): Uint32Array {
  return buildFromVdbDetailed(grid, opts).image;
}

/** As {@link buildFromVdb}, but returns the computed metadata alongside the image. */
export function buildFromVdbDetailed(grid: VdbGrid, opts: BuildFromVdbOptions = {}): BuiltGrid {
  const { voxelSize, translation } = uniformTransform(grid.transform);
  const buildOpts: BuildFromLeavesOptions = {
    voxelSize,
    worldOrigin: translation,
    background: grid.background,
    gridName: opts.gridName ?? grid.name ?? "grid",
    gridClass: opts.gridClass ?? "FogVolume",
  };
  return buildFromLeavesDetailed(grid.iterLeaves(), FLOAT_LEAF_CODEC, buildOpts);
}

/**
 * Extracts a uniform scale + translation from a parsed `.vdb` transform's
 * row-major 4x4 index->world matrix. Throws (with a pointer to the GPU-resample
 * roadmap) on any rotation, shear or non-uniform scale — the serializer's Map is
 * axis-aligned and uniform-scale in v1.
 */
export function uniformTransform(t: VdbTransformInfo): {
  voxelSize: number;
  translation: [number, number, number];
} {
  const m = t.matrix;
  if (m.length !== 16) {
    throw new Error(`buildFromVdb: expected a 4x4 (16-element) transform matrix, got ${m.length}`);
  }
  const sx = m[0]!;
  const sy = m[5]!;
  const sz = m[10]!;
  // Off-diagonal terms of the linear 3x3 block must vanish (no rotation/shear).
  const offDiagonal = [m[1]!, m[2]!, m[4]!, m[6]!, m[8]!, m[9]!];
  const eps = 1e-6 * Math.max(1, Math.abs(sx), Math.abs(sy), Math.abs(sz));
  const rotated = offDiagonal.some((v) => Math.abs(v) > eps);
  const nonUniform = Math.abs(sx - sy) > eps || Math.abs(sx - sz) > eps;
  if (rotated || nonUniform) {
    throw new Error(
      `buildFromVdb: transform "${t.type}" is ${rotated ? "rotated/sheared" : "non-uniformly scaled"} ` +
        `(scale [${sx}, ${sy}, ${sz}]${rotated ? `, off-diagonal ${JSON.stringify(offDiagonal)}` : ""}). ` +
        `v1 supports uniform scale + translate only; rotation/shear/anisotropy need GPU resampling (roadmap).`,
    );
  }
  return { voxelSize: sx, translation: [m[3]!, m[7]!, m[11]!] };
}
