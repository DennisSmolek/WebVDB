/**
 * `.wvdb` dense level → NanoVDB FLOAT grid with identical voxels (K3.2, R1 §6 Step 2 A/B).
 *
 * The sparse-vs-dense A/B renders the same voxels through both samplers, so the only difference
 * is the sampler (hardware trilinear on `r8unorm` vs the sparse 8-tap trilinear). This takes the
 * exact texels the dense source uploads (`wvdbDenseVolume`, his 8-bit `clamp(v / scale, 0, 1)`) and
 * writes each one as the f32 the GPU decodes from `r8unorm`: `k / 255`. Texel `(x, y, z)` becomes
 * index voxel `(x, y, z)`; non-zero texels are active, zero texels are the 0 background (so the
 * grid's stats are the mean of the non-zero texels, his `getDensityStats` rule up to the f32 sum).
 *
 * The unit box of the volume maps to the index box `[0, dims − 1]` (pass it as the sparse source's
 * `indexBox`: the built grid's own bbox is the active voxels' only). The Map is a uniform scale by
 * the x voxel size of the `.wvdb` box with the first voxel centre at `bbox.min + voxel / 2`: only
 * informative, since the froxel renderer places volumes by their box, not the grid's Map.
 */

import { VDBToolsError } from "../errors.js";
import { vdbBuildFromDenseDetailed, type VDBBuiltGrid } from "../nanovdb/serialize.js";
import type { WvdbDenseVolume } from "./dense.js";

export interface WvdbDenseToNvdbOptions {
  /** Channel to convert (a scalar one). Default `'density'`. */
  channel?: "density" | "mask" | "emission";
  /** Grid name. Default: the channel's grid name in the `.wvdb`. */
  gridName?: string;
}

/** The converted grid plus the index box the dense texture covered. */
export interface WvdbNvdbGrid extends VDBBuiltGrid {
  /** `[0, 0, 0]` to `dims − 1`: the dense texture's voxels (inclusive). */
  readonly textureIndexBox: { min: [number, number, number]; max: [number, number, number] };
}

/** Converts one scalar channel of a dense `.wvdb` level to a NanoVDB FLOAT grid, voxel for voxel. */
export function wvdbDenseToNvdb(volume: WvdbDenseVolume, o: WvdbDenseToNvdbOptions = {}): WvdbNvdbGrid {
  const channel = o.channel ?? "density";
  const tex = volume[channel];
  if (!tex) throw new VDBToolsError("VDB_BAD_OPTION", `wvdbDenseToNvdb: the volume has no ${channel} channel`);
  if (tex.texelBytes !== 1) throw new VDBToolsError("VDB_BAD_OPTION", `wvdbDenseToNvdb: the ${channel} channel is not scalar (r8unorm)`);
  const [w, h, d] = tex.dimensions;
  const values = new Float32Array(w * h * d);
  // Texels are x fastest (x + (y + z·h)·w); vdbBuildFromDense reads x-major ((x·h + y)·d + z).
  for (let z = 0; z < d; z++)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) values[(x * h + y) * d + z] = tex.data[x + (y + z * h) * w]! / 255;
  const bbox = volume.bbox;
  const voxelSize = bbox ? (bbox[1][0] - bbox[0][0]) / w : 1;
  const worldOrigin: [number, number, number] = bbox
    ? [bbox[0][0] + voxelSize / 2, bbox[0][1] + (bbox[1][1] - bbox[0][1]) / h / 2, bbox[0][2] + (bbox[1][2] - bbox[0][2]) / d / 2]
    : [0, 0, 0];
  const built = vdbBuildFromDenseDetailed(values, [w, h, d], {
    voxelSize: voxelSize > 0 ? voxelSize : 1,
    worldOrigin,
    background: 0,
    gridName: o.gridName ?? volume.gridNames[channel] ?? channel,
    gridClass: "FogVolume",
  });
  return { ...built, textureIndexBox: { min: [0, 0, 0], max: [w - 1, h - 1, d - 1] } };
}
