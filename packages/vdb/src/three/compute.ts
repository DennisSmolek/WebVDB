/**
 * three.js wrappers over the core compute tools (K2.3). v1's raw-WebGPU
 * kernels (`gridStats`, `valueTransform`, `decodeToAtlas` over a
 * `NanoVDBGrid` and the vendored WGSL text) moved into the engine-free core
 * (`core/compute/`) and now read grids from a `VDBGridPool`; these wrappers
 * take the renderer instead of a device (`getDevice(renderer)`), submit their
 * own encoder and return a promise.
 *
 * | v1 | core (`@pmndrs/vdb`) | this file (`@pmndrs/vdb/three`) |
 * |---|---|---|
 * | `vdbGridStatsGpu(device, grid, pnanovdbSource, opts)` | `vdbGridStatsGpu(device, pool, id, opts)` → `VDBReadback` | `vdbGridStats(renderer, pool, id, opts)` → `Promise` |
 * | `vdbDecodeToAtlasGpu(device, grid, pnanovdbSource, opts)` | `vdbDecodeToAtlasGpu(device, pool, id, opts)` → `VDBAtlasReadback` | `vdbDecodeToAtlas(renderer, pool, id, opts)` → `Promise`, plus a ready `Data3DTexture` |
 * | `valueTransform(device, grid, pnanovdbSource, wgslBodyOrPreset)` | `vdbApplyValueTransform(pool, id, sourceImage, fnOrPreset)` (CPU + `pool.replace`) | — (use the core function) |
 *
 * The `vdb` prefix keeps these apart from the core functions of the same job
 * (D17). For recording into your own encoder (the D8 pattern), use the core
 * functions directly with `getDevice(renderer)`; then submit, and
 * `job.submitted().read()`.
 */
import { Data3DTexture, FloatType, LinearFilter, RedFormat, UnsignedByteType } from "three";
import type { VDBGridPool } from "../core/pool/pool.js";
import { vdbGridStatsGpu, type VDBGridStats } from "../core/compute/stats.js";
import { vdbDecodeToAtlasGpu, type VDBAtlas, type VDBDecodeToAtlasOptions } from "../core/compute/atlas.js";
import { getDevice, type VDBThreeRendererLike } from "./device.js";

/** Options of {@link vdbGridStats}. */
export interface VDBThreeGridStatsOptions {
  /** Histogram bins over `[rootMin, rootMax]`. Default 256. */
  histogramBins?: number;
}

/** What {@link vdbGridStats} resolves with: the core's `VDBGridStats` (v1's result plus the histogram `range`). */
export type VDBGridStatsResult = VDBGridStats;

/**
 * Statistics of grid `gridId`: active-voxel min/max/mean/count and a histogram
 * (see the core `gridStats`). The grid must be resident (`pool.flush()` or an
 * update submitted) and the pool must live on the renderer's device. Records
 * into its own encoder and submits it. Rejects (never throws) on bad input.
 *
 * @param renderer The initialized `WebGPURenderer` whose device the pool lives on.
 * @param pool The pool that holds the grid.
 * @param gridId The grid's pool id; it must be resident.
 * @param options Histogram bin count.
 */
export function vdbGridStats(
  renderer: VDBThreeRendererLike,
  pool: VDBGridPool,
  gridId: number,
  options: VDBThreeGridStatsOptions = {},
): Promise<VDBGridStatsResult> {
  try {
    return vdbGridStatsGpu(getDevice(renderer), pool, gridId, options.histogramBins === undefined ? {} : { histogramBins: options.histogramBins }).read();
  } catch (e) {
    return Promise.reject(e);
  }
}

/** Options of {@link vdbDecodeToAtlas}: the core's, without the encoder/readback plumbing. */
export type VDBThreeDecodeToAtlasOptions = Omit<VDBDecodeToAtlasOptions, "encoder" | "readback">;

/** v1's result plus a `Data3DTexture` over `data`. */
export interface VDBDecodeToAtlasResult extends VDBAtlas {
  /**
   * `data` as a three `Data3DTexture`: `RedFormat`, `UnsignedByteType`
   * (`uint8`) or `FloatType` (`float32`), linear filtering, `unpackAlignment`
   * 1, `needsUpdate` set. Sample it with TSL `texture3D(texture, uvw).r`.
   * Linear filtering of a `float32` atlas needs the `float32-filterable`
   * feature; use `uint8` (the default) otherwise. Dispose it yourself.
   *
   * Texel `i` of an axis with `n` texels holds the grid at index
   * `min + (i + 0.5) * (max - min + 1) / n - 0.5`: voxel `min + i` at full
   * resolution (K2.6). The texture spans the voxel-centred box
   * `[min - 0.5, max + 0.5]` of the index bbox (world: that box times the
   * voxel size, through the grid transform).
   */
  texture: Data3DTexture;
}

/**
 * Dense decode of grid `gridId`'s index bbox (see the core `decodeToAtlas`,
 * including its voxel-centre sample convention), read back and wrapped in a
 * `Data3DTexture`. Records into its own encoder and submits it.
 *
 * @param renderer The initialized `WebGPURenderer` whose device the pool lives on.
 * @param pool The pool that holds the grid.
 * @param gridId The grid's pool id; it must be resident.
 * @param options Resolution cap, filter, format and normalization range.
 */
export async function vdbDecodeToAtlas(
  renderer: VDBThreeRendererLike,
  pool: VDBGridPool,
  gridId: number,
  options: VDBThreeDecodeToAtlasOptions = {},
): Promise<VDBDecodeToAtlasResult> {
  const { encoder: _e, readback: _r, ...rest } = options as VDBDecodeToAtlasOptions;
  const atlas = await vdbDecodeToAtlasGpu(getDevice(renderer), pool, gridId, rest).read();
  const [dx, dy, dz] = atlas.dims;
  const texture = new Data3DTexture(atlas.data, dx, dy, dz);
  texture.format = RedFormat;
  texture.type = atlas.data instanceof Uint8Array ? UnsignedByteType : FloatType;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return { ...atlas, texture };
}
