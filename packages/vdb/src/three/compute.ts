/**
 * three.js wrappers over the core compute tools (K2.3). v1's raw-WebGPU
 * kernels (`gridStats`, `valueTransform`, `decodeToAtlas` over a
 * `NanoVDBGrid` and the vendored WGSL text) moved into the engine-free core
 * (`core/compute/`) and now read grids from a `VDBGridPool`; these wrappers
 * take the renderer instead of a device (`getDevice(renderer)`) and keep v1's
 * promise-returning shape.
 *
 * | v1 | now |
 * |---|---|
 * | `gridStats(device, grid, pnanovdbSource, opts)` | `gridStats(renderer, pool, gridId, opts)` |
 * | `decodeToAtlas(device, grid, pnanovdbSource, opts)` | `decodeToAtlas(renderer, pool, gridId, opts)`, plus a ready `Data3DTexture` |
 * | `valueTransform(device, grid, pnanovdbSource, wgslBodyOrPreset)` | `valueTransform(pool, gridId, sourceImage, fnOrPreset)` (CPU + `pool.replace`) |
 *
 * For recording into your own encoder (the D8 pattern), use the core
 * functions directly with `getDevice(renderer)`.
 */
import { Data3DTexture, FloatType, LinearFilter, RedFormat, UnsignedByteType } from "three";
import type { VDBGridPool } from "../core/pool/pool.js";
import { gridStats as coreGridStats, type VDBGridStats } from "../core/compute/stats.js";
import { decodeToAtlas as coreDecodeToAtlas, type VDBAtlas, type VDBDecodeToAtlasOptions } from "../core/compute/atlas.js";
import { applyValueTransform, type VDBValueTransform, type VDBValueTransformPreset } from "../core/compute/value-transform.js";
import { getDevice, type ThreeRendererLike } from "./device.js";

/** Options of {@link gridStats}. */
export interface GridStatsOptions {
  /** Histogram bins over `[rootMin, rootMax]`. Default 256. */
  histogramBins?: number;
}

/** v1's result shape, plus `range` (the histogram range). */
export type GridStatsResult = VDBGridStats;

/**
 * Statistics of grid `gridId`: active-voxel min/max/mean/count and a histogram
 * (see the core `gridStats`). The grid must be resident (`pool.flush()` or an
 * update submitted) and the pool must live on the renderer's device.
 */
export function gridStats(renderer: ThreeRendererLike, pool: VDBGridPool, gridId: number, options: GridStatsOptions = {}): Promise<GridStatsResult> {
  try {
    return coreGridStats(getDevice(renderer), pool, gridId, options.histogramBins === undefined ? {} : { histogramBins: options.histogramBins }).read();
  } catch (e) {
    return Promise.reject(e);
  }
}

/** Options of {@link decodeToAtlas}: the core's, without the encoder/readback plumbing. */
export type DecodeToAtlasOptions = Omit<VDBDecodeToAtlasOptions, "encoder" | "readback">;

/** v1's result plus a `Data3DTexture` over `data`. */
export interface DecodeToAtlasResult extends VDBAtlas {
  /**
   * `data` as a three `Data3DTexture`: `RedFormat`, `UnsignedByteType`
   * (`uint8`) or `FloatType` (`float32`), linear filtering, `unpackAlignment`
   * 1, `needsUpdate` set. Sample it with TSL `texture3D(texture, uvw).r`.
   * Linear filtering of a `float32` atlas needs the `float32-filterable`
   * feature; use `uint8` (the default) otherwise. Dispose it yourself.
   */
  texture: Data3DTexture;
}

/**
 * Dense decode of grid `gridId`'s index bbox (see the core `decodeToAtlas`),
 * read back and wrapped in a `Data3DTexture`.
 */
export async function decodeToAtlas(
  renderer: ThreeRendererLike,
  pool: VDBGridPool,
  gridId: number,
  options: DecodeToAtlasOptions = {},
): Promise<DecodeToAtlasResult> {
  const { encoder: _e, readback: _r, ...rest } = options as VDBDecodeToAtlasOptions;
  const atlas = await coreDecodeToAtlas(getDevice(renderer), pool, gridId, rest).read();
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

/**
 * v1's `valueTransform`, now on the CPU: applies `transform` to the leaf
 * values of Float grid `gridId` and stages the result with `pool.replace`
 * (core `applyValueTransform`; no renderer involved). `sourceImage` is the
 * image the grid holds; the returned image is the new one. It reaches the GPU
 * at the next `pool.update` / `pool.flush`.
 */
export function valueTransform(
  pool: VDBGridPool,
  gridId: number,
  sourceImage: Uint32Array,
  transform: VDBValueTransform | VDBValueTransformPreset,
): Uint32Array {
  return applyValueTransform(pool, gridId, sourceImage, transform);
}
