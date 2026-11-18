/**
 * `applyValueTransform`: per-voxel value edits for pool grids (K2.3).
 *
 * Decided for 0.1: the transform runs on the CPU (`transformLeafValues` in
 * `@pmndrs/vdb/tools`) and the result is staged with `pool.replace`. v1 ran a
 * WGSL body on the GPU against a `read_write` copy of the image; the pool binds
 * read-only to every module (D10) and keeps no CPU copy of an image, so the
 * host passes the source image it added.
 *
 * A GPU write path may follow if interactive use needs it (demo 04 on the
 * reference device is the test): copy the image range into a scratch buffer,
 * run the transform with its own `read_write` binding, then a core
 * `pool.replaceFromBuffer(id, buffer, offset)` that copies GPU to GPU in
 * `update`.
 */
import { VDBError } from "../errors.js";
import type { VDBGridPool } from "../pool/pool.js";
import { transformLeafValues } from "../../tools/cpu/compute.js";

/** A per-value function; the result is rounded to f32. */
export type VDBValueTransform = (value: number) => number;

/** Named transforms (v1's `VALUE_TRANSFORM_PRESETS`, as functions). */
export const VDB_VALUE_TRANSFORM_PRESETS: Readonly<Record<"double" | "half" | "negate" | "clamp01", VDBValueTransform>> = Object.freeze({
  double: (v: number) => v * 2,
  half: (v: number) => v * 0.5,
  negate: (v: number) => -v,
  clamp01: (v: number) => Math.min(Math.max(v, 0), 1),
});

export type VDBValueTransformPreset = keyof typeof VDB_VALUE_TRANSFORM_PRESETS;

/**
 * Applies `transform` (a function or a preset name) to every leaf voxel value
 * inside grid `gridId`'s index bbox, and stages the result with
 * `pool.replace(gridId, image)`. Returns the new image (the source is not
 * modified); it reaches the GPU at the next `pool.update` / `pool.flush`, in
 * place (same size, so no `version` change).
 *
 * `sourceImage` is the grid's current image (the pool keeps none); it must be
 * the grid in the pool, which is checked by type, byte size and index bbox.
 *
 * As in v1: Float grids only; tile and background values are not changed;
 * the image's statistics are not recomputed (they keep describing the source
 * values, so `gridStats` bins against the old range and `vdbNodeMax` bounds
 * may be stale). Like `replace`, the pool holds the returned array until
 * `settled()`; don't reuse it before then.
 */
export function applyValueTransform(
  pool: VDBGridPool,
  gridId: number,
  sourceImage: Uint32Array,
  transform: VDBValueTransform | VDBValueTransformPreset,
): Uint32Array {
  const where = "applyValueTransform";
  const fn = typeof transform === "string" ? VDB_VALUE_TRANSFORM_PRESETS[transform] : transform;
  if (typeof fn !== "function") {
    throw new VDBError(
      "VDB_BAD_OPTION",
      `${where}: transform must be a function (v) => number or one of ${Object.keys(VDB_VALUE_TRANSFORM_PRESETS).join(", ")}; got ${JSON.stringify(transform)}.`,
    );
  }
  const info = pool.info(gridId);
  if (info.type !== "float") {
    throw new VDBError(
      "VDB_UNSUPPORTED_GRID_TYPE",
      `${where}: grid ${gridId} is ${info.type}; value transforms need a Float grid (quantized leaves would need re-encoding). Transform the Float source, quantize() it, and pool.replace().`,
    );
  }
  if (!(sourceImage instanceof Uint32Array) || sourceImage.byteLength !== info.imageBytes) {
    throw new VDBError(
      "VDB_BAD_OPTION",
      `${where}: sourceImage must be grid ${gridId}'s current image (${info.imageBytes} bytes); got ${sourceImage instanceof Uint32Array ? `${sourceImage.byteLength} bytes` : typeof sourceImage}.`,
    );
  }
  const image = transformLeafValues(sourceImage, fn);
  pool.replace(gridId, image);
  return image;
}
