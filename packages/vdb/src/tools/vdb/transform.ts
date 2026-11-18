/**
 * transform.ts — `GridDescriptor`'s `Map` (the index<->world affine
 * transform), per `GridTransform.js`'s `readTransform` and `vdb-rs`'s
 * `read_transform` (both agree on field order: translation, scale,
 * voxelSize, scaleInverse, scaleInverseSq, scaleInverseDouble — each a
 * vec3d — for the translate-bearing maps; the no-translation maps drop the
 * leading `translation` read).
 *
 * Only axis-aligned maps are supported (uniform/non-uniform scale + optional
 * translate) — the samples in this corpus use `UniformScaleTranslateMap`.
 * Rotated/sheared maps (`UnitaryMap`, `NonlinearFrustumMap`, general 4x4
 * `AffineMap`) are out of scope (per the Phase 3 handoff's note that
 * `NanoVDBGrid.indexToWorld()` is axis-aligned-only today) and raise
 * `VdbUnsupportedError`.
 */

import type { ByteReader } from "./byte-reader.js";
import { VdbUnsupportedError } from "./errors.js";

export interface VdbTransform {
  type: string;
  /** 4x4 row-major index->world affine matrix. */
  matrix: number[];
  voxelSize: [number, number, number];
}

const IDENTITY_VEC3: [number, number, number] = [0, 0, 0];

function toMatrix(
  scale: [number, number, number],
  translation: [number, number, number],
): number[] {
  const [sx, sy, sz] = scale;
  const [tx, ty, tz] = translation;
  // prettier-ignore
  return [
    sx, 0,  0,  tx,
    0,  sy, 0,  ty,
    0,  0,  sz, tz,
    0,  0,  0,  1,
  ];
}

export function readTransform(reader: ByteReader): VdbTransform {
  const mapType = reader.string();

  if (mapType === "UniformScaleTranslateMap" || mapType === "ScaleTranslateMap") {
    const translation = reader.vec3d();
    const scale = reader.vec3d();
    const voxelSize = reader.vec3d();
    reader.vec3d(); // scaleInverse (unused — derivable, kept for stream alignment)
    reader.vec3d(); // scaleInverseSq
    reader.vec3d(); // scaleInverseDouble
    return { type: mapType, matrix: toMatrix(scale, translation), voxelSize };
  }

  if (mapType === "UniformScaleMap" || mapType === "ScaleMap") {
    const scale = reader.vec3d();
    const voxelSize = reader.vec3d();
    reader.vec3d(); // scaleInverse
    reader.vec3d(); // scaleInverseSq
    reader.vec3d(); // scaleInverseDouble
    return { type: mapType, matrix: toMatrix(scale, IDENTITY_VEC3), voxelSize };
  }

  if (mapType === "TranslationMap") {
    const translation = reader.vec3d();
    return { type: mapType, matrix: toMatrix([1, 1, 1], translation), voxelSize: [1, 1, 1] };
  }

  throw new VdbUnsupportedError(
    `transform map type "${mapType}" (only Uniform/ScaleTranslateMap, Uniform/ScaleMap, TranslationMap are supported — rotated/sheared/frustum maps need the loader to surface the raw Map, per the Phase 3 handoff)`,
  );
}
