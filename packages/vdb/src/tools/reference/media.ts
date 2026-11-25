/**
 * K6.R: the CPU side of the reference renderer's volume sources ({@link VDBReferenceMediumCpu}): the dense `.wvdb`
 * textures (twin of `vdbDenseSource`: hardware trilinear on the 8-bit texels, `sampleWvdbDenseTexture`), the sparse
 * pool (twin of `vdbSparseSourceGpu` at level 0: `tools/frame/sparse.ts`) and a constant medium (analytic tests).
 */

import { vdbSparseAlbedoCpu, vdbSparseDensityCpu, vdbSparseEmissionCpu, vdbSparseMaskCpu, type VDBSparseVolumeCpu } from "../frame/sparse.js";
import { sampleWvdbDenseTexture, type WvdbDenseVolume } from "../wvdb/dense.js";
import type { VDBRefVec3 } from "./scene.js";
import type { VDBReferenceMediumCpu } from "./trace.js";

/** The dense source at level 0: density × mask, the albedo texture's rgb, the emission texture. */
export function vdbReferenceDenseMediumCpu(volume: WvdbDenseVolume): VDBReferenceMediumCpu {
  const { density, mask, albedo, emission } = volume;
  return {
    density: (uvw) => sampleWvdbDenseTexture(density, uvw, 0) * (mask ? sampleWvdbDenseTexture(mask, uvw, 0) : 1),
    ...(albedo ? { albedo: (uvw) => [0, 1, 2].map((c) => sampleWvdbDenseTexture(albedo, uvw, 0, c)) as VDBRefVec3 } : {}),
    ...(emission ? { emission: (uvw) => sampleWvdbDenseTexture(emission, uvw, 0) } : {}),
  };
}

/** The sparse source at level 0 (one volume of a `vdbSparseSourceGpu`, as its CPU twin describes it). */
export function vdbReferenceSparseMediumCpu(vol: VDBSparseVolumeCpu): VDBReferenceMediumCpu {
  return {
    density: (uvw) => vdbSparseDensityCpu(vol, uvw, 0).value * vdbSparseMaskCpu(vol, uvw).value,
    ...(vol.albedo ? { albedo: (uvw) => vdbSparseAlbedoCpu(vol, uvw).value } : {}),
    ...(vol.emission ? { emission: (uvw) => vdbSparseEmissionCpu(vol, uvw).value } : {}),
  };
}

/** A constant medium (density `value`, emission channel `emission`), for the analytic tests. */
export function vdbReferenceConstantMediumCpu(value = 1, emission?: number): VDBReferenceMediumCpu {
  return { density: () => value, ...(emission !== undefined ? { emission: () => emission } : {}) };
}
