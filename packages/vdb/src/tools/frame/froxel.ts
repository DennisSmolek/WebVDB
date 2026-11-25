// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameInjectMaterial.js L59–101 (the premultiplied "over" blend of
//   overlapping volumes), src/three-openvdb/frame/webgpu/FrameTemporalMaterial.js L120–200 (the
//   neighbourhood clamp and blend), src/three-openvdb/frame/webgpu/FrameIntegrateMaterial.js L219–294
//   (the slice march with the T < 0.001 stop and the `sliceAlpha` coverage of the last slice).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: the core renderer's froxel stages (K6.1, design §3.4): an order-independent 'mixture' of the
// volumes in a froxel beside his 'over'; his integrate split into a per-column prefix scan and a two-tap
// composite. Divergences: DIVERGES.md (K6.1 rows).

/**
 * CPU twins (f64) of the core frame renderer's froxel stages (K6.1): `vdbFrameGatherCs`'s combine of
 * the volumes in one froxel, `vdbFrameResolveCs`'s temporal blend and column scan, and
 * `vdbFrameIntegrate`'s two-tap composite. Engine-free; the oracles of `e2e/k6-core.spec.ts`.
 */

import { vdbFrameFroxelSlice, vdbFrameSliceDepth, type Vec2, type Vec3, type Vec4 } from "./index.js";

/** One volume's lit froxel: unpremultiplied slice radiance and slice alpha (`vdbFrameInjectSlice` + lights). */
export interface VDBFrameVolumeFroxel {
  radiance: Vec3;
  alpha: number;
}

/** How the volumes in one froxel combine (`VDBFrameSettings.froxelMix`). */
export type VDBFrameFroxelMix = "mixture" | "over";

/** Optical depths above this are clamped (T = 1.3e-14), so an opaque volume gives a finite τ. */
export const vdbFrameMaxOpticalDepth = 32;

/**
 * The gather kernel's combine of the volumes in one froxel, premultiplied (rgb = L·α, a = α).
 *
 * `'over'` (his inject blend, in record order): each volume over the ones before it.
 * `'mixture'` (design §3.4): co-located media; τᵢ = −log(1 − αᵢ), α = 1 − exp(−Στᵢ),
 * L = Σ τᵢLᵢ / Στᵢ. Order-independent; one volume gives exactly (Lα, α).
 * Volumes with α ≤ 0 do not count (discarded froxels).
 */
export function vdbFrameGatherFroxel(volumes: readonly VDBFrameVolumeFroxel[], mode: VDBFrameFroxelMix): Vec4 {
  const lit = volumes.filter((v) => v.alpha > 0);
  if (lit.length === 0) return [0, 0, 0, 0];
  if (lit.length === 1) {
    const v = lit[0]!;
    return [v.radiance[0] * v.alpha, v.radiance[1] * v.alpha, v.radiance[2] * v.alpha, v.alpha];
  }
  if (mode === "over") {
    const out: Vec4 = [0, 0, 0, 0];
    for (const v of lit) {
      const k = 1 - v.alpha;
      out[0] = v.radiance[0] * v.alpha + out[0] * k;
      out[1] = v.radiance[1] * v.alpha + out[1] * k;
      out[2] = v.radiance[2] * v.alpha + out[2] * k;
      out[3] = v.alpha + out[3] * k;
    }
    return out;
  }
  let tau = 0;
  const tl: Vec3 = [0, 0, 0];
  for (const v of lit) {
    const t = Math.min(-Math.log(Math.max(1 - v.alpha, 0)), vdbFrameMaxOpticalDepth);
    tau += t;
    for (let c = 0; c < 3; c++) tl[c] = tl[c]! + t * v.radiance[c]!;
  }
  const alpha = 1 - Math.exp(-tau);
  return [(tl[0] / tau) * alpha, (tl[1] / tau) * alpha, (tl[2] / tau) * alpha, alpha];
}

/**
 * His temporal blend of one froxel (`FrameTemporalMaterial`): the reprojected history clamped to the
 * per-channel min and max of the current froxel and its neighbours, then `mix(history, current, blend)`
 * (`blend` from `vdbFrameReproject`).
 */
export function vdbFrameTemporalFroxel(current: Vec4, history: Vec4, neighbours: readonly Vec4[], blend: number): Vec4 {
  const lo = [...current] as Vec4;
  const hi = [...current] as Vec4;
  for (const n of neighbours) {
    for (let c = 0; c < 4; c++) {
      lo[c] = Math.min(lo[c]!, n[c]!);
      hi[c] = Math.max(hi[c]!, n[c]!);
    }
  }
  return [0, 1, 2, 3].map((c) => {
    const h = Math.min(Math.max(history[c]!, lo[c]!), hi[c]!);
    return h + (current[c]! - h) * blend;
  }) as Vec4;
}

/** Transmittance below which the scan stops adding slices (his integrate's break). */
export const vdbFrameScanStop = 0.001;

/**
 * The resolve kernel's column scan (his integrate loop over whole slices): boundary k holds the
 * radiance of slices 0..k−1 and their opacity A = 1 − T. `froxels[z]` is the premultiplied
 * (rgb, α) of slice z (α clamped to [0, 1]); a slice whose start has T < 0.001 adds nothing (his break).
 * Returns `froxels.length + 1` boundaries; boundary 0 is (0, 0, 0, 0).
 */
export function vdbFrameScanColumn(froxels: readonly Vec4[], o: { stop?: number } = {}): Vec4[] {
  const stop = o.stop ?? vdbFrameScanStop;
  const out: Vec4[] = [[0, 0, 0, 0]];
  const L: Vec3 = [0, 0, 0];
  let T = 1;
  for (const f of froxels) {
    const a = Math.min(Math.max(f[3], 0), 1);
    if (T >= stop && a > 0) {
      for (let c = 0; c < 3; c++) L[c] = L[c]! + T * f[c]!;
      T *= 1 - a;
    }
    out.push([L[0], L[1], L[2], 1 - T]);
  }
  return out;
}

/** What the composite reads at a pixel: radiance and transmittance (grey: one T). */
export interface VDBFrameIntegrated {
  radiance: Vec3;
  transmittance: number;
}

/**
 * The composite's two taps (design §3.4): the scene depth falls `c` ∈ [0, 1] into the slice between
 * boundaries `a` and `b` (each (L rgb, A = 1 − T)). With α = 1 − T_b / T_a:
 * radiance = L_a + (L_b − L_a)·(1 − (1 − α)^c)/α (→ c as α → 0), T = T_a·(1 − α)^c: his `sliceAlpha`.
 */
export function vdbFrameIntegratePrefix(a: Vec4, b: Vec4, c: number): VDBFrameIntegrated {
  const ta = 1 - a[3];
  const tb = 1 - b[3];
  const alpha = Math.min(Math.max(ta > 1e-6 ? 1 - tb / ta : 0, 0), 1);
  const keep = Math.pow(1 - alpha, c);
  const w = alpha > 1e-4 ? (1 - keep) / alpha : c;
  return {
    radiance: [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w],
    transmittance: Math.max(ta, 0) * keep,
  };
}

/**
 * The composite's slice of a scene view depth: the slice `k` it falls in on his log slices (`range` the
 * froxel near and far; ≤ depth − 1) and his coverage `c` of that slice, linear in view depth:
 * (depth − sliceStart) / (sliceEnd − sliceStart), clamped to [0, 1].
 */
export function vdbFrameSliceCoord(viewDepth: number, range: Vec2, depth: number): { k: number; c: number } {
  const s = Math.min(Math.max(vdbFrameFroxelSlice(viewDepth, range, depth), 0), depth);
  const k = Math.min(Math.floor(s), depth - 1);
  const start = vdbFrameSliceDepth(range, k / depth);
  const end = vdbFrameSliceDepth(range, (k + 1) / depth);
  return { k, c: Math.min(Math.max((viewDepth - start) / (end - start), 0), 1) };
}
