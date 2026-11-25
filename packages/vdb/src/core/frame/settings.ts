// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L191–246 (the settings: names, meanings and defaults).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free (K6.1); `froxelMix` added (design §3.4). Divergences: DIVERGES.md V33, V67, K6.1 rows.

/**
 * The settings of the core frame renderer (K6.1, design §2.2): his `frameRenderer.settings` (the same
 * names, meanings and v0.26 defaults as `@pmndrs/vdb/three` `VDBFrameRendererSettings`, which a unit test
 * checks field by field), plus the K6 fields. `./three` keeps its own declaration until K6.8.
 */

/** Phase functions (his `phase`). */
export type VDBFramePhaseName = "isotropic" | "henyey-greenstein" | "dual-lobe" | "lorenz-mie" | "lorenz-mie-rgb";

/** Multiple scattering modes (his `multipleScattering`). */
export type VDBFrameMultipleScatteringMode = boolean | "two-sided" | "octaves" | "neural";

/** The settings; see `VDBFrameRendererSettings` in `./three` for each of his fields. */
export interface VDBFrameSettings {
  width: number;
  height: number;
  pixelsPerFroxel: number;
  depth: number;
  subSteps: number;
  near: number;
  far: number;
  fadeRadius: number;
  jitter: boolean;
  temporal: boolean;
  temporalBlend: number;
  lighting: "volume" | "frame";
  shadowResolution: number;
  shadowCascades: number;
  shadowDistance: number;
  phase: VDBFramePhaseName;
  phaseG: number;
  phaseBackG: number;
  phaseBackWeight: number;
  phaseGExponent: number;
  srgbBlending: boolean;
  environmentLighting: "outward" | "directional" | "spherical-harmonics";
  environmentSamples: number;
  environmentShOrder: number;
  multipleScattering: VDBFrameMultipleScatteringMode;
  scatteringOctaves: number;
  scatteringOctaveExtinction: number;
  scatteringOctaveContribution: number;
  scatteringOctavePhase: number;
  spectral: boolean;
  castShadows: boolean;
  castShadowMapSize: number;
  castShadowCascades: number;
  castShadowDistance: number;
  castShadowSteps: number;
  castShadowUpdatesPerFrame: number;
  castShadowCoarse: boolean;
  /**
   * K6: how overlapping volumes combine in a froxel (design §3.4). `'mixture'` (default): co-located media,
   * order-independent (Σ τᵢ, Σ τᵢLᵢ). `'over'`: his draw-order premultiplied "over", in record order; internal,
   * for the A/B against the TSL path (K6.R decides; K6.8 deletes the loser). One volume per froxel: the same values.
   */
  froxelMix: "mixture" | "over";
}

/** His v0.26 defaults (`FrameRenderer.js` L191–246, as `VDBFrameRenderer.settings`), plus `froxelMix: 'mixture'`. */
export function vdbFrameDefaultSettings(): VDBFrameSettings {
  return {
    width: 512,
    height: 512,
    pixelsPerFroxel: 0,
    depth: 24,
    subSteps: 32,
    near: 0.1,
    far: Infinity,
    fadeRadius: Infinity,
    jitter: false,
    temporal: true,
    temporalBlend: 0.2,
    lighting: "frame",
    shadowResolution: 64,
    shadowCascades: 1,
    shadowDistance: 512,
    phase: "henyey-greenstein",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    phaseGExponent: 0.0,
    srgbBlending: true,
    environmentLighting: "directional",
    environmentSamples: 8,
    environmentShOrder: 1,
    multipleScattering: "neural",
    scatteringOctaves: 4,
    scatteringOctaveExtinction: 0.5,
    scatteringOctaveContribution: 0.5,
    scatteringOctavePhase: 0.5,
    spectral: false,
    castShadows: false,
    castShadowMapSize: 512,
    castShadowCascades: 1,
    castShadowDistance: 4000,
    castShadowSteps: 32,
    castShadowUpdatesPerFrame: 1,
    castShadowCoarse: false,
    froxelMix: "mixture",
  };
}
