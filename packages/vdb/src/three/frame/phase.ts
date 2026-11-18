// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/framePhase.js L5–58 (getFramePhase, getFramePhaseSpectrum),
//   L79–108 (getFramePhaseAsymmetry, isFrameDiffusion, getFrameTwoSidedDiffusion,
//   getFrameMultipleScattering).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; Lorenz-Mie tables (getFrameMieTexels) and the octave packing
// (getFrameScatteringOctaves) wait for K3.4. Divergences: DIVERGES.md (K3.1 rows).

import type { Vector3, Vector4 } from "three";
import { frameSpectrumWavelengths } from "./spectrum.js";

/** Phase functions of a volume (his `phase` prop). */
export type VDBFramePhase = "isotropic" | "henyey-greenstein" | "dual-lobe" | "lorenz-mie" | "lorenz-mie-rgb";

/** Multiple scattering modes (his `frameRenderer.settings.multipleScattering`). */
export type VDBFrameMultipleScattering = boolean | "two-sided" | "octaves" | "neural";

export interface VDBFramePhaseSource {
  phase: VDBFramePhase;
  phaseG: number;
  phaseBackG: number;
  phaseBackWeight: number;
  phaseGExponent: number;
}

const clampG = (g: unknown) => Math.min(0.99, Math.max(-0.99, Number(g) || 0.0));
const clampWeight = (weight: unknown) => Math.min(1.0, Math.max(0.0, Number(weight) || 0.0));

/** Lorenz-Mie mode of 'lorenz-mie-rgb' in the phase vector. */
export const frameMieRgbMode = 2.0;

/** Asymmetry g of his Lorenz-Mie table (frameMieTable.js). */
export const frameMieAsymmetry = 0.8526;

/**
 * Writes (forward g, back g, back weight, Lorenz-Mie mode) to target. Isotropic is (0, 0, 0, 0).
 * The Step 1 shaders compile the Lorenz-Mie term out; a Mie phase falls back to isotropic HG with
 * a warning from the renderer (K3.4 ports the tables).
 */
export const getFramePhase = (source: VDBFramePhaseSource, target: Vector4) => {
  if (source.phase === "henyey-greenstein") {
    return target.set(clampG(source.phaseG), 0.0, 0.0, 0.0);
  }

  if (source.phase === "dual-lobe") {
    return target.set(clampG(source.phaseG), clampG(source.phaseBackG), clampWeight(source.phaseBackWeight), 0.0);
  }

  if (source.phase === "lorenz-mie") {
    return target.set(0.0, 0.0, 0.0, 1.0);
  }

  if (source.phase === "lorenz-mie-rgb") {
    return target.set(0.0, 0.0, 0.0, frameMieRgbMode);
  }

  return target.set(0.0, 0.0, 0.0, 0.0);
};

/** Writes (red g scale, blue g scale, flag) to target; flag 1 when the channels have other phases. */
export const getFramePhaseSpectrum = (source: VDBFramePhaseSource, target: Vector3) => {
  const exponent = Number.isFinite(Number(source.phaseGExponent)) ? Number(source.phaseGExponent) : 0.0;

  if (source.phase === "lorenz-mie-rgb") {
    return target.set(1.0, 1.0, 1.0);
  }

  if ((source.phase === "henyey-greenstein" || source.phase === "dual-lobe") && exponent !== 0.0) {
    const [red, green, blue] = frameSpectrumWavelengths;

    return target.set(Math.pow(red / green, -exponent), Math.pow(blue / green, -exponent), 1.0);
  }

  return target.set(1.0, 1.0, 0.0);
};

/** Asymmetry g (mean cosine of the scattering angle) of the phase of source. */
export const getFramePhaseAsymmetry = (source: VDBFramePhaseSource) => {
  if (source.phase === "henyey-greenstein") {
    return clampG(source.phaseG);
  }

  if (source.phase === "dual-lobe") {
    const weight = clampWeight(source.phaseBackWeight);

    return clampG(source.phaseG) * (1.0 - weight) + clampG(source.phaseBackG) * weight;
  }

  if (source.phase === "lorenz-mie" || source.phase === "lorenz-mie-rgb") {
    return frameMieAsymmetry;
  }

  return 0.0;
};

/** Modes with the diffusion approximation: true (1D), 'two-sided' (2D) and 'neural'. */
export const isFrameDiffusion = (mode: VDBFrameMultipleScattering) => !!mode && mode !== "octaves";

/** 1.0 for multipleScattering 'two-sided'. */
export const getFrameTwoSidedDiffusion = (mode: VDBFrameMultipleScattering) => (mode === "two-sided" ? 1.0 : 0.0);

/** Multiple scattering value of the shadow marches: 1 − g for the diffusion modes, else 0. */
export const getFrameMultipleScattering = (mode: VDBFrameMultipleScattering, source: VDBFramePhaseSource) =>
  isFrameDiffusion(mode) ? 1.0 - getFramePhaseAsymmetry(source) : 0.0;
