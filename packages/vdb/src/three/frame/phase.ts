// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/framePhase.js L5–58 (getFramePhase, getFramePhaseSpectrum),
//   L79–108 (getFramePhaseAsymmetry, isFrameDiffusion, getFrameTwoSidedDiffusion,
//   getFrameMultipleScattering); K3.4: L61–76 (getFrameMieTexels, via tools/frame/mie.ts), L110–129
//   (frameMaxScatteringOctaves, getFrameScatteringOctaves).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; the Mie tables are our generator's output (tools/frame/mie-table.ts, identical
// numbers). Divergences: DIVERGES.md (K3.1, K3.4 rows).

import type { Vector3, Vector4 } from "three";
import { vdbFrameMieAsymmetry, vdbFrameMieRgbMode, vdbFrameMieTexels } from "../../tools/frame/mie.js";
import { vdbFrameMaxScatteringOctaves, vdbFrameScatteringOctaves, type VDBFrameOctaveSettings } from "../../tools/frame/index.js";
import { frameSpectrumWavelengths } from "./spectrum.js";

/** Phase functions of a volume (his `phase` prop). */
export type VDBFramePhase = "isotropic" | "henyey-greenstein" | "dual-lobe" | "lorenz-mie" | "lorenz-mie-rgb";

/** Multiple scattering modes (his `frameRenderer.settings.multipleScattering`). */
/**
 * Multiple scattering modes (his `frameRenderer.settings.multipleScattering`). K6.3: `'mlp'` (EXPERIMENTAL generic
 * MLP evaluator) and `'multi-scale'` (K7.3) render with `backend: 'core'` only; the TSL path runs them as 1D diffusion.
 */
export type VDBFrameMultipleScattering = boolean | "two-sided" | "octaves" | "neural" | "mlp" | "multi-scale";

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
export const frameMieRgbMode = vdbFrameMieRgbMode;

/** Asymmetry g of the Lorenz-Mie table. */
export const frameMieAsymmetry = vdbFrameMieAsymmetry;

/**
 * Texels of the Lorenz-Mie texture: the 'lorenz-mie-rgb' tables in red, green, blue and the
 * 'lorenz-mie' table in alpha (one texture keeps the texture count of the shaders below 16).
 */
export const getFrameMieTexels = vdbFrameMieTexels;

/**
 * Writes (forward g, back g, back weight, Lorenz-Mie mode) to target. Isotropic is (0, 0, 0, 0).
 * Lorenz-Mie mode: 0 off, 1 the 550 nm table, 2 the red, green and blue tables (frameMieRgbMode).
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

export const frameMaxScatteringOctaves = vdbFrameMaxScatteringOctaves;

/**
 * Indirect light octaves of multipleScattering 'octaves': writes (count N, extinction a, contribution b,
 * phase c) to target. Octave n (1 to N) has extinction × aⁿ, light × bⁿ and phase g × cⁿ. Count 0 is off.
 */
export const getFrameScatteringOctaves = (settings: VDBFrameOctaveSettings, target: Vector4) => target.fromArray(vdbFrameScatteringOctaves(settings));
