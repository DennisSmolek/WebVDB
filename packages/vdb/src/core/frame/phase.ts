// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/framePhase.js L5–58 (getFramePhase, getFramePhaseSpectrum), L79–108
//   (getFramePhaseAsymmetry, isFrameDiffusion, getFrameMultipleScattering).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free (plain tuples) for the core renderer (K6.1). Divergences: DIVERGES.md (K3.4 rows).

import type { VDBFrameMultipleScatteringMode, VDBFramePhaseName } from "./settings.js";

/** The phase fields of a volume after its overrides (his `VDBFramePhaseSource`). */
export interface VDBFramePhaseFields {
  phase: VDBFramePhaseName;
  phaseG: number;
  phaseBackG: number;
  phaseBackWeight: number;
  phaseGExponent: number;
}

const clampG = (g: unknown) => Math.min(0.99, Math.max(-0.99, Number(g) || 0.0));
const clampWeight = (w: unknown) => Math.min(1.0, Math.max(0.0, Number(w) || 0.0));
/** His `frameSpectrumWavelengths` (nm): red, green, blue. */
const WAVELENGTHS = [650.0, 550.0, 450.0] as const;
/** Lorenz-Mie mode of 'lorenz-mie-rgb'. */
const MIE_RGB_MODE = 2.0;
/** Asymmetry g of the Lorenz-Mie table (`vdbFrameMieAsymmetry`). */
const MIE_ASYMMETRY = 0.8526;

/** `getFramePhase`: (forward g, back g, back weight, Lorenz-Mie mode). */
export function vdbFramePhaseVector(p: VDBFramePhaseFields): [number, number, number, number] {
  if (p.phase === "henyey-greenstein") return [clampG(p.phaseG), 0, 0, 0];
  if (p.phase === "dual-lobe") return [clampG(p.phaseG), clampG(p.phaseBackG), clampWeight(p.phaseBackWeight), 0];
  if (p.phase === "lorenz-mie") return [0, 0, 0, 1];
  if (p.phase === "lorenz-mie-rgb") return [0, 0, 0, MIE_RGB_MODE];
  return [0, 0, 0, 0];
}

/** `getFramePhaseSpectrum`: (red g scale, blue g scale, flag). */
export function vdbFramePhaseSpectrum(p: VDBFramePhaseFields): [number, number, number] {
  const exponent = Number.isFinite(Number(p.phaseGExponent)) ? Number(p.phaseGExponent) : 0.0;
  if (p.phase === "lorenz-mie-rgb") return [1, 1, 1];
  if ((p.phase === "henyey-greenstein" || p.phase === "dual-lobe") && exponent !== 0.0) {
    const [red, green, blue] = WAVELENGTHS;
    return [Math.pow(red / green, -exponent), Math.pow(blue / green, -exponent), 1];
  }
  return [1, 1, 0];
}

/** `getFramePhaseAsymmetry`. */
export function vdbFramePhaseAsymmetry(p: VDBFramePhaseFields): number {
  if (p.phase === "henyey-greenstein") return clampG(p.phaseG);
  if (p.phase === "dual-lobe") {
    const w = clampWeight(p.phaseBackWeight);
    return clampG(p.phaseG) * (1 - w) + clampG(p.phaseBackG) * w;
  }
  if (p.phase === "lorenz-mie" || p.phase === "lorenz-mie-rgb") return MIE_ASYMMETRY;
  return 0;
}

/** `getFrameMultipleScattering`: 1 − g for the diffusion modes (true, 'two-sided', 'neural'), else 0. */
export function vdbFrameMultipleScatteringValue(mode: VDBFrameMultipleScatteringMode, p: VDBFramePhaseFields): number {
  return !!mode && mode !== "octaves" ? 1 - vdbFramePhaseAsymmetry(p) : 0;
}
