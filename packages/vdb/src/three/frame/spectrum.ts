// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameSpectrum.js L1–33 (wavelengths, Ångström exponents,
//   getFrameExtinctionSpectrum).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; the spectral slot groups (L35–42) wait for spectral 'frame' lighting (K3.5).
// Divergences: none.

import type { Vector3, Vector3Like } from "three";

/** Wavelengths in nm of the red, green and blue channels. Green is the reference (550 nm). */
export const frameSpectrumWavelengths = [650.0, 550.0, 450.0] as const;

const referenceWavelength = frameSpectrumWavelengths[1];

const toNumber = (value: unknown, fallback: number) => (Number.isFinite(Number(value)) ? Number(value) : fallback);

/** Measured Ångström exponents (his VDB-116 research; Dubovik 2002, Bodhaine 1999). */
export const frameAngstromExponents: Readonly<Record<string, number>> = {
  cloud: 0.0,
  dust: 0.3,
  smoke: 1.8,
  air: 4.09,
};

/** A preset name of {@link frameAngstromExponents} or a number. Unknown names give 0 (grey). */
export const getFrameAngstromExponent = (value: string | number | undefined) =>
  typeof value === "string" ? (frameAngstromExponents[value] ?? 0.0) : toNumber(value, 0.0);

/**
 * Extinction scale of each channel: extinctionScale × (wavelength / 550)^−angstromExponent. Only
 * spectral shaders read it; the grey Step 1 passes keep it for the uniform layout.
 */
export const getFrameExtinctionSpectrum = (
  frameVolume: { extinctionScale?: Vector3Like; angstromExponent?: string | number },
  target: Vector3,
) => {
  const scale = frameVolume.extinctionScale;
  const exponent = getFrameAngstromExponent(frameVolume.angstromExponent);
  const channel = (index: number, value: unknown) =>
    Math.max(0.0, toNumber(value, 1.0)) * Math.pow(frameSpectrumWavelengths[index]! / referenceWavelength, -exponent);

  return target.set(channel(0, scale?.x), channel(1, scale?.y), channel(2, scale?.z));
};
