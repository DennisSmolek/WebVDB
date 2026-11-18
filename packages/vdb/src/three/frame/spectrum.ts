// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameSpectrum.js (all, 42 lines: wavelengths, Ångström exponents,
//   getFrameExtinctionSpectrum; K3.5: the spectral slot groups, L35–42).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript. Divergences: none.

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
 * spectral shaders read it (renderer `settings.spectral`); the grey passes keep it for the uniform layout.
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

// NOTE Spectral transmittance of 'frame' lighting: light slots in groups. One pass for each group in use.
// Group 0: sky up, sky down and slot 0. Group g (1 to 3): slots 2g - 1 and 2g. Each direction is marched one time.
// NOTE A group has two entries of (vec3 x, vec3 y): sky (x up, y down) or a slot (x total, y direct),
// packed in 3 RGBA targets: (a.x, a.y.r), (a.y.gb, b.x.rg), (b.x.b, b.y).
export const frameSpectralGroupCount = 4;

/** Groups in use for the number of light slots in use (0 to 6). */
export const getFrameSpectralGroups = (slotCount: number) => 1 + Math.ceil(Math.max(0, slotCount - 1) / 2);
