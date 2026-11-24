// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameBlackbody.js L1–172 (the 'planck' / 'soot' emission colour table: its Planck ×
//   soot integration, XYZ → linear sRGB and normalization, and getFrameEmissionSpectrum);
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L236–256 (frameBlackbodyColor, the table read).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript, engine-free; the table read is a CPU twin of the shader's. His 'approximate' spectrum (the
// Shadertoy blackBody(), CC BY-NC-SA unless stated) is not ported (R1 §7): emission colour stays a caller hook,
// and this table is the hook the three adapter offers (`vdbFrameBlackbodyEmission`).
// D33: his table integrates the tabulated CIE 1931 2° colour matching functions (a CC BY-SA 4.0 dataset, CIE DOI
// 10.25039/CIE.DS.xvudnb9b). No CIE data is shipped here: the colour matching functions are the multi-lobe piecewise
// Gaussian fit (Equation 4, Table 1) of
//   C. Wyman, P.-P. Sloan, P. Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions",
//   Journal of Computer Graphics Techniques 2(2), 1–11, 2013. https://jcgt.org/published/0002/02/01/
// written from the paper's published formula and coefficients. Callers may pass their own colour matching functions
// (e.g. the exact CIE table, under its own licence) to every function below.

/**
 * K3.6: the physical emission spectra of his `FrameVolume.emissionSpectrum`: Planck radiance × the soot emissivity
 * λ^−exponent, integrated with the CIE 1931 2° colour matching functions (by default the analytic fit
 * {@link vdbCie1931Fit}, D33) at 380–780 nm in 5 nm steps, CIE XYZ → linear sRGB (IEC 61966-2-1), negatives to 0,
 * largest channel 1. A 256 × 16 RGBA table over log temperature (500–40,000 K) and soot exponent (0–3); `'planck'` is
 * the exponent-0 row.
 */

import type { Vec2, Vec3 } from "./index.js";

/** CIE 1931 2° colour matching functions (x̄, ȳ, z̄) at a wavelength in nm. */
export type VDBColorMatchingFunctions = (wavelength: number) => Vec3;

/** One piecewise Gaussian lobe: α · exp(−½ ((λ − β) · (λ < β ? γ : δ))²) (Wyman et al. 2013, Equation 4). */
const lobe = (wavelength: number, alpha: number, beta: number, gamma: number, delta: number): number => {
  const t = (wavelength - beta) * (wavelength < beta ? gamma : delta);
  return alpha * Math.exp(-0.5 * t * t);
};

/**
 * The CIE 1931 2° standard observer as the analytic multi-lobe fit of Wyman, Sloan & Shirley (JCGT 2(2), 2013,
 * Equation 4 with Table 1: three lobes for x̄, two each for ȳ and z̄). The paper's stated accuracy against the 1 nm
 * CIE curves: max squared error 2.0e-4 / 6.4e-5 / 4.9e-4, mean squared error 3.1e-5 / 7.1e-6 / 1.6e-5 (x̄ / ȳ / z̄).
 */
export function vdbCie1931Fit(wavelength: number): Vec3 {
  return [
    lobe(wavelength, 0.362, 442.0, 0.0624, 0.0374) + lobe(wavelength, 1.056, 599.8, 0.0264, 0.0323) + lobe(wavelength, -0.065, 501.1, 0.049, 0.0382),
    lobe(wavelength, 0.821, 568.8, 0.0213, 0.0247) + lobe(wavelength, 0.286, 530.9, 0.0613, 0.0322),
    lobe(wavelength, 1.217, 437.0, 0.0845, 0.0278) + lobe(wavelength, 0.681, 459.0, 0.0385, 0.0725),
  ];
}

/** His integration grid: 380–780 nm in 5 nm steps (81 samples). */
export const frameBlackbodyWavelengthStart = 380.0;
export const frameBlackbodyWavelengthStep = 5.0;
export const frameBlackbodyWavelengthCount = 81;

/** Second radiation constant h·c/k in nm·K. */
const c2 = 1.4388e7;

export const frameBlackbodyTableSize = 256;
export const frameBlackbodyTableRows = 16;
export const frameBlackbodyMinTemperature = 500.0;
export const frameBlackbodyMaxTemperature = 40000.0;
export const frameMaxSootExponent = 3.0;

/**
 * The table's colour of one temperature (K) and soot exponent: linear sRGB, largest channel 1 (0 when black).
 * `cmf`: the colour matching functions, sampled at his 81 wavelengths (default: the analytic fit, D33).
 */
export function vdbFrameBlackbodyRgb(temperature: number, exponent: number, cmf: VDBColorMatchingFunctions = vdbCie1931Fit): Vec3 {
  let x = 0;
  let y = 0;
  let z = 0;
  for (let i = 0; i < frameBlackbodyWavelengthCount; i++) {
    const wavelength = frameBlackbodyWavelengthStart + i * frameBlackbodyWavelengthStep;
    // Planck radiance without constant factors (the scale is removed at the end)
    const radiance = ((wavelength / 550.0) ** -exponent * (550.0 / wavelength) ** 5) / Math.expm1(c2 / (wavelength * temperature));
    const [xBar, yBar, zBar] = cmf(wavelength);
    x += radiance * xBar;
    y += radiance * yBar;
    z += radiance * zBar;
  }
  const r = Math.max(0, 3.2406 * x - 1.5372 * y - 0.4986 * z);
  const g = Math.max(0, -0.9689 * x + 1.8758 * y + 0.0415 * z);
  const b = Math.max(0, 0.0557 * x - 0.204 * y + 1.057 * z);
  const peak = Math.max(r, g, b);
  return peak > 0 ? [r / peak, g / peak, b / peak] : [0, 0, 0];
}

const tables = new WeakMap<VDBColorMatchingFunctions, Float32Array>();

/**
 * RGBA float texels, {@link frameBlackbodyTableSize} × {@link frameBlackbodyTableRows}, row = soot exponent (made once
 * per `cmf` function; default: the analytic fit, D33).
 */
export function vdbFrameBlackbodyTable(cmf: VDBColorMatchingFunctions = vdbCie1931Fit): Float32Array {
  const cached = tables.get(cmf);
  if (cached) return cached;
  const t = new Float32Array(frameBlackbodyTableSize * frameBlackbodyTableRows * 4);
  for (let row = 0; row < frameBlackbodyTableRows; row++) {
    const exponent = (frameMaxSootExponent * row) / (frameBlackbodyTableRows - 1);
    for (let column = 0; column < frameBlackbodyTableSize; column++) {
      const temperature = frameBlackbodyMinTemperature * (frameBlackbodyMaxTemperature / frameBlackbodyMinTemperature) ** (column / (frameBlackbodyTableSize - 1));
      const rgb = vdbFrameBlackbodyRgb(temperature, exponent, cmf);
      t.set([rgb[0], rgb[1], rgb[2], 1.0], (row * frameBlackbodyTableSize + column) * 4);
    }
  }
  tables.set(cmf, t);
  return t;
}

/**
 * frameBlackbodyColor (CPU twin): the table read with linear blending in log temperature and exponent between the
 * texels (`load`, no sampler), on the f32 table values (of the `cmf` table; default: the analytic fit, D33).
 * @param temperature Kelvin; clamped to the table's range.
 * @param exponent The soot exponent, clamped to `0..frameMaxSootExponent`.
 * @param cmf The colour matching functions the table integrates. Default {@link vdbCie1931Fit}.
 */
export function vdbFrameBlackbodyColor(temperature: number, exponent: number, cmf: VDBColorMatchingFunctions = vdbCie1931Fit): Vec3 {
  const t = vdbFrameBlackbodyTable(cmf);
  const scale = Math.log(frameBlackbodyMaxTemperature / frameBlackbodyMinTemperature);
  const x = Math.min(1, Math.max(0, Math.log(Math.max(temperature, 1e-3) / frameBlackbodyMinTemperature) / scale)) * (frameBlackbodyTableSize - 1);
  const y = Math.min(1, Math.max(0, exponent / frameMaxSootExponent)) * (frameBlackbodyTableRows - 1);
  const column = Math.min(Math.trunc(x), frameBlackbodyTableSize - 2);
  const row = Math.min(Math.trunc(y), frameBlackbodyTableRows - 2);
  const bx = x - column;
  const by = y - row;
  const texel = (dx: number, dy: number): Vec3 => {
    const i = ((row + dy) * frameBlackbodyTableSize + column + dx) * 4;
    return [t[i]!, t[i + 1]!, t[i + 2]!];
  };
  const mixv = (a: Vec3, b: Vec3, w: number): Vec3 => [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
  return mixv(mixv(texel(0, 0), texel(1, 0), bx), mixv(texel(0, 1), texel(1, 1), bx), by);
}

/** His `getFrameEmissionSpectrum` for the ported spectra: (table weight 1, soot exponent); 'planck' is exponent 0. An invalid soot exponent gives his 1.39. */
export function vdbFrameEmissionSpectrum(spectrum: "planck" | "soot", sootExponent = 1.39): Vec2 {
  if (spectrum === "planck") return [1, 0];
  const exponent = Number(sootExponent);
  const valid = Number.isFinite(exponent) ? exponent : 1.39;
  return [1, Math.min(frameMaxSootExponent, Math.max(0, valid))];
}
