// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameBlackbody.js L1–172 (the 'planck' / 'soot' emission colour table and
//   getFrameEmissionSpectrum); src/three-openvdb/frame/webgpu/frameShaderChunks.js L236–256 (frameBlackbodyColor,
//   the table read).
// Data: CIE 1931 2° standard observer colour matching functions, 380–780 nm in 5 nm steps, from
//   https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv (CIE public dataset), as in his table.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript, engine-free; the table read is a CPU twin of the shader's. His 'approximate' spectrum (the
// Shadertoy blackBody(), CC BY-NC-SA unless stated) is not ported (R1 §7): emission colour stays a caller hook,
// and this table is the hook the three adapter offers (`vdbFrameBlackbodyEmission`).

/**
 * K3.6: the physical emission spectra of his `FrameVolume.emissionSpectrum`: Planck radiance × the soot emissivity
 * λ^−exponent, integrated with the CIE 1931 colour matching functions, CIE XYZ → linear sRGB (IEC 61966-2-1),
 * negatives to 0, largest channel 1. A 256 × 16 RGBA table over log temperature (500–40,000 K) and soot exponent
 * (0–3); `'planck'` is the exponent-0 row.
 */

import type { Vec2, Vec3 } from "./index.js";

const cieStart = 380.0;
const cieStep = 5.0;
// (x̄, ȳ, z̄) per 5 nm, 380–780 nm
// prettier-ignore
const cie1931 = [
  0.001368, 0.000039, 0.006450001,
  0.002236, 0.000064, 0.01054999,
  0.004243, 0.00012, 0.02005001,
  0.00765, 0.000217, 0.03621,
  0.01431, 0.000396, 0.06785001,
  0.02319, 0.00064, 0.1102,
  0.04351, 0.00121, 0.2074,
  0.07763, 0.00218, 0.3713,
  0.13438, 0.004, 0.6456,
  0.21477, 0.0073, 1.0390501,
  0.2839, 0.0116, 1.3856,
  0.3285, 0.01684, 1.62296,
  0.34828, 0.023, 1.74706,
  0.34806, 0.0298, 1.7826,
  0.3362, 0.038, 1.77211,
  0.3187, 0.048, 1.7441,
  0.2908, 0.06, 1.6692,
  0.2511, 0.0739, 1.5281,
  0.19536, 0.09098, 1.28764,
  0.1421, 0.1126, 1.0419,
  0.09564, 0.13902, 0.8129501,
  0.05795001, 0.1693, 0.6162,
  0.03201, 0.20802, 0.46518,
  0.0147, 0.2586, 0.3533,
  0.0049, 0.323, 0.272,
  0.0024, 0.4073, 0.2123,
  0.0093, 0.503, 0.1582,
  0.0291, 0.6082, 0.1117,
  0.06327, 0.71, 0.07824999,
  0.1096, 0.7932, 0.05725001,
  0.1655, 0.862, 0.04216,
  0.2257499, 0.9148501, 0.02984,
  0.2904, 0.954, 0.0203,
  0.3597, 0.9803, 0.0134,
  0.4334499, 0.9949501, 0.008749999,
  0.5120501, 1, 0.005749999,
  0.5945, 0.995, 0.0039,
  0.6784, 0.9786, 0.002749999,
  0.7621, 0.952, 0.0021,
  0.8425, 0.9154, 0.0018,
  0.9163, 0.87, 0.001650001,
  0.9786, 0.8163, 0.0014,
  1.0263, 0.757, 0.0011,
  1.0567, 0.6949, 0.001,
  1.0622, 0.631, 0.0008,
  1.0456, 0.5668, 0.0006,
  1.0026, 0.503, 0.00034,
  0.9384, 0.4412, 0.00024,
  0.8544499, 0.381, 0.00019,
  0.7514, 0.321, 0.0001,
  0.6424, 0.265, 0.00004999999,
  0.5419, 0.217, 0.00003,
  0.4479, 0.175, 0.00002,
  0.3608, 0.1382, 0.00001,
  0.2835, 0.107, 0,
  0.2187, 0.0816, 0,
  0.1649, 0.061, 0,
  0.1212, 0.04458, 0,
  0.0874, 0.032, 0,
  0.0636, 0.0232, 0,
  0.04677, 0.017, 0,
  0.0329, 0.01192, 0,
  0.0227, 0.00821, 0,
  0.01584, 0.005723, 0,
  0.01135916, 0.004102, 0,
  0.008110916, 0.002929, 0,
  0.005790346, 0.002091, 0,
  0.004109457, 0.001484, 0,
  0.002899327, 0.001047, 0,
  0.00204919, 0.00074, 0,
  0.001439971, 0.00052, 0,
  0.0009999493, 0.0003611, 0,
  0.0006900786, 0.0002492, 0,
  0.0004760213, 0.0001719, 0,
  0.0003323011, 0.00012, 0,
  0.0002348261, 0.0000848, 0,
  0.0001661505, 0.00006, 0,
  0.000117413, 0.0000424, 0,
  0.00008307527, 0.00003, 0,
  0.00005870652, 0.0000212, 0,
  0.00004150994, 0.00001499, 0,
];

/** Second radiation constant h·c/k in nm·K. */
const c2 = 1.4388e7;

export const frameBlackbodyTableSize = 256;
export const frameBlackbodyTableRows = 16;
export const frameBlackbodyMinTemperature = 500.0;
export const frameBlackbodyMaxTemperature = 40000.0;
export const frameMaxSootExponent = 3.0;

/** The table's colour of one temperature (K) and soot exponent: linear sRGB, largest channel 1 (0 when black). */
export function vdbFrameBlackbodyRgb(temperature: number, exponent: number): Vec3 {
  let x = 0;
  let y = 0;
  let z = 0;
  for (let i = 0; i < cie1931.length; i += 3) {
    const wavelength = cieStart + (i / 3) * cieStep;
    // Planck radiance without constant factors (the scale is removed at the end)
    const radiance = ((wavelength / 550.0) ** -exponent * (550.0 / wavelength) ** 5) / Math.expm1(c2 / (wavelength * temperature));
    x += radiance * cie1931[i]!;
    y += radiance * cie1931[i + 1]!;
    z += radiance * cie1931[i + 2]!;
  }
  const r = Math.max(0, 3.2406 * x - 1.5372 * y - 0.4986 * z);
  const g = Math.max(0, -0.9689 * x + 1.8758 * y + 0.0415 * z);
  const b = Math.max(0, 0.0557 * x - 0.204 * y + 1.057 * z);
  const peak = Math.max(r, g, b);
  return peak > 0 ? [r / peak, g / peak, b / peak] : [0, 0, 0];
}

let table: Float32Array | null = null;

/** RGBA float texels, {@link frameBlackbodyTableSize} × {@link frameBlackbodyTableRows}, row = soot exponent (made once). */
export function vdbFrameBlackbodyTable(): Float32Array {
  if (table) return table;
  const t = new Float32Array(frameBlackbodyTableSize * frameBlackbodyTableRows * 4);
  for (let row = 0; row < frameBlackbodyTableRows; row++) {
    const exponent = (frameMaxSootExponent * row) / (frameBlackbodyTableRows - 1);
    for (let column = 0; column < frameBlackbodyTableSize; column++) {
      const temperature = frameBlackbodyMinTemperature * (frameBlackbodyMaxTemperature / frameBlackbodyMinTemperature) ** (column / (frameBlackbodyTableSize - 1));
      const rgb = vdbFrameBlackbodyRgb(temperature, exponent);
      t.set([rgb[0], rgb[1], rgb[2], 1.0], (row * frameBlackbodyTableSize + column) * 4);
    }
  }
  table = t;
  return t;
}

/**
 * frameBlackbodyColor (CPU twin): the table read with linear blending in log temperature and exponent between the
 * texels (`load`, no sampler), on the f32 table values.
 */
export function vdbFrameBlackbodyColor(temperature: number, exponent: number): Vec3 {
  const t = vdbFrameBlackbodyTable();
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
