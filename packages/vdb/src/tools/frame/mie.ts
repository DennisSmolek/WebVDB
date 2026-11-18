// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   scripts/mie-phase-table.mjs L12–147 (the BHMIE droplet sum, the Deirmendjian C1 radius integral,
//   the sphere normalization and the asymmetry; the generator of his frameMieTable.js /
//   frameMieRgbTable.js), src/three-openvdb/frame/core/framePhase.js L61–76 (getFrameMieTexels),
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L250–271 (frameMieTexel, the 'lorenz-mie' and
//   'lorenz-mie-rgb' table reads).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; the generator is a function (the script that writes ./mie-table.ts is
// scripts/k3/mie-phase-table.mjs); the table read is an f64 CPU twin. Divergences: DIVERGES.md (K3.4 rows).

/**
 * Lorenz-Mie phase tables of mjurczyk's `'lorenz-mie'` / `'lorenz-mie-rgb'` phase functions (K3.4,
 * R1 §5 and §7): cumulus water droplets (Deirmendjian C1, n(r) = r⁶ e^(−1.5 r), r in µm), the Mie
 * series after Bohren & Huffman's BHMIE, absorption ignored. Each table is the phase divided by the
 * isotropic phase 1/(4π) at 2048 angles θ = π u² (dense near the forward peak). The committed tables
 * (`./mie-table.ts`) are this generator's output (`scripts/k3/mie-phase-table.mjs`), not data copied
 * from his repo; the unit test regenerates them.
 */

import { frameMieRgbTable, frameMieTable, frameMieTableSize } from "./mie-table.js";

/** Texels of the tables (his `frameMieTableSize`). */
export const vdbFrameMieTableSize = frameMieTableSize;

/** Asymmetry g of the 'lorenz-mie' table (his `frameMieAsymmetry`; 'lorenz-mie-rgb' uses it too). */
export const vdbFrameMieAsymmetry = 0.8526;

/** Lorenz-Mie mode of 'lorenz-mie-rgb' in the phase vector (his `frameMieRgbMode`). */
export const vdbFrameMieRgbMode = 2.0;

/** The channels of 'lorenz-mie-rgb': wavelength (µm) and the refractive index of water there. */
export const vdbFrameMieRgbChannels: readonly { name: string; wavelength: number; refractiveIndex: number }[] = [
  { name: "Red", wavelength: 0.65, refractiveIndex: 1.3311 },
  { name: "Green", wavelength: 0.55, refractiveIndex: 1.3335 },
  { name: "Blue", wavelength: 0.45, refractiveIndex: 1.3374 },
];

/** 'lorenz-mie': 550 nm, n = 1.335. */
export const vdbFrameMieChannel = { wavelength: 0.55, refractiveIndex: 1.335 } as const;

export interface VDBFrameMieTableOptions {
  /** Table texels (2048). */
  tableSize?: number;
  /** Droplet radius range (µm) and trapezoid steps (0.05–30, 1200). */
  radiusMin?: number;
  radiusMax?: number;
  radiusSteps?: number;
}

export interface VDBFrameMieTableResult {
  /** Phase / (1 / 4π) at θ = π u², u = i / (size − 1). */
  table: Float64Array;
  /** Mean cosine of the scattering angle. */
  asymmetry: number;
  /** The forward peak (largest value). */
  peak: number;
}

/**
 * The droplet-distribution phase table at one wavelength (µm) and refractive index: his
 * `createTable`, f64. About 1–2 s per table at the default size (1201 radii × ~200 terms × 2048 angles).
 */
export function vdbFrameMieTable(wavelength: number, refractiveIndex: number, o: VDBFrameMieTableOptions = {}): VDBFrameMieTableResult {
  const tableSize = o.tableSize ?? 2048;
  const radiusMin = o.radiusMin ?? 0.05;
  const radiusMax = o.radiusMax ?? 30.0;
  const radiusSteps = o.radiusSteps ?? 1200;
  const angles = Array.from({ length: tableSize }, (_, index) => {
    const u = index / (tableSize - 1);
    return Math.PI * u * u;
  });
  const cosines = angles.map(Math.cos);
  const intensity = new Float64Array(tableSize);

  // (|S1|² + |S2|²) / 2 of one droplet, times weight, added to intensity (BHMIE).
  const addDroplet = (x: number, m: number, weight: number) => {
    const y = m * x;
    const termCount = Math.round(x + 4.0 * Math.cbrt(x) + 2.0);
    const recurrenceCount = Math.round(Math.max(termCount, Math.abs(y))) + 15;
    // Logarithmic derivative D(n), downward recurrence
    const logDerivative = new Float64Array(recurrenceCount + 1);
    for (let n = recurrenceCount; n >= 2; n--) logDerivative[n - 1] = n / y - 1.0 / (logDerivative[n]! + n / y);
    // Riccati-Bessel functions psi and chi, xi = psi − i chi
    let psi0 = Math.cos(x);
    let psi1 = Math.sin(x);
    let chi0 = -Math.sin(x);
    let chi1 = Math.cos(x);
    const aRe = new Float64Array(termCount);
    const aIm = new Float64Array(termCount);
    const bRe = new Float64Array(termCount);
    const bIm = new Float64Array(termCount);
    for (let n = 1; n <= termCount; n++) {
      const psi = ((2 * n - 1) * psi1) / x - psi0;
      const chi = ((2 * n - 1) * chi1) / x - chi0;
      // an = (fa psi − psi1) / (fa xi − xi1), bn the same with fb
      const coefficient = (factor: number, re: Float64Array, im: Float64Array) => {
        const numerator = factor * psi - psi1;
        const denominatorRe = factor * psi - psi1;
        const denominatorIm = -(factor * chi - chi1);
        const scale = denominatorRe * denominatorRe + denominatorIm * denominatorIm;
        re[n - 1] = (numerator * denominatorRe) / scale;
        im[n - 1] = (-numerator * denominatorIm) / scale;
      };
      coefficient(logDerivative[n]! / m + n / x, aRe, aIm);
      coefficient(m * logDerivative[n]! + n / x, bRe, bIm);
      psi0 = psi1;
      psi1 = psi;
      chi0 = chi1;
      chi1 = chi;
    }
    for (let j = 0; j < tableSize; j++) {
      const mu = cosines[j]!;
      let pi0 = 0.0;
      let pi1 = 1.0;
      let s1Re = 0.0;
      let s1Im = 0.0;
      let s2Re = 0.0;
      let s2Im = 0.0;
      for (let n = 1; n <= termCount; n++) {
        const fn = (2 * n + 1) / (n * (n + 1));
        const tau = n * mu * pi1 - (n + 1) * pi0;
        s1Re += fn * (aRe[n - 1]! * pi1 + bRe[n - 1]! * tau);
        s1Im += fn * (aIm[n - 1]! * pi1 + bIm[n - 1]! * tau);
        s2Re += fn * (aRe[n - 1]! * tau + bRe[n - 1]! * pi1);
        s2Im += fn * (aIm[n - 1]! * tau + bIm[n - 1]! * pi1);
        const pi = ((2 * n + 1) / n) * mu * pi1 - ((n + 1) / n) * pi0;
        pi0 = pi1;
        pi1 = pi;
      }
      intensity[j] = intensity[j]! + weight * 0.5 * (s1Re * s1Re + s1Im * s1Im + s2Re * s2Re + s2Im * s2Im);
    }
  };

  // Normalize to the isotropic phase: the integral over the sphere is 4π, isotropic is 1 (trapezoids in θ).
  const sphereIntegral = (values: ArrayLike<number>) => {
    let sum = 0.0;
    for (let j = 1; j < tableSize; j++) {
      const left = values[j - 1]! * Math.sin(angles[j - 1]!);
      const right = values[j]! * Math.sin(angles[j]!);
      sum += 0.5 * (left + right) * (angles[j]! - angles[j - 1]!);
    }
    return 2.0 * Math.PI * sum;
  };

  const radiusStep = (radiusMax - radiusMin) / radiusSteps;
  for (let i = 0; i <= radiusSteps; i++) {
    const radius = radiusMin + i * radiusStep;
    // Trapezoid rule over the radius, n(r) = r⁶ exp(−1.5 r)
    const weight = radius ** 6 * Math.exp(-1.5 * radius) * radiusStep * (i === 0 || i === radiusSteps ? 0.5 : 1.0);
    addDroplet((2.0 * Math.PI * radius) / wavelength, refractiveIndex, weight);
  }

  const scale = (4.0 * Math.PI) / sphereIntegral(intensity);
  const table = new Float64Array(tableSize);
  let peak = -Infinity;
  for (let j = 0; j < tableSize; j++) {
    table[j] = intensity[j]! * scale;
    peak = Math.max(peak, table[j]!);
  }
  const asymmetry = sphereIntegral(Array.from(table, (v, j) => v * cosines[j]!)) / (4.0 * Math.PI);
  return { table, asymmetry, peak };
}

/** His table number format: 6 significant digits. */
export const vdbFrameMieFormat = (value: number): number => Number(value.toPrecision(6));

/**
 * Texels of the Lorenz-Mie texture (his `getFrameMieTexels`): the 'lorenz-mie-rgb' tables in red,
 * green, blue and the 'lorenz-mie' table in alpha (one texture keeps the shaders under 16 textures).
 */
export function vdbFrameMieTexels(): Float32Array {
  const texels = new Float32Array(frameMieTableSize * 4);
  for (let i = 0; i < frameMieTableSize; i++) {
    texels[i * 4] = frameMieRgbTable[i * 3]!;
    texels[i * 4 + 1] = frameMieRgbTable[i * 3 + 1]!;
    texels[i * 4 + 2] = frameMieRgbTable[i * 3 + 2]!;
    texels[i * 4 + 3] = frameMieTable[i]!;
  }
  return texels;
}

let defaultTexels: Float32Array | null = null;

/**
 * frameMieTexel: the table texel pair at the angle of `cosTheta` (u = √(θ/π), texel u (N − 1),
 * `load` of index and index + 1, linear blend), RGBA = (red, green, blue, 'lorenz-mie').
 */
export function vdbFrameMieTexel(cosTheta: number, texels?: Float32Array): [number, number, number, number] {
  const t = texels ?? (defaultTexels ??= vdbFrameMieTexels());
  const u = Math.sqrt(Math.acos(Math.min(1, Math.max(-1, cosTheta))) / Math.PI);
  const texel = u * (frameMieTableSize - 1);
  const index = Math.min(Math.trunc(texel), frameMieTableSize - 2);
  const f = texel - index;
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    const a = t[index * 4 + c]!;
    const b = t[(index + 1) * 4 + c]!;
    out[c] = a + (b - a) * f;
  }
  return out;
}

/** frameMiePhase: the 'lorenz-mie' phase / isotropic at cos θ. */
export const vdbFrameMiePhase = (cosTheta: number): number => vdbFrameMieTexel(cosTheta)[3];

/** frameMieRgbPhase: the 'lorenz-mie-rgb' phases / isotropic of the three channels. */
export const vdbFrameMieRgbPhase = (cosTheta: number): [number, number, number] => {
  const t = vdbFrameMieTexel(cosTheta);
  return [t[0], t[1], t[2]];
};
