// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameMedium.js L1–76 (frameMediumPresets with their citations),
//   L78–90 (getFrameMediumProps, engine-free here).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript, engine-free (albedo and extinction scale as arrays; the three adapter
// `getFrameMediumProps` in `@pmndrs/vdb/three` makes his `Color` / `Vector3`). He has no generator for the
// table: its values are midpoints and fits of the cited measurements. `vdbFrameMediumMeasurements` holds
// those measurements as his notes cite them and `vdbFrameMediumFromMeasurements` is the fit (midpoint of a
// range, green albedo = mean of red and blue when not measured, phaseGExponent from the blue and red g); the
// unit tests check that the ported table is that fit at his precision (3 decimals). Divergences: none.

import { VDBToolsError } from "../errors.js";

/** A measured medium: linear albedo (red 650 nm, green 550 nm, blue 450 nm), Ångström exponent, phase. */
export interface VDBFrameMediumPreset {
  /** Linear albedo (red, green, blue). */
  readonly albedo: readonly [number, number, number];
  /** The Ångström exponent: how the extinction falls with wavelength. */
  readonly angstromExponent: number;
  /** The phase function family: Henyey-Greenstein, or the Lorenz-Mie RGB tables. */
  readonly phase: "henyey-greenstein" | "lorenz-mie-rgb";
  /** g at 550 nm. */
  readonly phaseG: number;
  /** Fits g at 450 nm and 650 nm: g of a channel is phaseG × (λ / 550)^−phaseGExponent. */
  readonly phaseGExponent: number;
}

// NOTE Measured media. Albedo is linear (red 650 nm, green 550 nm, blue 450 nm). A range gives its midpoint.
// NOTE A green albedo that is not measured is the mean of red and blue.
// NOTE phaseG is g at 550 nm. phaseGExponent fits g at 450 nm and 650 nm. Extinction depends on the scene scale, so it is not here.
/** The measured medium presets by name: `cloud`, `fog`, `ice`, `soot`, `white-smoke`, `dark-smoke`, `haze` and `dust`. */
export const vdbFrameMediumPresets: Readonly<Record<string, VDBFrameMediumPreset>> = {
  // NOTE Mie, Pope & Fry 1997. g 0.857 / 0.853 / 0.850 (blue / green / red): the 'lorenz-mie-rgb' tables.
  cloud: { albedo: [1.0, 1.0, 1.0], angstromExponent: 0.0, phase: "lorenz-mie-rgb", phaseG: 0.853, phaseGExponent: 0.0 },
  // NOTE Hess et al. 1998 (OPAC). g 0.866 at 550 nm.
  fog: { albedo: [1.0, 1.0, 1.0], angstromExponent: 0.0, phase: "henyey-greenstein", phaseG: 0.866, phaseGExponent: 0.0 },
  // NOTE OPAC, Baran 2012. g 0.76 to 0.78.
  ice: { albedo: [1.0, 1.0, 1.0], angstromExponent: 0.0, phase: "henyey-greenstein", phaseG: 0.77, phaseGExponent: 0.0 },
  // NOTE Fresh soot (black carbon), Bond & Bergstrom 2006. Albedo 0.20 to 0.30 (flat), g 0.3 to 0.4 (estimate).
  soot: { albedo: [0.25, 0.25, 0.25], angstromExponent: 1.0, phase: "henyey-greenstein", phaseG: 0.35, phaseGExponent: 0.0 },
  // NOTE Boreal forest smoke, Dubovik et al. 2002. Angstrom 1.0 to 2.3, g 0.69 / 0.61 (blue / red).
  "white-smoke": { albedo: [0.935, 0.9375, 0.94], angstromExponent: 1.65, phase: "henyey-greenstein", phaseG: 0.645, phaseGExponent: 0.335 },
  // NOTE Savanna smoke, Dubovik et al. 2002. Angstrom 1.4 to 2.2, g 0.64 / 0.53 (blue / red).
  "dark-smoke": { albedo: [0.84, 0.86, 0.88], angstromExponent: 1.8, phase: "henyey-greenstein", phaseG: 0.577, phaseGExponent: 0.513 },
  // NOTE Urban haze, Dubovik et al. 2002. Angstrom 1.2 to 2.5, g 0.68 / 0.59 (blue / red).
  haze: { albedo: [0.97, 0.975, 0.98], angstromExponent: 1.85, phase: "henyey-greenstein", phaseG: 0.629, phaseGExponent: 0.386 },
  // NOTE Saharan dust, Dubovik et al. 2002, Kim et al. 2011. Angstrom -0.1 to 0.7, g 0.73 / 0.752 / 0.71 (blue / green / red).
  dust: { albedo: [0.98, 0.944, 0.93], angstromExponent: 0.3, phase: "henyey-greenstein", phaseG: 0.752, phaseGExponent: 0.076 },
};

/** The preset names of {@link vdbFrameMediumPresets} (his GUI 'Medium' options without 'custom'). */
export const vdbFrameMediumNames = Object.keys(vdbFrameMediumPresets) as readonly string[];

/** Engine-free {@link VDBFrameMediumProps}: the volume props of a preset. */
export interface VDBFrameMediumProps {
  /** Linear albedo (the volume's `baseColor`). */
  albedo: [number, number, number];
  /** Always (1, 1, 1): extinction depends on the scene scale. */
  extinctionScale: [number, number, number];
  /** The preset's Ångström exponent. */
  angstromExponent: number;
  /** The preset's phase function family. */
  phase: VDBFrameMediumPreset["phase"];
  /** g at 550 nm. */
  phaseG: number;
  /** How g changes with wavelength (see {@link VDBFrameMediumPreset.phaseGExponent}). */
  phaseGExponent: number;
}

/**
 * His `getFrameMediumProps` without three: the volume props of a preset (base colour, extinction scale,
 * Ångström exponent, phase). Unknown names give null.
 */
export function vdbFrameMediumProps(name: string): VDBFrameMediumProps | null {
  const preset = Object.prototype.hasOwnProperty.call(vdbFrameMediumPresets, name) ? vdbFrameMediumPresets[name] : undefined;

  if (!preset) {
    return null;
  }

  return {
    albedo: [...preset.albedo],
    extinctionScale: [1.0, 1.0, 1.0],
    angstromExponent: preset.angstromExponent,
    phase: preset.phase,
    phaseG: preset.phaseG,
    phaseGExponent: preset.phaseGExponent,
  };
}

// ── The cited measurements and the fit that gives the table ─────────────────────────────────────────

/** A value or a measured range [min, max] (its midpoint is used). */
export type VDBFrameMeasured = number | readonly [number, number];

/**
 * The measurements his notes cite for each preset. `albedo`: red, green (null: not measured) and blue;
 * `g`: the asymmetry at 550 nm, or blue and red (and green when measured); `asymmetryFromMieRgb`: g is the
 * 'lorenz-mie-rgb' tables' green asymmetry (cloud).
 */
export interface VDBFrameMediumMeasurement {
  /** The source of the numbers. */
  readonly citation: string;
  /** Albedo as measured: red, green (null: not measured) and blue; each a value or a range. */
  readonly albedo: readonly [VDBFrameMeasured, VDBFrameMeasured | null, VDBFrameMeasured];
  /** The Ångström exponent, a value or a range. */
  readonly angstromExponent: VDBFrameMeasured;
  /** The phase function family. */
  readonly phase: VDBFrameMediumPreset["phase"];
  /** The measured asymmetry g: at 550 nm (`green`), at 450 nm (`blue`) and 650 nm (`red`), or taken from the Mie tables. */
  readonly g: { readonly green?: VDBFrameMeasured; readonly blue?: number; readonly red?: number; readonly asymmetryFromMieRgb?: boolean };
}

/** The measurements of his frameMedium.js notes (Pope & Fry 1997, Hess 1998, Baran 2012, Bond & Bergstrom 2006, Dubovik 2002, Kim 2011). */
export const vdbFrameMediumMeasurements: Readonly<Record<string, VDBFrameMediumMeasurement>> = {
  cloud: { citation: "Mie, Pope & Fry 1997", albedo: [1, 1, 1], angstromExponent: 0, phase: "lorenz-mie-rgb", g: { asymmetryFromMieRgb: true } },
  fog: { citation: "Hess et al. 1998 (OPAC)", albedo: [1, 1, 1], angstromExponent: 0, phase: "henyey-greenstein", g: { green: 0.866 } },
  ice: { citation: "OPAC, Baran 2012", albedo: [1, 1, 1], angstromExponent: 0, phase: "henyey-greenstein", g: { green: [0.76, 0.78] } },
  soot: { citation: "Bond & Bergstrom 2006", albedo: [[0.2, 0.3], [0.2, 0.3], [0.2, 0.3]], angstromExponent: 1, phase: "henyey-greenstein", g: { green: [0.3, 0.4] } },
  "white-smoke": { citation: "Dubovik et al. 2002 (boreal forest)", albedo: [0.935, null, 0.94], angstromExponent: [1.0, 2.3], phase: "henyey-greenstein", g: { blue: 0.69, red: 0.61 } },
  "dark-smoke": { citation: "Dubovik et al. 2002 (savanna)", albedo: [0.84, null, 0.88], angstromExponent: [1.4, 2.2], phase: "henyey-greenstein", g: { blue: 0.64, red: 0.53 } },
  haze: { citation: "Dubovik et al. 2002 (urban)", albedo: [0.97, null, 0.98], angstromExponent: [1.2, 2.5], phase: "henyey-greenstein", g: { blue: 0.68, red: 0.59 } },
  dust: { citation: "Dubovik et al. 2002, Kim et al. 2011 (Saharan)", albedo: [0.98, 0.944, 0.93], angstromExponent: [-0.1, 0.7], phase: "henyey-greenstein", g: { blue: 0.73, green: 0.752, red: 0.71 } },
};

const midpoint = (v: VDBFrameMeasured) => (typeof v === "number" ? v : (v[0] + v[1]) / 2);

/**
 * The phase g exponent e of g(λ) = g₅₅₀ (λ / 550)^−e through g at 450 nm (blue) and 650 nm (red):
 * e = ln(g_blue / g_red) / ln(650 / 450).
 */
export function vdbFrameFitPhaseGExponent(gBlue: number, gRed: number): number {
  return Math.log(gBlue / gRed) / Math.log(650 / 450);
}

/**
 * A preset from its measurements (unrounded): midpoints of ranges; a green albedo that is not measured is the
 * mean of red and blue; phaseGExponent fits the blue and red g; g at 550 nm is the measured green g, else the
 * blue g moved to 550 nm by the fit. `mieRgbGreenAsymmetry` is the green asymmetry of the 'lorenz-mie-rgb'
 * tables (`vdbFrameMieTable`), for cloud.
 * @param m The cited measurements of one preset.
 * @param mieRgbGreenAsymmetry Required when `m.g.asymmetryFromMieRgb` is set; ignored otherwise.
 */
export function vdbFrameMediumFromMeasurements(m: VDBFrameMediumMeasurement, mieRgbGreenAsymmetry?: number): VDBFrameMediumPreset {
  const red = midpoint(m.albedo[0]);
  const blue = midpoint(m.albedo[2]);
  const green = m.albedo[1] === null ? (red + blue) / 2 : midpoint(m.albedo[1]);
  const { g } = m;
  const phaseGExponent = g.blue !== undefined && g.red !== undefined ? vdbFrameFitPhaseGExponent(g.blue, g.red) : 0;
  let phaseG: number;

  if (g.asymmetryFromMieRgb) {
    if (mieRgbGreenAsymmetry === undefined) throw new VDBToolsError("VDB_BAD_OPTION", "vdbFrameMediumFromMeasurements: pass the 'lorenz-mie-rgb' green asymmetry");
    phaseG = mieRgbGreenAsymmetry;
  } else if (g.green !== undefined) {
    phaseG = midpoint(g.green);
  } else {
    phaseG = g.blue! * Math.pow(450 / 550, phaseGExponent);
  }

  return { albedo: [red, green, blue], angstromExponent: midpoint(m.angstromExponent), phase: m.phase, phaseG, phaseGExponent };
}
