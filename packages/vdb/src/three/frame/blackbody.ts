// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L222–256 (the blackbody table texture and
//   frameBlackbodyColor), L1003–1010 (the 'planck' / 'soot' branch of sampleEmission).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; an `emissionColor` hook of `VDBFrameVolume` (the volume-source seam) instead of a built-in
// spectrum switch; his 'approximate' (Shadertoy) spectrum is not ported (R1 §7, DIVERGES V24). The table is engine-free
// (`@pmndrs/vdb/tools` `vdbFrameBlackbodyTable`, with a CPU twin of the read). D33: the colour matching functions are
// the analytic fit of Wyman, Sloan & Shirley (JCGT 2(2), 2013), not his CIE 1931 table (CC BY-SA 4.0); `options.cmf`
// takes others (e.g. the exact table, supplied by the caller under its own licence).

/**
 * K3.6: his physical emission spectra as an emission colour hook:
 *
 *   const flames = vdbFrameBlackbodyEmission("soot", 1.39);
 *   new VDBFrameVolume(source, { emissionColor: flames, temperatureRange: [0, 3000], emissiveIntensity: 13 });
 *   flames.sootExponent.value = 2.0;   // a uniform: no rebuild
 *
 * `'planck'`: blackbody colour; `'soot'`: × λ^−exponent (0–3). The colour is normalized (largest channel 1); the
 * volume's power still grows with temperature⁴ (`sampleEmission`). The CIE 1931 colour matching functions are the
 * analytic fit of Wyman, Sloan & Shirley (2013; D33): against the CIE table the normalized colour moves by ≤ 0.021
 * per channel over 1,000–40,000 K (chromaticity Δxy ≤ 0.0016 from 3,000 K up, 0.015 at 1,000 K where the fit's red
 * tail past ~680 nm is short).
 */

import * as THREE from "three";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import {
  frameBlackbodyMaxTemperature,
  frameBlackbodyMinTemperature,
  frameBlackbodyTableRows,
  frameBlackbodyTableSize,
  frameMaxSootExponent,
  vdbCie1931Fit,
  vdbFrameBlackbodyTable,
  vdbFrameEmissionSpectrum,
  type VDBColorMatchingFunctions,
} from "../../tools/frame/blackbody.js";

const { clamp, float, int, ivec2, log, max, min, mix, texture, uniform } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;

const tableTextures = new WeakMap<VDBColorMatchingFunctions, THREE.DataTexture>();

/** The 256 × 16 RGBA float table as a texture (made once per `cmf`; read with `load`, no filtering). */
export function getFrameBlackbodyMap(cmf: VDBColorMatchingFunctions = vdbCie1931Fit): THREE.DataTexture {
  let tableTexture = tableTextures.get(cmf);
  if (!tableTexture) {
    tableTexture = new THREE.DataTexture(vdbFrameBlackbodyTable(cmf), frameBlackbodyTableSize, frameBlackbodyTableRows, THREE.RGBAFormat, THREE.FloatType);
    tableTexture.minFilter = THREE.NearestFilter;
    tableTexture.magFilter = THREE.NearestFilter;
    tableTexture.generateMipmaps = false;
    tableTexture.needsUpdate = true;
    tableTextures.set(cmf, tableTexture);
  }

  return tableTexture;
}

/** frameBlackbodyColor: the table at a temperature (K) and soot exponent, linear in log temperature and exponent. */
export const frameBlackbodyColor = (map: N, temperature: N, exponent: N): N => {
  const temperatureScale = Math.log(frameBlackbodyMaxTemperature / frameBlackbodyMinTemperature);
  const logTemperature = log(max(temperature, 1e-3).div(frameBlackbodyMinTemperature));
  const x = clamp(logTemperature.div(temperatureScale), 0.0, 1.0)
    .mul(frameBlackbodyTableSize - 1)
    .toVar();
  const y = clamp(exponent.div(frameMaxSootExponent), 0.0, 1.0)
    .mul(frameBlackbodyTableRows - 1)
    .toVar();
  const column = min(int(x), int(frameBlackbodyTableSize - 2)).toVar();
  const row = min(int(y), int(frameBlackbodyTableRows - 2)).toVar();
  const blendX = x.sub(float(column)).toVar();
  const texel = (offsetX: number, offsetY: number) => map.load(ivec2(column.add(offsetX), row.add(offsetY))).rgb;

  return mix(mix(texel(0, 0), texel(1, 0), blendX), mix(texel(0, 1), texel(1, 1), blendX), y.sub(float(row)));
};

/** An emission colour hook with its soot exponent uniform. */
export interface VDBFrameBlackbodyHook {
  (temperature: N): N;
  /** The soot exponent (0–3; 'planck': 0). Set `.value` at runtime. */
  readonly sootExponent: { value: number };
  readonly spectrum: "planck" | "soot";
  /** The table texture this hook reads (one per colour matching functions; D33). */
  readonly table: THREE.DataTexture;
}

/**
 * The `emissionColor` hook of his `'planck'` / `'soot'` emission spectra (`VDBFrameVolume` / `vdbVolumeNodes`
 * `emissionColor`). Temperatures ≤ 0 give black (the source's `sampleEmission` already selects them out).
 * `options.cmf`: the CIE 1931 colour matching functions (x̄, ȳ, z̄ at a wavelength in nm, read at 380–780 nm in 5 nm
 * steps); default the analytic fit `vdbCie1931Fit` (D33). Pass the exact CIE table here to match his colours exactly.
 */
export function vdbFrameBlackbodyEmission(spectrum: "planck" | "soot" = "planck", sootExponent = 1.39, options: { cmf?: VDBColorMatchingFunctions } = {}): VDBFrameBlackbodyHook {
  const [, exponent] = vdbFrameEmissionSpectrum(spectrum, sootExponent);
  const exponentNode = uniform(exponent);
  const table = getFrameBlackbodyMap(options.cmf);
  const map = texture(table);
  const hook = ((temperature: N) => frameBlackbodyColor(map, temperature, exponentNode)) as VDBFrameBlackbodyHook;

  Object.defineProperty(hook, "sootExponent", { value: exponentNode });
  Object.defineProperty(hook, "spectrum", { value: spectrum });
  Object.defineProperty(hook, "table", { value: table });

  return hook;
}
