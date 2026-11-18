// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L222–256 (the blackbody table texture and
//   frameBlackbodyColor), L1003–1010 (the 'planck' / 'soot' branch of sampleEmission).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; an `emissionColor` hook of `VDBFrameVolume` (the volume-source seam) instead of a built-in
// spectrum switch; his 'approximate' (Shadertoy) spectrum is not ported (R1 §7, DIVERGES V24). The table is engine-free
// (`@pmndrs/vdb/tools` `vdbFrameBlackbodyTable`, with the CIE 1931 data and a CPU twin of the read).

/**
 * K3.6: his physical emission spectra as an emission colour hook:
 *
 *   const flames = vdbFrameBlackbodyEmission("soot", 1.39);
 *   new VDBFrameVolume(source, { emissionColor: flames, temperatureRange: [0, 3000], emissiveIntensity: 13 });
 *   flames.sootExponent.value = 2.0;   // a uniform: no rebuild
 *
 * `'planck'`: blackbody colour; `'soot'`: × λ^−exponent (0–3). The colour is normalized (largest channel 1); the
 * volume's power still grows with temperature⁴ (`sampleEmission`).
 */

import * as THREE from "three";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { frameBlackbodyMaxTemperature, frameBlackbodyMinTemperature, frameBlackbodyTableRows, frameBlackbodyTableSize, frameMaxSootExponent, vdbFrameBlackbodyTable, vdbFrameEmissionSpectrum } from "../../tools/frame/blackbody.js";

const { clamp, float, int, ivec2, log, max, min, mix, texture, uniform } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;

let tableTexture: THREE.DataTexture | null = null;

/** The 256 × 16 RGBA float table as a texture (made once; read with `load`, no filtering). */
export function getFrameBlackbodyMap(): THREE.DataTexture {
  if (!tableTexture) {
    tableTexture = new THREE.DataTexture(vdbFrameBlackbodyTable(), frameBlackbodyTableSize, frameBlackbodyTableRows, THREE.RGBAFormat, THREE.FloatType);
    tableTexture.minFilter = THREE.NearestFilter;
    tableTexture.magFilter = THREE.NearestFilter;
    tableTexture.generateMipmaps = false;
    tableTexture.needsUpdate = true;
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
}

/**
 * The `emissionColor` hook of his `'planck'` / `'soot'` emission spectra (`VDBFrameVolume` / `vdbVolumeNodes`
 * `emissionColor`). Temperatures ≤ 0 give black (the source's `sampleEmission` already selects them out).
 */
export function vdbFrameBlackbodyEmission(spectrum: "planck" | "soot" = "planck", sootExponent = 1.39): VDBFrameBlackbodyHook {
  const [, exponent] = vdbFrameEmissionSpectrum(spectrum, sootExponent);
  const exponentNode = uniform(exponent);
  const map = texture(getFrameBlackbodyMap());
  const hook = ((temperature: N) => frameBlackbodyColor(map, temperature, exponentNode)) as VDBFrameBlackbodyHook;

  Object.defineProperty(hook, "sootExponent", { value: exponentNode });
  Object.defineProperty(hook, "spectrum", { value: spectrum });

  return hook;
}
