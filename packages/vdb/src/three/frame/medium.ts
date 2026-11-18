// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameMedium.js L78–90 (getFrameMediumProps).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; the preset table is engine-free in `tools/frame/medium.ts` (`vdbFrameMediumPresets`).
// Divergences: none.

import * as THREE from "three";
import { vdbFrameMediumProps, type VDBFrameMediumPreset } from "../../tools/frame/medium.js";

/** {@link getFrameMediumProps} result: `VDBFrameVolume` props of a measured medium. */
export interface VDBFrameMediumVolumeProps {
  /** Linear albedo. */
  baseColor: THREE.Color;
  /** (1, 1, 1): extinction depends on the scene scale, so a preset does not set it. */
  extinctionScale: THREE.Vector3;
  angstromExponent: number;
  phase: VDBFrameMediumPreset["phase"];
  phaseG: number;
  phaseGExponent: number;
}

/**
 * `VDBFrameVolume` props of a measured medium (`'cloud'`, `'fog'`, `'ice'`, `'soot'`, `'white-smoke'`,
 * `'dark-smoke'`, `'haze'`, `'dust'`; `@pmndrs/vdb/tools` `vdbFrameMediumPresets`): base colour, extinction
 * scale, Ångström exponent and phase. Use them as constructor props or assign them to a volume, and turn on
 * the renderer's `settings.spectral` (his examples also set the renderer's 'frame' phase to the preset's).
 * Unknown names give null.
 */
export const getFrameMediumProps = (name: string): VDBFrameMediumVolumeProps | null => {
  const props = vdbFrameMediumProps(name);

  if (!props) {
    return null;
  }

  return {
    baseColor: new THREE.Color().setRGB(...props.albedo, THREE.LinearSRGBColorSpace),
    extinctionScale: new THREE.Vector3(...props.extinctionScale),
    angstromExponent: props.angstromExponent,
    phase: props.phase,
    phaseG: props.phaseG,
    phaseGExponent: props.phaseGExponent,
  };
};
