// #region material-params
import { Color, Vector3 } from 'three/webgpu';
// #endregion

import type { VDBVolumeMaterial } from '@pmndrs/vdb/three';

export function materialParams(material: VDBVolumeMaterial): void {
    // #region material-params
    // The parameters are uniforms: change them between frames, nothing recompiles.
    material.densityScale.value = 2; // extinction per world unit at value 1
    material.stepSize.value = 0.5; // march step, in voxels
    material.maxSteps.value = 512;
    material.sunDirection.value.copy(new Vector3(0.3, 1, 0.2).normalize());
    material.sunColor.value.set(new Color(1, 0.95, 0.85));
    material.shadowSteps.value = 8; // 0 turns the sun shadow march off
    material.anisotropy.value = 0.3; // Henyey-Greenstein g of the sun's phase
    material.jitter.value = 1; // per-pixel start jitter; 0 for a deterministic image
    // #endregion
}
