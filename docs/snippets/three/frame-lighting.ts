// #region frame-lighting
import { vdbFrameLights } from '@pmndrs/vdb/three';
// #endregion

import type { VDBFrameVolume } from '@pmndrs/vdb/three';

export function frameLighting(volume: VDBFrameVolume): void {
    // #region frame-lighting
    const settings = volume.frameRenderer.settings; // shared by every volume that uses this renderer

    // 'volume': each volume lights itself with its own shadow march.
    // 'frame' (default): a world shadow grid shared by all volumes, with up to 4 cascades.
    settings.lighting = 'frame';
    settings.shadowResolution = 64;
    settings.shadowCascades = 2;

    // The scene's environment as lights: 'outward', 'directional' (default) or 'spherical-harmonics'.
    settings.environmentLighting = 'spherical-harmonics';
    settings.environmentShOrder = 2;

    // Which of the scene's light types this volume uses.
    volume.lights = vdbFrameLights.useDirectionalLights | vdbFrameLights.usePointLights | vdbFrameLights.useEnvironment;
    // #endregion
}
