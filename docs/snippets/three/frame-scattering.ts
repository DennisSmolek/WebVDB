// #region frame-scattering

// #endregion

import type { VDBFrameVolume } from '@pmndrs/vdb/three';

export function frameScattering(volume: VDBFrameVolume): void {
    // #region frame-scattering
    const settings = volume.frameRenderer.settings;

    // false: single scattering. true: 1D diffusion toward the light. 'two-sided': 2D slab diffusion.
    // 'octaves': indirect octaves. 'neural' (default): a small trained network for the scene lights.
    settings.multipleScattering = 'octaves';
    settings.scatteringOctaves = 4; // each octave n has extinction a^n, light b^n, phase g c^n
    settings.scatteringOctaveExtinction = 0.5;
    settings.scatteringOctaveContribution = 0.5;
    settings.scatteringOctavePhase = 0.5;

    // Phase functions: 'isotropic', 'henyey-greenstein', 'dual-lobe', 'lorenz-mie' or 'lorenz-mie-rgb'.
    // The renderer's phase is every volume's ('frame' lighting lights them all in one pass with it).
    settings.phase = 'henyey-greenstein';
    settings.phaseG = 0.6;
    // 'volume' lighting only: one volume's own phase, field by field (null: the renderer's again).
    volume.phaseG = 0.3;
    // #endregion
}
