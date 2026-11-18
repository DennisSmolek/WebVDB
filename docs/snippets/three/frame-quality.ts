// #region frame-quality

// #endregion

import type { VDBFrameVolume } from '@pmndrs/vdb/three';

export function frameQuality(volume: VDBFrameVolume): void {
    // #region frame-quality
    const settings = volume.frameRenderer.settings;
    settings.width = 256; // froxel grid, default 512 x 512 x 24
    settings.height = 256;
    settings.depth = 32;
    settings.pixelsPerFroxel = 4; // at most one froxel per 4 x 4 canvas pixels (0: no cap)
    settings.subSteps = 16;
    settings.temporal = true; // reproject the froxels over frames
    settings.jitter = true;

    volume.shadowSteps = 16; // the shadow march of this volume
    volume.mipmaps = true; // sample the source's coarser levels when a froxel spans many voxels
    volume.hdda = true; // sparse sources: skip empty space in the shadow march (same image, off by default)
    // #endregion
}
