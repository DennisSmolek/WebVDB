// #region three-compute
import { vdbDecodeToAtlas, vdbGridStats } from '@pmndrs/vdb/three';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';
import type { WebGPURenderer } from 'three/webgpu';

export async function threeCompute(renderer: WebGPURenderer, pool: VDBGridPool, id: number): Promise<void> {
    // #region three-compute
    // The core's vdbGridStatsGpu and vdbDecodeToAtlasGpu through the renderer's device; the grid must be resident.
    const stats = await vdbGridStats(renderer, pool, id, { histogramBins: 128 });
    console.log(stats.min, stats.max, stats.mean, stats.activeVoxelCount);

    // The atlas also comes back as a three Data3DTexture, ready for a material.
    const atlas = await vdbDecodeToAtlas(renderer, pool, id, { maxDim: 128, format: 'float32', normalize: false });
    console.log(atlas.dims, atlas.texture);
    // #endregion
}
