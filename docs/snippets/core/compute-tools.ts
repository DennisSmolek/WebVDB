// #region compute-tools
import { decodeToAtlas, gridStats } from '@pmndrs/vdb';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';

export async function computeTools(device: GPUDevice, pool: VDBGridPool, gridId: number): Promise<void> {
    // #region compute-tools
    // The grid must be resident (an update() or flush() ran after its add()).
    const stats = await gridStats(device, pool, gridId, { histogramBins: 64 }).read();
    console.log(stats.min, stats.max, stats.mean, stats.activeVoxelCount);

    // Dense decode of the index box: x fastest, at most 128 voxels per axis, normalized to 0..255.
    const atlas = await decodeToAtlas(device, pool, gridId, { maxDim: 128, format: 'uint8' }).read();
    console.log(atlas.dims, atlas.data.length);

    // To record into your own encoder instead: pass { encoder }, submit it, then `job.submitted().read()`.
    // #endregion
}
