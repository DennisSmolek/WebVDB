// #region coarse-first
import { vdbStreamCoarseFirst } from '@pmndrs/vdb/three';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';
import type { VDBGridNode, VDBVolumeMaterial } from '@pmndrs/vdb/three';

export async function coarseFirst(pool: VDBGridPool, grids: VDBGridNode, material: VDBVolumeMaterial, url: string): Promise<number> {
    // #region coarse-first
    // A file baked with bakeCoarseFirst() holds the preview grid first. The preview is added with the
    // full size reserved, then replaced in place when the full grid arrives: the same pool id throughout.
    const { id } = await vdbStreamCoarseFirst(pool, await fetch(url), {
        onGrid: ({ action, final, voxelRatio }) => {
            pool.flush(); // the preview, or the replacement, reaches the GPU
            grids.update();
            // A preview's voxels are voxelRatio times larger: keep the same optical depth, then restore it.
            material.densityScale.value = final ? 1 : voxelRatio;
            console.log(action, final ? 'full grid' : 'preview');
        },
    });
    // #endregion
    return id;
}
