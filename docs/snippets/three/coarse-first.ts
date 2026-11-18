// #region coarse-first
import { vdbStreamCoarseFirst } from '@pmndrs/vdb/three';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';
import type { Mesh } from 'three';
import type { VDBGridNode, VDBVolumeMaterial } from '@pmndrs/vdb/three';

export async function coarseFirst(pool: VDBGridPool, grids: VDBGridNode, material: VDBVolumeMaterial, mesh: Mesh, url: string): Promise<number> {
    // #region coarse-first
    // A file baked with bakeCoarseFirst() holds the preview grid first. The preview is added with the
    // full size reserved, then replaced in place when the full grid arrives: the same pool id throughout.
    const { id } = await vdbStreamCoarseFirst(pool, await fetch(url), {
        // After each swap: pool.flush() + grids.update(), the proxy box from pool.info(id), and densityScale
        // = its current value x voxelRatio (a preview's voxels are larger), back to the base at the full grid.
        bind: { grids, material, mesh },
        onGrid: ({ action, final }) => console.log(action, final ? 'full grid' : 'preview'),
    });
    // #endregion
    return id;
}
