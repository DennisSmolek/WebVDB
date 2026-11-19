// #region coarse-first-frame
import { VDBFrameVolume, VDBGridNode, vdbSparseSource, vdbStreamCoarseFirst, type VDBCoarseFirstBind } from '@pmndrs/vdb/three';
// #endregion

import type { Scene, WebGPURenderer } from 'three/webgpu';
import type { VDBGridPool } from '@pmndrs/vdb';

export async function coarseFirstFrame(renderer: WebGPURenderer, pool: VDBGridPool, scene: Scene, url: string): Promise<number> {
    // #region coarse-first-frame
    // Into the froxel renderer: one node, source and volume, made at the first grid (the preview).
    const bind: VDBCoarseFirstBind = {}; // later swaps: pool.flush() + grids.update()
    const { id } = await vdbStreamCoarseFirst(pool, await fetch(url), {
        bind,
        onGrid: (event) => {
            if (event.action !== 'add') return; // a swap: the source and the volume follow by themselves
            pool.flush();
            // Every grid type of the family (a baked FpN preview, an Fp8 full grid): one module reads both.
            const grids = new VDBGridNode(renderer, pool, {
                profile: 'march',
                helpers: ['cellBoundTrilinear', 'hddaCellEnd', 'fastTrilinear'],
                types: event.types,
            });
            bind.grids = grids;
            // followImage: after a swap the source re-reads the grid's boxes and rebuilds its pyramid, and the
            // volume moves its box. No new node, source, volume or shader.
            const source = vdbSparseSource(grids, { density: event.id }, { images: { density: event.grid.image }, followImage: true });
            scene.add(new VDBFrameVolume(source, { extinction: 100 })); // froxels: no voxelRatio scale
        },
    });
    // #endregion
    return id;
}
