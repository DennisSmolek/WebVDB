// #region frame-sparse
import { VDBFrameVolume, VDBGridNode, vdbSparseSource } from '@pmndrs/vdb/three';
// #endregion

import type { Scene, WebGPURenderer } from 'three/webgpu';
import type { VDBGridPool } from '@pmndrs/vdb';

export function frameSparse(renderer: WebGPURenderer, pool: VDBGridPool, scene: Scene, image: Uint32Array): VDBFrameVolume {
    // #region frame-sparse
    const id = pool.add(image);
    pool.flush();

    // A sparse source: a pool grid, at full resolution. The module needs the march profile and the
    // helpers the source's HDDA and node-average paths use.
    const grids = new VDBGridNode(renderer, pool, {
        profile: 'march',
        helpers: ['cellBoundTrilinear', 'hddaCellEnd', 'nodeAverage'],
    });
    const source = vdbSparseSource(
        grids,
        { density: id }, // optional: mask, albedo (one grid or three), emission
        { lod: 'coarse-pyramid', pyramidMax: 128, pyramidBuild: 'gpu', images: { density: image } },
    );

    const volume = new VDBFrameVolume(source, { extinction: 100, hdda: false }); // the box is the grid's world box
    scene.add(volume);
    // #endregion
    return volume;
}
