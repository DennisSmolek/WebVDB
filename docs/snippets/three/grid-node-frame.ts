// #region grid-node-frame
import { VDBGridNode } from '@pmndrs/vdb/three';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';
import type { WebGPURenderer } from 'three/webgpu';
import type { Object3D, Camera } from 'three/webgpu';

export function gridNodeFrame(renderer: WebGPURenderer, pool: VDBGridPool, scene: Object3D, camera: Camera, replacement: Uint32Array, id: number): () => void {
    // #region grid-node-frame
    const grids = new VDBGridNode(renderer, pool, { profile: 'sample' });
    grids.check(); // throws VDB_TYPE_NOT_COMPILED if the pool holds a grid type the module lacks

    const frame = () => {
        pool.replace(id, replacement); // any pool change is staged...
        pool.flush(); // ...and must be on the GPU before three renders
        grids.update(); // rebinds the node when the pool grew into a new buffer (pool.version moved)
        renderer.render(scene, camera);
    };
    // #endregion
    return frame;
}
