// #region grid-node-tsl
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { ivec3, positionWorld, vec3 } from 'three/tsl';
import { VDBGridNode, vdbReadValue, vdbSample } from '@pmndrs/vdb/three';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';
import type { WebGPURenderer } from 'three/webgpu';

export function gridNodeTsl(renderer: WebGPURenderer, pool: VDBGridPool, id: number): MeshBasicNodeMaterial {
    // #region grid-node-tsl
    // profile 'sample' (the default) includes vdbSample; 'read' is smaller and only has vdbReadValue.
    const grids = new VDBGridNode(renderer, pool, { profile: 'sample', sampler: 'trilinear' });

    const material = new MeshBasicNodeMaterial();
    // The grid's value at this fragment's world position, through the grid's own transform
    // (pass { space: 'index' } for index-space positions).
    const density = vdbSample(grids, id, positionWorld);
    material.colorNode = vec3(density);

    // The voxel at integer index coordinates (any profile). `id` may be a number or a uint node.
    const voxel = vdbReadValue(grids, id, ivec3(0, 0, 0));
    material.opacityNode = voxel.add(1).clamp(0, 1);
    // #endregion
    return material;
}
