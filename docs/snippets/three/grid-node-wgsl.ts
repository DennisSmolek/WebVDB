// #region grid-node-wgsl
import { wgslFn } from 'three/tsl';
import { VDBGridNode } from '@pmndrs/vdb/three';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';
import type { WebGPURenderer } from 'three/webgpu';

export function gridNodeWgsl(renderer: WebGPURenderer, pool: VDBGridPool): unknown {
    // #region grid-node-wgsl
    const grids = new VDBGridNode(renderer, pool, { profile: 'sample' });

    // grids.module is the generated WGSL module as a code node. wgslFn parses only the first `fn`
    // in your text, so the module's helpers (vdbAccessorInit, vdbSample, ...) come in as an include.
    const shade = wgslFn(
        /* wgsl */ `
        fn shade(g: u32, p: vec3f) -> f32 {
            var acc = vdbAccessorInit(g);
            return vdbSample(&acc, vdbWorldToIndex(g, p));
        }`,
        [grids.module],
    );
    // #endregion
    return shade;
}
