// #region pool-bind

// #endregion

import type { VDBSlot } from '@pmndrs/vdb';
import type { VDBGridPool } from '@pmndrs/vdb';

export function poolBind(device: GPUDevice, pool: VDBGridPool, layout: GPUBindGroupLayout, slot: VDBSlot): (encoder: GPUCommandEncoder) => GPUBindGroup {
    // #region pool-bind
    let bindGroup: GPUBindGroup | undefined;
    let boundVersion = -1;

    return (encoder) => {
        pool.update(encoder); // 1. before any bind group is built or reused this frame
        if (!bindGroup || boundVersion !== pool.version) {
            // 2. the buffer is a new object after a growth: rebuild when `version` moved
            bindGroup = device.createBindGroup({ layout, entries: [pool.bindGroupEntry(slot)] });
            boundVersion = pool.version;
        }
        return bindGroup; // 3. the passes that read the pool, then submit the encoder
    };
    // #endregion
}
