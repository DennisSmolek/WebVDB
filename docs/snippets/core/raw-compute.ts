// #region raw-compute
import { vdbModuleEntries, vdbModuleWgsl } from '@pmndrs/vdb';
import type { VDBSlotModuleOptions } from '@pmndrs/vdb';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';

export async function rawCompute(device: GPUDevice, pool: VDBGridPool, gridId: number): Promise<Float32Array> {
    // #region raw-compute
    const options: VDBSlotModuleOptions = { slot: { group: 0, firstBinding: 0 }, types: ['float'], profile: 'sample' };

    // The host's own half of the shader. The generated module declares `vdbPool` and the vdb* functions.
    const code =
        vdbModuleWgsl(options) +
        /* wgsl */ `
        @group(1) @binding(0) var<storage, read_write> result: array<f32>;
        @group(1) @binding(1) var<uniform> grid: u32;

        @compute @workgroup_size(8)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var acc = vdbAccessorInit(grid);
            // World-space samples along x, through the grid's own transform.
            let world = vec3f(f32(id.x) * 0.1, 0.0, 0.0);
            result[id.x] = vdbSample(&acc, vdbWorldToIndex(grid, world));
        }`;

    const poolLayout = device.createBindGroupLayout({ entries: vdbModuleEntries(options.slot) });
    const ownLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const pipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [poolLayout, ownLayout] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });

    const count = 8;
    const result = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const gridUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(gridUniform, 0, new Uint32Array([gridId]));
    const staging = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const encoder = device.createCommandEncoder();
    pool.update(encoder); // before the bind group, and before the pass that reads the pool
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: poolLayout, entries: [pool.bindGroupEntry(options.slot)] }));
    pass.setBindGroup(1, device.createBindGroup({ layout: ownLayout, entries: [{ binding: 0, resource: { buffer: result } }, { binding: 1, resource: { buffer: gridUniform } }] }));
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(result, 0, staging, 0, count * 4);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    // #endregion
    return values;
}
