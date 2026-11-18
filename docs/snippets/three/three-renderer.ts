// #region three-renderer
import { WebGPURenderer } from 'three/webgpu';
import { VDBGridPool } from '@pmndrs/vdb';
import { getDevice } from '@pmndrs/vdb/three';
// #endregion

export async function threeRenderer(): Promise<{ renderer: WebGPURenderer; pool: VDBGridPool }> {
    // #region three-renderer
    // The host owns the device. Pass it to three, so renderer.dispose() never destroys it or the pool's buffer.
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });

    const renderer = new WebGPURenderer({ device });
    await renderer.init();

    getDevice(renderer) === device; // true: the renderer's WebGPU backend holds your device
    const pool = await VDBGridPool.create(getDevice(renderer));
    // #endregion
    return { renderer, pool };
}
