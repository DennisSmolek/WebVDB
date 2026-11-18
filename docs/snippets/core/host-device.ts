// #region host-device
import { VDBGridPool } from '@pmndrs/vdb';
// #endregion

export async function hostDevice(): Promise<VDBGridPool> {
    // #region host-device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    // The pool grows into the adapter's storage-buffer limits, so ask for them up front.
    const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
    const reason = VDBGridPool.unsupportedReason(device); // null when the device can hold a pool
    if (reason) throw new Error(reason);

    const pool = await VDBGridPool.create(device, { label: 'volumes' });
    // #endregion
    return pool;
}
