// #region pool-lifecycle
import { VDBGridPool } from '@pmndrs/vdb';
// #endregion

export async function poolLifecycle(device: GPUDevice, image: Uint32Array, next: Uint32Array, largest: number): Promise<void> {
    // #region pool-lifecycle
    const pool = await VDBGridPool.create(device);

    // add() validates and stages the image, and returns its id at once. Nothing is on the GPU yet.
    // `capacity` reserves room for later frames, so replace() writes in place.
    const id = pool.add(image, { label: 'smoke', capacity: largest });

    // update() applies staged changes. flush() is update() on an encoder of its own, submitted at once.
    pool.flush();
    pool.info(id).resident; // true: the image is on the GPU

    pool.replace(id, next); // staged; fits the reserved capacity, so no `version` change
    pool.setTransform(id, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1]); // column-major object-to-world
    pool.flush();

    pool.remove(id); // the id is reused by the next add()
    pool.flush();

    pool.dispose(); // frees what the pool allocated; the device stays yours
    // #endregion
}
