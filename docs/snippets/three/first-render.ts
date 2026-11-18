// #region first-render
import * as THREE from 'three/webgpu';
import { VDBGridPool } from '@pmndrs/vdb';
import { NvdbFile } from '@pmndrs/vdb/tools';
import { VDBGridNode, VDBVolumeMaterial, getDevice, vdbProxyGeometry } from '@pmndrs/vdb/three';
// #endregion

export async function firstRender(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, url: string): Promise<void> {
    // #region first-render
    const image = (await NvdbFile.fromURL(url)).gridImage(0);

    const pool = await VDBGridPool.create(getDevice(renderer));
    const id = pool.add(image);
    pool.flush(); // the grid is on the GPU after this
    await pool.settled();

    // One VDBGridNode per renderer. The ray clip the material needs is an opt-in helper.
    const grids = new VDBGridNode(renderer, pool, { helpers: ['rayClipSupport'] });
    const material = new VDBVolumeMaterial(renderer, grids, id, { densityScale: 1, stepSize: 0.75, shadowSteps: 12 });
    scene.add(new THREE.Mesh(vdbProxyGeometry(pool.info(id)), material)); // a box around the grid's index bbox

    renderer.setAnimationLoop(() => {
        grids.update(); // rebinds only if the pool replaced its buffer
        renderer.render(scene, camera);
    });
    // #endregion
}
