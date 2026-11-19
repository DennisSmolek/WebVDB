// #region frame-dense
import { readWvdb, wvdbDenseVolume } from '@pmndrs/vdb/tools';
import { VDBFrameVolume, vdbDenseTextures } from '@pmndrs/vdb/three';
// #endregion

import type { Scene } from 'three/webgpu';

export function frameDense(scene: Scene, wvdb: ArrayBuffer): VDBFrameVolume {
    // #region frame-dense
    // A dense source: the .wvdb's density as an 8-bit 3D texture with a mip atlas.
    const source = vdbDenseTextures(wvdbDenseVolume(readWvdb(wvdb)));
    const volume = new VDBFrameVolume(source, { extinction: 100 }); // the box is the file's bbox
    scene.add(volume); // renderer.render(scene, camera) draws it, froxels first

    volume.scatterColor.set('#ffffff');
    // A measured medium in one call: albedo and extinction spectrum on the volume, the phase on the renderer.
    volume.setMedium('fog');
    // volume.bbox is the resolved box (here the file's): place the volume by it.
    volume.position.copy(volume.bbox.getCenter(volume.position)).negate();
    // #endregion
    return volume;
}
