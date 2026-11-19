// #region frame-sequence
import { readWvdb, wvdbDenseVolume, type WvdbDenseVolume } from '@pmndrs/vdb/tools';
import { VDBFrameVolume, VDBSequence, vdbDenseTextures } from '@pmndrs/vdb/three';
// #endregion

// #region frame-sparse-copy
import { wvdbDenseToNvdb } from '@pmndrs/vdb/tools';
import { VDBGridNode, vdbSparseSource } from '@pmndrs/vdb/three';
// #endregion

import type { Scene, WebGPURenderer } from 'three/webgpu';
import type { VDBGridPool } from '@pmndrs/vdb';

export function frameSequence(scene: Scene, urls: string[], first: WvdbDenseVolume): (now: number) => void {
    // #region frame-sequence
    // One dense source for the whole sequence, made from a frame that has every channel.
    const source = vdbDenseTextures(first);
    const volume = new VDBFrameVolume(source, {
        // One mean for every frame, so the extinction does not pump (his sequences use the mean of all frames).
        densityMean: first.meanDensity,
    });
    scene.add(volume);

    // VDBSequence plays any frame type into a { source }: the volume follows each replace by itself, and a
    // frame without a channel the source has reads its absent value (emission 0, mask and albedo 1).
    const sequence = new VDBSequence<WvdbDenseVolume>({
        urls,
        fps: 24,
        loader: async (url) => wvdbDenseVolume(readWvdb(new Uint8Array(await (await fetch(url)).arrayBuffer()))),
    });
    sequence.start({ source });
    const frame = (now: number) => sequence.update(now); // each rendered frame
    // #endregion
    return frame;
}

export function frameSparseCopy(renderer: WebGPURenderer, pool: VDBGridPool, dense: WvdbDenseVolume): VDBFrameVolume {
    // #region frame-sparse-copy
    // The same voxels as a sparse grid, lined up with the dense source in one spread: the texture's index box,
    // the dense mean (normalizeDensity divides by the same value) and the .wvdb box.
    const grid = wvdbDenseToNvdb(dense);
    const id = pool.add(grid.image);
    const grids = new VDBGridNode(renderer, pool, { profile: 'march', helpers: ['cellBoundTrilinear', 'hddaCellEnd'] });
    const source = vdbSparseSource(grids, { density: id }, { ...grid.sparseSourceOptions, lod: 'coarse-pyramid' });
    return new VDBFrameVolume(source);
    // #endregion
}
