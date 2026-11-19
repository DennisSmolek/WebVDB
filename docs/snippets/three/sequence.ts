// #region sequence
import { NvdbFile } from '@pmndrs/vdb/tools';
import { VDBSequence } from '@pmndrs/vdb/three';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';

export async function sequence(pool: VDBGridPool, largestFrameBytes: number, firstFrame: Uint32Array): Promise<(now: number) => void> {
    // #region sequence
    // Reserve the largest frame, so every replace() writes in place: no move, no `version` change,
    // no rebind of the VDBGridNode, no new bind group.
    const id = pool.add(firstFrame, { capacity: largestFrameBytes });

    const sequence = new VDBSequence({
        urls: (index) => `/frames/smoke_${String(index).padStart(4, '0')}.nvdb`,
        frameCount: 120,
        fps: 24,
        prefetch: 4, // frames decoded ahead
        loopMode: 'repeat', // or 'pingpong', or 'once' (hold the last frame)
        // The default loader fetches and parses a .nvdb; override it to load from anywhere.
        loader: async (url) => (await NvdbFile.fromURL(url)).gridImage(0),
        onError: (error, index) => console.warn('frame', index, error),
    });
    await sequence.preload(); // the first frames, so playback starts without a stall
    sequence.start({ pool, id });

    // Each rendered frame: update() may replace() the grid, then flush and render as usual.
    const frame = (now: number) => {
        sequence.update(now);
        pool.flush();
    };
    // #endregion
    return frame;
}
