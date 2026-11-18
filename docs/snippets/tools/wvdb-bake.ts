// #region wvdb-bake
import { nvdbToWvdb, vdbToWvdb, writeWvdb } from '@pmndrs/vdb/tools';
// #endregion

export function wvdbBake(vdbBytes: ArrayBuffer, image: Uint32Array): Uint8Array[] {
    // #region wvdb-bake
    // From a .vdb (all float grids) or a NanoVDB grid image: point-sampled over the bbox, like his CLI.
    const fromVdb = vdbToWvdb(vdbBytes, { resolution: 128, lods: 3 });
    const fromNano = nvdbToWvdb(image, { resolution: 128 });

    // Or write dense grids you made yourself.
    const dims: [number, number, number] = [32, 32, 32];
    const written = writeWvdb(
        [{ name: 'density', background: 0, bbox: null, finest: { dimensions: dims, values: new Float32Array(32 * 32 * 32) } }],
        { lods: 2 },
    );
    // #endregion
    return [fromVdb.bytes, fromNano.bytes, written.bytes];
}
