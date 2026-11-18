// #region wvdb-read
import { readWvdb, wvdbDenseVolume } from '@pmndrs/vdb/tools';
// #endregion

export function wvdbRead(bytes: ArrayBuffer) {
    // #region wvdb-read
    const file = readWvdb(bytes); // the whole file; { maxLod } or { partial: true } read a prefix
    for (const grid of file.grids) console.log(grid.name, grid.kind, grid.class, grid.levels.length);

    // The finest decoded LOD of the density grid as 8-bit texels with a mip atlas: what the dense
    // frame-renderer source uploads.
    const volume = wvdbDenseVolume(file, { density: 'density' });
    console.log(volume.dimensions, volume.meanDensity);
    // #endregion
    return volume;
}
