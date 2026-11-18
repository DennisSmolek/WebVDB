// #region parse-vdb
import { buildFromVdb, parseVdb } from '@pmndrs/vdb/tools';
// #endregion

export function parseVdbSample(bytes: ArrayBuffer): Uint32Array[] {
    // #region parse-vdb
    const file = parseVdb(bytes); // uncompressed, zlib or Blosc (LZ4 / zlib); half-float storage; active tiles

    const images: Uint32Array[] = [];
    for (const grid of file.grids) {
        console.log(grid.name, grid.valueType, grid.activeVoxelCount, grid.indexBBox);
        grid.readValue([0, 0, 0]); // { value, active }

        // A float grid, or a vec3 grid (Vec3f), becomes a NanoVDB grid image. The streaming build never
        // makes a dense array, and the .vdb's uniform scale and translate go into the NanoVDB map.
        if (grid.valueType !== 'ptdataidx32') images.push(buildFromVdb(grid));
    }
    // #endregion
    return images;
}
