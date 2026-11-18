// #region parse-points
import { parseVdb } from '@pmndrs/vdb/tools';
// #endregion

export function parsePoints(bytes: ArrayBuffer): Float32Array | undefined {
    // #region parse-points
    const file = parseVdb(bytes);
    for (const grid of file.grids) {
        if (grid.valueType !== 'ptdataidx32') continue; // a point-data grid
        const points = grid.points();
        console.log(grid.pointCount, grid.attributes.map((a) => a.name), grid.groupNames);
        return points.positions; // Float32Array of xyz; points.attributes holds the named attribute arrays
    }
    // #endregion
    return undefined;
}
