// #region build-dense
import { buildFromDense, inspect, writeNvdb } from '@pmndrs/vdb/tools';
// #endregion

export function buildDense(): ArrayBuffer {
    // #region build-dense
    const dims: [number, number, number] = [64, 64, 64];
    const values = new Float32Array(dims[0] * dims[1] * dims[2]); // x fastest
    for (let z = 0; z < dims[2]; z++)
        for (let y = 0; y < dims[1]; y++)
            for (let x = 0; x < dims[0]; x++) {
                const d = Math.hypot(x - 32, y - 32, z - 32);
                values[x + dims[0] * (y + dims[1] * z)] = Math.max(0, 1 - d / 24);
            }

    // Zero voxels become inactive background; the rest go into a sparse 5-4-3 tree.
    const image = buildFromDense(values, dims, { voxelSize: 0.1, gridName: 'density', gridClass: 'FogVolume' });
    console.log(inspect(image)); // { gridType, gridClass, voxelCount, nodeCounts, memoryBreakdown }

    return writeNvdb([image], { codec: 'zip' });
    // #endregion
}
