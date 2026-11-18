// #region resample
import { vdbResampleError, vdbResampleGrid, vdbTransform } from '@pmndrs/vdb/tools';
// #endregion

export function resampleSample(image: Uint32Array): Uint32Array {
    // #region resample
    // Resample over a world box (default: the grid's own) to about 128 voxels on the longest side,
    // with a box filter, and store the result as Fp8.
    const baked = vdbResampleGrid(image, { resolution: 128, filter: 'box', quantize: 'fp8' });
    console.log(baked.dimensions, baked.bbox, baked.voxelCount);

    // How far is it from the source? The mean absolute error as a share of the mean value.
    console.log(vdbResampleError(image, baked.image).percent);

    // A metadata-only edit of the grid's map (uniform scale and translate), with no voxel churn.
    return vdbTransform(baked.image, { voxelSize: 0.05, worldOrigin: [0, 0, 0] });
    // #endregion
}
