// #region quantize
import { collapseGrid, quantize } from '@pmndrs/vdb/tools';
// #endregion

export function quantizeSample(image: Uint32Array): { fp8: Uint32Array; fpn: Uint32Array; collapsed: Uint32Array } {
    // #region quantize
    // Per-leaf min and quantum plus packed codes. Fp8 is one byte per voxel ('fp4' and 'fp16' are 4 and
    // 16 bits); FpN picks the bit width per leaf from the tolerance (an absolute error bound in value units).
    const fp8 = quantize(image, 'fp8');
    const fpn = quantize(image, 'fpn', 0.001);

    // collapseGrid() folds constant leaves into tiles (a smaller image, the same values).
    // quantize() does the same when asked: quantize(image, 'fpn', 0.001, { collapse: true }).
    const collapsed = collapseGrid(image).image;
    // #endregion
    return { fp8, fpn, collapsed };
}
