// #region quantize
import { vdbCollapseGrid, vdbQuantize } from '@pmndrs/vdb/tools';
// #endregion

export function quantizeSample(image: Uint32Array): { fp8: Uint32Array; fpn: Uint32Array; half: Uint32Array; collapsed: Uint32Array } {
    // #region quantize
    // Per-leaf min and quantum plus packed codes. Fp8 is one byte per voxel ('fp4' and 'fp16' are 4 and
    // 16 bits); FpN picks the bit width per leaf from the tolerance (an absolute error bound in value units).
    const fp8 = vdbQuantize(image, 'fp8');
    const fpn = vdbQuantize(image, 'fpn', 0.001);
    // NanoVDB Half: every value as an IEEE binary16 (2 bytes per voxel, ~3 significant digits, no per-leaf range).
    const half = vdbQuantize(image, 'half');

    // vdbCollapseGrid() folds constant leaves into tiles (a smaller image, the same values).
    // vdbQuantize() does the same when asked: vdbQuantize(image, 'fpn', 0.001, { collapse: true }).
    const collapsed = vdbCollapseGrid(image).image;
    // #endregion
    return { fp8, fpn, half, collapsed };
}
