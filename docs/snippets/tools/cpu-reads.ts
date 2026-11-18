// #region cpu-reads
import { readValue, sampleTrilinear } from '@pmndrs/vdb/tools';
// #endregion

export function cpuReads(image: Uint32Array): void {
    // #region cpu-reads
    // The same tree descent as the WGSL module, in TypeScript: an oracle for GPU results, no GPU needed.
    // Index space: the integer coordinates are voxels. (Float, Fp8 and FpN grids.)
    console.log(readValue(image, [10, 20, 30])); // { value, active }
    console.log(sampleTrilinear(image, [10.5, 20.25, 30])); // continuous index-space sample
    // #endregion
}
