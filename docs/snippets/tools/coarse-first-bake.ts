// #region coarse-first-bake
import { vdbBakeCoarseFirst, nvdbLodName, parseNvdbLodName } from '@pmndrs/vdb/tools';
// #endregion

export function coarseFirstBake(image: Uint32Array): Uint8Array {
    // #region coarse-first-bake
    // One .nvdb holding [<name>.lod48, <name>]: a coarse preview first, the full grid after it.
    const file = vdbBakeCoarseFirst(image, { previewResolution: 48, quantize: 'fp8', fullQuantize: 'fpn', codec: 'zip' });

    console.log(nvdbLodName('cloud', 48)); // the name of the preview grid
    console.log(parseNvdbLodName(nvdbLodName('cloud', 48))); // { baseName: 'cloud', lod: 48 }
    // #endregion
    return file;
}
