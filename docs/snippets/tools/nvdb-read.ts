// #region nvdb-read
import { NvdbFile } from '@pmndrs/vdb/tools';
// #endregion

export async function nvdbRead(url: string): Promise<Uint32Array> {
    // #region nvdb-read
    const file = await NvdbFile.fromURL(url); // or NvdbFile.fromArrayBuffer(bytes)
    for (const meta of file.grids) console.log(meta.name, meta.gridType, meta.gridClass, meta.voxelSize);

    // The grid image is what VDBGridPool.add() takes, byte for byte. Float, Fp4, Fp8, Fp16, FpN and Half grids load.
    const image = file.gridImage(0); // decompressed when the file is zipped
    // #endregion
    return image;
}
