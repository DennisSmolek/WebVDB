// #region nvdb-read
import { NanoVDBFile } from '@pmndrs/vdb/tools';
// #endregion

export async function nvdbRead(url: string): Promise<Uint32Array> {
    // #region nvdb-read
    const file = await NanoVDBFile.fromURL(url); // or NanoVDBFile.fromArrayBuffer(bytes)
    for (const meta of file.grids) console.log(meta.name, meta.gridType, meta.gridClass, meta.voxelSize);

    // The grid image is what VDBGridPool.add() takes, byte for byte. Float, Fp4, Fp8, Fp16 and FpN grids load.
    const image = file.gridImage(0); // decompressed when the file is zipped
    // #endregion
    return image;
}
