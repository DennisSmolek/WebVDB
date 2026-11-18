// #region nvdb-stream
import { readNvdbStream } from '@pmndrs/vdb/tools';
// #endregion

export async function nvdbStream(url: string): Promise<void> {
    // #region nvdb-stream
    const response = await fetch(url);
    // Each grid is emitted as soon as its bytes (raw or zipped) are complete, so the first grid can be
    // on screen while the rest downloads.
    for await (const grid of readNvdbStream(response, { onProgress: (p) => console.log(p.bytesReceived, p.bytesTotal) })) {
        console.log(grid.index, grid.baseName, grid.lod, grid.image.byteLength);
    }
    // #endregion
}
