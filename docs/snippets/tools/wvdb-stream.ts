// #region wvdb-stream
import { WvdbStreamReader } from '@pmndrs/vdb/tools';
// #endregion

export async function wvdbStream(url: string): Promise<void> {
    // #region wvdb-stream
    const reader = new WvdbStreamReader();
    const response = await fetch(url);
    const body = response.body;
    if (!body) return;
    const stream = body.getReader();
    for (;;) {
        const { done, value } = await stream.read();
        if (done) break;
        // Each complete LOD is an event, coarsest first.
        for (const event of reader.push(value)) console.log('lod', event.lod, 'complete', event.complete);
    }
    reader.end();
    // #endregion
}
