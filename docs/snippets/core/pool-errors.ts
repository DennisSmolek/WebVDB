// #region pool-errors
import { VDBError } from '@pmndrs/vdb';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';

export async function poolErrors(pool: VDBGridPool): Promise<void> {
    // #region pool-errors
    try {
        await pool.settled(); // resolves when every growth so far is confirmed by the device
    } catch (error) {
        // A failed growth rolled the pool back to its previous buffer; `lastError` holds the same error.
        if (error instanceof VDBError && error.code === 'VDB_POOL_LIMIT') console.warn(pool.lastError?.message);
        else throw error;
    }
    // #endregion
}
