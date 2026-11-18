// #region module-options
import { VDB_ALL_SCALAR_TYPES, checkModule, vdbModuleEntries, vdbModuleWgsl } from '@pmndrs/vdb';
import type { VDBSlotModuleOptions } from '@pmndrs/vdb';
// #endregion

import type { VDBGridPool } from '@pmndrs/vdb';

export function moduleOptions(pool: VDBGridPool): string {
    // #region module-options
    const options: VDBSlotModuleOptions = {
        slot: { group: 0, firstBinding: 0 }, // where the pool is bound: @group(0) @binding(0) var<storage, read> vdbPool
        types: ['float', 'fp8'], // or VDB_ALL_SCALAR_TYPES to never rebuild the pipeline when a new type arrives
        profile: 'march', // 'read' < 'sample' < 'march'; each includes the one before
        helpers: ['cellBoundTrilinear', 'hddaCellEnd'], // opt-in WGSL helpers
    };
    const wgsl = vdbModuleWgsl(options); // deterministic: the same options give the same bytes
    const entries = vdbModuleEntries(options.slot); // the bind group layout entries for that slot

    checkModule(pool, options); // throws VDB_TYPE_NOT_COMPILED if the pool holds a type the module lacks
    // #endregion
    void VDB_ALL_SCALAR_TYPES;
    void entries;
    return wgsl;
}
