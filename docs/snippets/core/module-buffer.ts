// #region module-buffer
import { vdbModuleWgsl } from '@pmndrs/vdb';
// #endregion

export function moduleBuffer(): string {
    // #region module-buffer
    // Buffer mode declares no binding: the module reads the host's `array<u32>` through this expression.
    // This is how the three.js adapter binds the pool through three's own storage node.
    return vdbModuleWgsl({ buffer: 'vdbPool.value', types: ['float'], profile: 'sample' });
    // #endregion
}
