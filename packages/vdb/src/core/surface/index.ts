/**
 * Level-set surfaces and point sprites in the core (K6.6, design §3.9): WGSL functions for hosts that shade
 * with their own materials, and raw reference passes for engine-free hosts. K6.6 exports through
 * `core/internal.ts` only; the public surface lands in K6.8 (design §2.1).
 */

export { vdbLevelSetSurfaceWgsl } from "./level-set-wgsl.js";
export {
  VDB_LEVEL_SET_BOX_TRIANGLES,
  VDB_LEVEL_SET_PASS_DRAW,
  VDB_LEVEL_SET_PASS_FRAME,
  vdbCreateLevelSetPassGpu,
  vdbLevelSetPassWgsl,
  vdbLevelSetProxyCorners,
} from "./level-set-pass.js";
export type { VDBLevelSetPassDraw, VDBLevelSetPassFrame, VDBLevelSetPassGpu, VDBLevelSetPassGpuOptions, VDBLevelSetPassPool } from "./level-set-pass.js";
export {
  VDB_POINT_PASS_FRAME,
  VDB_POINT_SPRITE_CORNERS,
  vdbCreatePointPassGpu,
  vdbPointPassWgsl,
  vdbPointPixelScale,
  vdbPointSpriteWgsl,
} from "./points.js";
export type { VDBPointPassData, VDBPointPassFrame, VDBPointPassGpu, VDBPointPassGpuOptions } from "./points.js";
