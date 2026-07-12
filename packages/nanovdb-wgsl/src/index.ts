/**
 * nanovdb-wgsl — renderer-agnostic NanoVDB traversal WGSL module + TS loader.
 *
 * The WGSL module itself is vendored in `vendor/pnanovdb.wgsl` (see
 * vendor/VENDOR.md for the fork policy). The `.nvdb` file loader
 * (`NanoVDBFile`, Phase 1 of docs/PLAN.md) lives in `./nvdb-file.ts`; this
 * module just re-exports its public surface.
 */

export {
  FILE_HEADER_SIZE,
  FILE_METADATA_SIZE,
  GRID_DATA_SIZE,
  MAGIC_ASCII,
  Codec,
  SUPPORTED_GRID_TYPES,
  NanoVDBFile,
} from "./nvdb-file.js";
export type { GridMetadata } from "./nvdb-file.js";

/**
 * Resolvable URL of the vendored WGSL module, for runtimes that fetch it
 * (Node tests, non-bundler setups). Vite/bundler consumers should import
 * `nanovdb-wgsl/pnanovdb.wgsl?raw` instead.
 */
export const pnanovdbWgslUrl = new URL("../vendor/pnanovdb.wgsl", import.meta.url);

/**
 * Pure-TypeScript CPU reference for NanoVDB voxel lookup/sampling (Phase 1/2,
 * docs/PLAN.md) — a byte-for-byte transliteration of the vendored
 * `PNanoVDB.h` root -> upper -> lower -> leaf descent, restricted to grid
 * types FLOAT/FP8/FPN. This used to be unimportable from a browser/bundler
 * context (`./cpu/stride-tables.ts` loaded its layout constants via
 * `node:fs.readFileSync`); it now reads from a baked, browser-safe generated
 * module instead (`./cpu/stride-tables.generated.ts`), so these are safe to
 * import from any Vite/browser page as well as Node.
 */
export { readValue } from "./cpu/read-value.js";
export type { ReadResult } from "./cpu/read-value.js";
export { sampleTrilinear } from "./cpu/sample-trilinear.js";
export { probeCoords, probePoints } from "./cpu/probe-coords.js";
export type { ProbeOpts } from "./cpu/probe-coords.js";

/**
 * Layout-constant accessors underlying `readValue` — exposed for callers
 * that need to walk NanoVDB tree structure generically (any grid type, not
 * just FLOAT/FP8/FPN) without decoding leaf values themselves, e.g.
 * enumerating LEAF node origins for a visualization. See `./cpu/
 * stride-tables.ts` for the full accessor surface and its data source.
 */
export { defineNumber, gridTypeConstantsForId } from "./cpu/stride-tables.js";
export type { GridTypeConstants } from "./cpu/stride-tables.js";
