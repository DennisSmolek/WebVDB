import type { VDBGridTypeName, VDBSlot } from "../types.js";

/**
 * Generator profiles; each includes the previous one. `march` and `majorant`
 * are reserved for K1.4b and throw `VDB_BAD_OPTION` until then.
 */
export type VDBModuleProfile = "read" | "sample" | "march" | "majorant";

/** Options of `vdbModuleWgsl` (docs/design/K1-core.md §3.1). Exactly one of `slot` / `buffer`. */
export interface VDBModuleOptions {
  /** Declare `@group(g) @binding(b) var<storage, read> vdbPool: array<u32>;`. */
  readonly slot?: VDBSlot | undefined;
  /**
   * Declare nothing and read the host's `array<u32>` through this WGSL
   * expression (engine adapters, e.g. `'vdbPool.value'` from three's TSL).
   */
  readonly buffer?: string | undefined;
  /** Grid types to decode; others compile out (`pool.types`). At least one. */
  readonly types: Iterable<VDBGridTypeName>;
  readonly profile: VDBModuleProfile;
  /** `sample`+: `'trilinear'` (default) or `'nearest'` (rounds to the closest voxel). */
  readonly sampler?: "trilinear" | "nearest" | undefined;
  /** `read`+: `false` drops `vdbIsActive`. Default `true`. */
  readonly isActive?: boolean | undefined;
  /**
   * `sample`+: `'background'` (default, NanoVDB: the tree's background value)
   * or `'zero'` (0 outside the index bbox `[min, max + 1]`, pathtracer convention).
   */
  readonly outside?: "background" | "zero" | undefined;
}

/** What `checkModule` needs from a pool (structural, so tests and adapters can pass plain objects). */
export interface VDBModulePoolLike {
  readonly types: Iterable<VDBGridTypeName>;
  readonly layoutVersion: number;
}
