import type { VDBGridTypeName, VDBSlot } from "../types.js";

/**
 * Generator profiles; each includes the previous one. `majorant` is deferred
 * past 0.1 (D22) and throws `VDB_BAD_OPTION`.
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
   * or `'zero'`: sample the field that is 0 outside the index bbox `[min, max]`
   * (voxels). Trilinear is then 0 outside `[min - 1, max + 1)`, nearest outside
   * `[min - 0.5, max + 0.5)` (voxel centres at integers).
   */
  readonly outside?: "background" | "zero" | undefined;
}

/** What `checkModule` needs from a pool (structural, so tests and adapters can pass plain objects). */
export interface VDBModulePoolLike {
  readonly types: Iterable<VDBGridTypeName>;
  readonly layoutVersion: number;
}
