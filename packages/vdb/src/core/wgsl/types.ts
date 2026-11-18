/** Grid value types the generated WGSL can decode (NanoVDB `GridType` ids 1, 13–16). */
export type VDBGridTypeName = "float" | "fp4" | "fp8" | "fp16" | "fpn";

/** NanoVDB `GridType` id per name. */
export const VDB_GRID_TYPE_IDS: Readonly<Record<VDBGridTypeName, number>> = {
  float: 1,
  fp4: 13,
  fp8: 14,
  fp16: 15,
  fpn: 16,
};

/** Canonical order; the generator sorts `types` into it so equal sets give equal bytes. */
export const VDB_GRID_TYPE_NAMES: readonly VDBGridTypeName[] = ["float", "fp4", "fp8", "fp16", "fpn"];

/** Where a module's binding lands: `@group(group) @binding(firstBinding)`. */
export interface VDBSlot {
  readonly group: number;
  readonly firstBinding: number;
}

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
