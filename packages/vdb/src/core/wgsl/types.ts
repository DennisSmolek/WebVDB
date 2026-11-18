import type { VDBGridTypeName, VDBSlot } from "../types.js";

/**
 * Generator profiles; each includes the previous one. `majorant` is deferred
 * past 0.1 (D22) and throws `VDB_BAD_OPTION`.
 */
export type VDBModuleProfile = "read" | "sample" | "march" | "majorant";

/**
 * Opt-in WGSL helpers (K1.7a). None is emitted unless listed in `helpers`, so
 * a module without them is byte-identical to one generated before they
 * existed.
 *
 * - `'cellBoundTrilinear'` (`march`): `vdbCellBoundTrilinear(acc, ijk, dim) -> f32`,
 *   the max of `vdbNodeMax` over the 2x2x2 block of `dim`-cells starting at
 *   the cell that holds `ijk`: a bound for every trilinear sample whose base
 *   voxel `floor(p)` lies in that cell.
 * - `'hddaCellEnd'` (`march`): `vdbHddaCellEnd(h: ptr<function, VdbHdda>) -> f32`,
 *   the ray `t` where the HDDA's current cell ends (`min(next, tmax)`).
 * - `'nodeAverage'` (`march`): `vdbNodeAverage(acc, ijk, dim) -> f32`, the box
 *   mean of the aligned `dim`-cell (8: leaf, 128: lower) holding `ijk` from the
 *   node's average statistic (K3.2, the `'node-average'` LOD of the sparse
 *   volume source): a tile's value; a leaf's `ave * activeVoxels / 512`
 *   (inactive voxels count as 0, the fog background); a lower node's `ave`
 *   (the mean of its ACTIVE voxels; also for `dim > 128`). Without stats, the value at `ijk`.
 * - `'rayClipSupport'` (`sample`+): `vdbRayClipSupport(g, o, d, t) -> bool`,
 *   `vdbRayClip` against the sampler's support box: `[min - 1, max + 1)` for
 *   trilinear, `[min - 0.5, max + 0.5)` for nearest.
 * - `'fastTrilinear'` (`sample`+, sampler `'trilinear'`; K3.2d): `vdbSampleFast(acc, xyz) -> f32`,
 *   bit-identical to `vdbSample`: it takes the base voxel's lower node (the
 *   accessor's cached one, else one descent) and reads each of the 8 taps as
 *   one lower-node child lookup, the same work in every lane (no per-tap tree
 *   walk, no leaf-cache thrash where the cell straddles leaves); a base on a
 *   lower node's +x/+y/+z face (or, with `outside: 'zero'`, whose taps leave
 *   the index bbox) calls `vdbSample`. Reuse the accessor across samples to
 *   keep its cached lower node.
 * - `'levelSet'` (`march`, K3.7): `vdbLevelSetHit(g, o, d, t, sdfToIndex, maxSamples) -> bool`,
 *   the first zero crossing of a level set's trilinear SDF along an index-space
 *   ray (HDDA skip of tile and background cells above 0, sphere trace, bisection
 *   + secant), `vdbLevelSetGradient(acc, xyz, h) -> vec3f` (central differences,
 *   index space) and `vdbLevelSetNormal(g, xyz, h) -> vec3f` (world unit normal).
 *   CPU twins: `tools` `levelSetHit`, `levelSetGradient`, `levelSetNormal`.
 */
export type VDBModuleHelper = "cellBoundTrilinear" | "hddaCellEnd" | "levelSet" | "nodeAverage" | "rayClipSupport" | "fastTrilinear";

/** Options every binding mode shares. */
export interface VDBModuleCommonOptions {
  /**
   * Grid types to decode; others compile out and read 0. At least one.
   *
   * A module must be generated for every type the host may ever add to the
   * pool, or regenerated (and its pipeline rebuilt) when a new type arrives:
   * `pool.types` only lists the grids present now. Pass
   * `VDB_ALL_SCALAR_TYPES` to never rebuild (the largest module), and call
   * `vdbCheckModule(pool, options)` after each `add`.
   */
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
  /** Opt-in helpers ({@link VDBModuleHelper}); a set: order and duplicates don't matter. Default none. */
  readonly helpers?: Iterable<VDBModuleHelper> | undefined;
}

/** `slot` mode: the module declares `@group(g) @binding(b) var<storage, read> vdbPool: array<u32>;`. */
export interface VDBSlotModuleOptions extends VDBModuleCommonOptions {
  /** The same `VDBSlot` that `vdbModuleEntries(slot)` and `pool.bindGroupEntry(slot)` take. */
  readonly slot: VDBSlot;
  readonly buffer?: undefined;
}

/**
 * `buffer` mode: the module declares nothing and reads the host's
 * `array<u32>` through this WGSL expression (engine adapters, e.g.
 * `'vdbPool.value'` from three's TSL).
 */
export interface VDBBufferModuleOptions extends VDBModuleCommonOptions {
  readonly buffer: string;
  readonly slot?: undefined;
}

/**
 * Options of `vdbModuleWgsl` (docs/design/K1-core.md §3.1): exactly one of
 * `slot` / `buffer`. Annotate a host's options as `VDBSlotModuleOptions` to
 * read `options.slot` without a non-null assertion.
 */
export type VDBModuleOptions = VDBSlotModuleOptions | VDBBufferModuleOptions;

/**
 * The type/layout-only form of `vdbCheckModule`'s options: the types a module
 * decodes and, optionally, the pool layout it targets (default
 * `VDB_GPU_LAYOUT_VERSION`). No binding or profile needed.
 */
export interface VDBModuleTypeCheck {
  readonly types: Iterable<VDBGridTypeName>;
  readonly layoutVersion?: number | undefined;
}

/** What `vdbCheckModule` needs from a pool (structural, so tests and adapters can pass plain objects). */
export interface VDBModulePoolLike {
  readonly types: Iterable<VDBGridTypeName>;
  readonly layoutVersion: number;
}
