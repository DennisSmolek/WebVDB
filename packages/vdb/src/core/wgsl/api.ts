/**
 * The public `vdb*` WGSL API layered over the vendored traversal
 * (docs/design/K1-core.md §3.2). Written in the generated module's final names:
 * vendored declarations appear renamed (`vdb_*`, `VDB_*`), and the generator
 * pulls in exactly the vendored declarations this text references.
 *
 * Every pool read goes through `vdb_load`, the module's one buffer access,
 * which the generator emits for the chosen binding mode.
 *
 * Layers stack: `sample` includes `read`, `march` includes `sample`
 * (`majorant` is deferred past 0.1, D22). Each layer is a function of the
 * normalized options and the profile rank.
 *
 * Conventions every layer keeps:
 * - A grid id is usable when it is below `vdbGridCount()` and its record is
 *   live. Every per-grid function checks that first: a dead or out-of-range
 *   id never reads a zeroed record or image bytes as a record.
 * - Voxel centres sit at integer index coordinates (the vendored trilinear and
 *   NanoVDB `SampleFromVoxels`); HDDA cells are floor-based, so voxel `i`
 *   occupies `[i, i + 1)` for ray clipping and traversal.
 */

import type { VDBGridTypeName } from "../types.js";
import { levelSetWgsl } from "./level-set.js";
import type { VDBModuleHelper } from "./types.js";

/** Options the API layer depends on (normalized by the generator). */
export interface ApiOptions {
  readonly types: readonly VDBGridTypeName[];
  readonly isActive: boolean;
  readonly sampler: "trilinear" | "nearest";
  readonly outside: "background" | "zero";
  /** Opt-in helpers, canonical order; the generator has checked each one's profile. */
  readonly helpers: readonly VDBModuleHelper[];
}

/** Vendored grid-type constant names (after the rename) per compiled type. */
export const GRID_TYPE_CONSTANT: Readonly<Record<VDBGridTypeName, string>> = {
  float: "VDB_GRID_TYPE_FLOAT",
  fp4: "VDB_GRID_TYPE_FP4",
  fp8: "VDB_GRID_TYPE_FP8",
  fp16: "VDB_GRID_TYPE_FP16",
  fpn: "VDB_GRID_TYPE_FPN",
};

/** Whether the accessor carries the index bbox (only `sample`+ with `outside: 'zero'`). */
function carriesBBox(profileRank: number, o: ApiOptions): boolean {
  return profileRank >= 1 && o.outside === "zero";
}

/** Whether the accessor carries the stats flag and `valueMax` (`march`+). */
function carriesStats(profileRank: number): boolean {
  return profileRank >= 2;
}

function readLayer(o: ApiOptions, profileRank: number): string {
  const bbox = carriesBBox(profileRank, o);
  const stats = carriesStats(profileRank);
  const typeOk = o.types.map((t) => `t == ${GRID_TYPE_CONSTANT[t]}`).join(" || ");
  const extraFields = (bbox ? "\n    lo: vec3i,\n    hi: vec3i," : "") + (stats ? "\n    stats: bool,\n    vmax: f32," : "");
  const extraInit =
    (bbox ? "\n        a.lo = vdbIndexMin(g);\n        a.hi = vdbIndexMax(g);" : "") +
    (stats
      ? "\n        a.stats = (vdb_rec(g, VDB_REC_FLAGS) & VDB_REC_FLAG_HAS_STATS) != 0u;\n        a.vmax = bitcast<f32>(vdb_rec(g, VDB_REC_VALUE_MAX));"
      : "");
  const parts = [
    `struct VdbAccessor {
    buf: vdb_buf_t,
    gridType: u32,
    ok: bool,
    inner: vdb_readaccessor_t,${extraFields}
}`,
    `fn vdb_rec(g: u32, w: u32) -> u32 {
    return vdb_load(VDB_POOL_TABLE_WORD + g * VDB_POOL_RECORD_WORDS + w);
}`,
    `fn vdb_type_ok(t: u32) -> bool {
    return ${typeOk};
}`,
    `fn vdbGridCount() -> u32 {
    return vdb_load(VDB_HEADER_GRID_COUNT);
}`,
    `fn vdb_live(g: u32) -> bool {
    return g < vdbGridCount() && (vdb_rec(g, VDB_REC_FLAGS) & VDB_REC_FLAG_LIVE) != 0u;
}`,
    `fn vdbGridType(g: u32) -> u32 {
    if !vdb_live(g) {
        return 0u;
    }
    return vdb_rec(g, VDB_REC_GRID_TYPE);
}`,
    `fn vdbIndexMin(g: u32) -> vec3i {
    if !vdb_live(g) {
        return vec3i(0);
    }
    return bitcast<vec3i>(vec3u(vdb_rec(g, VDB_REC_INDEX_BBOX_MIN), vdb_rec(g, VDB_REC_INDEX_BBOX_MIN + 1u), vdb_rec(g, VDB_REC_INDEX_BBOX_MIN + 2u)));
}`,
    `fn vdbIndexMax(g: u32) -> vec3i {
    if !vdb_live(g) {
        return vec3i(-1);
    }
    return bitcast<vec3i>(vec3u(vdb_rec(g, VDB_REC_INDEX_BBOX_MAX), vdb_rec(g, VDB_REC_INDEX_BBOX_MAX + 1u), vdb_rec(g, VDB_REC_INDEX_BBOX_MAX + 2u)));
}`,
    `fn vdbAccessorInit(g: u32) -> VdbAccessor {
    var a: VdbAccessor;
    a.gridType = vdbGridType(g);
    a.ok = vdb_type_ok(a.gridType);
    var root = vdb_root_handle_t(0u);
    if a.ok {
        a.buf = vdb_make_buffer(vdb_rec(g, VDB_REC_IMAGE_OFFSET), vdb_rec(g, VDB_REC_IMAGE_BYTES) >> 2u);
        root = vdb_tree_get_root(a.buf, vdb_grid_get_tree(vdb_grid_handle_t(0u)));${extraInit}
    }
    var inner: vdb_readaccessor_t;
    vdb_readaccessor_init(&inner, root);
    a.inner = inner;
    return a;
}`,
    `fn vdbReadValue(acc: ptr<function, VdbAccessor>, ijk: vec3i) -> f32 {
    if !(*acc).ok {
        return 0.0;
    }
    var inner = (*acc).inner;
    let v = vdb_readaccessor_get_value_float((*acc).gridType, (*acc).buf, &inner, ijk);
    (*acc).inner = inner;
    return v;
}`,
  ];
  if (o.isActive) {
    parts.push(`fn vdbIsActive(acc: ptr<function, VdbAccessor>, ijk: vec3i) -> bool {
    if !(*acc).ok {
        return false;
    }
    var inner = (*acc).inner;
    let v = vdb_readaccessor_is_active((*acc).gridType, (*acc).buf, &inner, ijk);
    (*acc).inner = inner;
    return v;
}`);
  }
  return parts.join("\n");
}

/**
 * `vdbSample` per sampler and `outside` mode.
 *
 * `outside: 'background'` reads the tree everywhere (NanoVDB: outside the
 * tree the background value).
 *
 * `outside: 'zero'` samples the field that equals the tree inside the index
 * bbox `[min, max]` (voxels, inclusive) and 0 outside it, so both samplers
 * see the same box with voxel centres at integers:
 * - trilinear: each of the 8 taps outside `[min, max]` reads 0. A point is
 *   0 unless its base voxel `floor(xyz)` lies in `[min - 1, max]` (some tap
 *   inside), i.e. `xyz` in `[min - 1, max + 1)`.
 * - nearest: the voxel `floor(xyz + 0.5)` must lie in `[min, max]`, i.e.
 *   `xyz` in `[min - 0.5, max + 0.5)`; else 0.
 * CPU twins: `tools` `sampleTrilinear` / `sampleNearest` with `{ outside: 'zero' }`.
 */
function sampleFunction(o: ApiOptions, profileRank: number): string[] {
  const read = (ijk: string) => `vdb_readaccessor_get_value_float((*acc).gridType, (*acc).buf, &inner, ${ijk})`;
  if (!carriesBBox(profileRank, o)) {
    const lookup =
      o.sampler === "trilinear" ? "vdb_sample_trilinear_typed((*acc).gridType, (*acc).buf, &inner, xyz)" : read("vec3i(floor(xyz + vec3f(0.5)))");
    return [
      `fn vdbSample(acc: ptr<function, VdbAccessor>, xyz: vec3f) -> f32 {
    if !(*acc).ok {
        return 0.0;
    }
    var inner = (*acc).inner;
    let v = ${lookup};
    (*acc).inner = inner;
    return v;
}`,
    ];
  }
  if (o.sampler === "nearest") {
    return [
      `fn vdbSample(acc: ptr<function, VdbAccessor>, xyz: vec3f) -> f32 {
    let ijk = vec3i(floor(xyz + vec3f(0.5)));
    if !(*acc).ok || any(ijk < (*acc).lo) || any(ijk > (*acc).hi) {
        return 0.0;
    }
    var inner = (*acc).inner;
    let v = ${read("ijk")};
    (*acc).inner = inner;
    return v;
}`,
    ];
  }
  // Same tap order and lerp structure as the vendored sample_trilinear_typed.
  return [
    `fn vdb_tap0(acc: ptr<function, VdbAccessor>, ijk: vec3i) -> f32 {
    if any(ijk < (*acc).lo) || any(ijk > (*acc).hi) {
        return 0.0;
    }
    var inner = (*acc).inner;
    let v = ${read("ijk")};
    (*acc).inner = inner;
    return v;
}`,
    `fn vdbSample(acc: ptr<function, VdbAccessor>, xyz: vec3f) -> f32 {
    let b = vec3i(floor(xyz));
    if !(*acc).ok || any(b < (*acc).lo - vec3i(1)) || any(b > (*acc).hi) {
        return 0.0;
    }
    let u = xyz - vec3f(b);
    let x00 = mix(vdb_tap0(acc, b), vdb_tap0(acc, b + vec3i(1, 0, 0)), u.x);
    let x10 = mix(vdb_tap0(acc, b + vec3i(0, 1, 0)), vdb_tap0(acc, b + vec3i(1, 1, 0)), u.x);
    let x01 = mix(vdb_tap0(acc, b + vec3i(0, 0, 1)), vdb_tap0(acc, b + vec3i(1, 0, 1)), u.x);
    let x11 = mix(vdb_tap0(acc, b + vec3i(0, 1, 1)), vdb_tap0(acc, b + vec3i(1, 1, 1)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}`,
  ];
}

function sampleLayer(o: ApiOptions, profileRank: number): string {
  return [
    `fn vdb_row(b: u32) -> vec4f {
    return bitcast<vec4f>(vec4u(vdb_load(b), vdb_load(b + 1u), vdb_load(b + 2u), vdb_load(b + 3u)));
}`,
    `fn vdb_affine(g: u32, w: u32, p: vec4f) -> vec3f {
    if !vdb_live(g) {
        return vec3f(0.0);
    }
    let b = VDB_POOL_TABLE_WORD + g * VDB_POOL_RECORD_WORDS + w;
    return vec3f(dot(vdb_row(b), p), dot(vdb_row(b + 4u), p), dot(vdb_row(b + 8u), p));
}`,
    `fn vdbWorldToIndex(g: u32, p: vec3f) -> vec3f {
    return vdb_affine(g, VDB_REC_WORLD_TO_INDEX, vec4f(p, 1.0));
}`,
    `fn vdbWorldToIndexDir(g: u32, d: vec3f) -> vec3f {
    return vdb_affine(g, VDB_REC_WORLD_TO_INDEX, vec4f(d, 0.0));
}`,
    `fn vdbIndexToWorld(g: u32, p: vec3f) -> vec3f {
    return vdb_affine(g, VDB_REC_INDEX_TO_WORLD, vec4f(p, 1.0));
}`,
    `fn vdbIndexToWorldDir(g: u32, d: vec3f) -> vec3f {
    return vdb_affine(g, VDB_REC_INDEX_TO_WORLD, vec4f(d, 0.0));
}`,
    ...sampleFunction(o, profileRank),
    `fn vdbRayClip(g: u32, o: vec3f, d: vec3f, t: ptr<function, vec2f>) -> bool {
    if !vdb_live(g) {
        return false;
    }
    var t0 = (*t).x;
    var t1 = (*t).y;
    let hit = vdb_hdda_ray_clip(vec3f(vdbIndexMin(g)), vec3f(vdbIndexMax(g) + vec3i(1)), o, &t0, d, &t1);
    *t = vec2f(t0, t1);
    return hit && t0 <= t1;
}`,
  ].join("\n");
}

/**
 * `march` (design note §3.2): the vendored HDDA, the accessor's node size at a
 * voxel, and node stats.
 *
 * `vdbGetDim(acc, ijk)` is the vendored `readaccessor_get_dim`: 1 inside a
 * leaf, 8 in a lower-node tile, 128 in an upper-node tile, 4096 in a root tile
 * or the background (and 4096 for a grid the accessor can't read, so a walk
 * ends after one cell instead of stalling at dim 0).
 *
 * `vdbNodeMax(acc, ijk, dim)` bounds the aligned cell of size `dim` holding
 * `ijk`. With `a = vdbGetDim(acc, ijk)`:
 * - `a > 1` and `dim <= a`: the cell lies in one tile; returns its value
 *   (exact, active or not).
 * - otherwise the max statistic of the smallest node spanning the cell: the
 *   leaf for `dim <= 8`, lower for `<= 128`, upper for `<= 4096`, else root.
 *   NanoVDB stats cover ACTIVE values only: the result bounds the active
 *   voxels and active tiles of that node, not inactive values, and not a
 *   trilinear sample, whose taps reach one voxel into the neighbours.
 * - without stats (record flag b1 clear) the stats branches return the
 *   record's `valueMax`: the caller's bound from `pool.add`, else +Infinity
 *   (the pool convention for "no bound"). Test `vdbHasStats(g)` rather than
 *   relying on infinity arithmetic.
 * The node handles come from the accessor cache that `vdbGetDim` just filled
 * along `ijk`'s path (leaf, lower and upper are all valid whenever the level
 * below them is, see the vendored `readaccessor_iscached*`).
 */
function marchLayer(): string {
  return [
    `alias VdbHdda = vdb_hdda_t;`,
    `fn vdbHddaInit(h: ptr<function, VdbHdda>, o: vec3f, tmin: f32, d: vec3f, tmax: f32, dim: u32) {
    vdb_hdda_init(h, o, tmin, d, tmax, i32(dim));
}`,
    `fn vdbHddaUpdate(h: ptr<function, VdbHdda>, o: vec3f, d: vec3f, dim: u32) -> bool {
    return vdb_hdda_update(h, o, d, i32(dim));
}`,
    `fn vdbHddaStep(h: ptr<function, VdbHdda>) -> bool {
    return vdb_hdda_step(h);
}`,
    `fn vdbGetDim(acc: ptr<function, VdbAccessor>, ijk: vec3i) -> u32 {
    if !(*acc).ok {
        return 4096u;
    }
    var inner = (*acc).inner;
    let v = vdb_readaccessor_get_dim((*acc).gridType, (*acc).buf, &inner, ijk);
    (*acc).inner = inner;
    return v;
}`,
    `fn vdbHasStats(g: u32) -> bool {
    return vdb_live(g) && (vdb_rec(g, VDB_REC_FLAGS) & VDB_REC_FLAG_HAS_STATS) != 0u;
}`,
    `fn vdbValueMax(g: u32) -> f32 {
    if !vdb_live(g) {
        return 0.0;
    }
    return bitcast<f32>(vdb_rec(g, VDB_REC_VALUE_MAX));
}`,
    `fn vdbNodeMax(acc: ptr<function, VdbAccessor>, ijk: vec3i, dim: u32) -> f32 {
    if !(*acc).ok {
        return 0.0;
    }
    let a = vdbGetDim(acc, ijk);
    if a > 1u && dim <= a {
        return vdbReadValue(acc, ijk);
    }
    if !(*acc).stats {
        return (*acc).vmax;
    }
    let t = (*acc).gridType;
    let b = (*acc).buf;
    if dim <= 8u {
        return vdb_leaf_get_max_float(t, b, (*acc).inner.leaf);
    }
    if dim <= 128u {
        return vdb_lower_get_max_float(t, b, (*acc).inner.lower);
    }
    if dim <= 4096u {
        return vdb_upper_get_max_float(t, b, (*acc).inner.upper);
    }
    return vdb_root_get_max_float(t, b, (*acc).inner.root);
}`,
  ].join("\n");
}

/**
 * Opt-in helpers (K1.7a, the `helpers` option). Each is emitted only when
 * asked for, so modules without helpers keep their bytes (the size snapshot).
 *
 * `vdbCellBoundTrilinear(acc, ijk, dim)`: the max of `vdbNodeMax` over the
 * 2x2x2 block of aligned `dim`-cells whose min corner is the cell holding
 * `ijk` (that cell and its +x/+y/+z neighbours). A trilinear sample at `p`
 * reads voxels `floor(p) + {0,1}^3`, so when `floor(p)` lies in the cell every
 * tap lies in the block and the result bounds the sample, with `vdbNodeMax`'s
 * caveat: stats bound ACTIVE values, so inactive leaf voxels must be <= their
 * leaf's max (true of fog volumes). With `outside: 'zero'` and a negative
 * field, also take max(result, 0). This is the skip test a march needs; the
 * cell's own `vdbNodeMax` is not one (CLAUDE.md Traps). CPU twin: `tools`
 * `cellBoundTrilinear`.
 *
 * `vdbHddaCellEnd(h)`: the ray `t` where the HDDA's current cell ends,
 * `min(next.x, next.y, next.z, tmax)` of the vendored state, so a host
 * needn't name `VdbHdda`'s vendored fields. CPU twin: `tools` `hddaCellEnd`.
 *
 * `vdbNodeAverage(acc, ijk, dim)` (K3.2): the box mean of the aligned
 * `dim`-cell holding `ijk`, from node statistics instead of voxels (the
 * sparse volume source's `'node-average'` LOD). A cell inside one tile (or the
 * background) is that tile's value, exactly. Otherwise, for `dim <= 8`, the
 * leaf's average times its active fraction (`countOneBits` of the value mask
 * / 512): the exact mean when the leaf's inactive voxels are 0 (fog); for
 * `dim <= 128`, the lower node's average, which is the mean of its ACTIVE
 * voxels (NanoVDB stores no active count per lower node), so it overstates a
 * partly empty lower node; larger cells read the lower node too (`dim` is
 * meant to be 8 or 128; no upper level, to keep the module small). Without stats it is the value at `ijk` (no
 * prefilter). CPU twin: `tools` `nodeAverage`.
 *
 * `vdbRayClipSupport(g, o, d, t)`: `vdbRayClip` against the box where
 * `vdbSample` can read the grid's voxels: `[min - 1, max + 1]` for trilinear
 * (the taps of a point in `[min - 1, min)` reach voxel `min`) and
 * `[min - 0.5, max + 0.5]` for nearest. `vdbRayClip`'s box, `[min, max + 1]`,
 * is the voxel-cell box the HDDA walks; it cuts the min-side half voxel of the
 * trilinear support. CPU twin: `tools` `rayClipSupport`.
 */
function helperFunctions(o: ApiOptions): string[] {
  const out: string[] = [];
  if (o.helpers.includes("rayClipSupport")) {
    const pad = o.sampler === "nearest" ? "0.5" : "1.0";
    out.push(`fn vdbRayClipSupport(g: u32, o: vec3f, d: vec3f, t: ptr<function, vec2f>) -> bool {
    if !vdb_live(g) {
        return false;
    }
    var t0 = (*t).x;
    var t1 = (*t).y;
    let hit = vdb_hdda_ray_clip(vec3f(vdbIndexMin(g)) - vec3f(${pad}), vec3f(vdbIndexMax(g)) + vec3f(${pad}), o, &t0, d, &t1);
    *t = vec2f(t0, t1);
    return hit && t0 <= t1;
}`);
  }
  if (o.helpers.includes("cellBoundTrilinear")) {
    out.push(`fn vdbCellBoundTrilinear(acc: ptr<function, VdbAccessor>, ijk: vec3i, dim: u32) -> f32 {
    var m = vdbNodeMax(acc, ijk, dim);
    for (var k = 1u; k < 8u; k++) {
        let c = vec3i(vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u)) * i32(dim);
        m = max(m, vdbNodeMax(acc, ijk + c, dim));
    }
    return m;
}`);
  }
  if (o.helpers.includes("hddaCellEnd")) {
    out.push(`fn vdbHddaCellEnd(h: ptr<function, VdbHdda>) -> f32 {
    return min(min((*h).next.x, (*h).next.y), min((*h).next.z, (*h).tmax));
}`);
  }
  if (o.helpers.includes("nodeAverage")) {
    out.push(`fn vdbNodeAverage(acc: ptr<function, VdbAccessor>, ijk: vec3i, dim: u32) -> f32 {
    if !(*acc).ok {
        return 0.0;
    }
    let a = vdbGetDim(acc, ijk);
    if (a > 1u && dim <= a) || !(*acc).stats || dim <= 1u {
        return vdbReadValue(acc, ijk);
    }
    let t = (*acc).gridType;
    let b = (*acc).buf;
    if dim <= 8u {
        let leaf = (*acc).inner.leaf;
        var n = 0u;
        for (var w = 0u; w < 16u; w++) {
            n += countOneBits(vdb_read_uint32(b, vdb_address_offset(leaf.address, VDB_LEAF_OFF_VALUE_MASK + 4u * w)));
        }
        return vdb_leaf_get_ave_float(t, b, leaf) * (f32(n) / 512.0);
    }
    return vdb_lower_get_ave_float(t, b, (*acc).inner.lower);
}`);
  }
  // K3.7: the level-set zero-crossing finder and normal (./level-set.ts).
  if (o.helpers.includes("levelSet")) out.push(...levelSetWgsl());
  return out;
}

/**
 * API layers in profile order. A profile's API is every layer up to and
 * including its own (`majorant`, rank 3, is deferred past 0.1, D22).
 */
export const API_LAYERS: readonly ((o: ApiOptions, profileRank: number) => string)[] = [readLayer, sampleLayer, marchLayer];

/** The public API text for the profile at `profileRank` (0 = read, 1 = sample, 2 = march), then the requested helpers. */
export function apiWgsl(profileRank: number, o: ApiOptions): string {
  return [...API_LAYERS.slice(0, profileRank + 1).map((layer) => layer(o, profileRank)), ...helperFunctions(o)].join("\n");
}
