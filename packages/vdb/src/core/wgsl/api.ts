/**
 * The public `vdb*` WGSL API layered over the vendored traversal
 * (docs/design/K1-core.md §3.2). Written in the generated module's final names:
 * vendored declarations appear renamed (`vdb_*`, `VDB_*`), and the generator
 * pulls in exactly the vendored declarations this text references.
 *
 * Every pool read goes through `vdb_load`, the module's one buffer access,
 * which the generator emits for the chosen binding mode.
 *
 * Layers stack: `sample` includes `read`. `march` and `majorant` (K1.4b) add
 * layers here; each layer is a function of the normalized options.
 */

import type { VDBGridTypeName } from "./types.js";

/** Options the API layer depends on (normalized by the generator). */
export interface ApiOptions {
  readonly types: readonly VDBGridTypeName[];
  readonly isActive: boolean;
  readonly sampler: "trilinear" | "nearest";
  readonly outside: "background" | "zero";
}

/** Vendored grid-type constant names (after the rename) per compiled type. */
export const GRID_TYPE_CONSTANT: Readonly<Record<VDBGridTypeName, string>> = {
  float: "VDB_GRID_TYPE_FLOAT",
  fp4: "VDB_GRID_TYPE_FP4",
  fp8: "VDB_GRID_TYPE_FP8",
  fp16: "VDB_GRID_TYPE_FP16",
  fpn: "VDB_GRID_TYPE_FPN",
};

/** Whether the accessor carries the index bbox (only `sample` with `outside: 'zero'`). */
function carriesBBox(profileRank: number, o: ApiOptions): boolean {
  return profileRank >= 1 && o.outside === "zero";
}

function readLayer(o: ApiOptions, profileRank: number): string {
  const bbox = carriesBBox(profileRank, o);
  const typeOk = o.types.map((t) => `t == ${GRID_TYPE_CONSTANT[t]}`).join(" || ");
  const parts = [
    `struct VdbAccessor {
    buf: vdb_buf_t,
    gridType: u32,
    ok: bool,
    inner: vdb_readaccessor_t,${bbox ? "\n    lo: vec3f,\n    hi: vec3f," : ""}
}`,
    `fn vdb_rec(g: u32, w: u32) -> u32 {
    return vdb_load(VDB_POOL_TABLE_WORD + g * VDB_POOL_RECORD_WORDS + w);
}`,
    `fn vdb_type_ok(t: u32) -> bool {
    return ${typeOk};
}`,
    `fn vdbGridCount() -> u32 {
    return vdb_load(VDB_POOL_HEADER_GRID_COUNT);
}`,
    `fn vdbGridType(g: u32) -> u32 {
    if g >= vdbGridCount() || (vdb_rec(g, VDB_REC_FLAGS) & VDB_REC_FLAG_LIVE) == 0u {
        return 0u;
    }
    return vdb_rec(g, VDB_REC_GRID_TYPE);
}`,
    `fn vdbIndexMin(g: u32) -> vec3i {
    return bitcast<vec3i>(vec3u(vdb_rec(g, VDB_REC_INDEX_BBOX_MIN), vdb_rec(g, VDB_REC_INDEX_BBOX_MIN + 1u), vdb_rec(g, VDB_REC_INDEX_BBOX_MIN + 2u)));
}`,
    `fn vdbIndexMax(g: u32) -> vec3i {
    return bitcast<vec3i>(vec3u(vdb_rec(g, VDB_REC_INDEX_BBOX_MAX), vdb_rec(g, VDB_REC_INDEX_BBOX_MAX + 1u), vdb_rec(g, VDB_REC_INDEX_BBOX_MAX + 2u)));
}`,
    `fn vdbAccessorInit(g: u32) -> VdbAccessor {
    var a: VdbAccessor;
    a.gridType = vdbGridType(g);
    a.ok = vdb_type_ok(a.gridType);
    var root = vdb_root_handle_t(0u);
    if a.ok {
        a.buf = vdb_make_buffer(vdb_rec(g, VDB_REC_IMAGE_OFFSET), vdb_rec(g, VDB_REC_IMAGE_BYTES) >> 2u);
        root = vdb_tree_get_root(a.buf, vdb_grid_get_tree(vdb_grid_handle_t(0u)));${
          bbox ? "\n        a.lo = vec3f(vdbIndexMin(g));\n        a.hi = vec3f(vdbIndexMax(g) + vec3i(1));" : ""
        }
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

function sampleLayer(o: ApiOptions, profileRank: number): string {
  const bbox = carriesBBox(profileRank, o);
  const guard = bbox ? "!(*acc).ok || any(xyz < (*acc).lo) || any(xyz > (*acc).hi)" : "!(*acc).ok";
  const lookup =
    o.sampler === "trilinear"
      ? "vdb_sample_trilinear_typed((*acc).gridType, (*acc).buf, &inner, xyz)"
      : "vdb_readaccessor_get_value_float((*acc).gridType, (*acc).buf, &inner, vec3i(floor(xyz + vec3f(0.5))))";
  return [
    `fn vdb_row(b: u32) -> vec4f {
    return bitcast<vec4f>(vec4u(vdb_load(b), vdb_load(b + 1u), vdb_load(b + 2u), vdb_load(b + 3u)));
}`,
    `fn vdb_affine(g: u32, w: u32, p: vec4f) -> vec3f {
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
    `fn vdbSample(acc: ptr<function, VdbAccessor>, xyz: vec3f) -> f32 {
    if ${guard} {
        return 0.0;
    }
    var inner = (*acc).inner;
    let v = ${lookup};
    (*acc).inner = inner;
    return v;
}`,
    `fn vdbRayClip(g: u32, o: vec3f, d: vec3f, t: ptr<function, vec2f>) -> bool {
    var t0 = (*t).x;
    var t1 = (*t).y;
    let hit = vdb_hdda_ray_clip(vec3f(vdbIndexMin(g)), vec3f(vdbIndexMax(g) + vec3i(1)), o, &t0, d, &t1);
    *t = vec2f(t0, t1);
    return hit && t0 <= t1;
}`,
  ].join("\n");
}

/**
 * API layers in profile order. A profile's API is every layer up to and
 * including its own. K1.4b appends `march` (and `majorant`) here.
 */
export const API_LAYERS: readonly ((o: ApiOptions, profileRank: number) => string)[] = [readLayer, sampleLayer];

/** The public API text for the profile at `profileRank` (0 = read, 1 = sample, ...). */
export function apiWgsl(profileRank: number, o: ApiOptions): string {
  return API_LAYERS.slice(0, profileRank + 1)
    .map((layer) => layer(o, profileRank))
    .join("\n");
}
