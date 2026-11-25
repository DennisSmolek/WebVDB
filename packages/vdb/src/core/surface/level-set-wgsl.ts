/**
 * The level-set surface function (K6.6, design §3.9): the fragment-stage wrapper of the `'levelSet'`
 * helper's trace for hosts that shade with their own BRDF. `VDBLevelSetMaterial` (`./three`) calls it
 * through `wgslFn`; the raw reference pass (`vdbCreateLevelSetPassGpu`) calls it from its fragment entry.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/levelset/webgpu/LevelSetSurfaceMaterial.js`: perspective and orthographic view rays,
 * so shadow passes work). Copyright (c) mjurczyk/openvdb contributors, MIT (see THIRD_PARTY_NOTICES.md).
 * Changes: the ray comes from the pixel centre through the inverse projection, not from interpolated box
 * positions (V57); the hit, normal and NDC depth come back as one struct (DIVERGES.md).
 *
 * CPU twins: `tools` `levelSetPixelUv`, `levelSetViewRay`, `levelSetSurface` (`tools/cpu/level-set.ts`).
 */

/**
 * The WGSL text of the surface functions. Append it after a VDB module generated with profile `'march'`
 * and helpers including `'levelSet'` (`vdbModuleWgsl({ ..., profile: 'march', helpers: ['levelSet'] })`):
 * it calls `vdbLevelSetHit`, `vdbLevelSetNormal`, `vdbWorldToIndex` and `vdbWorldToIndexDir`. One VDB
 * module per shader (Traps); this text has no bindings of its own.
 *
 * ```wgsl
 * struct VdbLevelSetRay { origin: vec3f, dir: vec3f, span: vec2f }          // world space; dir unit length
 * struct VdbLevelSetSurface { hit: bool, t: f32, world: vec3f, normal: vec3f, ndcDepth: f32 }
 * fn vdbLevelSetPixelUv(fragCoord: vec2f, viewport: vec4f) -> vec2f;       // viewport = (x, y, width, height)
 * fn vdbLevelSetViewRay(uv: vec2f, projectionInverse: mat4x4f, cameraWorld: mat4x4f, orthographic: bool) -> VdbLevelSetRay;
 * fn vdbLevelSetSurface(g: u32, ray: VdbLevelSetRay, viewProjection: mat4x4f, sdfToIndex: f32, maxSamples: u32, normalStep: f32) -> VdbLevelSetSurface;
 * ```
 *
 * - `vdbLevelSetPixelUv`: the pixel centre (`@builtin(position).xy` is the centre already) as a viewport
 *   uv, x right and y down, in `[0, 1]`.
 * - `vdbLevelSetViewRay`: the pixel's view ray (V57). Perspective: from the camera position
 *   (`cameraWorld[3]`) through `projectionInverse · (ndc, 0.5, 1)`, `t` in `[0, 1e30]`. Orthographic: from
 *   that point on the camera plane (view z = 0) along the view axis, `t` in `[-1e5, 1e5]`, so a
 *   directional-light shadow camera sees surfaces behind its plane. `t` is in world units.
 * - `vdbLevelSetSurface`: traces grid `g` along `ray` (`vdbLevelSetHit` in the grid's index space;
 *   `sdfToIndex` index voxels per SDF unit, 1 / voxel size for a level set in its own units; `maxSamples`
 *   trilinear samples). On a hit: `t`, the world hit `origin + dir · t`, the outward world unit normal
 *   (`vdbLevelSetNormal`, central differences at `normalStep` index voxels through `W^T`), and the NDC depth
 *   `clip.z / clip.w` of `viewProjection · (world, 1)` (WebGPU depth: 0 near, 1 far, or reversed when the
 *   host's projection is). On a miss every field is zero and `hit` is false.
 *
 * Every `vdbSample` call site inlines the type-dispatched trilinear reader (K3.7: compile time). The
 * surface has the helper's four sites (march, closing sample, bisection, gradient loop); call it from
 * one place per shader.
 */
export function vdbLevelSetSurfaceWgsl(): string {
  return LEVEL_SET_SURFACE_WGSL;
}

const LEVEL_SET_SURFACE_WGSL = /* wgsl */ `struct VdbLevelSetRay {
    origin: vec3f,
    dir: vec3f,
    span: vec2f,
};

struct VdbLevelSetSurface {
    hit: bool,
    t: f32,
    world: vec3f,
    normal: vec3f,
    ndcDepth: f32,
};

fn vdbLevelSetPixelUv(fragCoord: vec2f, viewport: vec4f) -> vec2f {
    return (fragCoord - viewport.xy) / viewport.zw;
}

fn vdbLevelSetViewRay(uv: vec2f, projectionInverse: mat4x4f, cameraWorld: mat4x4f, orthographic: bool) -> VdbLevelSetRay {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let clip = projectionInverse * vec4f(ndc, 0.5, 1.0);
    let view = clip.xyz / clip.w;
    var r: VdbLevelSetRay;
    if orthographic {
        r.origin = (cameraWorld * vec4f(view.x, view.y, 0.0, 1.0)).xyz;
        r.dir = normalize((cameraWorld * vec4f(0.0, 0.0, -1.0, 0.0)).xyz);
        r.span = vec2f(-1.0e5, 1.0e5);
    } else {
        r.origin = cameraWorld[3].xyz;
        r.dir = normalize((cameraWorld * vec4f(normalize(view), 0.0)).xyz);
        r.span = vec2f(0.0, 1.0e30);
    }
    return r;
}

fn vdbLevelSetSurface(g: u32, ray: VdbLevelSetRay, viewProjection: mat4x4f, sdfToIndex: f32, maxSamples: u32, normalStep: f32) -> VdbLevelSetSurface {
    var s: VdbLevelSetSurface;
    var span = ray.span;
    if !vdbLevelSetHit(g, vdbWorldToIndex(g, ray.origin), vdbWorldToIndexDir(g, ray.dir), &span, sdfToIndex, maxSamples) {
        return s;
    }
    s.hit = true;
    s.t = span.x;
    s.world = ray.origin + ray.dir * span.x;
    s.normal = vdbLevelSetNormal(g, vdbWorldToIndex(g, s.world), normalStep);
    let clip = viewProjection * vec4f(s.world, 1.0);
    s.ndcDepth = clip.z / clip.w;
    return s;
}
`;
