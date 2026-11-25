// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameVolumeShadowMaterial.js L27–48 (frameShadowDepth), L90–172 (the Beer shadow
//   map pass); src/three-openvdb/frame/webgpu/FrameVolumeShadows.js L248–290 (getCascadeLookup), L344–372 (the
//   receiver transmittance); past the pin: his 62a6e3e (receivers read the state the map was drawn with).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: a compute kernel per map piece (K6.5, design §2.6, §3.7) instead of a fullscreen pass per tile or cube
// face: texels from the drawing camera's inverse view-projection (spot, cascades) or the cube face table (point); the
// maps are 2d-array textures (cascades / faces as layers; a point map is sampled through a cube view). Receivers are WGSL
// functions the host calls with the map and its drawn view (a uniform buffer the core writes when it draws the piece).
// CPU twins: tools/frame/shadows.ts. Divergences: DIVERGES.md (K6.5 rows).

/**
 * The volume shadow maps on meshes (K6.5):
 *
 * - `vdbFrameVolumeShadowCs` (8×8×1; a piece's size² × layers): one Beer map texel per thread from the last shadow
 *   grid cascade: (τ·first, τ·last, 0, τ) with first / last the depths (÷ far) of the first and last non-empty steps.
 * - `vdbVolumeShadowWgsl()`: `vdbVolumeShadow(map, smp, view, world, viewDepth)` (directional cascades and spot maps,
 *   a `texture_2d_array`) and `vdbVolumeShadowCube(map, smp, view, world)` (point maps through a `texture_cube` view):
 *   the transmittance at a receiver, read with `view`, the state the map was drawn with (his 62a6e3e by construction).
 */

import { vdbUniformStruct } from "../uniforms.js";

/** Most layers of a map (directional cascades; a point map has 6 faces). */
export const VDB_VOLUME_SHADOW_MAX_CASCADES = 4;

/** One map piece the shadow kernel draws (group 1, dynamic offset, 256 B stride). */
export const VDB_FRAME_SHADOW_PIECE = vdbUniformStruct("VdbFrameShadowPiece", [
  // clip → world of the camera that draws the piece (spot, cascade); unused for a cube
  { name: "inverseViewProjection", type: "mat4x4f" },
  // camera position (the perspective ray origin; the depth origin), perspective (0/1)
  { name: "camera", type: "vec4f" },
  // forward axis (orthographic depth), far plane (depths are stored ÷ far)
  { name: "forward", type: "vec4f" },
  // first layer, size (texels per side), kind (0 camera, 2 cube: 6 layers from the first), march steps
  { name: "info", type: "vec4u" },
]);

/** Dynamic-offset stride of the piece records. */
export const VDB_FRAME_SHADOW_PIECE_STRIDE = 256;

/** The drawn view of one map (a uniform buffer per map; the receivers' `view`). */
export const VDB_VOLUME_SHADOW_VIEW = vdbUniformStruct("VdbVolumeShadowView", [
  // the view-projection each layer was drawn with (cascades; spot: [0])
  { name: "viewProjection", type: "mat4x4f", count: VDB_VOLUME_SHADOW_MAX_CASCADES },
  // the ray origin of each layer (xyz) and its far plane (w)
  { name: "origin", type: "vec4f", count: VDB_VOLUME_SHADOW_MAX_CASCADES },
  // forward axis (xyz; orthographic depth), perspective (0/1)
  { name: "forward", type: "vec4f" },
  // the cascade far distances (view depth) the layers were drawn for
  { name: "breaks", type: "vec4f" },
  // kind (0 directional cascades, 1 spot, 2 point), cascade count, size, enabled (0: transmittance 1)
  { name: "info", type: "vec4u" },
]);

/** Group-1 bindings of the shadow map kernel. */
export const VOLUME_SHADOW_BINDINGS = { piece: 0, grid: 1, map: 2 } as const;

/** The receiver functions' WGSL (the `VdbVolumeShadowView` struct and the Beer depth maths) without the struct. */
const RECEIVER_FUNCTIONS = `// The depth of a world point for layer l of a drawn map: distance from the origin (perspective) or along forward
fn vdbVolumeShadowDepthOf(view: VdbVolumeShadowView, layer: u32, world: vec3f) -> f32 {
  let offset = world - view.origin[layer].xyz;
  return select(dot(offset, view.forward.xyz), length(offset), view.forward.w > 0.5);
}
// The Beer map texel at a receiver depth: the optical depth grows linearly from the first to the last density
fn vdbVolumeShadowBeer(texel: vec4f, depth: f32, far: f32) -> f32 {
  let opticalDepth = texel.w;
  if (!(opticalDepth > 1e-4)) {
    return 1.0;
  }
  let first = texel.x / opticalDepth * far;
  let last = texel.y / opticalDepth * far;
  let along = clamp((depth - first) / max(last - first, 1e-4), 0.0, 1.0);
  return exp(-opticalDepth * along);
}
// Directional (cascades by the receiver's view depth) and spot maps: the volume transmittance at a world point,
// read with the view the map was drawn with; 1 outside the map, or when the map is disabled
fn vdbVolumeShadow(map: texture_2d_array<f32>, smp: sampler, view: VdbVolumeShadowView, world: vec3f, viewDepth: f32) -> f32 {
  if (view.info.w == 0u) {
    return 1.0;
  }
  var layer = 0u;
  var uv: vec2f;
  if (view.info.x == 0u) {
    let count = view.info.y;
    for (var i = 0u; i < ${VDB_VOLUME_SHADOW_MAX_CASCADES - 1}u; i++) {
      if (viewDepth > view.breaks[i] && i + 1u < count) {
        layer += 1u;
      }
    }
    let clip = view.viewProjection[layer] * vec4f(world, 1.0);
    let tile = vec2f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    let lastBreak = view.breaks[min(max(count, 1u) - 1u, 3u)];
    if (!(viewDepth <= lastBreak && all(tile >= vec2f(0.0)) && all(tile <= vec2f(1.0)))) {
      return 1.0;
    }
    let inset = 0.5 / f32(view.info.z);
    uv = clamp(tile, vec2f(inset), vec2f(1.0 - inset));
  } else {
    let clip = view.viewProjection[0] * vec4f(world, 1.0);
    let ndc = clip.xy / clip.w;
    uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (!(clip.w > 0.0 && all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)))) {
      return 1.0;
    }
  }
  let texel = textureSampleLevel(map, smp, uv, layer, 0.0);
  return vdbVolumeShadowBeer(texel, vdbVolumeShadowDepthOf(view, layer, world), view.origin[layer].w);
}
// Point maps (a cube view of the 6 faces): the transmittance along the direction from the drawn light position
fn vdbVolumeShadowCube(map: texture_cube<f32>, smp: sampler, view: VdbVolumeShadowView, world: vec3f) -> f32 {
  if (view.info.w == 0u) {
    return 1.0;
  }
  let offset = world - view.origin[0].xyz;
  let texel = textureSampleLevel(map, smp, offset, 0.0);
  return vdbVolumeShadowBeer(texel, length(offset), view.origin[0].w);
}
`;

/**
 * WGSL for hosts that receive volume shadows in their own shaders (design §2.5): the `VdbVolumeShadowView` struct and
 *
 *     fn vdbVolumeShadow(map: texture_2d_array<f32>, smp: sampler, view: VdbVolumeShadowView, world: vec3f, viewDepth: f32) -> f32
 *     fn vdbVolumeShadowCube(map: texture_cube<f32>, smp: sampler, view: VdbVolumeShadowView, world: vec3f) -> f32
 *
 * Bind a map's `texture` (a `'2d-array'` view; point maps a `'cube'` view), a filtering sampler (linear, clamp to edge)
 * and its `view` buffer (`var<uniform>`). `viewDepth` is the receiver's distance in front of the camera (directional
 * cascades). Multiply the result into the light's shadow. CPU twin: tools/frame `vdbFrameVolumeShadowLookup`.
 */
export function vdbVolumeShadowWgsl(): string {
  return `${VDB_VOLUME_SHADOW_VIEW.wgsl}${RECEIVER_FUNCTIONS}`;
}

/** The shadow map kernel (after `vdbFrameUniformsWgsl()`; source-free: it reads the shadow grid stack). */
export function vdbFrameVolumeShadowWgsl(): string {
  const b = VOLUME_SHADOW_BINDINGS;
  return `${VDB_FRAME_SHADOW_PIECE.wgsl}
@group(1) @binding(${b.piece}) var<uniform> vdbFrameShadowPiece: VdbFrameShadowPiece;
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_3d<f32>;
@group(1) @binding(${b.map}) var vdbFrameShadowMapOut: texture_storage_2d_array<rgba16float, write>;

const VDB_FRAME_SHADOW_MAX_OPTICAL_DEPTH: f32 = 8.0;

// The cube face table (WebGPU / Vulkan face selection): face 0..5 (+x, -x, +y, -y, +z, -z), (a, b) = (s, t) * 2 - 1
fn vdbFrameShadowCubeDirection(face: u32, a: f32, b: f32) -> vec3f {
  switch (face) {
    case 0u: { return vec3f(1.0, -b, -a); }
    case 1u: { return vec3f(-1.0, -b, a); }
    case 2u: { return vec3f(a, 1.0, b); }
    case 3u: { return vec3f(a, -1.0, -b); }
    case 4u: { return vec3f(a, -b, 1.0); }
    default: { return vec3f(-a, -b, -1.0); }
  }
}
fn vdbFrameShadowUnproject(ndc: vec2f, z: f32) -> vec3f {
  let p = vdbFrameShadowPiece.inverseViewProjection * vec4f(ndc, z, 1.0);
  return p.xyz / p.w;
}
// frameShadowDepth for the piece's camera
fn vdbFrameShadowPieceDepth(world: vec3f) -> f32 {
  let offset = world - vdbFrameShadowPiece.camera.xyz;
  return select(dot(offset, vdbFrameShadowPiece.forward.xyz), length(offset), vdbFrameShadowPiece.camera.w > 0.5);
}

// One Beer map texel: the texel's light ray marched through the last grid cascade with the piece's steps, each
// sample at the middle of its step; stops past optical depth 8. Empty texels are 0 (the linear filter stays right).
@compute @workgroup_size(8, 8, 1)
fn vdbFrameVolumeShadowCs(@builtin(global_invocation_id) gid: vec3u) {
  let info = vdbFrameShadowPiece.info;
  let size = info.y;
  if (gid.x >= size || gid.y >= size) {
    return;
  }
  let st = (vec2f(gid.xy) + 0.5) / f32(size);
  var origin = vdbFrameShadowPiece.camera.xyz;
  var direction: vec3f;
  var layer = info.x;
  if (info.z == 2u) {
    direction = normalize(vdbFrameShadowCubeDirection(gid.z, st.x * 2.0 - 1.0, st.y * 2.0 - 1.0));
    layer = info.x + gid.z;
  } else {
    let ndc = vec2f(st.x * 2.0 - 1.0, 1.0 - st.y * 2.0);
    let near = vdbFrameShadowUnproject(ndc, 0.25);
    direction = normalize(vdbFrameShadowUnproject(ndc, 0.75) - near);
    if (vdbFrameShadowPiece.camera.w < 0.5) {
      origin = near - direction * dot(near - vdbFrameShadowPiece.camera.xyz, direction);
    }
  }
  var opticalDepth = 0.0;
  var first = 0.0;
  var last = 0.0;
  let count = u32(vdbFrame.gridLayout.x);
  if (count > 0u) {
    let cascade = count - 1u;
    let boxMin = vdbFrame.gridMin[cascade].xyz;
    let boxMax = boxMin + vdbFrame.gridSize[cascade].xyz;
    let inverseDirection = vec3f(1.0) / direction;
    let tA = (boxMin - origin) * inverseDirection;
    let tB = (boxMax - origin) * inverseDirection;
    let tMin = min(tA, tB);
    let tMax = max(tA, tB);
    let tStart = max(0.0, max(tMin.x, max(tMin.y, tMin.z)));
    let tEnd = min(tMax.x, min(tMax.y, tMax.z));
    if (tEnd > tStart) {
      let steps = max(info.w, 1u);
      let stepLength = (tEnd - tStart) / f32(steps);
      var firstT = -1.0;
      var lastT = -1.0;
      for (var i = 0u; i < steps; i++) {
        if (opticalDepth > VDB_FRAME_SHADOW_MAX_OPTICAL_DEPTH) {
          break;
        }
        let t = tStart + f32(i) * stepLength;
        let p = origin + direction * (t + stepLength * 0.5);
        let extinction = vdbFrameStackSample(vdbFrameShadowGrid, vdbFrame.shadowOrigin[cascade].xyz, vdbFrameGridUvw(cascade, p)).x;
        if (extinction > 0.0) {
          if (firstT < 0.0) {
            firstT = t;
          }
          lastT = t + stepLength;
          opticalDepth += extinction * stepLength;
        }
      }
      if (opticalDepth > 0.0) {
        let far = vdbFrameShadowPiece.forward.w;
        first = vdbFrameShadowPieceDepth(origin + direction * firstT) / far;
        last = vdbFrameShadowPieceDepth(origin + direction * lastT) / far;
      }
    }
  }
  textureStore(vdbFrameShadowMapOut, gid.xy, layer, vec4f(first * opticalDepth, last * opticalDepth, 0.0, opticalDepth));
}
`;
}

/** The `?probe=shadow` receiver kernel (harness): `vdbVolumeShadow` / `vdbVolumeShadowCube` on rows of (world, view depth). */
export function vdbVolumeShadowProbeWgsl(cube: boolean): string {
  return `${vdbVolumeShadowWgsl()}
@group(0) @binding(0) var<uniform> vdbProbeView: VdbVolumeShadowView;
@group(0) @binding(1) var vdbProbeMap: ${cube ? "texture_cube<f32>" : "texture_2d_array<f32>"};
@group(0) @binding(2) var vdbProbeSampler: sampler;
@group(0) @binding(3) var<storage, read_write> vdbProbeRows: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn vdbVolumeShadowProbeCs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= arrayLength(&vdbProbeRows)) {
    return;
  }
  let row = vdbProbeRows[gid.x];
  vdbProbeRows[gid.x] = vec4f(${cube ? "vdbVolumeShadowCube(vdbProbeMap, vdbProbeSampler, vdbProbeView, row.xyz)" : "vdbVolumeShadow(vdbProbeMap, vdbProbeSampler, vdbProbeView, row.xyz, row.w)"}, row.yzw);
}
`;
}
