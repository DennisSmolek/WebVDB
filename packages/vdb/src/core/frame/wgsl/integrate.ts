// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameIntegrateMaterial.js L33–138 (sRGB blending, scene depth, empty
//   depth), L219–294 (the slice march and composite: the `sliceAlpha` coverage of the cut slice).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: two taps of the integrated prefix at the slice boundaries replace the per-pixel slice march
// (design §3.4; the same partial-slice formula); x, y filtering acts on the prefix. Divergences:
// DIVERGES.md (K6.1 rows).

/**
 * The composite seam (design §2.5): `vdbFrameIntegrateWgsl()` gives hosts
 *
 *   struct VdbFrameIntegrated { radiance: vec3f, transmittance: vec3f }
 *   fn vdbFrameIntegrate(screenUv: vec2f, sceneViewDepth: f32) -> VdbFrameIntegrated;
 *   fn vdbFrameComposite(i: VdbFrameIntegrated, sceneColor: vec4f, srgbBlending: bool) -> vec4f;
 *
 * over `outputs.froxels.integrated` (bind with `vdbFrameIntegrateEntries`), and the core composite pass
 * calls the same functions. The pure functions (no bindings) are also exported one by one for hosts that
 * sample the prefix themselves (`./three` through `wgslFn`). CPU twins: tools/frame `vdbFrameSliceCoord`,
 * `vdbFrameIntegratePrefix`, `vdbFrameSrgbBlend`.
 */

import type { VDBSlot } from "../../types.js";
import { VDB_FRAME_UNIFORMS } from "../uniforms.js";

/** `fn vdbFrameSliceCoord(viewDepth, range, depth) -> vec2f`: (slice k, his coverage c of slice k). */
export const VDB_FRAME_SLICE_COORD_WGSL = `fn vdbFrameSliceCoord(viewDepth: f32, range: vec2f, depth: f32) -> vec2f {
  let ratio = range.y / range.x;
  let s = clamp(depth * log(max(viewDepth, range.x) / range.x) / log(ratio), 0.0, depth);
  let k = min(floor(s), depth - 1.0);
  let start = range.x * pow(ratio, k / depth);
  let end = range.x * pow(ratio, (k + 1.0) / depth);
  return vec2f(k, clamp((viewDepth - start) / (end - start), 0.0, 1.0));
}
`;

/** `fn vdbFrameIntegrateTaps(a, b, c) -> vec4f`: boundaries a, b = (L, 1 − T); returns (radiance, T). */
export const VDB_FRAME_INTEGRATE_TAPS_WGSL = `fn vdbFrameIntegrateTaps(a: vec4f, b: vec4f, c: f32) -> vec4f {
  let ta = 1.0 - a.w;
  let tb = 1.0 - b.w;
  let alpha = clamp(select(0.0, 1.0 - tb / ta, ta > 1e-6), 0.0, 1.0);
  let keep = select(pow(max(1.0 - alpha, 1e-30), c), 1.0, c <= 0.0);
  let w = select(c, (1.0 - keep) / alpha, alpha > 1e-4);
  return vec4f(a.rgb + (b.rgb - a.rgb) * w, max(ta, 0.0) * keep);
}
`;

/** sRGB transfer functions (three's) and `fn vdbFrameCompositeColor(r: vec4f, sceneColor, srgb: u32, clampColors: u32) -> vec4f`. */
export const VDB_FRAME_COMPOSITE_COLOR_WGSL = `fn vdbFrameSrgbOetfColor(color: vec3f) -> vec3f {
  let a = pow(color, vec3f(0.41666)) * 1.055 - 0.055;
  return select(a, color * 12.92, color <= vec3f(0.0031308));
}
fn vdbFrameSrgbEotfColor(color: vec3f) -> vec3f {
  let a = pow(color * 0.9478672986 + 0.0521327014, vec3f(2.4));
  return select(a, color * 0.0773993808, color <= vec3f(0.04045));
}
fn vdbFrameCompositeColor(r: vec4f, sceneColor: vec4f, srgb: u32, clampColors: u32) -> vec4f {
  let alpha = 1.0 - r.w;
  if (srgb == 0u) {
    return vec4f(r.rgb, alpha);
  }
  var volumeColor = vdbFrameSrgbOetfColor(r.rgb / alpha);
  var backgroundColor = vdbFrameSrgbOetfColor(sceneColor.rgb);
  if (clampColors != 0u) {
    volumeColor = clamp(volumeColor, vec3f(0.0), vec3f(1.0));
    backgroundColor = clamp(backgroundColor, vec3f(0.0), vec3f(1.0));
  }
  return vec4f(vdbFrameSrgbEotfColor(mix(backgroundColor, volumeColor, vec3f(alpha))), alpha + sceneColor.a * (1.0 - alpha));
}
`;

/** The pure integrate functions, keyed by name (one function each, for `wgslFn`). */
export const VDB_FRAME_INTEGRATE_FUNCTIONS = {
  vdbFrameSliceCoord: VDB_FRAME_SLICE_COORD_WGSL,
  vdbFrameIntegrateTaps: VDB_FRAME_INTEGRATE_TAPS_WGSL,
} as const;

/** Options of {@link vdbFrameIntegrateWgsl}. */
export interface VDBFrameIntegrateWgslOptions {
  /** Spectral (K6.4: a second prefix); K6.1 has only grey. */
  readonly spectral?: boolean;
  /** Where the three bindings start (texture, sampler, `VdbFrameUniforms`); default group 0, binding 0. */
  readonly slot?: VDBSlot;
  /**
   * Use the frame uniforms already declared as `vdbFrame` (the core composite) instead of declaring the
   * struct and its binding; the texture and sampler are still declared at `slot`.
   */
  readonly sharedUniforms?: boolean;
}

/** The integrate seam's WGSL (design §2.5). */
export function vdbFrameIntegrateWgsl(o: VDBFrameIntegrateWgslOptions = {}): string {
  if (o.spectral) throw new Error("vdbFrameIntegrateWgsl: spectral is K6.4");
  const slot = o.slot ?? { group: 0, firstBinding: 0 };
  const g = slot.group;
  const b = slot.firstBinding;
  const uniforms = o.sharedUniforms
    ? ""
    : `${VDB_FRAME_UNIFORMS.wgsl}@group(${g}) @binding(${b + 2}) var<uniform> vdbFrame: VdbFrameUniforms;\n`;
  return `${uniforms}@group(${g}) @binding(${b}) var vdbFrameIntegratedMap: texture_3d<f32>;
@group(${g}) @binding(${b + 1}) var vdbFrameIntegratedSampler: sampler;

struct VdbFrameIntegrated {
  radiance: vec3f,
  transmittance: vec3f,
};

${VDB_FRAME_SLICE_COORD_WGSL}
${VDB_FRAME_INTEGRATE_TAPS_WGSL}
${VDB_FRAME_COMPOSITE_COLOR_WGSL}
// The volumes in front of the scene depth at a screen position (bottom-left uv): two taps of the prefix
// at the boundaries around the depth's slice, bilinear in x, y.
fn vdbFrameIntegrate(screenUv: vec2f, sceneViewDepth: f32) -> VdbFrameIntegrated {
  let grid = vdbFrame.grid.xyz;
  let kc = vdbFrameSliceCoord(sceneViewDepth, vdbFrame.range.xy, grid.z);
  let xy = clamp(screenUv * grid.xy, vec2f(0.5), grid.xy - 0.5) / grid.xy;
  let a = textureSampleLevel(vdbFrameIntegratedMap, vdbFrameIntegratedSampler, vec3f(xy, (kc.x + 0.5) / (grid.z + 1.0)), 0.0);
  let b = textureSampleLevel(vdbFrameIntegratedMap, vdbFrameIntegratedSampler, vec3f(xy, (kc.x + 1.5) / (grid.z + 1.0)), 0.0);
  let r = vdbFrameIntegrateTaps(a, b, kc.y);
  return VdbFrameIntegrated(r.rgb, vec3f(r.w));
}
// The volumes over the scene colour: sRGB blending as WebGLRenderer (srgbBlending, clamped to [0, 1]
// without tone mapping) or premultiplied (blend with one, one-minus-src-alpha).
fn vdbFrameComposite(i: VdbFrameIntegrated, sceneColor: vec4f, srgbBlending: bool) -> vec4f {
  return vdbFrameCompositeColor(vec4f(i.radiance, i.transmittance.y), sceneColor, select(0u, 1u, srgbBlending), select(0u, 1u, vdbFrame.screen.z > 0.5));
}
`;
}

/** Bind group layout entries of {@link vdbFrameIntegrateWgsl} at `slot`: texture, filtering sampler, uniforms. */
export function vdbFrameIntegrateEntries(slot: VDBSlot = { group: 0, firstBinding: 0 }, visibility: GPUShaderStageFlags = 0x2 /* FRAGMENT */, sharedUniforms = false): GPUBindGroupLayoutEntry[] {
  const b = slot.firstBinding;
  return [
    { binding: b, visibility, texture: { sampleType: "float", viewDimension: "3d" } },
    { binding: b + 1, visibility, sampler: { type: "filtering" } },
    ...(sharedUniforms ? [] : [{ binding: b + 2, visibility, buffer: { type: "uniform" as const } }]),
  ];
}

/** The core composite pass's WGSL (after the frame uniforms): a fullscreen triangle scissored by the host call. */
export function vdbFrameCompositeWgsl(o: { depth: "depth" | "r32float"; srgb: boolean }): string {
  const depthType = o.depth === "depth" ? "texture_depth_2d" : "texture_2d<f32>";
  const depthRead = o.depth === "depth" ? "textureLoad(vdbFrameSceneDepth, pixel, 0)" : "textureLoad(vdbFrameSceneDepth, pixel, 0).r";
  return `${vdbFrameIntegrateWgsl({ slot: { group: 1, firstBinding: 0 }, sharedUniforms: true })}
@group(1) @binding(2) var vdbFrameSceneDepth: ${depthType};
${o.srgb ? "@group(1) @binding(3) var vdbFrameSceneColor: texture_2d<f32>;" : ""}

// The scene's view depth (three's perspective / orthographic depthToViewZ); empty depth: 1e20
fn vdbFrameSceneViewDepth(depth: f32) -> f32 {
  let near = vdbFrame.camera.x;
  let far = vdbFrame.camera.y;
  let reversed = (vdbFrame.flags.z & 1u) != 0u;
  let orthographic = (vdbFrame.flags.z & 2u) != 0u;
  let empty = select(depth >= 1.0, depth <= 0.0, reversed);
  if (empty) {
    return 1e20;
  }
  var viewZ: f32;
  if (orthographic) {
    viewZ = select((near - far) * depth - near, (far - near) * depth - far, reversed);
  } else {
    viewZ = select(near * far / ((far - near) * depth - far), near * far / ((near - far) * depth - near), reversed);
  }
  return -viewZ;
}

@vertex
fn vdbFrameCompositeVs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn vdbFrameCompositeFs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(fragCoord.xy);
  let size = vdbFrame.screen.xy;
  let screenUv = vec2f(fragCoord.x / size.x, 1.0 - fragCoord.y / size.y);
  let integrated = vdbFrameIntegrate(screenUv, vdbFrameSceneViewDepth(${depthRead}));
  if (integrated.transmittance.y >= 1.0) {
    discard;
  }
  ${o.srgb ? "return vdbFrameComposite(integrated, textureLoad(vdbFrameSceneColor, pixel, 0), true);" : "return vdbFrameComposite(integrated, vec4f(0.0), false);"}
}
`;
}
