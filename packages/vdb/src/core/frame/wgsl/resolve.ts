// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameTemporalMaterial.js L31–217 (reprojection into the previous froxel
//   grid, the motion-weighted blend, the 6-neighbour clamp);
//   src/three-openvdb/frame/webgpu/FrameIntegrateMaterial.js L219–294 (the slice march: T < 0.001 stop).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: one compute thread per froxel column walks the slices (design §3.4): the temporal blend runs
// inside the scan (the history sampled with hardware trilinear, the same clamps as his bilinear + slice
// lerp; z neighbours from the column), and the integrate march becomes a prefix of (radiance, opacity
// 1 − T) at the slice boundaries, read by the composite with two taps. Divergences: DIVERGES.md (K6.1 rows).

/**
 * `vdbFrameResolveCs` (design §3.3): per (x, y) column of the dispatch box, z = 0..D−1: with `temporal`
 * his reprojection and clamp (writes the next history), then his integrate step; writes boundary z + 1 of
 * the prefix W × H × (D + 1) = (Σ radiance, 1 − T). Outside this frame's area it writes zeros. CPU twins:
 * tools/frame `vdbFrameReproject`, `vdbFrameTemporalFroxel`, `vdbFrameScanColumn`.
 */

import { frameChunksWgsl } from "./chunks.js";

/** Group-1 bindings of the resolve kernel. */
export const RESOLVE_BINDINGS = { scatter: 0, integrated: 1, historyIn: 2, historyOut: 3 } as const;

/** The resolve kernel's WGSL after `vdbFrameUniformsWgsl()`. */
export function vdbFrameResolveWgsl(o: { temporal: boolean }): string {
  const b = RESOLVE_BINDINGS;
  const temporalBindings = o.temporal
    ? `@group(1) @binding(${b.historyIn}) var vdbFrameHistoryIn: texture_3d<f32>;
@group(1) @binding(${b.historyOut}) var vdbFrameHistoryOut: texture_storage_3d<rgba16float, write>;`
    : "";
  const temporalFns = o.temporal
    ? `
// His temporal pass for one froxel: reproject its centre into the previous froxel grid, blend the history
// (clamped to the current froxel and its 6 neighbours) with the current value by the motion.
fn vdbFrameTemporal(cell: vec3u, current: vec4f, below: vec4f, above: vec4f, hasBelow: bool, hasAbove: bool) -> vec4f {
  let grid = vdbFrame.grid;
  let slice = f32(cell.z);
  let uv = (vec2f(cell.xy) + 0.5) / grid.xy;
  let viewDepth = vdbFrameSliceDepth((slice + 0.5) / grid.z);
  let world = vdbFrame.cameraWorld * vec4f(vdbFrameViewPosition(uv, viewDepth), 1.0);
  let previousView = vdbFrame.previousView * world;
  let previousClip = vdbFrame.previousProjection * previousView;
  if (previousClip.w <= 0.0) {
    return current;
  }
  let previousUv = previousClip.xy / previousClip.w * 0.5 + 0.5;
  let previousRange = vdbFrame.range.zw;
  let previousT = log(max(-previousView.z, 1e-6) / previousRange.x) / log(previousRange.y / previousRange.x);
  let previousSlice = previousT * grid.z - 0.5;
  let inside = min(previousUv.x, previousUv.y) >= 0.0 && max(previousUv.x, previousUv.y) <= 1.0 && previousSlice >= -0.5 && previousSlice <= grid.z - 0.5;
  if (!inside) {
    return current;
  }
  // Bilinear in the slice (cells clamped half a cell inside), linear between the clamped slices: hardware trilinear.
  let historyUv = clamp(previousUv * grid.xy, vec2f(0.5), grid.xy - 0.5) / grid.xy;
  let historyZ = (clamp(previousSlice, 0.0, grid.z - 1.0) + 0.5) / grid.z;
  let history = textureSampleLevel(vdbFrameHistoryIn, vdbFrameLinear, vec3f(historyUv, historyZ), 0.0);
  let motion = length(vec3f((previousUv - uv) * grid.xy, previousSlice - slice));
  let blend = mix(vdbFrame.jitterFade.z, 1.0, clamp(motion * 4.0, 0.0, 1.0));
  let last = vec2i(grid.xy) - 1;
  let c = vec2i(cell.xy);
  let z = i32(cell.z);
  var lo = current;
  var hi = current;
  for (var n = 0; n < 4; n++) {
    var q = c;
    if (n == 0) { q.x = max(c.x - 1, 0); }
    if (n == 1) { q.x = min(c.x + 1, last.x); }
    if (n == 2) { q.y = max(c.y - 1, 0); }
    if (n == 3) { q.y = min(c.y + 1, last.y); }
    let neighbor = textureLoad(vdbFrameScatterIn, vec3i(q, z), 0);
    lo = min(lo, neighbor);
    hi = max(hi, neighbor);
  }
  if (hasBelow) {
    lo = min(lo, below);
    hi = max(hi, below);
  }
  if (hasAbove) {
    lo = min(lo, above);
    hi = max(hi, above);
  }
  return mix(clamp(history, lo, hi), current, vec4f(blend));
}
`
    : "";
  const resolveStep = o.temporal
    ? `    var froxel = vec4f(0.0);
    if (inSlices) {
      froxel = current;
      if (vdbFrame.jitterFade.w > 0.5) {
        froxel = vdbFrameTemporal(vec3u(x, y, z), current, below, above, z > 0u, z + 1u < depth);
      }
    }
    textureStore(vdbFrameHistoryOut, vec3u(x, y, z), froxel);`
    : `    let froxel = current;`;
  const clearHistory = o.temporal ? `      textureStore(vdbFrameHistoryOut, vec3u(x, y, z), vec4f(0.0));\n` : "";

  return `${frameChunksWgsl(["sliceDepth", "viewPosition"])}
@group(1) @binding(${b.scatter}) var vdbFrameScatterIn: texture_3d<f32>;
@group(1) @binding(${b.integrated}) var vdbFrameIntegratedOut: texture_storage_3d<rgba16float, write>;
${temporalBindings}

const VDB_FRAME_SCAN_STOP: f32 = 0.001;
${temporalFns}
@compute @workgroup_size(8, 8, 1)
fn vdbFrameResolveCs(@builtin(global_invocation_id) gid: vec3u) {
  let box = vdbFrame.dispatchBox;
  if (gid.x >= box.z || gid.y >= box.w) {
    return;
  }
  let x = box.x + gid.x;
  let y = box.y + gid.y;
  let depth = u32(vdbFrame.grid.z);
  let area = vdbFrame.areaRect;
  let areaSlices = vdbFrame.areaSlices;
  let p = vec2f(f32(x), f32(y));
  let inRect = p.x >= area.x && p.y >= area.y && p.x < area.z && p.y < area.w;
  textureStore(vdbFrameIntegratedOut, vec3u(x, y, 0u), vec4f(0.0));
  if (!inRect) {
    for (var z = 0u; z < depth; z++) {
${clearHistory}      textureStore(vdbFrameIntegratedOut, vec3u(x, y, z + 1u), vec4f(0.0));
    }
    return;
  }
  var radiance = vec3f(0.0);
  var transmittance = 1.0;
  var below = vec4f(0.0);
  var current = textureLoad(vdbFrameScatterIn, vec3u(x, y, 0u), 0);
  for (var z = 0u; z < depth; z++) {
    let above = textureLoad(vdbFrameScatterIn, vec3u(x, y, min(z + 1u, depth - 1u)), 0);
    let inSlices = f32(z) >= areaSlices.x && f32(z) < areaSlices.y;
${resolveStep}
    // his integrate step over a whole slice; nothing more once T < 0.001
    let alpha = clamp(froxel.w, 0.0, 1.0);
    if (transmittance >= VDB_FRAME_SCAN_STOP && alpha > 0.0) {
      radiance += transmittance * froxel.rgb;
      transmittance *= 1.0 - alpha;
    }
    textureStore(vdbFrameIntegratedOut, vec3u(x, y, z + 1u), vec4f(radiance, 1.0 - transmittance));
    below = current;
    current = above;
  }
}
`;
}
