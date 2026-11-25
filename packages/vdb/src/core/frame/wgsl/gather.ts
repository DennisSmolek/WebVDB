// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameInjectMaterial.js L142–284 (froxel pre-integration, the shadow
//   march), L395–398 (ambient and light probe), L453–532 (hemisphere and scene lights, radiance,
//   premultiplied output); src/three-openvdb/frame/webgpu/frameShaderChunks.js L894–1017 (wrapUv, gridUv,
//   sampleExtinction, sampleAlbedo, sampleEmission).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: one compute thread per froxel loops the volumes of a source group (his: one blended draw per
// volume); the shadow march is one function called from one loop over the hemisphere and scene lights
// (two density call sites per kernel); the light terms are loops over uniform counts (V55's compiled-in
// sets go away); overlapping volumes mix order-independently by default ('mixture'); multi-group frames
// accumulate in packed partials; the environment (K6.2b) reads the GPU environment buffer (its lights, or the
// 'outward' SH irradiance instead of three's PMREM). Octaves, two-sided, Lorenz-Mie: K6.3;
// spectral: K6.4. Divergences: DIVERGES.md (K6.1 rows).

/**
 * `vdbFrameGatherCs`: the 'volume'-lighting gather kernel (design §3.3, §3.4). One thread per froxel of
 * the dispatch box (the union of this frame's froxel area and what the scatter texture last held):
 * outside this frame's area it writes 0; inside it lights every volume of its source group in the froxel
 * (his inject pass) and combines them (`'mixture'` or `'over'`). CPU twins: tools/frame `vdbFrameInjectSlice`,
 * `vdbFrameShadowMarch`, `vdbFrameShadowedLightColor`, `vdbFrameGatherFroxel`.
 */

import type { VDBVolumeChannels } from "../../source/types.js";
import { frameChunksWgsl } from "./chunks.js";

/** Pipeline-key options of the gather kernel. */
export interface VDBFrameGatherOptions {
  /** How volumes in one froxel combine. */
  readonly mix: "mixture" | "over";
  /** More than one source group this frame: accumulate in the packed r32uint partials (design §3.2). */
  readonly multi: boolean;
  /** The source's channels (absent ones compile to constants). */
  readonly channels: VDBVolumeChannels;
}

/** Group-1 bindings of the gather kernel. */
export const GATHER_BINDINGS = { volumes: 0, group: 1, scatter: 2, partials: 3 } as const;

/**
 * The volume-record functions every source kernel shares (gather 'volume' and 'frame', gridDensity): his gridUv,
 * sampleExtinction, sampleAlbedo and sampleEmission over the group-2 source, reading `vdbFrameVolumes` (group 1).
 */
export function vdbFrameSourceFunctionsWgsl(channels: VDBVolumeChannels): string {
  const emission = channels.emission
    ? `  let e = vdbFrameVolumes.v[vi].emission;
  let value = vdbSrcEmission(vdbFrameVolumes.v[vi].slices.z, vdbFrameGridUv(vi, localPosition), lod);
  let temperature = mix(e.y, e.z, value);
  let power = pow(temperature / max(e.z, 1.0), 4.0);
  return select(vec3f(0.0), vec3f(1.0), temperature > 0.0) * power * e.x;`
    : `  return vec3f(0.0);`;
  return `fn vdbFrameMirror(x: f32) -> f32 {
  let period = x - 2.0 * floor(x / 2.0);
  return select(period, 2.0 - period, period > 1.0);
}
// gridUv: the local position as an offset, wrapped uvw (wrap 0 clamp, 1 repeat, 2 mirrored repeat)
fn vdbFrameGridUv(vi: u32, localPosition: vec3f) -> vec3f {
  let offsetWrap = vdbFrameVolumes.v[vi].offsetWrap;
  let uvw = localPosition + 0.5 + offsetWrap.xyz;
  if (offsetWrap.w > 1.5) {
    return vec3f(vdbFrameMirror(uvw.x), vdbFrameMirror(uvw.y), vdbFrameMirror(uvw.z));
  }
  if (offsetWrap.w > 0.5) {
    return fract(uvw);
  }
  return uvw;
}
// sampleExtinction: extinction per world unit (the mask is read at local + 0.5, no offset or wrap)
fn vdbFrameExtinction(vi: u32, localPosition: vec3f, lod: f32) -> f32 {
  let v = vdbFrameVolumes.v[vi].slices.z;
  let mask = vdbSrcMask(v, localPosition + 0.5, lod);
  let value = vdbSrcDensity(v, vdbFrameGridUv(vi, localPosition), lod) * mask;
  let d = vdbFrameVolumes.v[vi].density;
  return select(0.0, value, value > d.y) * d.x * vdbFrameVolumes.v[vi].baseExtinction.w / d.z;
}
// sampleAlbedo: base colour times the albedo channel
fn vdbFrameAlbedo(vi: u32, localPosition: vec3f, lod: f32) -> vec3f {
  return vdbFrameVolumes.v[vi].baseExtinction.xyz * vdbSrcAlbedo(vdbFrameVolumes.v[vi].slices.z, vdbFrameGridUv(vi, localPosition), lod);
}
// sampleEmission (colour white; emission colour data and hooks: K6.5)
fn vdbFrameEmission(vi: u32, localPosition: vec3f, lod: f32) -> vec3f {
${emission}
}
`;
}

/**
 * The slice pre-integration of one volume in one froxel (his inject / density passes, L142–200): the box segment,
 * the sub-steps at the froxel lod and the visibility-weighted mean position. Shared by the 'volume' and 'frame'
 * gathers (one density call site).
 */
export const VDB_FRAME_SLICE_WGSL = `// One volume's slice: (alpha, visible t, extinction sum / steps * segment, read lod); alpha 0: nothing.
struct VdbFrameSlice {
  alpha: f32,
  visibleT: f32,
  extinction: f32,
  readLod: f32,
  lod: f32,
  maxTexels: f32,
  localNear: vec3f,
  localFar: vec3f,
};
fn vdbFrameSliceOf(vi: u32, cell: vec3u, f: VdbFrameFroxel) -> VdbFrameSlice {
  var s: VdbFrameSlice;
  s.alpha = 0.0;
  let rect = vdbFrameVolumes.v[vi].rect;
  let slices = vdbFrameVolumes.v[vi].slices;
  let p = vec2f(cell.xy);
  if (p.x < rect.x || p.y < rect.y || p.x >= rect.z || p.y >= rect.w || cell.z < slices.x || cell.z >= slices.x + slices.y) {
    return s;
  }
  let v = slices.z;
  let viewToLocal = vdbFrameVolumes.v[vi].viewToLocal;
  s.localNear = (viewToLocal * vec4f(f.viewNear, 1.0)).xyz;
  s.localFar = (viewToLocal * vec4f(f.viewFar, 1.0)).xyz;
  let boxSegment = vdbFrameBoxSegment(s.localNear, s.localFar);
  if (boxSegment.x >= boxSegment.y) {
    return s;
  }
  let segment = boxSegment.y - boxSegment.x;
  let texels = vdbSrcTexels(v);
  s.maxTexels = max(texels.x, max(texels.y, texels.z));
  let worldScale = vdbFrameVolumes.v[vi].density.z;
  let mipmaps = vdbFrameVolumes.v[vi].march.y != 0u;
  let widthLod = vdbFrameFroxelLod(f, 0.0, worldScale, s.maxTexels);
  let sampledLod = select(0.0, min(widthLod, vdbFrameVolumes.v[vi].emission.w), mipmaps);
  let steps = vdbFrameBoxSteps(s.localNear, s.localFar, segment, sampledLod, s.maxTexels);
  let subStepCount = f32(steps);
  let subStepLength = length(f.viewFar - f.viewNear) * segment / subStepCount;
  s.lod = max(widthLod, vdbFrameLod(subStepLength / worldScale, s.maxTexels));
  s.readLod = select(0.0, s.lod, mipmaps);
  let opacity = vdbFrameVolumes.v[vi].scatterOpacity.w;

  // Pre-integrate the slice along the view ray; light the visibility-weighted mean position.
  var transmittance = 1.0;
  var visibleWeight = 0.0;
  var visibleTSum = 0.0;
  var extinctionSum = 0.0;
  let jitter = vdbFrameNoise(vdbFrame.jitterFade.x, f.fragCoord);
  for (var i = 0u; i < steps; i++) {
    let t = boxSegment.x + segment * ((f32(i) + jitter) / subStepCount);
    let extinction = vdbFrameExtinction(vi, mix(s.localNear, s.localFar, vec3f(t)), s.readLod) * opacity * vdbFrameDistanceFade(mix(f.viewNear, f.viewFar, vec3f(t)));
    let stepTransmittance = exp(-extinction * subStepLength);
    let weight = transmittance * (1.0 - stepTransmittance);
    visibleWeight += weight;
    visibleTSum += weight * t;
    extinctionSum += extinction;
    transmittance *= stepTransmittance;
  }
  s.alpha = 1.0 - transmittance;
  s.extinction = extinctionSum / subStepCount * segment;
  s.visibleT = select(-1.0, visibleTSum / visibleWeight, visibleWeight > 0.0);
  return s;
}
`;

/** The kernel's WGSL after the source module and `vdbFrameUniformsWgsl()` (group 1 + functions + entry). */
export function vdbFrameGatherWgsl(o: VDBFrameGatherOptions): string {
  const b = GATHER_BINDINGS;
  const bindings = [
    `@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;`,
    `@group(1) @binding(${b.group}) var<uniform> vdbFrameGroup: VdbFrameGroup;`,
    `@group(1) @binding(${b.scatter}) var vdbFrameScatter: texture_storage_3d<rgba16float, write>;`,
    ...(o.multi ? [`@group(1) @binding(${b.partials}) var vdbFramePartials: texture_storage_3d<r32uint, read_write>;`] : []),
  ].join("\n");

  const accumulate =
    o.mix === "over"
      ? `      // his premultiplied "over" blend, in record order
      sum = vec4f(r.rgb * r.w + sum.rgb * (1.0 - r.w), r.w + sum.w * (1.0 - r.w));`
      : `      // co-located media: sum optical depths and depth-weighted radiance
      let tau = -log(max(1.0 - r.w, VDB_FRAME_MIN_TRANSMITTANCE));
      sum += vec4f(r.rgb * tau, tau);
      single = vec4f(r.rgb * r.w, r.w);`;

  const finish =
    o.mix === "over"
      ? `fn vdbFrameFinish(sum: vec4f, count: u32, single: vec4f) -> vec4f {
  return sum;
}`
      : `fn vdbFrameFinish(sum: vec4f, count: u32, single: vec4f) -> vec4f {
  if (count == 1u) {
    return single;
  }
  if (sum.w <= 0.0) {
    return vec4f(0.0);
  }
  let alpha = 1.0 - exp(-sum.w);
  return vec4f(sum.rgb / sum.w * alpha, alpha);
}`;

  // Partials (multi-group): (sum.xy, sum.zw) as pack2x16float pairs at slice z and z + D.
  const combine =
    o.mix === "over"
      ? `fn vdbFrameCombine(earlier: vec4f, later: vec4f) -> vec4f {
  return vec4f(later.rgb + earlier.rgb * (1.0 - later.w), later.w + earlier.w * (1.0 - later.w));
}`
      : `fn vdbFrameCombine(earlier: vec4f, later: vec4f) -> vec4f {
  return earlier + later;
}`;

  const store = o.multi
    ? `  let role = vdbFrameGroup.range.z;
  let isFirst = (role & 1u) != 0u;
  let isLast = (role & 2u) != 0u;
  let depth = u32(vdbFrame.grid.z);
  if (!inside) {
    if (isLast) {
      textureStore(vdbFrameScatter, cell, vec4f(0.0));
    }
    return;
  }
  var total = sum;
  if (!isFirst) {
    let lo = textureLoad(vdbFramePartials, cell).x;
    let hi = textureLoad(vdbFramePartials, vec3u(cell.xy, cell.z + depth)).x;
    total = vdbFrameCombine(vec4f(unpack2x16float(lo), unpack2x16float(hi)), sum);
  }
  if (isLast) {
    textureStore(vdbFrameScatter, cell, vdbFrameFinish(total, 2u, single));
  } else {
    textureStore(vdbFramePartials, cell, vec4u(pack2x16float(total.xy), 0u, 0u, 0u));
    textureStore(vdbFramePartials, vec3u(cell.xy, cell.z + depth), vec4u(pack2x16float(total.zw), 0u, 0u, 0u));
  }`
    : `  textureStore(vdbFrameScatter, cell, vdbFrameFinish(sum, count, single));`;

  return `${frameChunksWgsl(["constants", "mat3", "froxel", "noise", "fade", "box", "lod", "phase", "shadow", "sh", "envSh"])}
${bindings}

const VDB_FRAME_MIN_TRANSMITTANCE: f32 = 1.2664165549e-14; // exp(-32): tau is at most 32

${vdbFrameSourceFunctionsWgsl(o.channels)}
${VDB_FRAME_SLICE_WGSL}
// The inject pass's shadow march inside one volume: (direct, total, reduced depth, effective depth).
// The one march of the kernel (one density call site), stopping when the depth passes 8. Samples sit at
// t_k = step (k + noise); with the volume's hdda flag (K6.1d) runs of samples the source proves empty are jumped
// over (vdbSrcSkipSpan; the dense source never skips). No skipping under wrap or an offset (samples leave the
// ray's cells). The source is asked only at the start and after a zero sample past the last cell end (K3.2).
fn vdbFrameShadowMarch(vi: u32, origin: vec3f, localDirection: vec3f, maxLocalDistance: f32, lod: f32, maxTexels: f32, albedo: vec3f, noise: f32) -> vec4f {
  let steps = vdbFrameVolumes.v[vi].march.x;
  let absorbance = vdbFrameVolumes.v[vi].density.w;
  if (absorbance <= 0.0 || steps == 0u) {
    return vec4f(1.0, 1.0, 0.0, 0.0);
  }
  let direction = normalize(localDirection);
  let marchDistance = min(vdbFrameBoxExit(origin, direction), maxLocalDistance);
  if (marchDistance <= 0.0) {
    return vec4f(1.0, 1.0, 0.0, 0.0);
  }
  let multipleScattering = vdbFrameVolumes.v[vi].phaseSpectrumMs.w;
  let mipmaps = vdbFrameVolumes.v[vi].march.y != 0u;
  let localStep = marchDistance / f32(steps);
  let stepLod = select(0.0, max(lod, vdbFrameLod(localStep, maxTexels)), mipmaps);
  let worldStep = localStep * length(vdbFrameMat3(vdbFrameVolumes.v[vi].localToView) * direction);
  let stepScale = worldStep * absorbance;
  let offsetWrap = vdbFrameVolumes.v[vi].offsetWrap;
  let skip = vdbFrameVolumes.v[vi].march.z != 0u && offsetWrap.w < 0.5 && all(offsetWrap.xyz == vec3f(0.0));
  let cutoff = max(vdbFrameVolumes.v[vi].density.y, 0.0);
  let tEnd = marchDistance + localStep;
  var depths = vec3f(0.0);
  var cellEnd = -1.0;
  var lastZero = true;
  var k = 0u;
  loop {
    if (k >= steps) {
      break;
    }
    let t = localStep * (f32(k) + noise);
    if (skip && lastZero && t >= cellEnd) {
      let span = vdbSrcSkipSpan(vdbFrameVolumes.v[vi].slices.z, origin + 0.5, direction, t, tEnd, stepLod, cutoff);
      cellEnd = span.y;
      if (span.x > t) {
        k = max(k + 1u, u32(max(ceil(span.x / localStep - noise), 0.0)));
        continue;
      }
    }
    let extinction = vdbFrameExtinction(vi, origin + direction * t, stepLod) * stepScale;
    lastZero = extinction <= 0.0;
    depths += vdbFrameShadowStep(extinction, vdbFrameShadowAbsorption(extinction, albedo), multipleScattering);
    if (vdbFrameShadowDepth(depths, multipleScattering) > 8.0) {
      break;
    }
    k++;
  }
  let shadow = vdbFrameShadowTransmittance(depths, multipleScattering);
  return vec4f(shadow.x, shadow.y, depths.z, depths.y);
}

// One volume in one froxel (his inject pass): (unpremultiplied radiance, slice alpha); alpha 0: nothing.
fn vdbFrameVolumeFroxel(vi: u32, cell: vec3u, f: VdbFrameFroxel) -> vec4f {
  let s = vdbFrameSliceOf(vi, cell, f);
  let alpha = s.alpha;
  if (s.visibleT < 0.0 || alpha <= 1e-5) {
    return vec4f(0.0);
  }
  let slices = vdbFrameVolumes.v[vi].slices;
  let lod = s.lod;
  let readLod = s.readLod;
  let maxTexels = s.maxTexels;
  let localPosition = mix(s.localNear, s.localFar, vec3f(s.visibleT));
  let localToView = vdbFrameVolumes.v[vi].localToView;
  let worldToLocal = vdbFrameVolumes.v[vi].worldToLocal;
  let viewPosition = (localToView * vec4f(localPosition, 1.0)).xyz;
  let worldPosition = (vdbFrame.cameraWorld * vec4f(viewPosition, 1.0)).xyz;
  let worldToLocalDirection = vdbFrameMat3(worldToLocal);
  let rayDirection = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (f.viewFar - f.viewNear));
  let shadowNoise = vdbFrameNoise(vdbFrame.jitterFade.x, f.fragCoord + vec2f(17.0, 31.0));
  let albedo = vdbFrameAlbedo(vi, localPosition, readLod);
  let phase = vdbFrameVolumes.v[vi].phase;
  let phaseSpectrumMs = vdbFrameVolumes.v[vi].phaseSpectrumMs;
  let diffusion = phaseSpectrumMs.w > 0.0;
  let lightMask = slices.w;

  var outwardNormal = vec3f(0.0, 1.0, 0.0);
  if (length(localPosition) > 1e-4) {
    outwardNormal = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (vdbFrameMat3(localToView) * localPosition));
  }

  var irradiance = vec3f(0.0);
  if ((lightMask & VDB_FRAME_LIGHT_AMBIENT) != 0u) {
    irradiance += vdbFrameLights.ambient.xyz;
    irradiance += vdbFrameShIrradiance(outwardNormal);
  }

  // One loop over the hemisphere lights (a sky and ground march pair each), the scene lights (one march each) and
  // the environment (K6.2b: its lights, one march each and the opposite one with diffusion; without lights the
  // 'outward' SH irradiance along the outward normal): the kernel's only shadow-march call site.
  let hemiCount = select(0u, u32(vdbFrameLights.ambient.w), (lightMask & VDB_FRAME_LIGHT_HEMISPHERE) != 0u);
  let sceneCount = vdbFrameLights.counts.x;
  let envOn = (lightMask & VDB_FRAME_LIGHT_ENVIRONMENT) != 0u && vdbFrameEnv.info.z != 0u;
  let envLights = select(0u, vdbFrameEnv.info.x, envOn);
  let items = hemiCount + sceneCount + select(0u, max(envLights, 1u), envOn);
  var environment = vec3f(0.0);
  for (var item = 0u; item < items; item++) {
    var direction = vec3f(0.0, 1.0, 0.0);
    var maxDistance = 1e6;
    var marches = 2u;
    var lightColor = vec3f(0.0);
    var lightDirection = vec3f(0.0, 1.0, 0.0);
    var spotAttenuation = 1.0;
    let li = item - hemiCount;
    if (item < hemiCount) {
      direction = worldToLocalDirection * vdbFrameLights.hemiUp[item].xyz;
    } else if (li >= sceneCount) {
      direction = worldToLocalDirection * select(outwardNormal, vdbFrameEnv.direction[min(li - sceneCount, 11u)].xyz, envLights > 0u);
      marches = select(1u, 2u, diffusion);
    } else {
      let position = vdbFrameLights.position[li];
      let color = vdbFrameLights.color[li];
      let spot = vdbFrameLights.spot[li];
      let isDirectional = position.w < 1.5;
      let isSpot = position.w > 2.5;
      let bit = select(select(VDB_FRAME_LIGHT_POINT, VDB_FRAME_LIGHT_SPOT, isSpot), VDB_FRAME_LIGHT_DIRECTIONAL, isDirectional);
      if (isSpot) {
        spotAttenuation = smoothstep(spot.w, color.w, dot(normalize(position.xyz - worldPosition), spot.xyz));
      }
      if ((lightMask & bit) == 0u || spotAttenuation <= 0.0) {
        continue;
      }
      let toLight = (worldToLocal * vec4f(position.xyz, 1.0)).xyz - localPosition;
      direction = select(toLight, worldToLocalDirection * position.xyz, isDirectional);
      maxDistance = select(length(toLight), 1e6, isDirectional);
      lightDirection = select(normalize(position.xyz - worldPosition), position.xyz, isDirectional);
      lightColor = color.rgb;
      marches = 1u;
    }
    var front = vec4f(1.0, 1.0, 0.0, 0.0);
    var back = vec4f(1.0, 1.0, 0.0, 0.0);
    for (var k = 0u; k < marches; k++) {
      let shadow = vdbFrameShadowMarch(vi, localPosition, select(direction, -direction, k == 1u), select(maxDistance, 1e6, k == 1u), lod, maxTexels, albedo, shadowNoise);
      if (k == 0u) {
        front = shadow;
      } else {
        back = shadow;
      }
    }
    if (item < hemiCount) {
      // Diffusion: sky and ground are an opposite pair, two-sided totals
      var sky = front.y;
      var ground = back.y;
      if (diffusion) {
        sky = vdbFrameIsotropicDiffusion(front.z, back.z, front.w);
        ground = vdbFrameIsotropicDiffusion(back.z, front.z, back.w);
      }
      irradiance += vdbFrameLights.hemiSky[item].xyz * 0.5 * sky;
      irradiance += vdbFrameLights.hemiGround[item].xyz * 0.5 * ground;
    } else if (li < sceneCount) {
      let light = vdbFramePhaseColor(phase, phaseSpectrumMs.xyz, dot(rayDirection, lightDirection)) * front.x + vec3f(front.y - front.x);
      irradiance += lightColor * light * spotAttenuation;
    } else if (envLights > 0u) {
      // an environment light: isotropic, no phase (diffusion: the two-sided total)
      environment += vdbFrameEnv.color[min(li - sceneCount, 11u)].xyz * select(front.y, vdbFrameIsotropicDiffusion(front.z, back.z, front.w), diffusion);
    } else if (diffusion) {
      // 'outward': the outward and the inward irradiance, each with its two-sided total (uniform light gives E)
      irradiance += vdbFrameEnvIrradiance(outwardNormal) * (vdbFrameIsotropicDiffusion(front.z, back.z, front.w) * 0.5);
      irradiance += vdbFrameEnvIrradiance(-outwardNormal) * (vdbFrameIsotropicDiffusion(back.z, front.z, back.w) * 0.5);
    } else {
      irradiance += vdbFrameEnvIrradiance(outwardNormal) * front.y;
    }
  }
  // SH lights can be negative: the environment sum is clamped (his max(environment, 0))
  irradiance += max(environment, vec3f(0.0));

  let scatter = vdbFrameVolumes.v[vi].scatterOpacity.xyz;
  let radiance = max(scatter, albedo * irradiance * (1.0 / VDB_FRAME_PI)) + vdbFrameEmission(vi, localPosition, readLod);
  return vec4f(radiance, alpha);
}

${finish}
${combine}

@compute @workgroup_size(8, 8, 1)
fn vdbFrameGatherCs(@builtin(global_invocation_id) gid: vec3u) {
  let box = vdbFrame.dispatchBox;
  if (gid.x >= box.z || gid.y >= box.w || f32(gid.z) >= vdbFrame.grid.z) {
    return;
  }
  let cell = vec3u(box.x + gid.x, box.y + gid.y, gid.z);
  let area = vdbFrame.areaRect;
  let areaSlices = vdbFrame.areaSlices;
  let p = vec2f(cell.xy);
  let z = f32(cell.z);
  let inside = p.x >= area.x && p.y >= area.y && p.x < area.z && p.y < area.w && z >= areaSlices.x && z < areaSlices.y;
  var sum = vec4f(0.0);
  var single = vec4f(0.0);
  var count = 0u;
  if (inside) {
    let f = vdbFrameFroxelAt(cell);
    let group = vdbFrameGroup.range;
    for (var i = 0u; i < group.y; i++) {
      let r = vdbFrameVolumeFroxel(group.x + i, cell, f);
      if (r.w <= 0.0) {
        continue;
      }
${accumulate}
      count += 1u;
    }
  }
${store}
}
`;
}
