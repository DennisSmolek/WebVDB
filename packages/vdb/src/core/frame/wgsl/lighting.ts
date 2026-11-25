// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameShadowDensityMaterial.js L12–131 (the shadow density pass),
//   FrameTransmittanceMaterial.js L43–276 (light slots, gridExtinction, march, skyTotal, the near-extinction skip,
//   the slot packing), FrameDensityMaterial.js L172–357 (the grey density pass), FrameLightingMaterial.js L277–437
//   (the grey lighting pass); frameShaderChunks.js L1061–1194 (createGridNodes: sampleShadows).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: compute kernels on stacked 3D textures (K6.2, design §3.3, §3.5): the shadow grid and the transmittance
// are sub-volumes of one 3D texture each, read with hardware trilinear (his 2D slice atlases with a manual z lerp);
// the density and lighting passes are one kernel (the density sums live in registers while a thread loops its
// source group; his 4 rgba16float density targets are gone); several source groups accumulate in packed partials.
// The environment terms ('outward', environment grid, environment light slot), octaves, two-sided, neural and
// spectral wait for K6.2b / K6.3 / K6.4. CPU twins: tools/frame (grid.ts, stack.ts). Divergences: DIVERGES.md.

/**
 * The 'frame' lighting kernels of the core renderer (K6.2):
 *
 * - `vdbFrameGridDensityCs` (per caster source group, 4×4×4): the extinction and absorption of the group's
 *   casters at each shadow grid voxel of every cascade (his shadow density pass, additive).
 * - `vdbFrameTransmittanceCs` (source-free, 4×4×4): from each voxel the march through the grid toward the light
 *   slots and the sky up and down; four sub-volumes per cascade (his 4 MRT targets).
 * - `vdbFrameGatherCs` 'frame' (per source group, 8×8×1): his density pass per volume (the slice pre-integration,
 *   the visible position, the shadow slots there) summed extinction-weighted in registers, then (last group) his
 *   lighting pass on the sums: premultiplied (radiance·α, α) into the scatter texture.
 */

import type { VDBVolumeChannels } from "../../source/types.js";
import { frameChunksWgsl } from "./chunks.js";
import { VDB_FRAME_SLICE_WGSL, vdbFrameSourceFunctionsWgsl } from "./gather.js";

/** Group-1 bindings of the shadow grid density kernel. */
export const GRID_DENSITY_BINDINGS = { volumes: 0, group: 1, grid: 2, partials: 3 } as const;
/** Group-1 bindings of the transmittance kernel. */
export const TRANSMITTANCE_BINDINGS = { grid: 0, transmittance: 1 } as const;
/** Group-1 bindings of the 'frame' gather kernel. */
export const FRAME_GATHER_BINDINGS = { volumes: 0, group: 1, scatter: 2, partials: 3, transmittance: 4 } as const;

/** Values per froxel the 'frame' gather sums (4 × vec4f), packed as 8 `pack2x16float` words in the partials. */
export const VDB_FRAME_SUM_WORDS = 8;

const GRID_CELL = `  let r = u32(vdbFrame.gridLayout.y);
  if (gid.x >= r || gid.y >= r || gid.z >= r * u32(vdbFrame.gridLayout.x)) {
    return;
  }
  let cascade = gid.z / r;
  let cell = vec3u(gid.xy, gid.z - cascade * r);
  let uvw = (vec3f(cell) + 0.5) / f32(r);
  let world = vdbFrame.gridMin[cascade].xyz + uvw * vdbFrame.gridSize[cascade].xyz;`;

/** The shadow grid density kernel (after the source module, the source WGSL and `vdbFrameUniformsWgsl()`). */
export function vdbFrameGridDensityWgsl(o: { multi: boolean; channels: VDBVolumeChannels }): string {
  const b = GRID_DENSITY_BINDINGS;
  const store = o.multi
    ? `  let role = vdbFrameGroup.casters.z;
  if ((role & 1u) == 0u) {
    sum += unpack2x16float(textureLoad(vdbFrameGridPartials, texel).x);
  }
  if ((role & 2u) != 0u) {
    textureStore(vdbFrameShadowGrid, texel, vec4f(sum, 0.0, 0.0));
  } else {
    textureStore(vdbFrameGridPartials, texel, vec4u(pack2x16float(sum), 0u, 0u, 0u));
  }`
    : `  textureStore(vdbFrameShadowGrid, texel, vec4f(sum, 0.0, 0.0));`;
  return `${frameChunksWgsl(["constants", "mat3", "froxel", "lod", "shadow"])}
@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;
@group(1) @binding(${b.group}) var<uniform> vdbFrameGroup: VdbFrameGroup;
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_storage_3d<rgba16float, write>;
${o.multi ? `@group(1) @binding(${b.partials}) var vdbFrameGridPartials: texture_storage_3d<r32uint, read_write>;\n` : ""}
${vdbFrameSourceFunctionsWgsl(o.channels)}
// His shadow density pass at one voxel: the summed (extinction, absorption) of the group's casters whose box holds
// the voxel centre, read at the voxel's lod (absorbance scales the shadow, not the look).
@compute @workgroup_size(4, 4, 4)
fn vdbFrameGridDensityCs(@builtin(global_invocation_id) gid: vec3u) {
${GRID_CELL}
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  let voxelSize = max(voxel.x, max(voxel.y, voxel.z));
  var sum = vec2f(0.0);
  let casters = vdbFrameGroup.casters;
  for (var i = 0u; i < casters.y; i++) {
    let vi = casters.x + i;
    let local = (vdbFrameVolumes.v[vi].worldToLocal * vec4f(world, 1.0)).xyz;
    let a = abs(local);
    if (max(a.x, max(a.y, a.z)) > 0.5) {
      continue;
    }
    let texels = vdbSrcTexels(vdbFrameVolumes.v[vi].slices.z);
    let maxTexels = max(texels.x, max(texels.y, texels.z));
    let lod = vdbFrameLod(voxelSize / vdbFrameVolumes.v[vi].density.z, maxTexels);
    let readLod = select(0.0, lod, vdbFrameVolumes.v[vi].march.y != 0u);
    let extinction = vdbFrameExtinction(vi, local, readLod) * vdbFrameVolumes.v[vi].density.w;
    if (extinction <= 0.0) {
      continue;
    }
    sum += vec2f(extinction, vdbFrameShadowAbsorption(extinction, vdbFrameAlbedo(vi, local, readLod)));
  }
  let texel = vec3u(vdbFrame.shadowOrigin[cascade].xyz) + cell;
${store}
}
`;
}

/** The transmittance kernel (after `vdbFrameUniformsWgsl()`; source-free). */
export function vdbFrameTransmittanceWgsl(): string {
  const b = TRANSMITTANCE_BINDINGS;
  return `${frameChunksWgsl(["constants", "noise", "shadow", "grid"])}
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_3d<f32>;
@group(1) @binding(${b.transmittance}) var vdbFrameTransmittanceOut: texture_storage_3d<rgba16float, write>;

// gridExtinction: (extinction, absorption) of the finest cascade at the position, 0 outside the grid
fn vdbFrameGridExtinction(world: vec3f) -> vec2f {
  let c = vdbFrameFindCascade(world, 0u);
  if (c < 0) {
    return vec2f(0.0);
  }
  let cascade = u32(c);
  return vdbFrameStackSample(vdbFrameShadowGrid, vdbFrame.shadowOrigin[cascade].xyz, vdbFrameGridUvw(cascade, world)).xy;
}

// His march: one voxel of the finest cascade at the probe per step on the direction's fastest axis (coarser
// cascades after leaving a fine one), each step sampled jitter of the way along it; stops outside the grid, at
// maxDistance or past depth 6. (direct, total, reduced depth, effective depth).
fn vdbFrameGridMarch(origin: vec3f, direction: vec3f, maxDistance: f32, jitter: f32) -> vec4f {
  let ms = vdbFrame.lighting.x;
  var depths = vec3f(0.0);
  var traveled = 0.0;
  var probe = origin;
  for (var i = 0u; i < 1024u; i++) {
    let c = vdbFrameFindCascade(probe, 0u);
    if (c < 0 || traveled >= maxDistance || vdbFrameShadowDepth(depths, ms) > 6.0) {
      break;
    }
    let perUnit = abs(direction) / vdbFrame.gridVoxel[u32(c)].xyz;
    let stepLength = min(1.0 / max(perUnit.x, max(perUnit.y, perUnit.z)), maxDistance - traveled);
    let s = vdbFrameGridExtinction(probe + direction * stepLength * jitter);
    depths += vdbFrameShadowStep(s.x, s.y, ms) * stepLength;
    probe += direction * stepLength;
    traveled += stepLength;
  }
  let t = vdbFrameShadowTransmittance(depths, ms);
  return vec4f(t.x, t.y, depths.z, depths.y);
}

// His transmittance pass at one voxel: 4 targets (sky up total, sky down total, slot 0, 1 total), (slots 2–5
// total), (slots 0–3 direct), (slots 4–5 direct, 1, 1); 1 where no extinction is within half a voxel.
@compute @workgroup_size(4, 4, 4)
fn vdbFrameTransmittanceCs(@builtin(global_invocation_id) gid: vec3u) {
${GRID_CELL}
  var t0 = vec4f(1.0);
  var t1 = vec4f(1.0);
  var t2 = vec4f(1.0);
  var t3 = vec4f(1.0);
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  var near = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let corner = vec3f(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u)) - 0.5;
    near += vdbFrameGridExtinction(world + corner * voxel).x;
  }
  if (near > 0.0) {
    let ms = vdbFrame.lighting.x;
    let jitter = vdbFrameNoise(vdbFrame.jitterFade.x, vdbFrameGridFragCoord(cascade, cell));
    let sky = normalize(vdbFrameLights.sky.xyz);
    let slots = u32(vdbFrame.gridLayout.w);
    // marches 0..5: the light slots; 6, 7: sky up and down (one march call site)
    var results: array<vec4f, 8>;
    for (var m = 0u; m < 8u; m++) {
      var direction = sky;
      var maxDistance = 1e20;
      results[m] = vec4f(1.0, 1.0, 0.0, 0.0);
      if (m < 6u) {
        if (m >= slots) {
          continue;
        }
        let light = vdbFrameLights.position[m];
        let toLight = light.xyz - world;
        let isDirection = light.w < 1.5;
        direction = select(normalize(toLight), normalize(light.xyz), isDirection);
        maxDistance = select(length(toLight), 1e20, isDirection);
      } else if (m == 7u) {
        direction = -sky;
      }
      results[m] = vdbFrameGridMarch(world, direction, maxDistance, jitter);
    }
    let up = results[6];
    let down = results[7];
    var skyUp = up.y;
    var skyDown = down.y;
    if (ms > 0.0) {
      skyUp = vdbFrameIsotropicDiffusion(up.z, down.z, up.w);
      skyDown = vdbFrameIsotropicDiffusion(down.z, up.z, down.w);
    }
    t0 = vec4f(skyUp, skyDown, results[0].y, results[1].y);
    t1 = vec4f(results[2].y, results[3].y, results[4].y, results[5].y);
    t2 = vec4f(results[0].x, results[1].x, results[2].x, results[3].x);
    t3 = vec4f(results[4].x, results[5].x, 1.0, 1.0);
  }
  let base = cascade * 4u;
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base].xyz) + cell, t0);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 1u].xyz) + cell, t1);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 2u].xyz) + cell, t2);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 3u].xyz) + cell, t3);
}
`;
}

/** The 'frame' gather kernel (after the source module, the source WGSL and `vdbFrameUniformsWgsl()`). */
export function vdbFrameLightingGatherWgsl(o: { multi: boolean; channels: VDBVolumeChannels }): string {
  const b = FRAME_GATHER_BINDINGS;
  const store = o.multi
    ? `  let role = vdbFrameGroup.range.z;
  let isFirst = (role & 1u) != 0u;
  let isLast = (role & 2u) != 0u;
  if (!inside) {
    if (isLast) {
      textureStore(vdbFrameScatter, cell, vec4f(0.0));
    }
    return;
  }
  let depth = u32(vdbFrame.grid.z);
  var words: array<u32, ${VDB_FRAME_SUM_WORDS}>;
  if (!isFirst) {
    for (var k = 0u; k < ${VDB_FRAME_SUM_WORDS}u; k++) {
      words[k] = textureLoad(vdbFramePartials, vec3u(cell.xy, cell.z + k * depth)).x;
    }
    sums.a += vec4f(unpack2x16float(words[0]), unpack2x16float(words[1]));
    sums.e += vec4f(unpack2x16float(words[2]), unpack2x16float(words[3]));
    sums.s0 += vec4f(unpack2x16float(words[4]), unpack2x16float(words[5]));
    sums.s1 += vec4f(unpack2x16float(words[6]), unpack2x16float(words[7]));
  }
  if (isLast) {
    textureStore(vdbFrameScatter, cell, vdbFrameLightFroxel(sums, f));
  } else {
    words = array<u32, ${VDB_FRAME_SUM_WORDS}>(
      pack2x16float(sums.a.xy), pack2x16float(sums.a.zw), pack2x16float(sums.e.xy), pack2x16float(sums.e.zw),
      pack2x16float(sums.s0.xy), pack2x16float(sums.s0.zw), pack2x16float(sums.s1.xy), pack2x16float(sums.s1.zw));
    for (var k = 0u; k < ${VDB_FRAME_SUM_WORDS}u; k++) {
      textureStore(vdbFramePartials, vec3u(cell.xy, cell.z + k * depth), vec4u(words[k], 0u, 0u, 0u));
    }
  }`
    : `  var out = vec4f(0.0);
  if (inside) {
    out = vdbFrameLightFroxel(sums, f);
  }
  textureStore(vdbFrameScatter, cell, out);`;

  return `${frameChunksWgsl(["constants", "mat3", "froxel", "noise", "fade", "box", "lod", "phase", "shadow", "sh", "grid"])}
@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;
@group(1) @binding(${b.group}) var<uniform> vdbFrameGroup: VdbFrameGroup;
@group(1) @binding(${b.scatter}) var vdbFrameScatter: texture_storage_3d<rgba16float, write>;
${o.multi ? `@group(1) @binding(${b.partials}) var vdbFramePartials: texture_storage_3d<r32uint, read_write>;\n` : ""}@group(1) @binding(${b.transmittance}) var vdbFrameTransmittanceStack: texture_3d<f32>;

${vdbFrameSourceFunctionsWgsl(o.channels)}
${VDB_FRAME_SLICE_WGSL}
// His density targets as sums: (albedo·σ, σ), (emission·σ, visible t·σ), (sky up, sky down, slot 0, 1)·σ, (slots 2–5)·σ
struct VdbFrameSums {
  a: vec4f,
  e: vec4f,
  s0: vec4f,
  s1: vec4f,
};

// sampleShadows: the 4 transmittance targets at a world position, 1 outside the grid (or without shadows), blended
// toward the next cascade within 10% of a cascade's sides
fn vdbFrameShadowsAt(world: vec3f) -> array<vec4f, 4> {
  var out = array<vec4f, 4>(vec4f(1.0), vec4f(1.0), vec4f(1.0), vec4f(1.0));
  if (vdbFrame.lighting.z < 0.5) {
    return out;
  }
  let found = vdbFrameFindCascade(world, 0u);
  if (found < 0) {
    return out;
  }
  let cascade = u32(found);
  let uvw = vdbFrameGridUvw(cascade, world);
  for (var k = 0u; k < 4u; k++) {
    out[k] = vdbFrameStackSample(vdbFrameTransmittanceStack, vdbFrame.transmittanceOrigin[cascade * 4u + k].xyz, uvw);
  }
  let side = min(uvw, vec3f(1.0) - uvw);
  let blend = 1.0 - smoothstep(0.0, 0.1, min(side.x, min(side.y, side.z)));
  if (blend > 0.0) {
    let next = vdbFrameFindCascade(world, cascade + 1u);
    if (next >= 0) {
      let n = u32(next);
      let nextUvw = vdbFrameGridUvw(n, world);
      for (var k = 0u; k < 4u; k++) {
        out[k] = mix(out[k], vdbFrameStackSample(vdbFrameTransmittanceStack, vdbFrame.transmittanceOrigin[n * 4u + k].xyz, nextUvw), vec4f(blend));
      }
    }
  }
  return out;
}

// His density pass for one volume in one froxel: the slice's mean extinction σ (over the whole slice), the
// visibility-weighted position, albedo and emission there, and the shadow slots at that world position (scene
// light slots: green phase × direct + scattered); σ-weighted. Zero when σ ≤ 0.
fn vdbFrameDensitySums(vi: u32, cell: vec3u, f: VdbFrameFroxel) -> VdbFrameSums {
  var out: VdbFrameSums;
  let s = vdbFrameSliceOf(vi, cell, f);
  let extinction = s.extinction;
  if (!(extinction > 0.0)) {
    return out;
  }
  let visibleT = select(0.5, s.visibleT, s.visibleT >= 0.0);
  let local = mix(s.localNear, s.localFar, vec3f(visibleT));
  let albedo = vdbFrameAlbedo(vi, local, s.readLod);
  let emission = vdbFrameEmission(vi, local, s.readLod) + vdbFrameVolumes.v[vi].scatterOpacity.xyz;
  let world = (vdbFrame.cameraWorld * vec4f(mix(f.viewNear, f.viewFar, vec3f(visibleT)), 1.0)).xyz;
  let shadows = vdbFrameShadowsAt(world);
  let ray = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (f.viewFar - f.viewNear));
  var slot = array<f32, 6>(shadows[0].z, shadows[0].w, shadows[1].x, shadows[1].y, shadows[1].z, shadows[1].w);
  var direct = array<f32, 6>(shadows[2].x, shadows[2].y, shadows[2].z, shadows[2].w, shadows[3].x, shadows[3].y);
  let sceneSlots = min(u32(vdbFrame.gridLayout.w), 6u);
  for (var i = 0u; i < sceneSlots; i++) {
    let light = vdbFrameLights.position[i];
    let direction = select(normalize(light.xyz - world), light.xyz, light.w < 1.5);
    let phase = vdbFramePhaseColor(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, dot(ray, direction));
    slot[i] = phase.y * direct[i] + (slot[i] - direct[i]);
  }
  out.a = vec4f(albedo * extinction, extinction);
  out.e = vec4f(emission * extinction, visibleT * extinction);
  out.s0 = min(vec4f(shadows[0].x, shadows[0].y, slot[0], slot[1]) * extinction, vec4f(6.0e4));
  out.s1 = min(vec4f(slot[2], slot[3], slot[4], slot[5]) * extinction, vec4f(6.0e4));
  return out;
}

// three's getDistanceAttenuation
fn vdbFrameDistanceAttenuation(distance: f32, cutoff: f32, decay: f32) -> f32 {
  let falloff = 1.0 / max(pow(distance, decay), 0.01);
  if (cutoff > 0.0) {
    let x = clamp(1.0 - pow(distance / cutoff, 4.0), 0.0, 1.0);
    return falloff * x * x;
  }
  return falloff;
}

// His lighting pass on the summed density of all volumes in the froxel: premultiplied (radiance·α, α)
fn vdbFrameLightFroxel(sums: VdbFrameSums, f: VdbFrameFroxel) -> vec4f {
  let extinction = sums.a.w;
  if (!(extinction > 0.0)) {
    return vec4f(0.0);
  }
  let visibleT = clamp(sums.e.w / extinction, 0.0, 1.0);
  let world = (vdbFrame.cameraWorld * vec4f(mix(f.viewNear, f.viewFar, vec3f(visibleT)), 1.0)).xyz;
  let ray = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (f.viewFar - f.viewNear));
  let s0 = sums.s0 / extinction;
  let s1 = sums.s1 / extinction;
  var slot = array<f32, 6>(s0.z, s0.w, s1.x, s1.y, s1.z, s1.w);
  // per-channel phases: the slots hold the green phase; the direct transmittance adds the red and blue difference
  let colorPhase = vdbFrame.framePhaseSpectrum.z > 0.5;
  var direct = array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  if (colorPhase) {
    let d = vdbFrameShadowsAt(world);
    direct = array<f32, 6>(d[2].x, d[2].y, d[2].z, d[2].w, d[3].x, d[3].y);
  }
  let sky = normalize(vdbFrameLights.sky.xyz);
  let skyUp = s0.x * 0.5;
  let skyDown = s0.y * 0.5;
  var irradiance = vdbFrameLights.ambient.xyz;
  irradiance += vdbFrameShIrradiance(sky) * skyUp;
  irradiance += vdbFrameShIrradiance(-sky) * skyDown;
  irradiance += vdbFrameLights.hemiSkySum.xyz * skyUp;
  irradiance += vdbFrameLights.hemiGroundSum.xyz * skyDown;
  let shadowLights = u32(vdbFrame.gridLayout.w);
  for (var i = 0u; i < vdbFrameLights.counts.x; i++) {
    let position = vdbFrameLights.position[i];
    let color = vdbFrameLights.color[i];
    let spot = vdbFrameLights.spot[i];
    let falloff = vdbFrameLights.falloff[i];
    let lightVector = position.xyz - world;
    let spotAttenuation = select(1.0, smoothstep(spot.w, color.w, dot(normalize(lightVector), spot.xyz)), position.w > 2.5);
    let distanceAttenuation = select(1.0, vdbFrameDistanceAttenuation(length(lightVector), falloff.x, falloff.y), position.w > 1.5);
    let direction = select(normalize(lightVector), position.xyz, position.w < 1.5);
    let phase = vdbFramePhaseColor(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, dot(ray, direction));
    let hasSlot = i < shadowLights && i < 6u;
    var factor = phase;
    if (hasSlot) {
      factor = vec3f(slot[i]);
      if (colorPhase) {
        factor += (phase - phase.y) * direct[i];
      }
    }
    irradiance += color.rgb * factor * spotAttenuation * distanceAttenuation;
  }
  let albedo = sums.a.rgb / extinction;
  let emission = sums.e.rgb / extinction;
  let radiance = albedo * irradiance * (1.0 / VDB_FRAME_PI) + emission;
  let alpha = 1.0 - exp(-extinction * length(f.viewFar - f.viewNear));
  return vec4f(radiance * alpha, alpha);
}

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
  var sums: VdbFrameSums;
  var f: VdbFrameFroxel;
  if (inside) {
    f = vdbFrameFroxelAt(cell);
    let group = vdbFrameGroup.range;
    for (var i = 0u; i < group.y; i++) {
      let r = vdbFrameDensitySums(group.x + i, cell, f);
      sums.a += r.a;
      sums.e += r.e;
      sums.s0 += r.s0;
      sums.s1 += r.s1;
    }
  }
${store}
}
`;
}
