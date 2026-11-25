// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameEmissionLight.js L59–60 (golden angle), L168–237 (the direction pass),
//   L259–279 (the reduce pass), L310–364 (the probe pass), L457–519 (the light node's irradiance);
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L222–256 (frameBlackbodyColor, the table read), L1003–1010
//   (the 'planck' / 'soot' branch of sampleEmission).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: compute kernels (K6.5, design §3.3, §3.8): the direction pass is one workgroup per (direction, emitter)
// with one lane per ray and a workgroup reduction (his: one fragment loops the 64 rays); the probes are a stacked 3D
// texture (3 sub-volumes per cascade) read with hardware trilinear (his: a 2D atlas with a manual z lerp). Emission
// colour is data: a 256-texel LUT row per volume over log temperature (his blackbody table pre-blended to the volume's
// soot exponent) or a validated user WGSL hook. CPU twins: tools/frame (emission.ts, blackbody.ts). Divergences:
// DIVERGES.md (K6.5 rows).

/**
 * Emission in the core renderer (K6.5):
 *
 * - `vdbFrameEmissionColorWgsl(...)`: the volume record's emission colour (white, a LUT row, or the user hook), used
 *   by `vdbFrameEmission` in every source kernel that lights or emits.
 * - `vdbFrameEmissionDirectionCs` (64 lanes = 8² rays; dispatch 16 directions × emitters of a source group): each
 *   volume's emitted light along 16 Fibonacci directions (intensity × ray area, radius; luminance-weighted centre).
 * - `vdbFrameEmissionReduceCs` (64 per workgroup; one thread per slot): the mean intensity and the emission centre.
 * - `vdbFrameEmissionProbeCs` (4×4×4 per probe voxel of every cascade): L1 irradiance of all emitters as point lights.
 * - `vdbEmissionProbeWgsl()`: `vdbEmissionIrradiance(probes, smp, grid, world, normal)` for host receivers.
 */

import type { VDBVolumeChannels } from "../../source/types.js";
import { vdbUniformStruct } from "../uniforms.js";

/** Directions per emitter (his `frameEmissionDirections`). */
export const VDB_FRAME_EMISSION_DIRECTIONS = 16;
/** Rays per side of the square that covers an emitter (his `frameEmissionRays`); one lane each. */
export const VDB_FRAME_EMISSION_RAYS = 8;
/** Most emitters per update (one direction-pass item each). */
export const VDB_FRAME_MAX_EMITTERS = 128;
/** Most probe cascades (his `frameEmissionMaxCascades`). */
export const VDB_FRAME_EMISSION_MAX_CASCADES = 4;
/** Texels of one emission colour LUT row (over log temperature, 500–40,000 K). */
export const VDB_FRAME_EMISSION_LUT_SIZE = 256;
/** The LUT's temperature range (his blackbody table's). */
export const VDB_FRAME_EMISSION_LUT_KELVIN = [500, 40000] as const;

/**
 * The emitters a direction pass reads (group 1): one segment per source group at a dynamic offset
 * ({@link VDB_FRAME_EMITTERS_STRIDE}); per item the volume record and slot, and its bounding sphere.
 */
export const VDB_FRAME_EMITTERS = vdbUniformStruct("VdbFrameEmitterList", [
  // items of this segment, steps per ray, slots, spare
  { name: "header", type: "vec4u" },
  // (volume record, slot, 0, 0)
  { name: "info", type: "vec4u", count: VDB_FRAME_MAX_EMITTERS },
  // (centre xyz, radius)
  { name: "sphere", type: "vec4f", count: VDB_FRAME_MAX_EMITTERS },
]);

/** Bytes between the emitter list's per-group segments (256-aligned dynamic offsets). */
export const VDB_FRAME_EMITTERS_STRIDE = Math.ceil(VDB_FRAME_EMITTERS.size / 256) * 256;

/** The probe grid (group 1 of the probe kernel; the receivers' `grid`). */
export const VDB_EMISSION_GRID = vdbUniformStruct("VdbEmissionGrid", [
  { name: "gridMin", type: "vec4f", count: VDB_FRAME_EMISSION_MAX_CASCADES },
  { name: "gridSize", type: "vec4f", count: VDB_FRAME_EMISSION_MAX_CASCADES },
  // texel origin of (cascade c, channel k) at [3c + k] in the probe stack (k: red, green, blue)
  { name: "origin", type: "vec4f", count: 3 * VDB_FRAME_EMISSION_MAX_CASCADES },
  // the centre of all emitters (xyz), active (1 when some emitter is visible)
  { name: "center", type: "vec4f" },
  // cascade count, probes per axis, emitters, spare
  { name: "info", type: "vec4f" },
]);

/** Group-1 bindings of the emission kernels. */
export const EMISSION_DIRECTION_BINDINGS = { volumes: 0, emitters: 2, lut: 3, directions: 4 } as const;
export const EMISSION_REDUCE_BINDINGS = { emitters: 0, directions: 1, out: 2 } as const;
export const EMISSION_PROBE_BINDINGS = { grid: 0, emitters: 1, probes: 2 } as const;

const LUT_SCALE = Math.log(VDB_FRAME_EMISSION_LUT_KELVIN[1] / VDB_FRAME_EMISSION_LUT_KELVIN[0]);

/**
 * The emission colour of the source functions: `march.w` of a volume record is its colour code (bits 0–1 the mode: 0
 * white, 1 LUT row, 2 the user hook; bits 2+ the LUT row). `lutBinding`: the kernel's group-1 binding of the LUT
 * texture (null: white only, no binding). `hook`: the validated user WGSL (`fn vdbUserEmissionColor(t: f32,
 * temperature: f32) -> vec3f`, t the normalized temperature), or null.
 */
export function vdbFrameEmissionColorWgsl(o: { lutBinding: number | null; hook: string | null }): string {
  if (o.lutBinding === null) {
    return `fn vdbFrameEmissionColor(code: u32, t: f32, temperature: f32) -> vec3f {
  return vec3f(1.0);
}
`;
  }
  return `${o.hook ? `${o.hook}\n` : ""}@group(1) @binding(${o.lutBinding}) var vdbFrameEmissionLut: texture_2d<f32>;
// frameBlackbodyColor on one LUT row: linear in log temperature between texels (textureLoad, no sampler)
fn vdbFrameEmissionLutColor(row: u32, temperature: f32) -> vec3f {
  let x = clamp(log(max(temperature, 1e-3) / ${VDB_FRAME_EMISSION_LUT_KELVIN[0].toFixed(1)}) / ${LUT_SCALE}, 0.0, 1.0) * ${(VDB_FRAME_EMISSION_LUT_SIZE - 1).toFixed(1)};
  let column = min(u32(x), ${VDB_FRAME_EMISSION_LUT_SIZE - 2}u);
  let blend = x - f32(column);
  return mix(textureLoad(vdbFrameEmissionLut, vec2u(column, row), 0).rgb, textureLoad(vdbFrameEmissionLut, vec2u(column + 1u, row), 0).rgb, blend);
}
// The colour of a volume's emission: white (mode 0), a LUT row (1), the user hook (2)
fn vdbFrameEmissionColor(code: u32, t: f32, temperature: f32) -> vec3f {
  let mode = code & 3u;
  if (mode == 1u) {
    return vdbFrameEmissionLutColor(code >> 2u, temperature);
  }
${o.hook ? `  if (mode == 2u) {
    return vdbUserEmissionColor(t, temperature);
  }
` : ""}  return vec3f(1.0);
}
`;
}

/** The direction kernel (after the source module, the source WGSL, `vdbFrameUniformsWgsl()` and the source functions). */
export function vdbFrameEmissionDirectionWgsl(): string {
  const b = EMISSION_DIRECTION_BINDINGS;
  const D = VDB_FRAME_EMISSION_DIRECTIONS;
  const R = VDB_FRAME_EMISSION_RAYS;
  return `${VDB_FRAME_EMITTERS.wgsl}@group(1) @binding(${b.emitters}) var<uniform> vdbFrameEmitters: VdbFrameEmitterList;
@group(1) @binding(${b.directions}) var vdbFrameEmissionDirectionsOut: texture_storage_2d<rgba32float, write>;

const VDB_FRAME_GOLDEN_ANGLE: f32 = ${Math.PI * (3 - Math.sqrt(5))};
var<workgroup> vdbFrameEmissionLanes: array<vec4f, ${2 * R * R}>;

// His direction pass for one (direction, emitter): lane r is ray r of the 8x8 square over the bounding sphere; each
// adds emission x alpha x the transmittance so far (absorbance scales the transmittance) and stops below 1%.
@compute @workgroup_size(${R * R}, 1, 1)
fn vdbFrameEmissionDirectionCs(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  // the list segment of this source group (a dynamic offset): its items are this dispatch's emitters
  if (wg.y >= vdbFrameEmitters.header.x || wg.x >= ${D}u) {
    return;
  }
  let item = wg.y;
  let vi = vdbFrameEmitters.info[item].x;
  let slot = vdbFrameEmitters.info[item].y;
  let sphere = vdbFrameEmitters.sphere[item];
  let index = f32(wg.x);
  let y = 1.0 - (index * 2.0 + 1.0) / ${D}.0;
  let ring = sqrt(max(0.0, 1.0 - y * y));
  let phi = index * VDB_FRAME_GOLDEN_ANGLE;
  let direction = vec3f(cos(phi) * ring, y, sin(phi) * ring);
  let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(direction.y) < 0.999);
  let tangent = normalize(cross(helper, direction));
  let bitangent = cross(direction, tangent);
  let radius = sphere.w;
  let diameter = radius * 2.0;
  let spacing = diameter / ${R}.0;
  let steps = vdbFrameEmitters.header.y;
  let stepCount = f32(steps);
  let u = (f32(lane % ${R}u) + 0.5) / ${R}.0 * 2.0 - 1.0;
  let v = (f32(lane / ${R}u) + 0.5) / ${R}.0 * 2.0 - 1.0;
  let worldStart = sphere.xyz + direction * radius + tangent * (u * radius) + bitangent * (v * radius);
  let worldEnd = worldStart - direction * diameter;
  let worldToLocal = vdbFrameVolumes.v[vi].worldToLocal;
  let localStart = (worldToLocal * vec4f(worldStart, 1.0)).xyz;
  let localEnd = (worldToLocal * vec4f(worldEnd, 1.0)).xyz;
  let segment = vdbFrameBoxSegment(localStart, localEnd);
  var intensity = vec3f(0.0);
  var center = vec4f(0.0);
  if (segment.x < segment.y) {
    let segmentLength = segment.y - segment.x;
    let stepLength = segmentLength * diameter / stepCount;
    let texels = vdbSrcTexels(vdbFrameVolumes.v[vi].slices.z);
    let maxTexels = max(texels.x, max(texels.y, texels.z));
    // Smoke reads the mip level of the step; emission level 0 (power grows with temperature^4, mipmaps lose it)
    let lod = select(0.0, vdbFrameLod(stepLength / vdbFrameVolumes.v[vi].density.z, maxTexels), vdbFrameVolumes.v[vi].march.y != 0u);
    let absorbance = vdbFrameVolumes.v[vi].density.w;
    var transmittance = 1.0;
    for (var s = 0u; s < steps; s++) {
      let t = segment.x + segmentLength * ((f32(s) + 0.5) / stepCount);
      let localPoint = mix(localStart, localEnd, vec3f(t));
      let extinction = vdbFrameExtinction(vi, localPoint, lod);
      if (extinction > 0.0) {
        let alpha = 1.0 - exp(-extinction * stepLength);
        let radiance = vdbFrameEmission(vi, localPoint, 0.0) * (transmittance * alpha);
        intensity += radiance;
        center += vec4f(mix(worldStart, worldEnd, vec3f(t)), 1.0) * dot(radiance, vec3f(0.2126, 0.7152, 0.0722));
        transmittance *= exp(-extinction * absorbance * stepLength);
        if (transmittance < 0.01) {
          break;
        }
      }
    }
  }
  vdbFrameEmissionLanes[lane] = vec4f(intensity, 0.0);
  vdbFrameEmissionLanes[lane + ${R * R}u] = center;
  workgroupBarrier();
  for (var width = ${(R * R) / 2}u; width > 0u; width = width >> 1u) {
    if (lane < width) {
      vdbFrameEmissionLanes[lane] += vdbFrameEmissionLanes[lane + width];
      vdbFrameEmissionLanes[lane + ${R * R}u] += vdbFrameEmissionLanes[lane + ${R * R}u + width];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let area = spacing * spacing;
    textureStore(vdbFrameEmissionDirectionsOut, vec2u(wg.x, slot), vec4f(vdbFrameEmissionLanes[0].xyz * area, radius));
    textureStore(vdbFrameEmissionDirectionsOut, vec2u(wg.x + ${D}u, slot), vdbFrameEmissionLanes[${R * R}u] * area);
  }
}
`;
}

/** The reduce kernel (after `vdbFrameUniformsWgsl()`; source-free). */
export function vdbFrameEmissionReduceWgsl(): string {
  const b = EMISSION_REDUCE_BINDINGS;
  const D = VDB_FRAME_EMISSION_DIRECTIONS;
  return `${VDB_FRAME_EMITTERS.wgsl}@group(1) @binding(${b.emitters}) var<uniform> vdbFrameEmitters: VdbFrameEmitterList;
@group(1) @binding(${b.directions}) var vdbFrameEmissionDirections: texture_2d<f32>;
@group(1) @binding(${b.out}) var vdbFrameEmitterOut: texture_storage_2d<rgba32float, write>;

// His reduce pass for one slot: the mean intensity of the 16 directions; the centre of the visible emission
@compute @workgroup_size(64, 1, 1)
fn vdbFrameEmissionReduceCs(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= vdbFrameEmitters.header.z) {
    return;
  }
  var intensity = vec4f(0.0);
  var center = vec4f(0.0);
  for (var i = 0u; i < ${D}u; i++) {
    intensity += textureLoad(vdbFrameEmissionDirections, vec2u(i, slot), 0);
    center += textureLoad(vdbFrameEmissionDirections, vec2u(i + ${D}u, slot), 0);
  }
  textureStore(vdbFrameEmitterOut, vec2u(0u, slot), intensity / ${D}.0);
  textureStore(vdbFrameEmitterOut, vec2u(1u, slot), select(vec4f(0.0), vec4f(center.xyz / max(center.w, 1e-12), 1.0), center.w > 0.0));
}
`;
}

/** The probe kernel (after `vdbFrameUniformsWgsl()`; source-free). */
export function vdbFrameEmissionProbeWgsl(): string {
  const b = EMISSION_PROBE_BINDINGS;
  return `${VDB_EMISSION_GRID.wgsl}@group(1) @binding(${b.grid}) var<uniform> vdbEmissionGrid: VdbEmissionGrid;
@group(1) @binding(${b.emitters}) var vdbFrameEmitterIn: texture_2d<f32>;
@group(1) @binding(${b.probes}) var vdbFrameEmissionProbesOut: texture_storage_3d<rgba16float, write>;

// His probe pass at one probe: every emitter a point light at its emission centre, softened near it by a quarter of
// its radius; L1 projection E(n) = E/4 + E/2 dot(l, n), the direction fading near the centre
@compute @workgroup_size(4, 4, 4)
fn vdbFrameEmissionProbeCs(@builtin(global_invocation_id) gid: vec3u) {
  let r = u32(vdbEmissionGrid.info.y);
  let count = u32(vdbEmissionGrid.info.x);
  if (gid.x >= r || gid.y >= r || gid.z >= r * count) {
    return;
  }
  let cascade = gid.z / r;
  let cell = vec3u(gid.xy, gid.z - cascade * r);
  let uvw = (vec3f(cell) + 0.5) / f32(r);
  let probe = vdbEmissionGrid.gridMin[cascade].xyz + uvw * vdbEmissionGrid.gridSize[cascade].xyz;
  var constant = vec3f(0.0);
  var red = vec3f(0.0);
  var green = vec3f(0.0);
  var blue = vec3f(0.0);
  let emitters = u32(vdbEmissionGrid.info.z);
  for (var i = 0u; i < emitters; i++) {
    let center = textureLoad(vdbFrameEmitterIn, vec2u(1u, i), 0);
    if (center.w > 0.0) {
      let emitter = textureLoad(vdbFrameEmitterIn, vec2u(0u, i), 0);
      let toProbe = probe - center.xyz;
      let distanceSquared = dot(toProbe, toProbe);
      let soft = emitter.w * 0.25;
      let softDistanceSquared = distanceSquared + soft * soft;
      let irradiance = emitter.rgb / softDistanceSquared;
      let toLight = toProbe * inverseSqrt(max(distanceSquared, 1e-12)) * (distanceSquared / softDistanceSquared * -0.5);
      constant += irradiance * 0.25;
      red += toLight * irradiance.r;
      green += toLight * irradiance.g;
      blue += toLight * irradiance.b;
    }
  }
  let base = cascade * 3u;
  textureStore(vdbFrameEmissionProbesOut, vec3u(vdbEmissionGrid.origin[base].xyz) + cell, vec4f(constant.r, red));
  textureStore(vdbFrameEmissionProbesOut, vec3u(vdbEmissionGrid.origin[base + 1u].xyz) + cell, vec4f(constant.g, green));
  textureStore(vdbFrameEmissionProbesOut, vec3u(vdbEmissionGrid.origin[base + 2u].xyz) + cell, vec4f(constant.b, blue));
}
`;
}

const RECEIVER = `fn vdbEmissionGridUvw(grid: VdbEmissionGrid, cascade: u32, world: vec3f) -> vec3f {
  return (world - grid.gridMin[cascade].xyz) / grid.gridSize[cascade].xyz;
}
// the finest cascade from first that holds the position half a probe in, or -1
fn vdbEmissionFindCascade(grid: VdbEmissionGrid, world: vec3f, first: u32) -> i32 {
  let inset = 0.5 / grid.info.y;
  for (var c = first; c < u32(grid.info.x); c++) {
    let uvw = vdbEmissionGridUvw(grid, c, world);
    if (min(uvw.x, min(uvw.y, uvw.z)) >= inset && max(uvw.x, max(uvw.y, uvw.z)) <= 1.0 - inset) {
      return i32(c);
    }
  }
  return -1;
}
// one channel's L1 probe of a cascade: hardware trilinear in its sub-volume, clamped to the half-texel box
fn vdbEmissionProbeSample(probes: texture_3d<f32>, smp: sampler, grid: VdbEmissionGrid, cascade: u32, channel: u32, uvw: vec3f) -> vec4f {
  let r = grid.info.y;
  let p = grid.origin[cascade * 3u + channel].xyz + clamp(uvw * r, vec3f(0.5), vec3f(r - 0.5));
  return textureSampleLevel(probes, smp, p / vec3f(textureDimensions(probes)), 0.0);
}
// The irradiance of the volumes' emission at a lit point: the finest cascade that holds it, blended into the next one
// within a tenth of its sides; past the last cascade its clamped edge probes fall off with distance squared
fn vdbEmissionIrradiance(probes: texture_3d<f32>, smp: sampler, grid: VdbEmissionGrid, world: vec3f, normal: vec3f) -> vec3f {
  if (grid.info.x < 0.5) {
    return vec3f(0.0);
  }
  var red = vec4f(0.0);
  var green = vec4f(0.0);
  var blue = vec4f(0.0);
  var falloff = 1.0;
  let found = vdbEmissionFindCascade(grid, world, 0u);
  if (found >= 0) {
    let cascade = u32(found);
    let uvw = vdbEmissionGridUvw(grid, cascade, world);
    red = vdbEmissionProbeSample(probes, smp, grid, cascade, 0u, uvw);
    green = vdbEmissionProbeSample(probes, smp, grid, cascade, 1u, uvw);
    blue = vdbEmissionProbeSample(probes, smp, grid, cascade, 2u, uvw);
    let side = min(uvw, vec3f(1.0) - uvw);
    let blend = 1.0 - smoothstep(0.0, 0.1, min(side.x, min(side.y, side.z)));
    if (blend > 0.0) {
      let next = vdbEmissionFindCascade(grid, world, cascade + 1u);
      if (next >= 0) {
        let n = u32(next);
        let nextUvw = vdbEmissionGridUvw(grid, n, world);
        red = mix(red, vdbEmissionProbeSample(probes, smp, grid, n, 0u, nextUvw), vec4f(blend));
        green = mix(green, vdbEmissionProbeSample(probes, smp, grid, n, 1u, nextUvw), vec4f(blend));
        blue = mix(blue, vdbEmissionProbeSample(probes, smp, grid, n, 2u, nextUvw), vec4f(blend));
      }
    }
  } else {
    let last = u32(grid.info.x) - 1u;
    let inset = 0.5 / grid.info.y;
    let uvw = clamp(vdbEmissionGridUvw(grid, last, world), vec3f(inset), vec3f(1.0 - inset));
    let toEdge = grid.gridMin[last].xyz + uvw * grid.gridSize[last].xyz - grid.center.xyz;
    let toPoint = world - grid.center.xyz;
    red = vdbEmissionProbeSample(probes, smp, grid, last, 0u, uvw);
    green = vdbEmissionProbeSample(probes, smp, grid, last, 1u, uvw);
    blue = vdbEmissionProbeSample(probes, smp, grid, last, 2u, uvw);
    falloff = min(1.0, dot(toEdge, toEdge) / max(dot(toPoint, toPoint), 1e-6));
  }
  let e = vec3f(red.x + dot(red.yzw, normal), green.x + dot(green.yzw, normal), blue.x + dot(blue.yzw, normal));
  return max(e, vec3f(0.0)) * (falloff * grid.center.w);
}
`;

/**
 * WGSL for hosts that light their surfaces with the volumes' emission (design §2.5): the `VdbEmissionGrid` struct and
 *
 *     fn vdbEmissionIrradiance(probes: texture_3d<f32>, smp: sampler, grid: VdbEmissionGrid, world: vec3f, normal: vec3f) -> vec3f
 *
 * Bind `outputs.emissionProbes.texture` (a `'3d'` view), a filtering sampler (linear, clamp to edge) and
 * `outputs.emissionProbes.grid` (`var<uniform>`). The result is irradiance (his light node's E(n) before the light's
 * colour × intensity); a Lambert surface adds albedo × E / π. CPU twin: tools/frame `vdbFrameEmissionLookup` +
 * `vdbFrameEmissionIrradiance` (on a stack sampler).
 */
export function vdbEmissionProbeWgsl(): string {
  return `${VDB_EMISSION_GRID.wgsl}${RECEIVER}`;
}

/** The `?probe=emission` receiver kernel (harness): rows of (world, 0), (normal, 0) → (E rgb, 0). */
export function vdbEmissionIrradianceProbeWgsl(): string {
  return `${vdbEmissionProbeWgsl()}
@group(0) @binding(0) var<uniform> vdbProbeGrid: VdbEmissionGrid;
@group(0) @binding(1) var vdbProbeStack: texture_3d<f32>;
@group(0) @binding(2) var vdbProbeSampler: sampler;
@group(0) @binding(3) var<storage, read_write> vdbProbeRows: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn vdbEmissionIrradianceProbeCs(@builtin(global_invocation_id) gid: vec3u) {
  let rows = arrayLength(&vdbProbeRows) / 2u;
  if (gid.x >= rows) {
    return;
  }
  let world = vdbProbeRows[gid.x * 2u].xyz;
  let normal = vdbProbeRows[gid.x * 2u + 1u].xyz;
  vdbProbeRows[gid.x * 2u] = vec4f(vdbEmissionIrradiance(vdbProbeStack, vdbProbeSampler, vdbProbeGrid, world, normal), 0.0);
}
`;
}

/** Whether `channels` emit (the emission kernels and the LUT binding exist only then). */
export const vdbFrameEmits = (channels: VDBVolumeChannels): boolean => !!channels.emission;
