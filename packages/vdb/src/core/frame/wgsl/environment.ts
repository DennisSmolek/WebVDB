// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameEnvironment.js L60–330 (setTexels, getDirectionalLights, getShLights: the
//   directional cells and the SH quadrature lights, now in one workgroup), FrameEnvironmentMapMaterial.js (the
//   64 × 32 map), FrameEnvironmentGridMaterial.js L201–259 (the grey environment grid pass).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes (K6.2b, design §3.3, §3.5, Q3): compute kernels on the host's equirect or cube texture; an exact-area
// (equirect) or supersampled (cube) work map and a GGX roughness-0.4 convolution instead of three's PMREM; the light
// extraction and an SH9 irradiance projection run on the GPU into a buffer later kernels bind as a uniform (no CPU
// readback); the environment grid is a 3D texture read with hardware trilinear. CPU twins: tools/frame
// (environment-gpu.ts, stack.ts). Divergences: DIVERGES.md (K6.2b rows).

/**
 * The environment kernels of the core frame renderer (K6.2b):
 *
 * - `vdbFrameEnvAreaCs` / `vdbFrameEnvCubeCs` (`envPrefilter`, 8×8×1 over 128 × 64): the host texture → the work map
 *   (rgba32float, map space, row 0 the zenith).
 * - `vdbFrameEnvBlurCs` (8×8×1 over 64 × 32): his 64 × 32 map (world space, his directions) as the GGX roughness-0.4
 *   convolution of the work map.
 * - `vdbFrameEnvReduceCs` (one workgroup of 64, compat's limits): his 'directional' / 'spherical-harmonics' lights from the map and
 *   the SH9 of the work map, into the environment buffer (`VdbFrameEnvironment`).
 * - `vdbFrameEnvironmentGridCs` (4×4×4 over the last cascade): his environment grid pass.
 */

import { frameEnvironmentMapHeight, frameEnvironmentMapWidth, frameMaxEnvironmentSamples } from "../../../tools/frame/environment.js";
import { VDB_FRAME_ENV_MAX_SPAN, VDB_FRAME_ENV_ROUGHNESS, VDB_FRAME_ENV_TIE, VDB_FRAME_ENV_WORK_HEIGHT, VDB_FRAME_ENV_WORK_WIDTH } from "../../../tools/frame/environment-gpu.js";
import { VDB_FRAME_ENV_PARAMS, VDB_FRAME_ENVIRONMENT, VDB_FRAME_MAX_ENVIRONMENT_LIGHTS } from "../uniforms.js";
import { frameChunksWgsl } from "./chunks.js";

/** Group-1 bindings of the prefilter kernels (area / cube). */
export const ENV_PREFILTER_BINDINGS = { source: 0, work: 1 } as const;
/** Group-1 bindings of the blur kernel. */
export const ENV_BLUR_BINDINGS = { work: 0, map: 1 } as const;
/** Group-1 bindings of the reduce kernel. */
export const ENV_REDUCE_BINDINGS = { work: 0, map: 1, environment: 2 } as const;
/** Group-1 bindings of the environment grid kernel. */
export const ENV_GRID_BINDINGS = { grid: 0, out: 1 } as const;

const f = (x: number) => (Number.isInteger(x) ? `${x}.0` : `${x}`);
// an f32 literal: 9 significant digits round-trip every f32
const f32 = (x: number) => {
  const v = String(Number(Math.fround(x).toPrecision(9)));
  return /[.e]/.test(v) ? v : `${v}.0`;
};

/**
 * Direction tables of the work map and his map (cos, sin of the azimuth per column; sin, cos of the polar angle per
 * row) and the rows' solid angles, f32-rounded: WGSL's sin / cos may be off by 2^-11 (absolute), which the GGX lobe
 * (D ∝ 1/d², d ≈ α² at its peak) would turn into ~1 % weight errors. Tables make the directions exact to f32.
 */
function trigTable(name: string, columns: number, rows: number, phi: (i: number) => number, theta: (j: number) => number): string {
  const n = Math.max(columns, rows);
  const items = Array.from({ length: n }, (_, i) => {
    const a = i < columns ? phi(i) : 0;
    const t = i < rows ? theta(i) : 0;
    return `vec4f(${f32(i < columns ? Math.cos(a) : 0)}, ${f32(i < columns ? Math.sin(a) : 0)}, ${f32(i < rows ? Math.sin(t) : 0)}, ${f32(i < rows ? Math.cos(t) : 0)})`;
  });
  return `const ${name} = array<vec4f, ${n}>(${items.join(", ")});`;
}
const WORK_TRIG = trigTable("VDB_FRAME_ENV_WORK_TRIG", VDB_FRAME_ENV_WORK_WIDTH, VDB_FRAME_ENV_WORK_HEIGHT, (i) => ((i + 0.5) / VDB_FRAME_ENV_WORK_WIDTH - 0.5) * 2 * Math.PI, (j) => ((j + 0.5) / VDB_FRAME_ENV_WORK_HEIGHT) * Math.PI);
const MAP_TRIG = trigTable("VDB_FRAME_ENV_MAP_TRIG", frameEnvironmentMapWidth, frameEnvironmentMapHeight, (i) => ((i + 0.5) / frameEnvironmentMapWidth) * 2 * Math.PI, (j) => ((j + 0.5) / frameEnvironmentMapHeight) * Math.PI);
const WORK_OMEGA = `const VDB_FRAME_ENV_WORK_OMEGA = array<f32, ${VDB_FRAME_ENV_WORK_HEIGHT}>(${Array.from({ length: VDB_FRAME_ENV_WORK_HEIGHT }, (_, y) => f32(((2 * Math.PI) / VDB_FRAME_ENV_WORK_WIDTH) * (Math.cos((y * Math.PI) / VDB_FRAME_ENV_WORK_HEIGHT) - Math.cos(((y + 1) * Math.PI) / VDB_FRAME_ENV_WORK_HEIGHT)))).join(", ")});`;
const MAP_OMEGA = `const VDB_FRAME_ENV_MAP_OMEGA = array<f32, ${frameEnvironmentMapHeight}>(${Array.from({ length: frameEnvironmentMapHeight }, (_, y) => f32(((2 * Math.PI) / frameEnvironmentMapWidth) * (Math.PI / frameEnvironmentMapHeight) * Math.sin(((y + 0.5) / frameEnvironmentMapHeight) * Math.PI))).join(", ")});`;

const PARAMS = `${VDB_FRAME_ENV_PARAMS.wgsl}@group(0) @binding(0) var<uniform> vdbFrameEnvParams: VdbFrameEnvParams;
const VDB_FRAME_ENV_PI: f32 = 3.141592653589793;
const VDB_FRAME_ENV_WORK: vec2u = vec2u(${VDB_FRAME_ENV_WORK_WIDTH}u, ${VDB_FRAME_ENV_WORK_HEIGHT}u);
const VDB_FRAME_ENV_MAP: vec2u = vec2u(${frameEnvironmentMapWidth}u, ${frameEnvironmentMapHeight}u);
// The map direction of an equirect point: u along x (u = atan2(z, x) / 2pi + 1/2), v from the top row (+Y)
fn vdbFrameEquirectDirection(u: f32, v: f32) -> vec3f {
  let theta = v * VDB_FRAME_ENV_PI;
  let phi = (u - 0.5) * 2.0 * VDB_FRAME_ENV_PI;
  return vec3f(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}
// cos(a) - cos(b) without the cancellation near the poles
fn vdbFrameEnvBand(a: f32, b: f32) -> f32 {
  return 2.0 * sin((a + b) * 0.5) * sin((b - a) * 0.5);
}
`;

/** `envPrefilter` for an equirect source: the exact-area solid-angle mean of the source over each work texel. */
export function vdbFrameEnvAreaWgsl(): string {
  const b = ENV_PREFILTER_BINDINGS;
  return `${PARAMS}
@group(1) @binding(${b.source}) var vdbFrameEnvSource: texture_2d<f32>;
@group(1) @binding(${b.work}) var vdbFrameEnvWorkOut: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn vdbFrameEnvAreaCs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= VDB_FRAME_ENV_WORK.x || gid.y >= VDB_FRAME_ENV_WORK.y) {
    return;
  }
  let size = vdbFrameEnvParams.source.xy;
  let sw = f32(size.x);
  let sh = f32(size.y);
  let u0 = f32(gid.x) / f32(VDB_FRAME_ENV_WORK.x);
  let u1 = f32(gid.x + 1u) / f32(VDB_FRAME_ENV_WORK.x);
  let v0 = f32(gid.y) / f32(VDB_FRAME_ENV_WORK.y);
  let v1 = f32(gid.y + 1u) / f32(VDB_FRAME_ENV_WORK.y);
  let i0 = u32(floor(u0 * sw));
  let i1 = min(size.x, u32(ceil(u1 * sw)));
  let j0 = u32(floor(v0 * sh));
  let j1 = min(size.y, u32(ceil(v1 * sh)));
  let si = max(1u, (i1 - i0 + ${VDB_FRAME_ENV_MAX_SPAN - 1}u) / ${VDB_FRAME_ENV_MAX_SPAN}u);
  let sj = max(1u, (j1 - j0 + ${VDB_FRAME_ENV_MAX_SPAN - 1}u) / ${VDB_FRAME_ENV_MAX_SPAN}u);
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var j = j0; j < j1; j += sj) {
    let a = max(f32(j) / sh, v0) * VDB_FRAME_ENV_PI;
    let b = min(f32(j + sj) / sh, v1) * VDB_FRAME_ENV_PI;
    if (!(b > a)) {
      continue;
    }
    let wv = vdbFrameEnvBand(a, b);
    for (var i = i0; i < i1; i += si) {
      let wu = min(f32(i + si) / sw, u1) - max(f32(i) / sw, u0);
      if (!(wu > 0.0)) {
        continue;
      }
      sum += textureLoad(vdbFrameEnvSource, vec2u(i, j), 0).rgb * (wu * wv);
      weight += wu * wv;
    }
  }
  textureStore(vdbFrameEnvWorkOut, gid.xy, vec4f(select(vec3f(0.0), sum / weight, weight > 0.0), 1.0));
}
`;
}

/** `envPrefilter` for a cube source (a 6-layer 2D array view, WebGPU's faces): k × k nearest sub-samples per texel. */
export function vdbFrameEnvCubeWgsl(): string {
  const b = ENV_PREFILTER_BINDINGS;
  return `${PARAMS}
@group(1) @binding(${b.source}) var vdbFrameEnvSource: texture_2d_array<f32>;
@group(1) @binding(${b.work}) var vdbFrameEnvWorkOut: texture_storage_2d<rgba32float, write>;

// WebGPU's cube face selection and orientation: the face and texel of a direction
fn vdbFrameCubeTexel(m: vec3f, size: u32) -> vec3u {
  let a = abs(m);
  var face = 0u;
  var ma = a.x;
  var sc = 0.0;
  var tc = 0.0;
  if (a.x >= a.y && a.x >= a.z) {
    face = select(1u, 0u, m.x >= 0.0);
    ma = a.x;
    sc = select(m.z, -m.z, m.x >= 0.0);
    tc = -m.y;
  } else if (a.y >= a.z) {
    face = select(3u, 2u, m.y >= 0.0);
    ma = a.y;
    sc = m.x;
    tc = select(-m.z, m.z, m.y >= 0.0);
  } else {
    face = select(5u, 4u, m.z >= 0.0);
    ma = a.z;
    sc = select(-m.x, m.x, m.z >= 0.0);
    tc = -m.y;
  }
  let s = (sc / ma + 1.0) * 0.5;
  let t = (tc / ma + 1.0) * 0.5;
  let last = f32(size - 1u);
  return vec3u(u32(clamp(floor(s * f32(size)), 0.0, last)), u32(clamp(floor(t * f32(size)), 0.0, last)), face);
}

@compute @workgroup_size(8, 8, 1)
fn vdbFrameEnvCubeCs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= VDB_FRAME_ENV_WORK.x || gid.y >= VDB_FRAME_ENV_WORK.y) {
    return;
  }
  let size = vdbFrameEnvParams.source.x;
  let k = max(vdbFrameEnvParams.source.w, 1u);
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var b = 0u; b < k; b++) {
    let v = (f32(gid.y) + (f32(b) + 0.5) / f32(k)) / f32(VDB_FRAME_ENV_WORK.y);
    let w = sin(v * VDB_FRAME_ENV_PI);
    for (var a = 0u; a < k; a++) {
      let u = (f32(gid.x) + (f32(a) + 0.5) / f32(k)) / f32(VDB_FRAME_ENV_WORK.x);
      let t = vdbFrameCubeTexel(vdbFrameEquirectDirection(u, v), size);
      sum += textureLoad(vdbFrameEnvSource, t.xy, t.z, 0).rgb * w;
      weight += w;
    }
  }
  textureStore(vdbFrameEnvWorkOut, gid.xy, vec4f(select(vec3f(0.0), sum / weight, weight > 0.0), 1.0));
}
`;
}

/**
 * The directions of the work map texels and of his 64 × 32 map texels (his vdbFrameEnvironmentMapDirection: phi = u 2pi,
 * theta = v pi) and their solid angles, from the f32 tables.
 */
const MAP_DIRECTION = `${WORK_TRIG}
${MAP_TRIG}
${WORK_OMEGA}
${MAP_OMEGA}
fn vdbFrameEnvWorkDirection(x: u32, y: u32) -> vec3f {
  let a = VDB_FRAME_ENV_WORK_TRIG[x];
  let b = VDB_FRAME_ENV_WORK_TRIG[y];
  return vec3f(b.z * a.x, b.w, b.z * a.y);
}
fn vdbFrameEnvMapDirection(x: u32, y: u32) -> vec3f {
  let a = VDB_FRAME_ENV_MAP_TRIG[x];
  let b = VDB_FRAME_ENV_MAP_TRIG[y];
  return vec3f(b.z * a.x, b.w, b.z * a.y);
}
fn vdbFrameEnvToMap(d: vec3f) -> vec3f {
  let r = vdbFrameEnvParams.rotation;
  return vec3f(dot(r[0].xyz, d), dot(r[1].xyz, d), dot(r[2].xyz, d));
}
fn vdbFrameEnvToWorld(m: vec3f) -> vec3f {
  let r = vdbFrameEnvParams.rotation;
  return r[0].xyz * m.x + r[1].xyz * m.y + r[2].xyz * m.z;
}
`;

/** The GGX roughness-0.4 convolution of the work map into his 64 × 32 map (world space). */
export function vdbFrameEnvBlurWgsl(): string {
  const b = ENV_BLUR_BINDINGS;
  const r = VDB_FRAME_ENV_ROUGHNESS;
  return `${PARAMS}${MAP_DIRECTION}
@group(1) @binding(${b.work}) var vdbFrameEnvWork: texture_2d<f32>;
@group(1) @binding(${b.map}) var vdbFrameEnvMapOut: texture_storage_2d<rgba32float, write>;

// The GGX weight D(h) n.l of an input direction cosine c (n = v = r)
fn vdbFrameEnvGgxWeight(c: f32) -> f32 {
  if (!(c > 0.0)) {
    return 0.0;
  }
  let a2 = ${f(r * r * r * r)};
  let d = (1.0 + c) * 0.5 * (a2 - 1.0) + 1.0;
  return a2 / (VDB_FRAME_ENV_PI * d * d) * c;
}

@compute @workgroup_size(8, 8, 1)
fn vdbFrameEnvBlurCs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= VDB_FRAME_ENV_MAP.x || gid.y >= VDB_FRAME_ENV_MAP.y) {
    return;
  }
  let m = vdbFrameEnvToMap(vdbFrameEnvMapDirection(gid.x, gid.y));
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var j = 0u; j < VDB_FRAME_ENV_WORK.y; j++) {
    let omega = VDB_FRAME_ENV_WORK_OMEGA[j];
    for (var i = 0u; i < VDB_FRAME_ENV_WORK.x; i++) {
      let mj = vdbFrameEnvWorkDirection(i, j);
      let w = vdbFrameEnvGgxWeight(dot(m, mj)) * omega;
      if (!(w > 0.0)) {
        continue;
      }
      sum += textureLoad(vdbFrameEnvWork, vec2u(i, j), 0).rgb * w;
      weight += w;
    }
  }
  textureStore(vdbFrameEnvMapOut, gid.xy, vec4f(select(vec3f(0.0), sum / weight, weight > 0.0), 1.0));
}
`;
}

/**
 * `envReduce`: one workgroup of 64 threads (compatibility devices allow 128). His setTexels (clamped radiance, midpoint solid angles, finite and
 * positive energy), 'directional' (brightest texel, fixed sky seeds, nearest seed, energy centroids, sorted by
 * power) or 'spherical-harmonics' lights (SH L1 / L2 at octahedron / icosahedron directions) from the 64 × 32 map;
 * SH9 of the work map in world space; colours and SH × intensity. Every reduction runs in uniform control flow.
 */
export function vdbFrameEnvReduceWgsl(): string {
  const b = ENV_REDUCE_BINDINGS;
  const texels = frameEnvironmentMapWidth * frameEnvironmentMapHeight;
  const work = VDB_FRAME_ENV_WORK_WIDTH * VDB_FRAME_ENV_WORK_HEIGHT;
  const golden = (1 + Math.sqrt(5)) / 2;
  const ringMax = frameMaxEnvironmentSamples - 2;
  const ringEntries = (ringMax * (ringMax + 1)) / 2;
  const ringTable: string[] = [];
  for (let n = 1; n <= ringMax; n++) for (let r = 0; r < n; r++) ringTable.push(`vec2f(${f32(Math.cos((2 * Math.PI * r) / n))}, ${f32(Math.sin((2 * Math.PI * r) / n))})`);
  const ico = [
    [0, 1, golden],
    [0, -1, golden],
    [0, 1, -golden],
    [0, -1, -golden],
    [1, golden, 0],
    [-1, golden, 0],
    [1, -golden, 0],
    [-1, -golden, 0],
    [golden, 0, 1],
    [-golden, 0, 1],
    [golden, 0, -1],
    [-golden, 0, -1],
  ].map(([x, y, z]) => {
    const l = Math.hypot(x!, y!, z!);
    return `vec3f(${f(x! / l)}, ${f(y! / l)}, ${f(z! / l)})`;
  });
  return `${PARAMS}${MAP_DIRECTION}${VDB_FRAME_ENVIRONMENT.wgsl}
@group(1) @binding(${b.work}) var vdbFrameEnvWork: texture_2d<f32>;
@group(1) @binding(${b.map}) var vdbFrameEnvMap: texture_2d<f32>;
@group(1) @binding(${b.environment}) var<storage, read_write> vdbFrameEnvOut: VdbFrameEnvironment;

const VDB_FRAME_ENV_TEXELS: u32 = ${texels}u;
const VDB_FRAME_ENV_WORK_TEXELS: u32 = ${work}u;
const VDB_FRAME_ENV_MAX_SEEDS: u32 = ${frameMaxEnvironmentSamples}u;
// near-ties (the brightest texel, the nearest seed) go to the first candidate (vdbFrameEnvironmentReduce's tolerance)
const VDB_FRAME_ENV_TIE: f32 = ${f(VDB_FRAME_ENV_TIE)};
// (cos, sin) of 2 pi r / n for n = 1..${frameMaxEnvironmentSamples - 2}, r < n, at n (n - 1) / 2 + r
const VDB_FRAME_ENV_RING = array<vec2f, ${ringEntries}>(${ringTable});
const VDB_FRAME_ENV_ICOSAHEDRON = array<vec3f, 12>(${ico.join(", ")});
const VDB_FRAME_ENV_OCTAHEDRON = array<vec3f, 6>(vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0));

var<workgroup> vdbFrameEnvSum: array<vec4f, 64>;
var<workgroup> vdbFrameEnvPeak: array<vec2f, 64>;

fn vdbFrameEnvLuminance(c: vec3f) -> f32 {
  return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}
// three's SH9 basis
fn vdbFrameEnvShBasis(d: vec3f, k: u32) -> f32 {
  switch (k) {
    case 0u: { return 0.282095; }
    case 1u: { return 0.488603 * d.y; }
    case 2u: { return 0.488603 * d.z; }
    case 3u: { return 0.488603 * d.x; }
    case 4u: { return 1.092548 * d.x * d.y; }
    case 5u: { return 1.092548 * d.y * d.z; }
    case 6u: { return 0.315392 * (3.0 * d.z * d.z - 1.0); }
    case 7u: { return 1.092548 * d.x * d.z; }
    default: { return 0.546274 * (d.x * d.x - d.y * d.y); }
  }
}
fn vdbFrameEnvFinite(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
// Workgroup sum of v (call from uniform control flow by every thread)
fn vdbFrameEnvReduceSum(t: u32, v: vec4f) -> vec4f {
  vdbFrameEnvSum[t] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (t < s) {
      vdbFrameEnvSum[t] += vdbFrameEnvSum[t + s];
    }
    workgroupBarrier();
  }
  let r = vdbFrameEnvSum[0];
  workgroupBarrier();
  return r;
}
// A map texel: (clamped radiance, luminance) and (direction, his midpoint solid angle)
fn vdbFrameEnvTexel(i: u32) -> array<vec4f, 2> {
  let x = i % VDB_FRAME_ENV_MAP.x;
  let y = i / VDB_FRAME_ENV_MAP.x;
  let c = max(textureLoad(vdbFrameEnvMap, vec2u(x, y), 0).rgb, vec3f(0.0));
  return array<vec4f, 2>(vec4f(c, vdbFrameEnvLuminance(c)), vec4f(vdbFrameEnvMapDirection(x, y), VDB_FRAME_ENV_MAP_OMEGA[y]));
}

@compute @workgroup_size(64, 1, 1)
fn vdbFrameEnvReduceCs(@builtin(local_invocation_index) t: u32) {
  let settings = vdbFrameEnvParams.settings;
  let mode = u32(settings.x + 0.5);
  let intensity = vdbFrameEnvParams.scale.x;
  let per = VDB_FRAME_ENV_TEXELS / 64u;

  // setTexels: energy, finiteness and the brightest texel: the first texel within VDB_FRAME_ENV_TIE (relative) of the
  // maximum, so near-ties do not depend on f32 rounding (his JS takes the first exact maximum)
  var energy = 0.0;
  var bad = 0.0;
  var brightest = 0.0;
  for (var n = 0u; n < per; n++) {
    let i = t + n * 64u;
    let raw = textureLoad(vdbFrameEnvMap, vec2u(i % VDB_FRAME_ENV_MAP.x, i / VDB_FRAME_ENV_MAP.x), 0).rgb;
    if (!vdbFrameEnvFinite(raw.x) || !vdbFrameEnvFinite(raw.y) || !vdbFrameEnvFinite(raw.z)) {
      bad += 1.0;
    }
    let tx = vdbFrameEnvTexel(i);
    energy += tx[0].w * tx[1].w;
    brightest = max(brightest, tx[0].w);
  }
  let totals = vdbFrameEnvReduceSum(t, vec4f(energy, bad, 0.0, 0.0));
  vdbFrameEnvPeak[t] = vec2f(brightest, 0.0);
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (t < s) {
      vdbFrameEnvPeak[t].x = max(vdbFrameEnvPeak[t].x, vdbFrameEnvPeak[t + s].x);
    }
    workgroupBarrier();
  }
  let threshold = vdbFrameEnvPeak[0].x * (1.0 - VDB_FRAME_ENV_TIE);
  workgroupBarrier();
  var first = f32(VDB_FRAME_ENV_TEXELS);
  for (var n = 0u; n < per; n++) {
    let i = t + n * 64u;
    if (vdbFrameEnvTexel(i)[0].w >= threshold) {
      first = min(first, f32(i));
    }
  }
  vdbFrameEnvPeak[t] = vec2f(first, 0.0);
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (t < s) {
      vdbFrameEnvPeak[t].x = min(vdbFrameEnvPeak[t].x, vdbFrameEnvPeak[t + s].x);
    }
    workgroupBarrier();
  }
  let peakIndex = min(u32(vdbFrameEnvPeak[0].x), VDB_FRAME_ENV_TEXELS - 1u);
  workgroupBarrier();
  let valid = totals.y == 0.0 && totals.x > 0.0 && vdbFrameEnvFinite(totals.x);

  // 'directional' seeds (his getDirectionalLights): the peak, then zenith / nadir (or the opposite pair), a horizon ring
  let pairs = settings.w > 0.5;
  let count = min(VDB_FRAME_ENV_MAX_SEEDS, max(1u, u32(max(round(settings.y), 1.0))));
  let peakDir = vdbFrameEnvMapDirection(peakIndex % VDB_FRAME_ENV_MAP.x, peakIndex / VDB_FRAME_ENV_MAP.x);
  var seeds: array<vec3f, ${frameMaxEnvironmentSamples}>;
  var seedCount = 1u;
  seeds[0] = peakDir;
  let rest = select(0u, count - 4u, count > 4u);
  var ring = select(i32(count) - 3, i32(rest - rest % 2u), pairs);
  if (pairs) {
    if (count >= 2u) {
      seeds[seedCount] = -peakDir;
      seedCount++;
    }
    if (count >= 4u) {
      seeds[seedCount] = vec3f(0.0, 1.0, 0.0);
      seeds[seedCount + 1u] = vec3f(0.0, -1.0, 0.0);
      seedCount += 2u;
    }
  } else {
    if (count >= 2u) {
      seeds[seedCount] = vec3f(0.0, 1.0, 0.0);
      seedCount++;
    }
    if (count >= 3u) {
      seeds[seedCount] = vec3f(0.0, -1.0, 0.0);
      seedCount++;
    }
  }
  // the ring starts opposite the peak's azimuth: (-peak.x, -peak.z) normalised, turned by 2 pi r / ring (a table, no
  // trigonometry: WGSL's sin / cos may be 2^-11 off, which moves cell boundaries across texels)
  let horizontal = vec2f(peakDir.x, peakDir.z);
  let h = length(horizontal);
  let start = select(vec2f(-1.0, 0.0), -horizontal / h, h > 1e-20);
  var ringTable = VDB_FRAME_ENV_RING;
  for (var r = 0; r < ring; r++) {
    let cs = ringTable[u32(ring * (ring - 1) / 2 + r)];
    seeds[seedCount] = vec3f(cs.x * start.x - cs.y * start.y, 0.0, cs.y * start.x + cs.x * start.y);
    seedCount++;
  }

  // Each texel to its nearest seed (the first of the closest): energy and luminance-weighted centroid per cell
  var cellEnergy: array<vec4f, ${frameMaxEnvironmentSamples}>;
  var cellCentroid: array<vec4f, ${frameMaxEnvironmentSamples}>;
  var shLights: array<vec4f, 9>;
  for (var n = 0u; n < per; n++) {
    let tx = vdbFrameEnvTexel(t + n * 64u);
    let d = tx[1].xyz;
    let omega = tx[1].w;
    var nearest = 0u;
    var nearestDot = -3.0;
    for (var c = 0u; c < seedCount; c++) {
      let k = dot(seeds[c], d);
      if (k > nearestDot + VDB_FRAME_ENV_TIE) {
        nearest = c;
        nearestDot = k;
      }
    }
    cellEnergy[nearest] += vec4f(tx[0].xyz * omega, 0.0);
    cellCentroid[nearest] += vec4f(d * (tx[0].w * omega), 0.0);
    for (var k = 0u; k < 9u; k++) {
      shLights[k] += vec4f(tx[0].xyz * (vdbFrameEnvShBasis(d, k) * omega), 0.0);
    }
  }
  // SH9 of the work map in world space (the 'outward' irradiance)
  var sh: array<vec4f, 9>;
  for (var n = 0u; n < VDB_FRAME_ENV_WORK_TEXELS / 64u; n++) {
    let i = t + n * 64u;
    let x = i % VDB_FRAME_ENV_WORK.x;
    let y = i / VDB_FRAME_ENV_WORK.x;
    let d = vdbFrameEnvToWorld(vdbFrameEnvWorkDirection(x, y));
    let c = max(textureLoad(vdbFrameEnvWork, vec2u(x, y), 0).rgb, vec3f(0.0)) * VDB_FRAME_ENV_WORK_OMEGA[y];
    for (var k = 0u; k < 9u; k++) {
      sh[k] += vec4f(c * vdbFrameEnvShBasis(d, k), 0.0);
    }
  }
  var cellsE: array<vec4f, ${frameMaxEnvironmentSamples}>;
  var cellsC: array<vec4f, ${frameMaxEnvironmentSamples}>;
  for (var c = 0u; c < VDB_FRAME_ENV_MAX_SEEDS; c++) {
    cellsE[c] = vdbFrameEnvReduceSum(t, cellEnergy[c]);
    cellsC[c] = vdbFrameEnvReduceSum(t, cellCentroid[c]);
  }
  var shL: array<vec4f, 9>;
  var shW: array<vec4f, 9>;
  for (var k = 0u; k < 9u; k++) {
    shL[k] = vdbFrameEnvReduceSum(t, shLights[k]);
    shW[k] = vdbFrameEnvReduceSum(t, sh[k]);
  }
  if (t != 0u) {
    return;
  }

  var out: VdbFrameEnvironment;
  out.info = vec4u(0u, mode, select(0u, 1u, valid), 0u);
  for (var k = 0u; k < 9u; k++) {
    out.sh[k] = select(vec4f(0.0), vec4f(shW[k].xyz * intensity, 0.0), valid);
  }
  var lightCount = 0u;
  if (valid && mode == 1u) {
    // lights with energy, sorted by power (descending; stable)
    var power: array<f32, ${frameMaxEnvironmentSamples}>;
    for (var c = 0u; c < seedCount; c++) {
      let e = cellsE[c].xyz;
      let p = vdbFrameEnvLuminance(e);
      if (!(p > 0.0)) {
        continue;
      }
      let centroid = cellsC[c].xyz;
      let direction = select(seeds[c], normalize(centroid), length(centroid) > 1e-6);
      var at = lightCount;
      while (at > 0u && power[at - 1u] < p) {
        power[at] = power[at - 1u];
        out.direction[at] = out.direction[at - 1u];
        out.color[at] = out.color[at - 1u];
        at--;
      }
      power[at] = p;
      out.direction[at] = vec4f(direction, 0.0);
      out.color[at] = vec4f(e / 4.0 * intensity, 0.0);
      lightCount++;
    }
  } else if (valid && mode == 2u) {
    // his getShLights: the projection evaluated at quadrature directions, (4 pi / M) L(w) / 4
    let order1 = u32(settings.z + 0.5) == 1u;
    let coefficients = select(9u, 4u, order1);
    lightCount = select(12u, 6u, order1);
    let weight = 4.0 * VDB_FRAME_ENV_PI / f32(lightCount) / 4.0;
    var icosahedron = VDB_FRAME_ENV_ICOSAHEDRON;
    var octahedron = VDB_FRAME_ENV_OCTAHEDRON;
    for (var m = 0u; m < lightCount; m++) {
      let d = select(icosahedron[m], octahedron[min(m, 5u)], order1);
      var color = vec3f(0.0);
      for (var k = 0u; k < coefficients; k++) {
        color += shL[k].xyz * (vdbFrameEnvShBasis(d, k) * weight);
      }
      out.direction[m] = vec4f(d, 0.0);
      out.color[m] = vec4f(color * intensity, 0.0);
    }
  }
  out.info.x = min(lightCount, ${VDB_FRAME_MAX_ENVIRONMENT_LIGHTS}u);
  vdbFrameEnvOut = out;
}
`;
}

/**
 * The environment grid kernel (after `vdbFrameUniformsWgsl(…, { environment: true })`; source-free): his grey
 * environment grid pass at each voxel of the last cascade, from the environment lights without a shadow slot.
 */
export function vdbFrameEnvironmentGridWgsl(): string {
  const b = ENV_GRID_BINDINGS;
  return `${frameChunksWgsl(["constants", "shadow", "octaveTotal", "grid"])}
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_3d<f32>;
@group(1) @binding(${b.out}) var vdbFrameEnvGridOut: texture_storage_3d<rgba16float, write>;

// (extinction, effective, reduced) depths of one-voxel steps from world along direction, staying in the cascade
fn vdbFrameEnvGridMarch(cascade: u32, world: vec3f, direction: vec3f, ms: f32) -> vec3f {
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  let perUnit = abs(direction) / voxel;
  let stepLength = 1.0 / max(perUnit.x, max(perUnit.y, perUnit.z));
  let origin = vdbFrame.shadowOrigin[cascade].xyz;
  var probe = world + direction * stepLength * 0.5;
  var depths = vec3f(0.0);
  for (var i = 0u; i < 1024u; i++) {
    let uvw = vdbFrameGridUvw(cascade, probe);
    if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0)) || vdbFrameShadowMarchDepth(depths, ms, vdbFrame.octaves) > 6.0) {
      break;
    }
    let s = vdbFrameStackSample(vdbFrameShadowGrid, origin, uvw);
    depths += vdbFrameShadowStep(s.x, s.y, ms) * stepLength;
    probe += direction * stepLength;
  }
  return depths;
}

@compute @workgroup_size(4, 4, 4)
fn vdbFrameEnvironmentGridCs(@builtin(global_invocation_id) gid: vec3u) {
  let r = u32(vdbFrame.gridLayout.y);
  if (gid.x >= r || gid.y >= r || gid.z >= r) {
    return;
  }
  let cascade = u32(vdbFrame.gridLayout.x) - 1u;
  let world = vdbFrame.gridMin[cascade].xyz + (vec3f(gid) + 0.5) / f32(r) * vdbFrame.gridSize[cascade].xyz;
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  let origin = vdbFrame.shadowOrigin[cascade].xyz;
  let first = u32(vdbFrame.environment.y);
  let count = select(0u, vdbFrameEnv.info.x, vdbFrameEnv.info.z != 0u);
  var unshadowed = vec3f(0.0);
  for (var i = first; i < count; i++) {
    unshadowed += vdbFrameEnv.color[i].xyz;
  }
  var near = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let corner = vec3f(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u)) - 0.5;
    near += vdbFrameStackSample(vdbFrameShadowGrid, origin, vdbFrameGridUvw(cascade, world + corner * voxel)).x;
  }
  // K6.3: the unshadowed light gets the octaves too
  var result = vec4f(max(unshadowed, vec3f(0.0)) * vdbFrameShadowTotal(1.0, vdbFrame.octaves), 1.0);
  if (near > 0.0) {
    let ms = vdbFrame.lighting.x;
    var irradiance = vec3f(0.0);
    for (var i = first; i < count; i++) {
      let direction = vdbFrameEnv.direction[i].xyz;
      var total = 0.0;
      // one march call site per direction: the light, then (diffusion) the opposite one
      var front = vec3f(0.0);
      var back = vec3f(0.0);
      for (var k = 0u; k < select(1u, 2u, ms > 0.0); k++) {
        let depths = vdbFrameEnvGridMarch(cascade, world, select(direction, -direction, k == 1u), ms);
        if (k == 0u) {
          front = depths;
        } else {
          back = depths;
        }
      }
      if (ms > 0.0) {
        total = vdbFrameIsotropicDiffusion(front.z, back.z, front.y);
      } else {
        total = vdbFrameShadowTotal(vdbFrameShadowTransmittance(front, ms, vdbFrame.octaves).y, vdbFrame.octaves);
      }
      irradiance += vdbFrameEnv.color[i].xyz * total;
    }
    result = vec4f(max(irradiance, vec3f(0.0)), 1.0);
  }
  textureStore(vdbFrameEnvGridOut, gid, result);
}
`;
}
