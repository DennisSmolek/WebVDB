// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L94–107 (frameSliceDepth, frameGetViewPosition),
//   L109–127 (frameGetFroxel), L129–142 (frameDistanceFade), L157–185 (getFroxelLod, getBoxSteps,
//   frameNoise), L187–203 (frameHenyeyGreenstein, frameHenyeyGreensteinColor), L273–315 (framePhaseWeight,
//   framePhaseColor), L330–373 (frameShadowAbsorption, frameShadowStep, frameShadowDepth,
//   frameShadowTransmittance), L403–414 (frameIsotropicDiffusion), L938–958 (boxExit, boxSegment);
//   three.js r187 getShIrradianceAt (MIT).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TSL → WGSL for the core compute kernels (K6.1); the froxel of a 3D texel replaces the atlas
// pixel (the noise keeps his virtual atlas coordinate); K6.3: the Lorenz-Mie table reads (frameMieTexel, L206–271), octaves
// (L316–399: frameShadowMarchDepth, frameOctaveLight, frameShadowTotal, frameShadowedLight) and frameTwoSidedDiffusion
// (L416–451); spectral chunks wait for K6.4. CPU twins: tools/frame (index.ts, froxel.ts, mie.ts). Divergences:
// DIVERGES.md (K3.1, K6.1 rows).

import { vdbFrameMieAsymmetry, vdbFrameMieRgbMode, vdbFrameMieTableSize } from "../../../tools/frame/mie.js";
import { vdbFrameMaxScatteringOctaves } from "../../../tools/frame/index.js";

/**
 * `frameChunksWgsl(roots)`: the source-free `vdbFrame*` WGSL functions the frame kernels share, emitted
 * as the dependency closure of `roots` in one fixed order (same roots, same bytes). They read the group-0
 * uniforms `vdbFrame` / `vdbFrameLights` (`vdbFrameUniformsWgsl`).
 */

/** Lorenz-Mie table texels (his `frameMieTableSize`), its asymmetry and the 'lorenz-mie-rgb' mode (tools/frame/mie.ts). */
const MIE_SIZE = vdbFrameMieTableSize;
const MIE_ASYMMETRY = vdbFrameMieAsymmetry;
const MIE_RGB_MODE = vdbFrameMieRgbMode;
const MAX_OCTAVES = vdbFrameMaxScatteringOctaves;

interface Chunk {
  readonly deps: readonly string[];
  readonly code: string;
}

/** The chunks in emission order (a chunk's deps come before it). */
const CHUNKS: ReadonlyArray<readonly [string, Chunk]> = [
  [
    "constants",
    {
      deps: [],
      code: `const VDB_FRAME_PI: f32 = 3.141592653589793;
const VDB_FRAME_FADE_END: f32 = 0.018315638888734179; // exp(-4)
const VDB_FRAME_LIGHT_AMBIENT: u32 = 32u;
const VDB_FRAME_LIGHT_POINT: u32 = 16u;
const VDB_FRAME_LIGHT_DIRECTIONAL: u32 = 8u;
const VDB_FRAME_LIGHT_SPOT: u32 = 4u;
const VDB_FRAME_LIGHT_HEMISPHERE: u32 = 2u;
const VDB_FRAME_LIGHT_ENVIRONMENT: u32 = 1u;
`,
    },
  ],
  [
    "mat3",
    {
      deps: [],
      code: `fn vdbFrameMat3(m: mat4x4f) -> mat3x3f {
  return mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
}
`,
    },
  ],
  [
    "sliceDepth",
    {
      deps: [],
      code: `// frameSliceDepth: view depth at the slice fraction t of the log range
fn vdbFrameSliceDepth(t: f32) -> f32 {
  return vdbFrame.range.x * pow(vdbFrame.range.y / vdbFrame.range.x, t);
}
`,
    },
  ],
  [
    "viewPosition",
    {
      deps: [],
      code: `// frameGetViewPosition: two points on the pixel ray (NDC depth 0 and 1), the one at viewDepth
fn vdbFrameViewPosition(uv: vec2f, viewDepth: f32) -> vec3f {
  let ndc = uv * 2.0 - 1.0;
  let nearPoint = vdbFrame.projectionInverse * vec4f(ndc, 0.0, 1.0);
  let farPoint = vdbFrame.projectionInverse * vec4f(ndc, 1.0, 1.0);
  let near = nearPoint.xyz / nearPoint.w;
  let far = farPoint.xyz / farPoint.w;
  let t = (viewDepth + near.z) / (near.z - far.z);
  return mix(near, far, vec3f(t));
}
`,
    },
  ],
  [
    "froxel",
    {
      deps: ["sliceDepth", "viewPosition"],
      code: `// frameGetFroxel for the 3D texel cell; fragCoord is his atlas pixel (the noise reads it)
struct VdbFrameFroxel {
  fragCoord: vec2f,
  uv: vec2f,
  sliceNear: f32,
  sliceFar: f32,
  viewNear: vec3f,
  viewFar: vec3f,
};
fn vdbFrameFroxelAt(cell: vec3u) -> VdbFrameFroxel {
  let grid = vdbFrame.grid;
  let slice = f32(cell.z);
  let row = floor(slice / grid.w);
  let tile = vec2f(slice - grid.w * row, row);
  let xy = vec2f(cell.xy) + 0.5;
  var f: VdbFrameFroxel;
  f.fragCoord = tile * grid.xy + xy;
  f.uv = (f.fragCoord - tile * grid.xy) / grid.xy;
  f.sliceNear = vdbFrameSliceDepth(slice / grid.z);
  f.sliceFar = vdbFrameSliceDepth((slice + 1.0) / grid.z);
  f.viewNear = vdbFrameViewPosition(f.uv, f.sliceNear);
  f.viewFar = vdbFrameViewPosition(f.uv, f.sliceFar);
  return f;
}
`,
    },
  ],
  [
    "noise",
    {
      deps: [],
      code: `// frameNoise: interleaved gradient noise (Jimenez 2014); 0.5 when jitter is off (negative)
fn vdbFrameNoise(jitter: f32, pixel: vec2f) -> f32 {
  let noise = fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
  return select(fract(noise + jitter), 0.5, jitter < 0.0);
}
`,
    },
  ],
  [
    "fade",
    {
      deps: ["constants"],
      code: `// frameDistanceFade: 1 at the camera, 0 at the fade radius; radius 0: no fade
fn vdbFrameDistanceFade(viewPosition: vec3f) -> f32 {
  let radius = vdbFrame.jitterFade.y;
  if (radius <= 0.0) {
    return 1.0;
  }
  let x = clamp(length(viewPosition) / radius, 0.0, 1.0);
  return (exp(x * x * -4.0) - VDB_FRAME_FADE_END) / (1.0 - VDB_FRAME_FADE_END);
}
`,
    },
  ],
  [
    "box",
    {
      deps: [],
      code: `// boxExit / boxSegment of the local unit box [-0.5, 0.5]^3
fn vdbFrameBoxExit(origin: vec3f, direction: vec3f) -> f32 {
  let inverseDirection = vec3f(1.0) / direction;
  let tMin = (vec3f(-0.5) - origin) * inverseDirection;
  let tMax = (vec3f(0.5) - origin) * inverseDirection;
  let far = max(tMin, tMax);
  return max(0.0, min(far.x, min(far.y, far.z)));
}
fn vdbFrameBoxSegment(start: vec3f, end: vec3f) -> vec2f {
  let inverseDirection = vec3f(1.0) / (end - start);
  let tMin = (vec3f(-0.5) - start) * inverseDirection;
  let tMax = (vec3f(0.5) - start) * inverseDirection;
  let near = min(tMin, tMax);
  let far = max(tMin, tMax);
  return vec2f(max(0.0, max(near.x, max(near.y, near.z))), min(1.0, min(far.x, min(far.y, far.z))));
}
`,
    },
  ],
  [
    "lod",
    {
      deps: ["viewPosition"],
      code: `// getLod: the level of a footprint of localSize (unit-box units)
fn vdbFrameLod(localSize: f32, maxTexels: f32) -> f32 {
  return max(0.0, log2(localSize * maxTexels));
}
// getFroxelLod: the level of the froxel width at the slice middle or the sub-step, the larger
fn vdbFrameFroxelLod(f: VdbFrameFroxel, subStepLength: f32, worldScale: f32, maxTexels: f32) -> f32 {
  let sliceMiddle = (f.sliceNear + f.sliceFar) * 0.5;
  let neighborUv = f.uv + vec2f(1.0 / vdbFrame.grid.x, 0.0);
  let froxelWidth = length(vdbFrameViewPosition(neighborUv, sliceMiddle) - vdbFrameViewPosition(f.uv, sliceMiddle));
  return vdbFrameLod(max(froxelWidth, subStepLength) / worldScale, maxTexels);
}
// getBoxSteps: two sub-steps per voxel of the read level across the in-box part, at most subSteps
fn vdbFrameBoxSteps(localNear: vec3f, localFar: vec3f, segment: f32, sampledLod: f32, maxTexels: f32) -> u32 {
  let boxLength = length(localFar - localNear) * segment;
  let steps = ceil(exp2(vdbFrameLod(boxLength, maxTexels) - sampledLod) * 2.0);
  return u32(clamp(steps, 1.0, f32(vdbFrame.flags.x)));
}
`,
    },
  ],
  [
    "phase",
    {
      deps: ["constants"],
      code: `// frameHenyeyGreenstein: HG over the isotropic phase; g = 0 gives 1
fn vdbFrameHenyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
}
fn vdbFrameHenyeyGreensteinColor(cosTheta: f32, g: vec3f) -> vec3f {
  let g2 = g * g;
  return (vec3f(1.0) - g2) / pow(max(vec3f(1.0) + g2 - g * 2.0 * cosTheta, vec3f(1e-4)), vec3f(1.5));
}
// K6.3: the Lorenz-Mie tables (V47; rgb 'lorenz-mie-rgb', a 'lorenz-mie'), ${MIE_SIZE} texels at the angle PI u^2, read
// with textureLoad and a linear blend (his frameMieTexel)
@group(0) @binding(4) var vdbFrameMieLut: texture_2d<f32>;
fn vdbFrameMieTexel(cosTheta: f32) -> vec4f {
  let u = sqrt(acos(clamp(cosTheta, -1.0, 1.0)) * (1.0 / VDB_FRAME_PI));
  let texel = u * ${MIE_SIZE - 1}.0;
  let index = min(i32(texel), ${MIE_SIZE - 2});
  let a = textureLoad(vdbFrameMieLut, vec2i(index, 0), 0);
  let b = textureLoad(vdbFrameMieLut, vec2i(index + 1, 0), 0);
  return mix(a, b, vec4f(texel - f32(index)));
}
// framePhaseWeight: (forward g, back g, back weight, Lorenz-Mie mode); mode > 0 mixes the 'lorenz-mie' table in
fn vdbFramePhaseWeight(phase: vec4f, cosTheta: f32) -> f32 {
  let w = mix(vdbFrameHenyeyGreenstein(cosTheta, phase.x), vdbFrameHenyeyGreenstein(cosTheta, phase.y), phase.z);
  if (phase.w > 0.0) {
    return mix(w, vdbFrameMieTexel(cosTheta).w, phase.w);
  }
  return w;
}
// framePhaseColor: spectrum (red g scale, blue g scale, flag); flag 0: one phase for all channels; 'lorenz-mie-rgb'
// (mode 2): its three tables
fn vdbFramePhaseColor(phase: vec4f, spectrum: vec3f, cosTheta: f32) -> vec3f {
  if (spectrum.z < 0.5) {
    return vec3f(vdbFramePhaseWeight(phase, cosTheta));
  }
  if (phase.w > ${MIE_RGB_MODE - 0.5}) {
    return vdbFrameMieTexel(cosTheta).xyz;
  }
  let scale = vec3f(spectrum.x, 1.0, spectrum.y);
  return mix(
    vdbFrameHenyeyGreensteinColor(cosTheta, clamp(scale * phase.x, vec3f(-0.99), vec3f(0.99))),
    vdbFrameHenyeyGreensteinColor(cosTheta, clamp(scale * phase.y, vec3f(-0.99), vec3f(0.99))),
    vec3f(phase.z),
  );
}
// framePhaseWeightScaled: every g x gScale; the Lorenz-Mie modes use HG with the table asymmetry x gScale
fn vdbFramePhaseWeightScaled(phase: vec4f, cosTheta: f32, gScale: f32) -> f32 {
  let lobes = mix(vdbFrameHenyeyGreenstein(cosTheta, phase.x * gScale), vdbFrameHenyeyGreenstein(cosTheta, phase.y * gScale), phase.z);
  return mix(lobes, vdbFrameHenyeyGreenstein(cosTheta, gScale * ${MIE_ASYMMETRY}), min(phase.w, 1.0));
}
`,
    },
  ],
  [
    "shadow",
    {
      deps: [],
      code: `// frameShadowAbsorption: the extinction the albedo (Rec. 709 luminance) does not scatter
fn vdbFrameShadowAbsorption(extinction: f32, albedo: vec3f) -> f32 {
  return extinction * (1.0 - clamp(dot(albedo, vec3f(0.2126, 0.7152, 0.0722)), 0.0, 1.0));
}
// frameShadowStep: (extinction, effective extinction, reduced scattering) per unit of length
fn vdbFrameShadowStep(extinction: f32, absorption: f32, multipleScattering: f32) -> vec3f {
  let absorbed = min(absorption, extinction);
  let reducedScattering = (extinction - absorbed) * multipleScattering;
  return vec3f(extinction, sqrt(absorbed * 3.0 * (absorbed + reducedScattering)), reducedScattering);
}
// frameShadowDepth: -log of the total transmittance (1D diffusion when multipleScattering > 0)
fn vdbFrameShadowDepth(depths: vec3f, multipleScattering: f32) -> f32 {
  return select(depths.x, min(depths.x, depths.y + log(depths.z * 0.75 + 1.0)), multipleScattering > 0.0);
}
// K6.3 frameShadowMarchDepth: the depth that stops a march; octaves (count, a, b, c): the last octave's sigma a^N depth
fn vdbFrameShadowMarchDepth(depths: vec3f, multipleScattering: f32, octaves: vec4f) -> f32 {
  if (octaves.x > 0.0) {
    return depths.x * pow(octaves.y, octaves.x);
  }
  return vdbFrameShadowDepth(depths, multipleScattering);
}
// frameShadowTransmittance: (direct, total); octaves: total holds 1 / (1 + optical depth) (K6.3)
fn vdbFrameShadowTransmittance(depths: vec3f, multipleScattering: f32, octaves: vec4f) -> vec2f {
  if (octaves.x > 0.0) {
    return vec2f(exp(-depths.x), 1.0 / (depths.x + 1.0));
  }
  return vec2f(exp(-depths.x), exp(-vdbFrameShadowDepth(depths, multipleScattering)));
}
// frameIsotropicDiffusion: two-sided slab total of isotropic light (white furnace), z0 = 2/3
fn vdbFrameIsotropicDiffusion(frontReduced: f32, backReduced: f32, frontEffective: f32) -> f32 {
  return (backReduced + 2.0 / 3.0) * 2.0 / (frontReduced + backReduced + 4.0 / 3.0) * exp(-frontEffective);
}
// K6.3 frameTwoSidedDiffusion ('two-sided'): 1D slab diffusion with a collimated source between the light side (front
// reduced depth f) and the far side (back b), the lowest side mode k^2 = 2 (PI / (f + b + 2 z0))^2, Robin boundaries
// z0 = 2/3; the diffuse plus the reduced beam exp(-f), times exp(-effective front), never below the direct light
fn vdbFrameTwoSidedDiffusion(direct: f32, frontReduced: f32, frontEffective: f32, backReduced: f32) -> f32 {
  let z0 = 2.0 / 3.0;
  let slab = frontReduced + backReduced;
  let k0 = 4.442882938158366 / (slab + 2.0 * z0);
  let k = k0 + (1.0 - step(0.01, abs(1.0 - k0 * k0))) * 0.01;
  let p = -3.0 / (1.0 - k * k);
  let kl = k * slab;
  let coshKl = (exp(kl) + exp(-kl)) * 0.5;
  let sinhKl = (exp(kl) - exp(-kl)) * 0.5;
  let c12 = k * -z0;
  let r1 = p * -(1.0 + z0);
  let c21 = coshKl + k * z0 * sinhKl;
  let c22 = sinhKl + k * z0 * coshKl;
  let r2 = -p * exp(-slab) * (1.0 - z0);
  let det = c22 - c12 * c21;
  let c = (r1 * c22 - c12 * r2) / det;
  let d = (r2 - c21 * r1) / det;
  let kf = k * frontReduced;
  let beam = exp(-frontReduced);
  let diffuse = p * beam + c * (exp(kf) + exp(-kf)) * 0.5 + d * (exp(kf) - exp(-kf)) * 0.5;
  return max(direct, (diffuse + beam) * exp(-frontEffective));
}
`,
    },
  ],
  [
    "octaveTotal",
    {
      deps: ["shadow"],
      code: `// K6.3 multipleScattering 'octaves' (count N, extinction a, contribution b, phase c), isotropic light
// frameShadowOpticalDepth: the optical depth back from the octave shadow value 1 / (1 + depth)
fn vdbFrameShadowOpticalDepth(shadowY: f32) -> f32 {
  return max(1.0 / max(shadowY, 1e-6) - 1.0, 0.0);
}
// frameOctaveLight of isotropic light: sum over n = 1..N of b^n exp(-a^n depth)
fn vdbFrameOctaveSum(octaves: vec4f, opticalDepth: f32) -> f32 {
  var light = 0.0;
  var extinctionScale = 1.0;
  var contribution = 1.0;
  for (var i = 0u; i < ${MAX_OCTAVES}u; i++) {
    if (f32(i + 1u) > octaves.x) {
      break;
    }
    extinctionScale *= octaves.y;
    contribution *= octaves.z;
    light += contribution * exp(-opticalDepth * extinctionScale);
  }
  return light;
}
// frameShadowTotal: total transmittance of isotropic light from the y shadow value; octaves: exp(-depth) + the octaves
fn vdbFrameShadowTotal(shadowY: f32, octaves: vec4f) -> f32 {
  if (octaves.x > 0.0) {
    let depth = vdbFrameShadowOpticalDepth(shadowY);
    return exp(-depth) + vdbFrameOctaveSum(octaves, depth);
  }
  return shadowY;
}
`,
    },
  ],
  [
    "octaves",
    {
      deps: ["phase", "octaveTotal"],
      code: `// frameOctaveLight: sum over n = 1..N of b^n phase(g c^n) exp(-a^n depth)
fn vdbFrameOctaveLight(octaves: vec4f, opticalDepth: f32, phase: vec4f, cosTheta: f32) -> f32 {
  var light = 0.0;
  var extinctionScale = 1.0;
  var contribution = 1.0;
  var gScale = 1.0;
  for (var i = 0u; i < ${MAX_OCTAVES}u; i++) {
    if (f32(i + 1u) > octaves.x) {
      break;
    }
    extinctionScale *= octaves.y;
    contribution *= octaves.z;
    gScale *= octaves.w;
    light += contribution * vdbFramePhaseWeightScaled(phase, cosTheta, gScale) * exp(-opticalDepth * extinctionScale);
  }
  return light;
}
// frameShadowedLight of each channel: phase x direct + the scattered light (total - direct, isotropic), or with
// octaves their phased sum from the octave shadow value
fn vdbFrameShadowedLight(phase: vec4f, spectrum: vec3f, octaves: vec4f, cosTheta: f32, shadow: vec2f) -> vec3f {
  let direct = vdbFramePhaseColor(phase, spectrum, cosTheta) * shadow.x;
  if (octaves.x > 0.0) {
    return direct + vec3f(vdbFrameOctaveLight(octaves, vdbFrameShadowOpticalDepth(shadow.y), phase, cosTheta));
  }
  return direct + vec3f(shadow.y - shadow.x);
}
`,
    },
  ],
  [
    "sh",
    {
      deps: [],
      code: `// three's getShIrradianceAt: irradiance of the light probe SH9 along a unit normal
fn vdbFrameShIrradiance(normal: vec3f) -> vec3f {
  let x = normal.x;
  let y = normal.y;
  let z = normal.z;
  var result = vdbFrameLights.probe[0].xyz * 0.886227;
  result += vdbFrameLights.probe[1].xyz * (2.0 * 0.511664) * y;
  result += vdbFrameLights.probe[2].xyz * (2.0 * 0.511664) * z;
  result += vdbFrameLights.probe[3].xyz * (2.0 * 0.511664) * x;
  result += vdbFrameLights.probe[4].xyz * (2.0 * 0.429043) * x * y;
  result += vdbFrameLights.probe[5].xyz * (2.0 * 0.429043) * y * z;
  result += vdbFrameLights.probe[6].xyz * (z * z * 0.743125 - 0.247708);
  result += vdbFrameLights.probe[7].xyz * (2.0 * 0.429043) * x * z;
  result += vdbFrameLights.probe[8].xyz * 0.429043 * (x * x - y * y);
  return result;
}
`,
    },
  ],
  [
    "envSh",
    {
      deps: [],
      code: `// K6.2b: irradiance of the environment buffer's SH9 along a unit normal (three's getShIrradianceAt; the
// Ramamoorthi-Hanrahan cosine convolution), clamped at 0
fn vdbFrameEnvIrradiance(normal: vec3f) -> vec3f {
  let x = normal.x;
  let y = normal.y;
  let z = normal.z;
  var result = vdbFrameEnv.sh[0].xyz * 0.886227;
  result += vdbFrameEnv.sh[1].xyz * (2.0 * 0.511664) * y;
  result += vdbFrameEnv.sh[2].xyz * (2.0 * 0.511664) * z;
  result += vdbFrameEnv.sh[3].xyz * (2.0 * 0.511664) * x;
  result += vdbFrameEnv.sh[4].xyz * (2.0 * 0.429043) * x * y;
  result += vdbFrameEnv.sh[5].xyz * (2.0 * 0.429043) * y * z;
  result += vdbFrameEnv.sh[6].xyz * (z * z * 0.743125 - 0.247708);
  result += vdbFrameEnv.sh[7].xyz * (2.0 * 0.429043) * x * z;
  result += vdbFrameEnv.sh[8].xyz * 0.429043 * (x * x - y * y);
  return max(result, vec3f(0.0));
}
`,
    },
  ],
  [
    "grid",
    {
      deps: [],
      code: `// The world shadow grid (his createGridNodes): gridUvw, insideGrid (half a voxel in), findCascade
fn vdbFrameGridUvw(cascade: u32, world: vec3f) -> vec3f {
  return (world - vdbFrame.gridMin[cascade].xyz) / vdbFrame.gridSize[cascade].xyz;
}
fn vdbFrameInsideGrid(uvw: vec3f) -> bool {
  let inset = 0.5 / vdbFrame.gridLayout.y;
  return min(uvw.x, min(uvw.y, uvw.z)) >= inset && max(uvw.x, max(uvw.y, uvw.z)) <= 1.0 - inset;
}
// the finest cascade from first that holds the position, or -1
fn vdbFrameFindCascade(world: vec3f, first: u32) -> i32 {
  let count = u32(vdbFrame.gridLayout.x);
  for (var c = first; c < count; c++) {
    if (vdbFrameInsideGrid(vdbFrameGridUvw(c, world))) {
      return i32(c);
    }
  }
  return -1;
}
// sampleTiles on a stacked 3D texture: hardware trilinear in the R^3 sub-volume at origin, clamped to its
// half-texel box (his clamp to edge in a slice and between slices)
fn vdbFrameStackSample(stack: texture_3d<f32>, origin: vec3f, uvw: vec3f) -> vec4f {
  let r = vdbFrame.gridLayout.y;
  let p = origin + clamp(uvw * r, vec3f(0.5), vec3f(r - 0.5));
  return textureSampleLevel(stack, vdbFrameLinear, p / vec3f(textureDimensions(stack)), 0.0);
}
// his grid atlas pixel of a voxel (the jitter noise reads it): tile cascade * R + z of an atlas with gridLayout.z columns
fn vdbFrameGridFragCoord(cascade: u32, cell: vec3u) -> vec2f {
  let r = u32(vdbFrame.gridLayout.y);
  let columns = u32(vdbFrame.gridLayout.z);
  let tile = cascade * r + cell.z;
  return vec2f(f32((tile % columns) * r + cell.x), f32((tile / columns) * r + cell.y)) + 0.5;
}
`,
    },
  ],
  [
    "srgb",
    {
      deps: [],
      code: `// three's sRGBTransferOETF / sRGBTransferEOTF
fn vdbFrameSrgbOetf(color: vec3f) -> vec3f {
  let a = pow(color, vec3f(0.41666)) * 1.055 - 0.055;
  return select(a, color * 12.92, color <= vec3f(0.0031308));
}
fn vdbFrameSrgbEotf(color: vec3f) -> vec3f {
  let a = pow(color * 0.9478672986 + 0.0521327014, vec3f(2.4));
  return select(a, color * 0.0773993808, color <= vec3f(0.04045));
}
`,
    },
  ],
];

const BY_NAME = new Map(CHUNKS);

/** Names of the chunks {@link frameChunksWgsl} knows. */
export type VDBFrameChunk = "constants" | "mat3" | "sliceDepth" | "viewPosition" | "froxel" | "noise" | "fade" | "box" | "lod" | "phase" | "shadow" | "octaveTotal" | "octaves" | "sh" | "envSh" | "grid" | "srgb";

/** The closure of `roots` (with dependencies), in the fixed chunk order. */
export function frameChunksWgsl(roots: Iterable<VDBFrameChunk>): string {
  const want = new Set<string>();
  const visit = (name: string) => {
    if (want.has(name)) return;
    const chunk = BY_NAME.get(name);
    if (!chunk) throw new Error(`frameChunksWgsl: unknown chunk ${name}`);
    want.add(name);
    for (const d of chunk.deps) visit(d);
  };
  for (const r of roots) visit(r);
  return CHUNKS.filter(([name]) => want.has(name))
    .map(([, c]) => c.code)
    .join("\n");
}
