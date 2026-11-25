// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L94–107 (frameSliceDepth, frameGetViewPosition),
//   L109–127 (frameGetFroxel), L129–142 (frameDistanceFade), L157–185 (getFroxelLod, getBoxSteps,
//   frameNoise), L187–203 (frameHenyeyGreenstein, frameHenyeyGreensteinColor), L273–315 (framePhaseWeight,
//   framePhaseColor), L330–373 (frameShadowAbsorption, frameShadowStep, frameShadowDepth,
//   frameShadowTransmittance), L403–414 (frameIsotropicDiffusion), L938–958 (boxExit, boxSegment);
//   three.js r187 getShIrradianceAt (MIT).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TSL → WGSL for the core compute kernels (K6.1); the froxel of a 3D texel replaces the atlas
// pixel (the noise keeps his virtual atlas coordinate); octaves, Lorenz-Mie and spectral chunks wait for
// K6.3/K6.4. CPU twins: tools/frame (index.ts, froxel.ts). Divergences: DIVERGES.md (K3.1, K6.1 rows).

/**
 * `frameChunksWgsl(roots)`: the source-free `vdbFrame*` WGSL functions the frame kernels share, emitted
 * as the dependency closure of `roots` in one fixed order (same roots, same bytes). They read the group-0
 * uniforms `vdbFrame` / `vdbFrameLights` (`vdbFrameUniformsWgsl`).
 */

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
      deps: [],
      code: `// frameHenyeyGreenstein: HG over the isotropic phase; g = 0 gives 1
fn vdbFrameHenyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
}
fn vdbFrameHenyeyGreensteinColor(cosTheta: f32, g: vec3f) -> vec3f {
  let g2 = g * g;
  return (vec3f(1.0) - g2) / pow(max(vec3f(1.0) + g2 - g * 2.0 * cosTheta, vec3f(1e-4)), vec3f(1.5));
}
// framePhaseWeight: (forward g, back g, back weight, Lorenz-Mie mode); Lorenz-Mie: K6.3
fn vdbFramePhaseWeight(phase: vec4f, cosTheta: f32) -> f32 {
  return mix(vdbFrameHenyeyGreenstein(cosTheta, phase.x), vdbFrameHenyeyGreenstein(cosTheta, phase.y), phase.z);
}
// framePhaseColor: spectrum (red g scale, blue g scale, flag); flag 0: one phase for all channels
fn vdbFramePhaseColor(phase: vec4f, spectrum: vec3f, cosTheta: f32) -> vec3f {
  if (spectrum.z < 0.5) {
    return vec3f(vdbFramePhaseWeight(phase, cosTheta));
  }
  let scale = vec3f(spectrum.x, 1.0, spectrum.y);
  return mix(
    vdbFrameHenyeyGreensteinColor(cosTheta, clamp(scale * phase.x, vec3f(-0.99), vec3f(0.99))),
    vdbFrameHenyeyGreensteinColor(cosTheta, clamp(scale * phase.y, vec3f(-0.99), vec3f(0.99))),
    vec3f(phase.z),
  );
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
// frameShadowTransmittance: (direct, total); octaves: K6.3
fn vdbFrameShadowTransmittance(depths: vec3f, multipleScattering: f32) -> vec2f {
  return vec2f(exp(-depths.x), exp(-vdbFrameShadowDepth(depths, multipleScattering)));
}
// frameIsotropicDiffusion: two-sided slab total of isotropic light (white furnace), z0 = 2/3
fn vdbFrameIsotropicDiffusion(frontReduced: f32, backReduced: f32, frontEffective: f32) -> f32 {
  return (backReduced + 2.0 / 3.0) * 2.0 / (frontReduced + backReduced + 4.0 / 3.0) * exp(-frontEffective);
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
export type VDBFrameChunk = "constants" | "mat3" | "sliceDepth" | "viewPosition" | "froxel" | "noise" | "fade" | "box" | "lod" | "phase" | "shadow" | "sh" | "srgb";

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
