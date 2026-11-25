/**
 * Demo 17's own passes: everything that is not a volume. A tiny forward renderer in raw WebGPU, so the page shows
 * exactly what a host gives the core and what it reads back:
 *
 * - `sky`: the equirect sky (the environment the core is given) and the sun's disc, as the background.
 * - `meshes`: the ground, an obelisk and a plinth (one vertex buffer), Lambert-lit by the sun and the sky. A mesh is a
 *   receiver of both volume outputs: the sun term is multiplied by the volumes' Beer shadow map (`vdbVolumeShadowWgsl`,
 *   `outputs.volumeShadows`), and the volumes' emission adds irradiance from the probes (`vdbEmissionProbeWgsl`,
 *   `outputs.emissionProbes`). Meshes also shadow each other through the sun's depth map.
 * - `shadow`: that depth map, drawn from the sun (meshes, and the level set through a second level-set pass with the
 *   sun's orthographic camera). The core takes it as an occluder (`VDBFrameInput.occluders`): mesh → volume shadows.
 * - `levelSetResolve`: shades the level-set pass's G-buffer (world normal, ray distance) with the same lighting.
 *
 * All four share group 0 (the {@link SceneUniforms}) and group 1 (the lighting inputs), so one WGSL `shade()` lights
 * every surface.
 */
import { VDB_EMISSION_GRID, VDB_VOLUME_SHADOW_VIEW, vdbEmissionProbeWgsl, vdbVolumeShadowWgsl } from "../../../../packages/vdb/src/core/internal.js";
import type { Vec3 } from "./raw-frame.js";

// ------------------------------------------------------------------------------------------------ uniforms

/** Group 0 binding 0 (std140-like, 16-byte fields): see `SCENE_WGSL`. */
export const SCENE_FLOATS = 4 * 16 + 12 * 4;

export interface SceneUniforms {
  viewProjection: ArrayLike<number>;
  projectionInverse: ArrayLike<number>;
  cameraWorld: ArrayLike<number>;
  sunViewProjection: ArrayLike<number>;
  cameraPosition: Vec3;
  cameraForward: Vec3;
  /** The way the sunlight travels (unit). */
  sunDirection: Vec3;
  /** 1 / shadow map size. */
  shadowTexel: number;
  /** Colour × intensity. */
  sunColor: Vec3;
  /** Scales the emission probes' irradiance (his emission light intensity). */
  emissionIntensity: number;
  skyAmbient: Vec3;
  /** Scales the sky background (the environment intensity). */
  environmentIntensity: number;
  groundAmbient: Vec3;
  /** Depth-map bias (NDC depth units). */
  shadowBias: number;
  fogColor: Vec3;
  /** 1 / fog distance (exponential). */
  fogDensity: number;
  sunDiscColor: Vec3;
  /** cos of the sun disc's angular radius. */
  sunDiscCos: number;
  levelSetColor: Vec3;
  /** World units the receiver moves along its normal before the depth-map test. */
  normalOffset: number;
  /** Receivers read the volume shadow map. */
  volumeShadows: boolean;
  /** Meshes cast on meshes. */
  meshShadows: boolean;
  /** Receivers read the emission probes. */
  emission: boolean;
  viewport: [number, number];
}

export function packScene(u: SceneUniforms): ArrayBuffer {
  const buffer = new ArrayBuffer(SCENE_FLOATS * 4);
  const f = new Float32Array(buffer);
  const n = new Uint32Array(buffer);
  f.set(Array.from(u.viewProjection), 0);
  f.set(Array.from(u.projectionInverse), 16);
  f.set(Array.from(u.cameraWorld), 32);
  f.set(Array.from(u.sunViewProjection), 48);
  let o = 64;
  const v4 = (a: readonly number[], w: number) => {
    f.set([a[0]!, a[1]!, a[2]!, w], o);
    o += 4;
  };
  v4(u.cameraPosition, 0);
  v4(u.cameraForward, 0);
  v4(u.sunDirection, u.shadowTexel);
  v4(u.sunColor, u.emissionIntensity);
  v4(u.skyAmbient, u.environmentIntensity);
  v4(u.groundAmbient, u.shadowBias);
  v4(u.fogColor, u.fogDensity);
  v4(u.sunDiscColor, u.sunDiscCos);
  v4(u.levelSetColor, u.normalOffset);
  n.set([u.volumeShadows ? 1 : 0, u.meshShadows ? 1 : 0, u.emission ? 1 : 0, 0], o);
  o += 4;
  f.set([u.viewport[0], u.viewport[1], 0, 0], o);
  return buffer;
}

const SCENE_WGSL = /* wgsl */ `
struct Scene {
  viewProjection: mat4x4f,
  projectionInverse: mat4x4f,
  cameraWorld: mat4x4f,
  sunViewProjection: mat4x4f,
  cameraPosition: vec4f,
  cameraForward: vec4f,
  sunDirection: vec4f,   // w: 1 / shadow map size
  sunColor: vec4f,       // w: emission light intensity
  skyAmbient: vec4f,     // w: environment intensity (background)
  groundAmbient: vec4f,  // w: depth-map bias
  fog: vec4f,            // w: 1 / fog distance
  sunDisc: vec4f,        // w: cos of the disc radius
  levelSetColor: vec4f,  // w: normal offset
  flags: vec4u,          // x: volume shadows, y: mesh shadows, z: emission
  viewport: vec4f,
};
@group(0) @binding(0) var<uniform> scene: Scene;
`;

/** Group 1: the lighting inputs every surface pass reads. */
const LIGHTING_BINDINGS_WGSL = /* wgsl */ `
@group(1) @binding(0) var meshShadowMap: texture_depth_2d;
@group(1) @binding(1) var meshShadowSampler: sampler_comparison;
@group(1) @binding(2) var volumeShadowMap: texture_2d_array<f32>;
@group(1) @binding(3) var<uniform> volumeShadowView: VdbVolumeShadowView;
@group(1) @binding(4) var linearSampler: sampler;
@group(1) @binding(5) var emissionProbes: texture_3d<f32>;
@group(1) @binding(6) var<uniform> emissionGrid: VdbEmissionGrid;
@group(1) @binding(7) var skyMap: texture_2d<f32>;
@group(1) @binding(8) var skySampler: sampler;
`;

/** One `shade()` for every surface: sun (mesh shadow × volume shadow) + sky hemisphere + the volumes' emission. */
const SHADE_WGSL = /* wgsl */ `
const PI = 3.141592653589793;

// The sun's depth map at a world point: 3 × 3 PCF, 1 outside the map
fn meshShadow(world: vec3f, n: vec3f) -> f32 {
  if (scene.flags.y == 0u) {
    return 1.0;
  }
  let p = scene.sunViewProjection * vec4f(world + n * scene.levelSetColor.w, 1.0);
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z > 1.0) {
    return 1.0;
  }
  let reference = p.z - scene.groundAmbient.w;
  let texel = scene.sunDirection.w;
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      sum += textureSampleCompareLevel(meshShadowMap, meshShadowSampler, uv + vec2f(f32(x), f32(y)) * texel, reference);
    }
  }
  return sum / 9.0;
}

fn shade(world: vec3f, n: vec3f, albedo: vec3f) -> vec3f {
  let toLight = -scene.sunDirection.xyz;
  let ndl = max(dot(n, toLight), 0.0);
  var sun = scene.sunColor.rgb * ndl;
  if (ndl > 0.0) {
    sun *= meshShadow(world, n);
    if (scene.flags.x != 0u) {
      // volume -> mesh: the core's Beer map, read with the view it was drawn with
      let viewDepth = dot(world - scene.cameraPosition.xyz, scene.cameraForward.xyz);
      sun *= vdbVolumeShadow(volumeShadowMap, linearSampler, volumeShadowView, world, viewDepth);
    }
  }
  let sky = mix(scene.groundAmbient.rgb, scene.skyAmbient.rgb, n.y * 0.5 + 0.5);
  var emitted = vec3f(0.0);
  if (scene.flags.z != 0u) {
    // the volumes' emission as irradiance at this point (a Lambert surface: albedo × E / π)
    emitted = vdbEmissionIrradiance(emissionProbes, linearSampler, emissionGrid, world, n) * (scene.sunColor.w / PI);
  }
  return albedo * (sun + sky + emitted);
}

fn fogged(color: vec3f, world: vec3f) -> vec3f {
  let d = length(world - scene.cameraPosition.xyz);
  return mix(scene.fog.rgb, color, exp(-d * scene.fog.w));
}

// The pixel-centre view ray (the level-set pass's vdbLevelSetViewRay, perspective)
fn viewRay(position: vec2f) -> vec3f {
  let uv = position / scene.viewport.xy;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let clip = scene.projectionInverse * vec4f(ndc, 0.5, 1.0);
  let view = clip.xyz / clip.w;
  return normalize((scene.cameraWorld * vec4f(normalize(view), 0.0)).xyz);
}
`;

// The core's receiver WGSL declares its structs (VdbVolumeShadowView, VdbEmissionGrid) and the two functions
const HEADER_WGSL = `${SCENE_WGSL}${LIGHTING_BINDINGS_WGSL}${vdbVolumeShadowWgsl()}${vdbEmissionProbeWgsl()}${SHADE_WGSL}`;

const FULLSCREEN_VS = /* wgsl */ `
@vertex
fn fullscreenVs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(corner * 2.0 - 1.0, 1.0, 1.0);
}
`;

const SKY_WGSL = /* wgsl */ `${HEADER_WGSL}${FULLSCREEN_VS}
@fragment
fn skyFs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let d = viewRay(position.xy);
  let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
  let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
  var color = textureSampleLevel(skyMap, skySampler, vec2f(u, v), 0.0).rgb * scene.skyAmbient.w;
  if (dot(d, -scene.sunDirection.xyz) > scene.sunDisc.w) {
    color += scene.sunDisc.rgb;
  }
  if (d.y < 0.0) {
    color = mix(color, scene.fog.rgb, 1.0);
  }
  return vec4f(color, 1.0);
}
`;

const MESH_WGSL = /* wgsl */ `${HEADER_WGSL}
struct MeshOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec4f,
};

@vertex
fn meshVs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) albedo: vec4f) -> MeshOut {
  var o: MeshOut;
  o.position = scene.viewProjection * vec4f(p, 1.0);
  o.world = p;
  o.normal = n;
  o.albedo = albedo;
  return o;
}

// albedo.w: 1 = the ground's procedural tiles
fn groundAlbedo(base: vec3f, world: vec3f) -> vec3f {
  let cell = floor(world.xz / 40.0);
  let checker = (i32(cell.x) + i32(cell.y)) & 1;
  let line = min(fract(world.xz / 40.0), vec2f(1.0) - fract(world.xz / 40.0));
  let edge = 1.0 - 0.18 * (1.0 - smoothstep(0.0, 0.015, min(line.x, line.y)));
  return base * select(1.0, 0.86, checker == 1) * edge;
}

@fragment
fn meshFs(i: MeshOut) -> @location(0) vec4f {
  let n = normalize(i.normal);
  var albedo = i.albedo.rgb;
  if (i.albedo.w > 0.5) {
    albedo = groundAlbedo(albedo, i.world);
  }
  return vec4f(fogged(shade(i.world, n, albedo), i.world), 1.0);
}

// The sun's depth map (the colour target only matches the level-set pass sharing the render pass)
@vertex
fn shadowVs(@location(0) p: vec3f) -> @builtin(position) vec4f {
  return scene.sunViewProjection * vec4f(p, 1.0);
}
@fragment
fn shadowFs() -> @location(0) vec4f {
  return vec4f(0.0);
}
`;

const RESOLVE_WGSL = /* wgsl */ `${HEADER_WGSL}${FULLSCREEN_VS}
@group(2) @binding(0) var levelSetGbuffer: texture_2d<f32>;

// The level-set pass wrote (world normal, ray distance) where its surface won the depth test
@fragment
fn resolveFs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let g = textureLoad(levelSetGbuffer, vec2i(position.xy), 0);
  if (!(g.w > 0.0)) {
    discard;
  }
  let world = scene.cameraPosition.xyz + viewRay(position.xy) * g.w;
  return vec4f(fogged(shade(world, normalize(g.xyz), scene.levelSetColor.rgb), world), 1.0);
}
`;

// ------------------------------------------------------------------------------------------------ geometry

/** Vertex: position (3), normal (3), albedo (3) + pattern (1). */
export const VERTEX_FLOATS = 10;

export interface Box {
  center: Vec3;
  size: Vec3;
  albedo: Vec3;
}

/** The ground quad and boxes as one triangle list. */
export function meshVertices(groundHalf: number, groundAlbedo: Vec3, boxes: readonly Box[]): Float32Array<ArrayBuffer> {
  const out: number[] = [];
  const quad = (a: Vec3, b: Vec3, c: Vec3, d: Vec3, n: Vec3, albedo: Vec3, pattern: number) => {
    for (const p of [a, b, c, a, c, d]) out.push(...p, ...n, ...albedo, pattern);
  };
  const g = groundHalf;
  quad([-g, 0, -g], [-g, 0, g], [g, 0, g], [g, 0, -g], [0, 1, 0], groundAlbedo, 1);
  for (const b of boxes) {
    const [cx, cy, cz] = b.center;
    const [hx, hy, hz] = b.size.map((s) => s / 2) as Vec3;
    const x0 = cx - hx, x1 = cx + hx, y0 = cy - hy, y1 = cy + hy, z0 = cz - hz, z1 = cz + hz;
    quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], [1, 0, 0], b.albedo, 0);
    quad([x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0], [-1, 0, 0], b.albedo, 0);
    quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [0, 1, 0], b.albedo, 0);
    quad([x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [0, -1, 0], b.albedo, 0);
    quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], b.albedo, 0);
    quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], b.albedo, 0);
  }
  return new Float32Array(out);
}

// ------------------------------------------------------------------------------------------------ pipelines

export interface ScenePipelines {
  sky: GPURenderPipeline;
  mesh: GPURenderPipeline;
  shadow: GPURenderPipeline;
  resolve: GPURenderPipeline;
  group0: GPUBindGroupLayout;
  group1: GPUBindGroupLayout;
  group2: GPUBindGroupLayout;
}

/** Builds the four pipelines (async, inside a validation scope). `color` is the HDR format, `depth` the depth format. */
export async function createScenePipelines(device: GPUDevice, color: GPUTextureFormat, depth: GPUTextureFormat, shadowColor: GPUTextureFormat): Promise<ScenePipelines> {
  const F = GPUShaderStage.FRAGMENT;
  const V = GPUShaderStage.VERTEX;
  const group0 = device.createBindGroupLayout({ label: "demo-17 scene", entries: [{ binding: 0, visibility: V | F, buffer: { type: "uniform" } }] });
  const group1 = device.createBindGroupLayout({
    label: "demo-17 lighting",
    entries: [
      { binding: 0, visibility: F, texture: { sampleType: "depth" } },
      { binding: 1, visibility: F, sampler: { type: "comparison" } },
      { binding: 2, visibility: F, texture: { sampleType: "float", viewDimension: "2d-array" } },
      { binding: 3, visibility: F, buffer: { type: "uniform" } },
      { binding: 4, visibility: F, sampler: { type: "filtering" } },
      { binding: 5, visibility: F, texture: { sampleType: "float", viewDimension: "3d" } },
      { binding: 6, visibility: F, buffer: { type: "uniform" } },
      { binding: 7, visibility: F, texture: { sampleType: "float" } },
      { binding: 8, visibility: F, sampler: { type: "filtering" } },
    ],
  });
  const group2 = device.createBindGroupLayout({ label: "demo-17 level-set gbuffer", entries: [{ binding: 0, visibility: F, texture: { sampleType: "unfilterable-float" } }] });
  const layout1 = device.createPipelineLayout({ bindGroupLayouts: [group0] });
  const layout2 = device.createPipelineLayout({ bindGroupLayouts: [group0, group1] });
  const layout3 = device.createPipelineLayout({ bindGroupLayouts: [group0, group1, group2] });
  const vertexBuffers: GPUVertexBufferLayout[] = [
    {
      arrayStride: VERTEX_FLOATS * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x4" },
      ],
    },
  ];
  device.pushErrorScope("validation");
  const skyModule = device.createShaderModule({ label: "demo-17 sky", code: SKY_WGSL });
  const meshModule = device.createShaderModule({ label: "demo-17 meshes", code: MESH_WGSL });
  const resolveModule = device.createShaderModule({ label: "demo-17 level-set resolve", code: RESOLVE_WGSL });
  const [sky, mesh, shadow, resolve] = await Promise.all([
    device.createRenderPipelineAsync({
      label: "demo-17 sky",
      layout: layout2,
      vertex: { module: skyModule, entryPoint: "fullscreenVs" },
      fragment: { module: skyModule, entryPoint: "skyFs", targets: [{ format: color }] },
      depthStencil: { format: depth, depthWriteEnabled: false, depthCompare: "always" },
    }),
    device.createRenderPipelineAsync({
      label: "demo-17 meshes",
      layout: layout2,
      vertex: { module: meshModule, entryPoint: "meshVs", buffers: vertexBuffers },
      fragment: { module: meshModule, entryPoint: "meshFs", targets: [{ format: color }] },
      primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
      depthStencil: { format: depth, depthWriteEnabled: true, depthCompare: "less" },
    }),
    device.createRenderPipelineAsync({
      label: "demo-17 sun depth map",
      layout: layout1,
      vertex: { module: meshModule, entryPoint: "shadowVs", buffers: [{ arrayStride: VERTEX_FLOATS * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
      fragment: { module: meshModule, entryPoint: "shadowFs", targets: [{ format: shadowColor, writeMask: 0 }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: depth, depthWriteEnabled: true, depthCompare: "less" },
    }),
    device.createRenderPipelineAsync({
      label: "demo-17 level-set resolve",
      layout: layout3,
      vertex: { module: resolveModule, entryPoint: "fullscreenVs" },
      fragment: { module: resolveModule, entryPoint: "resolveFs", targets: [{ format: color }] },
    }),
  ]);
  const error = await device.popErrorScope();
  if (error) throw new Error(`demo-17 scene pipelines: ${error.message}`);
  return { sky, mesh, shadow, resolve, group0, group1, group2 };
}

/** Placeholders for the lighting inputs before the core has made them (a disabled map reads 1, an empty grid 0). */
export interface LightingPlaceholders {
  volumeShadowMap: GPUTexture;
  volumeShadowView: GPUBuffer;
  emissionProbes: GPUTexture;
  emissionGrid: GPUBuffer;
}

export function createPlaceholders(device: GPUDevice): LightingPlaceholders {
  return {
    volumeShadowMap: device.createTexture({ label: "demo-17 no volume shadow", size: [1, 1, 1], format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING }),
    volumeShadowView: device.createBuffer({ label: "demo-17 no volume shadow view", size: VDB_VOLUME_SHADOW_VIEW.size, usage: GPUBufferUsage.UNIFORM }),
    emissionProbes: device.createTexture({ label: "demo-17 no emission", size: [1, 1, 1], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING }),
    emissionGrid: device.createBuffer({ label: "demo-17 no emission grid", size: VDB_EMISSION_GRID.size, usage: GPUBufferUsage.UNIFORM }),
  };
}
