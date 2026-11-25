/**
 * K6.5b: mesh → volume shadows (design §3.7). The host's depth shadow maps (`VDBFrameInput.occluders`) are bound in
 * four fixed group-1 slots of the kernels that light volumes (the 'frame' transmittance and gather, the 'volume'
 * gather) as `texture_2d_array<f32>` read with `textureLoad`: a depth texture binds as `unfilterable-float` like an
 * `r32float` one, so one binding type takes both, reversed depth is a uniform flag, and the 2 × 2 percentage-closer
 * filter computes exact bilinear weights (a comparison sampler would need the `depth` sample type, a second sampler
 * for reversed depth, and quantizes its weights on hardware). Empty slots hold a 1 × 1 placeholder; `VdbFrameLights`
 * carries the count (`counts.y`), each light's slot (`falloff.z` = slot + 1) and the maps' layers, bias, flags and
 * per-layer view-projections. Volumetric fog in Unreal and Frostbite samples the light's shadow map per froxel the same
 * way (Hillaire 2015). CPU twin: tools/frame/occluders.ts `vdbFrameOccluderVisibility`.
 */

/**
 * The first group-1 binding of the occluder slots in every kernel that reads them (12–15: clear of the kernels' own
 * bindings, so other passes can add theirs below).
 */
export const VDB_FRAME_OCCLUDER_BINDING = 12;

/** The four occluder bindings of a kernel's group 1, from `first`, and the visibility functions. */
export function vdbFrameOccluderWgsl(first: number = VDB_FRAME_OCCLUDER_BINDING): string {
  const bindings = [0, 1, 2, 3].map((k) => `@group(1) @binding(${first + k}) var vdbFrameOccluder${k}: texture_2d_array<f32>;`).join("\n");
  return `${bindings}

// 1 when the reference depth is in front of the stored one (lit), else 0
fn vdbFrameOccluderTest(map: texture_2d_array<f32>, texel: vec2i, layer: u32, maxTexel: vec2i, reference: f32, reversed: bool) -> f32 {
  let stored = textureLoad(map, clamp(texel, vec2i(0), maxTexel), layer, 0).x;
  return select(select(0.0, 1.0, reference <= stored), select(0.0, 1.0, reference >= stored), reversed);
}

// One map at a world position: the first layer whose clip volume holds it, the host's bias, 2 x 2 PCF with bilinear
// weights (or one texel); 1 outside every layer (three's frustum rule)
fn vdbFrameOccluderMapVisibility(map: texture_2d_array<f32>, slot: u32, world: vec3f) -> f32 {
  let info = vdbFrameLights.occluderInfo[slot];
  let layers = min(u32(info.x), 6u);
  let dims = vec2i(textureDimensions(map));
  let size = vec2f(dims);
  let maxTexel = dims - vec2i(1);
  let reversed = info.z > 0.5;
  for (var layer = 0u; layer < layers; layer++) {
    let clip = vdbFrameLights.occluderMatrix[slot * 6u + layer] * vec4f(world, 1.0);
    if (!(clip.w > 0.0)) {
      continue;
    }
    let ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
      continue;
    }
    let reference = select(ndc.z + info.y, ndc.z - info.y, reversed);
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (info.w > 0.5) {
      return vdbFrameOccluderTest(map, vec2i(floor(uv * size)), layer, maxTexel, reference, reversed);
    }
    let p = uv * size - 0.5;
    let base = floor(p);
    let f = p - base;
    let b = vec2i(base);
    let l00 = vdbFrameOccluderTest(map, b, layer, maxTexel, reference, reversed);
    let l10 = vdbFrameOccluderTest(map, b + vec2i(1, 0), layer, maxTexel, reference, reversed);
    let l01 = vdbFrameOccluderTest(map, b + vec2i(0, 1), layer, maxTexel, reference, reversed);
    let l11 = vdbFrameOccluderTest(map, b + vec2i(1, 1), layer, maxTexel, reference, reversed);
    let top = l00 + (l10 - l00) * f.x;
    let bottom = l01 + (l11 - l01) * f.x;
    return top + (bottom - top) * f.y;
  }
  return 1.0;
}

// The visibility of a world position from scene light 'light' (its index in VdbFrameLights): 1 without an occluder map
// (or with occlusion off), else the map's test. A march toward the light stops at the first occluder, so this is its
// transmittance factor.
fn vdbFrameOccluderVisibility(world: vec3f, light: u32) -> f32 {
  let slot = u32(vdbFrameLights.falloff[light].z);
  if (slot == 0u || slot > vdbFrameLights.counts.y) {
    return 1.0;
  }
  switch (slot) {
    case 1u: {
      return vdbFrameOccluderMapVisibility(vdbFrameOccluder0, 0u, world);
    }
    case 2u: {
      return vdbFrameOccluderMapVisibility(vdbFrameOccluder1, 1u, world);
    }
    case 3u: {
      return vdbFrameOccluderMapVisibility(vdbFrameOccluder2, 2u, world);
    }
    default: {
      return vdbFrameOccluderMapVisibility(vdbFrameOccluder3, 3u, world);
    }
  }
}
`;
}

/** The group-1 layout entries of the four occluder slots from `first`. */
export function vdbFrameOccluderLayoutEntries(visibility: GPUShaderStageFlags, first: number = VDB_FRAME_OCCLUDER_BINDING): GPUBindGroupLayoutEntry[] {
  return [0, 1, 2, 3].map((k) => ({ binding: first + k, visibility, texture: { sampleType: "unfilterable-float" as const, viewDimension: "2d-array" as const } }));
}
