/**
 * Uniform structs of the core frame renderer (K6.1, design §3.1, §5.1): one field list per struct
 * generates both the WGSL struct text and the TypeScript packer, so the two cannot disagree (a unit
 * test checks the offsets against WGSL's uniform layout rules).
 *
 * Spectral (K6.4) adds fields to the reserved slots (`extinctionSpectrum`, the `reserved*` vectors)
 * without moving any offset.
 */

/** Field types the generator supports. Arrays take only 16-byte-aligned element types (uniform rule). */
export type VDBUniformType = "f32" | "u32" | "i32" | "vec2f" | "vec2u" | "vec3f" | "vec3u" | "vec4f" | "vec4u" | "mat4x4f";

/** One field: name, type, optional array length. */
export interface VDBUniformField {
  readonly name: string;
  readonly type: VDBUniformType;
  readonly count?: number;
}

/** A field with its byte offset. */
export interface VDBUniformSlot {
  readonly name: string;
  readonly type: VDBUniformType;
  readonly count: number;
  readonly offset: number;
  /** Bytes between array elements (the element size for one value). */
  readonly stride: number;
}

/** A generated struct: its WGSL, byte size and field offsets. */
export interface VDBUniformStruct {
  readonly name: string;
  readonly size: number;
  readonly fields: ReadonlyMap<string, VDBUniformSlot>;
  readonly wgsl: string;
}

const ALIGN: Record<VDBUniformType, number> = { f32: 4, u32: 4, i32: 4, vec2f: 8, vec2u: 8, vec3f: 16, vec3u: 16, vec4f: 16, vec4u: 16, mat4x4f: 16 };
const SIZE: Record<VDBUniformType, number> = { f32: 4, u32: 4, i32: 4, vec2f: 8, vec2u: 8, vec3f: 12, vec3u: 12, vec4f: 16, vec4u: 16, mat4x4f: 64 };
const INTEGER = new Set<VDBUniformType>(["u32", "i32", "vec2u", "vec3u", "vec4u"]);

const roundUp = (n: number, a: number) => Math.ceil(n / a) * a;

/**
 * Lays out `fields` by WGSL's uniform address-space rules and returns the struct.
 *
 * @param name The WGSL struct name.
 * @param fields The fields in order.
 * @param size A fixed byte size (padding added at the end; throws if the fields do not fit).
 */
export function vdbUniformStruct(name: string, fields: readonly VDBUniformField[], size?: number): VDBUniformStruct {
  const out = new Map<string, VDBUniformSlot>();
  const lines: string[] = [];
  let offset = 0;
  for (const f of fields) {
    const count = f.count ?? 1;
    if (f.count !== undefined && ALIGN[f.type] < 16) throw new Error(`${name}.${f.name}: uniform arrays need 16-byte elements`);
    const align = ALIGN[f.type];
    const stride = f.count !== undefined ? roundUp(SIZE[f.type], 16) : SIZE[f.type];
    offset = roundUp(offset, align);
    out.set(f.name, { name: f.name, type: f.type, count, offset, stride });
    lines.push(`  ${f.name}: ${f.count !== undefined ? `array<${f.type}, ${f.count}>` : f.type},`);
    offset += f.count !== undefined ? stride * count : SIZE[f.type];
  }
  let total = roundUp(offset, 16);
  if (size !== undefined) {
    if (total > size) throw new Error(`${name}: ${total} bytes of fields exceed the fixed ${size}`);
    let pad = 0;
    for (let at = total; at < size; at += 16) lines.push(`  pad${pad++}: vec4f,`);
    total = size;
  }
  return { name, size: total, fields: out, wgsl: `struct ${name} {\n${lines.join("\n")}\n};\n` };
}

/** A packer over one ArrayBuffer: writes fields of structs at a base offset. */
export class VDBUniformWriter {
  readonly buffer: ArrayBuffer;
  readonly f32: Float32Array;
  readonly u32: Uint32Array;
  readonly i32: Int32Array;

  constructor(bytes: number) {
    this.buffer = new ArrayBuffer(bytes);
    this.f32 = new Float32Array(this.buffer);
    this.u32 = new Uint32Array(this.buffer);
    this.i32 = new Int32Array(this.buffer);
  }

  /**
   * Writes `value` into field `name` of struct `s` at byte `base` (array fields: `value` holds all elements
   * back to back, each the type's component count; mat4x4f takes 16 numbers column-major).
   */
  set(s: VDBUniformStruct, base: number, name: string, value: number | ArrayLike<number>): void {
    const slot = s.fields.get(name);
    if (!slot) throw new Error(`${s.name} has no field ${name}`);
    const comps = SIZE[slot.type] / 4;
    const values = typeof value === "number" ? [value] : value;
    const view = slot.type === "i32" ? this.i32 : INTEGER.has(slot.type) ? this.u32 : this.f32;
    const elements = Math.min(slot.count, Math.ceil(values.length / comps));
    for (let e = 0; e < elements; e++) {
      const at = (base + slot.offset + e * slot.stride) / 4;
      for (let c = 0; c < comps; c++) {
        const v = values[e * comps + c];
        if (v !== undefined) view[at + c] = v;
      }
    }
  }

  /** Zeroes `bytes` from `base`. */
  clear(base: number, bytes: number): void {
    new Uint8Array(this.buffer, base, bytes).fill(0);
  }
}

/** Most visible volumes per update (64 KiB of 512-byte records; compat devices bind 16 KiB: 32). */
export const VDB_FRAME_MAX_VOLUMES = 128;
/** Bytes of one volume record. */
export const VDB_FRAME_VOLUME_BYTES = 512;
/** Dynamic-offset stride of the group records (`minUniformBufferOffsetAlignment` ≤ 256). */
export const VDB_FRAME_GROUP_STRIDE = 256;
/** Most scene lights (his `frameMaxLights`). */
export const VDB_FRAME_MAX_LIGHTS = 16;
/** Most hemisphere lights (his `frameMaxHemisphereLights`). */
export const VDB_FRAME_MAX_HEMISPHERE_LIGHTS = 4;
/** Most shadow grid cascades (his `frameMaxShadowCascades`). */
export const VDB_FRAME_MAX_CASCADES = 4;
/** Light slots of the 'frame' transmittance (his `frameShadowLightSlots`). */
export const VDB_FRAME_SHADOW_SLOTS = 6;

/** Group 0 binding 0: camera, froxel layout, areas, jitter, fade, temporal and composite values. */
export const VDB_FRAME_UNIFORMS = vdbUniformStruct("VdbFrameUniforms", [
  // view → world; the inverse projection reconstructs the froxel rays and the composite's view depth
  { name: "cameraWorld", type: "mat4x4f" },
  { name: "projectionInverse", type: "mat4x4f" },
  // temporal: the previous frame's world → view, projection and froxel range
  { name: "previousView", type: "mat4x4f" },
  { name: "previousProjection", type: "mat4x4f" },
  // W, H, D and his atlas columns (the noise's virtual atlas coordinate)
  { name: "grid", type: "vec4f" },
  // near, far of the slices; previous near, far
  { name: "range", type: "vec4f" },
  // this frame's froxel area: rect cells (x0, y0, x1, y1), slices (first, end)
  { name: "areaRect", type: "vec4f" },
  { name: "areaSlices", type: "vec4f" },
  // the dispatch box: origin x, y and size w, h in cells (the union of this area and what was written before)
  { name: "dispatchBox", type: "vec4u" },
  // jitter (< 0: off), fade radius (0: off), temporal blend, use history (0/1)
  { name: "jitterFade", type: "vec4f" },
  // sub-steps, temporal on, composite flags (bit 0 reversed depth, bit 1 orthographic), spare
  { name: "flags", type: "vec4u" },
  // composite: screen size in pixels, clamp colours (0/1), spare
  { name: "screen", type: "vec4f" },
  // composite: the camera's near and far (three's depth → view Z), spare
  { name: "camera", type: "vec4f" },
  // K6.3: octaves (count, a, b, c); reserved for K6.3/K6.4 (two-sided, spectral)
  { name: "octaves", type: "vec4f" },
  { name: "reserved", type: "vec4f" },
  // K6.2 'frame' lighting: the world shadow grid cascades (his createGridUniforms; xyz, w unused)
  { name: "gridMin", type: "vec4f", count: VDB_FRAME_MAX_CASCADES },
  { name: "gridSize", type: "vec4f", count: VDB_FRAME_MAX_CASCADES },
  { name: "gridVoxel", type: "vec4f", count: VDB_FRAME_MAX_CASCADES },
  // cascade count, resolution R (voxels per side), his grid atlas columns (the noise coordinate), scene shadow slots
  { name: "gridLayout", type: "vec4f" },
  // texel origin of cascade c in the shadow grid stack (a stacked 3D texture, design §3.2)
  { name: "shadowOrigin", type: "vec4f", count: VDB_FRAME_MAX_CASCADES },
  // texel origin of transmittance target k of cascade c, at [4c + k], in the transmittance stack
  { name: "transmittanceOrigin", type: "vec4f", count: 4 * VDB_FRAME_MAX_CASCADES },
  // multiple scattering (1 − g or 0), two-sided (0/1; K6.3), use shadows (0/1), diffusion (0/1)
  { name: "lighting", type: "vec4f" },
  // the renderer's 'frame' phase (forward g, back g, back weight, Lorenz-Mie mode) and its spectrum (red, blue g scale, flag)
  { name: "framePhase", type: "vec4f" },
  { name: "framePhaseSpectrum", type: "vec4f" },
]);

/** Group 0 binding 1: the scene lights (his light uniforms), in slot order directional, point, spot. */
export const VDB_FRAME_LIGHTS = vdbUniformStruct("VdbFrameLights", [
  // ambient rgb, hemisphere count
  { name: "ambient", type: "vec4f" },
  // scene light count, spare
  { name: "counts", type: "vec4u" },
  // light probe SH9 (rgb in xyz)
  { name: "probe", type: "vec4f", count: 9 },
  { name: "hemiSky", type: "vec4f", count: VDB_FRAME_MAX_HEMISPHERE_LIGHTS },
  { name: "hemiGround", type: "vec4f", count: VDB_FRAME_MAX_HEMISPHERE_LIGHTS },
  { name: "hemiUp", type: "vec4f", count: VDB_FRAME_MAX_HEMISPHERE_LIGHTS },
  // his records: position (w 1 directional: xyz toward the light, 2 point, 3 spot), colour (w penumbra cos),
  // spot (xyz target → light, w cone cos), falloff (distance, decay)
  { name: "position", type: "vec4f", count: VDB_FRAME_MAX_LIGHTS },
  { name: "color", type: "vec4f", count: VDB_FRAME_MAX_LIGHTS },
  { name: "spot", type: "vec4f", count: VDB_FRAME_MAX_LIGHTS },
  { name: "falloff", type: "vec4f", count: VDB_FRAME_MAX_LIGHTS },
  // K6.2 'frame' lighting: the sky direction (his frameSkyDirection: the first hemisphere light's up, else +y) and the
  // sums of every hemisphere light's sky and ground colours (his frameHemiSkyColor / frameHemiGroundColor)
  { name: "sky", type: "vec4f" },
  { name: "hemiSkySum", type: "vec4f" },
  { name: "hemiGroundSum", type: "vec4f" },
]);

/** One volume record (512 B) in the `VdbFrameVolumes` array (group 1). */
export const VDB_FRAME_VOLUME = vdbUniformStruct(
  "VdbFrameVolume",
  [
    { name: "viewToLocal", type: "mat4x4f" },
    { name: "localToView", type: "mat4x4f" },
    { name: "worldToLocal", type: "mat4x4f" },
    // the volume's froxel rect (cells) and slices
    { name: "rect", type: "vec4f" },
    // first slice, slice count, source slot v, lights bitmask
    { name: "slices", type: "vec4u" },
    // scatter colour, opacity
    { name: "scatterOpacity", type: "vec4f" },
    // base colour (albedo), extinction
    { name: "baseExtinction", type: "vec4f" },
    // offset3D, wrap (0 clamp, 1 repeat, 2 mirror) as f32
    { name: "offsetWrap", type: "vec4f" },
    // (forward g, back g, back weight, Lorenz-Mie mode)
    { name: "phase", type: "vec4f" },
    // (red g scale, blue g scale, flag), multiple scattering (1 − g, or 0)
    { name: "phaseSpectrumMs", type: "vec4f" },
    // densityScale (normalized), densityCutoff, worldScale, absorbance
    { name: "density", type: "vec4f" },
    // emissiveIntensity, temperature min, max, lod levels the read may use (mipmaps ? levels : 0)
    { name: "emission", type: "vec4f" },
    // shadow steps, mipmaps (0/1), hdda (0/1), spare
    { name: "march", type: "vec4u" },
    // K6.4: extinction spectrum (rgb), spare
    { name: "extinctionSpectrum", type: "vec4f" },
  ],
  VDB_FRAME_VOLUME_BYTES,
);

/** One source group's record (group 1, dynamic offset, {@link VDB_FRAME_GROUP_STRIDE}). */
export const VDB_FRAME_GROUP = vdbUniformStruct("VdbFrameGroup", [
  // first volume record, count, role (bit 0 first group, bit 1 last group), group index
  { name: "range", type: "vec4u" },
  // K6.2: the group's shadow casters (first record, count, role among the caster groups, caster group index)
  { name: "casters", type: "vec4u" },
]);

/** The WGSL of the frame structs and the group-0 bindings every frame kernel declares. */
export function vdbFrameUniformsWgsl(maxVolumes = VDB_FRAME_MAX_VOLUMES): string {
  return `${VDB_FRAME_UNIFORMS.wgsl}${VDB_FRAME_LIGHTS.wgsl}${VDB_FRAME_VOLUME.wgsl}struct VdbFrameVolumes {
  v: array<VdbFrameVolume, ${maxVolumes}>,
};
${VDB_FRAME_GROUP.wgsl}@group(0) @binding(0) var<uniform> vdbFrame: VdbFrameUniforms;
@group(0) @binding(1) var<uniform> vdbFrameLights: VdbFrameLights;
@group(0) @binding(2) var vdbFrameLinear: sampler;
`;
}
