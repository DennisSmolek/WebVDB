// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L838–877 (sampleMipAtlas / sampleMipmapped, here over the
//   packed pyramid atlas) and L914–936 (sampleExtinction's density read).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: the WGSL the TSL sparse source (`three/sparse-source.ts`, K3.2–K3.2d, G16a) wraps in `wgslFn`s, lifted
// into raw WGSL over a per-volume uniform table (one source serves every volume of a pool) and one packed r16float
// pyramid atlas. Divergences: DIVERGES.md (K3.2 rows; K6.7 row).

/**
 * The WGSL and the uniform record of the core sparse source (K6.7, `vdbSparseSourceGpu`). Every `vdbSrc*` function
 * takes `v: u32`, the volume's source-local slot, which indexes `vdbSrcVolumes`. The maths is the TSL source's
 * (`three/sparse-source.ts`): the same sample position, the same three LOD strategies and the same HDDA rules.
 */

import type { VDBSlot } from "../types.js";

/** One field of the per-volume record: a name and a 16-byte WGSL type. */
interface RecordField {
  readonly name: string;
  readonly type: "vec4u" | "vec4f";
  readonly doc: string;
}

/**
 * The per-volume record (`VdbSrcVolume`), one field list for the WGSL struct and the TS packer. Every field is a
 * 16-byte vector, so offsets are `index × 16` and the struct has no padding.
 */
export const VDB_SPARSE_RECORD_FIELDS: readonly RecordField[] = [
  { name: "ids", type: "vec4u", doc: "pool ids: density, mask, emission; w: flags (1 mask, 2 albedo, 4 albedo rgb, 8 emission)" },
  { name: "albedo", type: "vec4u", doc: "albedo pool ids r, g, b (grey: three times the same id)" },
  { name: "lod", type: "vec4f", doc: "x: levels above level 0, y: pyramid base level L0, z: pyramid atlas levels, w: density stats max" },
  { name: "densityMin", type: "vec4f", doc: "density index box min (xyz)" },
  { name: "densitySize", type: "vec4f", doc: "density index box size in voxels (xyz)" },
  { name: "maskMin", type: "vec4f", doc: "mask index box min" },
  { name: "maskSize", type: "vec4f", doc: "mask index box size" },
  { name: "albedoMin", type: "vec4f", doc: "albedo index box min" },
  { name: "albedoSize", type: "vec4f", doc: "albedo index box size" },
  { name: "emissionMin", type: "vec4f", doc: "emission index box min" },
  { name: "emissionSize", type: "vec4f", doc: "emission index box size" },
  { name: "pyrBase", type: "vec4f", doc: "pyramid base origin in the packed atlas (texels)" },
  { name: "pyrSize", type: "vec4f", doc: "pyramid base size (texels)" },
  { name: "pyrMip", type: "vec4f", doc: "xyz: mip atlas origin in the packed atlas; w: his atlas width" },
];

/** Bytes of one `VdbSrcVolume`. */
export const VDB_SPARSE_RECORD_BYTES = VDB_SPARSE_RECORD_FIELDS.length * 16;

/** Byte offset of a record field. */
export function sparseRecordOffset(name: string): number {
  const i = VDB_SPARSE_RECORD_FIELDS.findIndex((f) => f.name === name);
  if (i < 0) throw new Error(`sparse record: no field ${name}`);
  return i * 16;
}

/** Record flags (`ids.w`). */
export const SPARSE_FLAG = { mask: 1, albedo: 2, albedoRgb: 4, emission: 8 } as const;

/** The values of one record (see {@link VDB_SPARSE_RECORD_FIELDS}). */
export interface SparseRecord {
  ids: readonly [number, number, number, number];
  albedo: readonly [number, number, number, number];
  lod: readonly [number, number, number, number];
  densityMin: readonly number[];
  densitySize: readonly number[];
  maskMin: readonly number[];
  maskSize: readonly number[];
  albedoMin: readonly number[];
  albedoSize: readonly number[];
  emissionMin: readonly number[];
  emissionSize: readonly number[];
  pyrBase: readonly number[];
  pyrSize: readonly number[];
  pyrMip: readonly number[];
}

/** Writes `record` at byte `offset` of `bytes` (the packer of the generated struct). */
export function packSparseRecord(bytes: ArrayBuffer, offset: number, record: SparseRecord): void {
  const u = new Uint32Array(bytes, offset, VDB_SPARSE_RECORD_BYTES / 4);
  const f = new Float32Array(bytes, offset, VDB_SPARSE_RECORD_BYTES / 4);
  VDB_SPARSE_RECORD_FIELDS.forEach((field, i) => {
    const value = record[field.name as keyof SparseRecord];
    for (let c = 0; c < 4; c++) {
      const x = value[c] ?? 0;
      if (field.type === "vec4u") u[i * 4 + c] = x >>> 0;
      else f[i * 4 + c] = x;
    }
  });
}

/** The generated struct text. */
export function sparseRecordStructWgsl(): string {
  return `struct VdbSrcVolume {\n${VDB_SPARSE_RECORD_FIELDS.map((f) => `    ${f.name}: ${f.type}, // ${f.doc}`).join("\n")}\n}`;
}

/** What the WGSL depends on (the source's compile-time configuration). */
export interface SparseWgslConfig {
  readonly slot: VDBSlot;
  readonly profile: "sample" | "march";
  readonly lod: "none" | "coarse-pyramid" | "node-average";
  readonly channels: { readonly mask: boolean; readonly albedo: boolean; readonly emission: boolean };
  readonly fastTrilinear: boolean;
  readonly cacheAccessor: boolean;
  readonly clamp01: boolean;
  readonly skipEmpty: boolean;
  readonly maxVolumes: number;
}

/** Bindings after the pool's (which is at `slot.firstBinding`, declared by the VDB module). */
export function sparseBindings(slot: VDBSlot, lod: SparseWgslConfig["lod"]): { table: number; pyramid: number | null; sampler: number | null } {
  const b = slot.firstBinding;
  return lod === "coarse-pyramid" ? { table: b + 1, pyramid: b + 2, sampler: b + 3 } : { table: b + 1, pyramid: null, sampler: null };
}

const R = "vdbSrcVolumes[v]";

/** The `vdbSrc*` WGSL (without the VDB module, which must precede it in the same shader). */
export function sparseSourceWgsl(c: SparseWgslConfig): string {
  const { slot } = c;
  const bind = sparseBindings(slot, c.lod);
  const out: string[] = [
    `// vdbSparseSourceGpu (K6.7): ${c.lod}, ${c.maxVolumes} volumes. Needs the VDB module at group ${slot.group}, binding ${slot.firstBinding}.`,
    sparseRecordStructWgsl(),
    `@group(${slot.group}) @binding(${bind.table}) var<uniform> vdbSrcVolumes: array<VdbSrcVolume, ${c.maxVolumes}>;`,
  ];
  if (bind.pyramid !== null) {
    out.push(`@group(${slot.group}) @binding(${bind.pyramid}) var vdbSrcPyramid: texture_3d<f32>;`);
    out.push(`@group(${slot.group}) @binding(${bind.sampler}) var vdbSrcSampler: sampler;`);
  }

  // uvw → sample index position, clamped to the voxels (his clamp-to-edge); voxel centres at integers.
  out.push(`
fn vdbSrcSamplePos(bmin: vec3f, bsize: vec3f, uvw: vec3f) -> vec3f {
    return clamp(bmin + uvw * bsize - 0.5, bmin, bmin + bsize - 1.0);
}`);

  // Level 0 (K3.2d): vdbSampleFast and/or one accessor per invocation kept in a private variable (copied in and
  // out: WGSL can't pass a ptr<private> where the API takes ptr<function>). Same values either way.
  const call = c.fastTrilinear ? "vdbSampleFast" : "vdbSample";
  out.push(
    c.cacheAccessor
      ? `
var<private> vdbSrcAcc: VdbAccessor;
var<private> vdbSrcAccGrid: u32 = 0xffffffffu;
fn vdbSrcLevel0(g: u32, xyz: vec3f) -> f32 {
    if vdbSrcAccGrid != g {
        vdbSrcAcc = vdbAccessorInit(g);
        vdbSrcAccGrid = g;
    }
    var a = vdbSrcAcc;
    let value = ${call}(&a, xyz);
    vdbSrcAcc = a;
    return value;
}`
      : `
fn vdbSrcLevel0(g: u32, xyz: vec3f) -> f32 {
    var a = vdbAccessorInit(g);
    return ${call}(&a, xyz);
}`,
  );

  if (c.lod === "node-average") {
    out.push(`
fn vdbSrcNodeAverage(g: u32, p: vec3f, dim: u32) -> f32 {
    var acc = vdbAccessorInit(g);
    let fd = f32(dim);
    let q = (p - vec3f(0.5 * fd - 0.5)) / fd;
    let base = floor(q);
    let w = q - base;
    let c = vec3i(base) * i32(dim);
    // One call site in a loop: SwiftShader compiles 8 inlined tree walks very slowly.
    var n: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) {
        n[k] = vdbNodeAverage(&acc, c + vec3i(vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u)) * i32(dim), dim);
    }
    let x00 = n[0] + w.x * (n[1] - n[0]);
    let x10 = n[2] + w.x * (n[3] - n[2]);
    let x01 = n[4] + w.x * (n[5] - n[4]);
    let x11 = n[6] + w.x * (n[7] - n[6]);
    let y0 = x00 + w.y * (x10 - x00);
    let y1 = x01 + w.y * (x11 - x01);
    return y0 + w.z * (y1 - y0);
}
// Level 0 -> 3 (leaf averages) -> 7 (lower-node averages), two taps at most; one level-0 call site.
fn vdbSrcNodeLod(g: u32, p: vec3f, lod: f32) -> f32 {
    let fine = lod < 3.0;
    var a = 0.0;
    if fine {
        a = vdbSrcLevel0(g, p);
    } else {
        a = vdbSrcNodeAverage(g, p, 8u);
    }
    if lod <= 0.0 {
        return a;
    }
    if lod >= 7.0 {
        return vdbSrcNodeAverage(g, p, 128u);
    }
    let b = vdbSrcNodeAverage(g, p, select(128u, 8u, fine));
    let t = select((lod - 3.0) / 4.0, lod / 3.0, fine);
    return mix(a, b, t);
}`);
  }

  if (c.lod === "coarse-pyramid") {
    // His sampleMipAtlas / sampleMipmapped over the volume's blocks of the packed atlas. Every tap clamps to its
    // level's half-texel box, so a block never reads its neighbours.
    out.push(`
fn vdbSrcPyrTap(texel: vec3f) -> f32 {
    return textureSampleLevel(vdbSrcPyramid, vdbSrcSampler, texel / vec3f(textureDimensions(vdbSrcPyramid)), 0.0).r;
}
fn vdbSrcPyrBase(v: u32, uvw: vec3f) -> f32 {
    let size = ${R}.pyrSize.xyz;
    return vdbSrcPyrTap(clamp(uvw * size, vec3f(0.5), size - 0.5) + ${R}.pyrBase.xyz);
}
fn vdbSrcPyrMipAtlas(v: u32, level: f32, uvw: vec3f) -> f32 {
    let atlasWidth = ${R}.pyrMip.w;
    let size = max(vec3f(1.0), ceil(${R}.pyrSize.xyz / exp2(level)));
    let offset = atlasWidth - atlasWidth / exp2(level - 1.0);
    let texel = clamp(uvw * size, vec3f(0.5), size - 0.5);
    return vdbSrcPyrTap(vec3f(texel.x + offset, texel.y, texel.z) + ${R}.pyrMip.xyz);
}
// lod relative to the base level L0.
fn vdbSrcPyrSample(v: u32, uvw: vec3f, lod: f32) -> f32 {
    let levels = ${R}.lod.z;
    let level = min(lod, levels);
    if level <= 0.0 {
        return vdbSrcPyrBase(v, uvw);
    }
    let level0 = floor(level);
    var low = 0.0;
    if level0 < 0.5 {
        low = vdbSrcPyrBase(v, uvw);
    } else {
        low = vdbSrcPyrMipAtlas(v, level0, uvw);
    }
    let high = vdbSrcPyrMipAtlas(v, min(level0 + 1.0, levels), uvw);
    return mix(low, high, level - level0);
}`);
  }

  const norm = (e: string) => (c.clamp01 ? `clamp(${e}, 0.0, 1.0)` : e);
  let density: string;
  if (c.lod === "none") {
    density = `    return ${norm("vdbSrcLevel0(g, p)")};`;
  } else if (c.lod === "node-average") {
    density = `    let l = clamp(lod, 0.0, ${R}.lod.x);
    return ${norm("vdbSrcNodeLod(g, p, l)")};`;
  } else {
    // lod in (0, L0) blends level 0 toward the pyramid base; lod >= L0 samples the pyramid. One level-0 call site.
    density = `    let l = clamp(lod, 0.0, ${R}.lod.x);
    let L0 = ${R}.lod.y;
    var fine = 0.0;
    if l < L0 {
        fine = vdbSrcLevel0(g, p);
    }
    var r = 0.0;
    if l <= 0.0 {
        r = fine;
    } else if l < L0 {
        r = mix(fine, vdbSrcPyrSample(v, uvw, 0.0), l / L0);
    } else {
        r = vdbSrcPyrSample(v, uvw, l - L0);
    }
    return ${norm("r")};`;
  }
  const { mask, albedo, emission } = c.channels;
  out.push(`
fn vdbSrcDensity(v: u32, uvw: vec3f, lod: f32) -> f32 {
    let g = ${R}.ids.x;
    let p = vdbSrcSamplePos(${R}.densityMin.xyz, ${R}.densitySize.xyz, uvw);
${density}
}
fn vdbSrcMask(v: u32, uvw: vec3f, lod: f32) -> f32 {
${
  mask
    ? `    if (${R}.ids.w & ${SPARSE_FLAG.mask}u) == 0u {
        return 1.0;
    }
    return vdbSrcLevel0(${R}.ids.y, vdbSrcSamplePos(${R}.maskMin.xyz, ${R}.maskSize.xyz, uvw));`
    : "    return 1.0;"
}
}
fn vdbSrcAlbedo(v: u32, uvw: vec3f, lod: f32) -> vec3f {
${
  albedo
    ? `    let flags = ${R}.ids.w;
    if (flags & ${SPARSE_FLAG.albedo}u) == 0u {
        return vec3f(1.0);
    }
    let p = vdbSrcSamplePos(${R}.albedoMin.xyz, ${R}.albedoSize.xyz, uvw);
    let ids = ${R}.albedo;
    let n = select(1u, 3u, (flags & ${SPARSE_FLAG.albedoRgb}u) != 0u);
    var rgb = vec3f(0.0);
    for (var k = 0u; k < n; k++) {
        rgb[k] = vdbSrcLevel0(ids[k], p);
    }
    if n == 1u {
        rgb = vec3f(rgb.x);
    }
    return clamp(rgb, vec3f(0.0), vec3f(1.0));`
    : "    return vec3f(1.0);"
}
}
fn vdbSrcEmission(v: u32, uvw: vec3f, lod: f32) -> f32 {
${
  emission
    ? `    if (${R}.ids.w & ${SPARSE_FLAG.emission}u) == 0u {
        return 0.0;
    }
    return vdbSrcLevel0(${R}.ids.z, vdbSrcSamplePos(${R}.emissionMin.xyz, ${R}.emissionSize.xyz, uvw));`
    : "    return 0.0;"
}
}
fn vdbSrcTexels(v: u32) -> vec3f {
    return ${R}.densitySize.xyz;
}
fn vdbSrcLodLevels(v: u32) -> f32 {
    return ${R}.lod.x;
}`);

  if (c.profile === "march") {
    if (!c.skipEmpty) {
      out.push(`
fn vdbSrcSkipSpan(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32, lod: f32, cutoff: f32) -> vec2f {
    return vec2f(t0, 1e30);
}`);
    } else {
      let rules = "";
      if (c.lod === "node-average") {
        rules = `
    let l = clamp(lod, 0.0, ${R}.lod.x);
    if l > 0.0 {
        dilate = 1u;
        if l > 3.0 {
            enabled = false;
        }
    }`;
      } else if (c.lod === "coarse-pyramid") {
        rules = `
    let l = clamp(lod, 0.0, ${R}.lod.x);
    if l > 0.0 {
        dilate = 1u;
        if ${R}.lod.y > 2.0 || l > 2.0 {
            enabled = false;
        }
    }`;
      }
      out.push(`
// The HDDA empty-space query (K3.2; twin tools vdbFrameSkipEmpty): over 8^3 cells of o + d t in index space.
fn vdbSrcSkipCells(g: u32, o: vec3f, d: vec3f, t0: f32, t1: f32, cutoff: f32, dilate: u32) -> vec2f {
    var acc = vdbAccessorInit(g);
    var h: VdbHdda;
    vdbHddaInit(&h, o, t0, d, t1, 8u);
    let back = 1e-3 / max(length(d), 1e-6);
    var ts = t0;
    for (var n = 0u; n < 64u; n++) {
        let te = vdbHddaCellEnd(&h);
        var b = 0.0;
        if dilate == 0u {
            b = vdbCellBoundTrilinear(&acc, h.voxel, 8u);
        } else {
            b = -1e30;
            for (var k = 0u; k < 27u; k++) {
                let c = vec3i(i32(k % 3u) - 1, i32((k / 3u) % 3u) - 1, i32(k / 9u) - 1) * 8;
                b = max(b, vdbNodeMax(&acc, h.voxel + c, 8u));
            }
        }
        if b > cutoff {
            return vec2f(max(t0, ts - back), te);
        }
        ts = te;
        if !vdbHddaStep(&h) {
            return vec2f(t1, t1);
        }
    }
    let r = max(t0, ts - back);
    return vec2f(r, r);
}
// The march o + d t is in uvw (the source box [0, 1]^3, no offset or wrap: callers that offset or wrap must not
// skip) and its samples are read at lod. Returns (ts, te): every sample with t < ts is exactly 0, the samples in
// [ts, te) must be read, ask again from te; (t0, 1e30) when no exact bound exists (coarse levels past the
// dilated block) or lod > 3 (the K3.2 cost rule). cutoff: the caller's density cutoff (0 with a mask).
fn vdbSrcSkipSpan(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32, lod: f32, cutoff: f32) -> vec2f {
    var enabled = lod <= 3.0;
    var dilate = 0u;${rules}
    if !enabled {
        return vec2f(t0, 1e30);
    }
    let bmin = ${R}.densityMin.xyz;
    let bsize = ${R}.densitySize.xyz;
    var c = max(cutoff, 0.0);${
      mask
        ? `
    if (${R}.ids.w & ${SPARSE_FLAG.mask}u) != 0u {
        c = 0.0;
    }`
        : ""
    }
    return vdbSrcSkipCells(${R}.ids.x, bmin - 0.5 + o * bsize, d * bsize, t0, t1, c, dilate);
}`);
    }
    out.push(`
// The frozen group-2 form: exact for level-0 samples with any density cutoff >= 0.
fn vdbSrcSkipEmpty(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32) -> f32 {
    return vdbSrcSkipSpan(v, o, d, t0, t1, 0.0, 0.0).x;
}
// Conservative: the volume's density stats max (the box is ignored).
fn vdbSrcMaxIn(v: u32, uvwMin: vec3f, uvwMax: vec3f) -> f32 {
    return ${R}.lod.w;
}`);
  }
  return out.join("\n") + "\n";
}
