// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L834–1028 (createVolumeNodes: sampleMipAtlas,
//   sampleMipmapped, sample3D, sampleOptional), src/three-openvdb/frame/core/FrameVolume.js L14–22 and
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L483–487 (texture setup: linear filter, clamp to edge,
//   no GPU mipmaps), src/three-openvdb/frame/webgpu/frameMipmaps.js L183–189 (mip atlas texture).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TSL → WGSL, raw WebGPU upload; mip level count from textureDimensions instead of a uniform.
// Divergences: DIVERGES.md V23–V25.

/**
 * `vdbDenseSource(device, volume)`: the dense `.wvdb` volume source on raw
 * WebGPU (K3.0, R1 §4.2). It uploads the bytes `wvdbDenseVolume` (tools) built,
 * as his renderer does: one 3D texture per channel (`r8unorm` density, mask and
 * emission; `rgba8unorm` albedo) plus a CPU box mip atlas per channel, sampled
 * with hardware trilinear (linear min/mag, clamp to edge, no GPU mipmaps).
 *
 * D10 says kernels need no textures; that holds for the sparse source. A
 * kernel compiled against a dense source binds textures, by the host's choice.
 */

import { VDBError } from "../errors.js";
import type { VDBSlot } from "../types.js";
import type { WvdbDenseTexture, WvdbDenseVolume } from "../../tools/wvdb/dense.js";
import type { VDBVolumeChannels, VDBVolumeSource, VDBVolumeSourceWgslOptions } from "./types.js";

/** `GPUTextureUsage.TEXTURE_BINDING | COPY_DST` (0x04 | 0x02); the globals do not exist in Node. */
const TEXTURE_USAGE = 0x04 | 0x02;
/** `GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE`. */
const STAGE_FRAGMENT_COMPUTE = 0x2 | 0x4;

type ChannelName = "density" | "mask" | "albedo" | "emission";
const CHANNELS: readonly ChannelName[] = ["density", "mask", "albedo", "emission"];
const CAP: Record<ChannelName, string> = { density: "Density", mask: "Mask", albedo: "Albedo", emission: "Emission" };

/** Options of {@link vdbDenseSource}. */
export interface VDBDenseSourceOptions {
  /** Label prefix of the GPU objects. */
  label?: string;
}

interface Uploaded {
  map: GPUTexture;
  mip: GPUTexture | null;
}

const bad = (message: string) => new VDBError("VDB_BAD_OPTION", message);

function upload(device: GPUDevice, data: Uint8Array<ArrayBuffer>, dims: readonly number[], format: GPUTextureFormat, texelBytes: number, label: string): GPUTexture {
  const [w, h, d] = dims as [number, number, number];
  if (data.byteLength !== w * h * d * texelBytes) {
    throw bad(`vdbDenseSource: ${label} has ${data.byteLength} bytes, expected ${w}x${h}x${d}x${texelBytes}`);
  }
  const texture = device.createTexture({ label, size: [w, h, d], dimension: "3d", format, usage: TEXTURE_USAGE });
  // writeTexture has no 256-byte row alignment rule (only buffer copies do); FrameVolume uses unpackAlignment 1.
  device.queue.writeTexture({ texture }, data, { bytesPerRow: w * texelBytes, rowsPerImage: h }, [w, h, d]);
  return texture;
}

function uploadChannel(device: GPUDevice, tex: WvdbDenseTexture, label: string): Uploaded {
  const map = upload(device, tex.data, tex.dimensions, tex.format, tex.texelBytes, `${label}`);
  const mip = tex.mip ? upload(device, tex.mip.data, tex.mip.dimensions, tex.format, tex.texelBytes, `${label} mip atlas`) : null;
  return { map, mip };
}

function present(volume: WvdbDenseVolume): Record<ChannelName, WvdbDenseTexture | null> {
  return { density: volume.density, mask: volume.mask, albedo: volume.albedo, emission: volume.emission };
}

function checkSlot(slot: VDBSlot, where: string): VDBSlot {
  if (!slot || !Number.isInteger(slot.group) || slot.group < 0 || !Number.isInteger(slot.firstBinding) || slot.firstBinding < 0) {
    throw bad(`${where}: slot must be { group, firstBinding } with non-negative integers, got ${JSON.stringify(slot)}.`);
  }
  return slot;
}

/** WGSL for one channel's sample function (his sampleMipmapped / sampleMipAtlas, L838–877). */
function channelWgsl(name: ChannelName, hasMip: boolean): string {
  const C = CAP[name];
  const map = `vdbSrc${C}Map`;
  if (!hasMip) {
    return `fn vdbSrcSample${C}(uvw: vec3f, lod: f32) -> vec4f {\n  return textureSampleLevel(${map}, vdbSrcSampler, uvw, 0.0);\n}\n`;
  }
  const mip = `vdbSrc${C}Mip`;
  return `fn vdbSrcMipAtlas${C}(level: f32, uvw: vec3f) -> vec4f {
  let texels = vec3f(textureDimensions(${map}));
  let atlas = vec3f(textureDimensions(${mip}));
  let size = max(vec3f(1.0), ceil(texels / exp2(level)));
  let offset = atlas.x - atlas.x / exp2(level - 1.0);
  let texel = clamp(uvw * size, vec3f(0.5), size - 0.5);
  return textureSampleLevel(${mip}, vdbSrcSampler, vec3f(texel.x + offset, texel.y, texel.z) / atlas, 0.0);
}
fn vdbSrcSample${C}(uvw: vec3f, lod: f32) -> vec4f {
  let level = min(lod, vdbSrcLevelCount(textureDimensions(${map})));
  if (level <= 0.0) {
    return textureSampleLevel(${map}, vdbSrcSampler, uvw, 0.0);
  }
  let level0 = floor(level);
  var low: vec4f;
  if (level0 < 0.5) {
    low = textureSampleLevel(${map}, vdbSrcSampler, uvw, 0.0);
  } else {
    low = vdbSrcMipAtlas${C}(level0, uvw);
  }
  let high = vdbSrcMipAtlas${C}(min(level0 + 1.0, vdbSrcLevelCount(textureDimensions(${map}))), uvw);
  return mix(low, high, level - level0);
}
`;
}

class DenseSource implements VDBVolumeSource {
  readonly kind = "dense" as const;
  texels: [number, number, number];
  lodLevels: number;
  readonly channels: VDBVolumeChannels;
  stats: { meanNonZero: number; max: number; activeVoxels: number };
  /** Local unit box → uvw: translate by 0.5 (column-major). */
  readonly boxToSource = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0.5, 0.5, 1]);
  version = 0;
  readonly #device: GPUDevice;
  readonly #label: string;
  #volume: WvdbDenseVolume;
  #textures: Partial<Record<ChannelName, Uploaded>> = {};
  #sampler: GPUSampler;
  #disposed = false;

  constructor(device: GPUDevice, volume: WvdbDenseVolume, o: VDBDenseSourceOptions) {
    this.#device = device;
    this.#label = o.label ?? "vdb dense source";
    this.#volume = volume;
    this.channels = { mask: !!volume.mask, albedo: !!volume.albedo, emission: !!volume.emission };
    // FrameRenderer.js L483–487 + Data3DTexture defaults: linear, clamp to edge, no mipmaps.
    this.#sampler = device.createSampler({
      label: `${this.#label} sampler`,
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "nearest",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge",
    });
    this.texels = [...volume.dimensions] as [number, number, number];
    this.lodLevels = volume.density.mip?.levels ?? 0;
    this.stats = { meanNonZero: volume.meanDensity, max: volume.maxDensity, activeVoxels: volume.activeTexels };
    this.#upload(volume);
  }

  #upload(volume: WvdbDenseVolume): void {
    const textures = present(volume);
    for (const name of CHANNELS) {
      const tex = textures[name];
      if (tex) this.#textures[name] = uploadChannel(this.#device, tex, `${this.#label} ${name}`);
    }
  }

  #destroyTextures(): void {
    for (const t of Object.values(this.#textures)) {
      t.map.destroy();
      t.mip?.destroy();
    }
    this.#textures = {};
  }

  #check(): void {
    if (this.#disposed) throw new VDBError("VDB_DISPOSED", "vdbDenseSource: the source was disposed; create a new one.");
  }

  /** The textures as uploaded (for tests and adapters). */
  get volume(): WvdbDenseVolume {
    return this.#volume;
  }

  /**
   * Swap in another LOD (or frame) of the same grids. The channel set and mip
   * presence must match; a size change makes new textures and bumps `version`
   * (rebuild bind groups; the WGSL does not change).
   */
  replace(volume: WvdbDenseVolume): void {
    this.#check();
    const now = present(this.#volume);
    const next = present(volume);
    for (const name of CHANNELS) {
      if (!!now[name] !== !!next[name] || !!now[name]?.mip !== !!next[name]?.mip) {
        throw bad(`vdbDenseSource.replace: the ${name} channel or its mip atlas appears or disappears; create a new source (the WGSL differs).`);
      }
    }
    const sameSize = (a: WvdbDenseTexture | null, b: WvdbDenseTexture | null) =>
      !a || !b || (a.dimensions.every((v, i) => v === b.dimensions[i]) && (!a.mip || a.mip.dimensions.every((v, i) => v === b.mip!.dimensions[i])));
    if (CHANNELS.every((n) => sameSize(now[n], next[n]))) {
      for (const name of CHANNELS) {
        const tex = next[name];
        const up = this.#textures[name];
        if (!tex || !up) continue;
        const [w, h, d] = tex.dimensions;
        this.#device.queue.writeTexture({ texture: up.map }, tex.data, { bytesPerRow: w * tex.texelBytes, rowsPerImage: h }, [w, h, d]);
        if (tex.mip && up.mip) {
          const [aw, ah, ad] = tex.mip.dimensions;
          this.#device.queue.writeTexture({ texture: up.mip }, tex.mip.data, { bytesPerRow: aw * tex.texelBytes, rowsPerImage: ah }, [aw, ah, ad]);
        }
      }
    } else {
      this.#destroyTextures();
      this.#upload(volume);
      this.version++;
    }
    this.#volume = volume;
    this.texels = [...volume.dimensions] as [number, number, number];
    this.lodLevels = volume.density.mip?.levels ?? 0;
    this.stats = { meanNonZero: volume.meanDensity, max: volume.maxDensity, activeVoxels: volume.activeTexels };
  }

  /** Bindings in order: sampler, then per present channel its texture and its mip atlas. */
  #bindings(slot: VDBSlot): { name: string; binding: number; kind: "sampler" | "texture"; texture?: GPUTexture }[] {
    let b = slot.firstBinding;
    const out: { name: string; binding: number; kind: "sampler" | "texture"; texture?: GPUTexture }[] = [
      { name: "vdbSrcSampler", binding: b++, kind: "sampler" },
    ];
    for (const name of CHANNELS) {
      const t = this.#textures[name];
      if (!t) continue;
      out.push({ name: `vdbSrc${CAP[name]}Map`, binding: b++, kind: "texture", texture: t.map });
      if (t.mip) out.push({ name: `vdbSrc${CAP[name]}Mip`, binding: b++, kind: "texture", texture: t.mip });
    }
    return out;
  }

  wgsl(o: VDBVolumeSourceWgslOptions): string {
    this.#check();
    const slot = checkSlot(o.slot, "vdbDenseSource.wgsl");
    if (o.profile !== "sample" && o.profile !== "march") throw bad(`vdbDenseSource.wgsl: profile must be 'sample' or 'march', got ${JSON.stringify(o.profile)}.`);
    const on = (name: Exclude<ChannelName, "density">) => this.channels[name] && o.channels?.[name] !== false;
    const lines: string[] = [`// vdbDenseSource: dense .wvdb volume source (K3.0). ${this.#label}`];
    for (const b of this.#bindings(slot)) {
      lines.push(`@group(${slot.group}) @binding(${b.binding}) var ${b.name}: ${b.kind === "sampler" ? "sampler" : "texture_3d<f32>"};`);
    }
    lines.push(`
// frameMipmaps.js getMipLevelCount: ceil(log2(max side)), exactly, in integers.
fn vdbSrcLevelCount(dims: vec3u) -> f32 {
  let m = max(max(dims.x, dims.y), max(dims.z, 1u));
  if (m <= 1u) {
    return 0.0;
  }
  return f32(32u - countLeadingZeros(m - 1u));
}
`);
    const sampled: ChannelName[] = ["density", ...(["mask", "albedo", "emission"] as const).filter(on)];
    for (const name of sampled) lines.push(channelWgsl(name, !!this.#textures[name]?.mip));
    lines.push(`fn vdbSrcDensity(v: u32, uvw: vec3f, lod: f32) -> f32 {
  return vdbSrcSampleDensity(uvw, lod).r;
}
fn vdbSrcMask(v: u32, uvw: vec3f, lod: f32) -> f32 {
  return ${on("mask") ? "vdbSrcSampleMask(uvw, lod).r" : "1.0"};
}
fn vdbSrcAlbedo(v: u32, uvw: vec3f, lod: f32) -> vec3f {
  return ${on("albedo") ? "vdbSrcSampleAlbedo(uvw, lod).rgb" : "vec3f(1.0)"};
}
fn vdbSrcEmission(v: u32, uvw: vec3f, lod: f32) -> f32 {
  return ${on("emission") ? "vdbSrcSampleEmission(uvw, lod).r" : "0.0"};
}
fn vdbSrcTexels(v: u32) -> vec3f {
  return vec3f(textureDimensions(vdbSrcDensityMap));
}
fn vdbSrcLodLevels(v: u32) -> f32 {
  return ${this.#textures.density?.mip ? "vdbSrcLevelCount(textureDimensions(vdbSrcDensityMap))" : "0.0"};
}
`);
    if (o.profile === "march") {
      // Dense: no empty-space structure; the march steps everywhere (R1 §4.2).
      lines.push(`fn vdbSrcSkipEmpty(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32) -> f32 {
  return t0;
}
fn vdbSrcMaxIn(v: u32, uvwMin: vec3f, uvwMax: vec3f) -> f32 {
  return ${this.stats.max.toFixed(9)};
}
`);
    }
    return lines.join("\n");
  }

  layoutEntries(slot: VDBSlot, visibility: GPUShaderStageFlags = STAGE_FRAGMENT_COMPUTE): GPUBindGroupLayoutEntry[] {
    this.#check();
    checkSlot(slot, "vdbDenseSource.layoutEntries");
    return this.#bindings(slot).map((b) =>
      b.kind === "sampler"
        ? { binding: b.binding, visibility, sampler: { type: "filtering" } }
        : { binding: b.binding, visibility, texture: { sampleType: "float", viewDimension: "3d" } },
    );
  }

  bindEntries(slot: VDBSlot): GPUBindGroupEntry[] {
    this.#check();
    checkSlot(slot, "vdbDenseSource.bindEntries");
    return this.#bindings(slot).map((b) => ({
      binding: b.binding,
      resource: b.kind === "sampler" ? this.#sampler : b.texture!.createView({ dimension: "3d" }),
    }));
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#destroyTextures();
    this.#disposed = true;
  }
}

/** A dense source with `replace` (LOD swap). */
export type VDBDenseSource = VDBVolumeSource & {
  readonly kind: "dense";
  readonly volume: WvdbDenseVolume;
  replace(volume: WvdbDenseVolume): void;
};

/**
 * Upload a dense `.wvdb` volume (`wvdbDenseVolume` from `@pmndrs/vdb/tools`) on
 * the host's device. The source owns its textures and sampler; `dispose()` frees
 * the textures (samplers have no destroy).
 */
export function vdbDenseSource(device: GPUDevice, volume: WvdbDenseVolume, o: VDBDenseSourceOptions = {}): VDBDenseSource {
  if (!device || typeof (device as { createTexture?: unknown }).createTexture !== "function") {
    throw new VDBError("VDB_UNSUPPORTED_DEVICE", "vdbDenseSource: pass the host's GPUDevice.");
  }
  if (!volume || volume.kind !== "dense" || !volume.density) {
    throw bad("vdbDenseSource: pass a dense volume from wvdbDenseVolume() (@pmndrs/vdb/tools).");
  }
  return new DenseSource(device, volume, o);
}
