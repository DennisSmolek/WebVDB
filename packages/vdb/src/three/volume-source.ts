// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L65–66 (min/maxComponent), L729–763
//   (createVolumeUniforms), L765–798 (setMipUniforms), L834–1028 (createVolumeNodes);
//   src/three-openvdb/frame/core/FrameVolume.js L14–22, L357–405 (Data3DTexture setup);
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L483–487 (configureTexture);
//   src/three-openvdb/frame/webgpu/frameMipmaps.js L183–189 (mip atlas texture).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; absent channels compile to constants instead of runtime flags; the
// Shadertoy blackbody and the CIE table are not ported (emission colour is a caller hook).
// Divergences: DIVERGES.md V23–V25.

/**
 * The dense `.wvdb` volume source in three (K3.0, R1 §4.2).
 *
 *   const file = readWvdb(bytes);
 *   const volume = wvdbDenseVolume(file);                 // @pmndrs/vdb/tools: his 8-bit texels + mip atlas
 *   const source = vdbDenseTextures(volume);              // three Data3DTextures, set up as his FrameVolume
 *   const uniforms = vdbVolumeUniforms(source, { extinction: 100 });
 *   const nodes = vdbVolumeNodes(source, uniforms, { useMipmaps: true });
 *   // inside a TSL Fn: nodes.sampleExtinction(localPos, lod), nodes.sampleAlbedo(...), nodes.boxSegment(...)
 *
 * `vdbVolumeNodes` returns the shape of his `createVolumeNodes()` (`getLod`,
 * `sampleExtinction`, `sampleAlbedo`, `sampleEmission`, `insideBox`, `boxExit`,
 * `boxSegment`), so the K3 port of his materials changes one import per file.
 * Positions are in the local unit box [-0.5, 0.5]³.
 */

import * as THREE from "three";
import {
  If,
  abs,
  ceil,
  clamp,
  exp2,
  float,
  floor,
  fract,
  log2,
  max,
  min,
  mix,
  mod,
  pow,
  select,
  texture3D,
  uniform,
  vec2,
  vec3,
  vec4,
} from "three/tsl";
import { VDBError } from "../core/errors.js";
import type { VDBVolumeChannels, VDBVolumeSourceInfo } from "../core/source/types.js";
import type { WvdbDenseTexture, WvdbDenseVolume } from "../tools/wvdb/dense.js";
import type { VDBTslNode } from "./grid-node.js";

type ChannelName = "density" | "mask" | "albedo" | "emission";
const CHANNELS: readonly ChannelName[] = ["density", "mask", "albedo", "emission"];

/** The three textures of one channel. */
export interface VDBDenseChannelTextures {
  readonly map: THREE.Data3DTexture;
  readonly mip: THREE.Data3DTexture | null;
}

/** A dense source as three textures (the descriptive part of `VDBVolumeSource`; no raw bindings). */
export interface VDBThreeDenseSource extends VDBVolumeSourceInfo {
  readonly kind: "dense";
  readonly volume: WvdbDenseVolume;
  readonly textures: Readonly<Record<ChannelName, VDBDenseChannelTextures | null>>;
  /** Swap in another LOD or frame (same channels). New sizes make new textures and bump `version`; call `uniforms.update(source)`. */
  replace(volume: WvdbDenseVolume): void;
  dispose(): void;
}

/**
 * FrameVolume.js L14–22 `createTexture3D` + FrameRenderer.js L483–487
 * `configureTexture` (and frameMipmaps.js L183–189 for the atlas): an 8-bit
 * `Data3DTexture`, `RedFormat` or `RGBAFormat`, `unpackAlignment` 1, linear
 * min/mag, no mipmaps, default clamp-to-edge wrapping.
 */
function texture3DFrom(data: Uint8Array<ArrayBuffer>, dims: readonly number[], texelBytes: 1 | 4): THREE.Data3DTexture {
  const [w, h, d] = dims as [number, number, number];
  const texture = new THREE.Data3DTexture(data, w, h, d);
  texture.format = texelBytes === 4 ? THREE.RGBAFormat : THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.unpackAlignment = 1;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

function channelTextures(tex: WvdbDenseTexture | null): VDBDenseChannelTextures | null {
  if (!tex) return null;
  return {
    map: texture3DFrom(tex.data, tex.dimensions, tex.texelBytes),
    mip: tex.mip ? texture3DFrom(tex.mip.data, tex.mip.dimensions, tex.texelBytes) : null,
  };
}

const channelsOf = (v: WvdbDenseVolume): Record<ChannelName, WvdbDenseTexture | null> => ({
  density: v.density,
  mask: v.mask,
  albedo: v.albedo,
  emission: v.emission,
});

class ThreeDenseSource implements VDBThreeDenseSource {
  readonly kind = "dense" as const;
  readonly boxToSource = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0.5, 0.5, 1]);
  readonly channels: VDBVolumeChannels;
  volume: WvdbDenseVolume;
  textures: Record<ChannelName, VDBDenseChannelTextures | null>;
  texels: [number, number, number];
  lodLevels: number;
  stats: { meanNonZero: number; max: number; activeVoxels: number };
  version = 0;

  constructor(volume: WvdbDenseVolume) {
    this.volume = volume;
    this.channels = { mask: !!volume.mask, albedo: !!volume.albedo, emission: !!volume.emission };
    const c = channelsOf(volume);
    this.textures = { density: channelTextures(c.density), mask: channelTextures(c.mask), albedo: channelTextures(c.albedo), emission: channelTextures(c.emission) };
    this.texels = [...volume.dimensions] as [number, number, number];
    this.lodLevels = volume.density.mip?.levels ?? 0;
    this.stats = { meanNonZero: volume.meanDensity, max: volume.maxDensity, activeVoxels: volume.activeTexels };
  }

  replace(volume: WvdbDenseVolume): void {
    const now = channelsOf(this.volume);
    const next = channelsOf(volume);
    for (const name of CHANNELS) {
      if (!!now[name] !== !!next[name] || !!now[name]?.mip !== !!next[name]?.mip) {
        throw new VDBError("VDB_BAD_OPTION", `vdbDenseTextures.replace: the ${name} channel or its mip atlas appears or disappears; make a new source.`);
      }
    }
    // WVDBLoader.js L37–48 updateTexture: a new size disposes the GPU texture first.
    let resized = false;
    for (const name of CHANNELS) {
      const t = this.textures[name];
      const v = next[name];
      if (!t || !v) continue;
      const set = (tex: THREE.Data3DTexture, data: Uint8Array<ArrayBuffer>, dims: readonly number[]) => {
        const [width, height, depth] = dims as [number, number, number];
        const img = tex.image as { width: number; height: number; depth: number };
        const changed = img.width !== width || img.height !== height || img.depth !== depth;
        tex.image = { data, width, height, depth } as never;
        if (changed) {
          tex.dispose();
          resized = true;
        }
        tex.needsUpdate = true;
      };
      set(t.map, v.data, v.dimensions);
      if (t.mip && v.mip) set(t.mip, v.mip.data, v.mip.dimensions);
    }
    if (resized) this.version++;
    this.volume = volume;
    this.texels = [...volume.dimensions] as [number, number, number];
    this.lodLevels = volume.density.mip?.levels ?? 0;
    this.stats = { meanNonZero: volume.meanDensity, max: volume.maxDensity, activeVoxels: volume.activeTexels };
  }

  dispose(): void {
    for (const t of Object.values(this.textures)) {
      t?.map.dispose();
      t?.mip?.dispose();
    }
  }
}

/** Upload a dense volume (`wvdbDenseVolume`, `@pmndrs/vdb/tools`) as three textures, set up as his `FrameVolume` does. */
export function vdbDenseTextures(volume: WvdbDenseVolume): VDBThreeDenseSource {
  if (!volume || volume.kind !== "dense" || !volume.density) {
    throw new VDBError("VDB_BAD_OPTION", "vdbDenseTextures: pass a dense volume from wvdbDenseVolume() (@pmndrs/vdb/tools).");
  }
  return new ThreeDenseSource(volume);
}

/** Initial values of {@link vdbVolumeUniforms} (his `FrameVolume` props of the same names and defaults). */
export interface VDBVolumeUniformParameters {
  /** Optical depth of the longest box side at the mean density, with `normalizeDensity` (default 100). */
  extinction?: number;
  /** Density multiplier (default 1); with `normalizeDensity` it divides by `stats.meanNonZero`. */
  densityScale?: number;
  normalizeDensity?: boolean;
  densityCutoff?: number;
  baseColor?: THREE.ColorRepresentation;
  emissiveIntensity?: number;
  /** Kelvin at emission values 0 and 1 (default [0, 6500]). */
  temperatureRange?: [number, number];
  offset3D?: THREE.Vector3Like;
  wrap3D?: THREE.Wrapping;
  /** `matrixWorld.getMaxScaleOnAxis()` of the volume (default 1). */
  worldScale?: number;
}

/** His `createVolumeUniforms()` subset the source reads (frameShaderChunks.js L729–763), as live TSL uniforms. */
export interface VDBVolumeUniforms {
  readonly worldScale: VDBTslNode;
  readonly gridTexels: VDBTslNode;
  readonly mipAtlasSize: VDBTslNode;
  readonly maps: Readonly<Record<ChannelName, VDBTslNode | null>>;
  readonly mipMaps: Readonly<Record<ChannelName, VDBTslNode | null>>;
  readonly mipLevels: Readonly<Record<ChannelName, VDBTslNode>>;
  readonly baseColor: VDBTslNode;
  readonly extinction: VDBTslNode;
  readonly densityScale: VDBTslNode;
  readonly densityCutoff: VDBTslNode;
  readonly emissiveIntensity: VDBTslNode;
  readonly temperatureRange: VDBTslNode;
  readonly offset3D: VDBTslNode;
  readonly wrap3D: VDBTslNode;
  /** Re-read textures, sizes and the normalized density scale after `source.replace` (setMipUniforms, L773–798). */
  update(source: VDBThreeDenseSource, o?: { densityScale?: number; normalizeDensity?: boolean }): void;
}

/** FrameVolume.js L561–568 `getDensityScale`. */
const densityScaleOf = (source: VDBThreeDenseSource, densityScale: number, normalize: boolean): number =>
  normalize && source.stats.meanNonZero > 0 ? densityScale / source.stats.meanNonZero : densityScale;

/** Live uniforms for one dense source (his `createVolumeUniforms` + `setVolumeUniforms` + `setMipUniforms`). */
export function vdbVolumeUniforms(source: VDBThreeDenseSource, p: VDBVolumeUniformParameters = {}): VDBVolumeUniforms {
  const tex = source.textures;
  let densityScale = p.densityScale ?? 1.0;
  let normalize = p.normalizeDensity ?? true;
  const maps = {} as Record<ChannelName, VDBTslNode | null>;
  const mipMaps = {} as Record<ChannelName, VDBTslNode | null>;
  const mipLevels = {} as Record<ChannelName, VDBTslNode>;
  for (const name of CHANNELS) {
    maps[name] = tex[name] ? texture3D(tex[name]!.map) : null;
    mipMaps[name] = tex[name]?.mip ? texture3D(tex[name]!.mip!) : null;
    mipLevels[name] = uniform(0.0);
  }
  const gridTexels = uniform(new THREE.Vector3(1, 1, 1));
  const mipAtlasSize = uniform(new THREE.Vector3(1, 1, 1));
  const densityScaleU = uniform(densityScaleOf(source, densityScale, normalize));
  const u: VDBVolumeUniforms = {
    worldScale: uniform(p.worldScale ?? 1.0),
    gridTexels,
    mipAtlasSize,
    maps,
    mipMaps,
    mipLevels,
    baseColor: uniform(new THREE.Color(p.baseColor ?? 0xffffff)),
    extinction: uniform(p.extinction ?? 100.0),
    densityScale: densityScaleU,
    densityCutoff: uniform(p.densityCutoff ?? 0.0),
    emissiveIntensity: uniform(p.emissiveIntensity ?? 1.0),
    temperatureRange: uniform(new THREE.Vector2(...(p.temperatureRange ?? [0.0, 6500.0]))),
    offset3D: uniform(new THREE.Vector3().copy((p.offset3D ?? { x: 0, y: 0, z: 0 }) as THREE.Vector3)),
    wrap3D: uniform(p.wrap3D ?? THREE.ClampToEdgeWrapping, "int"),
    update(s, o = {}) {
      if (o.densityScale !== undefined) densityScale = o.densityScale;
      if (o.normalizeDensity !== undefined) normalize = o.normalizeDensity;
      densityScaleU.value = densityScaleOf(s, densityScale, normalize);
      const [w, h, d] = s.texels;
      gridTexels.value.set(w, h, d);
      const densityMip = s.textures.density?.mip;
      if (densityMip) {
        const img = densityMip.image as { width: number; height: number; depth: number };
        mipAtlasSize.value.set(img.width, img.height, img.depth);
      }
      for (const name of CHANNELS) {
        const t = s.textures[name];
        if (t && maps[name]) maps[name].value = t.map;
        if (t?.mip && mipMaps[name]) mipMaps[name].value = t.mip;
        // Every channel shares the density size (wvdbDenseVolume checks it), so all levels are the mip's.
        mipLevels[name].value = densityMip && t?.mip ? s.volume[name]!.mip!.levels : 0;
      }
    },
  };
  u.update(source);
  return u;
}

/** Options of {@link vdbVolumeNodes}. */
export interface VDBVolumeNodesOptions {
  /** Sample the CPU mip atlas at `lod` (his shaders built with `useMipmaps`). Default: the source has mips. */
  useMipmaps?: boolean;
  /**
   * Emission colour from temperature (Kelvin node) → vec3 node. Default: white.
   * His `'approximate'` blackbody is from Shadertoy (CC BY-NC-SA unless stated) and is not
   * ported (R1 §7); the CIE `'planck'`/`'soot'` table lands with emission in K3.6.
   */
  emissionColor?: (temperature: VDBTslNode) => VDBTslNode;
}

/** The node factory of his `createVolumeNodes()` (frameShaderChunks.js L834–1028). */
export interface VDBVolumeNodes {
  getLod(localSize: VDBTslNode): VDBTslNode;
  sampleExtinction(localPosition: VDBTslNode, lod: VDBTslNode): VDBTslNode;
  sampleAlbedo(localPosition: VDBTslNode, lod: VDBTslNode): VDBTslNode;
  sampleEmission(localPosition: VDBTslNode, lod: VDBTslNode): VDBTslNode;
  insideBox(localPosition: VDBTslNode): VDBTslNode;
  boxExit(origin: VDBTslNode, direction: VDBTslNode): VDBTslNode;
  boxSegment(start: VDBTslNode, end: VDBTslNode): VDBTslNode;
}

/** frameShaderChunks.js L65–66. */
export const vdbMinComponent = (v: VDBTslNode): VDBTslNode => min(v.x, min(v.y, v.z));
export const vdbMaxComponent = (v: VDBTslNode): VDBTslNode => max(v.x, max(v.y, v.z));

/**
 * Sampling of one dense source, in his `createVolumeNodes` shape. Call the
 * sample functions inside a TSL `Fn` body. Absent channels give his fallbacks
 * (mask 1, albedo `baseColor`, emission 0) without sampling.
 */
export function vdbVolumeNodes(source: VDBThreeDenseSource, u: VDBVolumeUniforms, o: VDBVolumeNodesOptions = {}): VDBVolumeNodes {
  const useMipmaps = o.useMipmaps ?? source.lodLevels > 0;
  const { gridTexels, mipAtlasSize } = u;
  const T = THREE as unknown as Record<string, number>;

  // L838–846: level 1 and up of the mip atlas (layout of frameMipmaps.js).
  const sampleMipAtlas = (mipMap: VDBTslNode, level: VDBTslNode, uvw: VDBTslNode): VDBTslNode => {
    const size = max(vec3(1.0), ceil(gridTexels.div(exp2(level))));
    const offset = mipAtlasSize.x.sub(mipAtlasSize.x.div(exp2(level.sub(1.0))));
    const texel = clamp(uvw.mul(size), vec3(0.5), size.sub(0.5));
    return mipMap.sample(vec3(texel.x.add(offset), texel.y, texel.z).div(mipAtlasSize)).level(float(0.0));
  };

  // L849–872: linear between two mip levels (LinearMipmapLinearFilter).
  const sampleMipmapped = (map: VDBTslNode, mipMap: VDBTslNode, levels: VDBTslNode, position: VDBTslNode, lod: VDBTslNode): VDBTslNode => {
    const uvw = vec3(position).toVar();
    const result = vec4(0.0).toVar();
    const level = min(lod, levels).toVar();
    If(level.lessThanEqual(0.0), () => {
      result.assign(map.sample(uvw).level(float(0.0)));
    }).Else(() => {
      const level0 = floor(level).toVar();
      const low = vec4(0.0).toVar();
      If(level0.lessThan(0.5), () => {
        low.assign(map.sample(uvw).level(float(0.0)));
      }).Else(() => {
        low.assign(sampleMipAtlas(mipMap, level0, uvw));
      });
      const high = sampleMipAtlas(mipMap, min(level0.add(1.0), levels), uvw);
      result.assign(mix(low, high, level.sub(level0)));
    });
    return result;
  };

  // L874–877 sample3D (a channel without a mip atlas samples level 0).
  const sample3D = (name: ChannelName, uvw: VDBTslNode, lod: VDBTslNode): VDBTslNode => {
    const map = u.maps[name]!;
    const mip = u.mipMaps[name];
    return useMipmaps && mip ? sampleMipmapped(map, mip, u.mipLevels[name], uvw, lod) : map.sample(uvw).level(float(0.0));
  };

  // L892
  const getLod = (localSize: VDBTslNode) => max(0.0, log2(localSize.mul(vdbMaxComponent(gridTexels))));

  // L894–911
  const mirrorAxis = (value: VDBTslNode) => {
    const period = mod(value, 2.0);
    return select(period.greaterThan(1.0), float(2.0).sub(period), period);
  };
  const wrapUv = (uvw: VDBTslNode) =>
    select(
      u.wrap3D.equal(T.RepeatWrapping!),
      fract(uvw),
      select(u.wrap3D.equal(T.MirroredRepeatWrapping!), vec3(mirrorAxis(uvw.x), mirrorAxis(uvw.y), mirrorAxis(uvw.z)), uvw),
    );
  const gridUv = (localPosition: VDBTslNode) => wrapUv(localPosition.add(0.5).add(u.offset3D));

  // L914–936: extinction per world unit. The mask is read at local + 0.5 (no offset or wrap), as his.
  const sampleExtinction = (localPosition: VDBTslNode, lod: VDBTslNode) => {
    const mask = source.channels.mask ? sample3D("mask", localPosition.add(0.5), lod).r : float(1.0);
    const value = sample3D("density", gridUv(localPosition), lod).r.mul(mask);
    return select(value.greaterThan(u.densityCutoff), value, float(0.0)).mul(u.densityScale).mul(u.extinction).div(u.worldScale);
  };

  // L938–958
  const insideBox = (localPosition: VDBTslNode) => vdbMaxComponent(abs(localPosition)).lessThanEqual(0.5);
  const boxExit = (origin: VDBTslNode, direction: VDBTslNode) => {
    const inverseDirection = vec3(1.0).div(direction);
    const tMin = vec3(-0.5).sub(origin).mul(inverseDirection);
    const tMax = vec3(0.5).sub(origin).mul(inverseDirection);
    return max(0.0, vdbMinComponent(max(tMin, tMax)));
  };
  const boxSegment = (start: VDBTslNode, end: VDBTslNode) => {
    const inverseDirection = vec3(1.0).div(end.sub(start));
    const tMin = vec3(-0.5).sub(start).mul(inverseDirection);
    const tMax = vec3(0.5).sub(start).mul(inverseDirection);
    return vec2(max(0.0, vdbMaxComponent(min(tMin, tMax))), min(1.0, vdbMinComponent(max(tMin, tMax))));
  };

  // L960–971
  const sampleAlbedo = (localPosition: VDBTslNode, lod: VDBTslNode) =>
    source.channels.albedo ? u.baseColor.mul(sample3D("albedo", gridUv(localPosition), lod).rgb) : u.baseColor.mul(vec3(1.0));

  // L990–1017, colour via the caller's hook (see VDBVolumeNodesOptions.emissionColor).
  const emissionColor = o.emissionColor ?? (() => vec3(1.0));
  const sampleEmission = (localPosition: VDBTslNode, lod: VDBTslNode) => {
    if (!source.channels.emission) return vec3(0.0);
    const range = u.temperatureRange;
    const value = sample3D("emission", gridUv(localPosition), lod).r;
    const temperature = mix(range.x, range.y, value);
    const power = pow(temperature.div(max(range.y, 1.0)), 4.0);
    const color = select(temperature.greaterThan(0.0), emissionColor(temperature), vec3(0.0));
    return color.mul(power).mul(u.emissiveIntensity);
  };

  return { getLod, sampleExtinction, sampleAlbedo, sampleEmission, insideBox, boxExit, boxSegment };
}
