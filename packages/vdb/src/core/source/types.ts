/**
 * The volume-source seam of the K3 renderer (R1 §4.2): the renderer's shading
 * reads density, mask, albedo and emission through one interface, with two
 * sources behind it, the sparse `VDBGridPool` (K3.2) and dense `.wvdb` levels
 * (`vdbDenseSource`, K3.0). It mirrors mjurczyk's `createVolumeNodes()`
 * (frameShaderChunks.js L834–1028), so his materials port with call-site edits.
 */

import type { VDBSlot } from "../types.js";

/** Optional channels of a source; absent channels compile to constants (mask 1, albedo 1, emission 0). */
export interface VDBVolumeChannels {
  readonly mask: boolean;
  readonly albedo: boolean;
  readonly emission: boolean;
}

/** The descriptive half of a source: what every adapter needs, whatever it binds. */
export interface VDBVolumeSourceInfo {
  readonly kind: "sparse" | "dense";
  /** Voxels across the box at lod 0 (dense: the texture size; sparse: the index bbox extent). */
  readonly texels: readonly [number, number, number];
  /** Prefiltered levels available above level 0 (0 = level 0 only). */
  readonly lodLevels: number;
  readonly channels: VDBVolumeChannels;
  /** `meanNonZero` feeds `normalizeDensity`; `max` bounds the density; `activeVoxels` counts non-zero voxels. */
  readonly stats: { readonly meanNonZero: number; readonly max: number; readonly activeVoxels: number };
  /** Column-major 4x4: local unit box [-0.5, 0.5]³ → source space (dense: uvw in [0, 1]³; sparse: index space). */
  readonly boxToSource: Float32Array;
  /** Bumps when bindings must be rebuilt (new textures after a LOD swap, pool growth). */
  readonly version: number;
}

/** Options of {@link VDBVolumeSource.wgsl}. */
export interface VDBVolumeSourceWgslOptions {
  readonly slot: VDBSlot;
  /** `march` adds `vdbSrcSkipEmpty` and `vdbSrcMaxIn`. */
  readonly profile: "sample" | "march";
  /** Turn channels the source has off (compiled to constants). Channels it lacks are always constants. */
  readonly channels?: Partial<VDBVolumeChannels>;
}

/**
 * A volume source for raw WebGPU kernels. Generated WGSL (all names `vdbSrc*`):
 *
 *   fn vdbSrcDensity (v: u32, uvw: vec3f, lod: f32) -> f32;   // dense: [0, 1]
 *   fn vdbSrcMask    (v: u32, uvw: vec3f, lod: f32) -> f32;   // 1.0 when absent
 *   fn vdbSrcAlbedo  (v: u32, uvw: vec3f, lod: f32) -> vec3f; // vec3(1) when absent
 *   fn vdbSrcEmission(v: u32, uvw: vec3f, lod: f32) -> f32;   // 0.0 when absent
 *   fn vdbSrcTexels  (v: u32) -> vec3f;
 *   fn vdbSrcLodLevels(v: u32) -> f32;
 *   // profile 'march':
 *   fn vdbSrcSkipEmpty(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32) -> f32;
 *   fn vdbSrcMaxIn   (v: u32, uvwMin: vec3f, uvwMax: vec3f) -> f32;
 *
 * `uvw` is in [0, 1]³ of the source box, already offset and wrapped by the
 * caller; `lod` is in levels (log2 of the footprint in lod-0 voxels).
 */
export interface VDBVolumeSource extends VDBVolumeSourceInfo {
  wgsl(o: VDBVolumeSourceWgslOptions): string;
  layoutEntries(slot: VDBSlot, visibility?: GPUShaderStageFlags): GPUBindGroupLayoutEntry[];
  bindEntries(slot: VDBSlot): GPUBindGroupEntry[];
  dispose(): void;
}
