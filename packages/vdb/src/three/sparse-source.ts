// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L729–763 (createVolumeUniforms, the source
//   half), L834–1028 (createVolumeNodes: getLod, sampleExtinction, sampleAlbedo, sampleEmission and
//   the box helpers, through ./volume-source.ts).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: the density is read from a sparse NanoVDB grid in a VDBGridPool (vdbSample, trilinear)
// instead of an 8-bit 3D texture; three LOD strategies instead of his box mip atlas; an HDDA
// empty-space query for marches. Divergences: DIVERGES.md (K3.2 rows), V23–V25.

/**
 * The sparse volume source (K3.2, R1 §4.2–4.3): a `VDBGridPool` grid behind the same seam as the
 * dense `.wvdb` source, so `VDBFrameVolume` renders NanoVDB grids.
 *
 *   const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
 *   const source = vdbSparseSource(grids, { density: id }, { lod: "coarse-pyramid", images: { density: image } });
 *   const volume = new VDBFrameVolume(source, { extinction: 100 });   // box: the grid's world box
 *
 * Positions are his local unit box [-0.5, 0.5]³; uvw = local + 0.5 (+ offset, wrapped) maps onto the
 * channel grid's index cell box `[min − ½, max + ½]` (voxel centres at integers), so uvw
 * (i + 0.5) / N lands on voxel min + i as on a dense texel centre. The sample position is clamped to
 * `[min, max]` (his clamp-to-edge). Level 0 is the pool's trilinear `vdbSample`; with the node helper
 * `'fastTrilinear'` (recommended, K3.2d) it is `vdbSampleFast` through one accessor per shader invocation
 * (`cacheAccessor`): the same values, 1.5–2× faster on the hardware GPU (docs/handoffs/K3-2d.md).
 *
 * LOD (`lod`, his `sampleExtinction(localPos, lod)` at lod ≥ 1; R1 §4.3):
 * - `'none'`: level 0 always (no prefilter; aliases when a froxel spans many voxels). lodLevels 0.
 * - `'coarse-pyramid'` (default, the K3.2 A/B pick): a dense r16f pyramid from L₀ = max(1, ⌈log2(maxTexels / pyramidMax)⌉):
 *   the base is a box resample of the grid, the levels above it his box mip atlas. lod in (0, L₀)
 *   blends level 0 toward L₀; lod ≥ L₀ samples the pyramid as his mip atlas. GPU bytes: ~2 B × (base
 *   + atlas) texels. Built on the GPU from the pool grid by default (`pyramidBuild: 'gpu'`, K3.2a: the
 *   core's `vdbCreateBoxPyramid`, no CPU image needed) and rebuilt when the grid's image changes
 *   (`pool.replace`: a sequence frame), checked once per render that samples it; `'cpu'` keeps the
 *   K3.2 path (`@pmndrs/vdb/tools` `vdbBoxPyramid` from `images.density`, static). Both give the same
 *   texels to ≤ 1 f16 ulp (e2e/k3-pyramid.spec.ts).
 * - `'node-average'`: no extra memory: trilinear over the node averages (`vdbNodeAverage`) at
 *   level 3 (leaves, the exact 8³ box mean for fog leaves) and level 7 (lower nodes, the mean of
 *   their active voxels); lod in (0, 3) blends level 0 → 3, (3, 7) blends 3 → 7. Needs stats.
 *
 * HDDA: `skipEmpty` lets a march jump samples that are provably 0 (`VDBFrameVolume` `hdda`, default
 * off since D30 / D31: it slowed the shadow march on Apple GPUs; the image is the same either way).
 *
 * Following the image (`followImage`, G16a): the index boxes, world box and pyramid are fixed when the source is
 * made unless it follows the grids' images. Then a `pool.replace` that changes a channel grid's box (a coarse-first
 * swap: the preview, then the full grid in the same id) is re-read from `pool.info` once the new image is resident:
 * new box uniforms, a new GPU pyramid of the new size (its textures swapped into the same texture nodes), the LOD
 * range as a uniform, and a new `version`, which a `VDBFrameVolume` follows with its box. No shader is rebuilt, and the
 * pyramid kernels of every type the `VDBGridNode` decodes are compiled up front.
 *
 * Mask, albedo and emission grids (optional, scalar; albedo may be three grids for RGB) are read at
 * level 0 only. One VDB module per shader: every grid goes through the one `VDBGridNode`.
 */

import * as THREE from "three";
import * as TSL from "three/tsl";
import { code, wgslFn } from "three/tsl";
import { VDBError } from "../core/errors.js";
import type { VDBVolumeChannels } from "../core/source/types.js";
import { vdbCreateBoxPyramid, warmBoxPyramidPipelines, type VDBBoxPyramidGpu } from "../core/compute/pyramid.js";
import { vdbBoxPyramid } from "../tools/frame/sparse.js";
import { vdbBoxPyramidLayout } from "../tools/frame/pyramid-layout.js";
import { getGPUTexture, webgpuBackendOf } from "./device.js";
import { leafVoxels, treeNodes } from "../tools/cpu/march.js";
import { vdbSample, type VDBGridNode, type VDBTslNode } from "./grid-node.js";
import {
  vdbBoxNodes,
  vdbDensityScaleOf,
  vdbGridUv,
  vdbMaxComponent,
  vdbMipSampler,
  type VDBThreeSourceFactory,
  type VDBVolumeNodes,
  type VDBVolumeNodesOptions,
  type VDBVolumeUniformParameters,
  type VDBVolumeUniforms,
} from "./volume-source.js";

const { If, clamp, float, int, log2, max, min, mix, pow, select, texture3D, uint, uniform, vec2, vec3 } = TSL as unknown as Record<string, VDBTslNode>;

type Vec3 = [number, number, number];

/** The LOD strategies of the sparse source (R1 §4.3); see the file comment. */
export type VDBSparseLod = "none" | "coarse-pyramid" | "node-average";

/** The pool grid ids of the channels. `albedo`: one grey grid or three (RGB). */
export interface VDBSparseSourceGrids {
  density: number;
  mask?: number;
  albedo?: number | readonly [number, number, number];
  emission?: number;
}

/** Inclusive voxel index box. */
export interface VDBIndexBox {
  min: Vec3;
  max: Vec3;
}

/** Options of {@link vdbSparseSource}. */
export interface VDBSparseSourceOptions {
  /**
   * LOD strategy. Default `'coarse-pyramid'`, the K3.2 A/B's pick (closest to his mips, low flicker, cheap to
   * sample; needs `images.density`); see docs/handoffs/K3-2.md.
   */
  lod?: VDBSparseLod;
  /** `'coarse-pyramid'`: base level texels on the longest side at most (default 128). */
  pyramidMax?: number;
  /**
   * `'coarse-pyramid'`: where the pyramid is built. `'gpu'` (default, K3.2a): a compute pass over the
   * pool grid on the renderer's device (the core's `vdbCreateBoxPyramid`), rebuilt when the grid's
   * `imageRevision` changes; needs an initialized WebGPURenderer behind the `VDBGridNode`. `'cpu'`:
   * `vdbBoxPyramid` from `images.density` (the K3.2 path, static; the GPU build's twin).
   */
  pyramidBuild?: "gpu" | "cpu";
  /** `'none'` (default): raw floats. `'clamp01'`: clamp each density sample to [0, 1] (his 8-bit range; parity A/B). */
  normalize?: "none" | "clamp01";
  /** CPU copies of the grid images: the density stats (and a `pyramidBuild: 'cpu'` pyramid) read `density`. */
  images?: { density?: Uint32Array };
  /**
   * Density stats (`normalizeDensity` divides by `meanNonZero`). Default: computed from `images.density`.
   * Either way they describe the image at creation; after the density grid's image changes (`pool.replace`,
   * its `imageRevision`) the source reads the new image back from the pool and recomputes them (K3.2e; see
   * `statsUpdates`).
   */
  stats?: { meanNonZero: number; max: number; activeVoxels: number };
  /** Density index box the unit box covers (default: the grid's index bbox). Aux grids use their own. */
  indexBox?: VDBIndexBox;
  /**
   * K3.6: index boxes the unit box covers in the aux grids (default: each grid's own index bbox). Grids converted
   * from one dense `.wvdb` level (`wvdbDenseToNvdb`) share the texture's box (`textureIndexBox`); pass it so a
   * sparse emission or mask channel lines up with the density (its active bbox is usually smaller).
   */
  auxIndexBoxes?: Partial<Record<"mask" | "albedo" | "emission", VDBIndexBox>>;
  /** World box (default: the density cell box through the grid's transform, as an AABB). */
  bbox?: [Vec3, Vec3];
  /**
   * K3.2d: level-0 samples through `vdbSampleFast` (the 8 trilinear taps as lower-node child lookups, no
   * tree walk per tap; bit-identical to `vdbSample`). Needs the `VDBGridNode` helper `'fastTrilinear'`.
   * Default: on when the node has the helper.
   */
  fastTrilinear?: boolean;
  /**
   * K3.2d: keep one accessor per shader invocation (a WGSL `var<private>`) across the samples of a march
   * instead of a fresh one per sample, so consecutive samples reuse its cached leaf/lower/upper nodes. The
   * values are the same either way. Default true (the K3.2d hardware A/B: with `fastTrilinear`, whose taps
   * start from the cached lower node, 10–25% off the source passes; alone it changes little).
   */
  cacheAccessor?: boolean;
  /**
   * G16a: follow the grids' images. When a channel grid's `imageRevision` moves (a `pool.replace`, e.g. a
   * `vdbStreamCoarseFirst` swap from the preview to the full grid) and every channel grid is resident again, the
   * source re-reads its index boxes, world box and transform from `pool.info`, rebuilds the coarse pyramid at the
   * new size and bumps `version` (a `VDBFrameVolume` whose box is the source's follows it). Boxes, sizes and LOD
   * levels are uniforms in this mode, so nothing recompiles; give the `VDBGridNode` every grid type the images will
   * have (`types`, e.g. `VDBCoarseFirstEvent.types`). Default false: the boxes are fixed when the source is made.
   * Not with `indexBox`, `auxIndexBoxes`, `bbox` or `pyramidBuild: 'cpu'`.
   */
  followImage?: boolean;
}

/** What the source needs from a pool: `info` and the ids' grids (a `VDBGridPool`). */
interface PoolInfoLike {
  info(id: number): { indexBBox: { min: Vec3; max: Vec3 }; indexToWorld: Float64Array; hasStats: boolean; capacity: number; type: string; imageRevision?: number; resident?: boolean };
}

/** The coarse pyramid of a `'coarse-pyramid'` sparse source. */
export interface VDBSparsePyramid {
  /** Where it was built (see `pyramidBuild`). */
  readonly build: "gpu" | "cpu";
  /** L₀: the base level in level-0 voxels. */
  readonly level: number;
  /** Texels of the base level. */
  readonly dimensions: [number, number, number];
  /** His atlas of the levels above the base (null for a 1³ base); `data` on the CPU path only. */
  readonly atlas: { readonly dimensions: [number, number, number]; readonly levels: number; readonly data?: Float32Array } | null;
  /** L₀ + atlas levels. */
  readonly lodLevels: number;
  /** The base level's f32 texels (CPU path only). */
  readonly data?: Float32Array;
  /** r16f textures: the base level and the atlas. */
  readonly map: THREE.Data3DTexture;
  readonly mip: THREE.Data3DTexture | null;
  /** The core GPU pyramid writing into `map` / `mip` (GPU path), else null. */
  readonly gpu: VDBBoxPyramidGpu | null;
}

/** The sparse source in three. */
export interface VDBThreeSparseSource extends VDBThreeSourceFactory {
  readonly grids: VDBGridNode;
  readonly ids: Readonly<VDBSparseSourceGrids>;
  readonly lod: VDBSparseLod;
  readonly normalize: "none" | "clamp01";
  /** The density index box the unit box covers. */
  readonly indexBox: VDBIndexBox;
  /** `'coarse-pyramid'` only: the pyramid and its textures. */
  readonly pyramid: VDBSparsePyramid | null;
  /** K3.2d: level 0 through `vdbSampleFast` (see the option). */
  readonly fastTrilinear: boolean;
  /** K3.2d: one accessor per invocation across samples (see the option). */
  readonly cacheAccessor: boolean;
  /** G16a: the boxes follow the grids' images (see the option). */
  readonly followImage: boolean;
  /** G16a: times the boxes (and the pyramid, if its size changed) were re-read after an image change. */
  readonly layoutUpdates: number;
  /**
   * GPU bytes: the grids' pool capacity, the pyramid textures (r16f) and the GPU build's scratch
   * buffers (kept for rebuilds; 0 on the CPU path).
   */
  readonly gpuBytes: { readonly pool: number; readonly pyramid: number; readonly pyramidScratch: number };
  /** GPU pyramid builds recorded so far (the first, then one per image change). */
  readonly pyramidBuilds: number;
  /** Density stats (`normalizeDensity`'s `meanNonZero`): of the density image at creation, then of the latest one read back (K3.2e). */
  readonly stats: { readonly meanNonZero: number; readonly max: number; readonly activeVoxels: number };
  /** Times the stats were recomputed after an image change (K3.2e). */
  readonly statsUpdates: number;
  /**
   * K3.2e: if the density grid's image changed since the stats were computed and it is resident, reads it
   * back from the pool (own encoder, submitted) and recomputes the stats (tools' twin of the CPU path);
   * resolves to whether it did. Renders call it themselves (the density scale uniform's render update) and
   * apply the new mean to the uniforms when the readback lands (K5.9a: a per-draw value, e.g. `VDBFrameVolume.densityScale`, is never overwritten); call it after `pool.update` was
   * submitted, as `grids.update()`.
   */
  refreshStats(): Promise<boolean>;
  /**
   * GPU path: rebuilds the pyramid now (own encoder, submitted) if the density grid's image changed
   * since the last build and it is resident; returns whether it did. Renders that sample the pyramid
   * call it themselves; call it after `pool.update` was submitted, as `grids.update()`.
   */
  refreshPyramid(): boolean;
  /**
   * The source's WGSL helpers as `wgslFn` nodes (including the grid module), for probes and tests:
   * `vdbFrameNodeAverage(g, p, dim) -> f32` (CPU twin `tools` `vdbFrameNodeAverageSample`) and
   * `vdbFrameSkipEmpty(g, o, d, t0, t1, cutoff, dilate) -> vec2f` (twin `vdbFrameSkipEmpty`).
   */
  nodeAverageFn(): VDBTslNode;
  skipEmptyFn(): VDBTslNode;
  dispose(): void;
}

/** Uniforms of a sparse source: his `VDBVolumeUniforms` plus the channel boxes and pyramid sizes. */
export interface VDBSparseVolumeUniforms extends VDBVolumeUniforms {
  /** Per channel: index min and size of the index box (voxels). */
  readonly boxes: Readonly<Record<"density" | "mask" | "albedo" | "emission", { min: VDBTslNode; size: VDBTslNode } | null>>;
  readonly pyramidLevel: VDBTslNode;
  readonly pyramidLevels: VDBTslNode;
  readonly pyramidTexels: VDBTslNode;
  readonly pyramidAtlasSize: VDBTslNode;
}

const bad = (message: string) => new VDBError("VDB_BAD_OPTION", `vdbSparseSource: ${message}`);

/** G16a: the pyramid texture nodes of a following source's uniforms (their textures change with the layout). */
const pyramidTextures = new WeakMap<object, { map: VDBTslNode; mip: VDBTslNode }>();

/** The non-zero density stats of an image (all leaf voxels, active or not, and active tiles). */
function imageStats(image: Uint32Array, clamp01: boolean): { meanNonZero: number; max: number; activeVoxels: number } {
  let sum = 0;
  let count = 0;
  let maxValue = 0;
  const add = (v: number, n: number) => {
    const x = clamp01 ? Math.min(Math.max(v, 0), 1) : v;
    if (x !== 0) {
      sum += x * n;
      count += n;
      maxValue = Math.max(maxValue, x);
    }
  };
  for (const node of treeNodes(image)) {
    for (const t of node.activeTiles) add(t.value, t.dim ** 3);
    if (node.level !== 0) continue;
    const { values } = leafVoxels(image, node.address);
    for (let n = 0; n < 512; n++) add(values[n]!, 1);
  }
  return { meanNonZero: count > 0 ? sum / count : 0, max: maxValue, activeVoxels: count };
}

/** World AABB of an index cell box through a 3x4 row-major affine. */
function worldBox(box: VDBIndexBox, m: Float64Array): [Vec3, Vec3] {
  const lo: Vec3 = [Infinity, Infinity, Infinity];
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const p = [c & 1 ? box.max[0] + 0.5 : box.min[0] - 0.5, c & 2 ? box.max[1] + 0.5 : box.min[1] - 0.5, c & 4 ? box.max[2] + 0.5 : box.min[2] - 0.5];
    for (let r = 0; r < 3; r++) {
      const v = m[r * 4]! * p[0]! + m[r * 4 + 1]! * p[1]! + m[r * 4 + 2]! * p[2]! + m[r * 4 + 3]!;
      lo[r] = Math.min(lo[r]!, v);
      hi[r] = Math.max(hi[r]!, v);
    }
  }
  return [lo, hi];
}

/** A half-float r16f Data3DTexture (linear, clamp-to-edge, no mipmaps); `data` null: GPU-written (no upload). */
function halfTexture(data: Float32Array | null, dims: readonly number[]): THREE.Data3DTexture {
  let half: Uint16Array | null = null;
  if (data) {
    half = new Uint16Array(data.length);
    for (let i = 0; i < data.length; i++) half[i] = THREE.DataUtils.toHalfFloat(data[i]!);
  }
  const [w, h, d] = dims as Vec3;
  const texture = new THREE.Data3DTexture(half, w, h, d);
  // A GPU-written texture: three allocates it (createTexture) but never uploads (no data to upload).
  if (!half) texture.source.dataReady = false;
  texture.format = THREE.RedFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

// WGSL helpers over the grid module (one module per shader: the VDBGridNode's).
const NODE_AVERAGE_WGSL = /* wgsl */ `fn vdbFrameNodeAverage(g: u32, p: vec3f, dim: u32) -> f32 {
    var acc = vdbAccessorInit(g);
    let fd = f32(dim);
    let q = (p - vec3f(0.5 * fd - 0.5)) / fd;
    let base = floor(q);
    let w = q - base;
    let c = vec3i(base) * i32(dim);
    // One call site in a loop: SwiftShader compiles 8 inlined tree walks very slowly.
    var v: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) {
        v[k] = vdbNodeAverage(&acc, c + vec3i(vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u)) * i32(dim), dim);
    }
    let x00 = v[0] + w.x * (v[1] - v[0]);
    let x10 = v[2] + w.x * (v[3] - v[2]);
    let x01 = v[4] + w.x * (v[5] - v[4]);
    let x11 = v[6] + w.x * (v[7] - v[6]);
    let y0 = x00 + w.y * (x10 - x00);
    let y1 = x01 + w.y * (x11 - x01);
    return y0 + w.z * (y1 - y0);
}`;

// The 'node-average' density at a lod (one call per sample: level 0 → 3 → 7 blends, two taps at most).
const nodeLodWgsl = (sample: string) => /* wgsl */ `fn vdbFrameNodeLod(g: u32, p: vec3f, lod: f32) -> f32 {
    var acc = vdbAccessorInit(g);
    if lod <= 0.0 {
        return ${sample}(&acc, p);
    }
    let fine = lod < 3.0;
    var a = 0.0;
    if fine {
        a = ${sample}(&acc, p);
    } else {
        a = vdbFrameNodeAverage(g, p, 8u);
    }
    if lod >= 7.0 {
        return vdbFrameNodeAverage(g, p, 128u);
    }
    let b = vdbFrameNodeAverage(g, p, select(128u, 8u, fine));
    let t = select((lod - 3.0) / 4.0, lod / 3.0, fine);
    return mix(a, b, t);
}`;

// K3.2d: level-0 density through the source's own sampler: vdbSampleFast (helper 'fastTrilinear') and/or one
// accessor per invocation kept in a private variable (copied in and out: WGSL can't pass a ptr<private> where
// the API takes ptr<function>). Same values as vdbSample with a fresh accessor (the accessor cache only skips
// the descent).
const SAMPLE_CACHE_WGSL = /* wgsl */ `var<private> vdbFrameAcc: VdbAccessor;
var<private> vdbFrameAccGrid: u32 = 0xffffffffu;`;
const sampleWgsl = (fast: boolean, cached: boolean) => {
  const call = fast ? "vdbSampleFast" : "vdbSample";
  return cached
    ? /* wgsl */ `fn vdbFrameSample(g: u32, xyz: vec3f) -> f32 {
    if vdbFrameAccGrid != g {
        vdbFrameAcc = vdbAccessorInit(g);
        vdbFrameAccGrid = g;
    }
    var a = vdbFrameAcc;
    let v = ${call}(&a, xyz);
    vdbFrameAcc = a;
    return v;
}`
    : /* wgsl */ `fn vdbFrameSample(g: u32, xyz: vec3f) -> f32 {
    var a = vdbAccessorInit(g);
    return ${call}(&a, xyz);
}`;
};

// The HDDA empty-space query; CPU twin tools vdbFrameSkipEmpty (see there for the contract).
const SKIP_EMPTY_WGSL = /* wgsl */ `fn vdbFrameSkipEmpty(g: u32, o: vec3f, d: vec3f, t0: f32, t1: f32, cutoff: f32, dilate: u32) -> vec2f {
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
}`;

class SparseSource implements VDBThreeSparseSource {
  readonly kind = "sparse" as const;
  readonly grids: VDBGridNode;
  readonly ids: Readonly<VDBSparseSourceGrids>;
  readonly lod: VDBSparseLod;
  readonly normalize: "none" | "clamp01";
  readonly channels: VDBVolumeChannels;
  // G16a: the layout fields below are re-read after an image change when `followImage` is on.
  indexBox: VDBIndexBox;
  auxBoxes: Record<"mask" | "albedo" | "emission", VDBIndexBox | null>;
  texels: [number, number, number];
  lodLevels: number;
  stats: { meanNonZero: number; max: number; activeVoxels: number };
  statsUpdates = 0;
  boxToSource: Float32Array;
  bbox: [Vec3, Vec3];
  pyramid: VDBSparsePyramid | null;
  gpuBytes: { pool: number; pyramid: number; pyramidScratch: number };
  readonly fastTrilinear: boolean;
  readonly cacheAccessor: boolean;
  readonly followImage: boolean;
  pyramidBuilds = 0;
  layoutUpdates = 0;
  /** G16a: the newest channel imageRevision the layout describes, the pool revision last checked, the uniforms to re-apply. */
  private layoutRevision = -1;
  private layoutPoolRevision = -1;
  private readonly layoutListeners = new Set<() => void>();
  /** G16a: `pyramidMax` as given (the pyramid is re-laid out with it). */
  private readonly pyramidMaxOption: { pyramidMax?: number };
  /** G16a: the atlas node's texture when the layout has no atlas (a 1³ zero texture of its own). */
  private emptyAtlas: THREE.Data3DTexture | null = null;
  private poolRevisionSeen = -1;
  private readonly poolIds: number[];
  private versionSeen = { poolRevision: -1, value: 0 };
  private nodeAverageCache: ((...a: unknown[]) => VDBTslNode) | null = null;
  private nodeLodCache: ((...a: unknown[]) => VDBTslNode) | null = null;
  private skipCache: ((...a: unknown[]) => VDBTslNode) | null = null;
  private sampleCache: ((...a: unknown[]) => VDBTslNode) | null = null;
  /** K3.2e: the density imageRevision the stats describe, the pool revision last checked, and a readback in flight. */
  private statsRevision = -1;
  private statsPoolRevisionSeen = -1;
  private statsReadback: Promise<boolean> | null = null;
  /** K5.9a: each of this source's uniforms re-applies its density scale with the new mean when a readback lands. */
  private readonly statsListeners = new Set<() => void>();

  constructor(grids: VDBGridNode, ids: VDBSparseSourceGrids, o: VDBSparseSourceOptions) {
    const pool = grids.pool as unknown as Partial<PoolInfoLike>;
    if (typeof pool.info !== "function") throw bad("the VDBGridNode's pool has no info(); pass a VDBGridPool.");
    const info = (id: number) => (pool as PoolInfoLike).info(id);
    const lod = o.lod ?? "coarse-pyramid";
    if (lod !== "none" && lod !== "coarse-pyramid" && lod !== "node-average") throw bad(`unknown lod ${JSON.stringify(lod)}; use 'none', 'coarse-pyramid' or 'node-average'.`);
    const normalize = o.normalize ?? "none";
    if (normalize !== "none" && normalize !== "clamp01") throw bad(`unknown normalize ${JSON.stringify(normalize)}; use 'none' or 'clamp01'.`);
    const helpers = new Set<string>(grids.options.helpers ?? []);
    if (grids.options.profile !== "march" || !helpers.has("cellBoundTrilinear") || !helpers.has("hddaCellEnd") || (lod === "node-average" && !helpers.has("nodeAverage"))) {
      throw bad(
        `make the VDBGridNode with profile 'march' and helpers ['cellBoundTrilinear', 'hddaCellEnd'${lod === "node-average" ? ", 'nodeAverage'" : ""}] ` +
          `(got profile '${grids.options.profile}', helpers [${[...helpers].join(", ")}]).`,
      );
    }
    const fastTrilinear = o.fastTrilinear ?? helpers.has("fastTrilinear");
    if (fastTrilinear && !helpers.has("fastTrilinear")) throw bad("fastTrilinear needs the VDBGridNode helper 'fastTrilinear' (add it to helpers), or pass fastTrilinear: false.");
    if (fastTrilinear && grids.options.sampler === "nearest") throw bad("fastTrilinear needs the VDBGridNode's sampler 'trilinear'.");
    this.fastTrilinear = fastTrilinear;
    this.cacheAccessor = o.cacheAccessor ?? true;
    const followImage = o.followImage ?? false;
    if (typeof followImage !== "boolean") throw bad("followImage must be a boolean.");
    if (followImage) {
      const fixed = (["indexBox", "auxIndexBoxes", "bbox"] as const).filter((k) => o[k] !== undefined);
      if (fixed.length) throw bad(`followImage re-reads every box from the pool; drop ${fixed.join(", ")} (fixed boxes).`);
      if (lod === "coarse-pyramid" && o.pyramidBuild === "cpu") throw bad("followImage rebuilds the pyramid from the pool on the GPU; use pyramidBuild 'gpu' (the default).");
    }
    this.followImage = followImage;
    this.pyramidMaxOption = o.pyramidMax !== undefined ? { pyramidMax: o.pyramidMax } : {};
    const density = info(ids.density);
    if (lod === "node-average" && !density.hasStats) throw bad("lod 'node-average' reads node statistics; this grid has none (rebuild it with stats).");
    this.grids = grids;
    this.ids = Object.freeze({ ...ids, ...(Array.isArray(ids.albedo) ? { albedo: [...ids.albedo] as unknown as [number, number, number] } : {}) });
    this.lod = lod;
    this.normalize = normalize;
    this.indexBox = o.indexBox ? { min: [...o.indexBox.min], max: [...o.indexBox.max] } : { min: [...density.indexBBox.min], max: [...density.indexBBox.max] };
    const albedoId = Array.isArray(ids.albedo) ? ids.albedo[0] : (ids.albedo as number | undefined);
    const box = (id: number | undefined, given?: VDBIndexBox) =>
      id === undefined ? null : given ? { min: [...given.min] as Vec3, max: [...given.max] as Vec3 } : { min: [...info(id).indexBBox.min] as Vec3, max: [...info(id).indexBBox.max] as Vec3 };
    const aux = o.auxIndexBoxes ?? {};
    this.auxBoxes = { mask: box(ids.mask, aux.mask), albedo: box(albedoId, aux.albedo), emission: box(ids.emission, aux.emission) };
    this.channels = { mask: ids.mask !== undefined, albedo: ids.albedo !== undefined, emission: ids.emission !== undefined };
    const b = this.indexBox;
    this.texels = [b.max[0] - b.min[0] + 1, b.max[1] - b.min[1] + 1, b.max[2] - b.min[2] + 1];
    if (this.texels.some((t) => !(t > 0))) throw bad("the density index box is empty.");
    const image = o.images?.density;
    if (o.stats) this.stats = { ...o.stats };
    else if (image) this.stats = imageStats(image, normalize === "clamp01");
    else throw bad("pass images.density (the CPU image, for the density stats) or stats.");
    this.statsRevision = (density as { imageRevision?: number }).imageRevision ?? -1;
    // Local unit box → sample index space: p = min − ½ + (local + ½)·size (column-major).
    const [sx, sy, sz] = this.texels;
    this.boxToSource = new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, b.min[0] - 0.5 + 0.5 * sx, b.min[1] - 0.5 + 0.5 * sy, b.min[2] - 0.5 + 0.5 * sz, 1]);
    this.bbox = o.bbox ? [[...o.bbox[0]], [...o.bbox[1]]] : worldBox(b, density.indexToWorld);

    let pyramid: VDBSparsePyramid | null = null;
    if (lod === "coarse-pyramid") {
      const build = o.pyramidBuild ?? "gpu";
      if (build !== "gpu" && build !== "cpu") throw bad(`unknown pyramidBuild ${JSON.stringify(build)}; use 'gpu' or 'cpu'.`);
      const pyramidMax = o.pyramidMax !== undefined ? { pyramidMax: o.pyramidMax } : {};
      if (build === "cpu") {
        if (!image) throw bad("pyramidBuild 'cpu' builds the pyramid from images.density; pass it, or use pyramidBuild 'gpu' (the default).");
        const p = vdbBoxPyramid(image, { indexBox: b, ...pyramidMax });
        pyramid = {
          build,
          level: p.level,
          dimensions: p.dimensions,
          atlas: p.atlas,
          lodLevels: p.lodLevels,
          data: p.data,
          map: halfTexture(p.data, p.dimensions),
          mip: p.atlas ? halfTexture(p.atlas.data, p.atlas.dimensions) : null,
          gpu: null,
        };
      } else {
        const renderer = (grids as { renderer?: unknown }).renderer;
        let device: GPUDevice;
        try {
          device = webgpuBackendOf(renderer, "vdbSparseSource").device;
        } catch {
          throw bad("pyramidBuild 'gpu' (the default for lod 'coarse-pyramid') needs the VDBGridNode's initialized WebGPURenderer; use pyramidBuild 'cpu' with images.density otherwise.");
        }
        const initTexture = (renderer as { initTexture?: (t: THREE.Texture) => void }).initTexture;
        if (typeof initTexture !== "function") throw bad("pyramidBuild 'gpu' needs renderer.initTexture (three r187 WebGPURenderer).");
        // G16a: a following source compiles the pyramid kernels of every type the node decodes now, so a swap to
        // another grid type creates no pipeline.
        if (followImage) warmBoxPyramidPipelines(device, grids.options.types);
        pyramid = this.gpuPyramid(b, ids.density);
      }
    }
    this.pyramid = pyramid;
    this.lodLevels = lod === "none" ? 0 : lod === "node-average" ? 7 : pyramid!.lodLevels;
    const poolIds = new Set([ids.density, ids.mask, ids.emission, ...(Array.isArray(ids.albedo) ? ids.albedo : [ids.albedo])].filter((v): v is number => v !== undefined));
    this.poolIds = [...poolIds];
    this.gpuBytes = this.measure();
    this.layoutRevision = this.poolIds.reduce((r, id) => Math.max(r, info(id).imageRevision ?? 0), 0);
    this.refreshPyramid();
  }

  /** GPU bytes of the channel grids' pool ranges and the pyramid (see the interface). */
  private measure(): { pool: number; pyramid: number; pyramidScratch: number } {
    const pool = this.grids.pool as unknown as PoolInfoLike;
    const p = this.pyramid;
    const atlasTexels = p?.atlas ? p.atlas.dimensions[0] * p.atlas.dimensions[1] * p.atlas.dimensions[2] : 0;
    const pyramidBytes = p ? 2 * (p.dimensions[0] * p.dimensions[1] * p.dimensions[2] + atlasTexels) : 0;
    return { pool: this.poolIds.reduce((s, id) => s + pool.info(id).capacity, 0), pyramid: pyramidBytes, pyramidScratch: p?.gpu?.gpuBytes.scratch ?? 0 };
  }

  /** The GPU pyramid of `box` (K3.2a): three textures the core's `vdbCreateBoxPyramid` writes into. */
  private gpuPyramid(box: VDBIndexBox, densityId: number): VDBSparsePyramid {
    const renderer = this.grids.renderer as unknown as { initTexture(t: THREE.Texture): void };
    const device = webgpuBackendOf(renderer, "vdbSparseSource").device;
    const texels = [0, 1, 2].map((a) => box.max[a]! - box.min[a]! + 1);
    const layout = vdbBoxPyramidLayout(texels, this.pyramidMaxOption);
    const map = halfTexture(null, layout.dimensions);
    const mip = layout.atlas ? halfTexture(null, layout.atlas.dimensions) : null;
    map.name = "vdbSparseSource pyramid base";
    if (mip) mip.name = "vdbSparseSource pyramid atlas";
    for (const t of [map, mip]) if (t) renderer.initTexture(t);
    const gpu = vdbCreateBoxPyramid(device, this.grids.pool as never, densityId, {
      indexBox: box,
      ...this.pyramidMaxOption,
      targets: { base: getGPUTexture(renderer as never, map), atlas: mip ? getGPUTexture(renderer as never, mip) : null },
      label: "vdbSparseSource pyramid",
    });
    return {
      build: "gpu",
      level: layout.level,
      dimensions: [...layout.dimensions],
      atlas: layout.atlas ? { dimensions: [...layout.atlas.dimensions], levels: layout.atlas.levels.length } : null,
      lodLevels: layout.lodLevels,
      map,
      mip,
      gpu,
    };
  }

  /** G16a: the atlas texture a following source's shaders bind: the pyramid's, else a 1³ zero texture (never read). */
  atlasTexture(): THREE.Data3DTexture {
    if (this.pyramid?.mip) return this.pyramid.mip;
    if (!this.emptyAtlas) {
      this.emptyAtlas = halfTexture(new Float32Array(1), [1, 1, 1]);
      this.emptyAtlas.name = "vdbSparseSource empty atlas";
    }
    return this.emptyAtlas;
  }

  /**
   * G16a (`followImage`): re-reads the layout when a channel grid's image changed and every channel grid is resident
   * again (the update that writes the new image has run: the GPU must not read it through the old boxes, nor the
   * old image through the new ones). Cheap when the pool's revision did not move; called from `version` (a
   * `VDBFrameVolume` asks once per frame, before it places its box) and from the render updates.
   */
  syncLayout(): boolean {
    if (!this.followImage) return false;
    const pool = this.grids.pool as unknown as { revision?: number } & PoolInfoLike;
    if (typeof pool.revision !== "number" || pool.revision === this.layoutPoolRevision) return false;
    let revision = 0;
    let resident = true;
    for (const id of this.poolIds) {
      let i: ReturnType<PoolInfoLike["info"]>;
      try {
        i = pool.info(id);
      } catch {
        return false; // a removed grid: keep the last layout
      }
      revision = Math.max(revision, i.imageRevision ?? 0);
      if (i.resident === false) resident = false;
    }
    if (revision === this.layoutRevision) {
      this.layoutPoolRevision = pool.revision;
      return false;
    }
    if (!resident) return false; // look again once the update has written it
    this.relayout();
    this.layoutRevision = revision;
    this.layoutPoolRevision = pool.revision;
    return true;
  }

  private relayout(): void {
    const pool = this.grids.pool as unknown as PoolInfoLike;
    const ids = this.ids;
    const density = pool.info(ids.density);
    const b: VDBIndexBox = { min: [...density.indexBBox.min], max: [...density.indexBBox.max] };
    const old = this.indexBox;
    const moved = b.min.some((v, a) => v !== old.min[a]) || b.max.some((v, a) => v !== old.max[a]);
    this.indexBox = b;
    this.texels = [b.max[0] - b.min[0] + 1, b.max[1] - b.min[1] + 1, b.max[2] - b.min[2] + 1];
    const [sx, sy, sz] = this.texels;
    this.boxToSource = new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, b.min[0] - 0.5 + 0.5 * sx, b.min[1] - 0.5 + 0.5 * sy, b.min[2] - 0.5 + 0.5 * sz, 1]);
    this.bbox = worldBox(b, density.indexToWorld);
    const auxBox = (id: number | undefined): VDBIndexBox | null => (id === undefined ? null : { min: [...pool.info(id).indexBBox.min] as Vec3, max: [...pool.info(id).indexBBox.max] as Vec3 });
    this.auxBoxes = { mask: auxBox(ids.mask), albedo: auxBox(Array.isArray(ids.albedo) ? ids.albedo[0] : (ids.albedo as number | undefined)), emission: auxBox(ids.emission) };
    // A new box needs a pyramid of its size (and box min); the same box rebuilds in place (refreshPyramid).
    const previous = this.pyramid;
    if (moved && previous?.gpu) {
      this.pyramid = this.gpuPyramid(b, ids.density);
      this.lodLevels = this.pyramid.lodLevels;
      this.poolRevisionSeen = -1;
    }
    this.gpuBytes = this.measure();
    this.layoutUpdates++;
    for (const apply of this.layoutListeners) apply();
    // After the uniforms point at the new textures: the old ones go (three destroys them after submitted work).
    if (previous !== this.pyramid && previous) {
      previous.gpu?.destroy();
      previous.map.dispose();
      previous.mip?.dispose();
    }
  }

  /**
   * Content version: the newest `imageRevision` of the channel grids (K3.2a), so a `pool.replace` (a
   * sequence frame) changes it and the 'frame' lighting caster signature (V44) draws the shadow grid
   * again. Read through the pool's `revision` first (cheap when nothing changed).
   */
  get version(): number {
    // G16a: a following source's version is the image revision its layout describes (it moves once the new image is
    // resident and the boxes were re-read), so a VDBFrameVolume re-reads its box and the uniforms then.
    if (this.followImage) {
      this.syncLayout();
      return this.layoutRevision;
    }
    const pool = this.grids.pool as unknown as { revision?: number; info(id: number): { imageRevision?: number } };
    if (typeof pool.revision !== "number") return 0;
    if (pool.revision !== this.versionSeen.poolRevision) {
      let value = 0;
      for (const id of this.poolIds) {
        try {
          value = Math.max(value, pool.info(id).imageRevision ?? 0);
        } catch {
          // A removed grid: keep the others' revision.
        }
      }
      this.versionSeen = { poolRevision: pool.revision, value };
    }
    return this.versionSeen.value;
  }

  refreshPyramid(): boolean {
    this.syncLayout();
    const gpu = this.pyramid?.gpu;
    if (!gpu) return false;
    // Cheap test first: the pool's revision moves on any change; the grid's imageRevision tells whether this one did.
    const pool = this.grids.pool as unknown as { revision: number; device: GPUDevice };
    if (pool.revision === this.poolRevisionSeen) return false;
    const device = pool.device;
    const encoder = device.createCommandEncoder({ label: "vdbSparseSource pyramid" });
    if (!gpu.update(encoder)) {
      // Not resident yet (the update that writes it is still to come): look again next time.
      if (!gpu.stale) this.poolRevisionSeen = pool.revision;
      return false;
    }
    device.queue.submit([encoder.finish()]);
    this.poolRevisionSeen = pool.revision;
    this.pyramidBuilds++;
    return true;
  }

  refreshStats(): Promise<boolean> {
    if (this.statsReadback) return this.statsReadback;
    const pool = this.grids.pool as unknown as {
      revision?: number;
      device?: GPUDevice;
      buffer?: GPUBuffer;
      info(id: number): { imageRevision?: number; resident?: boolean; imageOffset?: number | null; imageBytes?: number };
    };
    // Cheap test first (once per render): the pool's revision moves on any change.
    if (typeof pool.revision !== "number" || pool.revision === this.statsPoolRevisionSeen || !pool.device || !pool.buffer) return Promise.resolve(false);
    let info: ReturnType<typeof pool.info>;
    try {
      info = pool.info(this.ids.density);
    } catch {
      return Promise.resolve(false); // removed: keep the last stats
    }
    const revision = info.imageRevision ?? -1;
    if (revision === this.statsRevision) {
      this.statsPoolRevisionSeen = pool.revision;
      return Promise.resolve(false);
    }
    // Not written yet (the update that writes it is still to come): look again next time.
    if (!info.resident || info.imageOffset === null || info.imageOffset === undefined || !info.imageBytes) return Promise.resolve(false);
    const { device, buffer } = pool;
    const bytes = info.imageBytes;
    const staging = device.createBuffer({ label: "vdbSparseSource stats readback", size: bytes, usage: 0x1 | 0x8 }); // MAP_READ | COPY_DST (no WebGPU globals in Node tests)
    const encoder = device.createCommandEncoder({ label: "vdbSparseSource stats readback" });
    encoder.copyBufferToBuffer(buffer, info.imageOffset, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    const seen = pool.revision;
    this.statsReadback = staging
      .mapAsync(0x1) // GPUMapMode.READ
      .then(() => {
        const image = new Uint32Array(staging.getMappedRange().slice(0));
        this.stats = imageStats(image, this.normalize === "clamp01");
        this.statsRevision = revision;
        this.statsPoolRevisionSeen = seen;
        this.statsUpdates++;
        for (const apply of this.statsListeners) apply();
        return true;
      })
      .catch(() => false)
      .finally(() => {
        staging.destroy();
        this.statsReadback = null;
      });
    return this.statsReadback;
  }

  nodeAverageFn(): (...a: unknown[]) => VDBTslNode {
    this.nodeAverageCache ??= wgslFn(NODE_AVERAGE_WGSL, [this.grids.module] as never) as unknown as (...a: unknown[]) => VDBTslNode;
    return this.nodeAverageCache;
  }

  private nodeLodFn(): (...a: unknown[]) => VDBTslNode {
    this.nodeLodCache ??= wgslFn(nodeLodWgsl(this.fastTrilinear ? "vdbSampleFast" : "vdbSample"), [this.grids.module, this.nodeAverageFn()] as never) as unknown as (...a: unknown[]) => VDBTslNode;
    return this.nodeLodCache;
  }

  /** `vdbFrameSample(g, xyz) -> f32`: level 0 (index space) as configured (`fastTrilinear`, `cacheAccessor`). */
  sampleFn(): (...a: unknown[]) => VDBTslNode {
    this.sampleCache ??= wgslFn(sampleWgsl(this.fastTrilinear, this.cacheAccessor), (this.cacheAccessor ? [this.grids.module, code(SAMPLE_CACHE_WGSL, [this.grids.module])] : [this.grids.module]) as never) as unknown as (
      ...a: unknown[]
    ) => VDBTslNode;
    return this.sampleCache;
  }

  skipEmptyFn(): (...a: unknown[]) => VDBTslNode {
    this.skipCache ??= wgslFn(SKIP_EMPTY_WGSL, [this.grids.module] as never) as unknown as (...a: unknown[]) => VDBTslNode;
    return this.skipCache;
  }

  createUniforms(p: VDBVolumeUniformParameters): VDBSparseVolumeUniforms {
    let densityScale = p.densityScale ?? 1.0;
    let normalize = p.normalizeDensity ?? true;
    const boxUniform = (bx: VDBIndexBox | null) =>
      bx ? { min: uniform(new THREE.Vector3(...bx.min)), size: uniform(new THREE.Vector3(bx.max[0] - bx.min[0] + 1, bx.max[1] - bx.min[1] + 1, bx.max[2] - bx.min[2] + 1)) } : null;
    const levels = uniform(this.lodLevels);
    // K3.2e: the mean follows the density image (refreshStats, checked once per render, reads a replaced image
    // back). K5.9a: the new mean is applied once, when the readback lands; the render update never writes the
    // value, so a value set per draw (VDBFrameVolume's densityScale, through setVolumeUniforms) stays as set.
    const densityScaleU = uniform(vdbDensityScaleOf(this, densityScale, normalize)).onRenderUpdate(() => {
      this.syncLayout();
      void this.refreshStats();
      return undefined;
    });
    this.statsListeners.add(() => {
      densityScaleU.value = vdbDensityScaleOf(this, densityScale, normalize);
    });
    const pyr = this.pyramid;
    // G16a: a following source samples its pyramid through texture nodes made here (one pair per uniforms), so a new
    // layout swaps their textures; the atlas node always exists (a 1³ texture when the layout has none).
    const textures = this.followImage && pyr ? { map: texture3D(pyr.map), mip: texture3D(this.atlasTexture()) } : null;
    const u: VDBSparseVolumeUniforms = {
      worldScale: uniform(p.worldScale ?? 1.0),
      gridTexels: uniform(new THREE.Vector3(...this.texels)),
      mipAtlasSize: uniform(new THREE.Vector3(1, 1, 1)),
      maps: { density: null, mask: null, albedo: null, emission: null },
      mipMaps: { density: null, mask: null, albedo: null, emission: null },
      // getSampledLod reads density (and mask) levels; aux channels are level 0 only but report the same.
      mipLevels: { density: levels, mask: levels, albedo: levels, emission: levels },
      baseColor: uniform(new THREE.Color(p.baseColor ?? 0xffffff)),
      extinction: uniform(p.extinction ?? 100.0),
      densityScale: densityScaleU,
      densityCutoff: uniform(p.densityCutoff ?? 0.0),
      emissiveIntensity: uniform(p.emissiveIntensity ?? 1.0),
      temperatureRange: uniform(new THREE.Vector2(...(p.temperatureRange ?? [0.0, 6500.0]))),
      offset3D: uniform(new THREE.Vector3().copy((p.offset3D ?? { x: 0, y: 0, z: 0 }) as THREE.Vector3)),
      wrap3D: uniform(p.wrap3D ?? THREE.ClampToEdgeWrapping, "int"),
      boxes: { density: boxUniform(this.indexBox), mask: boxUniform(this.auxBoxes.mask), albedo: boxUniform(this.auxBoxes.albedo), emission: boxUniform(this.auxBoxes.emission) },
      // The GPU pyramid rebuilds (on an image change) from here: once per render that samples it.
      pyramidLevel: pyr?.gpu ? uniform(pyr.level).onRenderUpdate(() => void this.refreshPyramid()) : uniform(pyr?.level ?? 0),
      pyramidLevels: uniform(pyr?.atlas?.levels ?? 0),
      pyramidTexels: uniform(new THREE.Vector3(...(pyr?.dimensions ?? [1, 1, 1]))),
      pyramidAtlasSize: uniform(new THREE.Vector3(...(pyr?.atlas?.dimensions ?? [1, 1, 1]))),
      update: (s, o = {}) => {
        if (s !== this) throw bad("these uniforms belong to another source.");
        if (o.densityScale !== undefined) densityScale = o.densityScale;
        if (o.normalizeDensity !== undefined) normalize = o.normalizeDensity;
        densityScaleU.value = vdbDensityScaleOf(this, densityScale, normalize);
        if (this.followImage) applyLayout();
      },
    };
    // G16a: the layout as uniform values (and texture swaps), applied again after every re-read.
    const setBox = (b: { min: VDBTslNode; size: VDBTslNode } | null, bx: VDBIndexBox | null) => {
      if (!b || !bx) return;
      b.min.value.set(...bx.min);
      b.size.value.set(bx.max[0] - bx.min[0] + 1, bx.max[1] - bx.min[1] + 1, bx.max[2] - bx.min[2] + 1);
    };
    const applyLayout = () => {
      u.gridTexels.value.set(...this.texels);
      setBox(u.boxes.density, this.indexBox);
      setBox(u.boxes.mask, this.auxBoxes.mask);
      setBox(u.boxes.albedo, this.auxBoxes.albedo);
      setBox(u.boxes.emission, this.auxBoxes.emission);
      levels.value = this.lodLevels;
      const p = this.pyramid;
      if (!p) return;
      u.pyramidLevel.value = p.level;
      u.pyramidLevels.value = p.atlas?.levels ?? 0;
      u.pyramidTexels.value.set(...p.dimensions);
      u.pyramidAtlasSize.value.set(...(p.atlas?.dimensions ?? [1, 1, 1]));
      if (textures) {
        textures.map.value = p.map;
        textures.mip.value = this.atlasTexture();
      }
    };
    if (this.followImage) {
      this.layoutListeners.add(applyLayout);
      if (textures) pyramidTextures.set(u, textures);
    }
    return u;
  }

  createNodes(u0: VDBVolumeUniforms, o: VDBVolumeNodesOptions): VDBVolumeNodes {
    const u = u0 as VDBSparseVolumeUniforms;
    if (!u.boxes) throw bad("createNodes needs this source's uniforms (vdbVolumeUniforms(source)).");
    const { grids, ids, lod } = this;
    const useMipmaps = o.useMipmaps ?? this.lodLevels > 0;
    const gridUv = vdbGridUv(u);
    const clamp01 = this.normalize === "clamp01";
    // G16a: a following source's LOD range is a uniform (it changes with the layout); otherwise a constant.
    const lodMax = this.followImage ? u.mipLevels.density : float(this.lodLevels);
    const followTextures = this.followImage ? pyramidTextures.get(u) : undefined;

    // uvw → sample index space, clamped to the voxels (his clamp-to-edge).
    const samplePos = (box: { min: VDBTslNode; size: VDBTslNode }, uvw: VDBTslNode) =>
      clamp(box.min.add(uvw.mul(box.size)).sub(0.5), box.min, box.min.add(box.size).sub(1.0));
    const level0 =
      this.fastTrilinear || this.cacheAccessor ? (id: number, p: VDBTslNode) => this.sampleFn()(uint(id), p) : (id: number, p: VDBTslNode) => vdbSample(grids, id, p, { space: "index" });

    const densityAt = (uvw: VDBTslNode, lodIn: VDBTslNode): VDBTslNode => {
      const p = samplePos(u.boxes.density!, uvw).toVar();
      if (!useMipmaps || lod === "none") return level0(ids.density, p);
      const l = clamp(lodIn, 0.0, lodMax).toVar();
      if (lod === "node-average") return this.nodeLodFn()(uint(ids.density), p, l);
      const r = float(0.0).toVar();
      // coarse-pyramid
      const pyr = this.pyramid!;
      if (this.followImage && !followTextures) throw bad("createNodes needs this source's uniforms (vdbVolumeUniforms(source)).");
      const pyrMap = followTextures ? followTextures.map : texture3D(pyr.map);
      const pyrMip = followTextures ? followTextures.mip : pyr.mip ? texture3D(pyr.mip) : null;
      const { sampleMipmapped } = vdbMipSampler(u.pyramidTexels, u.pyramidAtlasSize);
      const coarse = (rel: VDBTslNode) => (pyrMip ? sampleMipmapped(pyrMap, pyrMip, u.pyramidLevels, uvw, rel).r : pyrMap.sample(uvw).level(float(0.0)).r);
      // K3.2d: one level-0 call site (each inlines the whole tree walk), shared by the two branches below.
      const fine = float(0.0).toVar();
      If(l.lessThan(u.pyramidLevel), () => {
        fine.assign(level0(ids.density, p));
      });
      If(l.lessThanEqual(0.0), () => {
        r.assign(fine);
      })
        .ElseIf(l.lessThan(u.pyramidLevel), () => {
          r.assign(mix(fine, coarse(float(0.0)), l.div(u.pyramidLevel)));
        })
        .Else(() => {
          r.assign(coarse(l.sub(u.pyramidLevel)));
        });
      return r;
    };

    // L892
    const getLod = (localSize: VDBTslNode) => max(0.0, log2(localSize.mul(vdbMaxComponent(u.gridTexels))));

    // L914–936: extinction per world unit; the mask at local + 0.5 (no offset or wrap), as his.
    const sampleExtinction = (localPosition: VDBTslNode, lodIn: VDBTslNode) => {
      const mask = this.channels.mask ? level0(ids.mask!, samplePos(u.boxes.mask!, localPosition.add(0.5))) : float(1.0);
      const raw = densityAt(gridUv(localPosition), lodIn);
      const value = (clamp01 ? clamp(raw, 0.0, 1.0) : raw).mul(mask);
      return select(value.greaterThan(u.densityCutoff), value, float(0.0)).mul(u.densityScale).mul(u.extinction).div(u.worldScale);
    };

    // L960–971 (level 0)
    const sampleAlbedo = (localPosition: VDBTslNode) => {
      if (!this.channels.albedo) return u.baseColor.mul(vec3(1.0));
      const p = samplePos(u.boxes.albedo!, gridUv(localPosition)).toVar();
      const a = ids.albedo!;
      const rgb = Array.isArray(a) ? vec3(level0(a[0], p), level0(a[1], p), level0(a[2], p)) : vec3(level0(a as number, p));
      return u.baseColor.mul(clamp(rgb, 0.0, 1.0));
    };

    // L990–1017 (level 0), colour via the caller's hook (V24).
    const emissionColor = o.emissionColor ?? (() => vec3(1.0));
    const sampleEmission = (localPosition: VDBTslNode) => {
      if (!this.channels.emission) return vec3(0.0);
      const range = u.temperatureRange;
      const value = level0(ids.emission!, samplePos(u.boxes.emission!, gridUv(localPosition)));
      const temperature = mix(range.x, range.y, value);
      const power = pow(temperature.div(max(range.y, 1.0)), 4.0);
      const color = select(temperature.greaterThan(0.0), emissionColor(temperature), vec3(0.0));
      return color.mul(power).mul(u.emissiveIntensity);
    };

    // HDDA empty-space query (R1 §4.3 "HDDA gains"): exact only where the bound covers every voxel a
    // sample reads: level 0 (the trilinear 2x2x2 cell block), or a coarse level whose taps reach at
    // most 8 voxels past the cell (dilated 3x3x3 block): node averages up to level 3, pyramid bases
    // up to level 2. Wrapping or an offset moves samples off the ray's cells: no skipping then.
    const skipEmpty = (origin: VDBTslNode, direction: VDBTslNode, t0: VDBTslNode, t1: VDBTslNode, lodIn: VDBTslNode) => {
      const box = u.boxes.density!;
      const l = useMipmaps && lod !== "none" ? clamp(lodIn, 0.0, lodMax) : float(0.0);
      const dilate = int(0).toVar();
      // Cost rule (K3.2 A/B): a walk visits 8³ cells at 8 node lookups each, so it only pays while samples are at
      // most one cell apart (the march's lod ≤ 3, i.e. steps ≤ 8 voxels); sparser marches sample everything.
      const enabled = u.wrap3D
        .equal(int(THREE.ClampToEdgeWrapping))
        .and(vdbMaxComponent(u.offset3D.abs()).equal(0.0))
        .and(lodIn.lessThanEqual(3.0))
        .toVar();
      if (useMipmaps && lod === "node-average") {
        If(l.greaterThan(0.0), () => {
          dilate.assign(1);
          If(l.greaterThan(3.0), () => {
            enabled.assign(false);
          });
        });
      } else if (useMipmaps && lod === "coarse-pyramid") {
        If(l.greaterThan(0.0), () => {
          dilate.assign(1);
          If(u.pyramidLevel.greaterThan(2.0).or(l.greaterThan(2.0)), () => {
            enabled.assign(false);
          });
        });
      }
      const result = vec2(t0, 1e30).toVar();
      If(enabled, () => {
        const o2 = box.min.sub(0.5).add(origin.add(0.5).mul(box.size));
        const d2 = direction.mul(box.size);
        // The mask multiplies the density, so with a mask only a zero bound is a safe skip.
        const cutoff = this.channels.mask ? float(0.0) : max(u.densityCutoff, 0.0);
        result.assign(this.skipEmptyFn()(uint(ids.density), o2, d2, t0, t1, cutoff, dilate.toUint()));
      });
      return result;
    };

    return { getLod, sampleExtinction, sampleAlbedo, sampleEmission, ...vdbBoxNodes, skipEmpty };
  }

  dispose(): void {
    this.pyramid?.gpu?.destroy();
    this.pyramid?.map.dispose();
    this.pyramid?.mip?.dispose();
    this.emptyAtlas?.dispose();
    this.layoutListeners.clear();
  }
}

/**
 * A sparse volume source over pool grids (see the file comment). `grids` must be a `VDBGridNode`
 * made with profile `'march'` and helpers `['cellBoundTrilinear', 'hddaCellEnd']` (+ `'nodeAverage'`
 * for that LOD). The source holds no grid buffer of its own (the pool's), only the pyramid textures
 * (and, building on the GPU, the build's scratch buffers).
 */
export function vdbSparseSource(grids: VDBGridNode, ids: VDBSparseSourceGrids, options: VDBSparseSourceOptions = {}): VDBThreeSparseSource {
  if (!grids || typeof grids !== "object" || !("module" in grids)) throw bad("pass a VDBGridNode.");
  if (!ids || !Number.isInteger(ids.density)) throw bad("ids.density must be a pool grid id.");
  return new SparseSource(grids, ids, options);
}
