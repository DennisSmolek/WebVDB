// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0), through the TSL sparse source
// (`three/sparse-source.ts`, K3.2): frameShaderChunks.js L729–763 (createVolumeUniforms, the source half) and
// L834–1028 (createVolumeNodes).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free; one source per pool serving every volume through a uniform table indexed by `v`; the
// coarse pyramids of all volumes packed in one r16float texture. Divergences: DIVERGES.md (K3.2 rows, K6.7 row).

/**
 * `vdbSparseSourceGpu(pool, options)` (K6.7, design K6-core-renderer.md §2.1): the sparse volume source of the core
 * renderer, on raw WebGPU. One source serves every volume of one `VDBGridPool`:
 *
 *   const source = vdbSparseSourceGpu(pool, { followImage: true });
 *   const slot = source.volume({ density: id });       // a VDBVolumeSlot; slot.v indexes the source's table
 *   // per frame, in the host's encoder:
 *   pool.update(encoder);
 *   source.update(encoder);                             // pyramids, layouts, the table (after the pool's update)
 *   // bind group 2: source.layoutEntries(slot2) / source.bindEntries(slot2) (rebuild when source.version moves)
 *   // WGSL: source.moduleWgsl(slot2) (or the renderer's one module with source.module's helpers) + source.wgsl(...)
 *
 * Group 2 at `slot` = { group, firstBinding: b }: b the pool (the VDB module's storage buffer, D10), b + 1 the volume
 * table (`array<VdbSrcVolume, maxVolumes>`, uniform), and with `lod: 'coarse-pyramid'` b + 2 the packed pyramid
 * atlas (r16float `texture_3d<f32>`) and b + 3 its sampler (linear, clamp to edge).
 *
 * The WGSL depends only on the options (LOD strategy, channels, `maxVolumes`, sampling options): boxes, pool ids,
 * LOD ranges and pyramid placements are values in the table, so adding volumes, a sequence frame (`pool.replace`)
 * or a coarse-first swap (`followImage`) never changes it, and with `types` fixed (the default
 * `VDB_ALL_SCALAR_TYPES`; the pyramid kernels of every type are compiled at creation when following images) a
 * swap creates no pipeline and no shader module (G16a).
 *
 * Per volume (see {@link VDBSparseVolumeGpu}): `texels`, `lodLevels`, `stats`, `boxToSource`, `bbox` and `version`
 * describe that volume; the source's own copies of the {@link VDBVolumeSourceInfo} fields are aggregates.
 */

import { VDBError } from "../errors.js";
import type { VDBGridInfo, VDBGridPool } from "../pool/pool.js";
import { VDB_ALL_SCALAR_TYPES, type VDBGridTypeName, type VDBSlot } from "../types.js";
import { vdbModuleEntries, vdbModuleWgsl } from "../wgsl/generate.js";
import type { VDBModuleHelper } from "../wgsl/types.js";
import { vdbCreateBoxPyramid, warmBoxPyramidPipelines, type VDBBoxPyramidGpu } from "../compute/pyramid.js";
import { vdbBoxPyramidLayout, vdbPackPyramidBlocks, type VDBBoxPyramidLayout } from "../../tools/frame/pyramid-layout.js";
import { vdbSparseDensityStats } from "../../tools/frame/sparse.js";
import type { VDBVolumeChannels, VDBVolumeSlot, VDBVolumeSource, VDBVolumeSourceWgslOptions } from "./types.js";
import { packSparseRecord, sparseBindings, sparseSourceWgsl, SPARSE_FLAG, VDB_SPARSE_RECORD_BYTES, type SparseRecord, type SparseWgslConfig } from "./sparse-wgsl.js";

type Vec3 = [number, number, number];

/** WebGPU flag values (the globals do not exist in Node). */
const BUFFER = { MAP_READ: 0x1, COPY_DST: 0x8, UNIFORM: 0x40 } as const;
const TEXTURE = { COPY_SRC: 0x1, COPY_DST: 0x2, TEXTURE_BINDING: 0x4 } as const;
const STAGE_FRAGMENT_COMPUTE = 0x2 | 0x4;
/** Updates a replaced GPU object outlives (the core's retire rule: destroyed two updates later). */
const RETIRE_UPDATES = 2;

/** The LOD strategies (R1 §4.3; `three/sparse-source.ts`). */
export type VDBSparseLodGpu = "none" | "coarse-pyramid" | "node-average";

/** Inclusive voxel index box. */
export interface VDBSparseIndexBox {
  /** Smallest voxel index (inclusive). */
  min: Vec3;
  /** Largest voxel index (inclusive). */
  max: Vec3;
}

/** Options of {@link vdbSparseSourceGpu}. */
export interface VDBSparseSourceGpuOptions {
  /** LOD strategy (default `'coarse-pyramid'`, the K3.2 pick). */
  lod?: VDBSparseLodGpu;
  /** `'coarse-pyramid'`: base level texels on the longest side at most (default 128). */
  pyramidMax?: number;
  /** `'clamp01'` clamps each density sample to [0, 1] (his 8-bit range); default `'none'`. */
  normalize?: "none" | "clamp01";
  /** Grid types the source's volumes may hold (default `VDB_ALL_SCALAR_TYPES`); the VDB module must decode them. */
  types?: readonly VDBGridTypeName[];
  /**
   * G16a: follow the grids' images. When a channel grid's `imageRevision` moves and every channel grid of the volume
   * is resident, `update()` re-reads its boxes, world box and pyramid size from `pool.info` (values in the table, so
   * nothing recompiles) and bumps the volume's `version`. Not with per-volume `indexBox`, `auxIndexBoxes` or `bbox`.
   */
  followImage?: boolean;
  /** `march` profile: the HDDA empty-space query (`vdbSrcSkipSpan`); default true. False: it never skips (smaller module). */
  skipEmpty?: boolean;
  /** K3.2d: level 0 through `vdbSampleFast` (module helper `'fastTrilinear'`); default true. */
  fastTrilinear?: boolean;
  /** K3.2d: one accessor per invocation across samples (`var<private>`); default true. */
  cacheAccessor?: boolean;
  /** Channels the WGSL compiles (default: grows with the volumes' channels; fixed when given). */
  channels?: Partial<VDBVolumeChannels>;
  /** Volume slots (the WGSL table size, default 64; 224 bytes each, ≤ the device's uniform binding size). */
  maxVolumes?: number;
  /** Label prefix of the GPU objects. */
  label?: string;
}

/** The pool ids of one volume's channels. `albedo`: one grey grid or three (RGB). */
export interface VDBSparseVolumeGrids {
  /** Pool id of the density grid. */
  density: number;
  /** Pool id of an optional mask grid (multiplies the density). */
  mask?: number;
  /** Pool id of one grey albedo grid, or three ids for RGB. */
  albedo?: number | readonly [number, number, number];
  /** Pool id of an optional emission (temperature) grid. */
  emission?: number;
}

/** Per-volume options of {@link VDBSparseSourceGpu.volume}. */
export interface VDBSparseVolumeGpuOptions {
  /** Density index box the unit box covers (default: the grid's index bbox). Not with `followImage`. */
  indexBox?: VDBSparseIndexBox;
  /** Index boxes of the aux grids (default: each grid's own bbox). Not with `followImage`. */
  auxIndexBoxes?: Partial<Record<"mask" | "albedo" | "emission", VDBSparseIndexBox>>;
  /** World box (default: the density cell box through the grid's transform, as an AABB). Not with `followImage`. */
  bbox?: [Vec3, Vec3];
  /** Density stats; default from `image`, else read back from the pool by `update()` once the grid is resident. */
  stats?: { meanNonZero: number; max: number; activeVoxels: number };
  /** The density grid's CPU image (for the stats). */
  image?: Uint32Array;
}

/** One volume of a sparse source: its slot and its own layout (the {@link VDBVolumeSlot} fields describe it). */
export interface VDBSparseVolumeGpu extends VDBVolumeSlot {
  readonly kind: "sparse";
  readonly source: VDBSparseSourceGpu;
  /** The pool ids of the channels. */
  readonly ids: Readonly<VDBSparseVolumeGrids>;
  /** The density index box the unit box covers. */
  readonly indexBox: VDBSparseIndexBox;
  /** World box (AABB of the density cell box). */
  readonly bbox: [Vec3, Vec3];
  /** `'coarse-pyramid'`: the pyramid layout and where its blocks sit in the packed atlas (null before the first update). */
  readonly pyramid: { readonly layout: VDBBoxPyramidLayout; readonly base: Vec3 | null; readonly atlas: Vec3 | null } | null;
  /** Times the stats were recomputed from a read-back image. */
  readonly statsUpdates: number;
  /** Times the layout was re-read after an image change (`followImage`). */
  readonly layoutUpdates: number;
  /** True once `remove()` ran. */
  readonly removed: boolean;
  /** Frees the slot (its `v` may be reused by a later volume) and its pyramid. */
  remove(): void;
}

/** What `update()` did. */
export interface VDBSparseSourceUpdate {
  /** Pyramid builds recorded into the encoder. */
  readonly pyramidBuilds: number;
  /** Volumes whose layout was re-read (`followImage`). */
  readonly relayouts: number;
  /** True when `version` moved since the previous `update()` (a new atlas texture or pool buffer): rebuild bind group 2. */
  readonly rebind: boolean;
}

/** The sparse source of the core renderer. */
export interface VDBSparseSourceGpu extends VDBVolumeSource {
  readonly kind: "sparse";
  readonly pool: VDBGridPool;
  readonly lod: VDBSparseLodGpu;
  readonly followImage: boolean;
  readonly maxVolumes: number;
  /** What the shader's one VDB module must include (profile, helpers, types). */
  readonly module: { readonly profile: "sample" | "march"; readonly helpers: readonly VDBModuleHelper[]; readonly types: readonly VDBGridTypeName[] };
  /** Live volumes, by slot. */
  readonly volumes: readonly VDBSparseVolumeGpu[];
  /** GPU bytes the source allocated: the table, the packed atlas, the pyramids' scratch buffers. */
  readonly gpuBytes: { readonly table: number; readonly atlas: number; readonly pyramidScratch: number };
  /** Pyramid builds recorded so far. */
  readonly pyramidBuilds: number;
  /** Adds a volume; returns its slot (lowest free `v`). */
  volume(grids: VDBSparseVolumeGrids, options?: VDBSparseVolumeGpuOptions): VDBSparseVolumeGpu;
  /**
   * Records this frame's source work into the host's encoder: re-reads followed layouts, packs and (re)builds the
   * pyramids whose grid image changed or whose block moved, writes the volume table (`queue.writeBuffer`) and
   * starts stats readbacks. Call once per submitted encoder, after `pool.update(encoder)` in the same encoder, and
   * build bind group 2 after it (Traps).
   */
  update(encoder: GPUCommandEncoder): VDBSparseSourceUpdate;
  /** The VDB module for `slot` with this source's needs (plus `helpers`), for shaders that sample only this source. */
  moduleWgsl(slot: VDBSlot, helpers?: Iterable<VDBModuleHelper>): string;
  /** Resolves when the stats readbacks in flight have landed. */
  settled(): Promise<void>;
  /** The packed pyramid atlas (r16float; a 1³ placeholder before the first pyramid), for tests and readbacks. */
  readonly atlas: GPUTexture | null;
}

const bad = (message: string) => new VDBError("VDB_BAD_OPTION", `vdbSparseSourceGpu: ${message}`);

function checkSlot(slot: VDBSlot, where: string): VDBSlot {
  if (!slot || !Number.isInteger(slot.group) || slot.group < 0 || !Number.isInteger(slot.firstBinding) || slot.firstBinding < 0) {
    throw bad(`${where}: slot must be { group, firstBinding } with non-negative integers, got ${JSON.stringify(slot)}.`);
  }
  return slot;
}

const copyBox = (b: { min: readonly number[]; max: readonly number[] }): VDBSparseIndexBox => ({ min: [b.min[0]!, b.min[1]!, b.min[2]!], max: [b.max[0]!, b.max[1]!, b.max[2]!] });
const sizeOf = (b: VDBSparseIndexBox): Vec3 => [b.max[0] - b.min[0] + 1, b.max[1] - b.min[1] + 1, b.max[2] - b.min[2] + 1];
const sameBox = (a: VDBSparseIndexBox, b: VDBSparseIndexBox) => a.min.every((v, i) => v === b.min[i]) && a.max.every((v, i) => v === b.max[i]);

/** World AABB of an index cell box through a 3x4 row-major affine (the TSL source's `worldBox`). */
function worldBox(box: VDBSparseIndexBox, m: Float64Array): [Vec3, Vec3] {
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

/** Local unit box → sample index space: p = min − ½ + (local + ½)·size (column-major; the TSL source's). */
function boxToIndex(b: VDBSparseIndexBox): Float32Array {
  const [sx, sy, sz] = sizeOf(b);
  return new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, b.min[0] - 0.5 + 0.5 * sx, b.min[1] - 0.5 + 0.5 * sy, b.min[2] - 0.5 + 0.5 * sz, 1]);
}

/** A coarse pyramid shared by the volumes with the same density grid and box. */
interface PyramidEntry {
  readonly key: string;
  readonly gridId: number;
  readonly box: VDBSparseIndexBox;
  readonly layout: VDBBoxPyramidLayout;
  refs: number;
  gpu: VDBBoxPyramidGpu | null;
  base: Vec3 | null;
  atlas: Vec3 | null;
  /** A (re)build is needed whatever the grid's revision (new or moved block). */
  dirty: boolean;
}

interface Readback {
  readonly buffer: GPUBuffer;
  readonly revision: number;
  readonly recordedAt: number;
  mapping: Promise<void> | null;
}

class SparseVolume implements VDBSparseVolumeGpu {
  readonly kind = "sparse" as const;
  readonly source: SparseSource;
  readonly v: number;
  readonly ids: Readonly<VDBSparseVolumeGrids>;
  readonly channels: VDBVolumeChannels;
  indexBox: VDBSparseIndexBox;
  auxBoxes: Record<"mask" | "albedo" | "emission", VDBSparseIndexBox | null>;
  texels: [number, number, number];
  lodLevels = 0;
  stats: { meanNonZero: number; max: number; activeVoxels: number };
  boxToSource: Float32Array;
  bbox: [Vec3, Vec3];
  statsUpdates = 0;
  layoutUpdates = 0;
  removed = false;
  entry: PyramidEntry | null = null;
  /** The channel grids' newest imageRevision the layout describes (followImage) and the pool revision last checked. */
  layoutRevision = 0;
  layoutPoolRevision = -1;
  /** The density imageRevision the stats describe; -1 = none yet (read back once resident). */
  statsRevision: number;
  readback: Readback | null = null;
  readonly poolIds: number[];
  #versionSeen = { poolRevision: -1, value: 0 };

  constructor(source: SparseSource, v: number, ids: VDBSparseVolumeGrids, o: VDBSparseVolumeGpuOptions) {
    this.source = source;
    this.v = v;
    const albedo = Array.isArray(ids.albedo) ? ([...ids.albedo] as unknown as [number, number, number]) : ids.albedo;
    this.ids = Object.freeze({ ...ids, ...(albedo !== undefined ? { albedo } : {}) });
    this.channels = { mask: ids.mask !== undefined, albedo: ids.albedo !== undefined, emission: ids.emission !== undefined };
    this.poolIds = [...new Set([ids.density, ids.mask, ids.emission, ...(Array.isArray(albedo) ? albedo : [albedo])].filter((x): x is number => x !== undefined))];
    const pool = source.pool;
    const density = pool.info(ids.density);
    this.indexBox = copyBox(o.indexBox ?? density.indexBBox);
    this.texels = sizeOf(this.indexBox);
    if (this.texels.some((t) => !(t > 0))) throw bad("the density index box is empty.");
    const aux = o.auxIndexBoxes ?? {};
    const box = (id: number | undefined, given?: VDBSparseIndexBox) => (id === undefined ? null : copyBox(given ?? pool.info(id).indexBBox));
    this.auxBoxes = { mask: box(ids.mask, aux.mask), albedo: box(this.albedoId, aux.albedo), emission: box(ids.emission, aux.emission) };
    this.boxToSource = boxToIndex(this.indexBox);
    this.bbox = o.bbox ? [[...o.bbox[0]], [...o.bbox[1]]] : worldBox(this.indexBox, density.indexToWorld);
    const clamp01 = source.normalize === "clamp01";
    this.stats = o.stats ? { ...o.stats } : o.image ? vdbSparseDensityStats(o.image, clamp01) : { meanNonZero: 0, max: 0, activeVoxels: 0 };
    this.statsRevision = o.stats || o.image ? density.imageRevision : -1;
    this.layoutRevision = this.poolIds.reduce((r, id) => Math.max(r, pool.info(id).imageRevision), 0);
  }

  get albedoId(): number | undefined {
    const a = this.ids.albedo;
    return Array.isArray(a) ? a[0] : (a as number | undefined);
  }

  /** Content version: the image revision the layout describes (following), else the newest channel imageRevision. */
  get version(): number {
    if (this.source.followImage) return this.layoutRevision;
    const pool = this.source.pool;
    if (pool.revision !== this.#versionSeen.poolRevision) {
      let value = 0;
      for (const id of this.poolIds) {
        try {
          value = Math.max(value, pool.info(id).imageRevision);
        } catch {
          // A removed grid: keep the others' revision.
        }
      }
      this.#versionSeen = { poolRevision: pool.revision, value };
    }
    return this.#versionSeen.value;
  }

  get pyramid(): VDBSparseVolumeGpu["pyramid"] {
    const e = this.entry;
    return e ? { layout: e.layout, base: e.base, atlas: e.atlas } : null;
  }

  /** Re-reads every box from the pool (followImage). */
  relayout(): void {
    const pool = this.source.pool;
    const density = pool.info(this.ids.density);
    this.indexBox = copyBox(density.indexBBox);
    this.texels = sizeOf(this.indexBox);
    this.boxToSource = boxToIndex(this.indexBox);
    this.bbox = worldBox(this.indexBox, density.indexToWorld);
    const aux = (id: number | undefined) => (id === undefined ? null : copyBox(pool.info(id).indexBBox));
    this.auxBoxes = { mask: aux(this.ids.mask), albedo: aux(this.albedoId), emission: aux(this.ids.emission) };
    this.layoutUpdates++;
  }

  /** The table record. */
  record(): SparseRecord {
    const ids = this.ids;
    const albedo = ids.albedo === undefined ? [0, 0, 0] : Array.isArray(ids.albedo) ? ids.albedo : [ids.albedo as number, ids.albedo as number, ids.albedo as number];
    const flags =
      (this.channels.mask ? SPARSE_FLAG.mask : 0) |
      (this.channels.albedo ? SPARSE_FLAG.albedo : 0) |
      (Array.isArray(ids.albedo) ? SPARSE_FLAG.albedoRgb : 0) |
      (this.channels.emission ? SPARSE_FLAG.emission : 0);
    const minOf = (b: VDBSparseIndexBox | null) => (b ? b.min : [0, 0, 0]);
    const sizeOfBox = (b: VDBSparseIndexBox | null) => (b ? sizeOf(b) : [1, 1, 1]);
    const e = this.entry;
    const layout = e?.layout;
    return {
      ids: [ids.density, ids.mask ?? 0, ids.emission ?? 0, flags],
      albedo: [albedo[0]!, albedo[1]!, albedo[2]!, 0],
      lod: [this.lodLevels, layout?.level ?? 0, layout?.atlas?.levels.length ?? 0, this.stats.max],
      densityMin: this.indexBox.min,
      densitySize: this.texels,
      maskMin: minOf(this.auxBoxes.mask),
      maskSize: sizeOfBox(this.auxBoxes.mask),
      albedoMin: minOf(this.auxBoxes.albedo),
      albedoSize: sizeOfBox(this.auxBoxes.albedo),
      emissionMin: minOf(this.auxBoxes.emission),
      emissionSize: sizeOfBox(this.auxBoxes.emission),
      pyrBase: e?.base ?? [0, 0, 0],
      pyrSize: layout?.dimensions ?? [1, 1, 1],
      pyrMip: [...(e?.atlas ?? [0, 0, 0]), layout?.atlas?.dimensions[0] ?? 1],
    };
  }

  remove(): void {
    this.source.removeVolume(this);
  }
}

class SparseSource implements VDBSparseSourceGpu {
  readonly kind = "sparse" as const;
  readonly pool: VDBGridPool;
  readonly lod: VDBSparseLodGpu;
  readonly normalize: "none" | "clamp01";
  readonly followImage: boolean;
  readonly maxVolumes: number;
  readonly module: VDBSparseSourceGpu["module"];
  readonly boxToSource = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  pyramidBuilds = 0;
  readonly #device: GPUDevice;
  readonly #label: string;
  readonly #types: readonly VDBGridTypeName[];
  readonly #pyramidMax: number;
  readonly #fixedChannels: Partial<VDBVolumeChannels> | null;
  readonly #channels: { -readonly [K in keyof VDBVolumeChannels]: boolean } = { mask: false, albedo: false, emission: false };
  readonly #wgslBase: Omit<SparseWgslConfig, "slot" | "profile" | "channels">;
  readonly #slots: (SparseVolume | null)[] = [];
  readonly #pyramids = new Map<string, PyramidEntry>();
  readonly #table: GPUBuffer;
  readonly #sampler: GPUSampler | null;
  #atlas: GPUTexture | null = null;
  #ownVersion = 0;
  #lastVersion = -1;
  #packDirty = false;
  #tableDirty = true;
  #updates = 0;
  #retired: { at: number; destroy: () => void }[] = [];
  readonly #mappings = new Set<Promise<void>>();
  #disposed = false;

  constructor(pool: VDBGridPool, o: VDBSparseSourceGpuOptions) {
    if (!pool || typeof (pool as { info?: unknown }).info !== "function" || !(pool as { device?: unknown }).device) throw bad("pass a VDBGridPool.");
    this.pool = pool;
    this.#device = pool.device;
    this.#label = o.label ?? "vdb sparse source";
    const lod = o.lod ?? "coarse-pyramid";
    if (lod !== "none" && lod !== "coarse-pyramid" && lod !== "node-average") throw bad(`unknown lod ${JSON.stringify(lod)}; use 'none', 'coarse-pyramid' or 'node-average'.`);
    this.lod = lod;
    const normalize = o.normalize ?? "none";
    if (normalize !== "none" && normalize !== "clamp01") throw bad(`unknown normalize ${JSON.stringify(normalize)}; use 'none' or 'clamp01'.`);
    this.normalize = normalize;
    if (o.followImage !== undefined && typeof o.followImage !== "boolean") throw bad("followImage must be a boolean.");
    this.followImage = o.followImage ?? false;
    const types = o.types ?? VDB_ALL_SCALAR_TYPES;
    if (!Array.isArray(types) || types.length === 0 || types.some((t) => !VDB_ALL_SCALAR_TYPES.includes(t))) throw bad(`types must list grid types (${VDB_ALL_SCALAR_TYPES.join(", ")}), got ${JSON.stringify(types)}.`);
    this.#types = Object.freeze([...types]);
    this.#pyramidMax = o.pyramidMax ?? 128;
    if (!Number.isInteger(this.#pyramidMax) || this.#pyramidMax < 1) throw bad(`pyramidMax must be a positive integer, got ${o.pyramidMax}.`);
    const skipEmpty = o.skipEmpty ?? true;
    const fastTrilinear = o.fastTrilinear ?? true;
    const cacheAccessor = o.cacheAccessor ?? true;
    const uniformLimit = (this.#device.limits as { maxUniformBufferBindingSize?: number } | undefined)?.maxUniformBufferBindingSize ?? 65536;
    const maxVolumes = o.maxVolumes ?? 64;
    const cap = Math.floor(uniformLimit / VDB_SPARSE_RECORD_BYTES);
    if (!Number.isInteger(maxVolumes) || maxVolumes < 1 || maxVolumes > cap) throw bad(`maxVolumes must be an integer in [1, ${cap}] (${VDB_SPARSE_RECORD_BYTES} B each in one uniform binding), got ${o.maxVolumes}.`);
    this.maxVolumes = maxVolumes;
    this.#fixedChannels = o.channels ? { ...o.channels } : null;
    if (this.#fixedChannels) Object.assign(this.#channels, { mask: !!o.channels!.mask, albedo: !!o.channels!.albedo, emission: !!o.channels!.emission });
    this.#wgslBase = { lod, fastTrilinear, cacheAccessor, clamp01: normalize === "clamp01", skipEmpty, maxVolumes };
    const helpers: VDBModuleHelper[] = [];
    if (skipEmpty) helpers.push("cellBoundTrilinear", "hddaCellEnd");
    if (lod === "node-average") helpers.push("nodeAverage");
    if (fastTrilinear) helpers.push("fastTrilinear");
    this.module = Object.freeze({ profile: skipEmpty || lod === "node-average" ? "march" : "sample", helpers: Object.freeze(helpers), types: this.#types });

    this.#table = this.#device.createBuffer({ label: `${this.#label} volumes`, size: maxVolumes * VDB_SPARSE_RECORD_BYTES, usage: BUFFER.UNIFORM | BUFFER.COPY_DST });
    if (lod === "coarse-pyramid") {
      this.#sampler = this.#device.createSampler({
        label: `${this.#label} pyramid sampler`,
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "nearest",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
        addressModeW: "clamp-to-edge",
      });
      this.#atlas = this.#newAtlas([1, 1, 1]);
      // G16a: a following source compiles the pyramid kernels of every type now, so a swap creates no pipeline.
      if (this.followImage) warmBoxPyramidPipelines(this.#device, this.#types);
    } else {
      this.#sampler = null;
    }
  }

  // ---- VDBVolumeSourceInfo (aggregates over the live volumes; per volume, read the slot) ----

  get channels(): VDBVolumeChannels {
    return { ...this.#channels };
  }

  get texels(): [number, number, number] {
    const t: Vec3 = [1, 1, 1];
    for (const s of this.volumes) for (let a = 0; a < 3; a++) t[a] = Math.max(t[a]!, s.texels[a]!);
    return t;
  }

  get lodLevels(): number {
    return this.volumes.reduce((m, s) => Math.max(m, s.lodLevels), 0);
  }

  get stats(): { meanNonZero: number; max: number; activeVoxels: number } {
    let sum = 0;
    let count = 0;
    let max = 0;
    for (const s of this.volumes) {
      sum += s.stats.meanNonZero * s.stats.activeVoxels;
      count += s.stats.activeVoxels;
      max = Math.max(max, s.stats.max);
    }
    return { meanNonZero: count > 0 ? sum / count : 0, max, activeVoxels: count };
  }

  /** Bumps when bind group 2 must be rebuilt: a new atlas texture, or the pool's buffer replaced (`pool.version`). */
  get version(): number {
    return this.#ownVersion + this.pool.version;
  }

  get volumes(): readonly SparseVolume[] {
    return this.#slots.filter((s): s is SparseVolume => s !== null);
  }

  get atlas(): GPUTexture | null {
    return this.#atlas;
  }

  get gpuBytes(): VDBSparseSourceGpu["gpuBytes"] {
    const a = this.#atlas;
    let scratch = 0;
    for (const e of this.#pyramids.values()) scratch += e.gpu?.gpuBytes.scratch ?? 0;
    return { table: this.#table.size, atlas: a ? 2 * a.width * a.height * a.depthOrArrayLayers : 0, pyramidScratch: scratch };
  }

  // ---- volumes ----

  #check(): void {
    if (this.#disposed) throw new VDBError("VDB_DISPOSED", "vdbSparseSourceGpu: the source was disposed; create a new one.");
  }

  #checkTypes(infos: readonly { id: number; type: VDBGridTypeName }[], what: string): void {
    const missing = infos.filter((i) => !this.#types.includes(i.type));
    if (missing.length === 0) return;
    throw new VDBError(
      "VDB_TYPE_NOT_COMPILED",
      `vdbSparseSourceGpu: ${what} is ${missing.map((m) => `grid ${m.id}: ${m.type}`).join(", ")}, but the source compiles only ${this.#types.join(", ")}; ` +
        `it would read 0. Make the source with types that include every type the images will have (a coarse-first stream lists them in VDBCoarseFirstEvent.types).`,
    );
  }

  volume(grids: VDBSparseVolumeGrids, o: VDBSparseVolumeGpuOptions = {}): VDBSparseVolumeGpu {
    this.#check();
    if (!grids || !Number.isInteger(grids.density)) throw bad("volume(): grids.density must be a pool grid id.");
    if (Array.isArray(grids.albedo) && grids.albedo.length !== 3) throw bad("volume(): albedo must be one id or three (RGB).");
    if (this.followImage) {
      const fixed = (["indexBox", "auxIndexBoxes", "bbox"] as const).filter((k) => o[k] !== undefined);
      if (fixed.length) throw bad(`followImage re-reads every box from the pool; drop ${fixed.join(", ")} (fixed boxes).`);
    }
    const ids = [grids.density, grids.mask, grids.emission, ...(Array.isArray(grids.albedo) ? grids.albedo : [grids.albedo])].filter((x): x is number => x !== undefined);
    const infos: VDBGridInfo[] = ids.map((id) => this.pool.info(id));
    this.#checkTypes(infos, "a channel grid");
    if (this.lod === "node-average" && !infos[0]!.hasStats) throw bad("lod 'node-average' reads node statistics; this grid has none (rebuild it with stats).");
    const channels = { mask: grids.mask !== undefined, albedo: grids.albedo !== undefined, emission: grids.emission !== undefined };
    if (this.#fixedChannels) {
      const off = (Object.keys(channels) as (keyof VDBVolumeChannels)[]).filter((k) => channels[k] && !this.#channels[k]);
      if (off.length) throw bad(`volume(): the source compiles channels ${JSON.stringify(this.#channels)}; this volume has ${off.join(", ")} (make the source with channels that include it).`);
    }
    let v = this.#slots.indexOf(null);
    if (v < 0) v = this.#slots.length;
    if (v >= this.maxVolumes) throw new VDBError("VDB_POOL_FULL", `vdbSparseSourceGpu: all ${this.maxVolumes} volume slots are used (maxVolumes).`);
    const volume = new SparseVolume(this, v, grids, o);
    this.#slots[v] = volume;
    for (const k of Object.keys(channels) as (keyof VDBVolumeChannels)[]) if (channels[k]) this.#channels[k] = true;
    this.#attachPyramid(volume);
    this.#tableDirty = true;
    return volume;
  }

  removeVolume(volume: SparseVolume): void {
    if (volume.removed || this.#slots[volume.v] !== volume) return;
    volume.removed = true;
    this.#slots[volume.v] = null;
    while (this.#slots.length && this.#slots[this.#slots.length - 1] === null) this.#slots.pop();
    this.#detachPyramid(volume);
    const r = volume.readback;
    // Not mapping yet: its copy may sit in an encoder that is not submitted (mapping ones free themselves).
    if (r && !r.mapping) this.#retire(() => r.buffer.destroy());
    this.#tableDirty = true;
  }

  #attachPyramid(volume: SparseVolume): void {
    if (this.lod !== "coarse-pyramid") {
      volume.lodLevels = this.lod === "node-average" ? 7 : 0;
      return;
    }
    const box = volume.indexBox;
    const key = `${volume.ids.density}:${box.min.join(",")}:${box.max.join(",")}`;
    let entry = this.#pyramids.get(key);
    if (!entry) {
      entry = { key, gridId: volume.ids.density, box: copyBox(box), layout: vdbBoxPyramidLayout(sizeOf(box), { pyramidMax: this.#pyramidMax }), refs: 0, gpu: null, base: null, atlas: null, dirty: true };
      this.#pyramids.set(key, entry);
      this.#packDirty = true;
    }
    entry.refs++;
    volume.entry = entry;
    volume.lodLevels = entry.layout.lodLevels;
  }

  #detachPyramid(volume: SparseVolume): void {
    const entry = volume.entry;
    volume.entry = null;
    if (!entry || --entry.refs > 0) return;
    this.#pyramids.delete(entry.key);
    const gpu = entry.gpu;
    if (gpu) this.#retire(() => gpu.destroy());
    this.#packDirty = true;
  }

  #retire(destroy: () => void): void {
    this.#retired.push({ at: this.#updates, destroy });
  }

  #newAtlas(size: Vec3): GPUTexture {
    return this.#device.createTexture({
      label: `${this.#label} pyramid atlas`,
      size,
      dimension: "3d",
      format: "r16float",
      usage: TEXTURE.TEXTURE_BINDING | TEXTURE.COPY_DST | TEXTURE.COPY_SRC,
    });
  }

  /** Packs every pyramid's blocks; grows the atlas when they no longer fit; moved or new blocks rebuild. */
  #pack(): void {
    this.#packDirty = false;
    const entries = [...this.#pyramids.values()];
    const blocks: (readonly number[])[] = [];
    for (const e of entries) {
      blocks.push(e.layout.dimensions);
      if (e.layout.atlas) blocks.push(e.layout.atlas.dimensions);
    }
    const max = (this.#device.limits as { maxTextureDimension3D?: number } | undefined)?.maxTextureDimension3D ?? 2048;
    const packing = vdbPackPyramidBlocks(blocks, [max, max, max]);
    const old = this.#atlas!;
    const grow = packing.size.some((s, a) => s > [old.width, old.height, old.depthOrArrayLayers][a]!);
    if (grow) {
      const size: Vec3 = [Math.max(old.width, packing.size[0]), Math.max(old.height, packing.size[1]), Math.max(old.depthOrArrayLayers, packing.size[2])];
      this.#atlas = this.#newAtlas(size);
      this.#retire(() => old.destroy());
      this.#ownVersion++;
    }
    let i = 0;
    for (const e of entries) {
      const base = packing.origins[i++]!;
      const atlas = e.layout.atlas ? packing.origins[i++]! : null;
      const moved = !e.base || e.base.some((v, a) => v !== base[a]) || (atlas !== null && (!e.atlas || e.atlas.some((v, a) => v !== atlas[a])));
      if (!moved && !grow && e.gpu) continue;
      e.base = [...base];
      e.atlas = atlas ? [...atlas] : null;
      const previous = e.gpu;
      if (previous) this.#retire(() => previous.destroy());
      const target = this.#atlas!;
      e.gpu = vdbCreateBoxPyramid(this.#device, this.pool, e.gridId, {
        indexBox: e.box,
        level: e.layout.level,
        targets: { base: target, atlas: e.layout.atlas ? target : null, baseOrigin: e.base, ...(e.atlas ? { atlasOrigin: e.atlas } : {}) },
        label: `${this.#label} pyramid grid ${e.gridId}`,
      });
      e.dirty = true;
    }
    this.#tableDirty = true;
  }

  /** G16a: re-reads a following volume's layout once every channel grid's new image is resident. */
  #syncLayout(volume: SparseVolume): boolean {
    const pool = this.pool;
    if (pool.revision === volume.layoutPoolRevision) return false;
    let revision = 0;
    let resident = true;
    const infos: { id: number; type: VDBGridTypeName }[] = [];
    for (const id of volume.poolIds) {
      let i: VDBGridInfo;
      try {
        i = pool.info(id);
      } catch {
        return false; // a removed grid: keep the last layout
      }
      revision = Math.max(revision, i.imageRevision);
      if (!i.resident) resident = false;
      infos.push({ id, type: i.type });
    }
    if (revision === volume.layoutRevision) {
      volume.layoutPoolRevision = pool.revision;
      return false;
    }
    if (!resident) return false; // look again once the update has written it
    this.#checkTypes(infos, "a channel grid's new image");
    const before = copyBox(volume.indexBox);
    volume.relayout();
    if (this.lod === "coarse-pyramid" && !sameBox(before, volume.indexBox)) {
      this.#detachPyramid(volume);
      this.#attachPyramid(volume);
    }
    volume.layoutRevision = revision;
    volume.layoutPoolRevision = pool.revision;
    this.#tableDirty = true;
    return true;
  }

  update(encoder: GPUCommandEncoder): VDBSparseSourceUpdate {
    this.#check();
    this.#updates++;
    // Retire: destroyed two updates after they were replaced (an encoder recorded before may still name them).
    const keep: { at: number; destroy: () => void }[] = [];
    for (const r of this.#retired) {
      if (this.#updates - r.at > RETIRE_UPDATES) r.destroy();
      else keep.push(r);
    }
    this.#retired = keep;

    let relayouts = 0;
    if (this.followImage) for (const s of this.volumes) if (this.#syncLayout(s)) relayouts++;
    if (this.#packDirty) this.#pack();

    let builds = 0;
    for (const e of this.#pyramids.values()) {
      const gpu = e.gpu;
      if (!gpu) continue;
      let info: VDBGridInfo;
      try {
        info = this.pool.info(e.gridId);
      } catch {
        continue; // removed from the pool: keep the last pyramid
      }
      if (!info.resident) continue; // the update that writes it is still to come
      if (e.dirty) {
        gpu.record(encoder);
        e.dirty = false;
        builds++;
      } else if (gpu.update(encoder)) {
        builds++;
      }
    }
    this.pyramidBuilds += builds;

    this.#stats(encoder);

    if (this.#tableDirty) {
      this.#tableDirty = false;
      const bytes = new ArrayBuffer(Math.max(1, this.#slots.length) * VDB_SPARSE_RECORD_BYTES);
      for (const s of this.volumes) packSparseRecord(bytes, s.v * VDB_SPARSE_RECORD_BYTES, s.record());
      this.#device.queue.writeBuffer(this.#table, 0, new Uint8Array(bytes));
    }
    const version = this.version;
    const rebind = version !== this.#lastVersion;
    this.#lastVersion = version;
    return { pyramidBuilds: builds, relayouts, rebind };
  }

  /** K3.2e: a changed density image's stats, read back from the pool: copied in this encoder, mapped next update. */
  #stats(encoder: GPUCommandEncoder): void {
    const clamp01 = this.normalize === "clamp01";
    for (const s of this.volumes) {
      const r = s.readback;
      if (r && !r.mapping && r.recordedAt < this.#updates) {
        // The encoder that holds the copy was submitted (one update per submitted encoder).
        const mapping = r.buffer
          .mapAsync(0x1)
          .then(() => {
            if (s.removed) return;
            s.stats = vdbSparseDensityStats(new Uint32Array(r.buffer.getMappedRange().slice(0)), clamp01);
            s.statsRevision = r.revision;
            s.statsUpdates++;
            this.#tableDirty = true;
          })
          .catch(() => {})
          .finally(() => {
            r.buffer.destroy();
            if (s.readback === r) s.readback = null;
            this.#mappings.delete(mapping);
          });
        r.mapping = mapping;
        this.#mappings.add(mapping);
        continue;
      }
      if (r) continue;
      let info: VDBGridInfo;
      try {
        info = this.pool.info(s.ids.density);
      } catch {
        continue;
      }
      if (info.imageRevision === s.statsRevision || !info.resident || info.imageOffset === null) continue;
      const bytes = Math.ceil(info.imageBytes / 4) * 4;
      const buffer = this.#device.createBuffer({ label: `${this.#label} stats readback`, size: bytes, usage: BUFFER.MAP_READ | BUFFER.COPY_DST });
      encoder.copyBufferToBuffer(this.pool.buffer, info.imageOffset, buffer, 0, bytes);
      s.readback = { buffer, revision: info.imageRevision, recordedAt: this.#updates, mapping: null };
    }
  }

  async settled(): Promise<void> {
    while (this.#mappings.size) await Promise.all([...this.#mappings]);
  }

  // ---- group-2 contract ----

  #config(o: VDBVolumeSourceWgslOptions): SparseWgslConfig {
    const slot = checkSlot(o.slot, "wgsl");
    if (o.profile !== "sample" && o.profile !== "march") throw bad(`wgsl: profile must be 'sample' or 'march', got ${JSON.stringify(o.profile)}.`);
    const on = (k: keyof VDBVolumeChannels) => this.#channels[k] && o.channels?.[k] !== false;
    return { ...this.#wgslBase, slot, profile: o.profile, channels: { mask: on("mask"), albedo: on("albedo"), emission: on("emission") } };
  }

  wgsl(o: VDBVolumeSourceWgslOptions): string {
    this.#check();
    return sparseSourceWgsl(this.#config(o));
  }

  moduleWgsl(slot: VDBSlot, helpers: Iterable<VDBModuleHelper> = []): string {
    this.#check();
    return vdbModuleWgsl({ slot: checkSlot(slot, "moduleWgsl"), types: this.#types, profile: this.module.profile, helpers: new Set([...this.module.helpers, ...helpers]) });
  }

  layoutEntries(slot: VDBSlot, visibility: GPUShaderStageFlags = STAGE_FRAGMENT_COMPUTE): GPUBindGroupLayoutEntry[] {
    this.#check();
    const b = sparseBindings(checkSlot(slot, "layoutEntries"), this.lod);
    const entries: GPUBindGroupLayoutEntry[] = [
      ...vdbModuleEntries({ slot, visibility }),
      { binding: b.table, visibility, buffer: { type: "uniform", minBindingSize: this.maxVolumes * VDB_SPARSE_RECORD_BYTES } },
    ];
    if (b.pyramid !== null) {
      entries.push({ binding: b.pyramid, visibility, texture: { sampleType: "float", viewDimension: "3d" } });
      entries.push({ binding: b.sampler!, visibility, sampler: { type: "filtering" } });
    }
    return entries;
  }

  bindEntries(slot: VDBSlot): GPUBindGroupEntry[] {
    this.#check();
    const b = sparseBindings(checkSlot(slot, "bindEntries"), this.lod);
    const entries: GPUBindGroupEntry[] = [this.pool.bindGroupEntry(slot), { binding: b.table, resource: { buffer: this.#table } }];
    if (b.pyramid !== null) {
      entries.push({ binding: b.pyramid, resource: this.#atlas!.createView({ dimension: "3d" }) });
      entries.push({ binding: b.sampler!, resource: this.#sampler! });
    }
    return entries;
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const r of this.#retired) r.destroy();
    this.#retired = [];
    for (const e of this.#pyramids.values()) e.gpu?.destroy();
    this.#pyramids.clear();
    for (const s of this.volumes) {
      s.removed = true;
      if (s.readback && !s.readback.mapping) s.readback.buffer.destroy();
    }
    this.#slots.length = 0;
    this.#atlas?.destroy();
    this.#atlas = null;
    this.#table.destroy();
  }
}

/**
 * A sparse volume source over the volumes of one `VDBGridPool` (see the file comment). The source owns its volume
 * table, the packed pyramid atlas and the pyramids' scratch buffers; it never touches the pool's buffer or the
 * device. `dispose()` frees what it allocated.
 *
 * @param pool The pool whose grids the volumes read (the source creates its objects on `pool.device`).
 * @param options LOD, normalization, types, `followImage`, `skipEmpty`, sampling and size options.
 */
export function vdbSparseSourceGpu(pool: VDBGridPool, options: VDBSparseSourceGpuOptions = {}): VDBSparseSourceGpu {
  return new SparseSource(pool, options);
}
