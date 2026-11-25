// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameMipmaps.js L14–106 (2x2x2 box downsample with clamped edge
//   taps, the mip atlas layout), as an f32 compute pass.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: GPU compute over a pool grid; the base level is the area-weighted box filter of
// `vdbResampleGrid`; f16 packing is three's `DataUtils.toHalfFloat`. Divergences: DIVERGES.md (K3.2a row).

/**
 * `vdbCreateBoxPyramid` (K3.2a): the sparse source's coarse pyramid (K3.2 `'coarse-pyramid'`) built on
 * the GPU from a pool grid, recorded into the host's encoder. CPU twin: `@pmndrs/vdb/tools`
 * `vdbBoxPyramid` + three's `DataUtils.toHalfFloat` (the K3.2 CPU path); parity ≤ 1 f16 ulp
 * (e2e/k3-pyramid.spec.ts).
 *
 * Passes, one compute pass (dispatches see each other's writes):
 * 1. **base** (one thread per base texel): the area-weighted box mean of the grid's voxels over the
 *    texel's cell, `vdbResampleGrid`'s `'box'` filter on the index cell box `[min − ½, max + ½]` with
 *    `⌈texels / 2^L₀⌉` texels. Overlaps are exact integers in units of 1/dims voxel (no f32 rounding
 *    of the footprint), summed separably x → y → z in the CPU's order. A texel whose taps touch an
 *    active voxel flags its 8³ texel block (atomicOr). A texel whose footprint lies in inactive
 *    zero tiles only (no leaf: `vdbGetDim` > 1) is 0 without the voxel loop, as on the CPU.
 * 2. **mask**: texels of an unflagged block become 0 (the CPU path drops leaves with no active voxel,
 *    and its dense readout fills them with 0).
 * 3. **levels**: his 2x2x2 box mips with clamped edge taps, level by level, into his atlas layout
 *    (`vdbBoxPyramidLayout`), from the f32 base (not the f16 one, as the CPU).
 * 4. **pack**: f32 → f16 (three's `toHalfFloat`: truncation, its denormal and overflow tables) into
 *    256-byte rows, then `copyBufferToTexture` into the r16float base and atlas textures.
 *
 * The textures are the pyramid's own (`TEXTURE_BINDING | COPY_DST | COPY_SRC`) or, with `targets`,
 * the host's (a three texture's `GPUTexture`): r16float, `COPY_DST`, the layout's sizes; then the
 * pyramid never destroys them. Scratch buffers (f32 levels, block flags, the packed rows, parameters)
 * stay alive for rebuilds (a sequence replaces the grid every frame) until `destroy()`.
 */
import type { VDBGridPool } from "../pool/pool.js";
import type { VDBVec3 } from "../pool/layout.js";
import type { VDBGridTypeName } from "../types.js";
import { VDBError } from "../errors.js";
import { vdbModuleEntries, vdbModuleWgsl } from "../wgsl/generate.js";
import { checkTarget, COMPUTE_SLOT, USAGE, WORKGROUP_AXIS } from "./shared.js";
import { vdbBoxPyramidLayout, type VDBBoxPyramidLayout } from "../../tools/frame/pyramid-layout.js";

const W = WORKGROUP_AXIS;
const STAGE_COMPUTE = 0x4;
const TEXTURE_USAGE = { COPY_SRC: 0x1, COPY_DST: 0x2, TEXTURE_BINDING: 0x4 } as const;
/** Uniform slot stride (minUniformBufferOffsetAlignment's default). */
const SLOT = 256;
const PARAMS_BYTES = 96;

const PARAMS_WGSL = /* wgsl */ `
struct PyrParams {
    grid: u32,
    srcFromBase: u32,
    srcOffset: u32,
    dstOffset: u32,
    boxMin: vec3i,
    rowWords: u32,
    texels: vec3u,
    blocksX: u32,
    dims: vec3u,
    blocksY: u32,
    srcDims: vec3u,
    pad1: u32,
    atlasDims: vec3u,
    pad2: u32,
};
@group(0) @binding(1) var<uniform> pyr: PyrParams;
@group(1) @binding(0) var<storage, read_write> pyrBase: array<f32>;
@group(1) @binding(1) var<storage, read_write> pyrAtlas: array<f32>;
@group(1) @binding(2) var<storage, read_write> pyrFlags: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> pyrPacked: array<u32>;
`;

const KERNELS_WGSL = /* wgsl */ `
fn pyrBlock(j: vec3u) -> u32 {
    let b = j >> vec3u(3u);
    return b.x + (b.y + b.z * pyr.blocksY) * pyr.blocksX;
}

// Box filter of the voxels under texel j (vdbResampleGrid 'box'): texel j covers [j·T, (j+1)·T] and voxel
// r covers [r·D, (r+1)·D] in units of 1/D voxel (T texels of the box, D base texels), so overlaps are
// exact integers and the weights ov / T. Summed x, then y, then z, in the CPU's order.
@compute @workgroup_size(${W}, ${W}, ${W})
fn csPyrBase(@builtin(global_invocation_id) j: vec3u) {
    let D = pyr.dims;
    if any(j >= D) {
        return;
    }
    let T = pyr.texels;
    let b0 = j * T;
    let b1 = b0 + T;
    let r0 = b0 / D;
    let r1 = (b1 + D - vec3u(1u)) / D - vec3u(1u);
    let tf = vec3f(T);
    var acc = vdbAccessorInit(pyr.grid);
    // Empty space: when every 8³ cell under the footprint is an inactive tile of value 0 (no leaf), the CPU
    // sum is exactly 0 with no active tap. Skips the voxel loop for most texels of a sparse cloud.
    let c0 = (pyr.boxMin + vec3i(r0)) >> vec3u(3u);
    let c1 = (pyr.boxMin + vec3i(r1)) >> vec3u(3u);
    var empty = true;
    for (var cz = c0.z; cz <= c1.z && empty; cz++) {
        for (var cy = c0.y; cy <= c1.y && empty; cy++) {
            for (var cx = c0.x; cx <= c1.x && empty; cx++) {
                let c = vec3i(cx, cy, cz) * 8;
                empty = vdbGetDim(&acc, c) > 1u && vdbReadValue(&acc, c) == 0.0 && !vdbIsActive(&acc, c);
            }
        }
    }
    if empty {
        pyrBase[j.x + (j.y + j.z * D.y) * D.x] = 0.0;
        return;
    }
    var touched = false;
    var sz = 0.0;
    for (var z = r0.z; z <= r1.z; z++) {
        let wz = f32(min(b1.z, (z + 1u) * D.z) - max(b0.z, z * D.z)) / tf.z;
        var sy = 0.0;
        for (var y = r0.y; y <= r1.y; y++) {
            let wy = f32(min(b1.y, (y + 1u) * D.y) - max(b0.y, y * D.y)) / tf.y;
            var sx = 0.0;
            for (var x = r0.x; x <= r1.x; x++) {
                let wx = f32(min(b1.x, (x + 1u) * D.x) - max(b0.x, x * D.x)) / tf.x;
                let ijk = pyr.boxMin + vec3i(vec3u(x, y, z));
                sx += wx * vdbReadValue(&acc, ijk);
                touched = touched || vdbIsActive(&acc, ijk);
            }
            sy += wy * sx;
        }
        sz += wz * sy;
    }
    pyrBase[j.x + (j.y + j.z * D.y) * D.x] = sz;
    if touched {
        atomicOr(&pyrFlags[pyrBlock(j)], 1u);
    }
}

// Texels of an 8³ block with no active tap read 0 (the CPU drops that leaf; its readout fills 0).
@compute @workgroup_size(${W}, ${W}, ${W})
fn csPyrMask(@builtin(global_invocation_id) j: vec3u) {
    let D = pyr.dims;
    if any(j >= D) {
        return;
    }
    if atomicLoad(&pyrFlags[pyrBlock(j)]) == 0u {
        pyrBase[j.x + (j.y + j.z * D.y) * D.x] = 0.0;
    }
}

fn pyrSrc(x: u32, y: u32, z: u32) -> f32 {
    if pyr.srcFromBase != 0u {
        return pyrBase[x + (y + z * pyr.srcDims.y) * pyr.srcDims.x];
    }
    return pyrAtlas[pyr.srcOffset + x + (y + z * pyr.atlasDims.y) * pyr.atlasDims.x];
}

// His downsample (frameMipmaps.js): the 2x2x2 mean with edge taps clamped, into the atlas at dstOffset.
@compute @workgroup_size(${W}, ${W}, ${W})
fn csPyrLevel(@builtin(global_invocation_id) p: vec3u) {
    if any(p >= pyr.dims) {
        return;
    }
    let s = pyr.srcDims - vec3u(1u);
    let x0 = min(p.x * 2u, s.x);
    let x1 = min(p.x * 2u + 1u, s.x);
    let y0 = min(p.y * 2u, s.y);
    let y1 = min(p.y * 2u + 1u, s.y);
    let z0 = min(p.z * 2u, s.z);
    let z1 = min(p.z * 2u + 1u, s.z);
    let sum = pyrSrc(x0, y0, z0) + pyrSrc(x1, y0, z0) + pyrSrc(x0, y1, z0) + pyrSrc(x1, y1, z0)
        + pyrSrc(x0, y0, z1) + pyrSrc(x1, y0, z1) + pyrSrc(x0, y1, z1) + pyrSrc(x1, y1, z1);
    pyrAtlas[pyr.dstOffset + p.x + (p.y + p.z * pyr.atlasDims.y) * pyr.atlasDims.x] = sum / 8.0;
}

// three's DataUtils.toHalfFloat (fox-toolkit tables: truncation, denormals, overflow to infinity).
fn pyrHalf(v: f32) -> u32 {
    let f = bitcast<u32>(clamp(v, -65504.0, 65504.0));
    let e = i32((f >> 23u) & 0xffu) - 127;
    var base = 0u;
    var shift = 24u;
    if e < -27 {
        base = 0u;
        shift = 24u;
    } else if e < -14 {
        base = 0x400u >> u32(-e - 14);
        shift = u32(-e - 1);
    } else if e <= 15 {
        base = u32(e + 15) << 10u;
        shift = 13u;
    } else if e < 128 {
        base = 0x7c00u;
        shift = 24u;
    } else {
        base = 0x7c00u;
        shift = 13u;
    }
    if (f & 0x80000000u) != 0u {
        base = base | 0x8000u;
    }
    return base + ((f & 0x7fffffu) >> shift);
}

// Two texels per word, rows of rowWords words (bytesPerRow multiple of 256), at dstOffset words.
@compute @workgroup_size(${W}, ${W}, ${W})
fn csPyrPack(@builtin(global_invocation_id) p: vec3u) {
    let size = pyr.srcDims;
    let words = (size.x + 1u) / 2u;
    if p.x >= words || p.y >= size.y || p.z >= size.z {
        return;
    }
    let x = p.x * 2u;
    var lo = 0u;
    var hi = 0u;
    if pyr.srcFromBase != 0u {
        let row = (p.y + p.z * size.y) * size.x;
        lo = pyrHalf(pyrBase[row + x]);
        if x + 1u < size.x {
            hi = pyrHalf(pyrBase[row + x + 1u]);
        }
    } else {
        let row = (p.y + p.z * size.y) * size.x;
        lo = pyrHalf(pyrAtlas[row + x]);
        if x + 1u < size.x {
            hi = pyrHalf(pyrAtlas[row + x + 1u]);
        }
    }
    pyrPacked[pyr.dstOffset + (p.y + p.z * size.y) * pyr.rowWords + p.x] = lo | (hi << 16u);
}
`;

/** The WGSL of the pyramid kernels for grids of `type` (exported for tests and the size of the shader). */
export function boxPyramidWgsl(type: VDBGridTypeName): string {
  return `${vdbModuleWgsl({ slot: COMPUTE_SLOT, types: [type], profile: "march" })}\n${PARAMS_WGSL}\n${KERNELS_WGSL}`;
}

interface PyramidPipelines {
  base: GPUComputePipeline;
  mask: GPUComputePipeline;
  level: GPUComputePipeline;
  pack: GPUComputePipeline;
  group0: GPUBindGroupLayout;
  group1: GPUBindGroupLayout;
  error: Promise<GPUError | null>;
}

const pipelines = new WeakMap<GPUDevice, Map<VDBGridTypeName, PyramidPipelines>>();

function pyramidPipelines(device: GPUDevice, type: VDBGridTypeName): PyramidPipelines {
  let byType = pipelines.get(device);
  if (!byType) pipelines.set(device, (byType = new Map()));
  const cached = byType.get(type);
  if (cached) return cached;
  device.pushErrorScope("validation");
  const group0 = device.createBindGroupLayout({
    label: `vdb boxPyramid ${type} group 0`,
    entries: [
      ...vdbModuleEntries({ slot: COMPUTE_SLOT, visibility: STAGE_COMPUTE }),
      { binding: 1, visibility: STAGE_COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAMS_BYTES } },
    ],
  });
  const group1 = device.createBindGroupLayout({
    label: `vdb boxPyramid ${type} group 1`,
    entries: [0, 1, 2, 3].map((binding) => ({ binding, visibility: STAGE_COMPUTE, buffer: { type: "storage" as const } })),
  });
  const module = device.createShaderModule({ label: `vdb boxPyramid ${type}`, code: boxPyramidWgsl(type) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [group0, group1] });
  const make = (entryPoint: string) => device.createComputePipeline({ label: `vdb boxPyramid ${entryPoint} ${type}`, layout, compute: { module, entryPoint } });
  const p: PyramidPipelines = {
    base: make("csPyrBase"),
    mask: make("csPyrMask"),
    level: make("csPyrLevel"),
    pack: make("csPyrPack"),
    group0,
    group1,
    error: device.popErrorScope(),
  };
  byType.set(type, p);
  return p;
}

/**
 * @internal Compiles (once per device and type, cached) the pyramid pipelines for grids of `types`, so a later
 * pyramid over a grid of one of them (G16a: a sparse source following a coarse-first swap to another grid type)
 * creates no pipeline. Returns the pipelines' validation results.
 */
export function warmBoxPyramidPipelines(device: GPUDevice, types: Iterable<VDBGridTypeName>): Promise<GPUError | null>[] {
  return [...types].map((type) => pyramidPipelines(device, type).error);
}

/** Options of {@link vdbCreateBoxPyramid}. */
export interface VDBBoxPyramidGpuOptions {
  /** Inclusive voxel index box the pyramid covers. Default: the grid's index bbox now (fixed afterwards). */
  indexBox?: { min: VDBVec3; max: VDBVec3 };
  /** Texels on the longest side of the base level at most (default 128). */
  pyramidMax?: number;
  /** Base level override (default `vdbPyramidBaseLevel`). */
  level?: number;
  /**
   * Write into these r16float textures (`COPY_DST`; `atlas` null exactly when the layout has no atlas)
   * instead of allocating. The pyramid never destroys them (a three texture's `GPUTexture`, for example).
   * Without origins they must have the layout's sizes; with `baseOrigin` / `atlasOrigin` (K6.7: the core
   * sparse source packs every volume's pyramid into one texture) the block at the origin must fit, and
   * `base` and `atlas` may be the same texture.
   */
  targets?: { base: GPUTexture; atlas: GPUTexture | null; baseOrigin?: VDBVec3; atlasOrigin?: VDBVec3 };
  /** Debug label of the GPU objects. */
  label?: string;
}

/** Options of {@link VDBBoxPyramidGpu.record}. */
export interface VDBBoxPyramidRecordOptions {
  /** Timestamps around the build's one compute pass (the `timestamp-query` feature). */
  timestampWrites?: GPUComputePassTimestampWrites;
}

/** A coarse pyramid on the GPU (see the file comment). */
export interface VDBBoxPyramidGpu {
  /** The pool id of the grid the pyramid is built from. */
  readonly gridId: number;
  /** The sizes and level layout of the textures. */
  readonly layout: VDBBoxPyramidLayout;
  /** r16float, the base level (`layout.dimensions`). */
  readonly base: GPUTexture;
  /** r16float, his atlas of the levels above (`layout.atlas.dimensions`), or null. */
  readonly atlas: GPUTexture | null;
  /** Bytes: `textures` (r16f; counted whoever owns them) and `scratch` (this pyramid's buffers). */
  readonly gpuBytes: { readonly textures: number; readonly scratch: number };
  /** The grid's `imageRevision` the last recorded build read, or null before the first. */
  readonly builtRevision: number | null;
  /** True when the grid's image changed since the last recorded build (or none was recorded). */
  readonly stale: boolean;
  /** Pipeline validation (null = valid); the first build's errors surface here. */
  readonly error: Promise<GPUError | null>;
  /**
   * Records a (re)build into `encoder`. The grid must be resident (`pool.update(encoder)` recorded
   * earlier in this encoder or submitted before); call after the pool's update, as for bind groups.
   *
   * @param encoder The encoder the build is recorded into; the host submits it.
   * @param options Timestamp writes for the build pass.
   */
  record(encoder: GPUCommandEncoder, options?: VDBBoxPyramidRecordOptions): void;
  /** Records a rebuild when {@link stale} and the grid is resident; returns whether it did. */
  update(encoder: GPUCommandEncoder): boolean;
  /** Frees the scratch buffers and the textures this pyramid allocated (not `targets`). Idempotent. */
  destroy(): void;
}

const align = (n: number, a: number) => Math.ceil(n / a) * a;
const bad = (message: string): never => {
  throw new VDBError("VDB_BAD_OPTION", `vdbCreateBoxPyramid: ${message}`);
};

class BoxPyramidGpu implements VDBBoxPyramidGpu {
  readonly gridId: number;
  readonly layout: VDBBoxPyramidLayout;
  readonly base: GPUTexture;
  readonly atlas: GPUTexture | null;
  readonly gpuBytes: { textures: number; scratch: number };
  builtRevision: number | null = null;
  error: Promise<GPUError | null>;
  readonly #device: GPUDevice;
  readonly #pool: VDBGridPool;
  readonly #boxMin: VDBVec3;
  readonly #owned: (GPUBuffer | GPUTexture)[] = [];
  readonly #baseF32: GPUBuffer;
  readonly #atlasF32: GPUBuffer;
  readonly #flags: GPUBuffer;
  readonly #packed: GPUBuffer;
  readonly #params: GPUBuffer;
  readonly #rows: { base: number; atlas: number; atlasOffset: number };
  readonly #blocks: [number, number, number];
  /** Where the base and the atlas land in their textures (K6.7 packing; 0 otherwise). */
  #origins: { base: VDBVec3; atlas: VDBVec3 } = { base: [0, 0, 0], atlas: [0, 0, 0] };
  #group1: GPUBindGroup | null = null;
  #destroyed = false;

  constructor(device: GPUDevice, pool: VDBGridPool, gridId: number, o: VDBBoxPyramidGpuOptions) {
    if (!device || typeof (device as { createBuffer?: unknown }).createBuffer !== "function") bad("expected a GPUDevice as the first argument.");
    if (!pool || typeof (pool as { info?: unknown }).info !== "function") bad("expected a VDBGridPool as the second argument.");
    if (pool.device !== device) bad("the pool lives on a different GPUDevice. Pass pool.device.");
    // Residency is checked when a build is recorded; creating needs only the grid's info.
    const info = pool.info(gridId);
    const box = o.indexBox ?? info.indexBBox;
    const texels = [0, 1, 2].map((a) => box.max[a]! - box.min[a]! + 1);
    if (texels.some((t) => !(t > 0) || !Number.isInteger(t))) bad(`empty or non-integer index box ${JSON.stringify(box)}.`);
    this.layout = vdbBoxPyramidLayout(texels, { ...(o.pyramidMax !== undefined ? { pyramidMax: o.pyramidMax } : {}), ...(o.level !== undefined ? { level: o.level } : {}) });
    this.#device = device;
    this.#pool = pool;
    this.gridId = gridId;
    this.#boxMin = [box.min[0], box.min[1], box.min[2]];
    const label = o.label ?? `vdb boxPyramid grid ${gridId}`;
    const { dimensions: dims, atlas } = this.layout;
    const baseCount = dims[0] * dims[1] * dims[2];
    const atlasCount = atlas ? atlas.dimensions[0] * atlas.dimensions[1] * atlas.dimensions[2] : 0;

    if (o.targets) {
      const check = (t: GPUTexture | null, want: readonly number[] | null, origin: VDBVec3 | undefined, what: string) => {
        if (!want) {
          if (t) bad(`targets.${what} must be null: the layout has no ${what}.`);
          return;
        }
        if (!t) return bad(`targets.${what} is missing (the layout has one, ${want.join("x")}).`);
        const size = [t.width, t.height, t.depthOrArrayLayers];
        const fits = origin
          ? origin.every((v, a) => Number.isInteger(v) && v >= 0 && v + want[a]! <= size[a]!)
          : size.every((v, a) => v === want[a]);
        if (t.format !== "r16float" || !fits || (t.usage & TEXTURE_USAGE.COPY_DST) === 0) {
          const where = origin ? ` holding ${want.join("x")} at ${origin.join(",")}` : ` of ${want.join("x")}`;
          bad(`targets.${what} must be an r16float 3D texture${where} with COPY_DST (got ${t.format} ${t.width}x${t.height}x${t.depthOrArrayLayers}, usage ${t.usage}).`);
        }
      };
      check(o.targets.base, dims, o.targets.baseOrigin, "base");
      check(o.targets.atlas, atlas?.dimensions ?? null, o.targets.atlasOrigin, "atlas");
      this.#origins = { base: o.targets.baseOrigin ?? [0, 0, 0], atlas: o.targets.atlasOrigin ?? [0, 0, 0] };
      this.base = o.targets.base;
      this.atlas = o.targets.atlas;
    } else {
      const texture = (size: readonly number[], what: string) => {
        const t = device.createTexture({
          label: `${label} ${what}`,
          size: [size[0]!, size[1]!, size[2]!],
          dimension: "3d",
          format: "r16float",
          usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST | TEXTURE_USAGE.COPY_SRC,
        });
        this.#owned.push(t);
        return t;
      };
      this.base = texture(dims, "base");
      this.atlas = atlas ? texture(atlas.dimensions, "atlas") : null;
    }

    const rowWords = (w: number) => align(w * 2, 256) / 4;
    const baseWords = rowWords(dims[0]) * dims[1] * dims[2];
    const atlasWords = atlas ? rowWords(atlas.dimensions[0]) * atlas.dimensions[1] * atlas.dimensions[2] : 0;
    this.#rows = { base: rowWords(dims[0]), atlas: atlas ? rowWords(atlas.dimensions[0]) : 0, atlasOffset: align(baseWords, 64) };
    this.#blocks = [Math.ceil(dims[0] / 8), Math.ceil(dims[1] / 8), Math.ceil(dims[2] / 8)];
    const buffer = (size: number, usage: number, what: string) => {
      const b = device.createBuffer({ label: `${label} ${what}`, size: align(Math.max(4, size), 4), usage });
      this.#owned.push(b);
      return b;
    };
    this.#baseF32 = buffer(baseCount * 4, USAGE.STORAGE, "base f32");
    this.#atlasF32 = buffer(atlasCount * 4, USAGE.STORAGE | USAGE.COPY_DST, "atlas f32");
    this.#flags = buffer(this.#blocks[0] * this.#blocks[1] * this.#blocks[2] * 4, USAGE.STORAGE | USAGE.COPY_DST, "block flags");
    this.#packed = buffer((this.#rows.atlasOffset + atlasWords) * 4, USAGE.STORAGE | USAGE.COPY_SRC, "packed f16");
    const slots = 4 + (atlas?.levels.length ?? 0);
    this.#params = buffer(slots * SLOT, USAGE.UNIFORM | USAGE.COPY_DST, "params");
    this.gpuBytes = {
      textures: 2 * (baseCount + atlasCount),
      scratch: [this.#baseF32, this.#atlasF32, this.#flags, this.#packed, this.#params].reduce((s, b) => s + b.size, 0),
    };
    this.error = pyramidPipelines(device, info.type).error;
    this.#writeParams();
  }

  get stale(): boolean {
    if (this.#destroyed) return false;
    return this.builtRevision === null || this.#pool.info(this.gridId).imageRevision !== this.builtRevision;
  }

  /** Per-dispatch parameter slots: 0 base/mask, 1 pack base, 2 pack atlas, 3.. levels. */
  #writeParams(): void {
    const { dimensions: dims, atlas, texels } = this.layout;
    const slots = 4 + (atlas?.levels.length ?? 0);
    const bytes = new ArrayBuffer(slots * SLOT);
    const put = (slot: number, p: { srcFromBase?: number; srcOffset?: number; dstOffset?: number; rowWords?: number; dims: readonly number[]; srcDims?: readonly number[] }) => {
      const u = new Uint32Array(bytes, slot * SLOT, PARAMS_BYTES / 4);
      const i = new Int32Array(bytes, slot * SLOT, PARAMS_BYTES / 4);
      u[0] = this.gridId;
      u[1] = p.srcFromBase ?? 0;
      u[2] = p.srcOffset ?? 0;
      u[3] = p.dstOffset ?? 0;
      for (let a = 0; a < 3; a++) {
        i[4 + a] = this.#boxMin[a]!;
        u[8 + a] = texels[a]!;
        u[12 + a] = p.dims[a]!;
        u[16 + a] = p.srcDims?.[a] ?? 0;
        u[20 + a] = atlas?.dimensions[a] ?? 1;
      }
      u[7] = p.rowWords ?? 0;
      u[11] = this.#blocks[0];
      u[15] = this.#blocks[1];
    };
    put(0, { dims });
    put(1, { srcFromBase: 1, rowWords: this.#rows.base, dims, srcDims: dims });
    if (atlas) {
      put(2, { srcFromBase: 0, rowWords: this.#rows.atlas, dstOffset: this.#rows.atlasOffset, dims: atlas.dimensions, srcDims: atlas.dimensions });
      atlas.levels.forEach((level, l) => {
        const below = l === 0 ? { offset: 0, dimensions: dims } : atlas.levels[l - 1]!;
        put(3 + l, { srcFromBase: l === 0 ? 1 : 0, srcOffset: below.offset, dstOffset: level.offset, dims: level.dimensions, srcDims: below.dimensions });
      });
    }
    this.#device.queue.writeBuffer(this.#params, 0, bytes);
  }

  record(encoder: GPUCommandEncoder, options: VDBBoxPyramidRecordOptions = {}): void {
    if (this.#destroyed) throw new VDBError("VDB_DISPOSED", "vdbCreateBoxPyramid: record() after destroy().");
    const info = checkTarget("vdbCreateBoxPyramid", this.#device, this.#pool, this.gridId);
    const p = pyramidPipelines(this.#device, info.type);
    // Group 0 names the pool's current buffer: rebuilt every record (a growth replaces it).
    const group0 = this.#device.createBindGroup({
      layout: p.group0,
      entries: [this.#pool.bindGroupEntry(COMPUTE_SLOT), { binding: 1, resource: { buffer: this.#params, size: PARAMS_BYTES } }],
    });
    this.#group1 ??= this.#device.createBindGroup({
      layout: p.group1,
      entries: [this.#baseF32, this.#atlasF32, this.#flags, this.#packed].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const { dimensions: dims, atlas } = this.layout;
    const groups = (size: readonly number[]): [number, number, number] => [Math.ceil(size[0]! / W), Math.ceil(size[1]! / W), Math.ceil(size[2]! / W)];
    encoder.clearBuffer(this.#flags);
    if (atlas) encoder.clearBuffer(this.#atlasF32);
    const pass = encoder.beginComputePass({ label: "vdb boxPyramid", ...(options.timestampWrites ? { timestampWrites: options.timestampWrites } : {}) });
    pass.setBindGroup(1, this.#group1);
    const dispatch = (pipeline: GPUComputePipeline, slot: number, size: readonly number[]) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0, [slot * SLOT]);
      pass.dispatchWorkgroups(...groups(size));
    };
    dispatch(p.base, 0, dims);
    dispatch(p.mask, 0, dims);
    atlas?.levels.forEach((level, l) => dispatch(p.level, 3 + l, level.dimensions));
    dispatch(p.pack, 1, [Math.ceil(dims[0] / 2), dims[1], dims[2]]);
    if (atlas) dispatch(p.pack, 2, [Math.ceil(atlas.dimensions[0] / 2), atlas.dimensions[1], atlas.dimensions[2]]);
    pass.end();
    encoder.copyBufferToTexture({ buffer: this.#packed, offset: 0, bytesPerRow: this.#rows.base * 4, rowsPerImage: dims[1] }, { texture: this.base, origin: this.#origins.base }, [dims[0], dims[1], dims[2]]);
    if (atlas && this.atlas) {
      const [aw, ah, ad] = atlas.dimensions;
      encoder.copyBufferToTexture({ buffer: this.#packed, offset: this.#rows.atlasOffset * 4, bytesPerRow: this.#rows.atlas * 4, rowsPerImage: ah }, { texture: this.atlas, origin: this.#origins.atlas }, [aw, ah, ad]);
    }
    this.builtRevision = info.imageRevision;
  }

  update(encoder: GPUCommandEncoder): boolean {
    if (this.#destroyed) return false;
    const info = this.#pool.info(this.gridId);
    if (!info.resident || info.imageRevision === this.builtRevision) return false;
    this.record(encoder);
    return true;
  }

  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    for (const r of this.#owned) r.destroy();
  }
}

/**
 * Creates the coarse pyramid of pool grid `gridId` on the GPU (see the file comment) and returns it
 * unbuilt: record the build with `pyramid.record(encoder)` (the grid must be resident), submit, and
 * rebuild with `pyramid.update(encoder)` after the grid's image changes (`imageRevision`). The pool
 * must live on `device`. Frees only what it allocated (`destroy()`).
 *
 *     const pyramid = vdbCreateBoxPyramid(device, pool, id);
 *     pool.update(encoder);
 *     pyramid.update(encoder);           // first build, then only on a replace
 *     device.queue.submit([encoder.finish()]);
 *
 * @param device The host's `GPUDevice`; the pool must live on it.
 * @param pool The pool that holds the grid.
 * @param gridId The grid's pool id.
 * @param options Index box, size cap, level, target textures and label ({@link VDBBoxPyramidGpuOptions}).
 */
export function vdbCreateBoxPyramid(device: GPUDevice, pool: VDBGridPool, gridId: number, options: VDBBoxPyramidGpuOptions = {}): VDBBoxPyramidGpu {
  return new BoxPyramidGpu(device, pool, gridId, options);
}
