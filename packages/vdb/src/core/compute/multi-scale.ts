// Portions derived from mjurczyk/openvdb (MIT), commit 4e2d46f ("Add opt-in multi-scale density lookups for
// indirect light", after v0.29):
//   src/three-openvdb/frame/webgpu/frameMultiScaleChunks.js L1–71 (frameMultiScaleLight);
//   src/three-openvdb/frame/webgpu/FrameGridMipMaterial.js (the binomial branch: weights 1 3 3 1 on 4 x 4 x 4 taps);
//   src/three-openvdb/frame/webgpu/frameNeuralChunks.js (createNeuralMipNodes: sampleLevel, unit).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: WGSL compute on 3D textures (his levels are z-slice atlases drawn by a fragment pass); the chain is
// built in f32 and written as f16 by buffer copies (rg16float in core WebGPU, no storage-texture format needed);
// zero-padded binomial levels on an exactly halving layout by default (his clamped edges behind `edge: 'clamp'`);
// the lookup is a text WGSL function over host-bound textures. CPU twins: `tools/frame/multi-scale.ts`.
// Divergences: DIVERGES.md (K7.3 row).

/**
 * `vdbCreateMultiScaleMips` (K7.3): the binomial mip chain of a 3D grid texture for `'multi-scale'` multiple
 * scattering (mjurczyk 4e2d46f, Disney "Deep Scattering" §4.2), recorded into the host's encoder, plus the WGSL
 * lookup a transmittance kernel calls (`vdbFrameMultiScaleWgsl`). Engine-free; internal (K6.3 consumes it).
 *
 * Build, one compute pass (dispatches see each other's writes):
 * 1. **level** k = 1 … levels − 1 (one thread per texel of level k): the separable 1 3 3 1 sum of the 4 × 4 × 4
 *    taps 2x − 1 … 2x + 2 of level k − 1, / 512, in f32 into a scratch chain. Level 1 reads the host's texture
 *    (`textureLoad`, any float format, the selected channels, a region at `origin`); level k ≥ 2 reads the chain.
 * 2. **pack** (one thread per output word): f32 → f16 (round to nearest even, `vdbFloatToHalf`) into rows of
 *    256-byte multiples, then `copyBufferToTexture` into mip k − 1 of the output texture.
 *
 * Output: one 3D texture, mip j = level j + 1, sizes halving exactly (`vdbFrameMultiScaleLayout`). Format
 * r16float / rg16float / rgba16float by channel count; all three are filterable and copy targets in core WebGPU,
 * so no optional feature is needed (rg16float is not storage-capable without `texture-formats-tier1`, which is
 * why the kernel writes a buffer, not a storage texture). The texture is the chain's own
 * (`TEXTURE_BINDING | COPY_DST | COPY_SRC`) or the host's (`target`, never destroyed here).
 *
 * Lifetime: `resize` replaces the texture and the scratch (`version` bumps); the old ones are retired, as the
 * pool's buffers: destroyed after two more `record`s, once `queue.onSubmittedWorkDone()` asked then resolves, so
 * a bind group the host built on the old `view` for the frame in flight stays valid. `destroy()` frees everything
 * this object allocated (retired included), never `target`.
 */
import { VDBError } from "../errors.js";
import { USAGE, WORKGROUP_AXIS } from "./shared.js";
import {
  VDB_FRAME_MULTI_SCALE_DEFAULT_LEVELS,
  vdbFrameMultiScaleLayout,
  type VDBFrameMultiScaleEdge,
  type VDBFrameMultiScaleLayout,
} from "../../tools/frame/multi-scale.js";

const W = WORKGROUP_AXIS;
const STAGE_COMPUTE = 0x4;
const TEXTURE_USAGE = { COPY_SRC: 0x1, COPY_DST: 0x2, TEXTURE_BINDING: 0x4 } as const;
/** Uniform slot stride (minUniformBufferOffsetAlignment's default). */
const SLOT = 256;
const PARAMS_BYTES = 112;

/** Output formats of the chain: components per texel. */
export type VDBMultiScaleFormat = "r16float" | "rg16float" | "rgba16float";
const FORMAT_CHANNELS: Record<VDBMultiScaleFormat, number> = { r16float: 1, rg16float: 2, rgba16float: 4 };

const BUILD_WGSL = /* wgsl */ `
struct MsParams {
    srcOrigin: vec3u,
    fromTexture: u32,
    srcDims: vec3u,
    srcOffset: u32,
    srcExtent: vec3u,
    dstOffset: u32,
    dims: vec3u,
    edgeClamp: u32,
    dstExtent: vec3u,
    channels: u32,
    select: vec4u,
    rowWords: u32,
    texelChannels: u32,
    packOffset: u32,
    pad0: u32,
};
@group(0) @binding(0) var<uniform> ms: MsParams;
@group(0) @binding(1) var msSource: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> msScratch: array<u32>;

fn msWeight(i: i32) -> f32 {
    return select(3.0, 1.0, i == 0 || i == 3);
}

// Tap p of the source level, channel c: 0 outside it ('zero'), or clamped to its valid texels ('clamp').
fn msTap(p: vec3i, c: u32) -> f32 {
    var q = p;
    if ms.edgeClamp != 0u {
        q = clamp(p, vec3i(0), vec3i(ms.srcExtent) - vec3i(1));
    } else if any(p < vec3i(0)) || any(p >= vec3i(ms.srcDims)) {
        return 0.0;
    }
    if ms.fromTexture != 0u {
        let t = textureLoad(msSource, vec3u(q) + ms.srcOrigin, 0);
        return t[ms.select[c]];
    }
    let i = (u32(q.x) + (u32(q.y) + u32(q.z) * ms.srcDims.y) * ms.srcDims.x) * ms.channels + c;
    return bitcast<f32>(msScratch[ms.srcOffset + i]);
}

// One texel of level k: the 1 3 3 1 binomial of the 4 x 4 x 4 taps 2x - 1 ... 2x + 2 (x, then y, then z), / 512.
@compute @workgroup_size(${W}, ${W}, ${W})
fn csMsLevel(@builtin(global_invocation_id) id: vec3u) {
    if any(id >= ms.dims) {
        return;
    }
    var e = id;
    if ms.edgeClamp != 0u {
        e = min(id, ms.dstExtent - vec3u(1u));
    }
    let b = vec3i(e) * 2 - vec3i(1);
    let o = (id.x + (id.y + id.z * ms.dims.y) * ms.dims.x) * ms.channels;
    for (var c = 0u; c < ms.channels; c++) {
        var sz = 0.0;
        for (var l = 0; l < 4; l++) {
            var sy = 0.0;
            for (var j = 0; j < 4; j++) {
                var sx = 0.0;
                for (var i = 0; i < 4; i++) {
                    sx += msWeight(i) * msTap(b + vec3i(i, j, l), c);
                }
                sy += msWeight(j) * sx;
            }
            sz += msWeight(l) * sy;
        }
        msScratch[ms.dstOffset + o + c] = bitcast<u32>(sz / 512.0);
    }
}

// f32 -> f16 bits, round to nearest even (tools vdbFloatToHalf, after NanoVDB HalfLite).
fn msHalf(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let ui = bits & 0x7fffffffu;
    var h = (bits >> 16u) & 0x8000u;
    if ui >= 0x38800000u {
        if ui >= 0x7f800000u {
            h = h | 0x7c00u;
            if ui == 0x7f800000u {
                return h;
            }
            let m = (ui & 0x7fffffu) >> 13u;
            return h | m | select(0u, 1u, m == 0u);
        }
        if ui > 0x477fefffu {
            return h | 0x7c00u;
        }
        let r = ui - 0x38000000u;
        return h | ((r + 0x00000fffu + ((r >> 13u) & 1u)) >> 13u);
    }
    if ui < 0x33000001u {
        return h;
    }
    let shift = 0x7eu - (ui >> 23u);
    let m = 0x800000u | (ui & 0x7fffffu);
    let r = m << (32u - shift);
    h = h | (m >> shift);
    if r > 0x80000000u || (r == 0x80000000u && (h & 1u) != 0u) {
        h = h + 1u;
    }
    return h;
}

// Half number n of a row: texel n / texelChannels, component n % texelChannels (0 past the level's channels).
fn msRowHalf(y: u32, z: u32, n: u32) -> u32 {
    let x = n / ms.texelChannels;
    let c = n % ms.texelChannels;
    if x >= ms.dims.x || c >= ms.channels {
        return 0u;
    }
    let i = (x + (y + z * ms.dims.y) * ms.dims.x) * ms.channels + c;
    return msHalf(bitcast<f32>(msScratch[ms.dstOffset + i]));
}

// Two halves per word, rows of rowWords words (bytesPerRow a multiple of 256), at packOffset words.
@compute @workgroup_size(${W}, ${W}, ${W})
fn csMsPack(@builtin(global_invocation_id) id: vec3u) {
    let words = (ms.dims.x * ms.texelChannels + 1u) / 2u;
    if id.x >= words || id.y >= ms.dims.y || id.z >= ms.dims.z {
        return;
    }
    let lo = msRowHalf(id.y, id.z, id.x * 2u);
    let hi = msRowHalf(id.y, id.z, id.x * 2u + 1u);
    msScratch[ms.packOffset + (id.y + id.z * ms.dims.y) * ms.rowWords + id.x] = lo | (hi << 16u);
}
`;

/** The WGSL of the mip build kernels (exported for tests). */
export function multiScaleMipsWgsl(): string {
  return BUILD_WGSL;
}

interface Pipelines {
  level: GPUComputePipeline;
  pack: GPUComputePipeline;
  group0: GPUBindGroupLayout;
  error: Promise<GPUError | null>;
}

const pipelines = new WeakMap<GPUDevice, Pipelines>();

function multiScalePipelines(device: GPUDevice): Pipelines {
  const cached = pipelines.get(device);
  if (cached) return cached;
  device.pushErrorScope("validation");
  const group0 = device.createBindGroupLayout({
    label: "vdb multiScaleMips group 0",
    entries: [
      { binding: 0, visibility: STAGE_COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAMS_BYTES } },
      { binding: 1, visibility: STAGE_COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: STAGE_COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const module = device.createShaderModule({ label: "vdb multiScaleMips", code: BUILD_WGSL });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [group0] });
  const make = (entryPoint: string) => device.createComputePipeline({ label: `vdb multiScaleMips ${entryPoint}`, layout, compute: { module, entryPoint } });
  const p: Pipelines = { level: make("csMsLevel"), pack: make("csMsPack"), group0, error: device.popErrorScope() };
  pipelines.set(device, p);
  return p;
}

/** Options of {@link vdbCreateMultiScaleMips}. */
export interface VDBMultiScaleMipsOptions {
  /** Base texels per axis: the grid region the levels filter (one cascade of the shadow grid). */
  size: readonly [number, number, number];
  /** Levels including the base (2 to 8, default 6: his `multiScaleLevels` maximum). */
  levels?: number;
  /** Source components that become the level channels, in order (default [0, 1]: extinction, absorption). */
  channels?: readonly number[];
  /** Output format (default by channel count: 1 r16float, 2 rg16float, 3–4 rgba16float). */
  format?: VDBMultiScaleFormat;
  /** Edge rule (default `'zero'`; `'clamp'` is his clamp-to-edge). */
  edge?: VDBFrameMultiScaleEdge;
  /**
   * Write into this 3D texture instead of allocating: `format`, the layout's level-1 size, levels − 1 mips,
   * `COPY_DST`. Never destroyed here.
   */
  target?: GPUTexture;
  /** Debug label of the GPU objects. */
  label?: string;
}

/** Options of {@link VDBMultiScaleMips.record}. */
export interface VDBMultiScaleRecordOptions {
  /** First texel of the base region in the source texture (default 0, 0, 0): a cascade of a stacked grid. */
  origin?: readonly [number, number, number];
  /** Timestamps around the build's one compute pass (the `timestamp-query` feature). */
  timestampWrites?: GPUComputePassTimestampWrites;
}

/** Options of {@link VDBMultiScaleMips.resize}. */
export interface VDBMultiScaleResizeOptions {
  /** Levels (default: unchanged). */
  levels?: number;
  /** The host texture for the new size (required when the chain was made with `target`). */
  target?: GPUTexture;
}

/** The mip chain of a grid texture on the GPU (see the file comment). */
export interface VDBMultiScaleMips {
  /** The level sizes (levels 1 … levels − 1 = mips 0 … levels − 2 of {@link texture}). */
  readonly layout: VDBFrameMultiScaleLayout;
  /** Components per level texel (the `channels` option's length). */
  readonly channels: number;
  /** The texture's format. */
  readonly format: VDBMultiScaleFormat;
  /** The levels as one 3D texture with levels − 1 mips. */
  readonly texture: GPUTexture;
  /** A 3D view of every mip of {@link texture}: bind it as the lookup's `mips`. */
  readonly view: GPUTextureView;
  /** Bumps when `resize` replaces {@link texture}: rebuild bind groups that hold {@link view}. */
  readonly version: number;
  /** Bytes: `textures` (counted whoever owns them) and `scratch` (this chain's buffers). */
  readonly gpuBytes: { readonly textures: number; readonly scratch: number };
  /** Pipeline validation (null = valid). */
  readonly error: Promise<GPUError | null>;
  /**
   * Records a build of every level from `source` into `encoder`; the host submits it. Record it after the
   * pass that writes `source` (in the same encoder or an earlier submit).
   *
   * @param encoder The host's encoder.
   * @param source The base grid: a 3D texture with `TEXTURE_BINDING` and a float sample type, or a 3D view of
   *   one (then the size and format are the host's to check). Holds the region at `origin`.
   * @param options Region origin and timestamps. The origin lives in a uniform written with `queue.writeBuffer`,
   *   so builds from two origins need two submits.
   */
  record(encoder: GPUCommandEncoder, source: GPUTexture | GPUTextureView, options?: VDBMultiScaleRecordOptions): void;
  /** New base size (and levels): new texture and scratch, `version` bumps, the old ones retired (file comment). */
  resize(size: readonly [number, number, number], options?: VDBMultiScaleResizeOptions): void;
  /** Frees what this chain allocated, retired resources included (never `target`). Idempotent. */
  destroy(): void;
}

const align = (n: number, a: number) => Math.ceil(n / a) * a;
const bad = (message: string): never => {
  throw new VDBError("VDB_BAD_OPTION", `vdbCreateMultiScaleMips: ${message}`);
};

/** Updates a retired texture must outlive before the chain asks the queue about it (the pool's rule). */
const RETIRE_AFTER_RECORDS = 2;

interface Resources {
  layout: VDBFrameMultiScaleLayout;
  texture: GPUTexture;
  view: GPUTextureView;
  owned: (GPUBuffer | GPUTexture)[];
  scratch: GPUBuffer;
  params: GPUBuffer;
  /** Per level: chain word offset, packed word offset, packed row words. */
  offsets: { chain: number; pack: number; rowWords: number }[];
  gpuBytes: { textures: number; scratch: number };
}

const isTexture = (s: GPUTexture | GPUTextureView): s is GPUTexture => typeof (s as GPUTexture).createView === "function";

class MultiScaleMips implements VDBMultiScaleMips {
  readonly channels: number;
  readonly format: VDBMultiScaleFormat;
  readonly error: Promise<GPUError | null>;
  version = 0;
  readonly #device: GPUDevice;
  readonly #select: readonly number[];
  readonly #edge: VDBFrameMultiScaleEdge;
  readonly #label: string;
  readonly #hostTarget: boolean;
  #r: Resources;
  #records = 0;
  #retired: { resources: (GPUBuffer | GPUTexture)[]; since: number; waiting: boolean }[] = [];
  #destroyed = false;

  constructor(device: GPUDevice, o: VDBMultiScaleMipsOptions) {
    if (!device || typeof (device as { createBuffer?: unknown }).createBuffer !== "function") bad("expected a GPUDevice as the first argument.");
    if (!o || typeof o !== "object") bad("expected an options object with size.");
    const select = o.channels ?? [0, 1];
    if (!Array.isArray(select) || select.length < 1 || select.length > 4 || select.some((c) => !Number.isInteger(c) || c < 0 || c > 3)) {
      bad(`channels must be 1 to 4 component indices from 0 to 3, got ${JSON.stringify(o.channels)}.`);
    }
    const format = o.format ?? (select.length === 1 ? "r16float" : select.length === 2 ? "rg16float" : "rgba16float");
    if (!(format in FORMAT_CHANNELS)) bad(`format must be r16float, rg16float or rgba16float, got ${JSON.stringify(o.format)}.`);
    if (FORMAT_CHANNELS[format] < select.length) bad(`format ${format} holds ${FORMAT_CHANNELS[format]} channel(s); channels has ${select.length}.`);
    const edge = o.edge ?? "zero";
    if (edge !== "zero" && edge !== "clamp") bad(`edge must be 'zero' or 'clamp', got ${JSON.stringify(o.edge)}.`);
    this.#device = device;
    this.#select = [...select];
    this.channels = select.length;
    this.format = format;
    this.#edge = edge;
    this.#label = o.label ?? "vdb multiScaleMips";
    this.#hostTarget = !!o.target;
    this.#r = this.#allocate(o.size, o.levels ?? VDB_FRAME_MULTI_SCALE_DEFAULT_LEVELS, o.target);
    this.error = multiScalePipelines(device).error;
  }

  get layout(): VDBFrameMultiScaleLayout {
    return this.#r.layout;
  }
  get texture(): GPUTexture {
    return this.#r.texture;
  }
  get view(): GPUTextureView {
    return this.#r.view;
  }
  get gpuBytes(): { textures: number; scratch: number } {
    return this.#r.gpuBytes;
  }

  #allocate(size: readonly number[], levels: number, target: GPUTexture | undefined): Resources {
    let layout: VDBFrameMultiScaleLayout;
    try {
      layout = vdbFrameMultiScaleLayout(size, levels);
    } catch (e) {
      return bad(e instanceof Error ? e.message.replace(/^.*?vdbFrameMultiScaleLayout: /, "") : String(e));
    }
    const device = this.#device;
    const F = FORMAT_CHANNELS[this.format];
    const C = this.channels;
    const top = layout.mips[0]!.dimensions;
    const mipCount = layout.levels - 1;
    const owned: (GPUBuffer | GPUTexture)[] = [];
    let texture: GPUTexture;
    if (target) {
      if (target.format !== this.format || target.dimension !== "3d" || target.width !== top[0] || target.height !== top[1] || target.depthOrArrayLayers !== top[2] || target.mipLevelCount !== mipCount || (target.usage & TEXTURE_USAGE.COPY_DST) === 0) {
        bad(
          `target must be a ${this.format} 3D texture of ${top.join("x")} with ${mipCount} mip levels and COPY_DST ` +
            `(got ${target.format} ${target.dimension} ${target.width}x${target.height}x${target.depthOrArrayLayers}, ${target.mipLevelCount} mips, usage ${target.usage}).`,
        );
      }
      texture = target;
    } else {
      if (this.#hostTarget) bad("this chain writes into a host target: pass options.target for the new size.");
      texture = device.createTexture({
        label: `${this.#label} levels`,
        size: [top[0], top[1], top[2]],
        dimension: "3d",
        format: this.format,
        mipLevelCount: mipCount,
        usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST | TEXTURE_USAGE.COPY_SRC,
      });
      owned.push(texture);
    }
    const offsets: Resources["offsets"] = [];
    let chain = 0;
    for (const m of layout.mips) {
      offsets.push({ chain, pack: 0, rowWords: align(m.dimensions[0] * F * 2, 256) / 4 });
      chain += m.dimensions[0] * m.dimensions[1] * m.dimensions[2] * C;
    }
    let pack = align(chain, 64);
    layout.mips.forEach((m, i) => {
      offsets[i]!.pack = pack;
      pack = align(pack + offsets[i]!.rowWords * m.dimensions[1] * m.dimensions[2], 64);
    });
    const scratch = device.createBuffer({ label: `${this.#label} scratch`, size: Math.max(4, pack * 4), usage: USAGE.STORAGE | USAGE.COPY_SRC });
    const params = device.createBuffer({ label: `${this.#label} params`, size: 2 * mipCount * SLOT, usage: USAGE.UNIFORM | USAGE.COPY_DST });
    owned.push(scratch, params);
    const textureBytes = layout.mips.reduce((s, m) => s + m.dimensions[0] * m.dimensions[1] * m.dimensions[2] * F * 2, 0);
    const r: Resources = {
      layout,
      texture,
      view: texture.createView({ label: `${this.#label} view`, dimension: "3d" }),
      owned,
      scratch,
      params,
      offsets,
      gpuBytes: { textures: textureBytes, scratch: scratch.size + params.size },
    };
    this.#writeParams(r, [0, 0, 0]);
    return r;
  }

  /** Slots: level k at 2(k − 1), its pack at 2(k − 1) + 1. Only the level-1 slot depends on `origin`. */
  #writeParams(r: Resources, origin: readonly number[]): void {
    const { layout, offsets } = r;
    const bytes = new ArrayBuffer(2 * layout.mips.length * SLOT);
    const F = FORMAT_CHANNELS[this.format];
    layout.mips.forEach((m, i) => {
      const fromBase = i === 0;
      const srcDims = fromBase ? layout.size : layout.mips[i - 1]!.dimensions;
      const srcExtent = fromBase ? layout.size : layout.mips[i - 1]!.extent;
      for (const pack of [false, true]) {
        const u = new Uint32Array(bytes, (2 * i + (pack ? 1 : 0)) * SLOT, PARAMS_BYTES / 4);
        for (let a = 0; a < 3; a++) {
          u[a] = fromBase ? origin[a]! : 0;
          u[4 + a] = srcDims[a]!;
          u[8 + a] = srcExtent[a]!;
          u[12 + a] = m.dimensions[a]!;
          u[16 + a] = m.extent[a]!;
        }
        u[3] = fromBase ? 1 : 0;
        u[7] = fromBase ? 0 : offsets[i - 1]!.chain;
        u[11] = offsets[i]!.chain;
        u[15] = this.#edge === "clamp" ? 1 : 0;
        u[19] = this.channels;
        for (let c = 0; c < 4; c++) u[20 + c] = this.#select[c] ?? 0;
        u[24] = offsets[i]!.rowWords;
        u[25] = F;
        u[26] = offsets[i]!.pack;
      }
    });
    this.#device.queue.writeBuffer(r.params, 0, bytes);
  }

  #origin: readonly number[] = [0, 0, 0];

  record(encoder: GPUCommandEncoder, source: GPUTexture | GPUTextureView, options: VDBMultiScaleRecordOptions = {}): void {
    if (this.#destroyed) throw new VDBError("VDB_DISPOSED", "vdbCreateMultiScaleMips: record() after destroy().");
    if (!source || typeof source !== "object") bad("record: expected the source GPUTexture or GPUTextureView.");
    const origin = options.origin ?? [0, 0, 0];
    if (origin.length !== 3 || origin.some((v) => !Number.isInteger(v) || v < 0)) bad(`record: origin must be 3 integers >= 0, got ${JSON.stringify(origin)}.`);
    const r = this.#r;
    const size = r.layout.size;
    let view: GPUTextureView;
    if (isTexture(source)) {
      const f = source.format;
      if (source.dimension !== "3d" || (source.usage & TEXTURE_USAGE.TEXTURE_BINDING) === 0 || /int|depth|stencil/.test(f)) {
        bad(`record: the source must be a 3D texture with TEXTURE_BINDING and a float format (got ${source.dimension} ${f}, usage ${source.usage}).`);
      }
      const extent = [source.width, source.height, source.depthOrArrayLayers];
      if ([0, 1, 2].some((a) => origin[a]! + size[a]! > extent[a]!)) {
        bad(`record: the region (origin ${origin.join(",")}, size ${size.join("x")}) is outside the source (${extent.join("x")}).`);
      }
      view = source.createView({ dimension: "3d", baseMipLevel: 0, mipLevelCount: 1 });
    } else {
      view = source;
    }
    this.#records++;
    this.#reap();
    if (origin.some((v, a) => v !== this.#origin[a])) {
      this.#origin = [...origin];
      this.#writeParams(r, origin);
    }
    const device = this.#device;
    const p = multiScalePipelines(device);
    const group = device.createBindGroup({
      label: `${this.#label} group`,
      layout: p.group0,
      entries: [
        { binding: 0, resource: { buffer: r.params, size: PARAMS_BYTES } },
        { binding: 1, resource: view },
        { binding: 2, resource: { buffer: r.scratch } },
      ],
    });
    const F = FORMAT_CHANNELS[this.format];
    const groups = (s: readonly number[]): [number, number, number] => [Math.ceil(s[0]! / W), Math.ceil(s[1]! / W), Math.ceil(s[2]! / W)];
    const pass = encoder.beginComputePass({ label: "vdb multiScaleMips", ...(options.timestampWrites ? { timestampWrites: options.timestampWrites } : {}) });
    r.layout.mips.forEach((m, i) => {
      pass.setPipeline(p.level);
      pass.setBindGroup(0, group, [2 * i * SLOT]);
      pass.dispatchWorkgroups(...groups(m.dimensions));
    });
    r.layout.mips.forEach((m, i) => {
      pass.setPipeline(p.pack);
      pass.setBindGroup(0, group, [(2 * i + 1) * SLOT]);
      pass.dispatchWorkgroups(...groups([Math.ceil((m.dimensions[0] * F) / 2), m.dimensions[1], m.dimensions[2]]));
    });
    pass.end();
    r.layout.mips.forEach((m, i) => {
      const o = r.offsets[i]!;
      encoder.copyBufferToTexture(
        { buffer: r.scratch, offset: o.pack * 4, bytesPerRow: o.rowWords * 4, rowsPerImage: m.dimensions[1] },
        { texture: r.texture, mipLevel: i },
        [m.dimensions[0], m.dimensions[1], m.dimensions[2]],
      );
    });
  }

  resize(size: readonly [number, number, number], options: VDBMultiScaleResizeOptions = {}): void {
    if (this.#destroyed) throw new VDBError("VDB_DISPOSED", "vdbCreateMultiScaleMips: resize() after destroy().");
    const next = this.#allocate(size, options.levels ?? this.#r.layout.levels, options.target);
    this.#retired.push({ resources: this.#r.owned, since: this.#records, waiting: false });
    this.#r = next;
    this.#origin = [0, 0, 0];
    this.version++;
  }

  /** The pool's rule: retired at least two records ago, then destroyed once the queue says that work is done. */
  #reap(): void {
    const ready = this.#retired.filter((r) => !r.waiting && this.#records - r.since >= RETIRE_AFTER_RECORDS);
    if (ready.length === 0) return;
    for (const r of ready) r.waiting = true;
    const destroy = (): void => {
      if (this.#destroyed) return; // destroy() freed them
      this.#retired = this.#retired.filter((r) => !ready.includes(r));
      for (const r of ready) for (const x of r.resources) x.destroy();
    };
    this.#device.queue.onSubmittedWorkDone().then(destroy, destroy);
  }

  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    for (const r of this.#retired) for (const x of r.resources) x.destroy();
    this.#retired = [];
    for (const x of this.#r.owned) x.destroy();
  }
}

/**
 * Creates the binomial mip chain of a grid texture for `'multi-scale'` multiple scattering (see the file comment)
 * and returns it unbuilt: record a build with `mips.record(encoder, source)` after the pass that writes the grid,
 * then bind `mips.view` as the lookup's `mips` (`vdbFrameMultiScaleWgsl`).
 *
 *     const mips = vdbCreateMultiScaleMips(device, { size: [64, 64, 64] });
 *     mips.record(encoder, shadowGrid, { origin: [0, 0, 64 * lastCascade] });
 *     device.queue.submit([encoder.finish()]);
 *
 * @param device The host's `GPUDevice`.
 * @param options Size, levels, channels, format, edge rule, target and label ({@link VDBMultiScaleMipsOptions}).
 */
export function vdbCreateMultiScaleMips(device: GPUDevice, options: VDBMultiScaleMipsOptions): VDBMultiScaleMips {
  return new MultiScaleMips(device, options);
}

/** Bytes of the WGSL `VdbFrameMultiScale` struct ({@link vdbFrameMultiScaleParams}). */
export const VDB_FRAME_MULTI_SCALE_PARAMS_BYTES = 64;

/** The lookup's sampler: linear, clamp-to-edge (the lookup picks each mip with an explicit level). */
export const VDB_FRAME_MULTI_SCALE_SAMPLER: GPUSamplerDescriptor = {
  label: "vdb multiScale sampler",
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "nearest",
  addressModeU: "clamp-to-edge",
  addressModeV: "clamp-to-edge",
  addressModeW: "clamp-to-edge",
};

/** The lookup's grid (WGSL `VdbFrameMultiScale`; the CPU twin's `VDBFrameMultiScaleGrid` plus the base region). */
export interface VDBFrameMultiScaleParams {
  /** World box of the base region (his last cascade: gridMin, gridSize). */
  boxMin: readonly [number, number, number];
  boxSize: readonly [number, number, number];
  /** h: the grid's largest voxel side, world units. */
  unit: number;
  /** Levels read, base included (clamped in WGSL to the mips present + 1). */
  levels: number;
  /** 1 − g; 0 gives no scattered light. */
  multipleScattering: number;
  /** First texel of the base region in the base texture (default 0, 0, 0). */
  baseOrigin?: readonly [number, number, number];
  /** Base region texels per axis (the chain's `layout.size`). */
  baseSize: readonly [number, number, number];
}

/**
 * The 64 bytes of a WGSL `VdbFrameMultiScale` (into `target` at `byteOffset`, or a new buffer): boxMin, levels,
 * boxSize, unit, baseOrigin, multipleScattering, baseSize, pad.
 */
export function vdbFrameMultiScaleParams(p: VDBFrameMultiScaleParams, target: ArrayBuffer = new ArrayBuffer(VDB_FRAME_MULTI_SCALE_PARAMS_BYTES), byteOffset = 0): ArrayBuffer {
  const f = new Float32Array(target, byteOffset, 16);
  const u = new Uint32Array(target, byteOffset, 16);
  const o = p.baseOrigin ?? [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    f[a] = p.boxMin[a]!;
    f[4 + a] = p.boxSize[a]!;
    f[8 + a] = o[a]!;
    f[12 + a] = p.baseSize[a]!;
  }
  u[3] = p.levels;
  f[7] = p.unit;
  f[11] = p.multipleScattering;
  f[15] = 0;
  return target;
}

/** Options of {@link vdbFrameMultiScaleWgsl}. */
export interface VDBFrameMultiScaleWgslOptions {
  /** Base texture components of extinction and absorption (default [0, 1]; one entry: no absorption). */
  baseChannels?: readonly number[];
  /** Components of the mip texture (default: as many as `baseChannels`; 1 reads no absorption). */
  mipChannels?: number;
}

const COMPONENTS = ["x", "y", "z", "w"];

/**
 * The WGSL lookup of `'multi-scale'` multiple scattering, as text to paste into a host shader once (not part of
 * the VDB module, so module bytes and the size snapshot never move). Declares:
 *
 *     struct VdbFrameMultiScale { boxMin: vec3f, levels: u32, boxSize: vec3f, unit: f32,
 *                                 baseOrigin: vec3f, multipleScattering: f32, baseSize: vec3f, pad0: f32 }
 *     fn vdbFrameMultiScaleSample(base: texture_3d<f32>, mips: texture_3d<f32>, smp: sampler,
 *                                 ms: VdbFrameMultiScale, level: u32, world: vec3f) -> vec2f
 *     fn vdbFrameMultiScaleLight(base: texture_3d<f32>, mips: texture_3d<f32>, smp: sampler,
 *                                ms: VdbFrameMultiScale, world: vec3f, direction: vec3f, maxDistance: f32) -> f32
 *
 * `base` is the host's grid texture (filterable float), `mips` the chain's `view`, `smp` a sampler made from
 * {@link VDB_FRAME_MULTI_SCALE_SAMPLER}; the host binds all three at its own slots and fills the struct with
 * {@link vdbFrameMultiScaleParams}. `vdbFrameMultiScaleLight` returns the scattered light (total − direct) for a
 * unit direction to the light; the caller adds its march's direct transmittance and clamps to 1. No derivatives
 * (`textureSampleLevel`), so it runs in compute and in non-uniform control flow. CPU twins:
 * `tools/frame/multi-scale.ts`.
 */
export function vdbFrameMultiScaleWgsl(options: VDBFrameMultiScaleWgslOptions = {}): string {
  const base = options.baseChannels ?? [0, 1];
  if (base.length < 1 || base.length > 2 || base.some((c) => !Number.isInteger(c) || c < 0 || c > 3)) {
    throw new VDBError("VDB_BAD_OPTION", `vdbFrameMultiScaleWgsl: baseChannels must be 1 or 2 component indices from 0 to 3, got ${JSON.stringify(options.baseChannels)}.`);
  }
  const mipChannels = options.mipChannels ?? base.length;
  if (!Number.isInteger(mipChannels) || mipChannels < 1 || mipChannels > 4) {
    throw new VDBError("VDB_BAD_OPTION", `vdbFrameMultiScaleWgsl: mipChannels must be 1 to 4, got ${JSON.stringify(options.mipChannels)}.`);
  }
  const baseValue = base.length > 1 ? `vec2f(s.${COMPONENTS[base[0]!]!}, s.${COMPONENTS[base[1]!]!})` : `vec2f(s.${COMPONENTS[base[0]!]!}, 0.0)`;
  const mipValue = mipChannels > 1 ? "vec2f(s.x, s.y)" : "vec2f(s.x, 0.0)";
  return /* wgsl */ `
struct VdbFrameMultiScale {
    boxMin: vec3f,
    levels: u32,
    boxSize: vec3f,
    unit: f32,
    baseOrigin: vec3f,
    multipleScattering: f32,
    baseSize: vec3f,
    pad0: f32,
};

// (extinction, absorption) of a level at a world position: 0 outside the box; level 0 is the base region with
// its edges clamped; level k is mip k - 1 at base texel / 2^k.
fn vdbFrameMultiScaleSample(base: texture_3d<f32>, mips: texture_3d<f32>, smp: sampler, ms: VdbFrameMultiScale, level: u32, world: vec3f) -> vec2f {
    let uvw = (world - ms.boxMin) / ms.boxSize;
    if any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0)) {
        return vec2f(0.0);
    }
    let q = uvw * ms.baseSize;
    if level == 0u {
        let c = ms.baseOrigin + clamp(q, vec3f(0.5), ms.baseSize - vec3f(0.5));
        let s = textureSampleLevel(base, smp, c / vec3f(textureDimensions(base, 0u)), 0.0);
        return ${baseValue};
    }
    let dims = vec3f(textureDimensions(mips, level - 1u));
    let s = textureSampleLevel(mips, smp, q / (exp2(f32(level)) * dims), f32(level - 1u));
    return ${mipValue};
}

// Scattered light (total - direct) of a light slot: level k reads the bulk of the segment h (2^k - 1) to
// h (2^(k+1) - 1) toward the light (5 points: the axis and +-h 2^k on two axes at right angles), the 1D diffusion
// of the summed bulk gives total - its direct part (mjurczyk 4e2d46f, after Disney "Deep Scattering" 4.2).
fn vdbFrameMultiScaleLight(base: texture_3d<f32>, mips: texture_3d<f32>, smp: sampler, ms: VdbFrameMultiScale, world: vec3f, direction: vec3f, maxDistance: f32) -> f32 {
    let h = ms.unit;
    let g = ms.multipleScattering;
    // Frisvad frame, pole at -y.
    let a = 1.0 / max(direction.y + 1.0, 1e-4);
    let b = -direction.x * direction.z * a;
    var u = vec3f(1.0 - direction.x * direction.x * a, -direction.x, b);
    var v = vec3f(b, -direction.z, 1.0 - direction.z * direction.z * a);
    if direction.y < -0.9999 {
        u = vec3f(1.0, 0.0, 0.0);
        v = vec3f(0.0, 0.0, 1.0);
    }
    var depths = vec3f(0.0);
    let count = min(ms.levels, textureNumLevels(mips) + 1u);
    for (var k = 0u; k < count; k++) {
        let scale = h * exp2(f32(k));
        let start = scale - h;
        let segment = min(scale, maxDistance - start);
        if segment > 0.0 {
            let center = world + direction * (start + segment * 0.5);
            let bulk = (vdbFrameMultiScaleSample(base, mips, smp, ms, k, center)
                + vdbFrameMultiScaleSample(base, mips, smp, ms, k, center + u * scale)
                + vdbFrameMultiScaleSample(base, mips, smp, ms, k, center - u * scale)
                + vdbFrameMultiScaleSample(base, mips, smp, ms, k, center + v * scale)
                + vdbFrameMultiScaleSample(base, mips, smp, ms, k, center - v * scale)) * 0.2;
            let absorbed = min(bulk.y, bulk.x);
            let reduced = (bulk.x - absorbed) * g;
            depths += vec3f(bulk.x, sqrt(absorbed * 3.0 * (absorbed + reduced)), reduced) * segment;
        }
    }
    var total = depths.x;
    if g > 0.0 {
        total = min(depths.x, depths.y + log(depths.z * 0.75 + 1.0));
    }
    return max(exp(-total) - exp(-depths.x), 0.0);
}
`;
}
