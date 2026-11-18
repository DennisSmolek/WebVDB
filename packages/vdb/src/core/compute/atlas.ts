/**
 * `decodeToAtlas`: dense decode of one pool grid's index bbox into a flat f32
 * array (x fastest), recorded into the host's encoder (K2.3, ported from v1
 * `three/compute.ts`). CPU twin: `vdbDecodeToAtlasCpu` + `atlasFromRaw` in
 * `@pmndrs/vdb/tools`.
 *
 * One thread per output voxel samples the index bbox at the voxel-centre
 * position of its cell (below) with `vdbReadValue(floor(p + 0.5))`
 * (`'nearest'`) or `vdbSample` (`'trilinear'`, the vendored trilinear,
 * `outside: 'background'`).
 *
 * **Sampling convention (K2.6; voxel centres).** NanoVDB voxel centres sit at
 * integer index coordinates, and the bbox `[min, max]` (inclusive) holds
 * `size = max - min + 1` voxels, i.e. the cells `[min - 0.5, max + 0.5]`.
 * Cell `i` of an axis with `n` cells over `[min, max]` samples the index-space
 * position
 *
 *     p = min + (i + 0.5) * (max - min + 1) / n - 0.5
 *
 * (the centre of the i-th of `n` equal slices of `[min - 0.5, max + 0.5]`).
 * At full resolution (`n == size`) that is `p = min + i`: `'nearest'` reads
 * exactly voxel `min + i` (the last cell is voxel `max`) and `'trilinear'`
 * reads that voxel's value unblended. Downsampled, each cell samples the
 * centre of its slice. The atlas is a texture over the voxel-centred box
 * `[min - 0.5, max + 0.5]`: texel `i`'s centre `(i + 0.5) / n` maps to `p`
 * above. (Before K2.6 this was v1's `min + (i + 0.5) / n * size`, half a voxel
 * past the centres.) The GPU writes raw f32; normalizing and quantizing to
 * `uint8` happen on the CPU after readback, as in v1.
 *
 * The raw buffer is also the GPU-side output: with `readback: false` the
 * host keeps `buffer` (`STORAGE | COPY_SRC`, `dims` product f32s) for its own
 * passes and calls `dispose()` when done.
 */
import type { VDBGridPool } from "../pool/pool.js";
import { VDBError } from "../errors.js";
import {
  checkPositiveInt,
  checkTarget,
  ComputeJob,
  createInitialized,
  hostEncoder,
  kernelPipeline,
  packParams,
  USAGE,
  workgroups,
  WORKGROUP_AXIS,
  type KernelSource,
  type VDBReadback,
} from "./shared.js";
import { atlasDims, atlasFromRaw, bboxSize, type VDBAtlasFilter, type VDBAtlasFormat } from "../../tools/cpu/compute.js";

export type { VDBAtlasFilter, VDBAtlasFormat };

const W = WORKGROUP_AXIS;

const ATLAS: KernelSource = {
  entryPoint: "csAtlas",
  outputs: 1,
  footer: /* wgsl */ `
@group(1) @binding(0) var<storage, read_write> csAtlasOut: array<f32>;

@compute @workgroup_size(${W}, ${W}, ${W})
fn csAtlas(@builtin(global_invocation_id) gid: vec3u) {
    let dims = csParams.dims;
    if any(gid >= dims) {
        return;
    }
    // Voxel-centre position: min + (i + 0.5) * size / dims - 0.5. Multiply
    // before dividing and subtract the half voxel before adding min, so at
    // full resolution the offset is exactly i (the CPU twin does the same).
    let pos = vec3f(csParams.regionMin) + ((vec3f(gid) + vec3f(0.5)) * vec3f(csParams.regionSize) / vec3f(dims) - vec3f(0.5));
    var acc = vdbAccessorInit(csParams.grid);
    var value: f32;
    if csParams.filterMode == 1u {
        value = vdbSample(&acc, pos);
    } else {
        value = vdbReadValue(&acc, vec3i(floor(pos + vec3f(0.5))));
    }
    csAtlasOut[gid.x + gid.y * dims.x + gid.z * dims.x * dims.y] = value;
}
`,
};

/** Options of {@link vdbDecodeToAtlasGpu} (v1's, plus `encoder` and `readback`). */
export interface VDBDecodeToAtlasOptions {
  /** Per-axis output resolution cap (a literal clamp, not a proportional downscale). Default 256. */
  maxDim?: number;
  /** Default `'trilinear'`. */
  filter?: VDBAtlasFilter;
  /** Element type of `read()`'s `data`. Default `'uint8'`. */
  format?: VDBAtlasFormat;
  /** Map `range` to `[0, 1]` (`float32`) or `[0, 255]` (`uint8`) in `read()`. Default `true`. */
  normalize?: boolean;
  /** Normalization range. Default: the grid's root min/max statistics (`pool.info(id).rootMin/rootMax`). */
  range?: { min: number; max: number };
  /**
   * Record into this encoder. The host submits it, then calls
   * `job.submitted().read()` (`read()` before `submitted()` rejects with
   * `VDB_BAD_OPTION`). Omitted: the tool submits its own.
   */
  encoder?: GPUCommandEncoder;
  /** `false`: no readback; keep `buffer` for GPU use and `dispose()` it. Default `true`. */
  readback?: boolean;
}

/** What `read()` resolves with (v1's `DecodeToAtlasResult`). */
export interface VDBAtlas {
  /** `dims` product values, x fastest. */
  data: Uint8Array | Float32Array;
  dims: [number, number, number];
  /** The range used for normalization (echoed so the caller can undo it). */
  range: { min: number; max: number };
}

/** The pending atlas: a {@link VDBReadback} plus the GPU output. */
export interface VDBAtlasReadback extends VDBReadback<VDBAtlas> {
  readonly dims: [number, number, number];
  readonly range: { min: number; max: number };
  /** Raw f32 values on the GPU (`STORAGE | COPY_SRC`); freed by `read()` / `dispose()`. */
  readonly buffer: GPUBuffer;
}

const WHERE = "decodeToAtlas";

function checkAtlasOptions(o: VDBDecodeToAtlasOptions): void {
  checkPositiveInt(WHERE, "maxDim", o.maxDim ?? 256);
  if (o.filter !== undefined && o.filter !== "nearest" && o.filter !== "trilinear") {
    throw new VDBError("VDB_BAD_OPTION", `${WHERE}: filter must be "nearest" or "trilinear", got ${JSON.stringify(o.filter)}.`);
  }
  if (o.format !== undefined && o.format !== "uint8" && o.format !== "float32") {
    throw new VDBError("VDB_BAD_OPTION", `${WHERE}: format must be "uint8" or "float32", got ${JSON.stringify(o.format)}.`);
  }
  if (o.range !== undefined && !(Number.isFinite(o.range?.min) && Number.isFinite(o.range?.max))) {
    throw new VDBError("VDB_BAD_OPTION", `${WHERE}: range must be { min, max } with finite numbers, got ${JSON.stringify(o.range)}.`);
  }
}

/**
 * Records the decode of grid `gridId` of `pool` and returns a
 * {@link VDBAtlasReadback}: `const { data, dims } = await decodeToAtlas(device, pool, id).read()`
 * (with a host encoder: submit it, then `await job.submitted().read()`).
 * Cell `i` samples index position `min + (i + 0.5) * size / n - 0.5`, voxel
 * centres at full resolution (see the file comment).
 *
 * The grid must be resident and the pool must live on `device` (as for
 * `gridStats`). The tool frees only its own parameter, output and readback
 * buffers.
 */
export function vdbDecodeToAtlasGpu(device: GPUDevice, pool: VDBGridPool, gridId: number, options: VDBDecodeToAtlasOptions = {}): VDBAtlasReadback {
  checkAtlasOptions(options);
  const info = checkTarget(WHERE, device, pool, gridId);
  const maxDim = options.maxDim ?? 256;
  const filter = options.filter ?? "trilinear";
  const format = options.format ?? "uint8";
  const normalize = options.normalize ?? true;
  const range = options.range ? { min: options.range.min, max: options.range.max } : normalize ? { min: info.rootMin, max: info.rootMax } : { min: 0, max: 1 };
  const regionSize = bboxSize(info.indexBBox);
  const dims = atlasDims(info.indexBBox, maxDim);
  const cached = kernelPipeline(device, ATLAS, info.type);

  const params = createInitialized(
    device,
    "vdb decodeToAtlas params",
    USAGE.UNIFORM,
    new Uint8Array(packParams({ grid: gridId, filter: filter === "trilinear" ? 1 : 0, regionMin: info.indexBBox.min, regionSize, dims })),
  );
  const bytes = dims[0] * dims[1] * dims[2] * 4;
  const out = device.createBuffer({ label: "vdb decodeToAtlas", size: bytes, usage: USAGE.STORAGE | USAGE.COPY_SRC });
  const readback = options.readback ?? true;
  const staging = readback ? device.createBuffer({ label: "vdb decodeToAtlas readback", size: bytes, usage: USAGE.MAP_READ | USAGE.COPY_DST }) : null;

  const group0 = device.createBindGroup({ layout: cached.group0, entries: [pool.bindGroupEntry({ group: 0, firstBinding: 0 }), { binding: 1, resource: { buffer: params } }] });
  const group1 = device.createBindGroup({ layout: cached.group1, entries: [{ binding: 0, resource: { buffer: out } }] });

  const host = hostEncoder(device, options.encoder, "vdb decodeToAtlas");
  const pass = host.encoder.beginComputePass({ label: "vdb decodeToAtlas" });
  pass.setPipeline(cached.pipeline);
  pass.setBindGroup(0, group0);
  pass.setBindGroup(1, group1);
  pass.dispatchWorkgroups(...workgroups(dims));
  pass.end();
  if (staging) host.encoder.copyBufferToBuffer(out, 0, staging, 0, bytes);
  host.submit();

  const finish = (raw: ArrayBuffer): VDBAtlas => ({ data: atlasFromRaw(new Float32Array(raw), { format, normalize, range }), dims: [...dims], range });
  return new AtlasJob([params, out], staging, cached.error, finish, dims, range, out, host.selfSubmitted);
}

class AtlasJob extends ComputeJob<VDBAtlas> implements VDBAtlasReadback {
  constructor(
    owned: GPUBuffer[],
    staging: GPUBuffer | null,
    error: Promise<GPUError | null>,
    finish: (bytes: ArrayBuffer) => VDBAtlas,
    readonly dims: [number, number, number],
    readonly range: { min: number; max: number },
    readonly buffer: GPUBuffer,
    selfSubmitted: boolean,
  ) {
    super(WHERE, owned, staging, error, finish, selfSubmitted);
  }
}
