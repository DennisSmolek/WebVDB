/**
 * Shared plumbing of the core compute tools (K2.3): kernel assembly on a
 * slot-mode VDB module, per-device pipeline caching, and the readback handle.
 *
 * Each kernel is one shader: `vdbModuleWgsl({ slot: COMPUTE_SLOT, types:
 * [type], profile: 'sample' })` plus a host footer that calls only the public
 * `vdb*` API. Group 0 holds the pool (binding 0, `vdbModuleEntries`) and the
 * kernel's uniform parameters (binding 1); group 1 holds the kernel's
 * outputs. Default module options, so the WGSL size snapshot is unchanged.
 */
import { VDBError } from "../errors.js";
import type { VDBGridInfo, VDBGridPool } from "../pool/pool.js";
import type { VDBGridTypeName, VDBSlot } from "../types.js";
import { vdbModuleEntries, vdbModuleWgsl } from "../wgsl/generate.js";

/** Where the compute kernels bind the pool. */
export const COMPUTE_SLOT: VDBSlot = { group: 0, firstBinding: 0 };

/** Threads per workgroup per axis (4 x 4 x 4 = 64). */
export const WORKGROUP_AXIS = 4;

/** WebGPU flag values, restated so the core loads where the globals don't exist (Node). */
export const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
const STAGE_COMPUTE = 0x4;
const MAP_MODE_READ = 0x1;

/** Bytes of `CsParams` below. */
export const PARAMS_BYTES = 80;

/**
 * Uniform parameters shared by every kernel (`CsParams`, 80 B):
 * `grid, bins, filter, _` u32 at 0..15; `regionMin` vec3i at 16; `regionSize`
 * vec3u at 32; `dims` vec3u at 48; `range` vec2f at 64.
 */
export interface KernelParams {
  grid: number;
  bins?: number;
  filter?: number;
  regionMin: readonly [number, number, number];
  regionSize: readonly [number, number, number];
  dims?: readonly [number, number, number];
  range?: { min: number; max: number };
}

const PARAMS_WGSL = /* wgsl */ `
struct CsParams {
    grid: u32,
    bins: u32,
    filterMode: u32,
    pad0: u32,
    regionMin: vec3i,
    pad1: u32,
    regionSize: vec3u,
    pad2: u32,
    dims: vec3u,
    pad3: u32,
    range: vec2f,
    pad4: vec2f,
};
@group(0) @binding(1) var<uniform> csParams: CsParams;
`;

/** A kernel: its footer WGSL (after the module and `CsParams`) and its group-1 output bindings. */
export interface KernelSource {
  readonly entryPoint: string;
  readonly footer: string;
  readonly outputs: number;
}

interface CachedPipeline {
  pipeline: GPUComputePipeline;
  group0: GPUBindGroupLayout;
  group1: GPUBindGroupLayout;
  /** Validation result of the module + pipeline creation (null = valid). */
  error: Promise<GPUError | null>;
}

const pipelines = new WeakMap<GPUDevice, Map<string, CachedPipeline>>();

/** The full WGSL of `kernel` for grids of `type` (exported for tests and the size of the shader). */
export function kernelWgsl(kernel: KernelSource, type: VDBGridTypeName): string {
  return `${vdbModuleWgsl({ slot: COMPUTE_SLOT, types: [type], profile: "sample" })}\n${PARAMS_WGSL}\n${kernel.footer}`;
}

/** The compute pipeline of `kernel` for `type` on `device`, created once per device. */
export function kernelPipeline(device: GPUDevice, kernel: KernelSource, type: VDBGridTypeName): CachedPipeline {
  let byKey = pipelines.get(device);
  if (!byKey) pipelines.set(device, (byKey = new Map()));
  const key = `${kernel.entryPoint}:${type}`;
  let cached = byKey.get(key);
  if (cached) return cached;
  device.pushErrorScope("validation");
  const group0 = device.createBindGroupLayout({
    label: `vdb compute ${key} group 0`,
    entries: [
      ...vdbModuleEntries({ slot: COMPUTE_SLOT, visibility: STAGE_COMPUTE }),
      { binding: 1, visibility: STAGE_COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const group1 = device.createBindGroupLayout({
    label: `vdb compute ${key} group 1`,
    entries: Array.from({ length: kernel.outputs }, (_, binding) => ({
      binding,
      visibility: STAGE_COMPUTE,
      buffer: { type: "storage" as const },
    })),
  });
  const module = device.createShaderModule({ label: `vdb compute ${key}`, code: kernelWgsl(kernel, type) });
  const pipeline = device.createComputePipeline({
    label: `vdb compute ${key}`,
    layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1] }),
    compute: { module, entryPoint: kernel.entryPoint },
  });
  cached = { pipeline, group0, group1, error: device.popErrorScope() };
  byKey.set(key, cached);
  return cached;
}

function badOption(message: string): never {
  throw new VDBError("VDB_BAD_OPTION", message);
}

/** Checks `device` / `pool` / `gridId` and returns the grid's info. */
export function checkTarget(where: string, device: GPUDevice, pool: VDBGridPool, gridId: number): VDBGridInfo {
  if (!device || typeof (device as { createBuffer?: unknown }).createBuffer !== "function") {
    badOption(`${where}: expected a GPUDevice as the first argument (the device the pool lives on).`);
  }
  if (!pool || typeof (pool as { info?: unknown }).info !== "function") {
    badOption(`${where}: expected a VDBGridPool as the second argument.`);
  }
  if (pool.device !== device) {
    badOption(`${where}: the pool lives on a different GPUDevice. Pass pool.device (or create the pool on this device).`);
  }
  const info = pool.info(gridId);
  if (!info.resident) {
    badOption(
      `${where}: grid ${gridId} is not on the GPU yet. Call pool.update(encoder) (or pool.flush()) before recording compute work on it.`,
    );
  }
  const { min, max } = info.indexBBox;
  if (min.some((v, a) => v > max[a]!)) {
    badOption(`${where}: grid ${gridId} has an empty index bbox (${JSON.stringify(info.indexBBox)}); there are no voxels to process.`);
  }
  return info;
}

/** Checks a positive integer option. */
export function checkPositiveInt(where: string, name: string, value: number): void {
  if (!Number.isInteger(value) || value < 1) badOption(`${where}: ${name} must be an integer >= 1, got ${JSON.stringify(value)}.`);
}

/** Creates a buffer with initial contents (mapped at creation, no queue write). */
export function createInitialized(device: GPUDevice, label: string, usage: number, data: ArrayBufferView, size = data.byteLength): GPUBuffer {
  const buffer = device.createBuffer({ label, size: align4(Math.max(4, size)), usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

export function align4(n: number): number {
  return Math.ceil(n / 4) * 4;
}

/** The packed `CsParams` bytes. */
export function packParams(p: KernelParams): ArrayBuffer {
  const bytes = new ArrayBuffer(PARAMS_BYTES);
  const u = new Uint32Array(bytes);
  const i = new Int32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = p.grid;
  u[1] = p.bins ?? 0;
  u[2] = p.filter ?? 0;
  for (let a = 0; a < 3; a++) {
    i[4 + a] = p.regionMin[a]!;
    u[8 + a] = p.regionSize[a]!;
    u[12 + a] = p.dims?.[a] ?? 0;
  }
  f[16] = p.range?.min ?? 0;
  f[17] = p.range?.max ?? 0;
  return bytes;
}

/** Workgroup counts covering `size` threads per axis. */
export function workgroups(size: readonly [number, number, number]): [number, number, number] {
  return [Math.ceil(size[0] / WORKGROUP_AXIS), Math.ceil(size[1] / WORKGROUP_AXIS), Math.ceil(size[2] / WORKGROUP_AXIS)];
}

/**
 * Pending GPU work with a CPU result. Created by `gridStats` / `decodeToAtlas`
 * after their passes are recorded.
 *
 * Order of calls:
 * - **No `encoder` option** (the tool created and submitted its own): call
 *   {@link read} at any time.
 * - **Host `encoder`**: the work sits in the host's encoder until the host
 *   submits it. Submit, then call {@link submitted}, then {@link read}:
 *
 *       const job = gridStats(device, pool, id, { encoder });
 *       device.queue.submit([encoder.finish()]);
 *       const stats = await job.submitted().read();
 *
 *   `read()` before `submitted()` rejects with `VDB_BAD_OPTION` and touches
 *   nothing: mapping (or freeing) the tool's buffers before the submit would
 *   make that submit fail validation ("used in submit while destroyed" /
 *   "mapped").
 *
 * `read()` maps the readback, frees every buffer the tool allocated and
 * resolves with the result; keep it outside timed regions. Call
 * {@link dispose} instead to drop the work without reading it (after its
 * submit, or with the encoder discarded).
 */
export interface VDBReadback<T> {
  /**
   * Tells the job the host submitted the encoder it recorded into; returns
   * the job (`job.submitted().read()`). A no-op for work the tool submitted
   * itself.
   */
  submitted(): this;
  /**
   * Maps the result and frees the tool's buffers. With a host `encoder`,
   * call {@link submitted} first (else rejects with `VDB_BAD_OPTION`).
   */
  read(): Promise<T>;
  dispose(): void;
}

/** Owns the buffers of one tool call; frees exactly those. */
export class ComputeJob<T> implements VDBReadback<T> {
  readonly #owned: GPUBuffer[];
  readonly #staging: GPUBuffer | null;
  readonly #finish: (bytes: ArrayBuffer) => T;
  readonly #error: Promise<GPUError | null>;
  readonly #where: string;
  #result: Promise<T> | null = null;
  #disposed = false;
  /** False until the host says it submitted its encoder; true when the tool submitted its own. */
  #submitted: boolean;

  constructor(
    where: string,
    owned: GPUBuffer[],
    staging: GPUBuffer | null,
    error: Promise<GPUError | null>,
    finish: (bytes: ArrayBuffer) => T,
    selfSubmitted = true,
  ) {
    this.#where = where;
    this.#owned = owned;
    this.#staging = staging;
    this.#error = error;
    this.#finish = finish;
    this.#submitted = selfSubmitted;
  }

  submitted(): this {
    this.#submitted = true;
    return this;
  }

  read(): Promise<T> {
    if (this.#result) return this.#result;
    if (this.#disposed) return Promise.reject(new VDBError("VDB_DISPOSED", `${this.#where}: this result was disposed before read().`));
    if (!this.#submitted) {
      return Promise.reject(
        new VDBError(
          "VDB_BAD_OPTION",
          `${this.#where}: read() before the host submitted its encoder. The work was recorded into your encoder: ` +
            "device.queue.submit([encoder.finish()]), then job.submitted().read(). (Nothing was mapped or freed.)",
        ),
      );
    }
    this.#result = this.#read();
    return this.#result;
  }

  async #read(): Promise<T> {
    const staging = this.#staging;
    // Keeps the GPU output alive: the host frees it with dispose().
    if (!staging) throw new VDBError("VDB_BAD_OPTION", `${this.#where}: created with readback: false; nothing to read. Use the buffer, then dispose().`);
    try {
      const error = await this.#error;
      if (error) throw new VDBError("VDB_UNSUPPORTED_DEVICE", `${this.#where}: the compute pipeline failed validation: ${error.message}`);
      await staging.mapAsync(MAP_MODE_READ);
      const bytes = staging.getMappedRange().slice(0);
      staging.unmap();
      return this.#finish(bytes);
    } finally {
      this.dispose();
    }
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const b of this.#owned) b.destroy();
    this.#staging?.destroy();
  }
}

/** Creates an encoder when the host gave none; `submit` submits it only then. */
export function hostEncoder(
  device: GPUDevice,
  given: GPUCommandEncoder | undefined,
  label: string,
): { encoder: GPUCommandEncoder; submit(): void; readonly selfSubmitted: boolean } {
  if (given) return { encoder: given, submit() {}, selfSubmitted: false };
  const encoder = device.createCommandEncoder({ label });
  return { encoder, submit: () => device.queue.submit([encoder.finish()]), selfSubmitted: true };
}

export { MAP_MODE_READ };
