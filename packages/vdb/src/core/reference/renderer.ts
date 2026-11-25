/**
 * K6.R: the reference renderer on the host's device (design K6-core-renderer.md §7.4). A brute-force volumetric path
 * tracer over one volume source (dense textures or the sparse pool: the group-2 `vdbSrc*` contract), accumulated
 * progressively in rgba32float textures with per-pixel variance. Engine-free, internal only (no public export). The
 * CPU twin is `tools/reference/trace.ts` (same stream, same draw order).
 *
 *   const ref = await vdbCreateReferenceRendererGpu(device, { source, width, height });
 *   ref.setScene(scene, settings);                 // resets the accumulation
 *   const enc = device.createCommandEncoder();
 *   pool?.update(enc); source.update?.(enc);       // host order (Traps): the pool, the source, then the kernel
 *   ref.render(enc, 4);                            // 4 batches of settings.spp samples per pixel
 *   device.queue.submit([enc.finish()]);
 *   const accum = await ref.readback();            // sums of batch means → vdbReferenceStatistics (tools)
 *
 * One `render()` per submitted encoder (its batch records are written with `queue.writeBuffer`). The renderer frees
 * only what it made (textures, buffers); the source is the host's.
 */

import { VDBError } from "../errors.js";
import type { VDBSlot } from "../types.js";
import type { VDBVolumeSource } from "../source/types.js";
import {
  VDB_REFERENCE_MAX_HEMISPHERES,
  VDB_REFERENCE_MAX_LIGHTS,
  VDB_REFERENCE_MAX_STEPS,
  VDB_REFERENCE_MAX_VOLUMES,
  VDB_REFERENCE_RATIO_RR,
  vdbRefInvert4,
  vdbReferenceSettings,
  type VDBReferenceAccum,
  type VDBReferenceScene,
  type VDBReferenceSettings,
} from "../../tools/reference/index.js";
import {
  VDB_REF_BATCH_STRIDE,
  VDB_REF_FLAG_BOX_FILTER,
  VDB_REF_FLAG_EQUIRECT,
  VDB_REF_FLAG_ORTHOGRAPHIC,
  VDB_REF_HEMI_BYTES,
  VDB_REF_LIGHT_BYTES,
  VDB_REF_UNIFORM_OFFSETS as OFF,
  VDB_REF_VOLUME_BYTES,
  vdbReferenceKernelWgsl,
} from "./wgsl.js";

/** `GPUShaderStage.COMPUTE`, `GPUBufferUsage` and `GPUTextureUsage` bits (the globals do not exist in Node). */
const STAGE_COMPUTE = 0x4;
const BUFFER_UNIFORM = 0x40;
const BUFFER_COPY_DST = 0x8;
const BUFFER_MAP_READ = 0x1;
const TEX_COPY_SRC = 0x1;
const TEX_COPY_DST = 0x2;
const TEX_BINDING = 0x4;
const TEX_STORAGE = 0x8;
const MAP_READ = 0x1;

/** The source slot of the kernel (the K6 group-2 contract). */
export const VDB_REFERENCE_SOURCE_SLOT: VDBSlot = Object.freeze({ group: 2, firstBinding: 0 });
/** Most batches one `render()` records. */
export const VDB_REFERENCE_MAX_BATCHES_PER_RENDER = 64;

/** A source the kernel can read: the group-2 contract, plus the sparse source's module and update. */
export type VDBReferenceSourceGpu = VDBVolumeSource & {
  moduleWgsl?(slot: VDBSlot): string;
};

/** Options of {@link vdbCreateReferenceRendererGpu}. */
export interface VDBReferenceRendererGpuOptions {
  /** The source every volume of the scene reads (its slots: `VDBReferenceVolume.v`). One source per kernel (D10). */
  source: VDBReferenceSourceGpu;
  width: number;
  height: number;
  label?: string;
  /** Tests: patch the WGSL before it compiles (the fail-before mutations, `VDB_REFERENCE_MUTATIONS`). */
  wgslPatch?: (wgsl: string) => string;
}

/** The reference renderer (internal). */
export interface VDBReferenceRendererGpu {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  /** The composed WGSL (module + source + kernel). */
  readonly wgsl: string;
  /** Batches accumulated since the last {@link setScene} / {@link reset}. */
  readonly batches: number;
  /** Set the scene and settings (packs the uniforms, uploads an equirect map) and reset the accumulation. */
  setScene(scene: VDBReferenceScene, settings: Omit<VDBReferenceSettings, "width" | "height">): void;
  /** Drop the accumulation (the next batch starts from 0; batch indices restart, so the samples repeat). */
  reset(): void;
  /** Record `batches` dispatches (each `settings.spp` samples per pixel). One `render()` per submitted encoder. */
  render(encoder: GPUCommandEncoder, batches?: number): void;
  /** Read the accumulation back (after the encoder was submitted): sums of batch means and of their squares. */
  readback(): Promise<VDBReferenceAccum>;
  dispose(): void;
}

/** Pack `VdbRefUniforms` (layout in `wgsl.ts`). Exported for tests. */
export function vdbReferencePackUniforms(scene: VDBReferenceScene, settingsIn: VDBReferenceSettings): ArrayBuffer {
  const s = vdbReferenceSettings(settingsIn);
  const buf = new ArrayBuffer(OFF.size_);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const F = (byte: number, values: ArrayLike<number>) => f.set(Array.from(values as ArrayLike<number>), byte / 4);
  const U = (byte: number, value: number) => (u[byte / 4] = value >>> 0);
  const volumes = scene.volumes;
  const lights = scene.lights ?? [];
  const env = scene.environment ?? {};
  const hemis = env.hemispheres ?? [];
  if (volumes.length > VDB_REFERENCE_MAX_VOLUMES) throw new VDBError("VDB_BAD_OPTION", `reference renderer: at most ${VDB_REFERENCE_MAX_VOLUMES} volumes, got ${volumes.length}`);
  if (lights.length > VDB_REFERENCE_MAX_LIGHTS) throw new VDBError("VDB_BAD_OPTION", `reference renderer: at most ${VDB_REFERENCE_MAX_LIGHTS} lights, got ${lights.length}`);
  if (hemis.length > VDB_REFERENCE_MAX_HEMISPHERES) throw new VDBError("VDB_BAD_OPTION", `reference renderer: at most ${VDB_REFERENCE_MAX_HEMISPHERES} hemispheres, got ${hemis.length}`);
  F(OFF.cameraToWorld, scene.camera.cameraToWorld);
  F(OFF.projectionInverse, scene.camera.projectionInverse);
  U(OFF.size, s.width);
  U(OFF.size + 4, s.height);
  U(OFF.seed, s.seed);
  U(OFF.spp, s.spp);
  U(OFF.maxBounces, s.maxBounces);
  U(OFF.rrStart, s.rrStart);
  U(OFF.volumeCount, volumes.length);
  U(OFF.lightCount, lights.length);
  U(OFF.hemiCount, hemis.length);
  const eq = env.equirect;
  U(OFF.flags, (s.pixelFilter === "box" ? VDB_REF_FLAG_BOX_FILTER : 0) | (scene.camera.orthographic ? VDB_REF_FLAG_ORTHOGRAPHIC : 0) | (eq ? VDB_REF_FLAG_EQUIRECT : 0));
  U(OFF.envSize, eq?.width ?? 1);
  U(OFF.envSize + 4, eq?.height ?? 1);
  F(OFF.uniformEnv, [...(env.uniform ?? [0, 0, 0]), eq?.intensity ?? 1]);
  const r = eq?.rotation ?? [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let row = 0; row < 3; row++) F(OFF.envRotation + row * 16, [r[row * 3]!, r[row * 3 + 1]!, r[row * 3 + 2]!, 0]);
  hemis.forEach((h, i) => {
    const o = OFF.hemis + i * VDB_REF_HEMI_BYTES;
    const l = Math.hypot(...h.up);
    F(o, [...h.sky, 0]);
    F(o + 16, [...h.ground, 0]);
    F(o + 32, [h.up[0] / l, h.up[1] / l, h.up[2] / l, 0]);
  });
  volumes.forEach((v, i) => {
    const o = OFF.volumes + i * VDB_REF_VOLUME_BYTES;
    const e = v.emission ?? {};
    const c = e.color ?? [1, 1, 1];
    const k = e.intensity ?? 1;
    const range = e.temperatureRange ?? [0, 6500];
    F(o, vdbRefInvert4(v.localToWorld));
    F(o + 64, [v.sigma, v.sigma * (v.majorant ?? 1), v.cutoff ?? 0, v.shape === "sphere" ? 1 : 0]);
    F(o + 80, [...(v.albedo ?? [1, 1, 1]), v.emits ? 1 : 0]);
    F(o + 96, [v.phase?.g ?? 0, v.phase?.backG ?? 0, v.phase?.backWeight ?? 0, 0]);
    F(o + 112, [c[0] * k, c[1] * k, c[2] * k, 0]);
    F(o + 128, [range[0], range[1], 0, 0]);
    U(o + 144, v.v ?? 0);
  });
  lights.forEach((l, i) => {
    const o = OFF.lights + i * VDB_REF_LIGHT_BYTES;
    const norm = (d: readonly number[]) => {
      const n = Math.hypot(d[0]!, d[1]!, d[2]!);
      return [d[0]! / n, d[1]! / n, d[2]! / n];
    };
    if (l.kind === "directional") {
      F(o, [0, 0, 0, 0]);
      F(o + 16, [...norm(l.direction), 0]);
      F(o + 32, [...l.irradiance, 0]);
      F(o + 48, [0, 0, 0, 0]);
    } else if (l.kind === "point") {
      F(o, [...l.position, 1]);
      F(o + 16, [0, 0, 0, 0]);
      F(o + 32, [...l.intensity, 0]);
      F(o + 48, [l.inverseSquare ? 1 : 0, 0, 0, 0]);
    } else {
      F(o, [...l.position, 2]);
      F(o + 16, [...norm(l.direction), l.cosOuter]);
      F(o + 32, [...l.intensity, l.cosInner]);
      F(o + 48, [l.inverseSquare ? 1 : 0, 0, 0, 0]);
    }
  });
  return buf;
}

/** The composed WGSL of a reference kernel over `source` (module + source + kernel). */
export function vdbReferenceWgsl(source: VDBReferenceSourceGpu): string {
  const slot = VDB_REFERENCE_SOURCE_SLOT;
  const module = typeof source.moduleWgsl === "function" ? source.moduleWgsl(slot) : "";
  const kernel = vdbReferenceKernelWgsl({
    maxVolumes: VDB_REFERENCE_MAX_VOLUMES,
    maxLights: VDB_REFERENCE_MAX_LIGHTS,
    maxHemispheres: VDB_REFERENCE_MAX_HEMISPHERES,
    maxSteps: VDB_REFERENCE_MAX_STEPS,
    ratioRr: VDB_REFERENCE_RATIO_RR,
  });
  return `${module}\n${source.wgsl({ slot, profile: "sample" })}\n${kernel}`;
}

class ReferenceRenderer implements VDBReferenceRendererGpu {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  readonly wgsl: string;
  batches = 0;
  readonly #source: VDBReferenceSourceGpu;
  readonly #label: string;
  readonly #pipeline: GPUComputePipeline;
  readonly #layouts: GPUBindGroupLayout[];
  readonly #uniforms: GPUBuffer;
  readonly #batch: GPUBuffer;
  readonly #accum: GPUTexture[];
  #env: GPUTexture;
  #group0: GPUBindGroup;
  #group1: GPUBindGroup[] = [];
  #group2: GPUBindGroup | null = null;
  #sourceVersion = -1;
  #front = 0;
  #settings: Required<VDBReferenceSettings> | null = null;
  #disposed = false;

  constructor(device: GPUDevice, o: VDBReferenceRendererGpuOptions, wgsl: string, pipeline: GPUComputePipeline, layouts: GPUBindGroupLayout[]) {
    this.device = device;
    this.width = o.width;
    this.height = o.height;
    this.wgsl = wgsl;
    this.#source = o.source;
    this.#label = o.label ?? "vdb reference";
    this.#pipeline = pipeline;
    this.#layouts = layouts;
    this.#uniforms = device.createBuffer({ label: `${this.#label} uniforms`, size: OFF.size_, usage: BUFFER_UNIFORM | BUFFER_COPY_DST });
    this.#batch = device.createBuffer({ label: `${this.#label} batches`, size: VDB_REF_BATCH_STRIDE * VDB_REFERENCE_MAX_BATCHES_PER_RENDER, usage: BUFFER_UNIFORM | BUFFER_COPY_DST });
    // Two (sum, sum of squares) pairs, ping-ponged: 0,1 = pair A; 2,3 = pair B.
    this.#accum = [0, 1, 2, 3].map((i) =>
      device.createTexture({ label: `${this.#label} accum ${i}`, size: [o.width, o.height], format: "rgba32float", usage: TEX_BINDING | TEX_STORAGE | TEX_COPY_SRC }),
    );
    this.#env = this.#makeEnv(1, 1, new Float32Array(4));
    this.#group0 = device.createBindGroup({
      label: `${this.#label} group 0`,
      layout: layouts[0]!,
      entries: [
        { binding: 0, resource: { buffer: this.#uniforms } },
        { binding: 1, resource: { buffer: this.#batch, size: 16 } },
      ],
    });
    this.#buildGroup1();
  }

  #makeEnv(w: number, h: number, data: Float32Array): GPUTexture {
    const t = this.device.createTexture({ label: `${this.#label} environment`, size: [w, h], format: "rgba32float", usage: TEX_BINDING | TEX_COPY_DST });
    this.device.queue.writeTexture({ texture: t }, data as Float32Array<ArrayBuffer>, { bytesPerRow: w * 16, rowsPerImage: h }, [w, h]);
    return t;
  }

  #buildGroup1(): void {
    const view = (t: GPUTexture) => t.createView();
    const a = this.#accum;
    // Group 1 reading pair p and writing the other one.
    this.#group1 = [0, 1].map((p) =>
      this.device.createBindGroup({
        label: `${this.#label} group 1 (${p ? "B→A" : "A→B"})`,
        layout: this.#layouts[1]!,
        entries: [
          { binding: 0, resource: view(a[2 * p]!) },
          { binding: 1, resource: view(a[2 * p + 1]!) },
          { binding: 2, resource: view(a[2 * (1 - p)]!) },
          { binding: 3, resource: view(a[2 * (1 - p) + 1]!) },
          { binding: 4, resource: view(this.#env) },
        ],
      }),
    );
  }

  #check(): void {
    if (this.#disposed) throw new VDBError("VDB_DISPOSED", "reference renderer: disposed");
  }

  setScene(scene: VDBReferenceScene, settings: Omit<VDBReferenceSettings, "width" | "height">): void {
    this.#check();
    const full = { ...settings, width: this.width, height: this.height };
    this.#settings = vdbReferenceSettings(full);
    this.device.queue.writeBuffer(this.#uniforms, 0, vdbReferencePackUniforms(scene, full));
    const eq = scene.environment?.equirect;
    if (eq) {
      if (eq.data.length !== eq.width * eq.height * 4) throw new VDBError("VDB_BAD_OPTION", `reference renderer: the equirect map has ${eq.data.length} floats, expected ${eq.width}x${eq.height}x4`);
      this.#env.destroy();
      this.#env = this.#makeEnv(eq.width, eq.height, eq.data);
      this.#buildGroup1();
    }
    this.reset();
  }

  reset(): void {
    this.batches = 0;
    this.#front = 0;
  }

  render(encoder: GPUCommandEncoder, batches = 1): void {
    this.#check();
    if (!this.#settings) throw new VDBError("VDB_BAD_OPTION", "reference renderer: call setScene() before render()");
    if (!Number.isInteger(batches) || batches < 1 || batches > VDB_REFERENCE_MAX_BATCHES_PER_RENDER) {
      throw new VDBError("VDB_BAD_OPTION", `reference renderer: batches must be an integer in [1, ${VDB_REFERENCE_MAX_BATCHES_PER_RENDER}], got ${batches}`);
    }
    if (!this.#group2 || this.#source.version !== this.#sourceVersion) {
      this.#group2 = this.device.createBindGroup({ label: `${this.#label} group 2 (source)`, layout: this.#layouts[2]!, entries: this.#source.bindEntries(VDB_REFERENCE_SOURCE_SLOT) });
      this.#sourceVersion = this.#source.version;
    }
    const records = new Uint32Array((VDB_REF_BATCH_STRIDE / 4) * batches);
    for (let b = 0; b < batches; b++) {
      records[(b * VDB_REF_BATCH_STRIDE) / 4] = this.batches + b;
      records[(b * VDB_REF_BATCH_STRIDE) / 4 + 1] = this.batches + b > 0 ? 1 : 0;
    }
    this.device.queue.writeBuffer(this.#batch, 0, records);
    const pass = encoder.beginComputePass({ label: `${this.#label} trace` });
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(2, this.#group2);
    for (let b = 0; b < batches; b++) {
      pass.setBindGroup(0, this.#group0, [b * VDB_REF_BATCH_STRIDE]);
      pass.setBindGroup(1, this.#group1[this.#front]!);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
      this.#front = 1 - this.#front;
    }
    pass.end();
    this.batches += batches;
  }

  async readback(): Promise<VDBReferenceAccum> {
    this.#check();
    const rowBytes = Math.ceil((this.width * 16) / 256) * 256;
    const size = rowBytes * this.height;
    const staging = [0, 1].map((i) => this.device.createBuffer({ label: `${this.#label} readback ${i}`, size, usage: BUFFER_MAP_READ | BUFFER_COPY_DST }));
    const encoder = this.device.createCommandEncoder({ label: `${this.#label} readback` });
    for (let i = 0; i < 2; i++) {
      encoder.copyTextureToBuffer({ texture: this.#accum[2 * this.#front + i]! }, { buffer: staging[i]!, bytesPerRow: rowBytes, rowsPerImage: this.height }, [this.width, this.height]);
    }
    this.device.queue.submit([encoder.finish()]);
    const out = await Promise.all(
      staging.map(async (b) => {
        await b.mapAsync(MAP_READ);
        const raw = new Float32Array(b.getMappedRange().slice(0));
        b.destroy();
        const packed = new Float32Array(this.width * this.height * 4);
        for (let y = 0; y < this.height; y++) packed.set(raw.subarray((y * rowBytes) / 4, (y * rowBytes) / 4 + this.width * 4), y * this.width * 4);
        return packed;
      }),
    );
    return { width: this.width, height: this.height, batches: this.batches, sum: out[0]!, sumSq: out[1]! };
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const t of this.#accum) t.destroy();
    this.#env.destroy();
    this.#uniforms.destroy();
    this.#batch.destroy();
  }
}

/**
 * Create the reference renderer on the host's device: compiles the kernel for `options.source` (under a validation
 * error scope; WGSL errors throw `VDBError('VDB_BAD_OPTION')` with the compiler messages).
 */
export async function vdbCreateReferenceRendererGpu(device: GPUDevice, o: VDBReferenceRendererGpuOptions): Promise<VDBReferenceRendererGpu> {
  if (!device || typeof (device as { createComputePipelineAsync?: unknown }).createComputePipelineAsync !== "function") {
    throw new VDBError("VDB_UNSUPPORTED_DEVICE", "vdbCreateReferenceRendererGpu: pass the host's GPUDevice.");
  }
  if (!(o.width >= 1 && o.height >= 1 && Number.isInteger(o.width) && Number.isInteger(o.height))) {
    throw new VDBError("VDB_BAD_OPTION", `vdbCreateReferenceRendererGpu: width and height must be positive integers, got ${o.width}x${o.height}`);
  }
  let wgsl = vdbReferenceWgsl(o.source);
  if (o.wgslPatch) wgsl = o.wgslPatch(wgsl);
  const label = o.label ?? "vdb reference";
  device.pushErrorScope("validation");
  const layouts = [
    device.createBindGroupLayout({
      label: `${label} layout 0`,
      entries: [
        { binding: 0, visibility: STAGE_COMPUTE, buffer: { type: "uniform", minBindingSize: OFF.size_ } },
        { binding: 1, visibility: STAGE_COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 } },
      ],
    }),
    device.createBindGroupLayout({
      label: `${label} layout 1`,
      entries: [
        { binding: 0, visibility: STAGE_COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
        { binding: 1, visibility: STAGE_COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
        { binding: 2, visibility: STAGE_COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
        { binding: 3, visibility: STAGE_COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
        { binding: 4, visibility: STAGE_COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      ],
    }),
    device.createBindGroupLayout({ label: `${label} layout 2 (source)`, entries: o.source.layoutEntries(VDB_REFERENCE_SOURCE_SLOT, STAGE_COMPUTE) }),
  ];
  const module = device.createShaderModule({ label: `${label} kernel`, code: wgsl });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) {
    await device.popErrorScope();
    throw new VDBError("VDB_BAD_OPTION", `reference renderer WGSL: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("\n")}`);
  }
  const pipeline = await device.createComputePipelineAsync({
    label: `${label} pipeline`,
    layout: device.createPipelineLayout({ label: `${label} pipeline layout`, bindGroupLayouts: layouts }),
    compute: { module, entryPoint: "vdbRefTraceCs" },
  });
  const err = await device.popErrorScope();
  if (err) throw new VDBError("VDB_BAD_OPTION", `reference renderer pipeline: ${err.message}`);
  return new ReferenceRenderer(device, o, wgsl, pipeline, layouts);
}
