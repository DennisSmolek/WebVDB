/**
 * The environment input of the core frame renderer (K6.2b, design §2.3, §3.5, Q3): a host equirect or cube
 * `GPUTexture` reduced on the GPU (`envPrefilter`: the 128 × 64 work map and his 64 × 32 map; `envReduce`: lights and
 * SH) into the environment buffer every later kernel binds as a uniform (group 0 binding 3). No CPU readback: the first
 * frame already has its environment lights. Each stage reruns only when its inputs change (texture → work map,
 * rotation → map, settings and intensity → buffer).
 */

import { VDBError } from "../errors.js";
import { frameEnvironmentMapHeight, frameEnvironmentMapWidth, vdbFrameEnvironmentMode } from "../../tools/frame/environment.js";
import { VDB_FRAME_ENV_IDENTITY, VDB_FRAME_ENV_MODE, VDB_FRAME_ENV_WORK_HEIGHT, VDB_FRAME_ENV_WORK_WIDTH, vdbFrameEnvCubeSupersample } from "../../tools/frame/environment-gpu.js";
import { vdbSameValues } from "../../tools/frame/grid.js";
import { vdbFrameEnvBlurSpec, vdbFrameEnvPrefilterSpec, vdbFrameEnvReduceSpec } from "./kernels.js";
import { computePipeline } from "./pipelines.js";
import { BUFFER_USAGE, TEXTURE_USAGE } from "./resources.js";
import type { VDBFrameSettings } from "./settings.js";
import type { VDBFrameEnvironment } from "./types.js";
import { VDB_FRAME_ENV_PARAMS, VDB_FRAME_ENVIRONMENT, VDBUniformWriter } from "./uniforms.js";

/** The environment state an update reads. */
export interface VDBFrameEnvironmentInfo {
  /** An environment is given this update. */
  readonly present: boolean;
  /** Its lighting mode (settings.environmentLighting). */
  readonly mode: "outward" | "directional" | "spherical-harmonics";
  /** +1 whenever the environment buffer may hold other values (signatures of the grids that read it). */
  readonly version: number;
}

/** The projection of a host environment texture (design §2.3: 6 square layers → cube, else equirect). */
export function vdbFrameEnvironmentProjection(env: Pick<VDBFrameEnvironment, "texture" | "projection">): "equirect" | "cube" {
  if (env.projection) return env.projection;
  const t = env.texture;
  return t.depthOrArrayLayers === 6 && t.width === t.height ? "cube" : "equirect";
}

/** @internal The environment buffer, its parameters and maps, and the kernels that fill them. */
export class VDBFrameEnvironmentState implements VDBFrameEnvironmentInfo {
  readonly buffer: GPUBuffer;
  present = false;
  mode: VDBFrameEnvironmentInfo["mode"] = "outward";
  version = 0;
  /** Kernels recorded so far (tests, harness). */
  readonly runs = { prefilter: 0, blur: 0, reduce: 0 };

  readonly #device: GPUDevice;
  readonly #label: string;
  readonly #params: GPUBuffer;
  readonly #writer = new VDBUniformWriter(VDB_FRAME_ENV_PARAMS.size);
  #work: GPUTexture | null = null;
  #map: GPUTexture | null = null;
  #textureKey: unknown[] = [];
  #rotationKey: number[] = [];
  #reduceKey: unknown[] = [];
  #source: { texture: GPUTexture; projection: "equirect" | "cube" } | null = null;
  #pending = { prefilter: false, blur: false, reduce: false };
  #groups = new Map<string, { keys: readonly object[]; group: GPUBindGroup }>();

  constructor(device: GPUDevice, label: string) {
    this.#device = device;
    this.#label = label;
    // Zero-initialised: no environment (valid 0, count 0)
    this.buffer = device.createBuffer({ label: `${label} environment`, size: VDB_FRAME_ENVIRONMENT.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC });
    this.#params = device.createBuffer({ label: `${label} environment params`, size: VDB_FRAME_ENV_PARAMS.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
  }

  /** The work map and his map (harness probes); null before the first environment. */
  get maps(): { work: GPUTexture; map: GPUTexture } | null {
    return this.#work && this.#map ? { work: this.#work, map: this.#map } : null;
  }

  /** Reads this update's environment and settings; decides which kernels {@link record} runs. */
  prepare(env: VDBFrameEnvironment | null | undefined, s: Pick<VDBFrameSettings, "environmentLighting" | "environmentSamples" | "environmentShOrder" | "multipleScattering">): void {
    this.mode = vdbFrameEnvironmentMode(s);
    if (!env) {
      if (this.present || this.#textureKey.length > 0) {
        this.#device.queue.writeBuffer(this.buffer, 0, new ArrayBuffer(VDB_FRAME_ENVIRONMENT.size));
        this.version++;
      }
      this.present = false;
      this.#textureKey = [];
      this.#rotationKey = [];
      this.#reduceKey = [];
      this.#source = null;
      this.#pending = { prefilter: false, blur: false, reduce: false };
      return;
    }
    const texture = env.texture;
    if (!texture || typeof (texture as { createView?: unknown }).createView !== "function") {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRendererGpu.update: environment.texture must be a GPUTexture (TEXTURE_BINDING usage) on the renderer's device.");
    }
    const projection = vdbFrameEnvironmentProjection(env);
    if (projection === "cube" && (texture.depthOrArrayLayers < 6 || texture.width !== texture.height)) {
      throw new VDBError("VDB_BAD_OPTION", `VDBFrameRendererGpu.update: a cube environment needs 6 square layers (got ${texture.width}x${texture.height}x${texture.depthOrArrayLayers}).`);
    }
    this.present = true;
    const rotation = env.rotation ? Array.from(env.rotation).slice(0, 9) : [...VDB_FRAME_ENV_IDENTITY];
    const textureKey = [env.id, env.revision, texture, projection];
    const ms = s.multipleScattering;
    const modeCode = this.mode === "directional" ? VDB_FRAME_ENV_MODE.directional : this.mode === "spherical-harmonics" ? VDB_FRAME_ENV_MODE.sphericalHarmonics : VDB_FRAME_ENV_MODE.outward;
    const pairs = !!ms && ms !== "octaves";
    const intensity = Number.isFinite(env.intensity) ? env.intensity : 1;
    const reduceKey = [modeCode, s.environmentSamples, s.environmentShOrder, pairs, intensity];
    const prefilter = !vdbSameValues(textureKey, this.#textureKey);
    const blur = prefilter || !vdbSameValues(rotation, this.#rotationKey);
    const reduce = blur || !vdbSameValues(reduceKey, this.#reduceKey);
    this.#pending = { prefilter, blur, reduce };
    if (!reduce) return;
    this.#textureKey = textureKey;
    this.#rotationKey = rotation;
    this.#reduceKey = reduceKey;
    this.#source = { texture, projection };
    this.version++;
    const w = this.#writer;
    const P = VDB_FRAME_ENV_PARAMS;
    // the rows of R (column-major input): row r = (R[r], R[3 + r], R[6 + r])
    w.set(P, 0, "rotation", [0, 1, 2].flatMap((r) => [rotation[r]!, rotation[3 + r]!, rotation[6 + r]!, 0]));
    w.set(P, 0, "source", [texture.width, texture.height, projection === "cube" ? 1 : 0, projection === "cube" ? vdbFrameEnvCubeSupersample(texture.width) : 1]);
    w.set(P, 0, "settings", [modeCode, s.environmentSamples, s.environmentShOrder, pairs ? 1 : 0]);
    w.set(P, 0, "scale", [intensity, 0, 0, 0]);
    this.#device.queue.writeBuffer(this.#params, 0, w.buffer);
    if (!this.#work) {
      const usage = TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_SRC;
      this.#work = this.#device.createTexture({ label: `${this.#label} environment work map`, size: [VDB_FRAME_ENV_WORK_WIDTH, VDB_FRAME_ENV_WORK_HEIGHT], format: "rgba32float", usage });
      this.#map = this.#device.createTexture({ label: `${this.#label} environment map`, size: [frameEnvironmentMapWidth, frameEnvironmentMapHeight], format: "rgba32float", usage });
    }
  }

  #cached(name: string, keys: readonly object[], make: () => GPUBindGroup): GPUBindGroup {
    const hit = this.#groups.get(name);
    if (hit && hit.keys.length === keys.length && hit.keys.every((k, i) => k === keys[i])) return hit.group;
    const group = make();
    this.#groups.set(name, { keys, group });
    return group;
  }

  /** Records the kernels {@link prepare} asked for into `pass` (before the kernels that read the buffer). */
  record(pass: GPUComputePassEncoder): void {
    const p = this.#pending;
    const src = this.#source;
    if (!src || !this.#work || !this.#map || !(p.prefilter || p.blur || p.reduce)) return;
    const device = this.#device;
    const work = this.#work;
    const map = this.#map;
    const params = (layout: GPUBindGroupLayout) => this.#cached("params", [layout, this.#params], () => device.createBindGroup({ label: `${this.#label} environment group 0`, layout, entries: [{ binding: 0, resource: { buffer: this.#params } }] }));
    if (p.prefilter) {
      const built = computePipeline(device, vdbFrameEnvPrefilterSpec(src.projection));
      const view = src.texture.createView(src.projection === "cube" ? { dimension: "2d-array", baseArrayLayer: 0, arrayLayerCount: 6, baseMipLevel: 0, mipLevelCount: 1 } : { dimension: "2d", baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: 1 });
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, params(built.layouts[0]!));
      pass.setBindGroup(1, device.createBindGroup({ label: `${this.#label} environment prefilter group 1`, layout: built.layouts[1]!, entries: [{ binding: 0, resource: view }, { binding: 1, resource: work.createView() }] }));
      pass.dispatchWorkgroups(Math.ceil(VDB_FRAME_ENV_WORK_WIDTH / 8), Math.ceil(VDB_FRAME_ENV_WORK_HEIGHT / 8), 1);
      this.runs.prefilter++;
    }
    if (p.blur) {
      const built = computePipeline(device, vdbFrameEnvBlurSpec());
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, params(built.layouts[0]!));
      pass.setBindGroup(1, this.#cached("blur", [built.layouts[1]!, work, map], () => device.createBindGroup({ label: `${this.#label} environment blur group 1`, layout: built.layouts[1]!, entries: [{ binding: 0, resource: work.createView() }, { binding: 1, resource: map.createView() }] })));
      pass.dispatchWorkgroups(Math.ceil(frameEnvironmentMapWidth / 8), Math.ceil(frameEnvironmentMapHeight / 8), 1);
      this.runs.blur++;
    }
    if (p.reduce) {
      const built = computePipeline(device, vdbFrameEnvReduceSpec());
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, params(built.layouts[0]!));
      pass.setBindGroup(
        1,
        this.#cached("reduce", [built.layouts[1]!, work, map, this.buffer], () =>
          device.createBindGroup({
            label: `${this.#label} environment reduce group 1`,
            layout: built.layouts[1]!,
            entries: [
              { binding: 0, resource: work.createView() },
              { binding: 1, resource: map.createView() },
              { binding: 2, resource: { buffer: this.buffer } },
            ],
          }),
        ),
      );
      pass.dispatchWorkgroups(1, 1, 1);
      this.runs.reduce++;
    }
    this.#pending = { prefilter: false, blur: false, reduce: false };
  }

  /** Builds the environment pipelines (`prepare()`). */
  specs(projections: readonly ("equirect" | "cube")[]) {
    return [...projections.map((p) => vdbFrameEnvPrefilterSpec(p)), vdbFrameEnvBlurSpec(), vdbFrameEnvReduceSpec()];
  }

  dispose(): void {
    this.buffer.destroy();
    this.#params.destroy();
    this.#work?.destroy();
    this.#map?.destroy();
    this.#work = null;
    this.#map = null;
    this.#groups.clear();
  }
}
