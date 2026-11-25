// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L715–1061 (updateTargets, renderFrame, updateFroxelArea,
//   renderTemporal, renderInject: the per-frame order, jitter, temporal validity, previous matrices),
//   L1476–1716 (light records and their slot order: directional, point, spot), frameShaderChunks.js L800–830
//   (setVolumeUniforms).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: an engine-free renderer on the host's device and encoder (D8, D34): plain-data inputs, compute
// kernels on 3D textures, explicit bind groups, a retire queue (design §2–§4). Divergences: DIVERGES.md
// (K6.1 rows).

/**
 * `vdbCreateFrameRenderer(device, options)`: the core froxel renderer (K6.1: 'volume' lighting on dense
 * sources). Per frame:
 *
 *   const enc = device.createCommandEncoder();
 *   const u = frame.update(enc, { frameId, camera, viewport, lights });   // gather per source group, resolve
 *   frame.render(enc, { color, size, depth, sceneColor });               // or vdbFrameIntegrate in your pass
 *   device.queue.submit([enc.finish()]);
 *
 * The renderer records, never submits; it frees only what it made (D8). One `update()` per submitted
 * encoder (its uniforms go through `queue.writeBuffer`).
 */

import { VDBError } from "../errors.js";
import type { VDBVolumeSlot, VDBVolumeSource } from "../source/types.js";
import {
  vdbFrameAtlasLayout,
  vdbFrameClipCamera,
  vdbFrameClipRect,
  vdbFrameClipVolume,
  vdbFrameFroxelArea,
  vdbFrameFroxelRect,
  vdbFrameFroxelSize,
  vdbFrameVolumeSlices,
  vdbFroxelAreaUnion,
  vdbFroxelHeldSize,
  vdbMat4Invert,
  vdbMat4MaxScale,
  type VDBFroxelArea,
  type VDBFroxelRect,
} from "../../tools/frame/view.js";
import { GATHER_BINDINGS } from "./wgsl/gather.js";
import { RESOLVE_BINDINGS } from "./wgsl/resolve.js";
import { vdbFrameCompositeSpec, vdbFrameGatherSpec, vdbFrameResolveSpec, SOURCE_SLOT, type VDBFrameCompositeKey } from "./kernels.js";
import { computePipeline, prepareCompute, prepareRender, renderPipeline } from "./pipelines.js";
import { BUFFER_USAGE, createFroxelTexture, VDBRetireQueue } from "./resources.js";
import { vdbFrameDefaultSettings, type VDBFrameSettings } from "./settings.js";
import { vdbFrameMultipleScatteringValue, vdbFramePhaseSpectrum, vdbFramePhaseVector, type VDBFramePhaseFields } from "./phase.js";
import {
  VDB_FRAME_GROUP,
  VDB_FRAME_GROUP_STRIDE,
  VDB_FRAME_LIGHTS,
  VDB_FRAME_MAX_HEMISPHERE_LIGHTS,
  VDB_FRAME_MAX_LIGHTS,
  VDB_FRAME_MAX_VOLUMES,
  VDB_FRAME_UNIFORMS,
  VDB_FRAME_VOLUME,
  VDB_FRAME_VOLUME_BYTES,
  VDBUniformWriter,
} from "./uniforms.js";
import {
  vdbFrameDefaultVolumeParams,
  type VDBFrameInput,
  type VDBFrameLight,
  type VDBFrameOutputs,
  type VDBFrameRedrawStats,
  type VDBFrameTargets,
  type VDBFrameUpdate,
  type VDBFrameVolumeGpu,
  type VDBFrameVolumeParams,
  type VDBFroxelLayout,
} from "./types.js";

/** Options of {@link vdbCreateFrameRenderer}. */
export interface VDBFrameRendererGpuOptions {
  /** Initial settings (his names and v0.26 defaults, plus the K6 fields). */
  settings?: Partial<VDBFrameSettings>;
  /** The core composite pass (`render()`): the host's colour and depth formats; null: no `render()`. */
  composite?: { colorFormat: GPUTextureFormat; depthFormat: GPUTextureFormat } | null;
  /** Optional device features (K6.1 uses none; fast paths are deferred, design §5.4). */
  features?: { tier1?: boolean; f16?: boolean; subgroups?: boolean };
  /** Label prefix of the GPU objects. */
  label?: string;
}

/** The core frame renderer (design §2.2). */
export interface VDBFrameRendererGpu {
  readonly device: GPUDevice;
  /** Mutable; read at the next `update()`. */
  readonly settings: VDBFrameSettings;
  readonly outputs: VDBFrameOutputs;
  readonly redrawStats: VDBFrameRedrawStats;
  addVolume(slot: VDBVolumeSlot, params?: Partial<VDBFrameVolumeParams>): VDBFrameVolumeGpu;
  /** Builds every pipeline the current volumes and settings reach, with the async creators. */
  prepare(options?: { lighting?: ("volume" | "frame")[] }): Promise<void>;
  update(encoder: GPUCommandEncoder, frame: VDBFrameInput, options?: { froxels?: boolean }): VDBFrameUpdate;
  render(encoder: GPUCommandEncoder, targets: VDBFrameTargets): void;
  dispose(): void;
}

const DEPTH_FORMATS = new Set<GPUTextureFormat>(["depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8"]);
const WRAP_CODE = { clamp: 0, repeat: 1, mirror: 2 } as const;

class FrameVolume implements VDBFrameVolumeGpu {
  params: VDBFrameVolumeParams;
  visible = true;
  castsShadow = true;
  readonly slot: VDBVolumeSlot;
  readonly #owner: FrameRenderer;

  constructor(owner: FrameRenderer, slot: VDBVolumeSlot, params: Partial<VDBFrameVolumeParams>) {
    this.#owner = owner;
    this.slot = slot;
    this.params = { ...vdbFrameDefaultVolumeParams(), ...params };
  }

  remove(): void {
    this.#owner.removeVolume(this);
  }
}

/** One visible volume after the CPU work of an update. */
interface VolumeRecord {
  volume: FrameVolume;
  localToView: Float64Array;
  viewToLocal: Float64Array;
  worldToLocal: Float64Array;
  rect: VDBFroxelRect;
  first: number;
  count: number;
}

interface Textures {
  size: [number, number, number];
  scatter: GPUTexture;
  integrated: GPUTexture;
  history: [GPUTexture, GPUTexture] | null;
  partials: GPUTexture | null;
}

class FrameRenderer implements VDBFrameRendererGpu {
  readonly device: GPUDevice;
  readonly settings: VDBFrameSettings;
  readonly redrawStats: VDBFrameRedrawStats = {
    shadowDensity: { drawn: 0, skipped: 0 },
    transmittance: { drawn: 0, skipped: 0 },
    environmentGrid: { drawn: 0, skipped: 0 },
  };
  readonly outputs: { version: number; froxels: { integrated: GPUTexture; integratedAlpha: GPUTexture | null; layout: VDBFroxelLayout } | null; uniforms: GPUBuffer };

  readonly #label: string;
  readonly #composite: { colorFormat: GPUTextureFormat; depthFormat: GPUTextureFormat } | null;
  readonly #maxVolumes: number;
  readonly #retire = new VDBRetireQueue();
  readonly #volumes: FrameVolume[] = [];
  readonly #warned = new Set<string>();
  readonly #held = vdbFroxelHeldSize();
  readonly #frameWriter = new VDBUniformWriter(VDB_FRAME_UNIFORMS.size);
  readonly #lightsWriter = new VDBUniformWriter(VDB_FRAME_LIGHTS.size);
  readonly #volumesWriter: VDBUniformWriter;
  readonly #frameBuffer: GPUBuffer;
  readonly #lightsBuffer: GPUBuffer;
  readonly #volumesBuffer: GPUBuffer;
  #groupsBuffer: GPUBuffer;
  #groupsWriter: VDBUniformWriter;
  readonly #linear: GPUSampler;
  #textures: Textures | null = null;
  /** The rect each texture last held non-zero values in (the dispatch box also clears it). */
  readonly #written = new WeakMap<GPUTexture, VDBFroxelArea>();
  #historyIndex = 0;
  #historyValid = false;
  #previous: { view: Float64Array; projection: Float64Array; range: [number, number]; cameraId: unknown; frameId: number } | null = null;
  #group0: GPUBindGroup | null = null;
  readonly #sourceGroups = new WeakMap<VDBVolumeSource, { version: number; layout: GPUBindGroupLayout; group: GPUBindGroup }>();
  #recorded = false;
  #disposed = false;

  constructor(device: GPUDevice, o: VDBFrameRendererGpuOptions) {
    this.device = device;
    this.#label = o.label ?? "vdb frame";
    this.settings = { ...vdbFrameDefaultSettings(), ...(o.settings ?? {}) };
    this.#composite = o.composite ?? null;
    this.#maxVolumes = Math.min(VDB_FRAME_MAX_VOLUMES, Math.floor((device.limits?.maxUniformBufferBindingSize ?? 65536) / VDB_FRAME_VOLUME_BYTES));
    this.#frameBuffer = device.createBuffer({ label: `${this.#label} uniforms`, size: VDB_FRAME_UNIFORMS.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#lightsBuffer = device.createBuffer({ label: `${this.#label} lights`, size: VDB_FRAME_LIGHTS.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#volumesWriter = new VDBUniformWriter(this.#maxVolumes * VDB_FRAME_VOLUME_BYTES);
    this.#volumesBuffer = device.createBuffer({ label: `${this.#label} volumes`, size: this.#maxVolumes * VDB_FRAME_VOLUME_BYTES, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#groupsWriter = new VDBUniformWriter(4 * VDB_FRAME_GROUP_STRIDE);
    this.#groupsBuffer = device.createBuffer({ label: `${this.#label} groups`, size: 4 * VDB_FRAME_GROUP_STRIDE, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#linear = device.createSampler({
      label: `${this.#label} linear clamp`,
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "nearest",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge",
    });
    this.outputs = { version: 0, froxels: null, uniforms: this.#frameBuffer };
  }

  // ---------------------------------------------------------------- volumes

  addVolume(slot: VDBVolumeSlot, params: Partial<VDBFrameVolumeParams> = {}): VDBFrameVolumeGpu {
    this.#check();
    if (!slot || !slot.source || typeof slot.source.wgsl !== "function" || !Number.isInteger(slot.v)) {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRendererGpu.addVolume: pass a VDBVolumeSlot ({ source, v }); a dense source from vdbDenseSource() is its own slot.");
    }
    if (slot.source.kind !== "dense") {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRendererGpu.addVolume: sparse sources come with K6.7 (vdbSparseSourceGpu); K6.1 renders dense sources (vdbDenseSource).");
    }
    const v = new FrameVolume(this, slot, params);
    this.#volumes.push(v);
    return v;
  }

  removeVolume(v: FrameVolume): void {
    const i = this.#volumes.indexOf(v);
    if (i >= 0) this.#volumes.splice(i, 1);
  }

  // ---------------------------------------------------------------- pipelines

  #compositeKey(): VDBFrameCompositeKey | null {
    if (!this.#composite) return null;
    return {
      colorFormat: this.#composite.colorFormat,
      depth: DEPTH_FORMATS.has(this.#composite.depthFormat) ? "depth" : "r32float",
      srgb: this.settings.srgbBlending,
    };
  }

  async prepare(options: { lighting?: ("volume" | "frame")[] } = {}): Promise<void> {
    this.#check();
    if (options.lighting?.includes("frame")) this.#warnOnce("prepare-frame", "prepare: 'frame' lighting comes with K6.2; preparing 'volume' lighting only.");
    const jobs: Promise<void>[] = [];
    const sources = new Set(this.#volumes.map((v) => v.slot.source));
    for (const source of sources) {
      for (const multi of [false, true]) jobs.push(prepareCompute(this.device, vdbFrameGatherSpec(source, { mix: this.settings.froxelMix, multi, maxVolumes: this.#maxVolumes })));
    }
    jobs.push(prepareCompute(this.device, vdbFrameResolveSpec({ temporal: this.settings.temporal })));
    const key = this.#compositeKey();
    if (key) jobs.push(prepareRender(this.device, vdbFrameCompositeSpec(key)));
    await Promise.all(jobs);
  }

  // ---------------------------------------------------------------- update

  update(encoder: GPUCommandEncoder, frame: VDBFrameInput, options: { froxels?: boolean } = {}): VDBFrameUpdate {
    this.#check();
    this.#retire.advance();
    const s = this.settings;
    if (s.lighting !== "volume") {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRendererGpu: 'frame' lighting comes with K6.2; set settings.lighting = 'volume'.");
    }
    if (frame.environment) this.#warnOnce("environment", "the environment input is K6.2; K6.1 ignores it.");
    if (frame.occluders && frame.occluders.length > 0) this.#warnOnce("occluders", "occluder maps (mesh → volume shadows) are K6.5b; K6.1 ignores them.");
    if (frame.lights.some((l) => "shadow" in l && l.shadow)) this.#warnOnce("shadow", "volume shadow map requests are K6.5; K6.1 ignores them.");
    if (s.spectral) this.#warnOnce("spectral", "settings.spectral is K6.4; K6.1 renders grey.");
    if (s.castShadows) this.#warnOnce("castShadows", "settings.castShadows is K6.5; K6.1 ignores it.");
    if (s.multipleScattering === "two-sided") this.#warnOnce("two-sided", "multipleScattering 'two-sided' is K6.3; K6.1 runs it as true (1D diffusion).");
    if (s.multipleScattering === "octaves") this.#warnOnce("octaves", "multipleScattering 'octaves' is K6.3; K6.1 runs it as false.");

    if (options.froxels === false && this.#textures) return { outputs: this.outputs, recorded: this.#recorded, volumes: 0 };

    // Froxel size (his getFroxelSize) and the textures
    const size = vdbFrameFroxelSize(s, frame.viewport, this.#held);
    const maxDim = this.device.limits?.maxTextureDimension3D ?? 2048;
    let W = size.width;
    let H = size.height;
    let D = Math.max(1, Math.round(s.depth));
    if (W > maxDim || H > maxDim || D + 1 > maxDim || 2 * D > maxDim) {
      this.#warnOnce("size", `the froxel grid ${W}x${H}x${D} exceeds maxTextureDimension3D (${maxDim}); clamped.`);
      W = Math.min(W, maxDim);
      H = Math.min(H, maxDim);
      D = Math.min(D, Math.floor(maxDim / 2));
    }

    // Collect and clip the visible volumes (his FrameVolumeCollector)
    const cam = frame.camera;
    const clip = vdbFrameClipCamera(cam.view, cam.projection, { coordinateSystem: "webgpu", reversedDepth: !!cam.reversedDepth });
    let near = Infinity;
    let far = 0;
    const clipped: { volume: FrameVolume; c: NonNullable<ReturnType<typeof vdbFrameClipVolume>> }[] = [];
    for (const volume of this.#volumes) {
      if (!volume.visible) continue;
      const c = vdbFrameClipVolume(clip, volume.params.matrixWorld);
      if (!c) continue;
      clipped.push({ volume, c });
      near = Math.min(near, c.viewNear);
      far = Math.max(far, c.viewFar);
    }
    near = Math.max(near, s.near, cam.near);
    far = Math.min(far, s.far, cam.far);
    if (clipped.length === 0 || far <= near) {
      this.#recorded = false;
      return { outputs: this.outputs, recorded: false, volumes: 0 };
    }
    far = Math.max(far, near * 1.0001);
    const range: [number, number] = [near, far];

    // Rects, slices and the froxel area
    let records: VolumeRecord[] = [];
    for (const { volume, c } of clipped) {
      const rect = vdbFrameFroxelRect(c.localToView, cam.projection, W, H);
      if (!rect) continue;
      const slices = vdbFrameVolumeSlices(c.viewNear, c.viewFar, range, D);
      records.push({ volume, localToView: c.localToView, viewToLocal: c.viewToLocal, worldToLocal: c.worldToLocal, rect, first: slices.first, count: slices.count });
    }
    if (records.length > this.#maxVolumes) {
      this.#warnOnce("volumes", `${records.length} visible volumes exceed the ${this.#maxVolumes} a frame holds (uniform binding size); the rest are not drawn.`);
      records = records.slice(0, this.#maxVolumes);
    }
    const area = vdbFrameFroxelArea(records, W, H, D);
    if (records.length === 0 || area.count <= 0) {
      this.#recorded = false;
      return { outputs: this.outputs, recorded: false, volumes: 0 };
    }

    // Source groups: volumes sharing a source object, in first-appearance order
    const groups = new Map<VDBVolumeSource, VolumeRecord[]>();
    for (const r of records) {
      const list = groups.get(r.volume.slot.source);
      if (list) list.push(r);
      else groups.set(r.volume.slot.source, [r]);
    }
    const ordered = [...groups.values()].flat();
    const multi = groups.size > 1;

    // Textures (a new size or temporal / multi-group need retires the old ones and drops the history)
    const tex = this.#ensureTextures(W, H, D, s.temporal, multi);

    // Temporal validity (his historyValid && historyCamera === camera; plus a consecutive frameId)
    const prev = this.#previous;
    const useHistory = s.temporal && this.#historyValid && !!prev && prev.cameraId === (cam.id ?? null) && frame.frameId === prev.frameId + 1;
    if (!s.temporal) this.#historyValid = false;

    // The dispatch box: this area and every rect a texture this update writes still holds
    const historyOut = tex.history ? tex.history[1 - this.#historyIndex]! : null;
    let box: VDBFroxelArea = { rect: [...area.rect], first: 0, count: D };
    for (const t of [tex.scatter, tex.integrated, historyOut]) {
      const w = t && this.#written.get(t);
      if (w) box = vdbFroxelAreaUnion(box, w);
    }

    // Uniforms
    const projectionInverse = vdbMat4Invert(cam.projection);
    const cameraWorld = vdbMat4Invert(cam.view);
    const fw = this.#frameWriter;
    const U = VDB_FRAME_UNIFORMS;
    fw.clear(0, U.size);
    fw.set(U, 0, "cameraWorld", cameraWorld);
    fw.set(U, 0, "projectionInverse", projectionInverse);
    fw.set(U, 0, "previousView", prev?.view ?? cam.view);
    fw.set(U, 0, "previousProjection", prev?.projection ?? cam.projection);
    fw.set(U, 0, "grid", [W, H, D, vdbFrameAtlasLayout(W, H, D).columns]);
    fw.set(U, 0, "range", [near, far, prev?.range[0] ?? near, prev?.range[1] ?? far]);
    fw.set(U, 0, "areaRect", area.rect);
    fw.set(U, 0, "areaSlices", [area.first, area.first + area.count]);
    fw.set(U, 0, "dispatchBox", [box.rect[0], box.rect[1], box.rect[2] - box.rect[0], box.rect[3] - box.rect[1]]);
    const jitter = s.jitter ? ((frame.frameId % 1024) * 0.618034) % 1.0 : -1.0;
    const fade = Number.isFinite(s.fadeRadius) && s.fadeRadius > 0 ? s.fadeRadius : 0;
    fw.set(U, 0, "jitterFade", [jitter, fade, Math.min(1, Math.max(0.01, s.temporalBlend)), useHistory ? 1 : 0]);
    fw.set(U, 0, "flags", [Math.max(1, Math.round(s.subSteps)), s.temporal ? 1 : 0, (cam.reversedDepth ? 1 : 0) | (cam.orthographic ? 2 : 0), 0]);
    fw.set(U, 0, "camera", [cam.near, cam.far, 0, 0]);
    this.device.queue.writeBuffer(this.#frameBuffer, 0, fw.buffer);
    this.#packLights(frame.lights);
    this.#packVolumes(ordered);
    const groupList = [...groups.values()];
    this.#packGroups(groupList);

    // Record: gather per source group, then resolve
    const pass = encoder.beginComputePass({ label: `${this.#label} froxels` });
    const group0 = this.#group0Bind();
    groupList.forEach((list, gi) => {
      const source = list[0]!.volume.slot.source;
      const built = computePipeline(this.device, vdbFrameGatherSpec(source, { mix: s.froxelMix, multi, maxVolumes: this.#maxVolumes }));
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, this.#gatherBind(built.layouts[1]!, tex, multi), [gi * VDB_FRAME_GROUP_STRIDE]);
      pass.setBindGroup(2, this.#sourceBind(source, built.layouts[2]!));
      pass.dispatchWorkgroups(Math.ceil((box.rect[2] - box.rect[0]) / 8), Math.ceil((box.rect[3] - box.rect[1]) / 8), D);
    });
    const resolve = computePipeline(this.device, vdbFrameResolveSpec({ temporal: s.temporal }));
    pass.setPipeline(resolve.pipeline);
    pass.setBindGroup(0, group0);
    pass.setBindGroup(1, this.#resolveBind(resolve.layouts[1]!, tex, s.temporal));
    pass.dispatchWorkgroups(Math.ceil((box.rect[2] - box.rect[0]) / 8), Math.ceil((box.rect[3] - box.rect[1]) / 8), 1);
    pass.end();

    // Book-keeping: what each texture holds, the history swap, the previous frame
    for (const t of [tex.scatter, tex.integrated, historyOut]) if (t) this.#written.set(t, area);
    if (tex.history) {
      this.#historyIndex = 1 - this.#historyIndex;
      this.#historyValid = true;
    }
    this.#previous = { view: Float64Array.from(cam.view), projection: Float64Array.from(cam.projection), range, cameraId: cam.id ?? null, frameId: frame.frameId };
    const layout: VDBFroxelLayout = { width: W, height: H, depth: D, range, screenRect: vdbFrameClipRect(area.rect, W, H, 1) };
    this.outputs.froxels = { integrated: tex.integrated, integratedAlpha: null, layout };
    this.#recorded = true;
    return { outputs: this.outputs, recorded: true, volumes: ordered.length };
  }

  #ensureTextures(W: number, H: number, D: number, temporal: boolean, multi: boolean): Textures {
    const t = this.#textures;
    const sameSize = !!t && t.size[0] === W && t.size[1] === H && t.size[2] === D;
    if (t && sameSize && (!temporal || t.history) && (!multi || t.partials)) {
      return t;
    }
    let changed = false;
    const next: Textures = sameSize && t ? { ...t } : (() => {
      if (t) {
        this.#retire.retire(t.scatter);
        this.#retire.retire(t.integrated);
        t.history?.forEach((h) => this.#retire.retire(h));
        this.#retire.retire(t.partials);
      }
      changed = true;
      this.#historyValid = false;
      return {
        size: [W, H, D],
        scatter: createFroxelTexture(this.device, `${this.#label} scatter`, [W, H, D], "rgba16float"),
        integrated: createFroxelTexture(this.device, `${this.#label} integrated`, [W, H, D + 1], "rgba16float"),
        history: null,
        partials: null,
      } as Textures;
    })();
    if (temporal && !next.history) {
      next.history = [createFroxelTexture(this.device, `${this.#label} history 0`, [W, H, D], "rgba16float"), createFroxelTexture(this.device, `${this.#label} history 1`, [W, H, D], "rgba16float")];
      this.#historyValid = false;
      this.#historyIndex = 0;
    }
    if (multi && !next.partials) next.partials = createFroxelTexture(this.device, `${this.#label} partials`, [W, H, 2 * D], "r32uint", false);
    this.#textures = next;
    if (changed) this.outputs.version++;
    return next;
  }

  #group0Bind(): GPUBindGroup {
    if (!this.#group0) {
      const layout = computePipeline(this.device, vdbFrameResolveSpec({ temporal: false })).layouts[0]!;
      this.#group0 = this.device.createBindGroup({
        label: `${this.#label} group 0`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.#frameBuffer } },
          { binding: 1, resource: { buffer: this.#lightsBuffer } },
          { binding: 2, resource: this.#linear },
        ],
      });
    }
    return this.#group0;
  }

  readonly #bindCache = new Map<string, { keys: readonly object[]; group: GPUBindGroup }>();

  /** A bind group cached on the identity of everything it binds (rebuilt when one changes). */
  #cached(name: string, keys: readonly object[], make: () => GPUBindGroup): GPUBindGroup {
    const hit = this.#bindCache.get(name);
    if (hit && hit.keys.length === keys.length && hit.keys.every((k, i) => k === keys[i])) return hit.group;
    const group = make();
    this.#bindCache.set(name, { keys, group });
    return group;
  }

  #gatherBind(layout: GPUBindGroupLayout, tex: Textures, multi: boolean): GPUBindGroup {
    const b = GATHER_BINDINGS;
    return this.#cached(`gather${multi ? "-multi" : ""}`, [layout, tex.scatter, ...(multi ? [tex.partials!] : []), this.#groupsBuffer], () =>
      this.device.createBindGroup({
        label: `${this.#label} gather group 1`,
        layout,
        entries: [
          { binding: b.volumes, resource: { buffer: this.#volumesBuffer } },
          { binding: b.group, resource: { buffer: this.#groupsBuffer, size: VDB_FRAME_GROUP.size } },
          { binding: b.scatter, resource: tex.scatter.createView({ dimension: "3d" }) },
          ...(multi ? [{ binding: b.partials, resource: tex.partials!.createView({ dimension: "3d" }) }] : []),
        ],
      }),
    );
  }

  #resolveBind(layout: GPUBindGroupLayout, tex: Textures, temporal: boolean): GPUBindGroup {
    const b = RESOLVE_BINDINGS;
    const historyIn = tex.history ? tex.history[this.#historyIndex]! : null;
    const historyOut = tex.history ? tex.history[1 - this.#historyIndex]! : null;
    return this.#cached(`resolve${temporal ? `-${this.#historyIndex}` : ""}`, [layout, tex.scatter, tex.integrated, ...(temporal ? [historyIn!, historyOut!] : [])], () =>
      this.device.createBindGroup({
        label: `${this.#label} resolve group 1`,
        layout,
        entries: [
          { binding: b.scatter, resource: tex.scatter.createView({ dimension: "3d" }) },
          { binding: b.integrated, resource: tex.integrated.createView({ dimension: "3d" }) },
          ...(temporal
            ? [
                { binding: b.historyIn, resource: historyIn!.createView({ dimension: "3d" }) },
                { binding: b.historyOut, resource: historyOut!.createView({ dimension: "3d" }) },
              ]
            : []),
        ],
      }),
    );
  }

  #sourceBind(source: VDBVolumeSource, layout: GPUBindGroupLayout): GPUBindGroup {
    const hit = this.#sourceGroups.get(source);
    if (hit && hit.version === source.version && hit.layout === layout) return hit.group;
    const group = this.device.createBindGroup({ label: `${this.#label} source group 2`, layout, entries: source.bindEntries(SOURCE_SLOT) });
    this.#sourceGroups.set(source, { version: source.version, layout, group });
    return group;
  }

  // ---------------------------------------------------------------- packing

  #packLights(lights: readonly VDBFrameLight[]): void {
    const w = this.#lightsWriter;
    const L = VDB_FRAME_LIGHTS;
    w.clear(0, L.size);
    const ambient = [0, 0, 0];
    const probe = new Float32Array(36);
    const hemi: { sky: readonly number[]; ground: readonly number[]; up: number[] }[] = [];
    const directional: number[][] = [];
    const point: number[][] = [];
    const spot: number[][] = [];
    const norm = (v: readonly number[]) => {
      const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
      return [v[0]! / l, v[1]! / l, v[2]! / l];
    };
    let hemiTotal = 0;
    for (const l of lights) {
      switch (l.kind) {
        case "ambient":
          for (let i = 0; i < 3; i++) ambient[i] = ambient[i]! + l.color[i]!;
          break;
        case "probe":
          for (let i = 0; i < 9; i++) for (let c = 0; c < 3; c++) probe[i * 4 + c] = probe[i * 4 + c]! + (l.sh[i * 3 + c] ?? 0);
          break;
        case "hemisphere":
          hemiTotal++;
          if (hemi.length < VDB_FRAME_MAX_HEMISPHERE_LIGHTS) hemi.push({ sky: l.sky, ground: l.ground, up: norm(l.up) });
          break;
        case "directional": {
          const d = norm(l.direction);
          directional.push([-d[0]!, -d[1]!, -d[2]!, 1, ...l.color, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          break;
        }
        case "point":
          point.push([...l.position, 2, ...l.color, 0, 0, 0, 0, 0, l.distance, l.decay, 0, 0]);
          break;
        case "spot": {
          const d = norm(l.direction);
          spot.push([...l.position, 3, ...l.color, l.cosPenumbra, -d[0]!, -d[1]!, -d[2]!, l.cosAngle, l.distance, l.decay, 0, 0]);
          break;
        }
      }
    }
    const sorted = [...directional, ...point, ...spot];
    if (sorted.length > VDB_FRAME_MAX_LIGHTS || hemiTotal > VDB_FRAME_MAX_HEMISPHERE_LIGHTS) {
      this.#warnOnce("lights", `too many lights: only ${VDB_FRAME_MAX_LIGHTS} lights and ${VDB_FRAME_MAX_HEMISPHERE_LIGHTS} hemisphere lights are used.`);
    }
    const used = sorted.slice(0, VDB_FRAME_MAX_LIGHTS);
    w.set(L, 0, "ambient", [ambient[0]!, ambient[1]!, ambient[2]!, hemi.length]);
    w.set(L, 0, "counts", [used.length, 0, 0, 0]);
    w.set(L, 0, "probe", probe);
    w.set(L, 0, "hemiSky", hemi.flatMap((h) => [...h.sky, 0]));
    w.set(L, 0, "hemiGround", hemi.flatMap((h) => [...h.ground, 0]));
    w.set(L, 0, "hemiUp", hemi.flatMap((h) => [...h.up, 0]));
    w.set(L, 0, "position", used.flatMap((r) => r.slice(0, 4)));
    w.set(L, 0, "color", used.flatMap((r) => r.slice(4, 8)));
    w.set(L, 0, "spot", used.flatMap((r) => r.slice(8, 12)));
    w.set(L, 0, "falloff", used.flatMap((r) => r.slice(12, 16)));
    this.device.queue.writeBuffer(this.#lightsBuffer, 0, w.buffer);
  }

  #phaseOf(p: VDBFrameVolumeParams): VDBFramePhaseFields {
    const s = this.settings;
    return {
      phase: p.phase ?? s.phase,
      phaseG: p.phaseG ?? s.phaseG,
      phaseBackG: p.phaseBackG ?? s.phaseBackG,
      phaseBackWeight: p.phaseBackWeight ?? s.phaseBackWeight,
      phaseGExponent: p.phaseGExponent ?? s.phaseGExponent,
    };
  }

  #packVolumes(records: readonly VolumeRecord[]): void {
    const w = this.#volumesWriter;
    const V = VDB_FRAME_VOLUME;
    records.forEach((r, i) => {
      const base = i * VDB_FRAME_VOLUME_BYTES;
      const p = r.volume.params;
      const source = r.volume.slot.source;
      const phaseFields = this.#phaseOf(p);
      const phase = vdbFramePhaseVector(phaseFields);
      if (phase[3] > 0) {
        this.#warnOnce("mie", "the Lorenz-Mie phases are K6.3; K6.1 lights them with the phase's lobes off (isotropic).");
        phase[3] = 0;
      }
      const ms = vdbFrameMultipleScatteringValue(this.settings.multipleScattering === "octaves" ? false : this.settings.multipleScattering, phaseFields);
      const mean = p.densityMean ?? source.stats.meanNonZero;
      const densityScale = p.normalizeDensity && mean > 0 ? p.densityScale / mean : p.densityScale;
      const levels = source.lodLevels;
      w.clear(base, VDB_FRAME_VOLUME_BYTES);
      w.set(V, base, "viewToLocal", r.viewToLocal);
      w.set(V, base, "localToView", r.localToView);
      w.set(V, base, "worldToLocal", r.worldToLocal);
      w.set(V, base, "rect", r.rect);
      w.set(V, base, "slices", [r.first, r.count, r.volume.slot.v, p.lights >>> 0]);
      w.set(V, base, "scatterOpacity", [...p.scatterColor, p.opacity]);
      w.set(V, base, "baseExtinction", [...p.baseColor, p.extinction]);
      w.set(V, base, "offsetWrap", [...p.offset3D, WRAP_CODE[p.wrap3D] ?? 0]);
      w.set(V, base, "phase", phase);
      w.set(V, base, "phaseSpectrumMs", [...vdbFramePhaseSpectrum(phaseFields), ms]);
      w.set(V, base, "density", [densityScale, p.densityCutoff, vdbMat4MaxScale(p.matrixWorld), p.absorbance]);
      w.set(V, base, "emission", [p.emissiveIntensity, p.temperatureRange[0], p.temperatureRange[1], levels]);
      w.set(V, base, "march", [Math.max(0, Math.round(p.shadowSteps)), p.mipmaps && levels > 0 ? 1 : 0, p.hdda ? 1 : 0, 0]);
      w.set(V, base, "extinctionSpectrum", [1, 1, 1, 0]);
    });
    this.device.queue.writeBuffer(this.#volumesBuffer, 0, w.buffer, 0, Math.max(1, records.length) * VDB_FRAME_VOLUME_BYTES);
  }

  #packGroups(groups: readonly VolumeRecord[][]): void {
    const bytes = Math.max(4, groups.length) * VDB_FRAME_GROUP_STRIDE;
    if (this.#groupsBuffer.size < bytes) {
      this.#retire.retire(this.#groupsBuffer);
      this.#groupsBuffer = this.device.createBuffer({ label: `${this.#label} groups`, size: bytes, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
      this.#groupsWriter = new VDBUniformWriter(bytes);
    }
    const w = this.#groupsWriter;
    let first = 0;
    groups.forEach((list, i) => {
      const role = (i === 0 ? 1 : 0) | (i === groups.length - 1 ? 2 : 0);
      w.set(VDB_FRAME_GROUP, i * VDB_FRAME_GROUP_STRIDE, "range", [first, list.length, role, i]);
      first += list.length;
    });
    this.device.queue.writeBuffer(this.#groupsBuffer, 0, w.buffer, 0, groups.length * VDB_FRAME_GROUP_STRIDE);
  }

  // ---------------------------------------------------------------- render

  render(encoder: GPUCommandEncoder, targets: VDBFrameTargets): void {
    this.#check();
    const key = this.#compositeKey();
    if (!key) throw new VDBError("VDB_BAD_OPTION", "VDBFrameRendererGpu.render: create the renderer with options.composite ({ colorFormat, depthFormat }).");
    const froxels = this.outputs.froxels;
    if (!this.#recorded || !froxels) return;
    const srgb = key.srgb && (targets.outputColorSpace ?? "srgb") === "srgb";
    const finalKey: VDBFrameCompositeKey = { ...key, srgb };
    if (srgb && !targets.sceneColor) {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRendererGpu.render: sRGB blending reads the scene colour: pass targets.sceneColor (a copy of the colour target), or set settings.srgbBlending = false.");
    }
    const [sw, sh] = targets.size;
    // Only the screen-size fields move here; the rest is the update's.
    const fw = this.#frameWriter;
    fw.set(VDB_FRAME_UNIFORMS, 0, "screen", [sw, sh, (targets.clampColors ?? true) ? 1 : 0, 0]);
    const screen = VDB_FRAME_UNIFORMS.fields.get("screen")!;
    this.device.queue.writeBuffer(this.#frameBuffer, screen.offset, fw.buffer, screen.offset, 16);
    const built = renderPipeline(this.device, vdbFrameCompositeSpec(finalKey));
    const [x0, y0, x1, y1] = froxels.layout.screenRect;
    // clip → pixels (y down)
    const px0 = Math.max(0, Math.floor((x0 * 0.5 + 0.5) * sw));
    const px1 = Math.min(sw, Math.ceil((x1 * 0.5 + 0.5) * sw));
    const py0 = Math.max(0, Math.floor((1 - (y1 * 0.5 + 0.5)) * sh));
    const py1 = Math.min(sh, Math.ceil((1 - (y0 * 0.5 + 0.5)) * sh));
    if (px1 <= px0 || py1 <= py0) return;
    const group1 = this.device.createBindGroup({
      label: `${this.#label} composite group 1`,
      layout: built.layouts[1]!,
      entries: [
        { binding: 0, resource: froxels.integrated.createView({ dimension: "3d" }) },
        { binding: 1, resource: this.#linear },
        { binding: 2, resource: targets.depth },
        ...(srgb ? [{ binding: 3, resource: targets.sceneColor! }] : []),
      ],
    });
    const pass = encoder.beginRenderPass({ label: `${this.#label} composite`, colorAttachments: [{ view: targets.color, loadOp: "load", storeOp: "store" }] });
    pass.setPipeline(built.pipeline);
    pass.setBindGroup(0, this.#group0Bind());
    pass.setBindGroup(1, group1);
    pass.setScissorRect(px0, py0, px1 - px0, py1 - py0);
    pass.draw(3);
    pass.end();
  }

  // ---------------------------------------------------------------- lifecycle

  #warnOnce(key: string, message: string): void {
    if (this.#warned.has(key)) return;
    this.#warned.add(key);
    console.warn(`@pmndrs/vdb frame renderer: ${message}`);
  }

  #check(): void {
    if (this.#disposed) throw new VDBError("VDB_DISPOSED", "VDBFrameRendererGpu: the renderer was disposed; create a new one.");
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const t = this.#textures;
    if (t) {
      t.scatter.destroy();
      t.integrated.destroy();
      t.history?.forEach((h) => h.destroy());
      t.partials?.destroy();
    }
    this.#textures = null;
    this.#retire.drain();
    this.#frameBuffer.destroy();
    this.#lightsBuffer.destroy();
    this.#volumesBuffer.destroy();
    this.#groupsBuffer.destroy();
    this.outputs.froxels = null;
    this.#volumes.length = 0;
  }

  /** @internal The textures (harness probes and tests). */
  get debugTextures(): Readonly<Textures> | null {
    return this.#textures;
  }
}

/**
 * Creates the core frame renderer on the host's device (design §2.2). It makes its uniform buffers now and
 * its froxel textures at the first `update()` with volumes in view.
 *
 * @param device The host's `GPUDevice`.
 * @param options {@link VDBFrameRendererGpuOptions}.
 */
export function vdbCreateFrameRenderer(device: GPUDevice, options: VDBFrameRendererGpuOptions = {}): VDBFrameRendererGpu {
  if (!device || typeof (device as { createTexture?: unknown }).createTexture !== "function") {
    throw new VDBError("VDB_UNSUPPORTED_DEVICE", "vdbCreateFrameRenderer: pass the host's GPUDevice.");
  }
  return new FrameRenderer(device, options);
}

/** @internal The froxel textures of a renderer from {@link vdbCreateFrameRenderer} (harness probes). */
export function vdbFrameRendererTextures(r: VDBFrameRendererGpu): { scatter: GPUTexture; integrated: GPUTexture; history: readonly GPUTexture[] | null; partials: GPUTexture | null; size: readonly number[] } | null {
  return (r as unknown as { debugTextures: Textures | null }).debugTextures;
}
