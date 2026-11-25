// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameVolumeShadows.js L459–548 (entries, resize), L650–686 (spot and point
//   cameras), L830–989 (collectWork, the budgeted updateMaps); past the pin: his 62a6e3e (receivers read the drawn
//   state) and c13f2d5 (the budget counts only lights of the rendered scene).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free (K6.5, design §2.6): light data in, map textures and drawn-view uniform buffers out; the
// receivers read the drawn view from a buffer the core writes when it draws a piece; plain-data cameras (the
// directional cascades from tools/frame `vdbFrameShadowFitCascades`). Divergences: DIVERGES.md (K6.5 rows).

/**
 * The CPU half of the volume shadow maps (K6.5): one entry per light with a `shadow` request; each update fits its
 * cameras, finds the changed pieces (a spot map, a cube map, one cascade, or a new cascade layout), draws them oldest
 * first within `castShadowUpdatesPerFrame` (the first always; a tile or spot costs 1, a cube 6, a layout its cascades),
 * and writes every map's drawn view. Lights not in this frame's `lights` are not in the budget (his c13f2d5); their
 * maps are freed.
 */

import {
  frameMaxCastShadowSteps,
  frameShadowCoarseMapSize,
  frameShadowCoarseSteps,
  vdbFrameShadowFitCascades,
  vdbFrameShadowViewCamera,
} from "../../tools/frame/shadows.js";
import { vdbMat4Invert, vdbMat4Multiply } from "../../tools/frame/view.js";
import type { VDBCachedPipeline } from "./pipelines.js";
import { BUFFER_USAGE, TEXTURE_USAGE, type VDBRetireQueue } from "./resources.js";
import type { VDBFrameLight, VDBFrameVec3, VDBVolumeShadowDrawn, VDBVolumeShadowMap, VDBVolumeShadowRequest } from "./types.js";
import { VDBUniformWriter } from "./uniforms.js";
import { VDB_FRAME_SHADOW_PIECE, VDB_FRAME_SHADOW_PIECE_STRIDE, VDB_VOLUME_SHADOW_MAX_CASCADES, VDB_VOLUME_SHADOW_VIEW, VOLUME_SHADOW_BINDINGS } from "./wgsl/shadows.js";

type V3 = [number, number, number];
type Kind = "directional" | "spot" | "point";

const KIND_CODE: Record<Kind, number> = { directional: 0, spot: 1, point: 2 };
const LAYERS: Record<Kind, number> = { directional: VDB_VOLUME_SHADOW_MAX_CASCADES, spot: 1, point: 6 };

interface Item {
  signature: unknown[];
  lastDraw: number;
}

interface Cascade extends Item {
  viewProjection: Float64Array;
  origin: V3;
  far: number;
  drawn: { viewProjection: Float32Array; origin: V3; far: number };
}

/** The camera of a spot or point map this update. */
interface Camera {
  inverse: Float64Array;
  viewProjection: Float64Array | null;
  position: V3;
  forward: V3;
  perspective: boolean;
  far: number;
}

interface Entry {
  id: number | string;
  kind: Kind;
  size: number;
  texture: GPUTexture;
  viewBuffer: GPUBuffer;
  item: Item;
  cascades: Cascade[] | null;
  cascadeCount: number;
  breaks: number[];
  forward: V3;
  camera: Camera | null;
  drawn: { viewProjection: Float32Array[]; origin: V3[]; far: number[]; forward: V3; perspective: boolean; breaks: number[]; cascadeCount: number };
  everDrawn: boolean;
  enabled: boolean;
  revision: number;
  map: { -readonly [K in keyof VDBVolumeShadowMap]: VDBVolumeShadowMap[K] };
}

/** One dispatch of the shadow kernel: a piece record and its map. */
interface Dispatch {
  entry: Entry;
  layers: number;
  record: { inverse: ArrayLike<number>; camera: [number, number, number, number]; forward: [number, number, number, number]; info: [number, number, number, number] };
}

/** What `update` needs from the renderer. */
export interface VDBFrameShadowInputs {
  readonly lights: readonly VDBFrameLight[];
  readonly castShadows: boolean;
  readonly coarse: boolean;
  readonly steps: number;
  readonly budget: number;
  readonly cascades: number;
  readonly castShadowDistance: number;
  /** The frame camera: world → view, view → world, clip → view, near, far, orthographic. */
  readonly camera: { view: ArrayLike<number>; cameraWorld: ArrayLike<number>; projectionInverse: ArrayLike<number>; near: number; far: number; orthographic: boolean };
  /** The last shadow grid cascade (what the maps march), or null without casters. */
  readonly grid: { min: V3; size: V3 } | null;
  /** The grid's last-cascade version (his castVersion). */
  readonly castVersion: number;
}

const changed = (item: Item, signature: unknown[]): boolean => {
  const same = signature.length === item.signature.length && signature.every((v, i) => v === item.signature[i]);
  if (!same) item.signature = signature;
  return !same;
};

const f32 = (m: ArrayLike<number>) => Float32Array.from({ length: 16 }, (_, i) => m[i]!);
const norm = (v: readonly number[]): V3 => {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
  return [v[0]! / l, v[1]! / l, v[2]! / l];
};

/** A perspective camera over a spot light's cone (the default without a request view): WebGPU depth, look-at. */
function spotCamera(position: VDBFrameVec3, direction: VDBFrameVec3, cosAngle: number, distance: number): { viewProjection: Float64Array; near: number; far: number } {
  const near = 0.5;
  const far = distance > 0 ? distance : 500;
  const f = 1 / Math.tan(Math.acos(Math.min(1, Math.max(-1, cosAngle))));
  const projection = [f, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
  const z = norm([-direction[0], -direction[1], -direction[2]]);
  const up: V3 = Math.abs(z[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const x = norm([up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]);
  const y: V3 = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const world = [...x, 0, ...y, 0, ...z, 0, position[0], position[1], position[2], 1];
  return { viewProjection: vdbMat4Multiply(projection, vdbMat4Invert(world)), near, far };
}

/** The volume shadow maps of one core frame renderer. */
export class VDBFrameShadowMaps {
  /** The maps by light id (the renderer's `outputs.volumeShadows`). */
  readonly maps = new Map<number | string, VDBVolumeShadowMap>();
  /** Pieces drawn and waiting in the last update. */
  readonly stats = { drawn: 0, waiting: 0 };
  /** +1 when a map texture is a new object. */
  version = 0;

  readonly #device: GPUDevice;
  readonly #label: string;
  readonly #retire: VDBRetireQueue;
  readonly #entries = new Map<number | string, Entry>();
  readonly #viewWriter = new VDBUniformWriter(VDB_VOLUME_SHADOW_VIEW.size);
  #pieceBuffer: GPUBuffer | null = null;
  #pieceWriter: VDBUniformWriter | null = null;
  #dispatches: Dispatch[] = [];
  #renderCount = 0;
  readonly #groups = new WeakMap<GPUTexture, { keys: readonly object[]; group: GPUBindGroup }>();

  constructor(device: GPUDevice, label: string, retire: VDBRetireQueue) {
    this.#device = device;
    this.#label = label;
    this.#retire = retire;
  }

  /** True when some light this frame asks for a map (the renderer then draws the shadow grid). */
  static wanted(lights: readonly VDBFrameLight[], castShadows: boolean): boolean {
    return castShadows && lights.some((l) => (l.kind === "directional" || l.kind === "spot" || l.kind === "point") && !!l.shadow);
  }

  /** The CPU work of one update: entries, cameras, changed pieces within the budget, drawn views. Returns the dispatch count. */
  update(o: VDBFrameShadowInputs): number {
    this.#dispatches = [];
    this.stats.drawn = 0;
    this.stats.waiting = 0;
    if (!o.castShadows) {
      for (const e of [...this.#entries.values()]) this.#free(e);
      return 0;
    }
    this.#renderCount++;

    // Entries of this frame's shadow lights (first-seen order); lights not in the frame are freed
    const active: { entry: Entry; light: VDBFrameLight & { shadow?: VDBVolumeShadowRequest } }[] = [];
    const seen = new Set<number | string>();
    for (const light of o.lights) {
      if (light.kind !== "directional" && light.kind !== "spot" && light.kind !== "point") continue;
      if (!light.shadow || seen.has(light.id)) continue;
      seen.add(light.id);
      const size = o.coarse ? frameShadowCoarseMapSize : Math.max(1, Math.round(light.shadow.size));
      active.push({ entry: this.#entry(light.id, light.kind, size), light });
    }
    for (const e of [...this.#entries.values()]) if (!seen.has(e.id)) this.#free(e);
    for (const e of this.#entries.values()) e.enabled = false;

    if (!o.grid) {
      // No caster: no shadow; every piece draws again once there is one
      for (const { entry } of active) {
        entry.item.signature = [];
        entry.cascades?.forEach((c) => (c.signature = []));
      }
      this.#writeViews();
      return 0;
    }

    const steps = o.coarse ? frameShadowCoarseSteps : Math.min(frameMaxCastShadowSteps, Math.max(1, Math.round(o.steps)));
    const work: { item: Item; cost: number; draw: () => void }[] = [];
    for (const { entry, light } of active) {
      entry.enabled = true;
      this.#collect(entry, light, o, steps, work);
    }

    // Oldest first; the first piece always draws (a large piece does not wait forever)
    work.sort((a, b) => a.item.lastDraw - b.item.lastDraw);
    let budget = Math.max(1, Math.round(o.budget));
    work.forEach((piece, index) => {
      if (index > 0 && piece.cost > budget) {
        piece.item.signature = [];
        this.stats.waiting++;
        return;
      }
      piece.draw();
      piece.item.lastDraw = this.#renderCount;
      budget -= piece.cost;
      this.stats.drawn++;
    });

    this.#writePieces();
    this.#writeViews();
    return this.#dispatches.length;
  }

  /** Records the drawn pieces (after the shadow grid in the same pass). */
  record(pass: GPUComputePassEncoder, built: VDBCachedPipeline<GPUComputePipeline>, group0: GPUBindGroup, grid: GPUTexture): void {
    if (this.#dispatches.length === 0) return;
    pass.setPipeline(built.pipeline);
    pass.setBindGroup(0, group0);
    const layout = built.layouts[1]!;
    this.#dispatches.forEach((d, i) => {
      pass.setBindGroup(1, this.#bind(layout, d.entry.texture, grid), [i * VDB_FRAME_SHADOW_PIECE_STRIDE]);
      const groups = Math.ceil(d.entry.size / 8);
      pass.dispatchWorkgroups(groups, groups, d.layers);
    });
  }

  #bind(layout: GPUBindGroupLayout, map: GPUTexture, grid: GPUTexture): GPUBindGroup {
    const keys = [layout, grid, this.#pieceBuffer!];
    const hit = this.#groups.get(map);
    if (hit && hit.keys.every((k, i) => k === keys[i])) return hit.group;
    const b = VOLUME_SHADOW_BINDINGS;
    const group = this.#device.createBindGroup({
      label: `${this.#label} volume shadow group 1`,
      layout,
      entries: [
        { binding: b.piece, resource: { buffer: this.#pieceBuffer!, size: VDB_FRAME_SHADOW_PIECE.size } },
        { binding: b.grid, resource: grid.createView({ dimension: "3d" }) },
        { binding: b.map, resource: map.createView({ dimension: "2d-array" }) },
      ],
    });
    this.#groups.set(map, { keys, group });
    return group;
  }

  #entry(id: number | string, kind: Kind, size: number): Entry {
    let e = this.#entries.get(id);
    if (e && e.kind === kind && e.size === size) return e;
    if (e) this.#free(e);
    const texture = this.#device.createTexture({
      label: `${this.#label} volume shadow ${kind} ${String(id)}`,
      size: [size, size, LAYERS[kind]],
      format: "rgba16float",
      usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_SRC,
    });
    const viewBuffer = this.#device.createBuffer({ label: `${this.#label} volume shadow view ${String(id)}`, size: VDB_VOLUME_SHADOW_VIEW.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    const drawn = { viewProjection: [] as Float32Array[], origin: [] as V3[], far: [] as number[], forward: [0, 0, -1] as V3, perspective: kind !== "directional", breaks: [0, 0, 0, 0], cascadeCount: 0 };
    e = {
      id,
      kind,
      size,
      texture,
      viewBuffer,
      item: { signature: [], lastDraw: -1 },
      cascades:
        kind === "directional"
          ? Array.from({ length: VDB_VOLUME_SHADOW_MAX_CASCADES }, () => ({
              signature: [],
              lastDraw: -1,
              viewProjection: new Float64Array(16),
              origin: [0, 0, 0] as V3,
              far: 1,
              drawn: { viewProjection: new Float32Array(16), origin: [0, 0, 0] as V3, far: 1 },
            }))
          : null,
      cascadeCount: 0,
      breaks: [0, 0, 0, 0],
      forward: [0, 0, -1],
      camera: null,
      drawn,
      everDrawn: false,
      enabled: false,
      revision: 0,
      map: { light: id, kind, texture, layers: LAYERS[kind], size, view: viewBuffer, drawn: drawn as VDBVolumeShadowDrawn, enabled: false, revision: 0, lastDraw: [] },
    };
    this.#entries.set(id, e);
    this.maps.set(id, e.map);
    this.version++;
    return e;
  }

  #free(e: Entry): void {
    this.#retire.retire(e.texture);
    this.#retire.retire(e.viewBuffer);
    this.#entries.delete(e.id);
    this.maps.delete(e.id);
    this.version++;
  }

  /** His collectWork: the changed pieces of one light (fits cascades / sets the spot or cube camera first). */
  #collect(entry: Entry, light: VDBFrameLight & { shadow?: VDBVolumeShadowRequest }, o: VDBFrameShadowInputs, steps: number, work: { item: Item; cost: number; draw: () => void }[]): void {
    const request = light.shadow!;
    const view = request.view;
    if (entry.kind === "directional" && light.kind === "directional") {
      const cascades = entry.cascades!;
      let count: number;
      let breaks: number[];
      let forward: V3;
      if (view && typeof view === "object" && "cascades" in view) {
        count = Math.min(VDB_VOLUME_SHADOW_MAX_CASCADES, view.cascades.length);
        view.cascades.slice(0, count).forEach((c, i) => {
          const cam = vdbFrameShadowViewCamera(c.viewProjection, c.near, true);
          const cascade = cascades[i]!;
          cascade.viewProjection = Float64Array.from({ length: 16 }, (_, k) => c.viewProjection[k]!);
          cascade.origin = cam.position;
          cascade.far = c.far;
        });
        breaks = Array.from({ length: 4 }, (_, i) => view.breaks[Math.min(i, count - 1)] ?? 0);
        forward = norm(light.direction);
      } else {
        const g = o.grid!;
        const fit = vdbFrameShadowFitCascades({
          cameraWorld: o.camera.cameraWorld,
          projectionInverse: o.camera.projectionInverse,
          near: o.camera.near,
          far: o.camera.far,
          orthographic: o.camera.orthographic,
          castShadowDistance: o.castShadowDistance,
          count: o.cascades,
          size: entry.size,
          lightDirection: [light.direction[0], light.direction[1], light.direction[2]],
          casterBox: { min: g.min, max: [g.min[0] + g.size[0], g.min[1] + g.size[1], g.min[2] + g.size[2]] },
        });
        count = fit.count;
        breaks = fit.breaks;
        forward = fit.forward;
        fit.cascades.forEach((c, i) => {
          const cascade = cascades[i]!;
          cascade.viewProjection = c.viewProjection;
          cascade.origin = c.origin;
          cascade.far = c.far;
        });
      }
      entry.cascadeCount = count;
      entry.breaks = breaks;
      entry.forward = forward;
      const drawCascade = (i: number) => {
        const c = cascades[i]!;
        this.#dispatches.push({
          entry,
          layers: 1,
          record: { inverse: vdbMat4Invert(c.viewProjection), camera: [...c.origin, 0], forward: [...forward, c.far], info: [i, entry.size, 0, steps] },
        });
        c.drawn = { viewProjection: f32(c.viewProjection), origin: [...c.origin], far: c.far };
        c.lastDraw = this.#renderCount;
        entry.drawn.forward = [...forward];
        entry.drawn.perspective = false;
        entry.revision++;
        entry.everDrawn = true;
      };
      const cascadeSignature = (i: number) => [entry.size, o.castVersion, steps, ...cascades[i]!.viewProjection, cascades[i]!.far];
      // A new split layout: one piece draws every cascade, then receivers switch to it (never two layouts' tiles)
      if (changed(entry.item, [entry.size, count, ...breaks])) {
        for (let i = 0; i < count; i++) changed(cascades[i]!, cascadeSignature(i));
        work.push({
          item: entry.item,
          cost: count,
          draw: () => {
            for (let i = 0; i < count; i++) drawCascade(i);
            entry.drawn.cascadeCount = count;
            entry.drawn.breaks = [...breaks];
          },
        });
        return;
      }
      for (let i = 0; i < count; i++) if (changed(cascades[i]!, cascadeSignature(i))) work.push({ item: cascades[i]!, cost: 1, draw: () => drawCascade(i) });
      return;
    }

    // Spot: the request's camera (or one over the cone); point: a cube at the light
    let camera: Camera;
    let signature: unknown[];
    if (light.kind === "spot") {
      const v = view && typeof view === "object" && !("cascades" in view) && view.viewProjection ? { viewProjection: view.viewProjection, near: view.near, far: view.far, orthographic: !!view.orthographic } : { ...spotCamera(light.position, light.direction, light.cosAngle, light.distance), orthographic: false };
      const cam = vdbFrameShadowViewCamera(v.viewProjection, v.near, v.orthographic);
      camera = { inverse: cam.inverse, viewProjection: Float64Array.from({ length: 16 }, (_, k) => v.viewProjection[k]!), position: cam.position, forward: cam.forward, perspective: !v.orthographic, far: v.far };
      signature = [entry.size, o.castVersion, steps, ...camera.viewProjection!, v.near, v.far, v.orthographic];
    } else {
      const position = (light as { position: VDBFrameVec3 }).position;
      const far = view && typeof view === "object" && !("cascades" in view) ? view.far : (light as { distance: number }).distance > 0 ? (light as { distance: number }).distance : 500;
      camera = { inverse: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), viewProjection: null, position: [position[0], position[1], position[2]], forward: [0, 0, -1], perspective: true, far };
      signature = [entry.size, o.castVersion, steps, ...camera.position, far];
    }
    entry.camera = camera;
    if (!changed(entry.item, signature)) return;
    const point = entry.kind === "point";
    work.push({
      item: entry.item,
      // all six cube faces in one piece: the faces always match
      cost: point ? 6 : 1,
      draw: () => {
        this.#dispatches.push({
          entry,
          layers: point ? 6 : 1,
          record: { inverse: camera.inverse, camera: [...camera.position, camera.perspective ? 1 : 0], forward: [...camera.forward, camera.far], info: [0, entry.size, point ? 2 : 0, steps] },
        });
        entry.drawn = {
          viewProjection: camera.viewProjection ? [f32(camera.viewProjection)] : [],
          origin: [[...camera.position]],
          far: [camera.far],
          forward: [...camera.forward],
          perspective: camera.perspective,
          breaks: [0, 0, 0, 0],
          cascadeCount: 0,
        };
        entry.revision++;
        entry.everDrawn = true;
      },
    });
  }

  #writePieces(): void {
    const bytes = Math.max(1, this.#dispatches.length) * VDB_FRAME_SHADOW_PIECE_STRIDE;
    if (!this.#pieceBuffer || this.#pieceBuffer.size < bytes) {
      this.#retire.retire(this.#pieceBuffer);
      const size = Math.max(bytes, 16 * VDB_FRAME_SHADOW_PIECE_STRIDE);
      this.#pieceBuffer = this.#device.createBuffer({ label: `${this.#label} volume shadow pieces`, size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
      this.#pieceWriter = new VDBUniformWriter(size);
    }
    if (this.#dispatches.length === 0) return;
    const w = this.#pieceWriter!;
    const P = VDB_FRAME_SHADOW_PIECE;
    this.#dispatches.forEach((d, i) => {
      const base = i * VDB_FRAME_SHADOW_PIECE_STRIDE;
      w.clear(base, VDB_FRAME_SHADOW_PIECE_STRIDE);
      w.set(P, base, "inverseViewProjection", d.record.inverse);
      w.set(P, base, "camera", d.record.camera);
      w.set(P, base, "forward", d.record.forward);
      w.set(P, base, "info", d.record.info);
    });
    this.#device.queue.writeBuffer(this.#pieceBuffer, 0, w.buffer, 0, this.#dispatches.length * VDB_FRAME_SHADOW_PIECE_STRIDE);
  }

  /** Every map's drawn view (and its enabled flag) into its uniform buffer and the output object. */
  #writeViews(): void {
    const w = this.#viewWriter;
    const V = VDB_VOLUME_SHADOW_VIEW;
    for (const e of this.#entries.values()) {
      const cascades = e.cascades;
      const drawn: VDBVolumeShadowDrawn = cascades
        ? {
            viewProjection: cascades.slice(0, e.drawn.cascadeCount).map((c) => c.drawn.viewProjection),
            origin: cascades.slice(0, e.drawn.cascadeCount).map((c) => c.drawn.origin),
            far: cascades.slice(0, e.drawn.cascadeCount).map((c) => c.drawn.far),
            forward: e.drawn.forward,
            perspective: false,
            breaks: e.drawn.breaks,
            cascadeCount: e.drawn.cascadeCount,
          }
        : e.drawn;
      const enabled = e.enabled && e.everDrawn;
      w.clear(0, V.size);
      const matrices: number[] = [];
      for (let i = 0; i < VDB_VOLUME_SHADOW_MAX_CASCADES; i++) {
        const m = cascades ? cascades[i]!.drawn.viewProjection : (drawn.viewProjection[0] ?? new Float32Array(16));
        matrices.push(...(cascades || i === 0 ? m : new Float32Array(16)));
      }
      w.set(V, 0, "viewProjection", matrices);
      const origins: number[] = [];
      for (let i = 0; i < VDB_VOLUME_SHADOW_MAX_CASCADES; i++) {
        if (cascades) origins.push(...cascades[i]!.drawn.origin, cascades[i]!.drawn.far);
        else origins.push(...(drawn.origin[0] ?? [0, 0, 0]), drawn.far[0] ?? 1);
      }
      w.set(V, 0, "origin", origins);
      w.set(V, 0, "forward", [...drawn.forward, drawn.perspective ? 1 : 0]);
      w.set(V, 0, "breaks", drawn.breaks.slice(0, 4));
      w.set(V, 0, "info", [KIND_CODE[e.kind], drawn.cascadeCount, e.size, enabled ? 1 : 0]);
      this.#device.queue.writeBuffer(e.viewBuffer, 0, w.buffer);
      e.map.drawn = drawn;
      e.map.enabled = enabled;
      e.map.revision = e.revision;
      e.map.lastDraw = cascades ? [e.item.lastDraw, ...cascades.slice(0, e.cascadeCount).map((c) => c.lastDraw)] : [e.item.lastDraw];
    }
  }

  /** Frees every map (renderer dispose). */
  dispose(): void {
    for (const e of [...this.#entries.values()]) this.#free(e);
    this.#retire.retire(this.#pieceBuffer);
    this.#pieceBuffer = null;
  }
}
