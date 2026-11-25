// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameEmissionLight.js L138–287 (emitters, keys, slots, updateProbeGrid,
//   prepareProbes); src/three-openvdb/frame/webgpu/FrameEmissionLight.js L540–658 (targets, updateProbes);
//   src/three-openvdb/frame/core/frameBlackbody.js (the 'planck' / 'soot' table, via tools/frame/blackbody.ts).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free (K6.5, design §3.8): emission colour as data (LUT rows) plus a validated user WGSL hook; the
// emitter keys and probe cascades of his light, run by the core renderer on its volume records. Divergences:
// DIVERGES.md (K6.5 rows).

/**
 * The CPU half of the core renderer's emission (K6.5):
 *
 * - Colour: each volume's `emissionColor` becomes a record code (white, a row of the LUT texture, or the user hook).
 *   Rows are made once per distinct colour (his blackbody table pre-blended to the soot exponent, or the caller's LUT).
 * - Hooks: `vdbEmissionHookProblems` (names, reserved words, back-ticks, bindings) and `vdbValidateEmissionHook`
 *   (compiles under a validation scope) give `VDBError('VDB_BAD_WGSL')` naming the line.
 * - Probes: his emitter keys and power-of-two slots, the probe cascades (tools/frame `vdbFrameEmissionProbeGrid`), the
 *   emitter list and probe grid uniforms, the direction / emitter / probe textures.
 */

import { VDBError } from "../errors.js";
import { WGSL_RESERVED_WORDS } from "../wgsl/reserved.js";
import { vdbFrameEmissionLutRow, vdbFrameEmissionSpectrum } from "../../tools/frame/blackbody.js";
import { vdbFrameEmissionProbeGrid, vdbFrameEmitterBounds } from "../../tools/frame/emission.js";
import { vdbFrameStackLayout, type VDBFrameStackLayout } from "../../tools/frame/stack.js";
import { vdbHashText } from "./pipelines.js";
import { BUFFER_USAGE, TEXTURE_USAGE, type VDBRetireQueue } from "./resources.js";
import type { VDBEmissionColor, VDBFrameEmissionLightSettings, VDBFrameEmissionProbes, VDBFrameVec3 } from "./types.js";
import { VDBUniformWriter } from "./uniforms.js";
import {
  VDB_EMISSION_GRID,
  VDB_FRAME_EMISSION_DIRECTIONS,
  VDB_FRAME_EMISSION_LUT_SIZE,
  VDB_FRAME_EMISSION_MAX_CASCADES,
  VDB_FRAME_EMITTERS,
  VDB_FRAME_MAX_EMITTERS,
} from "./wgsl/emission.js";

type V3 = [number, number, number];

// ------------------------------------------------------------------ hooks

/** The line (1-based) of character `at` in `text`. */
const lineOf = (text: string, at: number) => text.slice(0, at).split("\n").length;

/**
 * Static checks of a user emission hook (design §3.8): it defines `fn vdbUserEmissionColor(`, every top-level name
 * starts with `vdbUser`, no reserved word is a name, no back-ticks, no bindings, entry points or directives.
 * Returns the first problem with its line, or null.
 */
export function vdbEmissionHookProblems(wgsl: string): string | null {
  if (typeof wgsl !== "string" || wgsl.trim() === "") return "the hook is empty";
  const tick = wgsl.indexOf("`");
  if (tick >= 0) return `line ${lineOf(wgsl, tick)}: back-ticks are not allowed`;
  // strip comments (keep line breaks so lines stay right)
  const code = wgsl.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " ")).replace(/\/\/[^\n]*/g, (m) => " ".repeat(m.length));
  for (const m of code.matchAll(/@(group|binding|compute|vertex|fragment|workgroup_size)\b|\b(enable|requires|diagnostic)\b/g)) {
    return `line ${lineOf(code, m.index!)}: '${m[0]}' is not allowed in a hook (no bindings, entry points or directives)`;
  }
  // top-level declarations (brace depth 0) must be vdbUser*
  let depth = 0;
  for (let i = 0; i < code.length; i++) {
    const c = code[i]!;
    if (c === "{") depth++;
    else if (c === "}") depth = Math.max(0, depth - 1);
    else if (depth === 0) {
      const m = /^(fn|struct|const|var|alias|override)\b(?:<[^>]*>)?\s+([A-Za-z_]\w*)/.exec(code.slice(i));
      if (m && (i === 0 || /[\s;}]/.test(code[i - 1]!))) {
        if (!m[2]!.startsWith("vdbUser")) return `line ${lineOf(code, i)}: top-level name '${m[2]}' must start with vdbUser`;
        i += m[0].length - 1;
      }
    }
  }
  for (const m of code.matchAll(/\b(?:let|var|fn|const|struct|alias)\s*(?:<[^>]*>)?\s+([A-Za-z_]\w*)|\b([A-Za-z_]\w*)\s*:\s*[A-Za-z_]/g)) {
    const name = m[1] ?? m[2]!;
    if (WGSL_RESERVED_WORDS.has(name)) return `line ${lineOf(code, m.index!)}: '${name}' is a reserved word`;
  }
  if (!/\bfn\s+vdbUserEmissionColor\s*\(/.test(code)) return "the hook must define fn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f";
  return null;
}

const validations = new WeakMap<GPUDevice, Map<string, Promise<void>>>();

/**
 * Validates a user emission hook on `device` once (cached per text): {@link vdbEmissionHookProblems}, then
 * `createShaderModule` + `getCompilationInfo` under `pushErrorScope('validation')` with a test entry that calls
 * `vdbUserEmissionColor(0.5, 1000.0)` as a `vec3f`.
 *
 * @throws VDBError `VDB_BAD_WGSL` naming the line (of the hook text).
 */
export function vdbValidateEmissionHook(device: GPUDevice, wgsl: string): Promise<void> {
  let byText = validations.get(device);
  if (!byText) validations.set(device, (byText = new Map()));
  const hit = byText.get(wgsl);
  if (hit) return hit;
  const problem = vdbEmissionHookProblems(wgsl);
  const job = problem
    ? Promise.reject(new VDBError("VDB_BAD_WGSL", `emission colour hook: ${problem}.`))
    : (async () => {
        const lines = wgsl.split("\n").length;
        device.pushErrorScope("validation");
        const module = device.createShaderModule({
          label: "vdb frame emission hook check",
          code: `${wgsl}\n@compute @workgroup_size(1, 1, 1)\nfn vdbUserHookCheckCs() {\n  let color: vec3f = vdbUserEmissionColor(0.5, 1000.0);\n  _ = color;\n}\n`,
        });
        const info = typeof module.getCompilationInfo === "function" ? await module.getCompilationInfo() : { messages: [] as GPUCompilationMessage[] };
        const scope = await device.popErrorScope();
        const error = info.messages.find((m) => m.type === "error");
        if (error) {
          const where = error.lineNum > lines ? `the hook's call vdbUserEmissionColor(f32, f32) -> vec3f` : `line ${error.lineNum}`;
          throw new VDBError("VDB_BAD_WGSL", `emission colour hook, ${where}: ${error.message}`);
        }
        if (scope) throw new VDBError("VDB_BAD_WGSL", `emission colour hook: ${scope.message}`);
      })();
  job.catch(() => {});
  byText.set(wgsl, job);
  return job;
}

// ------------------------------------------------------------------ colour LUT

/** The colour code of a volume: (mode, row) and the hook text when the mode is the hook. */
export interface VDBEmissionColorCode {
  readonly code: number;
  readonly hook: string | null;
}

/** The emission colour LUT texture (rgba32float, 256 × rows) and the hooks of one renderer. */
export class VDBFrameEmissionColors {
  /** +1 when `texture` is a new object (rebuild bind groups). */
  version = 0;
  #texture: GPUTexture;
  #capacity = 4;
  readonly #rows = new Map<string, number>();
  readonly #data: Float32Array<ArrayBuffer>[] = [];
  readonly #lutKeys = new WeakMap<object, string>();
  readonly #dirty = new Set<number>();
  readonly #hooks = new Map<string, "pending" | "valid" | VDBError>();
  readonly #device: GPUDevice;
  readonly #label: string;
  readonly #retire: VDBRetireQueue;

  constructor(device: GPUDevice, label: string, retire: VDBRetireQueue) {
    this.#device = device;
    this.#label = label;
    this.#retire = retire;
    this.#texture = this.#make(this.#capacity);
  }

  #make(rows: number): GPUTexture {
    return this.#device.createTexture({
      label: `${this.#label} emission colour LUT`,
      size: [VDB_FRAME_EMISSION_LUT_SIZE, rows],
      format: "rgba32float",
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    });
  }

  /** The LUT texture after this update's new rows (uploaded here). */
  texture(): GPUTexture {
    if (this.#data.length > this.#capacity) {
      while (this.#capacity < this.#data.length) this.#capacity *= 2;
      this.#retire.retire(this.#texture);
      this.#texture = this.#make(this.#capacity);
      this.version++;
      this.#data.forEach((_, i) => this.#dirty.add(i));
    }
    for (const row of this.#dirty) {
      this.#device.queue.writeTexture({ texture: this.#texture, origin: [0, row, 0] }, this.#data[row]!, { bytesPerRow: VDB_FRAME_EMISSION_LUT_SIZE * 16 }, [VDB_FRAME_EMISSION_LUT_SIZE, 1]);
    }
    this.#dirty.clear();
    return this.#texture;
  }

  #row(key: string, rgb: () => ArrayLike<number>): number {
    let row = this.#rows.get(key);
    if (row === undefined) {
      const values = rgb();
      const rgba = new Float32Array(VDB_FRAME_EMISSION_LUT_SIZE * 4);
      for (let i = 0; i < VDB_FRAME_EMISSION_LUT_SIZE; i++) {
        rgba[i * 4] = values[i * 3]!;
        rgba[i * 4 + 1] = values[i * 3 + 1]!;
        rgba[i * 4 + 2] = values[i * 3 + 2]!;
        rgba[i * 4 + 3] = 1;
      }
      row = this.#data.length;
      this.#data.push(rgba);
      this.#rows.set(key, row);
      this.#dirty.add(row);
    }
    return row;
  }

  /**
   * The record code of a colour (registers a LUT row, starts a hook's validation). A hook still validating draws white
   * this frame (an invalid pipeline would invalidate the host's command buffer); a failed one throws.
   */
  codeOf(color: VDBEmissionColor | undefined): VDBEmissionColorCode {
    if (!color || color === "white") return { code: 0, hook: null };
    if (typeof color === "object" && "wgsl" in color) {
      const state = this.#hookState(color.wgsl);
      if (state instanceof VDBError) throw state;
      return state === "valid" ? { code: 2, hook: color.wgsl } : { code: 0, hook: null };
    }
    if (typeof color === "object" && "spectrum" in color) {
      const [, exponent] = vdbFrameEmissionSpectrum(color.spectrum, color.exponent ?? 1.39);
      return { code: 1 | (this.#row(`spectrum:${exponent}`, () => vdbFrameEmissionLutRow(exponent)) << 2), hook: null };
    }
    if (typeof color === "object" && "lut" in color) {
      const lut = color.lut;
      if (!lut || lut.length !== VDB_FRAME_EMISSION_LUT_SIZE * 3) {
        throw new VDBError("VDB_BAD_OPTION", `emissionColor.lut: pass ${VDB_FRAME_EMISSION_LUT_SIZE} × rgb values (${VDB_FRAME_EMISSION_LUT_SIZE * 3} numbers) over log temperature 500–40,000 K.`);
      }
      // the LUT's content key, once per array object (a LUT is data: callers pass the same array each frame)
      let key = typeof lut === "object" ? this.#lutKeys.get(lut) : undefined;
      if (key === undefined) {
        key = `lut:${vdbHashText(Array.from(lut, (v) => v.toPrecision(9)).join(","))}`;
        this.#lutKeys.set(lut, key);
      }
      return { code: 1 | (this.#row(key, () => lut) << 2), hook: null };
    }
    throw new VDBError("VDB_BAD_OPTION", "emissionColor: pass 'white', { spectrum: 'planck' | 'soot', exponent }, { lut } or { wgsl }.");
  }

  #hookState(text: string): "pending" | "valid" | VDBError {
    let state = this.#hooks.get(text);
    if (state === undefined) {
      const problem = vdbEmissionHookProblems(text);
      if (problem) throw new VDBError("VDB_BAD_WGSL", `emission colour hook: ${problem}.`);
      this.#hooks.set(text, (state = "pending"));
      vdbValidateEmissionHook(this.#device, text).then(
        () => this.#hooks.set(text, "valid"),
        (e: unknown) => this.#hooks.set(text, e instanceof VDBError ? e : new VDBError("VDB_BAD_WGSL", `emission colour hook: ${String(e)}`)),
      );
    }
    return state;
  }

  /** Waits for every hook seen so far to validate (`prepare()`); throws the first failure. */
  async settled(colors: readonly (VDBEmissionColor | undefined)[] = []): Promise<void> {
    for (const c of colors) if (c && typeof c === "object" && "wgsl" in c) this.#hookState(c.wgsl);
    await Promise.all([...this.#hooks.keys()].map((text) => vdbValidateEmissionHook(this.#device, text)));
  }

  dispose(): void {
    this.#retire.retire(this.#texture);
  }
}

// ------------------------------------------------------------------ probes

/** One emitting volume of an update (a volume record with an emission channel that is drawn). */
export interface VDBFrameEmitterInput {
  /** The volume record index this update. */
  readonly record: number;
  /** The record's source group index (direction passes run per source group). */
  readonly group: number;
  /** Unit box → world. */
  readonly matrixWorld: ArrayLike<number>;
  /** Changes when the volume's direction pass would draw differently. */
  readonly key: string;
}

/** The probe work of one update. */
export interface VDBFrameEmissionPlan {
  /** Per source group index: the first item of the emitter list and its count (direction passes). */
  readonly groups: ReadonlyMap<number, { first: number; count: number }>;
  readonly reduce: boolean;
  readonly probes: boolean;
}

/** The emission probes of one renderer (his `FrameEmissionLight`, one per renderer; its `outputs` are `VDBFrameEmissionProbes`). */
export class VDBFrameEmissionProbeSet {
  /** The renderer's `outputs.emissionProbes` (null until an update with emitters and an emission light). */
  outputs: { -readonly [K in keyof VDBFrameEmissionProbes]: VDBFrameEmissionProbes[K] } | null = null;
  /** +1 when a probe texture or buffer is a new object. */
  version = 0;
  readonly #device: GPUDevice;
  readonly #label: string;
  readonly #retire: VDBRetireQueue;
  #slotCount = 0;
  #slotKeys: string[] = [];
  #probeKey = "";
  #directions: GPUTexture | null = null;
  #emitters: GPUTexture | null = null;
  #probes: GPUTexture | null = null;
  #layout: VDBFrameStackLayout | null = null;
  #resolution = 0;
  readonly #listBuffer: GPUBuffer;
  readonly #listWriter = new VDBUniformWriter(VDB_FRAME_EMITTERS.size);
  readonly #gridBuffer: GPUBuffer;
  readonly #gridWriter = new VDBUniformWriter(VDB_EMISSION_GRID.size);

  constructor(device: GPUDevice, label: string, retire: VDBRetireQueue) {
    this.#device = device;
    this.#label = label;
    this.#retire = retire;
    this.#listBuffer = device.createBuffer({ label: `${label} emitter list`, size: VDB_FRAME_EMITTERS.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#gridBuffer = device.createBuffer({ label: `${label} emission grid`, size: VDB_EMISSION_GRID.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
  }

  get listBuffer(): GPUBuffer {
    return this.#listBuffer;
  }
  get gridBuffer(): GPUBuffer {
    return this.#gridBuffer;
  }
  get directionTexture(): GPUTexture | null {
    return this.#directions;
  }
  get emitterTexture(): GPUTexture | null {
    return this.#emitters;
  }
  get probeTexture(): GPUTexture | null {
    return this.#probes;
  }
  /** Slots of the direction and emitter textures (rows). */
  get slots(): number {
    return this.#slotCount;
  }

  /**
   * His prepareProbes + updateProbeGrid: the emitters' slots and keys, the probe cascades around the emitters and the
   * camera; writes the emitter list and the grid. Returns null when nothing is drawn this update.
   */
  plan(settings: VDBFrameEmissionLightSettings | null | undefined, emitters: readonly VDBFrameEmitterInput[], cameraPosition: V3, steps: number): VDBFrameEmissionPlan | null {
    if (!settings) {
      this.#release();
      return null;
    }
    const list = emitters.slice(0, VDB_FRAME_MAX_EMITTERS);
    const count = list.length;
    if (count > this.#slotCount) {
      this.#slotCount = 2 ** Math.ceil(Math.log2(count));
      this.#slotKeys = [];
      this.#retire.retire(this.#directions);
      this.#retire.retire(this.#emitters);
      const make = (width: number, name: string) =>
        this.#device.createTexture({ label: `${this.#label} emission ${name}`, size: [width, this.#slotCount], format: "rgba32float", usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_SRC });
      this.#directions = make(2 * VDB_FRAME_EMISSION_DIRECTIONS, "directions");
      this.#emitters = make(2, "emitters");
      this.version++;
    }
    const resolution = Math.min(32, Math.max(2, Math.round(settings.resolution ?? 16)));
    if (!this.#probes || resolution !== this.#resolution) {
      this.#retire.retire(this.#probes);
      this.#resolution = resolution;
      this.#layout = vdbFrameStackLayout(resolution, 3 * VDB_FRAME_EMISSION_MAX_CASCADES, this.#device.limits?.maxTextureDimension3D ?? 2048);
      this.#probes = this.#device.createTexture({
        label: `${this.#label} emission probes`,
        size: this.#layout.size,
        dimension: "3d",
        format: "rgba16float",
        usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_SRC,
      });
      this.#probeKey = "";
      this.version++;
    }

    const changed: { item: VDBFrameEmitterInput; slot: number }[] = [];
    list.forEach((e, i) => {
      if (this.#slotKeys[i] !== e.key) {
        this.#slotKeys[i] = e.key;
        changed.push({ item: e, slot: i });
      }
    });

    const boxes = list.map((e) => vdbFrameEmitterBounds(e.matrixWorld));
    const grid =
      count > 0
        ? vdbFrameEmissionProbeGrid(
            { resolution, range: settings.range ?? 3, cascades: settings.cascades ?? 1, distance: settings.distance ?? 512 },
            boxes.map((b) => ({ min: b.min, max: b.max })),
            cameraPosition,
          )
        : null;
    const lastUpdate = { directions: 0, probes: false };
    if (!grid) {
      this.#probeKey = "";
      this.#writeGrid(null, 0);
      this.#publish(null, 0, lastUpdate);
      return null;
    }
    const probeKey = `${count},${grid.key}`;
    if (changed.length === 0 && probeKey === this.#probeKey) {
      this.#publish(grid, count, lastUpdate);
      return null;
    }
    this.#probeKey = probeKey;

    // The emitter list: changed emitters by source group (direction passes run per group)
    changed.sort((a, b) => a.item.group - b.item.group);
    const groups = new Map<number, { first: number; count: number }>();
    const w = this.#listWriter;
    const L = VDB_FRAME_EMITTERS;
    w.clear(0, L.size);
    w.set(L, 0, "header", [changed.length, Math.max(1, Math.round(steps)), this.#slotCount, 0]);
    const info: number[] = [];
    const sphere: number[] = [];
    changed.forEach(({ item, slot }, i) => {
      const g = groups.get(item.group);
      if (g) g.count++;
      else groups.set(item.group, { first: i, count: 1 });
      const b = boxes[list.indexOf(item)]!;
      info.push(item.record, slot, 0, 0);
      sphere.push(...b.center, b.radius);
    });
    w.set(L, 0, "info", info);
    w.set(L, 0, "sphere", sphere);
    this.#device.queue.writeBuffer(this.#listBuffer, 0, w.buffer);
    this.#writeGrid(grid, count);
    lastUpdate.directions = changed.length;
    lastUpdate.probes = true;
    this.#publish(grid, count, lastUpdate, true);
    return { groups, reduce: changed.length > 0, probes: true };
  }

  #writeGrid(grid: ReturnType<typeof vdbFrameEmissionProbeGrid> | null, emitters: number): void {
    const w = this.#gridWriter;
    const G = VDB_EMISSION_GRID;
    w.clear(0, G.size);
    if (grid) {
      w.set(G, 0, "gridMin", grid.min.flatMap((m) => [...m, 0]));
      w.set(G, 0, "gridSize", grid.size.flatMap((s) => [...s, 0]));
      w.set(G, 0, "center", [...grid.center, emitters > 0 ? 1 : 0]);
      w.set(G, 0, "info", [grid.count, grid.resolution, emitters, 0]);
    }
    w.set(G, 0, "origin", this.#layout!.origins.flatMap((o) => [...o, 0]));
    this.#device.queue.writeBuffer(this.#gridBuffer, 0, w.buffer);
  }

  #publish(grid: ReturnType<typeof vdbFrameEmissionProbeGrid> | null, emitters: number, lastUpdate: { directions: number; probes: boolean }, drawn = false): void {
    const layout = this.#layout!;
    const prev = this.outputs;
    this.outputs = {
      texture: this.#probes!,
      grid: this.#gridBuffer,
      layout: { size: layout.size, origins: layout.origins },
      state: grid
        ? { count: grid.count, resolution: grid.resolution, min: grid.min.map((m) => [...m] as VDBFrameVec3), size: grid.size.map((s) => [...s] as VDBFrameVec3), center: [...grid.center] as VDBFrameVec3, emitters }
        : { count: 0, resolution: this.#resolution, min: [], size: [], center: [0, 0, 0], emitters: 0 },
      revision: (prev?.revision ?? 0) + (drawn ? 1 : 0),
      lastUpdate,
    };
  }

  #release(): void {
    if (!this.outputs && !this.#probes) return;
    this.#retire.retire(this.#directions);
    this.#retire.retire(this.#emitters);
    this.#retire.retire(this.#probes);
    this.#directions = this.#emitters = this.#probes = null;
    this.#slotCount = 0;
    this.#slotKeys = [];
    this.#probeKey = "";
    this.#resolution = 0;
    this.outputs = null;
    this.version++;
  }

  dispose(): void {
    this.#release();
    this.#retire.retire(this.#listBuffer);
    this.#retire.retire(this.#gridBuffer);
  }
}
