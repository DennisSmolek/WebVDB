/**
 * The raw level-set reference pass (K6.6, design §3.9): level-set surfaces drawn by an engine-free host
 * into its own render pass, through the same `vdbLevelSetSurface` WGSL that `VDBLevelSetMaterial` calls.
 * Writes `frag_depth` (so surfaces depth-test against the host's meshes) and either a Lambert + ambient
 * shade or a G-buffer texel (world normal, ray distance).
 *
 *   const surfaces = vdbCreateLevelSetPassGpu(device, pool, { colorFormat: 'rgba8unorm', depthFormat: 'depth24plus' });
 *   // per frame, after pool.update(encoder):
 *   const pass = encoder.beginRenderPass({ colorAttachments: [...], depthStencilAttachment: { ... } });
 *   surfaces.record(pass, { view, projection, viewport: [0, 0, width, height] }, [{ grid: id, color: [1, 0.8, 0.6] }]);
 *   pass.end();
 *
 * Each draw rasterizes the back faces of the grid's index box `[min − 1.5, max + 1.5]` (the trilinear
 * support the trace clips to, half a voxel wider) mapped to world space, so the camera may be inside it;
 * the fragment ignores the box and traces the pixel-centre ray (V57). Uniforms go through
 * `queue.writeBuffer`: one `record()` per submitted encoder (the pool's contract).
 */

import { VDBError } from "../errors.js";
import { VDB_ALL_SCALAR_TYPES, type VDBGridTypeName } from "../types.js";
import type { VDBVec3 } from "../pool/layout.js";
import { vdbCheckModule, vdbModuleEntries, vdbModuleWgsl } from "../wgsl/generate.js";
import { prepareRender, renderPipeline, type VDBRenderSpec } from "../frame/pipelines.js";
import { BUFFER_USAGE, STAGE, VDBRetireQueue } from "../frame/resources.js";
import { VDBUniformWriter, vdbUniformStruct } from "../frame/uniforms.js";
import { vdbMat4Invert, vdbMat4Multiply, type VDBMat4 } from "../../tools/frame/view.js";
import { vdbLevelSetSurfaceWgsl } from "./level-set-wgsl.js";

/** What the pass reads from the pool (a `VDBGridPool` has all of it). */
export interface VDBLevelSetPassPool {
  readonly buffer: GPUBuffer;
  readonly version: number;
  readonly layoutVersion: number;
  readonly types: ReadonlySet<VDBGridTypeName>;
  info(id: number): {
    readonly indexBBox: { readonly min: readonly number[]; readonly max: readonly number[] };
    readonly indexToWorld: ArrayLike<number>;
    readonly voxelSize: readonly number[];
  };
}

/** Options of {@link vdbCreateLevelSetPassGpu}. */
export interface VDBLevelSetPassGpuOptions {
  /** Format of the one colour attachment. */
  readonly colorFormat: GPUTextureFormat;
  /** Format of the depth attachment (`frag_depth` needs one). */
  readonly depthFormat: GPUTextureFormat;
  /**
   * `'shade'` (default): `color · (ambient + lightColor · max(n · −lightDirection, 0))`, alpha 1, linear.
   * `'gbuffer'`: `(world normal xyz, ray distance t)`; use a float format (`rgba16float`, `rgba32float`).
   */
  readonly output?: "shade" | "gbuffer";
  /** Grid types the module decodes (fixed at creation; others read 0). Default: the pool's types now. */
  readonly types?: readonly VDBGridTypeName[];
  /** Compiled cap on a draw's `maxSamples`. Default 1024 (`VDB_LEVEL_SET_SAMPLES_CAP`). */
  readonly maxSamplesCap?: number;
  /** Depth test. Default `'less'` (`'greater'` for a reversed-depth host). */
  readonly depthCompare?: GPUCompareFunction;
  readonly label?: string;
}

/** The camera and lights of one {@link VDBLevelSetPassGpu.record}. Matrices are column-major. */
export interface VDBLevelSetPassFrame {
  /** World → view. */
  readonly view: VDBMat4;
  /** View → clip, WebGPU depth range (`[0, 1]`, or reversed). */
  readonly projection: VDBMat4;
  /** Default: `projection[11] === 0` (no perspective divide). */
  readonly orthographic?: boolean;
  /** The render pass's viewport in attachment pixels: `[x, y, width, height]`. */
  readonly viewport: readonly [number, number, number, number];
  /** `'shade'`: one directional light; `direction` is the way the light travels. Default `[-0.4, -0.8, -0.45]`, white. */
  readonly light?: { readonly direction: VDBVec3; readonly color: VDBVec3 } | null;
  /** `'shade'`: ambient radiance. Default `[0.15, 0.15, 0.15]`. */
  readonly ambient?: VDBVec3;
}

/** One level set to draw. */
export interface VDBLevelSetPassDraw {
  /** The grid's pool id. */
  readonly grid: number;
  /** `'shade'` albedo. Default white. */
  readonly color?: VDBVec3;
  /** Index voxels per SDF unit. Default `1 / pool.info(grid).voxelSize[0]`. */
  readonly sdfToIndex?: number;
  /** Trilinear samples per trace, clamped to `[1, maxSamplesCap]`. Default 512. */
  readonly maxSamples?: number;
  /** Central-difference step of the normal in index voxels. Default 1. */
  readonly normalStep?: number;
}

/** A level-set pass (see the file comment). */
export interface VDBLevelSetPassGpu {
  readonly device: GPUDevice;
  readonly output: "shade" | "gbuffer";
  /** Grid types the module decodes. */
  readonly types: readonly VDBGridTypeName[];
  /** The pass's whole WGSL (module + surface + entries), for inspection and size budgets. */
  readonly wgsl: string;
  /** The render pipeline's spec (cache key, layouts, state). */
  readonly spec: VDBRenderSpec;
  /** Builds the pipeline with `createRenderPipelineAsync` (optional warm-up). */
  prepare(): Promise<void>;
  /**
   * Records the draws into the host's render pass (its viewport must equal `frame.viewport`). Call after
   * `pool.update(encoder)` of this frame: the pool's bind group is rebuilt when `pool.version` moved.
   *
   * @throws VDBError `VDB_TYPE_NOT_COMPILED` when the pool holds a type the module doesn't decode;
   *   `VDB_BAD_OPTION` for a bad draw; `VDB_DISPOSED` after `dispose()`.
   */
  record(renderPass: GPURenderPassEncoder, frame: VDBLevelSetPassFrame, draws: readonly VDBLevelSetPassDraw[]): void;
  /** Destroys the buffers the pass made (never the pool, device or targets). Idempotent. */
  dispose(): void;
}

/** Group 0 binding 0: the camera and lights. */
export const VDB_LEVEL_SET_PASS_FRAME = vdbUniformStruct("VdbLevelSetPassFrame", [
  { name: "viewProjection", type: "mat4x4f" },
  { name: "projectionInverse", type: "mat4x4f" },
  { name: "cameraWorld", type: "mat4x4f" },
  { name: "viewport", type: "vec4f" },
  { name: "lightDirection", type: "vec4f" },
  { name: "lightColor", type: "vec4f" },
  { name: "ambient", type: "vec4f" },
  // x: orthographic
  { name: "flags", type: "vec4u" },
]);

/** Group 0 binding 1 (dynamic offset): one draw. */
export const VDB_LEVEL_SET_PASS_DRAW = vdbUniformStruct(
  "VdbLevelSetPassDraw",
  [
    // the proxy box's world corners, corner k = (x: k & 1, y: k & 2, z: k & 4)
    { name: "corners", type: "vec4f", count: 8 },
    { name: "color", type: "vec4f" },
    { name: "grid", type: "u32" },
    { name: "maxSamples", type: "u32" },
    { name: "sdfToIndex", type: "f32" },
    { name: "normalStep", type: "f32" },
  ],
  256,
);

/** Byte offset of draw 0 (frame record first, 256-aligned). */
const DRAWS_AT = 512;
const DRAW_STRIDE = 256;
/** Pool slot of the pass's module. */
const POOL_SLOT = { group: 1, firstBinding: 0 } as const;

/**
 * The 36 corner indices of the box's 12 triangles, counter-clockwise seen from outside in a right-handed
 * world (a unit test checks every triangle's normal points out). Culling front faces draws the back faces.
 */
export const VDB_LEVEL_SET_BOX_TRIANGLES: readonly number[] = Object.freeze([
  0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6,
]);

function bad(message: string): VDBError {
  return new VDBError("VDB_BAD_OPTION", `vdbCreateLevelSetPassGpu: ${message}`);
}

/** The entry points (after the module and the surface functions). */
function entriesWgsl(output: "shade" | "gbuffer", cap: number): string {
  const color =
    output === "gbuffer"
      ? "vec4f(s.normal, s.t)"
      : "vec4f(vdbLsDraw.color.rgb * (vdbLsFrame.ambient.rgb + vdbLsFrame.lightColor.rgb * max(dot(s.normal, -vdbLsFrame.lightDirection.xyz), 0.0)), 1.0)";
  return /* wgsl */ `${VDB_LEVEL_SET_PASS_FRAME.wgsl}
${VDB_LEVEL_SET_PASS_DRAW.wgsl}
@group(0) @binding(0) var<uniform> vdbLsFrame: VdbLevelSetPassFrame;
@group(0) @binding(1) var<uniform> vdbLsDraw: VdbLevelSetPassDraw;

@vertex
fn vdbLevelSetPassVs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    var box = array<u32, 36>(${VDB_LEVEL_SET_BOX_TRIANGLES.map((k) => `${k}u`).join(", ")});
    return vdbLsFrame.viewProjection * vec4f(vdbLsDraw.corners[box[i]].xyz, 1.0);
}

struct VdbLevelSetPassOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn vdbLevelSetPassFs(@builtin(position) position: vec4f) -> VdbLevelSetPassOut {
    let uv = vdbLevelSetPixelUv(position.xy, vdbLsFrame.viewport);
    let ray = vdbLevelSetViewRay(uv, vdbLsFrame.projectionInverse, vdbLsFrame.cameraWorld, vdbLsFrame.flags.x != 0u);
    let s = vdbLevelSetSurface(vdbLsDraw.grid, ray, vdbLsFrame.viewProjection, vdbLsDraw.sdfToIndex, clamp(vdbLsDraw.maxSamples, 1u, ${cap}u), vdbLsDraw.normalStep);
    if !s.hit || !(s.ndcDepth >= 0.0 && s.ndcDepth <= 1.0) {
        discard;
    }
    var o: VdbLevelSetPassOut;
    o.color = ${color};
    o.depth = s.ndcDepth;
    return o;
}
`;
}

/** The full WGSL of a level-set pass (module + `vdbLevelSetSurfaceWgsl()` + entries). */
export function vdbLevelSetPassWgsl(options: { types: readonly VDBGridTypeName[]; output?: "shade" | "gbuffer"; maxSamplesCap?: number }): string {
  const module = vdbModuleWgsl({ slot: POOL_SLOT, types: options.types, profile: "march", helpers: ["levelSet"] });
  return `${module}\n${vdbLevelSetSurfaceWgsl()}\n${entriesWgsl(options.output ?? "shade", options.maxSamplesCap ?? 1024)}`;
}

/** The 3x3 determinant of a 3x4 row-major affine's linear part. */
function det3(m: ArrayLike<number>): number {
  return m[0]! * (m[5]! * m[10]! - m[6]! * m[9]!) - m[1]! * (m[4]! * m[10]! - m[6]! * m[8]!) + m[2]! * (m[4]! * m[9]! - m[5]! * m[8]!);
}

/**
 * The 8 world corners of a grid's proxy box (`[min − 1.5, max + 1.5]` through the 3x4 row-major
 * index-to-world affine), `x, y, z, 1` each. A mirroring affine swaps the x corners so the triangles stay
 * counter-clockwise from outside.
 */
export function vdbLevelSetProxyCorners(info: ReturnType<VDBLevelSetPassPool["info"]>, out: Float32Array = new Float32Array(32)): Float32Array {
  const m = info.indexToWorld;
  const lo = info.indexBBox.min.map((v) => v - 1.5);
  const hi = info.indexBBox.max.map((v) => v + 1.5);
  const flip = det3(m) < 0 ? 1 : 0;
  for (let k = 0; k < 8; k++) {
    const c = k ^ flip;
    const p = [c & 1 ? hi[0]! : lo[0]!, c & 2 ? hi[1]! : lo[1]!, c & 4 ? hi[2]! : lo[2]!];
    for (let r = 0; r < 3; r++) out[k * 4 + r] = m[r * 4]! * p[0]! + m[r * 4 + 1]! * p[1]! + m[r * 4 + 2]! * p[2]! + m[r * 4 + 3]!;
    out[k * 4 + 3] = 1;
  }
  return out;
}

const DEFAULT_LIGHT: VDBVec3 = [-0.4, -0.8, -0.45];

/**
 * Creates a level-set pass on `device` for the grids of `pool` (see the file comment). The pipeline is
 * built at the first `record()` (or `prepare()`), cached per device by its WGSL and state.
 *
 * @param device The host's device (the pool's).
 * @param pool The `VDBGridPool` holding the level sets.
 * @param options Formats, output and module types ({@link VDBLevelSetPassGpuOptions}).
 * @throws VDBError `VDB_BAD_OPTION` for bad options.
 */
export function vdbCreateLevelSetPassGpu(device: GPUDevice, pool: VDBLevelSetPassPool, options: VDBLevelSetPassGpuOptions): VDBLevelSetPassGpu {
  if (!options || typeof options.colorFormat !== "string" || typeof options.depthFormat !== "string") throw bad("colorFormat and depthFormat are required.");
  const output = options.output ?? "shade";
  if (output !== "shade" && output !== "gbuffer") throw bad(`output must be 'shade' or 'gbuffer', got ${JSON.stringify(output)}.`);
  const cap = options.maxSamplesCap ?? 1024;
  if (!Number.isInteger(cap) || cap < 1) throw bad(`maxSamplesCap must be an integer >= 1, got ${JSON.stringify(options.maxSamplesCap)}.`);
  const types = [...(options.types ?? (pool.types.size > 0 ? [...pool.types] : VDB_ALL_SCALAR_TYPES))];
  const label = options.label ?? "vdb level-set pass";
  const wgsl = vdbLevelSetPassWgsl({ types, output, maxSamplesCap: cap });
  const visibility = STAGE.VERTEX | STAGE.FRAGMENT;
  const spec: VDBRenderSpec = {
    label: `${label} (${output})`,
    code: wgsl,
    vertexEntryPoint: "vdbLevelSetPassVs",
    fragmentEntryPoint: "vdbLevelSetPassFs",
    groups: [
      {
        entries: [
          { binding: 0, visibility, buffer: { type: "uniform" } },
          { binding: 1, visibility, buffer: { type: "uniform", hasDynamicOffset: true } },
        ],
      },
      { entries: vdbModuleEntries({ slot: POOL_SLOT, visibility: STAGE.FRAGMENT }) },
    ],
    targets: [{ format: options.colorFormat }],
    depthStencil: { format: options.depthFormat, depthWriteEnabled: true, depthCompare: options.depthCompare ?? "less" },
    primitive: { topology: "triangle-list", cullMode: "front", frontFace: "ccw" },
  };

  const retire = new VDBRetireQueue();
  let uniforms: GPUBuffer | null = null;
  let writer: VDBUniformWriter | null = null;
  let frameGroup: GPUBindGroup | null = null;
  let poolGroup: GPUBindGroup | null = null;
  let poolBuffer: GPUBuffer | null = null;
  let poolVersion = -1;
  let disposed = false;
  const corners = new Float32Array(32);

  const pipeline = () => renderPipeline(device, spec);

  function ensureUniforms(draws: number): void {
    const bytes = DRAWS_AT + Math.max(1, draws) * DRAW_STRIDE;
    if (uniforms && uniforms.size >= bytes) return;
    retire.retire(uniforms);
    const size = Math.max(bytes, uniforms ? uniforms.size * 2 : DRAWS_AT + 4 * DRAW_STRIDE);
    uniforms = device.createBuffer({ label: `${label} uniforms`, size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    writer = new VDBUniformWriter(size);
    frameGroup = device.createBindGroup({
      label: `${label} group 0`,
      layout: pipeline().layouts[0]!,
      entries: [
        { binding: 0, resource: { buffer: uniforms, offset: 0, size: VDB_LEVEL_SET_PASS_FRAME.size } },
        { binding: 1, resource: { buffer: uniforms, offset: 0, size: VDB_LEVEL_SET_PASS_DRAW.size } },
      ],
    });
  }

  function ensurePoolGroup(): void {
    if (poolGroup && pool.version === poolVersion && pool.buffer === poolBuffer) return;
    poolBuffer = pool.buffer;
    poolVersion = pool.version;
    poolGroup = device.createBindGroup({
      label: `${label} group 1 (pool)`,
      layout: pipeline().layouts[1]!,
      entries: [{ binding: POOL_SLOT.firstBinding, resource: { buffer: poolBuffer } }],
    });
  }

  function packFrame(w: VDBUniformWriter, frame: VDBLevelSetPassFrame): void {
    const s = VDB_LEVEL_SET_PASS_FRAME;
    if (!frame || !frame.view || !frame.projection || !frame.viewport) throw bad("record: frame needs view, projection and viewport.");
    const [, , vw, vh] = frame.viewport;
    if (!(vw > 0 && vh > 0)) throw bad(`record: viewport width and height must be > 0, got ${JSON.stringify(frame.viewport)}.`);
    const orthographic = frame.orthographic ?? frame.projection[11] === 0;
    w.set(s, 0, "viewProjection", vdbMat4Multiply(frame.projection, frame.view));
    w.set(s, 0, "projectionInverse", vdbMat4Invert(frame.projection));
    w.set(s, 0, "cameraWorld", vdbMat4Invert(frame.view));
    w.set(s, 0, "viewport", frame.viewport);
    const light = frame.light === undefined ? { direction: DEFAULT_LIGHT, color: [1, 1, 1] as VDBVec3 } : frame.light;
    if (light) {
      const d = light.direction;
      const l = Math.hypot(d[0], d[1], d[2]) || 1;
      w.set(s, 0, "lightDirection", [d[0] / l, d[1] / l, d[2] / l, 0]);
      w.set(s, 0, "lightColor", [...light.color, 0]);
    } else {
      w.set(s, 0, "lightDirection", [0, -1, 0, 0]);
      w.set(s, 0, "lightColor", [0, 0, 0, 0]);
    }
    w.set(s, 0, "ambient", [...(frame.ambient ?? [0.15, 0.15, 0.15]), 0]);
    w.set(s, 0, "flags", [orthographic ? 1 : 0, 0, 0, 0]);
  }

  function packDraw(w: VDBUniformWriter, base: number, d: VDBLevelSetPassDraw): void {
    const s = VDB_LEVEL_SET_PASS_DRAW;
    if (!d || !Number.isInteger(d.grid) || d.grid < 0) throw bad(`record: draw.grid must be a pool id, got ${JSON.stringify(d?.grid)}.`);
    const info = pool.info(d.grid);
    const sdfToIndex = d.sdfToIndex ?? 1 / info.voxelSize[0]!;
    if (!(Number.isFinite(sdfToIndex) && sdfToIndex > 0)) throw bad(`record: sdfToIndex must be a positive finite number, got ${sdfToIndex}.`);
    w.clear(base, DRAW_STRIDE);
    w.set(s, base, "corners", vdbLevelSetProxyCorners(info, corners));
    w.set(s, base, "color", [...(d.color ?? [1, 1, 1]), 1]);
    w.set(s, base, "grid", d.grid);
    w.set(s, base, "maxSamples", Math.max(1, Math.min(cap, Math.round(d.maxSamples ?? 512))));
    w.set(s, base, "sdfToIndex", sdfToIndex);
    w.set(s, base, "normalStep", d.normalStep ?? 1);
  }

  return {
    device,
    output,
    types,
    wgsl,
    spec,
    async prepare() {
      if (disposed) throw new VDBError("VDB_DISPOSED", `${label}: disposed.`);
      await prepareRender(device, spec);
    },
    record(renderPass, frame, draws) {
      if (disposed) throw new VDBError("VDB_DISPOSED", `${label}: disposed.`);
      if (draws.length === 0) return;
      vdbCheckModule(pool, { types });
      retire.advance();
      ensureUniforms(draws.length);
      const w = writer!;
      packFrame(w, frame);
      draws.forEach((d, i) => packDraw(w, DRAWS_AT + i * DRAW_STRIDE, d));
      device.queue.writeBuffer(uniforms!, 0, w.buffer, 0, DRAWS_AT + draws.length * DRAW_STRIDE);
      ensurePoolGroup();
      renderPass.setPipeline(pipeline().pipeline);
      renderPass.setBindGroup(1, poolGroup!);
      for (let i = 0; i < draws.length; i++) {
        renderPass.setBindGroup(0, frameGroup!, [DRAWS_AT + i * DRAW_STRIDE]);
        renderPass.draw(36);
      }
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      retire.drain();
      uniforms?.destroy();
      uniforms = null;
      frameGroup = null;
      poolGroup = null;
    },
  };
}
