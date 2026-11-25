/**
 * Point sprites (K6.6, design §3.9): the sprite size and attenuation as core WGSL
 * (`vdbPointSpriteWgsl`), which `VDBPoints` (`./three`) calls from its TSL vertex, and a raw sprite pass
 * (`vdbCreatePointPassGpu`) for engine-free hosts. The disc is the adapters' (three keeps his mipmapped
 * alpha map); the raw pass cuts an analytic disc.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/points/webgpu/VDBPoints.js`, `points/pointsCore.js`: one instanced quad per point,
 * the attenuated size in pixels with a 1-pixel minimum, the pixel scale `diameter · maxScale · zoom /
 * tan(fov / 2) · height / 2`) and three.js r187 (MIT) `PointsNodeMaterial.setupVertexSprite` (the clip-space
 * offset). Copyright (c) mjurczyk/openvdb contributors, MIT (see THIRD_PARTY_NOTICES.md). Changes: plain WGSL
 * and data; the raw pass's disc is analytic (DIVERGES.md).
 *
 * CPU twins: `tools` `pointSpriteSize`, `pointSpriteVertex`, `pointSpriteClip` (`tools/cpu/points.ts`).
 */

import { VDBError } from "../errors.js";
import type { VDBVec3 } from "../pool/layout.js";
import { prepareRender, renderPipeline, type VDBRenderSpec } from "../frame/pipelines.js";
import { BUFFER_USAGE, STAGE, VDBRetireQueue } from "../frame/resources.js";
import { VDBUniformWriter, vdbUniformStruct } from "../frame/uniforms.js";
import { vdbMat4MaxScale, vdbMat4Multiply, type VDBMat4 } from "../../tools/frame/view.js";

/**
 * The WGSL text of the sprite functions (no bindings, no module):
 *
 * ```wgsl
 * fn vdbPointSpriteSize(pixelScale: f32, viewDepth: f32, minSize: f32) -> f32;
 * fn vdbPointSpriteVertex(clipCentre: vec4f, corner: vec2f, sizePx: f32, viewportSize: vec2f) -> vec4f;
 * ```
 *
 * - `vdbPointSpriteSize`: the sprite's width in pixels, `max(pixelScale / viewDepth, minSize)` (his 1-pixel
 *   minimum: sprites under a pixel are still drawn); `viewDepth` is the point's distance along the view
 *   axis (`−view.z`).
 * - `vdbPointSpriteVertex`: a quad corner's clip position, `clipCentre + (corner · sizePx / (viewportSize / 2)
 *   · clipCentre.w, 0, 0)`; `corner` in `[−0.5, 0.5]²`, `viewportSize` in pixels. The quad is `sizePx` wide
 *   on screen whatever the depth.
 */
export function vdbPointSpriteWgsl(): string {
  return POINT_SPRITE_WGSL;
}

const POINT_SPRITE_WGSL = /* wgsl */ `fn vdbPointSpriteSize(pixelScale: f32, viewDepth: f32, minSize: f32) -> f32 {
    return max(pixelScale / viewDepth, minSize);
}

fn vdbPointSpriteVertex(clipCentre: vec4f, corner: vec2f, sizePx: f32, viewportSize: vec2f) -> vec4f {
    return clipCentre + vec4f(corner * sizePx / (viewportSize * 0.5) * clipCentre.w, 0.0, 0.0);
}
`;

/**
 * The pixel scale of {@link vdbPointSpriteSize} (pixels at view depth 1) for a point of world diameter
 * `diameter`: `diameter · maxScale(model) · projection[5] · height / 2`. For a perspective projection
 * `projection[5]` is `zoom / tan(fov / 2)`, his `getAttenuatedPointSize` · `height / 2`.
 *
 * @param diameter World diameter of a point (before the model scale).
 * @param projection Column-major view → clip.
 * @param height Viewport height in pixels.
 * @param model Column-major local → world (default identity).
 */
export function vdbPointPixelScale(diameter: number, projection: VDBMat4, height: number, model?: VDBMat4): number {
  return diameter * (model ? vdbMat4MaxScale(model) : 1) * projection[5]! * height * 0.5;
}

/** Options of {@link vdbCreatePointPassGpu}. */
export interface VDBPointPassGpuOptions {
  readonly colorFormat: GPUTextureFormat;
  /** Depth attachment format, or null for none. Default null. */
  readonly depthFormat?: GPUTextureFormat | null;
  /** Depth test when `depthFormat` is set. Default `'less'`. */
  readonly depthCompare?: GPUCompareFunction;
  readonly label?: string;
}

/** Points to draw (tools `vdbPointRenderData` returns one). */
export interface VDBPointPassData {
  /** Local positions, xyz per point. */
  readonly positions: Float32Array;
  /** Linear RGB per point (times the frame colour), or null. */
  readonly colors?: Float32Array | null;
  /** Points to draw. Default `positions.length / 3`. */
  readonly count?: number;
}

/** The camera of one {@link VDBPointPassGpu.record}. Matrices are column-major. */
export interface VDBPointPassFrame {
  readonly view: VDBMat4;
  readonly projection: VDBMat4;
  /** Local → world. Default identity. */
  readonly model?: VDBMat4;
  /** The render pass's viewport in attachment pixels: `[x, y, width, height]`. */
  readonly viewport: readonly [number, number, number, number];
  /** Pixels at view depth 1. Default {@link vdbPointPixelScale}`(diameter, projection, height, model)`. */
  readonly pixelScale?: number;
  /** World diameter for the default pixel scale. Default 1. */
  readonly diameter?: number;
  /** Smallest sprite in pixels. Default 1. */
  readonly minSize?: number;
  /** Colour (times each point's colour). Default white. */
  readonly color?: VDBVec3;
}

/** A raw point-sprite pass (see the file comment). */
export interface VDBPointPassGpu {
  readonly device: GPUDevice;
  /** Points of the last {@link setPoints}. */
  readonly count: number;
  /** The pipeline spec without (`[0]`) and with (`[1]`) the colour attribute. */
  readonly specs: readonly [VDBRenderSpec, VDBRenderSpec];
  /** Uploads the points into buffers the pass owns (the old ones are retired). */
  setPoints(data: VDBPointPassData): void;
  /** Builds both pipelines with the async creators (optional). */
  prepare(): Promise<void>;
  /** Records one instanced draw into the host's render pass (one `record()` per submitted encoder). */
  record(renderPass: GPURenderPassEncoder, frame: VDBPointPassFrame): void;
  /** Destroys the buffers the pass made. Idempotent. */
  dispose(): void;
}

/** Group 0 binding 0. */
export const VDB_POINT_PASS_FRAME = vdbUniformStruct("VdbPointPassFrame", [
  { name: "modelView", type: "mat4x4f" },
  { name: "projection", type: "mat4x4f" },
  { name: "viewport", type: "vec4f" },
  { name: "color", type: "vec4f" },
  // pixelScale, minSize
  { name: "params", type: "vec4f" },
]);

/** The quad's 6 corners (two triangles, three's sprite index order 0 1 2, 0 2 3). */
export const VDB_POINT_SPRITE_CORNERS: readonly (readonly [number, number])[] = Object.freeze([
  [-0.5, -0.5],
  [0.5, -0.5],
  [0.5, 0.5],
  [-0.5, -0.5],
  [0.5, 0.5],
  [-0.5, 0.5],
]);

/** The raw pass's WGSL (`colors`: with the per-point colour attribute). */
export function vdbPointPassWgsl(colors: boolean): string {
  const corners = VDB_POINT_SPRITE_CORNERS.map(([x, y]) => `vec2f(${x.toFixed(1)}, ${y.toFixed(1)})`).join(", ");
  return /* wgsl */ `${POINT_SPRITE_WGSL}
${VDB_POINT_PASS_FRAME.wgsl}
@group(0) @binding(0) var<uniform> vdbPtFrame: VdbPointPassFrame;

struct VdbPointPassVaryings {
    @builtin(position) position: vec4f,
    @location(0) corner: vec2f,
    @location(1) color: vec3f,
};

@vertex
fn vdbPointPassVs(@builtin(vertex_index) i: u32, @location(0) local: vec3f${colors ? ", @location(1) rgb: vec3f" : ""}) -> VdbPointPassVaryings {
    var quad = array<vec2f, 6>(${corners});
    let view = vdbPtFrame.modelView * vec4f(local, 1.0);
    let size = vdbPointSpriteSize(vdbPtFrame.params.x, -view.z, vdbPtFrame.params.y);
    var o: VdbPointPassVaryings;
    o.corner = quad[i];
    o.position = vdbPointSpriteVertex(vdbPtFrame.projection * view, o.corner, size, vdbPtFrame.viewport.zw);
    o.color = vdbPtFrame.color.rgb${colors ? " * rgb" : ""};
    return o;
}

@fragment
fn vdbPointPassFs(v: VdbPointPassVaryings) -> @location(0) vec4f {
    if dot(v.corner, v.corner) > 0.25 {
        discard;
    }
    return vec4f(v.color, 1.0);
}
`;
}

function bad(message: string): VDBError {
  return new VDBError("VDB_BAD_OPTION", `vdbCreatePointPassGpu: ${message}`);
}

/**
 * Creates a raw point-sprite pass on `device` (see the file comment): one instanced quad per point, an
 * analytic round disc, depth-tested when `depthFormat` is set.
 *
 * @throws VDBError `VDB_BAD_OPTION` for bad options or data.
 */
export function vdbCreatePointPassGpu(device: GPUDevice, options: VDBPointPassGpuOptions): VDBPointPassGpu {
  if (!options || typeof options.colorFormat !== "string") throw bad("colorFormat is required.");
  const label = options.label ?? "vdb point pass";
  const groups = [{ entries: [{ binding: 0, visibility: STAGE.VERTEX | STAGE.FRAGMENT, buffer: { type: "uniform" as const } }] }];
  const attribute = (location: number): GPUVertexBufferLayout => ({ arrayStride: 12, stepMode: "instance", attributes: [{ shaderLocation: location, offset: 0, format: "float32x3" }] });
  const spec = (colors: boolean): VDBRenderSpec => ({
    label: `${label}${colors ? " (colors)" : ""}`,
    code: vdbPointPassWgsl(colors),
    vertexEntryPoint: "vdbPointPassVs",
    fragmentEntryPoint: "vdbPointPassFs",
    groups,
    targets: [{ format: options.colorFormat }],
    buffers: colors ? [attribute(0), attribute(1)] : [attribute(0)],
    primitive: { topology: "triangle-list", cullMode: "none" },
    ...(options.depthFormat ? { depthStencil: { format: options.depthFormat, depthWriteEnabled: true, depthCompare: options.depthCompare ?? "less" } } : {}),
  });
  const specs = [spec(false), spec(true)] as const;
  const retire = new VDBRetireQueue();
  const writer = new VDBUniformWriter(VDB_POINT_PASS_FRAME.size);
  let uniforms: GPUBuffer | null = null;
  let group: GPUBindGroup | null = null;
  let positions: GPUBuffer | null = null;
  let colors: GPUBuffer | null = null;
  let count = 0;
  let disposed = false;
  const alive = () => {
    if (disposed) throw new VDBError("VDB_DISPOSED", `${label}: disposed.`);
  };
  const upload = (data: Float32Array, n: number, what: string): GPUBuffer => {
    const b = device.createBuffer({ label: `${label} ${what}`, size: Math.max(16, n * 12), usage: 0x20 /* VERTEX */ | BUFFER_USAGE.COPY_DST });
    if (n > 0) device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, n * 12);
    return b;
  };

  return {
    device,
    get count() {
      return count;
    },
    specs,
    setPoints(data) {
      alive();
      if (!data || !(data.positions instanceof Float32Array) || data.positions.length % 3 !== 0) throw bad("positions must be a Float32Array of xyz triples.");
      const n = data.count ?? data.positions.length / 3;
      if (!Number.isInteger(n) || n < 0 || n * 3 > data.positions.length) throw bad(`count ${n} does not fit ${data.positions.length / 3} positions.`);
      if (data.colors && data.colors.length < n * 3) throw bad("colors must hold count * 3 values.");
      retire.retire(positions);
      retire.retire(colors);
      positions = upload(data.positions, n, "positions");
      colors = data.colors ? upload(data.colors, n, "colors") : null;
      count = n;
    },
    async prepare() {
      alive();
      await Promise.all(specs.map((s) => prepareRender(device, s)));
    },
    record(renderPass, frame) {
      alive();
      retire.advance();
      if (!positions || count === 0) return;
      if (!frame || !frame.view || !frame.projection || !frame.viewport) throw bad("record: frame needs view, projection and viewport.");
      const [, , vw, vh] = frame.viewport;
      if (!(vw > 0 && vh > 0)) throw bad(`record: viewport width and height must be > 0, got ${JSON.stringify(frame.viewport)}.`);
      const s = VDB_POINT_PASS_FRAME;
      const modelView = frame.model ? vdbMat4Multiply(frame.view, frame.model) : frame.view;
      const pixelScale = frame.pixelScale ?? vdbPointPixelScale(frame.diameter ?? 1, frame.projection, vh, frame.model);
      writer.set(s, 0, "modelView", modelView);
      writer.set(s, 0, "projection", frame.projection);
      writer.set(s, 0, "viewport", frame.viewport);
      writer.set(s, 0, "color", [...(frame.color ?? [1, 1, 1]), 1]);
      writer.set(s, 0, "params", [pixelScale, frame.minSize ?? 1, 0, 0]);
      const built = renderPipeline(device, specs[colors ? 1 : 0]);
      if (!uniforms) {
        uniforms = device.createBuffer({ label: `${label} uniforms`, size: s.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
        group = device.createBindGroup({ label: `${label} group 0`, layout: built.layouts[0]!, entries: [{ binding: 0, resource: { buffer: uniforms } }] });
      }
      device.queue.writeBuffer(uniforms, 0, writer.buffer);
      renderPass.setPipeline(built.pipeline);
      renderPass.setBindGroup(0, group!);
      renderPass.setVertexBuffer(0, positions);
      if (colors) renderPass.setVertexBuffer(1, colors);
      renderPass.draw(6, count);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      retire.drain();
      uniforms?.destroy();
      positions?.destroy();
      colors?.destroy();
      uniforms = positions = colors = null;
      group = null;
    },
  };
}
