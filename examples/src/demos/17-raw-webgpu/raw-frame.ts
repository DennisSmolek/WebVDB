/**
 * The host half of a raw-WebGPU frame around the core frame renderer (K6.1c, K6.9): no three.js, no engine. Shared by
 * demo 17 (`main.ts`) and the `k6-core` harness's `?scene=` mode (`examples/src/harness/k6-raw-scene.ts`).
 *
 * - Camera maths in the conventions the core takes: column-major matrices, WebGPU clip depth [0, 1].
 * - {@link RawTargets}: the frame's linear HDR colour (`rgba16float`), the scene-colour copy his sRGB blend reads
 *   (`targets.sceneColor`), the depth buffer the composite reads, all at the drawing-buffer size.
 * - {@link RawPresenter}: tone mapping (none or three's ACESFilmic) and the sRGB encode into the canvas
 *   (`presentation=direct`) or into an `rgba8unorm` texture read back to a 2D canvas (`presentation=readback`, for
 *   headless SwiftShader, which has no WebGPU canvas compositor). Optionally it composites the volumes itself with
 *   `vdbFrameIntegrateWgsl` (the host-side seam) instead of the core's `render()` pass.
 * - {@link readTextureRows}: a texture read back as tight rows (top row first).
 *
 * The frame loop this supports (design K6-core-renderer.md §2.4):
 *
 *   const enc = device.createCommandEncoder();
 *   pool.update(enc); source.update(enc);             // host pool, then the core source (same encoder)
 *   const u = frame.update(enc, input);               // compute: grids, shadow maps, emission, froxels
 *   ...opaque passes into targets.hdr / targets.depth, reading u.outputs (volume shadows, emission probes)
 *   targets.copySceneColor(enc); frame.render(enc, targets.composite())   // or presenter.present(enc, { integrate })
 *   presenter.present(enc, ...); device.queue.submit([enc.finish()]);
 */
import { vdbFrameIntegrateEntries, vdbFrameIntegrateWgsl, type VDBFrameOutputs, type VDBFrameTargets } from "../../../../packages/vdb/src/core/internal.js";
import { vdbMat4Invert, vdbMat4Multiply } from "../../../../packages/vdb/src/tools/internal.js";

export type Vec3 = [number, number, number];

// ------------------------------------------------------------------------------------------------ camera maths

export const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
export const add = (a: readonly number[], b: readonly number[]): Vec3 => [a[0]! + b[0]!, a[1]! + b[1]!, a[2]! + b[2]!];
export const scale = (a: readonly number[], s: number): Vec3 => [a[0]! * s, a[1]! * s, a[2]! * s];
export const dot = (a: readonly number[], b: readonly number[]): number => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
export const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
export const normalize = (v: readonly number[]): Vec3 => {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
  return [v[0]! / l, v[1]! / l, v[2]! / l];
};

/** Perspective projection, column-major, WebGPU depth: view z = −near → 0, −far → 1. */
export function perspective(fovYRadians: number, aspect: number, near: number, far: number): number[] {
  const f = 1 / Math.tan(fovYRadians / 2);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
}

/** Orthographic projection, column-major, WebGPU depth: view z = −near → 0, −far → 1. */
export function orthographic(left: number, right: number, bottom: number, top: number, near: number, far: number): number[] {
  return [2 / (right - left), 0, 0, 0, 0, 2 / (top - bottom), 0, 0, 0, 0, 1 / (near - far), 0, -(right + left) / (right - left), -(top + bottom) / (top - bottom), near / (near - far), 1];
}

/** Camera → world of a camera at `eye` looking at `target` (three's `lookAt`: −z forward, `up` kept upright). */
export function lookAt(eye: readonly number[], target: readonly number[], up: readonly number[] = [0, 1, 0]): number[] {
  const z = normalize(sub(eye, target));
  let x = cross(up, z);
  if (Math.hypot(...x) < 1e-6) x = cross([0, 0, 1], z);
  x = normalize(x);
  const y = cross(z, x);
  return [...x, 0, ...y, 0, ...z, 0, eye[0]!, eye[1]!, eye[2]!, 1];
}

export const invert = (m: ArrayLike<number>): number[] => Array.from(vdbMat4Invert(m));
export const multiply = (a: ArrayLike<number>, b: ArrayLike<number>): number[] => Array.from(vdbMat4Multiply(a, b));

/** Unit box [−½, ½]³ → world box `[lo, hi]` (column-major): a volume's `params.matrixWorld` for an axis-aligned box. */
export function boxMatrix(lo: readonly number[], hi: readonly number[]): number[] {
  return [hi[0]! - lo[0]!, 0, 0, 0, 0, hi[1]! - lo[1]!, 0, 0, 0, 0, hi[2]! - lo[2]!, 0, (lo[0]! + hi[0]!) / 2, (lo[1]! + hi[1]!) / 2, (lo[2]! + hi[2]!) / 2, 1];
}

/** The camera of one frame: matrices the core and the host passes share. */
export interface RawCamera {
  /** Camera → world. */
  world: number[];
  /** World → view (the core's `camera.view`). */
  view: number[];
  projection: number[];
  viewProjection: number[];
  projectionInverse: number[];
  position: Vec3;
  /** Unit view direction (−z of the camera). */
  forward: Vec3;
  near: number;
  far: number;
}

export function rawCamera(world: ArrayLike<number>, projection: ArrayLike<number>, near: number, far: number): RawCamera {
  const w = Array.from(world);
  const view = invert(w);
  const p = Array.from(projection);
  return {
    world: w,
    view,
    projection: p,
    viewProjection: multiply(p, view),
    projectionInverse: invert(p),
    position: [w[12]!, w[13]!, w[14]!],
    forward: normalize([-w[8]!, -w[9]!, -w[10]!]),
    near,
    far,
  };
}

// ------------------------------------------------------------------------------------------------ targets

const USAGE = { COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04, RENDER_ATTACHMENT: 0x10 } as const;

/**
 * The frame's targets at the drawing-buffer size: `hdr` (linear radiance, `rgba16float`), `sceneColor` (its copy
 * before the volumes, for the core composite's sRGB blend) and `depth` (`depth32float`, also sampled by the composite).
 * `resize` makes new textures and destroys the old ones (WebGPU defers the destroy past submitted work).
 */
export class RawTargets {
  static readonly COLOR_FORMAT: GPUTextureFormat = "rgba16float";
  static readonly DEPTH_FORMAT: GPUTextureFormat = "depth32float";
  width = 0;
  height = 0;
  hdr!: GPUTexture;
  sceneColor!: GPUTexture;
  depth!: GPUTexture;
  /** +1 per resize: rebuild bind groups that read these textures. */
  version = 0;

  constructor(
    readonly device: GPUDevice,
    readonly label = "raw frame",
  ) {}

  resize(width: number, height: number): boolean {
    if (width === this.width && height === this.height) return false;
    this.destroy();
    const d = this.device;
    const size = [width, height];
    this.hdr = d.createTexture({ label: `${this.label} hdr`, size, format: RawTargets.COLOR_FORMAT, usage: USAGE.RENDER_ATTACHMENT | USAGE.TEXTURE_BINDING | USAGE.COPY_SRC });
    this.sceneColor = d.createTexture({ label: `${this.label} scene colour`, size, format: RawTargets.COLOR_FORMAT, usage: USAGE.COPY_DST | USAGE.TEXTURE_BINDING });
    this.depth = d.createTexture({ label: `${this.label} depth`, size, format: RawTargets.DEPTH_FORMAT, usage: USAGE.RENDER_ATTACHMENT | USAGE.TEXTURE_BINDING });
    this.width = width;
    this.height = height;
    this.version++;
    return true;
  }

  /** Copies the opaque frame into `sceneColor` (his sRGB blend reads the colour under the volumes). */
  copySceneColor(encoder: GPUCommandEncoder): void {
    encoder.copyTextureToTexture({ texture: this.hdr }, { texture: this.sceneColor }, [this.width, this.height]);
  }

  /** The core composite's targets (`VDBFrameRendererGpu.render`). `clampColors`: false when the host tone maps (his rule). */
  composite(clampColors: boolean): VDBFrameTargets {
    return { color: this.hdr.createView(), size: [this.width, this.height], depth: this.depth.createView(), sceneColor: this.sceneColor.createView(), clampColors };
  }

  destroy(): void {
    this.hdr?.destroy();
    this.sceneColor?.destroy();
    this.depth?.destroy();
  }
}

// ------------------------------------------------------------------------------------------------ presentation

export type Presentation = "direct" | "readback";

/** Present uniforms: exposure, tone map (0 none, 1 ACESFilmic), integrate (0/1), near, far, srgbBlending (0/1). */
const PRESENT_FLOATS = 8;

/** The present pass: (optionally) the host-side composite, then tone mapping and the sRGB encode. */
function presentWgsl(integrate: boolean): string {
  return /* wgsl */ `${integrate ? vdbFrameIntegrateWgsl({ slot: { group: 1, firstBinding: 0 } }) : ""}
struct Present { exposure: f32, toneMap: f32, integrate: f32, near: f32, far: f32, srgbBlending: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var hdrMap: texture_2d<f32>;
@group(0) @binding(1) var<uniform> present: Present;
${integrate ? "@group(0) @binding(2) var depthMap: texture_depth_2d;" : ""}

@vertex
fn presentVs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

// three's ACESFilmicToneMapping (exposure / 0.6, RRT + ODT fit, sRGB primaries)
fn acesFilmic(colorIn: vec3f) -> vec3f {
  let ACESInputMat = mat3x3f(vec3f(0.59719, 0.07600, 0.02840), vec3f(0.35458, 0.90834, 0.13383), vec3f(0.04823, 0.01566, 0.83777));
  let ACESOutputMat = mat3x3f(vec3f(1.60475, -0.10208, -0.00327), vec3f(-0.53108, 1.10813, -0.07276), vec3f(-0.07367, -0.00605, 1.07602));
  var color = ACESInputMat * (colorIn * (present.exposure / 0.6));
  let a = color * (color + 0.0245786) - 0.000090537;
  let b = color * (0.983729 * color + 0.4329510) + 0.238081;
  color = ACESOutputMat * (a / b);
  return clamp(color, vec3f(0.0), vec3f(1.0));
}
fn srgbEncode(c: vec3f) -> vec3f {
  let x = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x < vec3f(0.0031308));
}

@fragment
fn presentFs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  var color = textureLoad(hdrMap, pixel, 0).rgb;
  ${
    integrate
      ? `// The host-side composite (design §2.5): the volumes in front of this pixel's scene depth, over the scene colour
  let size = vec2f(textureDimensions(hdrMap));
  let depth = textureLoad(depthMap, pixel, 0);
  var viewDepth = 1e20;
  if (depth < 1.0) {
    viewDepth = present.near * present.far / (present.far - (present.far - present.near) * depth);
  }
  let screenUv = vec2f(position.x / size.x, 1.0 - position.y / size.y);
  let i = vdbFrameIntegrate(screenUv, viewDepth);
  if (i.transmittance.y < 1.0) {
    if (present.srgbBlending > 0.5) {
      color = vdbFrameComposite(i, vec4f(color, 1.0), true).rgb;
    } else {
      color = i.radiance + color * i.transmittance;
    }
  }`
      : ""
  }
  if (present.toneMap > 0.5) {
    color = acesFilmic(color);
  }
  return vec4f(srgbEncode(color), 1.0);
}
`;
}

interface PresentPipeline {
  pipeline: GPURenderPipeline;
  layouts: GPUBindGroupLayout[];
}

/**
 * Tone maps `targets.hdr` into the canvas (direct) or into an `rgba8unorm` texture that `read()` copies back
 * (readback), optionally compositing the volumes itself (`integrate`: the host-side seam, `vdbFrameIntegrateWgsl`).
 */
export class RawPresenter {
  readonly format: GPUTextureFormat;
  private context: GPUCanvasContext | null = null;
  private ctx2d: CanvasRenderingContext2D | null = null;
  private output: GPUTexture | null = null;
  private readback: GPUBuffer | null = null;
  private bytesPerRow = 0;
  private readonly uniforms: GPUBuffer;
  private readonly pipelines = new Map<boolean, Promise<PresentPipeline>>();
  private readonly linear: GPUSampler;
  private placeholder3d: GPUTexture | null = null;
  /** The last read-back frame (readback presentation), sRGB RGBA bytes, top row first. */
  last: Uint8ClampedArray<ArrayBuffer> | null = null;

  constructor(
    readonly device: GPUDevice,
    readonly canvas: HTMLCanvasElement,
    readonly presentation: Presentation,
  ) {
    this.format = presentation === "direct" ? navigator.gpu.getPreferredCanvasFormat() : "rgba8unorm";
    if (presentation === "direct") {
      this.context = canvas.getContext("webgpu")!;
      this.context.configure({ device, format: this.format, alphaMode: "opaque" });
    } else {
      this.ctx2d = canvas.getContext("2d");
    }
    this.uniforms = device.createBuffer({ label: "raw present uniforms", size: PRESENT_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.linear = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
  }

  /** Builds the present pipeline for one mode (validation errors throw). */
  pipeline(integrate: boolean): Promise<PresentPipeline> {
    let p = this.pipelines.get(integrate);
    if (!p) {
      p = this.build(integrate);
      this.pipelines.set(integrate, p);
    }
    return p;
  }

  private async build(integrate: boolean): Promise<PresentPipeline> {
    const d = this.device;
    const F = GPUShaderStage.FRAGMENT;
    const layouts = [
      d.createBindGroupLayout({
        label: "raw present group 0",
        entries: [
          { binding: 0, visibility: F, texture: { sampleType: "unfilterable-float" } },
          { binding: 1, visibility: F, buffer: { type: "uniform" } },
          ...(integrate ? ([{ binding: 2, visibility: F, texture: { sampleType: "depth" } }] as GPUBindGroupLayoutEntry[]) : []),
        ],
      }),
      ...(integrate ? [d.createBindGroupLayout({ label: "raw present integrate", entries: vdbFrameIntegrateEntries({ group: 1, firstBinding: 0 }) })] : []),
    ];
    d.pushErrorScope("validation");
    const module = d.createShaderModule({ label: `raw present${integrate ? " + integrate" : ""}`, code: presentWgsl(integrate) });
    const pipeline = await d.createRenderPipelineAsync({
      label: `raw present${integrate ? " + integrate" : ""}`,
      layout: d.createPipelineLayout({ bindGroupLayouts: layouts }),
      vertex: { module, entryPoint: "presentVs" },
      fragment: { module, entryPoint: "presentFs", targets: [{ format: this.format }] },
    });
    const error = await d.popErrorScope();
    if (error) throw new Error(`raw present pipeline: ${error.message}`);
    return { pipeline, layouts };
  }

  /** Sizes the canvas (CSS size × pixel ratio in direct; CSS size in readback) and returns the drawing-buffer size. */
  resize(pixelRatio: number): { width: number; height: number } {
    const c = this.canvas;
    const ratio = this.presentation === "direct" ? pixelRatio : 1;
    const width = Math.max(1, Math.floor((c.clientWidth || window.innerWidth) * ratio));
    const height = Math.max(1, Math.floor((c.clientHeight || window.innerHeight) * ratio));
    if (c.width !== width || c.height !== height) {
      c.width = width;
      c.height = height;
    }
    if (this.presentation === "readback" && (!this.output || this.output.width !== width || this.output.height !== height)) {
      this.output?.destroy();
      this.readback?.destroy();
      this.output = this.device.createTexture({ label: "raw readback output", size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      this.bytesPerRow = Math.ceil((width * 4) / 256) * 256;
      this.readback = this.device.createBuffer({ label: "raw readback", size: this.bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    return { width, height };
  }

  /**
   * Records the present pass (and, readback, the copy to the map buffer). `integrate`: composite the volumes here
   * (`outputs` and `targets` give the integrated froxels, the depth and the scene colour) instead of `frame.render()`.
   */
  present(
    encoder: GPUCommandEncoder,
    p: PresentPipeline,
    targets: RawTargets,
    o: { exposure: number; toneMap: boolean; integrate?: { outputs: VDBFrameOutputs; near: number; far: number; srgbBlending: boolean } | null },
  ): void {
    const d = this.device;
    const integrate = o.integrate ?? null;
    d.queue.writeBuffer(this.uniforms, 0, new Float32Array([o.exposure, o.toneMap ? 1 : 0, integrate ? 1 : 0, integrate?.near ?? 0, integrate?.far ?? 1, integrate?.srgbBlending ? 1 : 0, 0, 0]));
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: targets.hdr.createView() },
      { binding: 1, resource: { buffer: this.uniforms } },
    ];
    if (integrate) entries.push({ binding: 2, resource: targets.depth.createView() });
    const group0 = d.createBindGroup({ layout: p.layouts[0]!, entries });
    const view = this.presentation === "direct" ? this.context!.getCurrentTexture().createView() : this.output!.createView();
    const pass = encoder.beginRenderPass({ label: "raw present", colorAttachments: [{ view, loadOp: "clear", clearValue: [0, 0, 0, 1], storeOp: "store" }] });
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, group0);
    if (integrate) {
      const froxels = integrate.outputs.froxels;
      this.placeholder3d ??= d.createTexture({ label: "raw present empty froxels", size: [1, 1, 1], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING });
      pass.setBindGroup(
        1,
        d.createBindGroup({
          layout: p.layouts[1]!,
          entries: [
            { binding: 0, resource: (froxels?.integrated ?? this.placeholder3d).createView({ dimension: "3d" }) },
            { binding: 1, resource: this.linear },
            { binding: 2, resource: { buffer: integrate.outputs.uniforms } },
          ],
        }),
      );
    }
    pass.draw(3);
    pass.end();
    if (this.presentation === "readback") encoder.copyTextureToBuffer({ texture: this.output! }, { buffer: this.readback!, bytesPerRow: this.bytesPerRow }, [this.output!.width, this.output!.height]);
  }

  /** Readback: maps the copied frame, draws it on the 2D canvas and keeps it in `last`. Call after the submit. */
  async show(): Promise<Uint8ClampedArray<ArrayBuffer> | null> {
    if (this.presentation !== "readback") return null;
    const { width, height } = this.output!;
    await this.readback!.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(this.readback!.getMappedRange());
    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) pixels.set(src.subarray(y * this.bytesPerRow, y * this.bytesPerRow + width * 4), y * width * 4);
    this.readback!.unmap();
    this.ctx2d!.putImageData(new ImageData(pixels, width, height), 0, 0);
    this.last = pixels;
    return pixels;
  }

  dispose(): void {
    this.context?.unconfigure();
    this.output?.destroy();
    this.readback?.destroy();
    this.uniforms.destroy();
    this.placeholder3d?.destroy();
  }
}

// ------------------------------------------------------------------------------------------------ readback

/** Reads a 2D texture back as tight rows (top row first); `texelBytes` per texel. */
export async function readTextureRows(device: GPUDevice, texture: GPUTexture, texelBytes: number): Promise<Uint8Array> {
  const w = texture.width;
  const h = texture.height;
  const row = Math.ceil((w * texelBytes) / 256) * 256;
  const buffer = device.createBuffer({ label: "raw read texture", size: row * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row }, [w, h]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buffer.getMappedRange());
  const out = new Uint8Array(w * h * texelBytes);
  for (let y = 0; y < h; y++) out.set(src.subarray(y * row, y * row + w * texelBytes), y * w * texelBytes);
  buffer.unmap();
  buffer.destroy();
  return out;
}
