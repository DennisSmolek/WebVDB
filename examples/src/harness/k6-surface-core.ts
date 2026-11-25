/**
 * K6.6 harness helpers: the core level-set and point passes on a raw `GPUDevice` (no three objects), used by
 * `k3-levelset.ts` with `?backend=core`, and the sprite-vertex probe (core WGSL vs the CPU twin).
 */
import {
  VDBGridPool,
  vdbCreateLevelSetPassGpu,
  vdbCreatePointPassGpu,
  vdbPointSpriteWgsl,
  type VDBLevelSetPassDraw,
  type VDBLevelSetPassFrame,
  type VDBPointPassData,
  type VDBPointPassFrame,
} from "../../../packages/vdb/src/core/internal.js";
import { pointSpriteSize, pointSpriteVertex } from "../../../packages/vdb/src/tools/internal.js";

/** Reads a 2D texture back as tightly packed rows (top row first). */
export async function readTexture(device: GPUDevice, texture: GPUTexture, bytesPerPixel: number, aspect: GPUTextureAspect = "all"): Promise<ArrayBuffer> {
  const w = texture.width;
  const h = texture.height;
  const row = Math.ceil((w * bytesPerPixel) / 256) * 256;
  const buffer = device.createBuffer({ size: row * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture, aspect }, { buffer, bytesPerRow: row }, [w, h]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buffer.getMappedRange());
  const out = new Uint8Array(w * h * bytesPerPixel);
  for (let y = 0; y < h; y++) out.set(src.subarray(y * row, y * row + w * bytesPerPixel), y * w * bytesPerPixel);
  buffer.unmap();
  buffer.destroy();
  return out.buffer;
}

/** The output of one level-set pass render: colour (rgba32float) and depth (depth32float, cleared to 1). */
export interface CoreLevelSetImage {
  color: Float32Array;
  depth: Float32Array;
  /** Pipeline validation error, if any. */
  error: string | null;
}

/**
 * Renders the level-set pass into fresh `rgba32float` + `depth32float` targets of `w × h` (colour cleared
 * to 0, depth to 1) and reads both back.
 */
export async function renderLevelSetCore(
  device: GPUDevice,
  pool: VDBGridPool,
  output: "shade" | "gbuffer",
  frame: VDBLevelSetPassFrame,
  draws: readonly VDBLevelSetPassDraw[],
  w: number,
  h: number,
): Promise<CoreLevelSetImage> {
  const surfaces = vdbCreateLevelSetPassGpu(device, pool, { colorFormat: "rgba32float", depthFormat: "depth32float", output, label: `k6.6 level set (${output})` });
  await surfaces.prepare();
  const color = device.createTexture({ size: [w, h], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const depth = device.createTexture({ size: [w, h], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const encoder = device.createCommandEncoder();
  pool.update(encoder);
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: color.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }],
    depthStencilAttachment: { view: depth.createView(), depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1 },
  });
  surfaces.record(pass, frame, draws);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const image = {
    color: new Float32Array(await readTexture(device, color, 16)),
    depth: new Float32Array(await readTexture(device, depth, 4, "depth-only")),
    error: null as string | null,
  };
  color.destroy();
  depth.destroy();
  surfaces.dispose();
  return image;
}

/** Renders the point pass into an `rgba32float` target (cleared to black) with a `depth24plus` buffer. */
export async function renderPointsCore(device: GPUDevice, data: VDBPointPassData, frame: VDBPointPassFrame, w: number, h: number): Promise<{ rgba: Float32Array; count: number }> {
  const points = vdbCreatePointPassGpu(device, { colorFormat: "rgba32float", depthFormat: "depth24plus", label: "k6.6 points" });
  points.setPoints(data);
  await points.prepare();
  const color = device.createTexture({ size: [w, h], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const depth = device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: color.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
    depthStencilAttachment: { view: depth.createView(), depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1 },
  });
  points.record(pass, frame);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const rgba = new Float32Array(await readTexture(device, color, 16));
  const count = points.count;
  color.destroy();
  depth.destroy();
  points.dispose();
  return { rgba, count };
}

/** Twin mutations the sprite probe can apply (each must make the probe fail). */
export type SpriteMutation = "none" | "sprite-half" | "sprite-min" | "sprite-w";

/** A deterministic PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The sprite probe: a compute kernel runs `vdbPointSpriteVertex(clip, corner, vdbPointSpriteSize(scale,
 * depth, minSize), viewport)` over `n` random inputs (sizes from sub-pixel to hundreds of pixels, depths
 * 0.05–500, both signs of corner, w from 0.05 to 500) and the f32 twin computes the same; the error is in
 * f32 ulps of each output's magnitude.
 */
export async function spriteProbe(device: GPUDevice, mutate: SpriteMutation, n = 4096): Promise<{ compared: number; maxUlp: number; over: number; bound: number; first?: unknown }> {
  const IN = 12;
  const input = new Float32Array(n * IN);
  const r = rng(0x6b66);
  const corners = [-0.5, 0.5];
  for (let i = 0; i < n; i++) {
    const w = 0.05 + r() * 500;
    const row = [
      (r() * 2 - 1) * w, (r() * 2 - 1) * w, r() * w, w, // clip centre
      corners[i & 1]!, corners[(i >> 1) & 1]!, // corner
      r() < 0.2 ? r() * 2 : r() * 400, // pixel scale (some sub-pixel: the minimum applies)
      0.05 + r() * 500, // view depth
      r() < 0.5 ? 1 : r() * 3, // min size
      64 + Math.floor(r() * 2000), 64 + Math.floor(r() * 2000), // viewport
      0,
    ];
    input.set(row, i * IN);
  }
  const code = `${vdbPointSpriteWgsl()}
@group(0) @binding(0) var<storage, read> probeIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> probeOut: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= ${n}u {
        return;
    }
    let b = i * ${IN}u;
    let clip = vec4f(probeIn[b], probeIn[b + 1u], probeIn[b + 2u], probeIn[b + 3u]);
    let size = vdbPointSpriteSize(probeIn[b + 6u], probeIn[b + 7u], probeIn[b + 8u]);
    probeOut[i] = vdbPointSpriteVertex(clip, vec2f(probeIn[b + 4u], probeIn[b + 5u]), size, vec2f(probeIn[b + 9u], probeIn[b + 10u]));
}`;
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label: "k6.6 sprite probe", code });
  const pipeline = device.createComputePipeline({ label: "k6.6 sprite probe", layout: "auto", compute: { module, entryPoint: "main" } });
  const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, input);
  const outBuf = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, read, 0, n * 16);
  device.queue.submit([encoder.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(`sprite probe: ${err.message}`);
  await read.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  [inBuf, outBuf, read].forEach((b) => b.destroy());

  const ulp = (v: number) => {
    const a = Math.abs(Math.fround(v)) || 1e-30;
    const e = Math.floor(Math.log2(a));
    return 2 ** (e - 23);
  };
  const BOUND = 4;
  let maxUlp = 0;
  let over = 0;
  let first: unknown;
  for (let i = 0; i < n; i++) {
    const b = i * IN;
    const v = [...input.subarray(b, b + IN)];
    const clip = v.slice(0, 4);
    const corner: [number, number] = [v[4]!, v[5]!];
    const size = pointSpriteSize(v[6]!, v[7]!, mutate === "sprite-min" ? 0 : v[8]!);
    let viewport: [number, number] = [v[9]!, v[10]!];
    if (mutate === "sprite-half") viewport = [viewport[0] * 2, viewport[1] * 2];
    if (mutate === "sprite-w") clip[3] = 1;
    const twin = pointSpriteVertex(clip, corner, size, viewport);
    for (let c = 0; c < 4; c++) {
      const g = gpu[i * 4 + c]!;
      // The error in ulps of the larger of the output and the clip w (|x|, |y| <= w: the operands' scale).
      const scale = Math.max(Math.abs(twin[c]!), Math.abs(v[3]!));
      const e = Math.abs(g - twin[c]!) / ulp(scale);
      if (e > maxUlp) maxUlp = e;
      if (e > BOUND) {
        over++;
        first ??= { i, c, gpu: g, twin: twin[c], input: v };
      }
    }
  }
  return { compared: n * 4, maxUlp, over, bound: BOUND, ...(first ? { first } : {}) };
}
