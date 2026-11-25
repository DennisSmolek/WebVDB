/**
 * K6.5b harness helper: synthetic host occluder maps for the k6-core probes (`&occluders=1`), one per kind of the
 * probe lights, each with its CPU copy for `vdbFrameOccluderVisibility`:
 *
 * - the directional light (id 3): a 64² `depth32float` orthographic map drawn by a depth-only render pass (a tilted
 *   quad between the light and the volumes), read back for the twin; bilinear PCF, bias −0.001;
 * - the point light (id 4): a 32² × 6 `r32float` cube (three r187's WebGPU face table) of a sphere between the light
 *   and the volumes, ray-traced on the CPU; bilinear;
 * - the spot light (id 5): a 48² `r32float` perspective map of another sphere, **reversed depth**, one texel ('nearest').
 */
import type { VDBFrameOccluder } from "../../../packages/vdb/src/core/internal.js";
import { vdbMat4Invert, type VDBFrameOccluderCpu } from "../../../packages/vdb/src/tools/internal.js";

type V3 = [number, number, number];
type R3 = readonly [number, number, number];

const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const norm = (v: readonly number[]): V3 => {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  return [v[0]! / l, v[1]! / l, v[2]! / l];
};
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** World → view of a camera at `eye` looking at `at` (three's lookAt for cameras), column-major. */
function viewOf(eye: V3, at: V3, up: V3): Float64Array {
  const z = norm(sub(eye, at));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return vdbMat4Invert([...x, 0, ...y, 0, ...z, 0, ...eye, 1]);
}
const mul = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) out[c * 4 + r] = out[c * 4 + r]! + a[k * 4 + r]! * b[c * 4 + k]!;
  return out;
};
const perspective = (fovY: number, aspect: number, near: number, far: number) => {
  const f = 1 / Math.tan(fovY / 2);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
};
const orthographic = (half: number, near: number, far: number) => [1 / half, 0, 0, 0, 0, 1 / half, 0, 0, 0, 0, 1 / (near - far), 0, 0, 0, near / (near - far), 1];
/** z → w − z: a reversed-depth projection. */
const REVERSE = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1];
const clipOf = (m: ArrayLike<number>, p: readonly number[]) => [0, 1, 2, 3].map((r) => m[r]! * p[0]! + m[4 + r]! * p[1]! + m[8 + r]! * p[2]! + m[12 + r]!);

/** Ray-traced depth of a sphere per texel of a layer (NDC depth of the hit; 1, or 0 reversed, where nothing is hit). */
function sphereLayer(viewProjection: ArrayLike<number>, w: number, h: number, center: V3, radius: number, reversed: boolean): Float32Array<ArrayBuffer> {
  const inv = vdbMat4Invert(viewProjection);
  const out = new Float32Array(w * h);
  const unproject = (x: number, y: number, z: number): V3 => {
    const c = clipOf(inv, [x, y, z]);
    return [c[0]! / c[3]!, c[1]! / c[3]!, c[2]! / c[3]!];
  };
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const nx = ((x + 0.5) / w) * 2 - 1;
      const ny = 1 - ((y + 0.5) / h) * 2;
      const near = unproject(nx, ny, reversed ? 1 : 0);
      const far = unproject(nx, ny, reversed ? 0 : 1);
      const d = norm(sub(far, near));
      const oc = sub(near, center);
      const b = dot(oc, d);
      const disc = b * b - (dot(oc, oc) - radius * radius);
      let depth = reversed ? 0 : 1;
      if (disc >= 0) {
        const t = -b - Math.sqrt(disc);
        if (t > 0) {
          const hit = [near[0] + d[0] * t, near[1] + d[1] * t, near[2] + d[2] * t];
          const c = clipOf(viewProjection, hit);
          const z = c[2]! / c[3]!;
          if (z >= 0 && z <= 1) depth = z;
        }
      }
      out[y * w + x] = depth;
    }
  return out;
}

/** three r187 PointShadowNode's WebGPU cube faces (directions and ups). */
const CUBE_DIRECTIONS: V3[] = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const CUBE_UPS: V3[] = [[0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [0, 1, 0]];

export interface SynthOccluders {
  /** The core inputs. */
  inputs: VDBFrameOccluder[];
  /** CPU copies by light id. */
  cpu: Map<number, VDBFrameOccluderCpu>;
  dispose(): void;
}

async function readDepth32(device: GPUDevice, texture: GPUTexture, w: number, h: number): Promise<Float32Array> {
  const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture, aspect: "depth-only" }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, 1]);
  device.queue.submit([enc.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const src = new Float32Array(buffer.getMappedRange().slice(0));
  buffer.unmap();
  buffer.destroy();
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) out.set(src.subarray((y * bytesPerRow) / 4, (y * bytesPerRow) / 4 + w), y * w);
  return out;
}

/** A tilted quad (NDC rect x0, y0, x1, y1; depth z0 at the left, z1 at the right) into a cleared depth32float map. */
function drawQuad(device: GPUDevice, texture: GPUTexture, rect: number[], z: [number, number]): void {
  const module = device.createShaderModule({
    code: `struct Q { rect: vec4f, z: vec4f };
@group(0) @binding(0) var<uniform> q: Q;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var c = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  let t = c[i];
  return vec4f(mix(q.rect.x, q.rect.z, t.x), mix(q.rect.y, q.rect.w, t.y), mix(q.z.x, q.z.y, t.x), 1.0);
}`,
  });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    primitive: { topology: "triangle-list" },
    depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
  });
  const uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniform, 0, new Float32Array([...rect, z[0], z[1], 0, 0]));
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: texture.createView(), depthLoadOp: "clear", depthClearValue: 1, depthStoreOp: "store" } });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uniform } }] }));
  pass.draw(6);
  pass.end();
  device.queue.submit([enc.finish()]);
  uniform.destroy();
}

/**
 * The three maps of the k6-core probe lights: directional id 3 (`direction`), point id 4 (`position`), spot id 5
 * (`position`, `direction`); the volumes sit around the origin.
 */
export async function synthOccluders(device: GPUDevice, sun: { direction: R3 }, point: { position: R3 }, spot: { position: R3; direction: R3 }): Promise<SynthOccluders> {
  const textures: GPUTexture[] = [];
  const inputs: VDBFrameOccluder[] = [];
  const cpu = new Map<number, VDBFrameOccluderCpu>();
  const f32 = (m: ArrayLike<number>) => Float32Array.from(m as ArrayLike<number>);

  // Directional: orthographic, 10 units back from the origin along the light; a tilted quad between it and the volumes
  {
    const n = 64;
    const d = norm(sun.direction);
    const eye: V3 = [-d[0] * 10, -d[1] * 10, -d[2] * 10];
    const vp = f32(mul(orthographic(2.5, 0.5, 20), viewOf(eye, [0, 0, 0], [0, 1, 0])));
    const texture = device.createTexture({ label: "k6 occluder sun", size: [n, n, 1], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    textures.push(texture);
    drawQuad(device, texture, [-0.3, -0.35, 0.45, 0.5], [0.3, 0.34]);
    const depth = await readDepth32(device, texture, n, n);
    inputs.push({ light: 3, depth: texture.createView({ dimension: "2d-array" }), viewProjection: [vp], bias: -0.001, filter: "bilinear", revision: 1 });
    cpu.set(3, { viewProjection: [vp], bias: Math.fround(-0.001), reversedDepth: false, filter: "bilinear", size: [n, n], depthAt: (_l, x, y) => depth[y * n + x]! });
  }

  // Point: a cube of 6 faces (three's WebGPU table), a sphere 40% of the way to the origin
  {
    const n = 32;
    const p: V3 = [...point.position];
    const to = norm(sub([0, 0, 0], p));
    const center: V3 = [p[0] + to[0] * 1.3, p[1] + to[1] * 1.3, p[2] + to[2] * 1.3];
    const proj = perspective(Math.PI / 2, 1, 0.1, 10);
    const faces = CUBE_DIRECTIONS.map((dir, k) => f32(mul(proj, viewOf(p, [p[0] + dir[0], p[1] + dir[1], p[2] + dir[2]], CUBE_UPS[k]!))));
    const layers = faces.map((vp) => sphereLayer(vp, n, n, center, 0.45, false));
    const texture = device.createTexture({ label: "k6 occluder point", size: [n, n, 6], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    textures.push(texture);
    const all = new Float32Array(n * n * 6);
    layers.forEach((l, k) => all.set(l, k * n * n));
    device.queue.writeTexture({ texture }, all, { bytesPerRow: n * 4, rowsPerImage: n }, [n, n, 6]);
    inputs.push({ light: 4, depth: texture.createView({ dimension: "2d-array" }), viewProjection: faces, filter: "bilinear", revision: 1 });
    cpu.set(4, { viewProjection: faces, bias: 0, reversedDepth: false, filter: "bilinear", size: [n, n], depthAt: (l, x, y) => layers[l]![y * n + x]! });
  }

  // Spot: a perspective map along the cone, reversed depth, one texel; a sphere 35% of the way to the origin
  {
    const n = 48;
    const s: V3 = [...spot.position];
    const d = norm(spot.direction);
    const to = norm(sub([0, 0, 0], s));
    const center: V3 = [s[0] + to[0] * 1.6, s[1] + to[1] * 1.6, s[2] + to[2] * 1.6];
    const vp = f32(mul(REVERSE, mul(perspective(1.3, 1, 0.5, 12), viewOf(s, [s[0] + d[0], s[1] + d[1], s[2] + d[2]], [0, 1, 0]))));
    const layer = sphereLayer(vp, n, n, center, 0.5, true) as Float32Array<ArrayBuffer>;
    const texture = device.createTexture({ label: "k6 occluder spot", size: [n, n, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    textures.push(texture);
    device.queue.writeTexture({ texture }, layer, { bytesPerRow: n * 4, rowsPerImage: n }, [n, n, 1]);
    inputs.push({ light: 5, depth: texture.createView({ dimension: "2d-array" }), viewProjection: [vp], reversedDepth: true, filter: "nearest", revision: 1 });
    cpu.set(5, { viewProjection: [vp], bias: 0, reversedDepth: true, filter: "nearest", size: [n, n], depthAt: (_l, x, y) => layer[y * n + x]! });
  }
  return { inputs, cpu, dispose: () => textures.forEach((t) => t.destroy()) };
}
