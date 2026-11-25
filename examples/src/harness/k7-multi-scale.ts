/**
 * K7.3 harness: `'multi-scale'` multiple scattering on raw WebGPU (no three). The core mip build
 * (`vdbCreateMultiScaleMips`) against its CPU twin (`vdbFrameMultiScaleMips` + `vdbFrameMultiScaleToHalf`), texel
 * for texel on every mip, and the WGSL lookup (`vdbFrameMultiScaleWgsl`: `vdbFrameMultiScaleSample` /
 * `vdbFrameMultiScaleLight`) in a compute kernel against `vdbFrameMultiScaleSample` / `vdbFrameMultiScaleLight` over
 * the same f16 texels read back. Fixed seeds; seeded clouds on odd, non-cubic sizes.
 *
 * Build cases: rg16float → rg16float (edge zero and clamp), a region of a stacked grid (origin), an rgba16float
 * source with channels [2, 0] into rgba16float, one channel into r16float, an rg32float source (format-agnostic
 * input). Lookup cases: 1024 queries each on the zero chain and on the stacked region (origin, no bleed).
 * Retire check: a resize between frames, the old view still bound by a dispatch in the next frame's encoder.
 * Results: `window.__VDB_K7_MULTI_SCALE__` (e2e/k7-multi-scale.spec.ts).
 */
import {
  VDB_FRAME_MULTI_SCALE_PARAMS_BYTES,
  VDB_FRAME_MULTI_SCALE_SAMPLER,
  vdbCreateMultiScaleMips,
  vdbFrameMultiScaleParams,
  vdbFrameMultiScaleWgsl,
  type VDBMultiScaleFormat,
  type VDBMultiScaleMips,
} from "../../../packages/vdb/src/core/internal.js";
import {
  vdbFloatToHalf,
  vdbFrameMultiScaleLight,
  vdbFrameMultiScaleMips,
  vdbFrameMultiScaleSample,
  vdbFrameMultiScaleToHalf,
  vdbHalfToFloat,
  type VDBFrameMultiScaleEdge,
  type VDBFrameTexture3D,
} from "../../../packages/vdb/src/tools/internal.js";

type V3 = [number, number, number];

interface Compare {
  texels: number;
  exact: number;
  ulp1: number;
  over: number;
  maxUlp: number;
  firstBad?: { level: number; index: number; gpu: number; cpu: number };
}

interface BuildResult {
  name: string;
  format: string;
  dimensions: number[][];
  compare: Compare;
  /** Max |level value| (the chain is not all zeros). */
  peak: number;
}

interface LookupResult {
  name: string;
  queries: number;
  /** Max |gpu − cpu| of vdbFrameMultiScaleLight and the share of queries with scattered light > 1e-3. */
  light: { maxAbs: number; maxRel: number; nonZero: number; worst?: { i: number; gpu: number; cpu: number } };
  /** Max |gpu − cpu| / max(1, |cpu|) of vdbFrameMultiScaleSample (both channels, every level). */
  sample: { maxRel: number; outside: number };
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  builds: BuildResult[];
  lookups: LookupResult[];
  retire?: { version: number; errors: string[] };
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K7_MULTI_SCALE__?: Result;
  }
}

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, builds: [], lookups: [], uncaptured: [] };
window.__VDB_K7_MULTI_SCALE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

/** Mulberry32. */
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

/** A seeded cloud: a few soft blobs times noise in channel 0, a share of it in channel 1, more channels noise. */
function cloud(size: V3, channels: number, seed: number): Float64Array {
  const r = rng(seed);
  const blobs = Array.from({ length: 4 }, () => ({ c: [r() * size[0], r() * size[1], r() * size[2]], radius: (0.25 + r() * 0.3) * Math.min(...size) }));
  const data = new Float64Array(size[0] * size[1] * size[2] * channels);
  for (let z = 0; z < size[2]; z++) {
    for (let y = 0; y < size[1]; y++) {
      for (let x = 0; x < size[0]; x++) {
        let d = 0;
        for (const b of blobs) d = Math.max(d, 1 - Math.hypot(x - b.c[0]!, y - b.c[1]!, z - b.c[2]!) / b.radius);
        const i = (x + (y + z * size[1]) * size[0]) * channels;
        const ext = d > 0 ? d * 6 * (0.5 + r()) : 0;
        data[i] = ext;
        if (channels > 1) data[i + 1] = ext * 0.3 * r();
        for (let c = 2; c < channels; c++) data[i + c] = r() * 3;
      }
    }
  }
  return data;
}

const FORMAT_CHANNELS: Record<string, number> = { r16float: 1, rg16float: 2, rgba16float: 4, rg32float: 2 };

/** Uploads `values` (rounded to the format) to a new 3D texture; returns it and the values it holds. */
function upload(device: GPUDevice, format: GPUTextureFormat, size: V3, values: Float64Array): { texture: GPUTexture; held: Float64Array } {
  const channels = FORMAT_CHANNELS[format]!;
  const texture = device.createTexture({ size, dimension: "3d", format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const held = new Float64Array(values.length);
  let bytes: ArrayBufferView<ArrayBuffer>;
  if (format === "rg32float") {
    const f = Float32Array.from(values);
    held.set(f);
    bytes = f;
  } else {
    const h = new Uint16Array(values.length);
    for (let i = 0; i < values.length; i++) {
      h[i] = vdbFloatToHalf(values[i]!);
      held[i] = vdbHalfToFloat(h[i]!);
    }
    bytes = h;
  }
  const texelBytes = channels * (format === "rg32float" ? 4 : 2);
  device.queue.writeTexture({ texture }, bytes, { bytesPerRow: size[0] * texelBytes, rowsPerImage: size[1] }, size);
  return { texture, held };
}

/** Reads mip `mipLevel` of a 16-bit float 3D texture back (tight rows of `channels` halves per texel). */
async function readHalfMip(device: GPUDevice, texture: GPUTexture, mipLevel: number, size: readonly number[], channels: number): Promise<Uint16Array> {
  const [w, h, d] = size as V3;
  const rowBytes = w * channels * 2;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture, mipLevel }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Uint16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  const out = new Uint16Array(w * h * d * channels);
  const stride = bytesPerRow / 2;
  for (let r = 0; r < h * d; r++) out.set(raw.subarray(r * stride, r * stride + w * channels), r * w * channels);
  return out;
}

const ordered = (h: number) => (h & 0x8000 ? -(h & 0x7fff) : h);

function compareHalves(c: Compare, level: number, gpu: Uint16Array, cpu: Uint16Array): void {
  if (gpu.length !== cpu.length) throw new Error(`compare: ${gpu.length} GPU halves vs ${cpu.length} CPU`);
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(ordered(gpu[i]!) - ordered(cpu[i]!));
    c.texels++;
    if (d === 0) c.exact++;
    else if (d === 1) c.ulp1++;
    else {
      c.over++;
      c.firstBad ??= { level, index: i, gpu: vdbHalfToFloat(gpu[i]!), cpu: vdbHalfToFloat(cpu[i]!) };
    }
    c.maxUlp = Math.max(c.maxUlp, d);
  }
}

interface BuildCase {
  name: string;
  format: GPUTextureFormat;
  textureSize: V3;
  size: V3;
  origin?: V3;
  levels: number;
  channels?: number[];
  edge?: VDBFrameMultiScaleEdge;
  outFormat?: VDBMultiScaleFormat;
  seed: number;
}

interface Built {
  base: GPUTexture;
  held: Float64Array;
  mips: VDBMultiScaleMips;
  /** The GPU mips as f64 textures (decoded halves), level 1 first. */
  gpuMips: VDBFrameTexture3D[];
  spec: BuildCase;
}

async function build(device: GPUDevice, spec: BuildCase): Promise<Built> {
  const inChannels = FORMAT_CHANNELS[spec.format]!;
  const { texture: base, held } = upload(device, spec.format, spec.textureSize, cloud(spec.textureSize, inChannels, spec.seed));
  const mips = vdbCreateMultiScaleMips(device, {
    size: spec.size,
    levels: spec.levels,
    ...(spec.channels ? { channels: spec.channels } : {}),
    ...(spec.edge ? { edge: spec.edge } : {}),
    ...(spec.outFormat ? { format: spec.outFormat } : {}),
    label: `k7 ${spec.name}`,
  });
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder();
  mips.record(encoder, base, spec.origin ? { origin: spec.origin } : {});
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const scope = await device.popErrorScope();
  const pipeline = await mips.error;
  if (scope || pipeline) throw new Error(`${spec.name}: ${(scope ?? pipeline)!.message}`);

  const twin = vdbFrameMultiScaleMips(
    { data: held, dimensions: spec.textureSize, channels: inChannels },
    { levels: spec.levels, size: spec.size, ...(spec.origin ? { origin: spec.origin } : {}), ...(spec.channels ? { channels: spec.channels } : {}), ...(spec.edge ? { edge: spec.edge } : {}) },
  );
  const F = FORMAT_CHANNELS[mips.format]!;
  const compare: Compare = { texels: 0, exact: 0, ulp1: 0, over: 0, maxUlp: 0 };
  const gpuMips: VDBFrameTexture3D[] = [];
  let peak = 0;
  for (let k = 0; k < twin.levels.length; k++) {
    const dims = twin.layout.mips[k]!.dimensions;
    const gpu = await readHalfMip(device, mips.texture, k, dims, F);
    const cpu = vdbFrameMultiScaleToHalf(twin.levels[k]!, twin.channels, F);
    compareHalves(compare, k + 1, gpu, cpu);
    const decoded = new Float64Array(gpu.length);
    for (let i = 0; i < gpu.length; i++) {
      decoded[i] = vdbHalfToFloat(gpu[i]!);
      peak = Math.max(peak, Math.abs(decoded[i]!));
    }
    gpuMips.push({ data: decoded, dimensions: dims, channels: F });
  }
  const r: BuildResult = { name: spec.name, format: mips.format, dimensions: twin.layout.mips.map((m) => [...m.dimensions]), compare, peak };
  result.builds.push(r);
  log(
    `${spec.name} (${spec.format} → ${mips.format}, ${spec.edge ?? "zero"}): ${compare.exact}/${compare.texels} exact, ${compare.ulp1} at 1 ulp, ${compare.over} over (max ${compare.maxUlp} ulp); levels ${r.dimensions.map((d) => d.join("x")).join(", ")}`,
    compare.over === 0 ? "ok" : "fail",
  );
  return { base, held, mips, gpuMips, spec };
}

const LOOKUP_KERNEL = /* wgsl */ `
@group(0) @binding(0) var base: texture_3d<f32>;
@group(0) @binding(1) var mips: texture_3d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var<uniform> params: VdbFrameMultiScale;
@group(0) @binding(4) var<storage, read> queries: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> results: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let n = arrayLength(&queries) / 2u;
    if id.x >= n {
        return;
    }
    let a = queries[id.x * 2u];
    let b = queries[id.x * 2u + 1u];
    let light = vdbFrameMultiScaleLight(base, mips, smp, params, a.xyz, b.xyz, a.w);
    let s = vdbFrameMultiScaleSample(base, mips, smp, params, u32(b.w), a.xyz);
    results[id.x] = vec4f(light, s, 0.0);
}
`;

async function lookup(device: GPUDevice, built: Built, name: string, n: number, seed: number): Promise<VDBFrameTexture3D[]> {
  const { spec, mips } = built;
  const size = spec.size;
  const voxel: V3 = [0.5, 0.75, 0.5];
  const boxMin: V3 = [-3, 1, 2];
  const boxSize: V3 = [size[0] * voxel[0], size[1] * voxel[1], size[2] * voxel[2]];
  const grid = { boxMin, boxSize, unit: Math.max(...voxel), levels: spec.levels, multipleScattering: 0.2 };
  const params = vdbFrameMultiScaleParams({ ...grid, baseSize: size, ...(spec.origin ? { baseOrigin: spec.origin } : {}) });
  const r = rng(seed);
  const q = new Float32Array(n * 8);
  for (let i = 0; i < n; i++) {
    // 90% inside the box, 10% up to a voxel outside it.
    const world = [0, 1, 2].map((a) => boxMin[a]! + boxSize[a]! * (i % 10 === 9 ? -0.05 + r() * 1.1 : r()));
    const z = r() * 2 - 1;
    const phi = r() * 2 * Math.PI;
    const dir = [Math.sqrt(1 - z * z) * Math.cos(phi), z, Math.sqrt(1 - z * z) * Math.sin(phi)];
    if (i === 0) dir.splice(0, 3, 0, -1, 0); // the pole
    const maxDistance = i % 2 === 0 ? 1e20 : r() * 40 * grid.unit;
    q.set([world[0]!, world[1]!, world[2]!, maxDistance, dir[0]!, dir[1]!, dir[2]!, i % spec.levels], i * 8);
  }
  const module = device.createShaderModule({ label: "k7 lookup", code: `${vdbFrameMultiScaleWgsl()}\n${LOOKUP_KERNEL}` });
  device.pushErrorScope("validation");
  const pipeline = device.createComputePipeline({ label: "k7 lookup", layout: "auto", compute: { module, entryPoint: "main" } });
  const uniform = device.createBuffer({ size: VDB_FRAME_MULTI_SCALE_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniform, 0, params);
  const queries = device.createBuffer({ size: q.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(queries, 0, q);
  const out = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: built.base.createView() },
      { binding: 1, resource: mips.view },
      { binding: 2, resource: device.createSampler(VDB_FRAME_MULTI_SCALE_SAMPLER) },
      { binding: 3, resource: { buffer: uniform } },
      { binding: 4, resource: { buffer: queries } },
      { binding: 5, resource: { buffer: out } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  encoder.copyBufferToBuffer(out, 0, read, 0, n * 16);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  if (error) throw new Error(`${name} lookup: ${error.message}`);
  await read.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(read.getMappedRange().slice(0));
  read.destroy();
  for (const b of [uniform, queries, out]) b.destroy();

  const inChannels = FORMAT_CHANNELS[spec.format]!;
  const textures = {
    base: { data: built.held, dimensions: spec.textureSize, channels: inChannels },
    baseSize: size,
    ...(spec.origin ? { baseOrigin: spec.origin } : {}),
    mips: built.gpuMips,
  };
  const res: LookupResult = { name, queries: n, light: { maxAbs: 0, maxRel: 0, nonZero: 0 }, sample: { maxRel: 0, outside: 0 } };
  for (let i = 0; i < n; i++) {
    const o = i * 8;
    const world: V3 = [q[o]!, q[o + 1]!, q[o + 2]!];
    const dir: V3 = [q[o + 4]!, q[o + 5]!, q[o + 6]!];
    const cpu = vdbFrameMultiScaleLight((k, p) => vdbFrameMultiScaleSample(textures, grid, k, p), grid, world, dir, q[o + 3]!);
    const g = gpu[i * 4]!;
    const abs = Math.abs(g - cpu);
    if (abs > res.light.maxAbs) res.light.worst = { i, gpu: g, cpu };
    res.light.maxAbs = Math.max(res.light.maxAbs, abs);
    res.light.maxRel = Math.max(res.light.maxRel, abs / Math.max(cpu, 1e-3));
    if (cpu > 1e-3) res.light.nonZero++;
    const s = vdbFrameMultiScaleSample(textures, grid, q[o + 7]!, world);
    if (s[0] === 0 && s[1] === 0) res.sample.outside++;
    for (const c of [0, 1]) res.sample.maxRel = Math.max(res.sample.maxRel, Math.abs(gpu[i * 4 + 1 + c]! - s[c]!) / Math.max(1, Math.abs(s[c]!)));
  }
  result.lookups.push(res);
  log(`${name}: ${JSON.stringify(res)}`, res.light.maxAbs < 1e-3 ? "ok" : "fail");
  return built.gpuMips;
}

/** A resize between frames: the old view, still bound by a dispatch in the next encoder, must stay valid. */
async function retire(device: GPUDevice, built: Built): Promise<void> {
  const errors: string[] = [];
  device.pushErrorScope("validation");
  const { mips, base } = built;
  const old = mips.view;
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: "@group(0) @binding(0) var t: texture_3d<f32>;\n@group(0) @binding(1) var<storage, read_write> o: array<f32>;\n@compute @workgroup_size(1) fn main() { o[0] = textureLoad(t, vec3u(0u), 0).x; }" }),
      entryPoint: "main",
    },
  });
  const out = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  const oldGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: old }, { binding: 1, resource: { buffer: out } }] });
  mips.resize(built.spec.size);
  for (let frame = 0; frame < 3; frame++) {
    const encoder = device.createCommandEncoder();
    mips.record(encoder, base, built.spec.origin ? { origin: built.spec.origin } : {});
    if (frame === 0) {
      // The frame after the resize still draws with the bind group made before it.
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, oldGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const e = await device.popErrorScope();
  if (e) errors.push(e.message);
  out.destroy();
  result.retire = { version: mips.version, errors };
  log(`retire: ${JSON.stringify(result.retire)}`, errors.length === 0 ? "ok" : "fail");
}

async function main(): Promise<void> {
  root.textContent = "";
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const zero = await build(device, { name: "rg-zero", format: "rg16float", textureSize: [37, 24, 50], size: [37, 24, 50], levels: 6, seed: 1 });
  await build(device, { name: "rg-clamp", format: "rg16float", textureSize: [37, 24, 50], size: [37, 24, 50], levels: 6, edge: "clamp", seed: 1 });
  const stacked = await build(device, { name: "stacked", format: "rg16float", textureSize: [32, 32, 64], size: [32, 32, 32], origin: [0, 0, 32], levels: 5, seed: 2 });
  await build(device, { name: "rgba-select", format: "rgba16float", textureSize: [20, 28, 17], size: [20, 28, 17], levels: 4, channels: [2, 0], outFormat: "rgba16float", seed: 3 });
  await build(device, { name: "r-single", format: "rg16float", textureSize: [33, 33, 33], size: [33, 33, 33], levels: 6, channels: [1], seed: 4 });
  await build(device, { name: "rg32-source", format: "rg32float", textureSize: [24, 24, 24], size: [24, 24, 24], levels: 4, seed: 5 });

  await lookup(device, zero, "lookup rg-zero", 1024, 10);
  await lookup(device, stacked, "lookup stacked", 1024, 11);
  await retire(device, zero);
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
