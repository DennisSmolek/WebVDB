/**
 * K6.1 harness: the core frame renderer on raw WebGPU (no three). Each probe runs one kernel on seeded
 * synthetic data and compares its texels with the CPU twin (tools/frame, f64), within 2 f16 ulp:
 *
 * - `?probe=view`: the source-free WGSL chunks (froxel view maths, noise, fade, box, lod, phase, shadow
 *   maths, slice coordinate, two-tap integrate, sRGB composite) vs their twins.
 * - `?probe=gather` (`&mix=mixture|over`, `&groups=1|2`, `&mips=0|1`): `update()` on two overlapping
 *   synthetic dense volumes (albedo, mask and emission channels; every light kind) → the scatter texture vs
 *   `gatherTwin` (his inject pass on the CPU dense sampler + `vdbFrameGatherFroxel`). `&swap=1`: counts
 *   `create*Pipeline*` / `createShaderModule` calls in a second identical update (must be 0).
 * - `?probe=resolve` (`&temporal=0|1`): the resolve kernel on a synthetic scatter (and history) vs
 *   `vdbFrameTemporalFroxel` + `vdbFrameScanColumn`.
 * - `?probe=integrate`: the composite pass on a synthetic prefix and r32float depth vs
 *   `vdbFrameSliceCoord` + `vdbFrameIntegratePrefix`.
 * - `?probe=compile`: every K6.1 pipeline variant compiles (`getCompilationInfo`, validation scope).
 * - `?mutate=<term>`: flips one twin term (`mix`, `phase`, `extinction`, `fade`, `scan-stop`,
 *   `temporal-clamp`, `coverage`, `taps`); the probe must then fail.
 *
 * Results: `window.__VDB_K6_CORE__` (e2e/k6-core.spec.ts).
 */
import {
  CHUNK_PROBE_INPUTS,
  CHUNK_PROBE_OUTPUTS,
  VDB_FRAME_LIGHTS,
  VDB_FRAME_UNIFORMS,
  VDBUniformWriter,
  computePipeline,
  renderPipeline,
  vdbCreateFrameRenderer,
  vdbDenseSource,
  vdbFrameChunkProbeSpec,
  vdbFrameCompositeSpec,
  vdbFrameGatherSpec,
  vdbFrameRendererTextures,
  vdbFrameResolveSpec,
  type VDBFrameLight,
  type VDBFrameVolumeParams,
} from "../../../packages/vdb/src/core/internal.js";
import {
  vdbFloatToHalf,
  vdbFrameAtlasLayout,
  vdbFrameClipCamera,
  vdbFrameClipVolume,
  vdbFrameDistanceFade,
  vdbFrameFroxelRect,
  vdbFrameHenyeyGreenstein,
  vdbFrameIntegratePrefix,
  vdbFrameIsotropicDiffusion,
  vdbFrameNoise,
  vdbFramePhaseColor,
  vdbFrameScanColumn,
  vdbFrameShadowDepth,
  vdbFrameShadowStep,
  vdbFrameShadowTransmittance,
  vdbFrameSliceCoord,
  vdbFrameSliceDepth,
  vdbFrameSrgbBlend,
  vdbFrameViewPosition,
  vdbFrameVolumeSlices,
  vdbHalfToFloat,
  vdbMat4Invert,
  vdbMat4MaxScale,
  wvdbMipAtlas,
  type VDBFrameVec3,
  type VDBFrameVec4,
  type WvdbDenseTexture,
  type WvdbDenseVolume,
} from "../../../packages/vdb/src/tools/internal.js";
import { gatherTwin, integrateTwin, resolveTwin, vdbFrameReproject, type TwinFrame, type TwinMutation, type TwinVolume } from "./k6-core-twin.js";

interface Stats {
  compared: number;
  covered: number;
  over: number;
  maxUlp: number;
  maxAbs: number;
  bound: string;
  firstBad?: { at: number[]; gpu: number[]; cpu: number[] };
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  probe?: string;
  mutate?: string;
  stats?: Stats;
  /** `?probe=gather&swap=1`: builds in the second identical update. */
  builds?: { first: number; second: number };
  compile?: { variants: number; errors: string[] };
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K6_CORE__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const PROBE = params.get("probe") ?? "view";
const MUTATE = (params.get("mutate") ?? "none") as TwinMutation;
const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [], probe: PROBE, mutate: MUTATE };
window.__VDB_K6_CORE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

// ------------------------------------------------------------------ comparison

/** One f16 ulp at |x| (2^-24 in the subnormal range). */
const ulp16 = (x: number) => (Math.abs(x) < 2 ** -14 ? 2 ** -24 : 2 ** (Math.floor(Math.log2(Math.abs(x))) - 10));

function compare(stats: Stats, at: number[], gpu: readonly number[], cpu: readonly number[], ulps = 2, absFloor = 2 ** -20): void {
  stats.compared++;
  if (cpu.some((v) => v !== 0) || gpu.some((v) => v !== 0)) stats.covered++;
  let bad = false;
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(gpu[i]! - cpu[i]!);
    const u = d / ulp16(Math.max(Math.abs(cpu[i]!), Math.abs(gpu[i]!)));
    stats.maxAbs = Math.max(stats.maxAbs, d);
    if (Number.isFinite(u)) stats.maxUlp = Math.max(stats.maxUlp, u);
    if (!(d <= Math.max(ulps * ulp16(Math.max(Math.abs(cpu[i]!), Math.abs(gpu[i]!))), absFloor))) bad = true;
  }
  if (bad) {
    stats.over++;
    stats.firstBad ??= { at, gpu: [...gpu], cpu: [...cpu] };
  }
}

const newStats = (bound: string): Stats => ({ compared: 0, covered: 0, over: 0, maxUlp: 0, maxAbs: 0, bound });

// ------------------------------------------------------------------ GPU helpers

async function readTexture(device: GPUDevice, texture: GPUTexture, size: readonly number[], texelBytes: number): Promise<Uint8Array> {
  const [w, h, d] = size as [number, number, number];
  const bytesPerRow = Math.ceil((w * texelBytes) / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([enc.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buffer.getMappedRange());
  const out = new Uint8Array(w * h * d * texelBytes);
  for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) out.set(src.subarray((z * h + y) * bytesPerRow, (z * h + y) * bytesPerRow + w * texelBytes), (z * h + y) * w * texelBytes);
  buffer.unmap();
  buffer.destroy();
  return out;
}

/** rgba16float texels as numbers (x fastest, then y, then z). */
const halves = (bytes: Uint8Array): Float64Array => {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  return Float64Array.from(u16, (b) => vdbHalfToFloat(b));
};

function writeHalfTexture(device: GPUDevice, texture: GPUTexture, size: readonly number[], values: ArrayLike<number>): void {
  const u16 = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) u16[i] = vdbFloatToHalf(values[i]!);
  const [w, h, d] = size as [number, number, number];
  device.queue.writeTexture({ texture }, u16, { bytesPerRow: w * 8, rowsPerImage: h }, [w, h, d]);
}

/** A seeded PRNG (mulberry32). */
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

const roundHalf = (x: number) => vdbHalfToFloat(vdbFloatToHalf(x));

// ------------------------------------------------------------------ matrices

function perspective(fovY: number, aspect: number, near: number, far: number): number[] {
  const f = 1 / Math.tan(fovY / 2);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
}

function lookAt(eye: VDBFrameVec3, at: VDBFrameVec3): number[] {
  const sub = (a: VDBFrameVec3, b: VDBFrameVec3): VDBFrameVec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const n = (v: VDBFrameVec3): VDBFrameVec3 => {
    const l = Math.hypot(...v);
    return [v[0] / l, v[1] / l, v[2] / l];
  };
  const cross = (a: VDBFrameVec3, b: VDBFrameVec3): VDBFrameVec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const z = n(sub(eye, at));
  const x = n(cross([0, 1, 0], z));
  const y = cross(z, x);
  return [...x, 0, ...y, 0, ...z, 0, ...eye, 1]; // camera → world
}

function compose(t: VDBFrameVec3, angleY: number, s: VDBFrameVec3): number[] {
  const c = Math.cos(angleY);
  const n = Math.sin(angleY);
  return [c * s[0], 0, -n * s[0], 0, 0, s[1], 0, 0, n * s[2], 0, c * s[2], 0, t[0], t[1], t[2], 1];
}

// ------------------------------------------------------------------ synthetic dense volumes

function synthDense(dims: [number, number, number], seed: number, o: { albedo?: boolean; mask?: boolean; emission?: boolean; mips: boolean }): WvdbDenseVolume {
  const r = rng(seed);
  const ph = [r() * 6.28, r() * 6.28, r() * 6.28];
  const [W, H, D] = dims;
  // Smooth, low-gradient data (< ~3/255 per texel): hardware filter-weight precision stays far below 1 f16 ulp.
  const field = (k: number, base: number, amp: number) => {
    const data = new Uint8Array(W * H * D * k);
    for (let z = 0; z < D; z++)
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          for (let c = 0; c < k; c++) {
            const v = base + amp * Math.sin((x / W) * 3.1 + ph[0]! + c) * Math.cos((y / H) * 2.7 + ph[1]!) * Math.sin((z / D) * 2.3 + ph[2]! + 0.5 * c);
            data[((z * H + y) * W + x) * k + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
          }
    return data;
  };
  const tex = (data: Uint8Array<ArrayBuffer>, k: 1 | 4): WvdbDenseTexture => ({
    data,
    dimensions: [W, H, D],
    format: k === 1 ? "r8unorm" : "rgba8unorm",
    texelBytes: k,
    mip: o.mips ? wvdbMipAtlas(data, [W, H, D], k) : null,
  });
  const density = field(1, 0.55, 0.25);
  let sum = 0;
  let count = 0;
  let max = 0;
  for (const v of density) {
    if (v > 0) {
      sum += v / 255;
      count++;
    }
    max = Math.max(max, v / 255);
  }
  return {
    kind: "dense",
    normalize: "clamp01",
    lod: 0,
    dimensions: [W, H, D],
    bbox: [
      [-0.5, -0.5, -0.5],
      [0.5, 0.5, 0.5],
    ],
    density: tex(density, 1),
    mask: o.mask ? tex(field(1, 0.85, 0.12), 1) : null,
    albedo: o.albedo ? tex(field(4, 0.75, 0.2), 4) : null,
    emission: o.emission ? tex(field(1, 0.45, 0.2), 1) : null,
    densityStats: { sum, count },
    densityStatsSource: "texels",
    meanDensity: count > 0 ? sum / count : 0,
    maxDensity: max,
    activeTexels: count,
    gridNames: { density: "density" },
  } as WvdbDenseVolume;
}

// ------------------------------------------------------------------ probes

async function probeView(device: GPUDevice): Promise<void> {
  const rows = 1024;
  const row = CHUNK_PROBE_INPUTS + CHUNK_PROBE_OUTPUTS;
  const spec = vdbFrameChunkProbeSpec();
  const built = computePipeline(device, spec);
  const r = rng(7);
  const inputs = new Float32Array(rows * row);
  for (let i = 0; i < rows; i++) {
    const x = inputs.subarray(i * row, i * row + CHUNK_PROBE_INPUTS);
    x[0] = r(); // t
    x[1] = r(); // u
    x[2] = r(); // v
    x[3] = 0.2 + r() * 60; // view depth
    x[4] = i % 4 === 0 ? -1 : r(); // jitter
    x[5] = r() * 2 - 1; // cos
    x[6] = r() * 1.6 - 0.8; // g
    x[7] = r() * 3; // extinction
    x[8] = r() * 2; // absorption
    x[9] = i % 3 === 0 ? 0 : r(); // ms
    x[10] = r(); // c
    x[11] = r() * 0.95; // alpha
  }
  const projection = perspective(0.9, 1.5, 0.3, 200);
  const projectionInverse = vdbMat4Invert(projection);
  const range: [number, number] = [0.5, 120];
  const depth = 24;
  const fade = 40;
  const fw = new VDBUniformWriter(VDB_FRAME_UNIFORMS.size);
  fw.set(VDB_FRAME_UNIFORMS, 0, "projectionInverse", projectionInverse);
  fw.set(VDB_FRAME_UNIFORMS, 0, "grid", [64, 48, depth, 5]);
  fw.set(VDB_FRAME_UNIFORMS, 0, "range", [range[0], range[1], range[0], range[1]]);
  fw.set(VDB_FRAME_UNIFORMS, 0, "jitterFade", [0, fade, 0.2, 0]);
  const frameBuffer = device.createBuffer({ size: VDB_FRAME_UNIFORMS.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(frameBuffer, 0, fw.buffer);
  const lightsBuffer = device.createBuffer({ size: VDB_FRAME_LIGHTS.size, usage: GPUBufferUsage.UNIFORM });
  const storage = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(storage, 0, inputs);
  const readback = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(built.pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: built.layouts[0]!, entries: [{ binding: 0, resource: { buffer: frameBuffer } }, { binding: 1, resource: { buffer: lightsBuffer } }, { binding: 2, resource: device.createSampler() }] }));
  pass.setBindGroup(1, device.createBindGroup({ layout: built.layouts[1]!, entries: [{ binding: 0, resource: { buffer: storage } }] }));
  pass.dispatchWorkgroups(Math.ceil(rows / 64));
  pass.end();
  enc.copyBufferToBuffer(storage, 0, readback, 0, inputs.byteLength);
  device.queue.submit([enc.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  const stats = newStats("2 f16 ulp of each f32 output (f32 GPU vs f64 twin)");
  const f = Math.fround;
  for (let i = 0; i < rows; i++) {
    const x = Array.from(inputs.subarray(i * row, i * row + CHUNK_PROBE_INPUTS));
    const g = Array.from(out.subarray(i * row + CHUNK_PROBE_INPUTS, (i + 1) * row));
    const p = vdbFrameViewPosition(projectionInverse, [x[1]!, x[2]!], x[3]!);
    const segA: VDBFrameVec3 = [x[1]! * 2 - 1, x[2]! * 2 - 1, x[5]! * 2 - 1];
    const segB: VDBFrameVec3 = [x[6]! * 2 - 0.5, f(x[7]! * 0.1) * 2 - 0.5, x[0]! * 2 - 0.5];
    const d = [0, 1, 2].map((k) => segB[k]! - segA[k]!);
    const t0 = [0, 1, 2].map((k) => (-0.5 - segA[k]!) / d[k]!);
    const t1 = [0, 1, 2].map((k) => (0.5 - segA[k]!) / d[k]!);
    const seg = [Math.max(0, ...t0.map((v, k) => Math.min(v, t1[k]!))), Math.min(1, ...t0.map((v, k) => Math.max(v, t1[k]!)))];
    const pc = vdbFramePhaseColor([x[6]!, -0.3, 0.2, 0], [1.2, 0.8, 1], [1, 0, 0], [x[5]!, Math.sqrt(Math.max(0, 1 - x[5]! ** 2)), 0]);
    const st = vdbFrameShadowStep(x[7]!, x[8]!, x[9]!);
    const depths: VDBFrameVec3 = [st[0] * 3, st[1] * 3, st[2] * 3];
    const tr = vdbFrameShadowTransmittance(depths, x[9]!);
    const kc = vdbFrameSliceCoord(x[3]!, range, depth);
    const c = MUTATE === "coverage" ? Math.min(1, kc.c * 1.1) : kc.c;
    const taps = vdbFrameIntegratePrefix([x[7]!, x[8]!, x[9]!, f(x[11]! * 0.5)], [f(x[7]! + x[1]!), f(x[8]! + x[2]!), x[9]!, x[11]!], x[10]!);
    const radiance: VDBFrameVec3 = [f(x[1]! * 0.5), f(x[2]! * 0.5), f(x[6]! * 0.5)];
    const blend = vdbFrameSrgbBlend([x[0]!, f(x[5]! * 0.5 + 0.5), x[10]!], radiance, x[11]!, true);
    const cpu = [
      vdbFrameSliceDepth(range, x[0]!),
      p[0],
      p[1],
      p[2],
      vdbFrameNoise(x[4]!, [f(x[1]! * 977), f(x[2]! * 977)]),
      vdbFrameDistanceFade(fade, p),
      seg[0]!,
      seg[1]!,
      Math.max(0, Math.log2(f(x[3]! * 0.01) * 128)),
      vdbFrameHenyeyGreenstein(x[5]!, x[6]!),
      pc[0],
      pc[2],
      st[1],
      st[2],
      vdbFrameShadowDepth(depths, x[9]!),
      tr[0],
      tr[1],
      vdbFrameIsotropicDiffusion(st[2], x[8]!, st[1]),
      kc.k,
      c,
      taps.radiance[0],
      taps.transmittance,
      blend[0],
      x[11]! + 1 * (1 - x[11]!),
    ];
    // noise: fract() of large products amplifies f32 rounding (his formula); compare it on its own scale
    const tol = (k: number) => (k === 4 ? 2e-3 : 0);
    compare(stats, [i], g.map((v, k) => (tol(k) && Math.abs(v - cpu[k]!) < tol(k) ? cpu[k]! : v)), cpu, 2, 1e-5);
  }
  result.stats = stats;
}

async function probeGather(device: GPUDevice): Promise<void> {
  const mix = (params.get("mix") ?? "mixture") as "mixture" | "over";
  const groups = Number(params.get("groups") ?? 1);
  const mips = params.get("mips") !== "0";
  const [GW, GH, GD] = [24, 16, 12];
  const volA = synthDense([20, 16, 24], 1, { albedo: true, mips });
  const volB = synthDense([16, 20, 18], 2, { mask: true, emission: true, mips });
  const srcA = vdbDenseSource(device, volA, { label: "k6 A" });
  const srcB = groups === 2 ? vdbDenseSource(device, volB, { label: "k6 B" }) : srcA;
  const settings = { lighting: "volume" as const, width: GW, height: GH, depth: GD, subSteps: 16, temporal: false, jitter: false, multipleScattering: true, froxelMix: mix, phase: "dual-lobe" as const, fadeRadius: 9 };
  const renderer = vdbCreateFrameRenderer(device, { settings });
  const pA: Partial<VDBFrameVolumeParams> = { matrixWorld: compose([0, 0, 0], 0.4, [2, 1.6, 2.2]), extinction: 4, shadowSteps: 12, mipmaps: mips, phaseG: 0.6 };
  const pB: Partial<VDBFrameVolumeParams> = {
    matrixWorld: compose([0.7, 0.25, -0.3], -0.3, [1.8, 1.5, 1.7]),
    extinction: 3,
    baseColor: [0.9, 0.8, 0.7],
    shadowSteps: 10,
    mipmaps: mips,
    emissiveIntensity: 0.5,
    temperatureRange: [500, 3000],
    scatterColor: [0.01, 0.02, 0.03],
    densityCutoff: 0.05,
    wrap3D: "mirror",
    offset3D: [0.1, 0, -0.05],
  };
  const a = renderer.addVolume(srcA, pA);
  const b = renderer.addVolume(groups === 2 ? srcB : srcA, pB);
  const eyeWorld = lookAt([3.2, 1.6, 4.4], [0.2, 0, 0]);
  const view = vdbMat4Invert(eyeWorld);
  const projection = perspective(0.75, GW / GH, 0.1, 100);
  const lights: VDBFrameLight[] = [
    { id: 1, kind: "ambient", color: [0.05, 0.06, 0.08] },
    { id: 2, kind: "hemisphere", sky: [0.3, 0.4, 0.6], ground: [0.2, 0.15, 0.1], up: [0, 1, 0] },
    { id: 3, kind: "directional", direction: [-0.4, -0.8, -0.45], color: [1.4, 1.3, 1.2] },
    { id: 4, kind: "point", position: [2, 2, 1.5], color: [2, 1.4, 0.8], distance: 0, decay: 2 },
    { id: 5, kind: "spot", position: [-2.5, 3, 2], direction: [0.55, -0.65, -0.5], color: [1.5, 1.5, 2.2], distance: 0, decay: 2, cosAngle: Math.cos(0.6), cosPenumbra: Math.cos(0.6 * 0.6) },
    { id: 6, kind: "probe", sh: Array.from({ length: 27 }, (_, i) => (i < 3 ? 0.08 : 0.02 * Math.sin(i))) },
  ];
  const frame = (frameId: number) => ({ frameId, camera: { view, projection, near: 0.1, far: 100, id: 1 }, viewport: { drawingBuffer: [GW, GH] as [number, number], canvas: [GW, GH] as [number, number] }, lights });
  const counter = countBuilds(device);
  let enc = device.createCommandEncoder();
  const u = renderer.update(enc, frame(0));
  device.queue.submit([enc.finish()]);
  const first = counter.count;
  if (params.get("swap") === "1") {
    enc = device.createCommandEncoder();
    renderer.update(enc, frame(1));
    device.queue.submit([enc.finish()]);
    result.builds = { first, second: counter.count - first };
  }
  counter.restore();
  if (!u.recorded) throw new Error("update recorded nothing");
  const tex = vdbFrameRendererTextures(renderer)!;
  const scatter = halves(await readTexture(device, tex.scatter, [GW, GH, GD], 8));

  // The twin: the renderer's CPU work redone with the same tools functions
  const clip = vdbFrameClipCamera(view, projection, { coordinateSystem: "webgpu" });
  const vols = [
    { h: a, vol: volA, p: a.params },
    { h: b, vol: groups === 2 ? volB : volA, p: b.params },
  ];
  const clipped = vols.map((v) => ({ ...v, c: vdbFrameClipVolume(clip, v.p.matrixWorld)! }));
  const near = Math.max(Math.min(...clipped.map((v) => v.c.viewNear)), 0.1);
  const far = Math.min(Math.max(...clipped.map((v) => v.c.viewFar)), 100);
  const range: [number, number] = [near, Math.max(far, near * 1.0001)];
  const records: TwinVolume[] = clipped.map(({ vol, p, c }) => {
    const slices = vdbFrameVolumeSlices(c.viewNear, c.viewFar, range, GD);
    const g = p.phaseG ?? 0.8;
    return {
      volume: vol,
      rect: vdbFrameFroxelRect(c.localToView, projection, GW, GH)!,
      first: slices.first,
      count: slices.count,
      viewToLocal: c.viewToLocal,
      localToView: c.localToView,
      worldToLocal: c.worldToLocal,
      densityScale: p.densityScale / vol.meanDensity,
      densityCutoff: p.densityCutoff,
      extinction: p.extinction,
      worldScale: vdbMat4MaxScale(p.matrixWorld),
      absorbance: p.absorbance,
      opacity: p.opacity,
      baseColor: [...p.baseColor] as VDBFrameVec3,
      scatterColor: [...p.scatterColor] as VDBFrameVec3,
      offset: [...p.offset3D] as VDBFrameVec3,
      wrap: p.wrap3D === "mirror" ? 2 : p.wrap3D === "repeat" ? 1 : 0,
      phase: [g, -0.3, 0.2, 0],
      phaseSpectrum: [1, 1, 0],
      // diffusion: 1 − the dual-lobe asymmetry (back g −0.3, weight 0.2)
      multipleScattering: 1 - (g * 0.8 + -0.3 * 0.2),
      lights: p.lights,
      shadowSteps: p.shadowSteps,
      mipmaps: p.mipmaps && (vol.density.mip?.levels ?? 0) > 0,
      levels: vol.density.mip?.levels ?? 0,
      emissiveIntensity: p.emissiveIntensity,
      temperatureRange: [p.temperatureRange[0], p.temperatureRange[1]],
    };
  });
  // record order = source-group order: A's source first (one group: add order)
  const ordered = records;
  const twinFrame: TwinFrame = {
    grid: [GW, GH, GD],
    columns: vdbFrameAtlasLayout(GW, GH, GD).columns,
    range,
    projectionInverse: vdbMat4Invert(projection),
    cameraWorld: eyeWorld,
    jitter: -1,
    fadeRadius: settings.fadeRadius,
    subSteps: settings.subSteps,
    ambient: [0.05, 0.06, 0.08],
    probe: Array.from({ length: 9 }, (_, i) => [0, 1, 2].map((c) => (i * 3 + c < 3 ? 0.08 : 0.02 * Math.sin(i * 3 + c))) as VDBFrameVec3),
    hemi: [{ sky: [0.3, 0.4, 0.6], ground: [0.2, 0.15, 0.1], up: [0, 1, 0] }],
    lights: lightRecords(lights),
  };
  const stats = newStats("2 f16 ulp (rgba16float scatter vs the f64 twin)");
  for (let z = 0; z < GD; z++)
    for (let y = 0; y < GH; y++)
      for (let x = 0; x < GW; x++) {
        const i = ((z * GH + y) * GW + x) * 4;
        const gpu = [scatter[i]!, scatter[i + 1]!, scatter[i + 2]!, scatter[i + 3]!];
        const cpu = gatherTwin(twinFrame, ordered, [x, y, z], mix, MUTATE);
        compare(stats, [x, y, z], gpu, cpu, 2, 2 ** -20);
      }
  result.stats = stats;
  renderer.dispose();
  srcA.dispose();
  if (srcB !== srcA) srcB.dispose();
}

/** The lights as the renderer packs them (directional, point, spot; position w kind). */
function lightRecords(lights: readonly VDBFrameLight[]): TwinFrame["lights"] {
  const n = (v: readonly number[]) => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!);
    return [v[0]! / l, v[1]! / l, v[2]! / l];
  };
  const out: TwinFrame["lights"] = [];
  for (const kind of ["directional", "point", "spot"] as const) {
    for (const l of lights) {
      if (l.kind !== kind) continue;
      if (l.kind === "directional") {
        const d = n(l.direction);
        out.push({ position: [-d[0]!, -d[1]!, -d[2]!, 1], color: [...l.color, 0], spot: [0, 0, 0, 0] });
      } else if (l.kind === "point") out.push({ position: [...l.position, 2], color: [...l.color, 0], spot: [0, 0, 0, 0] });
      else if (l.kind === "spot") {
        const d = n(l.direction);
        out.push({ position: [...l.position, 3], color: [...l.color, l.cosPenumbra], spot: [-d[0]!, -d[1]!, -d[2]!, l.cosAngle] });
      }
    }
  }
  return out;
}

function countBuilds(device: GPUDevice): { readonly count: number; restore(): void } {
  const names = ["createShaderModule", "createComputePipeline", "createComputePipelineAsync", "createRenderPipeline", "createRenderPipelineAsync"] as const;
  const d = device as unknown as Record<string, (...a: unknown[]) => unknown>;
  const saved = names.map((name) => d[name]!);
  const state = { count: 0 };
  names.forEach((name, i) => {
    d[name] = (...a: unknown[]) => {
      state.count++;
      return saved[i]!.apply(device, a);
    };
  });
  return {
    get count() {
      return state.count;
    },
    restore() {
      names.forEach((name, i) => (d[name] = saved[i]!));
    },
  };
}

function frameUniforms(device: GPUDevice, values: Record<string, number | ArrayLike<number>>): { frame: GPUBuffer; lights: GPUBuffer } {
  const w = new VDBUniformWriter(VDB_FRAME_UNIFORMS.size);
  for (const [k, v] of Object.entries(values)) w.set(VDB_FRAME_UNIFORMS, 0, k, v);
  const frame = device.createBuffer({ size: VDB_FRAME_UNIFORMS.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(frame, 0, w.buffer);
  const lights = device.createBuffer({ size: VDB_FRAME_LIGHTS.size, usage: GPUBufferUsage.UNIFORM });
  return { frame, lights };
}

async function probeResolve(device: GPUDevice): Promise<void> {
  const temporal = params.get("temporal") === "1";
  const [W, H, D] = [20, 14, 16];
  const r = rng(11);
  const area = [3, 2, 17, 12];
  const slices = [2, 14];
  const scatter = new Float64Array(W * H * D * 4);
  const history = new Float64Array(W * H * D * 4);
  for (let i = 0; i < W * H * D; i++) {
    const z = Math.floor(i / (W * H));
    // dense deep slices: most columns pass T < 0.001 (the scan's stop)
    const a = r() < 0.15 ? 0 : z > 4 ? 0.6 + r() * 0.38 : r() * 0.4;
    const L = [r() * 2, r() * 2, r() * 2];
    for (let c = 0; c < 3; c++) {
      scatter[i * 4 + c] = roundHalf(L[c]! * a);
      history[i * 4 + c] = roundHalf(r() * 1.5);
    }
    scatter[i * 4 + 3] = roundHalf(a);
    history[i * 4 + 3] = roundHalf(r());
  }
  // gather leaves zeros outside its area
  for (let z = 0; z < D; z++)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        if (x >= area[0]! && x < area[2]! && y >= area[1]! && y < area[3]! && z >= slices[0]! && z < slices[1]!) continue;
        scatter.fill(0, ((z * H + y) * W + x) * 4, ((z * H + y) * W + x) * 4 + 4);
      }
  const projection = perspective(0.8, W / H, 0.2, 50);
  const camera = lookAt([0.5, 0.3, 3], [0, 0, 0]);
  const view = vdbMat4Invert(camera);
  const range: [number, number] = [0.5, 30];
  const blend = 0.2;
  const { frame, lights } = frameUniforms(device, {
    cameraWorld: camera,
    projectionInverse: vdbMat4Invert(projection),
    previousView: view,
    previousProjection: projection,
    grid: [W, H, D, vdbFrameAtlasLayout(W, H, D).columns],
    range: [range[0], range[1], range[0], range[1]],
    areaRect: area,
    areaSlices: slices,
    dispatchBox: [0, 0, W, H],
    jitterFade: [-1, 0, blend, temporal ? 1 : 0],
    flags: [16, temporal ? 1 : 0, 0, 0],
  });
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
  const mk = (d: number) => device.createTexture({ size: [W, H, d], dimension: "3d", format: "rgba16float", usage });
  const scatterTex = mk(D);
  const integrated = mk(D + 1);
  const historyIn = mk(D);
  const historyOut = mk(D);
  writeHalfTexture(device, scatterTex, [W, H, D], scatter);
  writeHalfTexture(device, historyIn, [W, H, D], history);
  const built = computePipeline(device, vdbFrameResolveSpec({ temporal }));
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(built.pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: built.layouts[0]!, entries: [{ binding: 0, resource: { buffer: frame } }, { binding: 1, resource: { buffer: lights } }, { binding: 2, resource: sampler }] }));
  pass.setBindGroup(
    1,
    device.createBindGroup({
      layout: built.layouts[1]!,
      entries: [
        { binding: 0, resource: scatterTex.createView() },
        { binding: 1, resource: integrated.createView() },
        ...(temporal ? [{ binding: 2, resource: historyIn.createView() }, { binding: 3, resource: historyOut.createView() }] : []),
      ],
    }),
  );
  pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const prefix = halves(await readTexture(device, integrated, [W, H, D + 1], 8));
  const histOut = temporal ? halves(await readTexture(device, historyOut, [W, H, D], 8)) : null;
  const at = (arr: Float64Array, x: number, y: number, z: number): VDBFrameVec4 => {
    const i = ((z * H + y) * W + x) * 4;
    return [arr[i]!, arr[i + 1]!, arr[i + 2]!, arr[i + 3]!];
  };
  const projectionInverse = vdbMat4Invert(projection);
  const stats = newStats("2 f16 ulp per boundary of the prefix (and of history′ with temporal)");
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const inRect = x >= area[0]! && x < area[2]! && y >= area[1]! && y < area[3]!;
      let expected: { prefix: VDBFrameVec4[]; history: VDBFrameVec4[] };
      if (!inRect) expected = { prefix: Array.from({ length: D + 1 }, () => [0, 0, 0, 0] as VDBFrameVec4), history: Array.from({ length: D }, () => [0, 0, 0, 0] as VDBFrameVec4) };
      else {
        const column = Array.from({ length: D }, (_, z) => at(scatter, x, y, z));
        expected = resolveTwin({
          column,
          neighbours: (z) => [at(scatter, Math.max(x - 1, 0), y, z), at(scatter, Math.min(x + 1, W - 1), y, z), at(scatter, x, Math.max(y - 1, 0), z), at(scatter, x, Math.min(y + 1, H - 1), z)],
          history: temporal
            ? (z) => {
                const uv: [number, number] = [(x + 0.5) / W, (y + 0.5) / H];
                const viewP = vdbFrameViewPosition(projectionInverse, uv, vdbFrameSliceDepth(range, (z + 0.5) / D));
                const w = [0, 1, 2].map((k) => camera[k]! * viewP[0] + camera[4 + k]! * viewP[1] + camera[8 + k]! * viewP[2] + camera[12 + k]!) as VDBFrameVec3;
                const rp = vdbFrameReproject(w, view, projection, range, [W, H, D], { uv, slice: z }, blend);
                if (!rp) return null;
                // static camera: the reprojection lands on the texel centre
                return at(history, Math.min(W - 1, Math.max(0, Math.round(rp.uv[0] * W - 0.5))), Math.min(H - 1, Math.max(0, Math.round(rp.uv[1] * H - 0.5))), Math.min(D - 1, Math.max(0, Math.round(rp.slice))));
              }
            : null,
          blend,
          inSlices: (z) => z >= slices[0]! && z < slices[1]!,
          mutate: MUTATE,
        });
      }
      for (let k = 0; k <= D; k++) compare(stats, [x, y, k], at(prefix, x, y, k), expected.prefix[k]!);
      if (histOut) for (let z = 0; z < D; z++) compare(stats, [x, y, z, -1], at(histOut, x, y, z), inRect ? expected.history[z]! : [0, 0, 0, 0]);
    }
  result.stats = stats;
}

async function probeIntegrate(device: GPUDevice): Promise<void> {
  const [W, H, D] = [40, 28, 10];
  const r = rng(23);
  const range: [number, number] = [1, 80];
  const prefix = new Float64Array(W * H * (D + 1) * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const froxels: VDBFrameVec4[] = Array.from({ length: D }, () => {
        const a = r() < 0.3 ? 0 : r() * 0.6;
        return [r() * a, r() * a, r() * a, a];
      });
      const col = vdbFrameScanColumn(froxels);
      col.forEach((b, k) => prefix.set(b.map(roundHalf), ((k * H + y) * W + x) * 4));
    }
  const near = 0.3;
  const far = 200;
  const depthValues = new Float32Array(W * H);
  const viewDepths = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (i % 17 === 0) {
      depthValues[i] = 1;
      viewDepths[i] = 1e20;
      continue;
    }
    const z = 0.5 + r() * 100;
    depthValues[i] = (far * (z - near)) / (z * (far - near));
    const d = depthValues[i]!;
    viewDepths[i] = -((near * far) / ((far - near) * d - far));
  }
  const { frame, lights } = frameUniforms(device, {
    grid: [W, H, D, 1],
    range: [range[0], range[1], range[0], range[1]],
    flags: [16, 0, 0, 0],
    screen: [W, H, 1, 0],
    camera: [near, far, 0, 0],
  });
  const prefixTex = device.createTexture({ size: [W, H, D + 1], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  writeHalfTexture(device, prefixTex, [W, H, D + 1], prefix);
  const depthTex = device.createTexture({ size: [W, H], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: depthTex }, depthValues, { bytesPerRow: W * 4 }, [W, H]);
  const color = device.createTexture({ size: [W, H], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const built = renderPipeline(device, vdbFrameCompositeSpec({ colorFormat: "rgba16float", depth: "r32float", srgb: false }));
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: color.createView(), loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }] });
  pass.setPipeline(built.pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: built.layouts[0]!, entries: [{ binding: 0, resource: { buffer: frame } }, { binding: 1, resource: { buffer: lights } }, { binding: 2, resource: sampler }] }));
  pass.setBindGroup(
    1,
    device.createBindGroup({
      layout: built.layouts[1]!,
      entries: [
        { binding: 0, resource: prefixTex.createView() },
        { binding: 1, resource: sampler },
        { binding: 2, resource: depthTex.createView() },
      ],
    }),
  );
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
  const out = halves(await readTexture(device, color, [W, H, 1], 8));
  const stats = newStats("2 f16 ulp (rgba16float target; froxel grid = screen, so taps sit on texel centres)");
  for (let py = 0; py < H; py++)
    for (let x = 0; x < W; x++) {
      const y = H - 1 - py; // texel y is bottom-up; pixel rows top-down
      const i = py * W + x;
      const cpu = integrateTwin((k) => {
        const j = ((k * H + y) * W + x) * 4;
        return [prefix[j]!, prefix[j + 1]!, prefix[j + 2]!, prefix[j + 3]!];
      }, viewDepths[i]!, range, D, MUTATE);
      const alpha = cpu[3];
      compare(stats, [x, py], [out[i * 4]!, out[i * 4 + 1]!, out[i * 4 + 2]!, out[i * 4 + 3]!], alpha > 0 ? cpu : [0, 0, 0, 0]);
    }
  result.stats = stats;
}

async function probeCompile(device: GPUDevice): Promise<void> {
  const vol = synthDense([8, 8, 8], 3, { albedo: true, mask: true, emission: true, mips: true });
  const plain = synthDense([8, 8, 8], 4, { mips: false });
  const specs = [
    ...[vol, plain].flatMap((v) => {
      const s = vdbDenseSource(device, v);
      return (["mixture", "over"] as const).flatMap((mix) => [false, true].map((multi) => vdbFrameGatherSpec(s, { mix, multi })));
    }),
    vdbFrameResolveSpec({ temporal: false }),
    vdbFrameResolveSpec({ temporal: true }),
    vdbFrameChunkProbeSpec(),
  ];
  const errors: string[] = [];
  for (const spec of specs) {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: spec.code });
    const info = await module.getCompilationInfo();
    for (const m of info.messages) if (m.type === "error") errors.push(`${spec.label}: ${m.lineNum}:${m.linePos} ${m.message}`);
    try {
      await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: spec.groups.map((g) => device.createBindGroupLayout(g)) }),
        compute: { module, entryPoint: spec.entryPoint },
      });
    } catch (e) {
      errors.push(`${spec.label}: ${String(e)}`);
    }
    const e = await device.popErrorScope();
    if (e) errors.push(`${spec.label}: ${e.message}`);
  }
  let variants = specs.length;
  for (const depth of ["depth", "r32float"] as const)
    for (const srgb of [false, true]) {
      const spec = vdbFrameCompositeSpec({ colorFormat: "rgba16float", depth, srgb });
      variants++;
      device.pushErrorScope("validation");
      const module = device.createShaderModule({ code: spec.code });
      const info = await module.getCompilationInfo();
      for (const m of info.messages) if (m.type === "error") errors.push(`${spec.label}: ${m.lineNum}:${m.linePos} ${m.message}`);
      try {
        await device.createRenderPipelineAsync({
          layout: device.createPipelineLayout({ bindGroupLayouts: spec.groups.map((g) => device.createBindGroupLayout(g)) }),
          vertex: { module, entryPoint: spec.vertexEntryPoint },
          fragment: { module, entryPoint: spec.fragmentEntryPoint, targets: [...spec.targets] },
        });
      } catch (e) {
        errors.push(`${spec.label}: ${String(e)}`);
      }
      const e = await device.popErrorScope();
      if (e) errors.push(`${spec.label}: ${e.message}`);
    }
  result.compile = { variants, errors };
}

async function main(): Promise<void> {
  root.textContent = "";
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  if (PROBE === "view") await probeView(device);
  else if (PROBE === "gather") await probeGather(device);
  else if (PROBE === "resolve") await probeResolve(device);
  else if (PROBE === "integrate") await probeIntegrate(device);
  else if (PROBE === "compile") await probeCompile(device);
  else throw new Error(`unknown ?probe=${PROBE}`);
  await device.queue.onSubmittedWorkDone();
  const s = result.stats;
  if (s) log(`${PROBE}${MUTATE !== "none" ? ` (mutate ${MUTATE})` : ""}: ${s.compared} texels (${s.covered} non-zero), max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over ${s.bound}`, s.over === 0 ? "ok" : "fail");
  if (result.compile) log(`compile: ${result.compile.variants} variants, ${result.compile.errors.length} errors\n${result.compile.errors.join("\n")}`, result.compile.errors.length ? "fail" : "ok");
  if (result.builds) log(`builds: ${result.builds.first} in the first update, ${result.builds.second} in the second`, result.builds.second === 0 ? "ok" : "fail");
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
