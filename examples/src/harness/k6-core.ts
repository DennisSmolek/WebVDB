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
  VDB_FRAME_ENVIRONMENT,
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
  vdbFrameRendererGrid,
  vdbFrameRendererTextures,
  vdbFrameRendererEnvironment,
  vdbFrameRendererScattering,
  vdbFramePhaseVector,
  vdbFramePhaseSpectrum,
  vdbFrameMultipleScatteringValue,
  type VDBFramePhaseName,
  type VDBFrameMultipleScatteringMode,
  vdbFrameGridMipSpec,
  VDBFrameEnvironmentState,
  vdbFrameResolveSpec,
  vdbFrameGridDensitySpec,
  vdbFrameTransmittanceSpec,
  vdbFrameVolumeShadowSpec,
  vdbFrameEmissionDirectionSpec,
  vdbFrameEmissionReduceSpec,
  vdbFrameEmissionProbeSpec,
  vdbFrameRendererEmission,
  vdbVolumeShadowProbeWgsl,
  vdbEmissionIrradianceProbeWgsl,
  vdbValidateEmissionHook,
  vdbFrameEnvPrefilterSpec,
  vdbFrameEnvBlurSpec,
  vdbFrameEnvReduceSpec,
  vdbFrameEnvironmentGridSpec,
  vdbSparseSourceGpu,
  type VDBFrameLight,
  type VDBFrameVolumeGpu,
  type VDBFrameVolumeParams,
  type VDBSparseSourceGpu,
  type VDBSparseVolumeGpu,
  type VDBVolumeSlot,
} from "../../../packages/vdb/src/core/internal.js";
import { VDBGridPool } from "@pmndrs/vdb";
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
  vdbMat4Multiply,
  wvdbMipAtlas,
  vdbBuildFromDense,
  vdbBoxPyramid,
  vdbPyramidToHalf,
  vdbSparseAlbedoCpu,
  vdbSparseDensityCpu,
  vdbSparseEmissionCpu,
  vdbSparseMaskCpu,
  vdbFrameDensityFroxel,
  vdbFrameFroxel,
  vdbFrameGridDensityVoxel,
  vdbFrameGridVoxelWorld,
  vdbFrameLightingFroxel,
  vdbFrameStackSample,
  vdbFrameStackShadows,
  vdbFrameTransmittanceVoxel3d,
  vdbFrameMieLutWords,
  vdbFrameMieTexelsHalf,
  vdbFrameMieTexel,
  vdbFramePhaseWeight,
  vdbFrameOctaveLight,
  vdbFrameScatteringOctaves,
  vdbFrameShadowTotal,
  vdbFrameTwoSidedDiffusion,
  vdbFrameShadowedLightColor,
  vdbFrameShadowMarchDepth,
  vdbFrameMlpForward,
  vdbFrameMlpInputs,
  vdbFrameMlpMipTexel,
  vdbFrameMlpSampleLevel,
  vdbFrameMultiScaleLight,
  vdbFrameMultiScaleSample,
  vdbFrameTexture3DLinear,
  vdbFrameMultiScaleMips,
  type VDBFrameMlp,
  vdbFrameShadowMapTexel,
  vdbFrameShadowMapRay,
  vdbFrameCubeFaceDirection,
  vdbFrameCubeFaceUv,
  vdbFrameVolumeShadowLookup,
  vdbMat4Vec4,
  vdbFrameOccluderVisibility,
  vdbFrameEmissionLutRow,
  vdbFrameEmissionLutColor,
  vdbFrameEmitterBounds,
  vdbFrameEmissionDirectionTexel,
  vdbFrameEmissionReduce,
  vdbFrameEmissionProbe,
  vdbFrameEmissionIrradiance,
  vdbFrameEmissionLookupWith,
  vdbFrameEnvironmentGridVoxel3d,
  vdbFrameEnvironmentAreaTexel,
  vdbFrameEnvironmentCubeTexel,
  vdbFrameEnvironmentBlurTexel,
  vdbFrameEnvironmentIrradianceSh,
  vdbFrameEnvironmentReduce,
  vdbFrameEnvironmentDensityTerm,
  vdbFrameGridUvw,
  vdbFrameInsideGrid,
  vdbFrameEnvironmentMapWidth,
  vdbFrameEnvironmentMapHeight,
  VDB_FRAME_ENV_WORK_WIDTH,
  VDB_FRAME_ENV_WORK_HEIGHT,
  type VDBFrameEnvImage,
  type VDBFrameEnvironmentGpuResult,
  type VDBFrameGridLayout,
  type VDBFrameTexture3d,
  type VDBSparseVolumeCpu,
  type VDBFrameVec3,
  type VDBFrameVec4,
  type WvdbDenseTexture,
  type WvdbDenseVolume,
} from "../../../packages/vdb/src/tools/internal.js";
import { synthOccluders, type SynthOccluders } from "./k6-occluders-synth.js";
import {
  albedoAt,
  denseSampler,
  emissionAt,
  extinctionAt,
  gatherTwin,
  integrateTwin,
  resolveTwin,
  vdbFrameReproject,
  type TwinFrame,
  type TwinMutation,
  type TwinSampler,
  type TwinVolume,
} from "./k6-core-twin.js";

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
  /** `?probe=gather`: `render()` (the core composite pass) vs its twin. */
  renderStats?: Stats;
  /** `?probe=gather&swap=1`: builds in the second identical update. */
  builds?: { first: number; second: number };
  /** `?probe=frame`: the shadow grid and the transmittance stacks vs their twins (`stats` is the scatter texture). */
  gridStats?: Stats;
  /** K6.2b `?probe=environment`: the work map, his map and the environment buffer vs their twins; `?probe=frame&env=`: the environment grid. */
  workStats?: Stats;
  mapStats?: Stats;
  bufferStats?: Stats;
  environmentGridStats?: Stats;
  /** K6.2b: kernel runs after identical / intensity / rotation updates, and the lights the GPU found. */
  environment?: { runs: unknown[]; count: number; mode: number; valid: number; directions: number[][]; colors: number[][] };
  transmittanceStats?: Stats;
  /** K6.3 `?probe=frame&ms=mlp|multi-scale`: the scattered-light term's mips vs their twin. */
  mipStats?: Stats;
  /** `?probe=frame`: live cascades, redraw counters after a second identical update. */
  frame?: { cascades: number; redraw: unknown };
  compile?: { variants: number; errors: string[]; seconds: Record<string, number> };
  /** K6.5 `?probe=shadow`: the receivers vs their twin (`stats` is the map); redraw counts. */
  receiverStats?: Stats;
  shadow?: Record<string, unknown>;
  /** K6.5b `&occluders=1`: twin lookups that found an occluder (visibility < 1) per kernel, and the maps bound. */
  occluders?: { maps: number; transmittance: number; density: number; lighting: number; gather: number };
  /** K6.5 `?probe=emission`: the reduce, probes and receiver vs their twins (`stats` is the direction pass). */
  reduceStats?: Stats;
  probeStats?: Stats;
  irradianceStats?: Stats;
  emission?: Record<string, unknown>;
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
/** K6.2b `?env=directional|spherical-harmonics|outward`: a synthetic environment for the frame and gather probes. */
const ENV_MODE = params.get("env");
// K6.3 `?phase=` (dual-lobe default; isotropic, henyey-greenstein, lorenz-mie, lorenz-mie-rgb) and `?ms=` (1 default, 0,
// two-sided, octaves, mlp, neural, multi-scale; `&edge=clamp`, `&levels=`) of the gather and frame probes
/** K6.3: the Lorenz-Mie texels as the core reads them (binary16). */
const MIE_HALF = vdbFrameMieTexelsHalf();
const PHASE_NAME = (params.get("phase") ?? "dual-lobe") as VDBFramePhaseName;
const MS_PARAM = params.get("ms");
const MS_MODE: VDBFrameMultipleScatteringMode = MS_PARAM === null || MS_PARAM === "1" ? true : MS_PARAM === "0" ? false : (MS_PARAM as VDBFrameMultipleScatteringMode);
/** Octave settings of `?ms=octaves` (off the defaults so a swapped term shows). */
const OCTAVE_SETTINGS = { scatteringOctaves: 3, scatteringOctaveExtinction: 0.6, scatteringOctaveContribution: 0.45, scatteringOctavePhase: 0.7 };
const MS_SETTINGS = { multipleScattering: MS_MODE, ...OCTAVE_SETTINGS, multiScaleEdge: (params.get("edge") ?? "zero") as "zero" | "clamp", multiScaleLevels: Number(params.get("levels") ?? 6) };
const MS_OCTAVES = vdbFrameScatteringOctaves(MS_SETTINGS) as VDBFrameVec4;
/** K6.5b `?occluders=1`: synthetic host depth maps for the directional, point and spot probe lights (frame and gather probes). */
const OCCLUDERS = params.get("occluders") === "1";
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

async function readTexture(device: GPUDevice, texture: GPUTexture, size: readonly number[], texelBytes: number, mipLevel = 0): Promise<Uint8Array> {
  const [w, h, d] = size as [number, number, number];
  const bytesPerRow = Math.ceil((w * texelBytes) / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture, mipLevel }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
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
  pass.setBindGroup(0, device.createBindGroup({ layout: built.layouts[0]!, entries: [{ binding: 0, resource: { buffer: frameBuffer } }, { binding: 1, resource: { buffer: lightsBuffer } }, { binding: 2, resource: device.createSampler() }, envEntry(device), mieEntry(device)] }));
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
    const tr = vdbFrameShadowTransmittance(depths, x[9]!, [0, 0, 0, 0]);
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
      // K6.3 (?mutate=mie: the table off; ?mutate=octaves: one octave fewer; ?mutate=two-sided: no back depth)
      vdbFramePhaseWeight([x[6]!, -0.3, 0.2, MUTATE === "mie" ? 0 : 0.5], [1, 0, 0], [x[5]!, Math.sqrt(Math.max(0, 1 - x[5]! ** 2)), 0]),
      vdbFrameMieTexel(x[5]!, MIE_HALF)[0],
      vdbFrameTwoSidedDiffusion(tr[0], depths[2], depths[1], MUTATE === "two-sided" ? 0 : x[8]!),
      vdbFrameShadowTotal(x[11]!, [MUTATE === "octaves" ? 2 : 3, 0.6, 0.45, 0.7]),
      vdbFrameOctaveLight([MUTATE === "octaves" ? 2 : 3, 0.6, 0.45, 0.7], x[7]!, [x[6]!, -0.3, 0.2, 0], x[5]!),
    ];
    // noise: fract() of large products amplifies f32 rounding (his formula); compare it on its own scale. K6.3 two-sided
    // diffusion: the slab solve subtracts cosh / sinh of k L (f32 cancellation, his formula): 5e-4 absolute
    const tol = (k: number) => (k === 4 ? 2e-3 : k === 26 ? 5e-4 : 0);
    compare(stats, [i], g.map((v, k) => (tol(k) && Math.abs(v - cpu[k]!) < tol(k) ? cpu[k]! : v)), cpu, 2, 1e-5);
  }
  result.stats = stats;
}

// ------------------------------------------------------------------ the sparse source (K6.1d)

type V3 = [number, number, number];

/** A dense field (x-major, z fastest: vdbBuildFromDense's order) as a FLOAT grid at `origin`. */
function sparseGrid(dims: V3, origin: V3, fn: (x: number, y: number, z: number) => number): Uint32Array {
  const values = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let x = 0; x < dims[0]; x++) for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) values[(x * dims[1] + y) * dims[2] + z] = fn(x, y, z);
  return vdbBuildFromDense(values, dims, { origin, voxelSize: 0.25 });
}

interface SparseFixture {
  pool: VDBGridPool;
  images: Map<number, Uint32Array>;
  ids: { d1: number; d2: number; m: number; r: number; g: number; b: number; e: number };
}

/** Synthetic grids with empty space (the HDDA has leaves and nodes to skip): two densities, a mask, RGB albedo, emission. */
async function sparseFixture(device: GPUDevice): Promise<SparseFixture> {
  const pool = await VDBGridPool.create(device, { label: "k6-core sparse" });
  const images = new Map<number, Uint32Array>();
  const add = (image: Uint32Array) => {
    const id = pool.add(image);
    images.set(id, image);
    return id;
  };
  const blob = (cx: number, cy: number, cz: number, r: number) => (x: number, y: number, z: number) => Math.max(0, 1 - Math.hypot(x - cx, y - cy, z - cz) / r);
  const a = blob(8, 7, 7, 7);
  const b = blob(17, 13, 15, 6);
  const D1 = sparseGrid([26, 22, 24], [0, 0, 0], (x, y, z) => {
    const v = a(x, y, z) * 1.1 + b(x, y, z) * (0.8 + 0.15 * Math.sin(x * 0.5 + z * 0.3));
    return v > 0.03 ? v : 0;
  });
  const c = blob(9, 11, 8, 8);
  const D2 = sparseGrid([20, 22, 18], [40, 0, 0], (x, y, z) => {
    const v = c(x, y, z) * (0.9 + 0.1 * Math.cos(y * 0.4));
    return v > 0.03 ? v : 0;
  });
  const ids = {
    d1: add(D1),
    d2: add(D2),
    m: add(sparseGrid([20, 22, 18], [40, 0, 0], (x, y, z) => 0.35 + 0.6 * Math.abs(Math.sin(x * 0.21 + y * 0.13 - z * 0.08)))),
    r: add(sparseGrid([26, 22, 24], [0, 0, 0], (x, y) => 0.4 + 0.5 * ((x + y) / 48))),
    g: add(sparseGrid([26, 22, 24], [0, 0, 0], (_x, y, z) => 0.6 + 0.3 * Math.cos(y * 0.2 + z * 0.1))),
    b: add(sparseGrid([26, 22, 24], [0, 0, 0], (x, _y, z) => 0.3 + (x * z) / 1300)),
    e: add(sparseGrid([20, 22, 18], [40, 0, 0], (x, y, z) => Math.max(0, 0.8 - Math.hypot(x - 10, y - 11, z - 9) / 14))),
  };
  pool.flush();
  await pool.settled();
  return { pool, images, ids };
}

/** The CPU side of one sparse volume (each aux grid under its own box, as the source defaults). */
function sparseCpu(fx: SparseFixture, source: VDBSparseSourceGpu, slot: VDBSparseVolumeGpu): VDBSparseVolumeCpu {
  const own = (id: number) => {
    const b = fx.pool.info(id).indexBBox;
    return { min: [...b.min] as V3, max: [...b.max] as V3 };
  };
  const ids = slot.ids;
  const albedo = ids.albedo === undefined ? undefined : Array.isArray(ids.albedo) ? (ids.albedo as number[]) : [ids.albedo as number];
  const density = fx.images.get(ids.density)!;
  return {
    lod: source.lod,
    density: { image: density, box: slot.indexBox },
    ...(ids.mask !== undefined ? { mask: { image: fx.images.get(ids.mask)!, box: own(ids.mask) } } : {}),
    ...(albedo ? { albedo: { images: albedo.map((id) => fx.images.get(id)!), box: own(albedo[0]!) } } : {}),
    ...(ids.emission !== undefined ? { emission: { image: fx.images.get(ids.emission)!, box: own(ids.emission) } } : {}),
    ...(source.lod === "coarse-pyramid" ? { pyramid: vdbPyramidToHalf(vdbBoxPyramid(density, { indexBox: slot.indexBox, pyramidMax: SPARSE_PYRAMID_MAX })) } : {}),
  };
}

const SPARSE_PYRAMID_MAX = 8;

/** The twin sampler of a sparse volume (`vdbSrc*` of `vdbSparseSourceGpu`). */
function sparseSampler(cpu: VDBSparseVolumeCpu, slot: VDBVolumeSlot, source: VDBSparseSourceGpu): TwinSampler {
  return {
    texels: [...slot.texels] as V3,
    levels: slot.lodLevels,
    hasEmission: source.channels.emission,
    density: (uvw, lod) => vdbSparseDensityCpu(cpu, uvw, lod).value,
    mask: (uvw) => vdbSparseMaskCpu(cpu, uvw).value,
    albedo: (uvw) => vdbSparseAlbedoCpu(cpu, uvw).value,
    emission: (uvw) => vdbSparseEmissionCpu(cpu, uvw).value,
  };
}

/** Two volumes (and optionally a third slot for a caster) on dense or sparse sources, one or two source groups. */
async function probeScene(device: GPUDevice, o: { kind: "dense" | "sparse"; groups: number; mips: boolean; extra?: boolean }) {
  if (o.kind === "dense") {
    const volA = synthDense([20, 16, 24], 1, { albedo: true, mips: o.mips });
    const volB = synthDense([16, 20, 18], 2, { mask: true, emission: true, mips: o.mips });
    const srcA = vdbDenseSource(device, volA, { label: "k6 A" });
    const srcB = o.groups === 2 ? vdbDenseSource(device, volB, { label: "k6 B" }) : srcA;
    const slots: VDBVolumeSlot[] = [srcA, srcB, ...(o.extra ? [srcA] : [])];
    const samplers = [denseSampler(volA), denseSampler(o.groups === 2 ? volB : volA), ...(o.extra ? [denseSampler(volA)] : [])];
    return {
      slots,
      samplers,
      dispose() {
        srcA.dispose();
        if (srcB !== srcA) srcB.dispose();
      },
    };
  }
  const fx = await sparseFixture(device);
  const lod = (params.get("lod") ?? "none") as "none" | "coarse-pyramid" | "node-average";
  const options = { lod, pyramidMax: SPARSE_PYRAMID_MAX, types: ["float" as const], maxVolumes: 8 };
  const srcA = vdbSparseSourceGpu(fx.pool, { ...options, label: "k6 sparse A" });
  const srcB = o.groups === 2 ? vdbSparseSourceGpu(fx.pool, { ...options, label: "k6 sparse B" }) : srcA;
  const { ids } = fx;
  const slotA = srcA.volume({ density: ids.d1, albedo: [ids.r, ids.g, ids.b] }, { image: fx.images.get(ids.d1)! });
  const slotB = srcB.volume({ density: ids.d2, mask: ids.m, emission: ids.e }, { image: fx.images.get(ids.d2)! });
  const slotC = o.extra ? srcA.volume({ density: ids.d1 }, { image: fx.images.get(ids.d1)! }) : null;
  const slots: VDBVolumeSlot[] = [slotA, slotB, ...(slotC ? [slotC] : [])];
  return {
    slots,
    // channels are known once every volume is added (a sparse source compiles the union)
    samplers: [sparseSampler(sparseCpu(fx, srcA, slotA), slotA, srcA), sparseSampler(sparseCpu(fx, srcB, slotB), slotB, srcB), ...(slotC ? [sparseSampler(sparseCpu(fx, srcA, slotC), slotC, srcA)] : [])],
    dispose() {
      srcA.dispose();
      if (srcB !== srcA) srcB.dispose();
      fx.pool.dispose();
    },
  };
}

/**
 * `?mutate=skip`: the shadow march's HDDA query treats cells whose bound is ≤ 0.6 as empty (skips samples that are
 * not); a GPU patch. It fails only when the march asks the source and skips, so it also shows the HDDA path runs.
 */
function patchShaders(device: GPUDevice): () => void {
  if (MUTATE !== "skip") return () => {};
  const d = device as unknown as { createShaderModule: (desc: GPUShaderModuleDescriptor) => GPUShaderModule };
  const saved = d.createShaderModule;
  d.createShaderModule = (desc) => saved.call(device, { ...desc, code: desc.code.replace("let cutoff = max(vdbFrameVolumes.v[vi].density.y, 0.0);", "let cutoff = 0.6;") });
  return () => (d.createShaderModule = saved);
}

/** The renderer's per-volume record twin of a volume (its params, slot and clip). */
function twinRecord(handle: VDBFrameVolumeGpu, sampler: TwinSampler, c: { localToView: ArrayLike<number>; viewToLocal: ArrayLike<number>; worldToLocal: ArrayLike<number> }, rect: [number, number, number, number], slices: { first: number; count: number }, g: number, ms: VDBFrameMultipleScatteringMode = true): TwinVolume {
  // K6.3: the renderer's phase packing for ?phase= (dual-lobe back g −0.3, weight 0.2) and ?ms=
  const fields = { phase: PHASE_NAME, phaseG: g, phaseBackG: -0.3, phaseBackWeight: 0.2, phaseGExponent: 0 };
  const p = handle.params;
  const slot = handle.slot;
  const mean = p.densityMean ?? slot.stats.meanNonZero;
  return {
    volume: sampler,
    rect,
    first: slices.first,
    count: slices.count,
    viewToLocal: c.viewToLocal,
    localToView: c.localToView,
    worldToLocal: c.worldToLocal,
    densityScale: p.normalizeDensity && mean > 0 ? p.densityScale / mean : p.densityScale,
    densityCutoff: p.densityCutoff,
    extinction: p.extinction,
    worldScale: vdbMat4MaxScale(p.matrixWorld),
    absorbance: p.absorbance,
    opacity: p.opacity,
    baseColor: [...p.baseColor] as VDBFrameVec3,
    scatterColor: [...p.scatterColor] as VDBFrameVec3,
    offset: [...p.offset3D] as VDBFrameVec3,
    wrap: p.wrap3D === "mirror" ? 2 : p.wrap3D === "repeat" ? 1 : 0,
    phase: vdbFramePhaseVector(fields),
    phaseSpectrum: vdbFramePhaseSpectrum(fields),
    // diffusion: 1 − the dual-lobe asymmetry (back g −0.3, weight 0.2)
    multipleScattering: vdbFrameMultipleScatteringValue(ms, fields),
    lights: p.lights,
    shadowSteps: p.shadowSteps,
    mipmaps: p.mipmaps && slot.lodLevels > 0,
    levels: slot.lodLevels,
    emissiveIntensity: p.emissiveIntensity,
    temperatureRange: [p.temperatureRange[0], p.temperatureRange[1]],
  };
}

const PROBE_LIGHTS: VDBFrameLight[] = [
  { id: 1, kind: "ambient", color: [0.05, 0.06, 0.08] },
  { id: 2, kind: "hemisphere", sky: [0.3, 0.4, 0.6], ground: [0.2, 0.15, 0.1], up: [0, 1, 0] },
  { id: 3, kind: "directional", direction: [-0.4, -0.8, -0.45], color: [1.4, 1.3, 1.2] },
  { id: 4, kind: "point", position: [2, 2, 1.5], color: [2, 1.4, 0.8], distance: 0, decay: 2 },
  { id: 5, kind: "spot", position: [-2.5, 3, 2], direction: [0.55, -0.65, -0.5], color: [1.5, 1.5, 2.2], distance: 0, decay: 2, cosAngle: Math.cos(0.6), cosPenumbra: Math.cos(0.6 * 0.6) },
  { id: 6, kind: "probe", sh: Array.from({ length: 27 }, (_, i) => (i < 3 ? 0.08 : 0.02 * Math.sin(i))) },
];
const PROBE_SH = Array.from({ length: 9 }, (_, i) => [0, 1, 2].map((c) => (i * 3 + c < 3 ? 0.08 : 0.02 * Math.sin(i * 3 + c))) as VDBFrameVec3);

/** K6.5b: the synthetic occluder maps of the probe lights (`?occluders=1`), or null. */
async function probeOccluders(device: GPUDevice): Promise<SynthOccluders | null> {
  if (!OCCLUDERS) return null;
  const at = <K extends VDBFrameLight["kind"]>(kind: K) => PROBE_LIGHTS.find((l) => l.kind === kind) as Extract<VDBFrameLight, { kind: K }>;
  return synthOccluders(device, at("directional"), at("point"), at("spot"));
}

/**
 * K6.5b: the twin's occluder lookup by sorted light index (directional, point, spot; `lightRecords` order), counting
 * the lookups that found an occluder; `?mutate=occluder-filter` reads one texel everywhere.
 */
function occluderTwin(o: SynthOccluders, counter: { n: number }): (index: number, world: VDBFrameVec3) => number {
  const ids = ["directional", "point", "spot"].map((kind) => PROBE_LIGHTS.find((l) => l.kind === kind)!.id as number);
  return (index, world) => {
    const cpu = o.cpu.get(ids[index]!);
    if (!cpu) return 1;
    const v = vdbFrameOccluderVisibility(MUTATE === "occluder-filter" ? { ...cpu, filter: "nearest" } : cpu, world);
    if (v < 1) counter.n++;
    return v;
  };
}

async function probeGather(device: GPUDevice): Promise<void> {
  const mix = (params.get("mix") ?? "mixture") as "mixture" | "over";
  const groups = Number(params.get("groups") ?? 1);
  const mips = params.get("mips") !== "0";
  const kind = params.get("source") === "sparse" ? "sparse" : "dense";
  const [GW, GH, GD] = [24, 16, 12];
  const restore = patchShaders(device);
  const scene = await probeScene(device, { kind, groups, mips });
  const ms = MS_MODE;
  const settings = {
    lighting: "volume" as const,
    width: GW,
    height: GH,
    depth: GD,
    subSteps: 16,
    temporal: false,
    jitter: false,
    ...MS_SETTINGS,
    froxelMix: mix,
    phase: PHASE_NAME,
    fadeRadius: 9,
    srgbBlending: false,
    environmentLighting: (ENV_MODE ?? "directional") as "directional" | "spherical-harmonics" | "outward",
    environmentSamples: 6,
    environmentShOrder: 1,
  };
  const renderer = vdbCreateFrameRenderer(device, { settings, composite: { colorFormat: "rgba16float", depthFormat: "depth32float" } });
  // K6.2b `?env=`: the synthetic sky (rotated and reflected); every environment light is marched in each volume
  const envSource = ENV_MODE ? synthEnvironment(device, "equirect", "rgba16float") : null;
  const environment = envSource ? { id: 1, revision: 0, texture: envSource.texture, projection: "equirect" as const, rotation: Float32Array.from(probeEnvRotation()), intensity: 0.6 } : null;
  // A skips empty space (hdda; the dense source never skips); B wraps with an offset, so it never skips
  const pA: Partial<VDBFrameVolumeParams> = { matrixWorld: compose([0, 0, 0], 0.4, [2, 1.6, 2.2]), extinction: 4, shadowSteps: 12, mipmaps: mips, phaseG: 0.6, hdda: true };
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
    hdda: true,
  };
  const a = renderer.addVolume(scene.slots[0]!, pA);
  const b = renderer.addVolume(scene.slots[1]!, pB);
  const eyeWorld = lookAt([3.2, 1.6, 4.4], [0.2, 0, 0]);
  const view = vdbMat4Invert(eyeWorld);
  const projection = perspective(0.75, GW / GH, 0.1, 100);
  const lights = PROBE_LIGHTS;
  const occ = await probeOccluders(device);
  const occluders = occ?.inputs ?? [];
  const frame = (frameId: number) => ({ frameId, camera: { view, projection, near: 0.1, far: 100, id: 1 }, viewport: { drawingBuffer: [GW, GH] as [number, number], canvas: [GW, GH] as [number, number] }, lights, environment, occluders });
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
  restore();
  if (!u.recorded) throw new Error("update recorded nothing");
  const tex = vdbFrameRendererTextures(renderer)!;
  const scatter = halves(await readTexture(device, tex.scatter, [GW, GH, GD], 8));
  const env = environment ? parseEnvironment(await readBuffer(device, vdbFrameRendererEnvironment(renderer).buffer, VDB_FRAME_ENVIRONMENT.size)) : null;
  if (env) result.environment = { runs: [], count: env.count, mode: env.mode, valid: env.valid ? 1 : 0, directions: env.directions, colors: env.colors };

  // The twin: the renderer's CPU work redone with the same tools functions
  const clip = vdbFrameClipCamera(view, projection, { coordinateSystem: "webgpu" });
  const vols = [
    { h: a, sampler: scene.samplers[0]!, p: a.params },
    { h: b, sampler: scene.samplers[1]!, p: b.params },
  ];
  const clipped = vols.map((v) => ({ ...v, c: vdbFrameClipVolume(clip, v.p.matrixWorld)! }));
  const near = Math.max(Math.min(...clipped.map((v) => v.c.viewNear)), 0.1);
  const far = Math.min(Math.max(...clipped.map((v) => v.c.viewFar)), 100);
  const range: [number, number] = [near, Math.max(far, near * 1.0001)];
  const records: TwinVolume[] = clipped.map(({ h, sampler, p, c }) =>
    twinRecord(h, sampler, c, vdbFrameFroxelRect(c.localToView, projection, GW, GH)!, vdbFrameVolumeSlices(c.viewNear, c.viewFar, range, GD), p.phaseG ?? 0.8, ms),
  );
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
    probe: PROBE_SH,
    hemi: [{ sky: [0.3, 0.4, 0.6], ground: [0.2, 0.15, 0.1], up: [0, 1, 0] }],
    lights: lightRecords(lights),
    ...(env ? { environment: env } : {}),
    octaves: MS_OCTAVES,
    twoSided: MS_MODE === "two-sided",
  };
  const gatherCount = { n: 0 };
  if (occ && MUTATE !== "occluder") twinFrame.occluder = occluderTwin(occ, gatherCount);
  const stats = newStats(`2 f16 ulp (rgba16float scatter vs the f64 twin; ${kind} source${occ ? ", occluder maps" : ""})`);
  for (let z = 0; z < GD; z++)
    for (let y = 0; y < GH; y++)
      for (let x = 0; x < GW; x++) {
        const i = ((z * GH + y) * GW + x) * 4;
        const gpu = [scatter[i]!, scatter[i + 1]!, scatter[i + 2]!, scatter[i + 3]!];
        const cpu = gatherTwin(twinFrame, ordered, [x, y, z], mix, MUTATE);
        compare(stats, [x, y, z], gpu, cpu, 2, 2 ** -20);
      }
  result.stats = stats;

  // render(): the core composite pass over an empty depth (froxel grid = target size: taps on texel centres) vs the
  // two-tap twin on the read-back prefix (boundary D at an infinite depth)
  const color = device.createTexture({ size: [GW, GH], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const depthTex = device.createTexture({ size: [GW, GH], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  enc = device.createCommandEncoder();
  enc.beginRenderPass({ colorAttachments: [{ view: color.createView(), loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }], depthStencilAttachment: { view: depthTex.createView(), depthLoadOp: "clear", depthClearValue: 1, depthStoreOp: "store" } }).end();
  renderer.render(enc, { color: color.createView(), size: [GW, GH], depth: depthTex.createView() });
  device.queue.submit([enc.finish()]);
  const composite = halves(await readTexture(device, color, [GW, GH, 1], 8));
  const prefix = halves(await readTexture(device, tex.integrated, [GW, GH, GD + 1], 8));
  const rstats = newStats("2 f16 ulp (render() over an empty depth vs the two-tap twin on the read-back prefix)");
  for (let py = 0; py < GH; py++)
    for (let x = 0; x < GW; x++) {
      const y = GH - 1 - py;
      const cpu = integrateTwin((k) => {
        const j = ((k * GH + y) * GW + x) * 4;
        return [prefix[j]!, prefix[j + 1]!, prefix[j + 2]!, prefix[j + 3]!];
      }, 1e20, range, GD, MUTATE === "taps" ? "taps" : "none");
      const i = (py * GW + x) * 4;
      compare(rstats, [x, py], [composite[i]!, composite[i + 1]!, composite[i + 2]!, composite[i + 3]!], cpu[3] > 0 ? cpu : [0, 0, 0, 0]);
    }
  result.renderStats = rstats;
  if (occ) result.occluders = { maps: occ.inputs.length, transmittance: 0, density: 0, lighting: 0, gather: gatherCount.n };
  renderer.dispose();
  scene.dispose();
  envSource?.texture.destroy();
  occ?.dispose();
}

/**
 * `?probe=frame` (K6.2): 'frame' lighting on two visible volumes and one caster out of the drawn set (`&source=`,
 * `&groups=`, `&cascades=`, `&ms=0`): the shadow grid stack vs `vdbFrameGridDensityVoxel`, the transmittance stack vs
 * `vdbFrameTransmittanceVoxel3d` on the read-back grid, and the fused gather's scatter vs his density + lighting twins
 * (`vdbFrameDensityFroxel`, `vdbFrameLightingFroxel`) on the read-back transmittance. A second identical update must
 * skip the grid and the transmittance (his redraw signatures).
 */
async function probeFrame(device: GPUDevice): Promise<void> {
  const groups = Number(params.get("groups") ?? 1);
  const cascades = Number(params.get("cascades") ?? 1);
  // K6.3: diffusion (1 − g) for every truthy mode but octaves
  const ms = !!MS_MODE && MS_MODE !== "octaves";
  const kind = params.get("source") === "sparse" ? "sparse" : "dense";
  const [GW, GH, GD] = [24, 16, 12];
  const R = 16;
  const scene = await probeScene(device, { kind, groups, mips: true, extra: true });
  const g = 0.6;
  const backG = -0.3;
  const backWeight = 0.2;
  const exponent = Number(params.get("exponent") ?? 0.5);
  const settings = {
    lighting: "frame" as const,
    width: GW,
    height: GH,
    depth: GD,
    subSteps: 16,
    temporal: false,
    jitter: false,
    ...MS_SETTINGS,
    phase: PHASE_NAME,
    phaseG: g,
    phaseBackG: backG,
    phaseBackWeight: backWeight,
    phaseGExponent: exponent,
    fadeRadius: 9,
    srgbBlending: false,
    shadowResolution: R,
    shadowCascades: cascades,
    shadowDistance: 2.5,
    // K6.2b `?env=`: the environment lighting mode of a synthetic equirect sky (rotated and reflected)
    environmentLighting: (ENV_MODE ?? "directional") as "directional" | "spherical-harmonics" | "outward",
    environmentSamples: 8,
    environmentShOrder: 2,
  };
  const renderer = vdbCreateFrameRenderer(device, { settings });
  const envSource = ENV_MODE ? synthEnvironment(device, "equirect", "rgba16float") : null;
  const environment = envSource ? { id: 1, revision: 0, texture: envSource.texture, projection: "equirect" as const, rotation: Float32Array.from(probeEnvRotation()), intensity: 0.6 } : null;
  const pA: Partial<VDBFrameVolumeParams> = { matrixWorld: compose([0, 0, 0], 0.4, [2, 1.6, 2.2]), extinction: 4, absorbance: 0.7, mipmaps: true, hdda: true };
  const pB: Partial<VDBFrameVolumeParams> = {
    matrixWorld: compose([0.7, 0.25, -0.3], -0.3, [1.8, 1.5, 1.7]),
    extinction: 3,
    baseColor: [0.9, 0.8, 0.7],
    mipmaps: true,
    emissiveIntensity: 0.5,
    temperatureRange: [500, 3000],
    scatterColor: [0.01, 0.02, 0.03],
    densityCutoff: 0.05,
    wrap3D: "mirror",
    offset3D: [0.1, 0, -0.05],
  };
  // C: a caster that is not drawn (his collector: casters out of the view still shadow), between the sun and A; with
  // two cascades its box holds the camera, so the fine cascade around the camera is not empty
  const pC: Partial<VDBFrameVolumeParams> = { matrixWorld: compose(cascades > 1 ? [1.6, 1.2, 2.4] : [0.6, 1.6, 0.4], 0.2, cascades > 1 ? [4, 2.4, 5] : [1.4, 0.5, 1.2]), extinction: 5, mipmaps: true };
  const a = renderer.addVolume(scene.slots[0]!, pA);
  const b = renderer.addVolume(scene.slots[1]!, pB);
  const c = renderer.addVolume(scene.slots[2]!, pC);
  c.visible = false;
  const eyeWorld = lookAt([3.2, 1.6, 4.4], [0.2, 0, 0]);
  const view = vdbMat4Invert(eyeWorld);
  const projection = perspective(0.75, GW / GH, 0.1, 100);
  const lights = PROBE_LIGHTS;
  const occ = await probeOccluders(device);
  const occluders = occ?.inputs ?? [];
  const frame = (frameId: number) => ({ frameId, camera: { view, projection, near: 0.1, far: 100, id: 1 }, viewport: { drawingBuffer: [GW, GH] as [number, number], canvas: [GW, GH] as [number, number] }, lights, environment, occluders });
  let enc = device.createCommandEncoder();
  const u = renderer.update(enc, frame(0));
  device.queue.submit([enc.finish()]);
  if (!u.recorded) throw new Error("update recorded nothing");
  const grid = vdbFrameRendererGrid(renderer)!;
  const tex = vdbFrameRendererTextures(renderer)!;
  const gt = grid.textures;
  const shadowStack: VDBFrameTexture3d = { data: halves(await readTexture(device, gt.shadow, gt.shadowLayout.size, 8)), size: gt.shadowLayout.size, channels: 4 };
  const transStack: VDBFrameTexture3d = { data: halves(await readTexture(device, gt.transmittance, gt.transmittanceLayout.size, 8)), size: gt.transmittanceLayout.size, channels: 4 };
  const scatter = halves(await readTexture(device, tex.scatter, [GW, GH, GD], 8));
  // K6.2b: the environment buffer and grid the kernels read (the buffer is checked by ?probe=environment)
  const env = parseEnvironment(await readBuffer(device, vdbFrameRendererEnvironment(renderer).buffer, VDB_FRAME_ENVIRONMENT.size));
  const envGridTex: VDBFrameTexture3d = { data: halves(await readTexture(device, gt.environment, [R, R, R], 8)), size: [R, R, R], channels: 4 };
  // K6.3: the scattered-light term's mips ('mlp': rgba16float box mips; 'multi-scale': K7.3's rg16float chain)
  const sc = vdbFrameRendererScattering(renderer);
  const mlpMips: VDBFrameTexture3d[] = [];
  if (sc.mips && MS_MODE !== true && (MS_MODE === "mlp" || MS_MODE === "neural")) {
    for (let k = 0; k < sc.mips.mipLevelCount; k++) {
      const side = Math.max(1, sc.mips.width >> k);
      mlpMips.push({ data: halves(await readTexture(device, sc.mips, [side, side, side], 8, k)), size: [side, side, side], channels: 4 });
    }
  }
  const msMips: VDBFrameTexture3d[] = [];
  if (sc.multiScale && MS_MODE === "multi-scale") {
    for (const [k, m] of sc.multiScale.layout.mips.entries()) {
      msMips.push({ data: halves(await readTexture(device, sc.multiScale.texture, m.dimensions, 4, k)), size: m.dimensions, channels: 2 });
    }
  }
  enc = device.createCommandEncoder();
  renderer.update(enc, frame(1));
  device.queue.submit([enc.finish()]);
  result.frame = { cascades: grid.count, redraw: JSON.parse(JSON.stringify(renderer.redrawStats)) };

  const layout: VDBFrameGridLayout = {
    count: grid.count,
    resolution: R,
    columns: vdbFrameAtlasLayout(R, R, R * cascades).columns,
    atlasSize: [1, 1],
    gridMin: grid.gridMin as VDBFrameVec3[],
    gridSize: grid.gridSize as VDBFrameVec3[],
    gridVoxel: grid.gridVoxel as VDBFrameVec3[],
  };
  const clip = vdbFrameClipCamera(view, projection, { coordinateSystem: "webgpu" });
  const handles = [a, b, c];
  const clips = handles.map((h) => vdbFrameClipVolume(clip, h.params.matrixWorld));
  const visibleClips = clips.slice(0, 2).map((x) => x!);
  const near = Math.max(Math.min(...visibleClips.map((v) => v.viewNear)), 0.1);
  const far = Math.min(Math.max(...visibleClips.map((v) => v.viewFar)), 100);
  const range: [number, number] = [near, Math.max(far, near * 1.0001)];
  const records = handles.map((h, i) => {
    const cl = clips[i] ?? { localToView: vdbMat4Multiply(view, h.params.matrixWorld), viewToLocal: vdbMat4Invert(vdbMat4Multiply(view, h.params.matrixWorld)), worldToLocal: vdbMat4Invert(h.params.matrixWorld), viewNear: 0, viewFar: 0 };
    const rect = i < 2 ? vdbFrameFroxelRect(cl.localToView, projection, GW, GH)! : ([0, 0, 0, 0] as [number, number, number, number]);
    return twinRecord(h, scene.samplers[i]!, cl, rect, i < 2 ? vdbFrameVolumeSlices(cl.viewNear, cl.viewFar, range, GD) : { first: 0, count: 0 }, 0.8);
  });
  const readLod = (r: TwinVolume, lod: number) => (r.mipmaps ? lod : 0);

  // (a) the shadow grid
  const gstats = newStats("2 f16 ulp (rgba16float shadow grid vs vdbFrameGridDensityVoxel)");
  const casters = records.map((r) => ({
    worldToLocal: r.worldToLocal,
    maxTexels: Math.max(...r.volume.texels),
    worldScale: r.worldScale,
    mipmaps: MUTATE === "grid-lod" ? false : r.mipmaps,
    absorbance: MUTATE === "grid-absorbance" ? 1 : r.absorbance,
    extinctionAt: (local: VDBFrameVec3, lod: number) => extinctionAt(r, local, lod),
    albedoAt: (local: VDBFrameVec3, lod: number) => albedoAt(r, local, lod),
  }));
  const at = (t: VDBFrameTexture3d, p: readonly number[]) => {
    const i = (p[0]! + (p[1]! + p[2]! * t.size[1]) * t.size[0]) * 4;
    return [t.data[i]!, t.data[i + 1]!, t.data[i + 2]!, t.data[i + 3]!];
  };
  for (let cc = 0; cc < layout.count; cc++) {
    const voxel = layout.gridVoxel[cc]!;
    for (let z = 0; z < R; z++)
      for (let y = 0; y < R; y++)
        for (let x = 0; x < R; x++) {
          const world = vdbFrameGridVoxelWorld(layout, cc, [x, y, z]);
          const e = vdbFrameGridDensityVoxel(world, Math.max(...voxel), casters);
          const o = gt.shadowLayout.origins[cc]!;
          compare(gstats, [cc, x, y, z], at(shadowStack, [o[0] + x, o[1] + y, o[2] + z]), [e[0], e[1], 0, 0], 2, 2 ** -20);
        }
  }
  result.gridStats = gstats;

  // (b) the transmittance from the read-back grid
  const phaseFields = { phase: PHASE_NAME, phaseG: g, phaseBackG: backG, phaseBackWeight: backWeight, phaseGExponent: exponent };
  const msValue = ms ? vdbFrameMultipleScatteringValue(MS_MODE, phaseFields) : 0;
  const lastCascade = layout.count - 1;
  const lastOrigin = layout.count > 0 ? gt.shadowLayout.origins[lastCascade]! : [0, 0, 0];

  // (a') K6.3 the scattered-light term's mips, each from the read-back level before it
  let slotTotal: ((direct: number, world: VDBFrameVec3, direction: VDBFrameVec3, opticalDepth: number, maxDistance: number) => number) | null = null;
  if (mlpMips.length > 0) {
    const mstats = newStats("2 f16 ulp (rgba16float 'mlp' feature mips vs vdbFrameMlpMipTexel on the read-back level before)");
    mlpMips.forEach((m, k) => {
      const source = k === 0 ? shadowStack : mlpMips[k - 1]!;
      const origin = k === 0 ? lastOrigin : [0, 0, 0];
      const size = k === 0 ? [R, R, R] : source.size;
      for (let z = 0; z < m.size[2]; z++)
        for (let y = 0; y < m.size[1]; y++)
          for (let x = 0; x < m.size[0]; x++) {
            // ?mutate=mlp-mip: the box one texel off along x
            const cpu = vdbFrameMlpMipTexel(source, MUTATE === "mlp-mip" ? [origin[0]! + 1, origin[1]!, origin[2]!] : origin, size, [x, y, z]);
            compare(mstats, [k, x, y, z], at(m, [x, y, z]).slice(0, 2), cpu, 2, 2 ** -20);
          }
    });
    result.mipStats = mstats;
    const mlp = sc.mlp!;
    const weights = MUTATE === "mlp-weights" ? Float32Array.from(mlp.weights, (w, i) => (i === 7 ? w + 0.5 : w)) : mlp.weights;
    const net: VDBFrameMlp = { ...mlp, weights };
    const h = Math.max(...layout.gridVoxel[lastCascade]!);
    const sample = (level: number, world: VDBFrameVec3) => vdbFrameMlpSampleLevel(layout, shadowStack, lastOrigin, mlpMips, level, world);
    slotTotal = (direct, world, direction, opticalDepth) => direct + vdbFrameMlpForward(net, vdbFrameMlpInputs(sample, net.levels, h, world, direction, MUTATE === "mlp-depth" ? 0 : opticalDepth));
  }
  if (msMips.length > 0) {
    const levels = MS_SETTINGS.multiScaleLevels;
    const edge = MUTATE === "ms-edge" ? (MS_SETTINGS.multiScaleEdge === "clamp" ? "zero" : "clamp") : MS_SETTINGS.multiScaleEdge;
    // the chain from the read-back last cascade (K7.3's twin), then the lookup twin on the read-back chain
    const base = { data: shadowStack.data, dimensions: [...shadowStack.size] as VDBFrameVec3, channels: 4 };
    const cpuMips = vdbFrameMultiScaleMips(base, { levels, edge, origin: [lastOrigin[0]!, lastOrigin[1]!, lastOrigin[2]!], size: [R, R, R], channels: [0, 1] });
    const mstats = newStats("1 f16 ulp (K7.3 rg16float multi-scale mips of the last cascade vs vdbFrameMultiScaleMips on the read-back grid)");
    msMips.forEach((m, k) => {
      const c = cpuMips.levels[k]!;
      for (let i = 0; i < m.size[0] * m.size[1] * m.size[2]; i++) compare(mstats, [k, i], [m.data[i * 2]!, m.data[i * 2 + 1]!], [c[i * 2]!, c[i * 2 + 1]!], 1, 2 ** -20);
    });
    result.mipStats = mstats;
    const textures = { base, baseOrigin: [lastOrigin[0]!, lastOrigin[1]!, lastOrigin[2]!] as VDBFrameVec3, baseSize: [R, R, R] as VDBFrameVec3, baseChannels: [0, 1], mips: msMips.map((m) => ({ data: m.data, dimensions: [...m.size] as VDBFrameVec3, channels: 2 })) };
    const box = { boxMin: layout.gridMin[lastCascade]!, boxSize: layout.gridSize[lastCascade]! };
    const msGrid = { unit: Math.max(...layout.gridVoxel[lastCascade]!), levels: MUTATE === "ms-levels" ? 2 : levels, multipleScattering: msValue };
    slotTotal = (direct, world, direction, _depth, maxDistance) => Math.min(direct + vdbFrameMultiScaleLight((k, p) => vdbFrameMultiScaleSample(textures, box, k, p), msGrid, world, direction, maxDistance), 1);
  }
  const slotLights = lightRecords(lights).slice(0, 6).map((l) => l.position);
  // K6.2b: his light result: the dominant 'directional' environment light in the next free slot (not with diffusion)
  const sceneSlotCount = slotLights.length;
  const envFlags = {
    useShadows: grid.count > 0,
    slot: !!environment && settings.environmentLighting === "directional" && sceneSlotCount < 6 && !ms && env.valid,
    grid: !!environment && settings.environmentLighting !== "outward" && grid.count > 0,
    diffusion: ms,
  };
  const envSlotUsed = envFlags.slot && env.count > 0;
  if (envSlotUsed && MUTATE !== "env-slot") slotLights.push([...env.directions[0]!, 1]);
  // K6.5b: the scene slots' occluder maps scale the scattered part at each voxel (envSlot: no map)
  const counts = { transmittance: { n: 0 }, density: { n: 0 }, lighting: { n: 0 } };
  const transmittanceOccluder = occ && MUTATE !== "occluder" && MUTATE !== "occluder-scattered" ? occluderTwin(occ, counts.transmittance) : null;
  const densityOccluder = occ && MUTATE !== "occluder" ? occluderTwin(occ, counts.density) : null;
  const lightingOccluder = occ && MUTATE !== "occluder" ? occluderTwin(occ, counts.lighting) : null;
  const tstats = newStats("2 f16 ulp (rgba16float transmittance vs vdbFrameTransmittanceVoxel3d on the read-back grid)");
  const gridAt = (cc: number, uvw: VDBFrameVec3): [number, number] => {
    const s = vdbFrameStackSample(shadowStack, gt.shadowLayout.origins[cc]!, R, uvw);
    return [s[0]!, s[1]!];
  };
  for (let cc = 0; cc < layout.count; cc++)
    for (let z = 0; z < R; z++)
      for (let y = 0; y < R; y++)
        for (let x = 0; x < R; x++) {
          const t = vdbFrameTransmittanceVoxel3d(gridAt, layout, cc, [x, y, z], slotLights, [0, 1, 0], MUTATE === "diffusion" ? 0 : msValue, -1, transmittanceOccluder, {
            octaves: MUTATE === "octaves" ? [0, 0, 0, 0] : MS_OCTAVES,
            twoSided: MS_MODE === "two-sided" && MUTATE !== "two-sided",
            slotTotal,
          });
          for (let k = 0; k < 4; k++) {
            const o = gt.transmittanceLayout.origins[4 * cc + k]!;
            compare(tstats, [cc, k, x, y, z], at(transStack, [o[0] + x, o[1] + y, o[2] + z]), t[k]!, 2, 2 ** -20);
          }
        }
  result.transmittanceStats = tstats;

  // (b') K6.2b: the environment grid from the read-back grid and buffer
  if (envFlags.grid && env.valid) {
    const first = envFlags.slot ? 1 : 0;
    const gridLights = env.directions.map((direction, i) => ({ direction, color: env.colors[i]! })).slice(MUTATE === "env-grid" ? first + 1 : first);
    const es = newStats("2 f16 ulp (rgba16float environment grid vs vdbFrameEnvironmentGridVoxel3d on the read-back grid)");
    for (let z = 0; z < R; z++)
      for (let y = 0; y < R; y++)
        for (let x = 0; x < R; x++) {
          const cpu = vdbFrameEnvironmentGridVoxel3d(gridAt, layout, [x, y, z], gridLights, msValue, MUTATE === "octaves" ? [0, 0, 0, 0] : MS_OCTAVES);
          compare(es, [x, y, z], at(envGridTex, [x, y, z]), cpu, 2, 2 ** -20);
        }
    result.environmentGridStats = es;
  }

  // (c) the fused gather from the read-back transmittance
  const shadowsAt =
    layout.count > 0
      ? (world: VDBFrameVec3) => vdbFrameStackShadows((cc, k, uvw) => vdbFrameStackSample(transStack, gt.transmittanceLayout.origins[4 * cc + k]!, R, uvw), layout, world)
      : null;
  // K6.3: the renderer's phase packing (?phase=; ?mutate=mie drops the Lorenz-Mie table)
  const phase = vdbFramePhaseVector(phaseFields) as VDBFrameVec4;
  if (MUTATE === "mie") phase[3] = 0;
  const phaseSpectrum = vdbFramePhaseSpectrum(phaseFields) as VDBFrameVec3;
  const octaves: VDBFrameVec4 = MUTATE === "octaves" ? [0, 0, 0, 0] : MS_OCTAVES;
  const sorted = lightRecords(lights);
  const sceneSlots = Math.min(sorted.length, 6);
  const projectionInverse = vdbMat4Invert(projection);
  const columns = vdbFrameAtlasLayout(GW, GH, GD).columns;
  const stats = newStats(`2 f16 ulp (rgba16float scatter vs his density + lighting twins; ${kind} source, ${groups} group(s), ${layout.count} cascade(s))`);
  for (let z = 0; z < GD; z++)
    for (let y = 0; y < GH; y++)
      for (let x = 0; x < GW; x++) {
        const tile: [number, number] = [z % columns, Math.floor(z / columns)];
        const fragCoord: [number, number] = [tile[0] * GW + x + 0.5, tile[1] * GH + y + 0.5];
        const sums: [VDBFrameVec4, VDBFrameVec4, VDBFrameVec4, VDBFrameVec4] = [
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
        ];
        for (const r of records.slice(0, 2)) {
          if (x < r.rect[0] || y < r.rect[1] || x >= r.rect[2] || y >= r.rect[3] || z < r.first || z >= r.first + r.count) continue;
          const d = vdbFrameDensityFroxel(fragCoord, {
            froxelGrid: [GW, GH, GD],
            columns,
            range,
            projectionInverse,
            cameraWorld: eyeWorld,
            viewToLocal: r.viewToLocal,
            maxTexels: Math.max(...r.volume.texels),
            mipLevels: r.mipmaps ? r.levels : 0,
            worldScale: r.worldScale,
            subSteps: settings.subSteps,
            jitter: -1,
            fadeRadius: settings.fadeRadius,
            opacity: r.opacity,
            baseColor: r.baseColor,
            scatterColor: r.scatterColor,
            extinctionAt: (local, lod) => extinctionAt(r, local, readLod(r, lod), MUTATE),
            albedoAt: (local, lod) => albedoAt(r, local, readLod(r, lod)),
            emissionAt: (local, lod) => emissionAt(r, local, readLod(r, lod)),
            shadows: null,
            shadowsAt,
            sceneLightCount: MUTATE === "slots" ? 0 : sceneSlots,
            lightPositions: sorted.map((l) => l.position),
            phase,
            phaseSpectrum,
            octaves,
            occluderAt: densityOccluder,
            environmentAt: environment
              ? (world, local, skyTotals) => {
                  const outward = MUTATE !== "env-outward" && local && Math.hypot(...local) > 1e-4 ? normalizeV(mat3Of(eyeWorld, mat3Of(r.localToView, local))) : ([0, 1, 0] as VDBFrameVec3);
                  const last = layout.count - 1;
                  const gridAt = () => {
                    if (last < 0) return null;
                    const uvw = vdbFrameGridUvw(layout, last, world);
                    return vdbFrameInsideGrid(R, uvw) ? (vdbFrameStackSample(envGridTex, [0, 0, 0], R, uvw).slice(0, 3) as VDBFrameVec3) : null;
                  };
                  return vdbFrameEnvironmentDensityTerm(env, envFlags, outward, [0, 1, 0], skyTotals, gridAt, octaves);
                }
              : null,
          });
          if (!d) continue;
          for (let k = 0; k < 4; k++) for (let ch = 0; ch < 4; ch++) sums[k]![ch]! += d[k]![ch]!;
        }
        const f = vdbFrameFroxel([GW, GH, GD], columns, range, projectionInverse, fragCoord);
        const cpu = vdbFrameLightingFroxel(sums, f.viewNear, f.viewFar, eyeWorld, {
          ambient: [0.05, 0.06, 0.08],
          hemiSky: [0.3, 0.4, 0.6],
          hemiGround: [0.2, 0.15, 0.1],
          lights: sorted.map((l) => ({ position: l.position, color: [l.color[0], l.color[1], l.color[2]] as VDBFrameVec3, spot: l.spot, penumbra: l.color[3], falloff: l.falloff })),
          shadowLightCount: sceneSlots,
          phase,
          phaseSpectrum,
          octaves,
          directAt: shadowsAt ? (world) => {
            const s = shadowsAt(world);
            return [s[2]![0]!, s[2]![1]!, s[2]![2]!, s[2]![3]!, s[3]![0]!, s[3]![1]!];
          } : () => [1, 1, 1, 1, 1, 1],
          environmentColors: envFlags.useShadows && envSlotUsed && MUTATE !== "env-slot" ? [env.colors[0]!] : [],
          environmentShadowSlot: sceneSlots,
          occluderAt: lightingOccluder,
          ...(MUTATE === "sky" ? {} : { probe: PROBE_SH, sky: [0, 1, 0] as VDBFrameVec3 }),
        });
        const i = ((z * GH + y) * GW + x) * 4;
        compare(stats, [x, y, z], [scatter[i]!, scatter[i + 1]!, scatter[i + 2]!, scatter[i + 3]!], cpu, 2, 2 ** -20);
      }
  result.stats = stats;
  result.environment = { runs: [{ ...vdbFrameRendererEnvironment(renderer).runs }], count: env.count, mode: env.mode, valid: env.valid ? 1 : 0, directions: env.directions, colors: env.colors };
  if (occ) result.occluders = { maps: occ.inputs.length, transmittance: counts.transmittance.n, density: counts.density.n, lighting: counts.lighting.n, gather: 0 };
  renderer.dispose();
  scene.dispose();
  envSource?.texture.destroy();
  occ?.dispose();
}

const normalizeV = (v: readonly number[]): VDBFrameVec3 => {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  return [v[0]! / l, v[1]! / l, v[2]! / l];
};
/** The upper 3x3 of a column-major 4x4 times v. */
const mat3Of = (m: ArrayLike<number>, v: readonly number[]): VDBFrameVec3 => [0, 1, 2].map((r) => m[r]! * v[0]! + m[4 + r]! * v[1]! + m[8 + r]! * v[2]!) as VDBFrameVec3;

/** The lights as the renderer packs them (directional, point, spot; position w kind). */
function lightRecords(lights: readonly VDBFrameLight[]): (TwinFrame["lights"][number] & { falloff: [number, number] })[] {
  const n = (v: readonly number[]) => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!);
    return [v[0]! / l, v[1]! / l, v[2]! / l];
  };
  const out: (TwinFrame["lights"][number] & { falloff: [number, number] })[] = [];
  for (const kind of ["directional", "point", "spot"] as const) {
    for (const l of lights) {
      if (l.kind !== kind) continue;
      if (l.kind === "directional") {
        const d = n(l.direction);
        out.push({ position: [-d[0]!, -d[1]!, -d[2]!, 1], color: [...l.color, 0], spot: [0, 0, 0, 0], falloff: [0, 0] });
      } else if (l.kind === "point") out.push({ position: [...l.position, 2], color: [...l.color, 0], spot: [0, 0, 0, 0], falloff: [l.distance, l.decay] });
      else if (l.kind === "spot") {
        const d = n(l.direction);
        out.push({ position: [...l.position, 3], color: [...l.color, l.cosPenumbra], spot: [-d[0]!, -d[1]!, -d[2]!, l.cosAngle], falloff: [l.distance, l.decay] });
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

/** Group 0 binding 3 (K6.2b): an empty environment buffer for the probes that run one kernel by hand. */
function envEntry(device: GPUDevice): GPUBindGroupEntry {
  return { binding: 3, resource: { buffer: device.createBuffer({ size: VDB_FRAME_ENVIRONMENT.size, usage: GPUBufferUsage.UNIFORM }) } };
}

/** Group 0 binding 4 (K6.3): the Lorenz-Mie table (binary16 pairs in a uniform buffer), as the renderer makes it. */
let mieLut: GPUBuffer | null = null;
function mieEntry(device: GPUDevice): GPUBindGroupEntry {
  if (!mieLut) {
    const words = vdbFrameMieLutWords();
    mieLut = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(mieLut, 0, words.buffer as ArrayBuffer);
  }
  return { binding: 4, resource: { buffer: mieLut } };
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
  pass.setBindGroup(0, device.createBindGroup({ layout: built.layouts[0]!, entries: [{ binding: 0, resource: { buffer: frame } }, { binding: 1, resource: { buffer: lights } }, { binding: 2, resource: sampler }, envEntry(device), mieEntry(device)] }));
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
  pass.setBindGroup(0, device.createBindGroup({ layout: built.layouts[0]!, entries: [{ binding: 0, resource: { buffer: frame } }, { binding: 1, resource: { buffer: lights } }, { binding: 2, resource: sampler }, envEntry(device), mieEntry(device)] }));
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

// ------------------------------------------------------------------ K6.2b environment

/** |gpu − cpu| ≤ rel · |cpu| + abs per value (f32 outputs); `maxUlp` holds the largest relative error. */
function compareRel(stats: Stats, at: number[], gpu: readonly number[], cpu: readonly number[], rel: number, abs: number): void {
  stats.compared++;
  if (cpu.some((v) => v !== 0) || gpu.some((v) => v !== 0)) stats.covered++;
  let bad = false;
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(gpu[i]! - cpu[i]!);
    stats.maxAbs = Math.max(stats.maxAbs, d);
    if (Math.abs(cpu[i]!) > abs) stats.maxUlp = Math.max(stats.maxUlp, d / Math.abs(cpu[i]!));
    if (!(d <= rel * Math.abs(cpu[i]!) + abs)) bad = true;
  }
  if (bad) {
    stats.over++;
    stats.firstBad ??= { at, gpu: [...gpu], cpu: [...cpu] };
  }
}

/** The synthetic host environment of the probes: a sky with a sun and a red blob (equirect) or six graded faces (cube). */
function synthEnvironment(device: GPUDevice, projection: "equirect" | "cube", format: "rgba16float" | "rgba32float"): { texture: GPUTexture; cpu: VDBFrameEnvImage[]; size: number } {
  const round = format === "rgba16float" ? roundHalf : (x: number) => Math.fround(x);
  const upload = (texture: GPUTexture, data: Float64Array, w: number, h: number, layer: number) => {
    const bytes = format === "rgba16float" ? Uint16Array.from(data, (v) => vdbFloatToHalf(v)) : Float32Array.from(data);
    device.queue.writeTexture({ texture, origin: [0, 0, layer] }, bytes, { bytesPerRow: w * (format === "rgba16float" ? 8 : 16), rowsPerImage: h }, [w, h, 1]);
  };
  if (projection === "equirect") {
    const [w, h] = [300, 150];
    const data = new Float64Array(w * h * 4);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const u = (x + 0.5) / w;
        const v = (y + 0.5) / h;
        let c = v < 0.5 ? [0.2 + 0.3 * (1 - v), 0.3 + 0.4 * (1 - v), 0.6 + 0.3 * (1 - v)] : [0.15 * v, 0.12 * v, 0.1 * v];
        if (Math.hypot((u - 0.7) * w, (v - 0.3) * h) < 2.2) c = [40, 36, 30];
        if (Math.hypot((u - 0.2) * w, (v - 0.45) * h) < 8) c = [2, 0.3, 0.2];
        for (let k = 0; k < 3; k++) data[(y * w + x) * 4 + k] = round(c[k]!);
        data[(y * w + x) * 4 + 3] = 1;
      }
    const texture = device.createTexture({ size: [w, h], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    upload(texture, data, w, h, 0);
    return { texture, cpu: [{ data, width: w, height: h }], size: w };
  }
  const size = 40;
  const texture = device.createTexture({ size: [size, size, 6], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const cpu: VDBFrameEnvImage[] = [];
  for (let f = 0; f < 6; f++) {
    const data = new Float64Array(size * size * 4);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        let c = [0.2 + 0.1 * f, 0.3 + (0.4 * y) / size, 0.5 + (0.3 * x) / size];
        if (f === 0 && Math.hypot(x - 12, y - 15) < 1.6) c = [30, 28, 25];
        for (let k = 0; k < 3; k++) data[(y * size + x) * 4 + k] = round(c[k]!);
        data[(y * size + x) * 4 + 3] = 1;
      }
    upload(texture, data, size, size, f);
    cpu.push({ data, width: size, height: size });
  }
  return { texture, cpu, size };
}

/** A world → map 3x3 (column-major): a reflection of y (three's flipY equirect) times rotations about y and x. */
function probeEnvRotation(): number[] {
  const [a, b] = [0.7, 0.3];
  const ry = [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
  const rx = [1, 0, 0, 0, Math.cos(b), -Math.sin(b), 0, Math.sin(b), Math.cos(b)];
  const mul = (p: number[], q: number[]) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => p[r * 3]! * q[c]! + p[r * 3 + 1]! * q[3 + c]! + p[r * 3 + 2]! * q[6 + c]!));
  const m = mul([1, 0, 0, 0, -1, 0, 0, 0, 1], mul(ry, rx)); // row-major
  return [0, 1, 2].flatMap((c) => [0, 1, 2].map((r) => m[r * 3 + c]!));
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, read, 0, size);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = read.getMappedRange().slice(0);
  read.unmap();
  read.destroy();
  return out;
}

/** The environment buffer as plain values. */
function parseEnvironment(bytes: ArrayBuffer): VDBFrameEnvironmentGpuResult & { info: number[] } {
  const u32 = new Uint32Array(bytes);
  const f32 = new Float32Array(bytes);
  const at = (name: string) => VDB_FRAME_ENVIRONMENT.fields.get(name)!.offset / 4;
  const vec = (name: string, i: number): VDBFrameVec3 => [f32[at(name) + i * 4]!, f32[at(name) + i * 4 + 1]!, f32[at(name) + i * 4 + 2]!];
  const info = [u32[0]!, u32[1]!, u32[2]!, u32[3]!];
  const count = info[0]!;
  return {
    info,
    valid: info[2] === 1,
    mode: info[1]!,
    count,
    sh: Array.from({ length: 9 }, (_, k) => vec("sh", k)),
    directions: Array.from({ length: count }, (_, i) => vec("direction", i)),
    colors: Array.from({ length: count }, (_, i) => vec("color", i)),
  };
}

/** Compares the environment buffer with its twin (lights by index: same count, directions, colours; the SH). */
function compareEnvironment(stats: Stats, gpu: VDBFrameEnvironmentGpuResult, cpu: VDBFrameEnvironmentGpuResult): void {
  compareRel(stats, [-1], [gpu.valid ? 1 : 0, gpu.mode, gpu.count], [cpu.valid ? 1 : 0, cpu.mode, cpu.count], 0, 0);
  const scale = Math.max(1e-6, ...cpu.colors.flat().map(Math.abs), ...cpu.sh.flat().map(Math.abs));
  for (let i = 0; i < Math.min(gpu.count, cpu.count); i++) {
    compareRel(stats, [0, i], gpu.directions[i]!, cpu.directions[i]!, 0, 2e-3);
    compareRel(stats, [1, i], gpu.colors[i]!, cpu.colors[i]!, 2e-3, 1e-5 * scale);
  }
  for (let k = 0; k < 9; k++) compareRel(stats, [2, k], gpu.sh[k]!, cpu.sh[k]!, 2e-3, 1e-5 * scale);
}

/** Reads the environment state's maps and buffer and compares them with the twins (`?probe=environment`). */
async function checkEnvironment(
  device: GPUDevice,
  maps: { work: GPUTexture; map: GPUTexture },
  buffer: GPUBuffer,
  o: { projection: "equirect" | "cube"; source: VDBFrameEnvImage[]; size: number; rotation: number[]; settings: { environmentLighting: string; environmentSamples: number; environmentShOrder: number; pairs: boolean; intensity: number } },
): Promise<VDBFrameEnvironmentGpuResult> {
  const WW = VDB_FRAME_ENV_WORK_WIDTH;
  const WH = VDB_FRAME_ENV_WORK_HEIGHT;
  const work = new Float32Array((await readTexture(device, maps.work, [WW, WH, 1], 16)).buffer);
  const map = new Float32Array((await readTexture(device, maps.map, [vdbFrameEnvironmentMapWidth, vdbFrameEnvironmentMapHeight, 1], 16)).buffer);
  const gpu = parseEnvironment(await readBuffer(device, buffer, VDB_FRAME_ENVIRONMENT.size));
  // (a) the work map from the host texture (f16-rounded source as uploaded)
  const ws = newStats("1e-3 relative + 1e-6 (rgba32float work map vs vdbFrameEnvironmentArea/CubeTexel)");
  const workW = MUTATE === "env-area" ? WW + 1 : WW;
  for (let y = 0; y < WH; y++)
    for (let x = 0; x < WW; x++) {
      const cpu = o.projection === "cube" ? vdbFrameEnvironmentCubeTexel(o.source, MUTATE === "env-area" ? o.size + 8 : o.size, x, y) : vdbFrameEnvironmentAreaTexel(o.source[0]!, x, y, workW);
      const i = (y * WW + x) * 4;
      compareRel(ws, [x, y], [work[i]!, work[i + 1]!, work[i + 2]!], cpu.slice(0, 3), 1e-3, 1e-6);
    }
  result.workStats = ws;
  // (b) his 64 x 32 map from the read-back work map
  const workImage: VDBFrameEnvImage = { data: work, width: WW, height: WH };
  const ms = newStats("1e-3 relative + 1e-6 (rgba32float map vs vdbFrameEnvironmentBlurTexel on the read-back work map)");
  for (let y = 0; y < vdbFrameEnvironmentMapHeight; y++)
    for (let x = 0; x < vdbFrameEnvironmentMapWidth; x++) {
      const cpu = vdbFrameEnvironmentBlurTexel(workImage, o.rotation, x, y, MUTATE === "env-blur" ? 0.3 : 0.4);
      const i = (y * vdbFrameEnvironmentMapWidth + x) * 4;
      compareRel(ms, [x, y], [map[i]!, map[i + 1]!, map[i + 2]!], cpu.slice(0, 3), 1e-3, 1e-6);
    }
  result.mapStats = ms;
  // (c) the buffer from the read-back maps
  const sh = vdbFrameEnvironmentIrradianceSh(workImage, MUTATE === "env-sh" ? [0, 0, -1, 0, 1, 0, 1, 0, 0] : o.rotation);
  const cpu = vdbFrameEnvironmentReduce(map, sh, { ...o.settings, pairs: MUTATE === "env-seeds" ? !o.settings.pairs : o.settings.pairs });
  const bs = newStats("count exact; directions 2e-3; colours and SH 2e-3 relative (environment buffer vs vdbFrameEnvironmentReduce on the read-back maps)");
  compareEnvironment(bs, gpu, cpu);
  result.bufferStats = bs;
  return gpu;
}

async function probeEnvironment(device: GPUDevice): Promise<void> {
  const projection: "equirect" | "cube" = params.get("projection") === "cube" ? "cube" : "equirect";
  const format = params.get("format") === "rgba32float" ? "rgba32float" : "rgba16float";
  const mode = (params.get("mode") ?? "directional") as "directional" | "spherical-harmonics" | "outward";
  const pairs = params.get("pairs") === "1";
  const order = Number(params.get("order") ?? 2);
  const rotation = params.get("rotate") === "1" ? probeEnvRotation() : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const src = synthEnvironment(device, projection, format);
  const state = new VDBFrameEnvironmentState(device, "k6 probe");
  const settings = { environmentLighting: mode, environmentSamples: Number(params.get("samples") ?? 8), environmentShOrder: order, multipleScattering: pairs };
  const env = { id: 1, revision: 0, texture: src.texture, projection, rotation: Float32Array.from(rotation), intensity: 0.7 };
  const run = (e: typeof env, s: typeof settings) => {
    state.prepare(e, s);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    state.record(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
    return { ...state.runs };
  };
  const runs: unknown[] = [run(env, settings)];
  const gpu = await checkEnvironment(device, state.maps!, state.buffer, { projection, source: src.cpu, size: src.size, rotation: Array.from(Float32Array.from(rotation)), settings: { ...settings, pairs, intensity: 0.7 } });
  // the stages rerun only for their inputs: identical (nothing), intensity (reduce), rotation (blur + reduce), revision (all)
  runs.push(run(env, settings), run({ ...env, intensity: 0.5 }, settings), run({ ...env, intensity: 0.5, rotation: Float32Array.from([0, 0, 1, 0, 1, 0, -1, 0, 0]) }, settings), run({ ...env, revision: 1 }, settings));
  result.environment = { runs, count: gpu.count, mode: gpu.mode, valid: gpu.valid ? 1 : 0, directions: gpu.directions, colors: gpu.colors };
  result.stats = result.workStats!;
  state.dispose();
  src.texture.destroy();
}

// ------------------------------------------------------------------ K6.5: volume shadow maps, emission

/** |gpu − cpu| ≤ rel · max(|cpu|, scale) per value (f32 outputs: the direction and emitter textures, receivers). */
function compareScaled(stats: Stats, at: number[], gpu: readonly number[], cpu: readonly number[], rel: number, scale: number): void {
  stats.compared++;
  if (cpu.some((v) => v !== 0) || gpu.some((v) => v !== 0)) stats.covered++;
  let bad = false;
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(gpu[i]! - cpu[i]!);
    stats.maxAbs = Math.max(stats.maxAbs, d);
    const r = d / Math.max(Math.abs(cpu[i]!), scale, 1e-30);
    stats.maxUlp = Math.max(stats.maxUlp, r);
    if (!(r <= rel)) bad = true;
  }
  if (bad) {
    stats.over++;
    stats.firstBad ??= { at, gpu: [...gpu], cpu: [...cpu] };
  }
}

/** rgba32float texels as numbers. */
const floats = (bytes: Uint8Array): Float64Array => Float64Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));

/** A clamp-to-edge bilinear read of one layer of a read-back rgba texture (s, t in [0, 1], t down the rows). */
function bilinearLayer(data: ArrayLike<number>, size: number, layer: number, s: number, t: number): [number, number, number, number] {
  const x = s * size - 0.5;
  const y = t * size - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const at = (xi: number, yi: number, c: number) => data[((layer * size + Math.min(size - 1, Math.max(0, yi))) * size + Math.min(size - 1, Math.max(0, xi))) * 4 + c]!;
  return [0, 1, 2, 3].map((c) => (at(x0, y0, c) * (1 - fx) + at(x0 + 1, y0, c) * fx) * (1 - fy) + (at(x0, y0 + 1, c) * (1 - fx) + at(x0 + 1, y0 + 1, c) * fx) * fy) as [number, number, number, number];
}

/** Runs a receiver probe kernel (`code`, entry `entry`) over `rows` (vec4f each) with its four group-0 bindings. */
async function runRowProbe(device: GPUDevice, code: string, entry: string, uniform: GPUBuffer, view: GPUTextureView, rows: Float32Array<ArrayBuffer>, viewDimension: GPUTextureViewDimension): Promise<Float32Array> {
  device.pushErrorScope("validation");
  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module: device.createShaderModule({ code }), entryPoint: entry } });
  const buffer = device.createBuffer({ size: rows.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, rows);
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
  const group = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: view },
      { binding: 2, resource: sampler },
      { binding: 3, resource: { buffer } },
    ],
  });
  const read = device.createBuffer({ size: rows.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(rows.length / 4 / 64));
  pass.end();
  enc.copyBufferToBuffer(buffer, 0, read, 0, rows.byteLength);
  device.queue.submit([enc.finish()]);
  const error = await device.popErrorScope();
  if (error) throw new Error(`receiver probe: ${error.message}`);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();
  buffer.destroy();
  return out;
}

/**
 * `?probe=shadow` (K6.5; `&light=dir|spot|point`, `&cascades=`, `&source=`): 'volume' lighting with castShadows over
 * two volumes and a caster that is not drawn. (a) Every texel of the light's Beer map vs `vdbFrameShadowMapTexel` on
 * the read-back shadow grid with the ray from the drawn view (`vdbFrameShadowMapRay` / the cube face table); (b) the
 * receiver functions (`vdbVolumeShadow` / `vdbVolumeShadowCube` in a compute kernel) at points on a ground plane and in
 * the volumes vs `vdbFrameVolumeShadowLookup` on the read-back map; (c) a second identical update draws nothing, a
 * moved light draws one piece.
 */
async function probeShadow(device: GPUDevice): Promise<void> {
  const light = (params.get("light") ?? "dir") as "dir" | "spot" | "point";
  const cascades = Number(params.get("cascades") ?? (light === "dir" ? 2 : 1));
  const kind = params.get("source") === "sparse" ? "sparse" : "dense";
  const [GW, GH, GD] = [24, 16, 12];
  const R = 16;
  const SIZE = 32;
  const STEPS = 24;
  const scene = await probeScene(device, { kind, groups: 1, mips: true, extra: true });
  const settings = {
    lighting: "volume" as const,
    width: GW,
    height: GH,
    depth: GD,
    subSteps: 8,
    temporal: false,
    jitter: false,
    multipleScattering: true,
    srgbBlending: false,
    shadowResolution: R,
    shadowCascades: 1,
    shadowDistance: 50,
    castShadows: true,
    castShadowCascades: cascades,
    castShadowSteps: STEPS,
    castShadowUpdatesPerFrame: 16,
    castShadowDistance: 40,
  };
  const renderer = vdbCreateFrameRenderer(device, { settings });
  renderer.addVolume(scene.slots[0]!, { matrixWorld: compose([0, 0, 0], 0.4, [2, 1.6, 2.2]), extinction: 4, mipmaps: true });
  renderer.addVolume(scene.slots[1]!, { matrixWorld: compose([0.7, 0.25, -0.3], -0.3, [1.8, 1.5, 1.7]), extinction: 3, mipmaps: true });
  const c = renderer.addVolume(scene.slots[2]!, { matrixWorld: compose([0.6, 1.6, 0.4], 0.2, [1.4, 0.5, 1.2]), extinction: 5, mipmaps: true });
  c.visible = false;
  const eyeWorld = lookAt([3.2, 1.6, 4.4], [0.2, 0, 0]);
  const view = vdbMat4Invert(eyeWorld);
  const projection = perspective(0.75, GW / GH, 0.1, 100);
  const spotPos: VDBFrameVec3 = [-2.5, 3, 2];
  const spotVP = vdbMat4Multiply(perspective(1.3, 1, 0.5, 12), vdbMat4Invert(lookAt(spotPos, [0.3, -0.2, -0.1])));
  const id = light === "dir" ? "sun" : light === "spot" ? "spot" : "lamp";
  const lightOf = (dx: number): VDBFrameLight =>
    light === "dir"
      ? { id, kind: "directional", direction: [-0.4 + dx, -0.8, -0.45], color: [1, 1, 1], shadow: { size: SIZE } }
      : light === "spot"
        ? { id, kind: "spot", position: spotPos, direction: [0.55, -0.65, -0.5], color: [1, 1, 1], distance: 0, decay: 2, cosAngle: Math.cos(0.65), cosPenumbra: Math.cos(0.4), shadow: { size: SIZE, view: { viewProjection: dx ? vdbMat4Multiply(perspective(1.3, 1, 0.5, 12), vdbMat4Invert(lookAt([spotPos[0] + dx, spotPos[1], spotPos[2]], [0.3, -0.2, -0.1]))) : spotVP, near: 0.5, far: 12 } } }
        : { id, kind: "point", position: [0.4 + dx, 2.9, 0.6], color: [1, 1, 1], distance: 0, decay: 2, shadow: { size: SIZE, view: { near: 0.1, far: 10 } } };
  const frame = (frameId: number, dx = 0) => ({ frameId, camera: { view, projection, near: 0.1, far: 100, id: 1 }, viewport: { drawingBuffer: [GW, GH] as [number, number], canvas: [GW, GH] as [number, number] }, lights: [lightOf(dx)] });
  let enc = device.createCommandEncoder();
  renderer.update(enc, frame(0));
  device.queue.submit([enc.finish()]);
  const map = renderer.outputs.volumeShadows.get(id);
  if (!map) throw new Error("no volume shadow map");
  const grid = vdbFrameRendererGrid(renderer)!;
  const gt = grid.textures;
  const shadowStack: VDBFrameTexture3d = { data: halves(await readTexture(device, gt.shadow, gt.shadowLayout.size, 8)), size: gt.shadowLayout.size, channels: 4 };
  const mapData = halves(await readTexture(device, map.texture, [SIZE, SIZE, map.layers], 8));
  const layout: VDBFrameGridLayout = {
    count: grid.count,
    resolution: R,
    columns: vdbFrameAtlasLayout(R, R, R).columns,
    atlasSize: [1, 1],
    gridMin: grid.gridMin as VDBFrameVec3[],
    gridSize: grid.gridSize as VDBFrameVec3[],
    gridVoxel: grid.gridVoxel as VDBFrameVec3[],
  };
  const last = grid.count - 1;
  const box = { min: layout.gridMin[last]!, max: layout.gridMin[last]!.map((v, k) => v + layout.gridSize[last]![k]!) as VDBFrameVec3 };
  const extinction = (world: VDBFrameVec3) => vdbFrameStackSample(shadowStack, gt.shadowLayout.origins[last]!, R, vdbFrameGridUvw(layout, last, world))[0]!;
  const drawn = { ...map.drawn, origin: map.drawn.origin.map((o) => [...o] as VDBFrameVec3), forward: [...map.drawn.forward] as VDBFrameVec3 };
  const layers = light === "dir" ? drawn.cascadeCount : light === "spot" ? 1 : 6;
  const steps = MUTATE === "map-steps" ? STEPS + 1 : STEPS;
  const stats = newStats(`2 f16 ulp (rgba16float Beer map vs vdbFrameShadowMapTexel on the read-back grid; ${light}, ${kind})`);
  let shadowed = 0;
  for (let layer = 0; layer < layers; layer++)
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const s = (x + 0.5) / SIZE;
        const t = (y + 0.5) / SIZE;
        let origin: VDBFrameVec3;
        let direction: VDBFrameVec3;
        let ray: { origin: VDBFrameVec3; forward: VDBFrameVec3; perspective: number; far: number };
        if (light === "point") {
          const d = vdbFrameCubeFaceDirection(layer, s * 2 - 1, t * 2 - 1);
          const l = Math.hypot(...d);
          origin = drawn.origin[0]!;
          direction = [d[0] / l, d[1] / l, d[2] / l];
          ray = { origin, forward: drawn.forward, perspective: 1, far: drawn.far[0]! };
        } else {
          const r = vdbFrameShadowMapRay(vdbMat4Invert(drawn.viewProjection[layer]!), drawn.origin[layer]!, drawn.perspective, [s * 2 - 1, 1 - t * 2]);
          origin = r.origin;
          direction = r.direction;
          ray = { origin: drawn.origin[layer]!, forward: drawn.forward, perspective: drawn.perspective ? 1 : 0, far: drawn.far[layer]! };
        }
        if (MUTATE === "map-far") ray = { ...ray, far: ray.far * 1.01 };
        const cpu = vdbFrameShadowMapTexel(extinction, origin, direction, box, steps, ray).map;
        const i = ((layer * SIZE + y) * SIZE + x) * 4;
        const gpu = [mapData[i]!, mapData[i + 1]!, mapData[i + 2]!, mapData[i + 3]!];
        if (cpu[3] > 1e-3) shadowed++;
        compare(stats, [layer, x, y], gpu, cpu, 2, 2 ** -20);
      }
  result.stats = stats;

  // (b) the receivers through the drawn view, at points on a ground plane below the volumes and inside them
  const rand = rng(11);
  const rows: number[] = [];
  const points: { world: VDBFrameVec3; depth: number }[] = [];
  const viewCpu = {
    kind: light === "dir" ? 0 : light === "spot" ? 1 : 2,
    viewProjection: drawn.viewProjection,
    origin: drawn.origin as VDBFrameVec3[],
    far: drawn.far,
    forward: drawn.forward as VDBFrameVec3,
    perspective: drawn.perspective,
    breaks: drawn.breaks,
    cascadeCount: drawn.cascadeCount,
    size: SIZE,
    enabled: map.enabled,
  };
  while (points.length < 1024) {
    const world: VDBFrameVec3 = points.length % 2 ? [rand() * 5 - 2.5, -1.4, rand() * 5 - 2.5] : [rand() * 3 - 1.5, rand() * 2 - 1, rand() * 3 - 1.5];
    if (light === "point") {
      const f = vdbFrameCubeFaceUv([world[0] - drawn.origin[0]![0], world[1] - drawn.origin[0]![1], world[2] - drawn.origin[0]![2]]);
      const m = 1.5 / SIZE;
      if (f.s < m || f.s > 1 - m || f.t < m || f.t > 1 - m) continue;
    }
    const v = vdbMat4Vec4(view, [...world, 1]);
    points.push({ world, depth: -v[2]! });
    rows.push(...world, -v[2]!);
  }
  const gpuRows = await runRowProbe(
    device,
    vdbVolumeShadowProbeWgsl(light === "point"),
    "vdbVolumeShadowProbeCs",
    map.view,
    map.texture.createView({ dimension: light === "point" ? "cube" : "2d-array" }),
    new Float32Array(rows),
    light === "point" ? "cube" : "2d-array",
  );
  const rstats = newStats("|d| ≤ 1e-3 (receiver transmittance vs vdbFrameVolumeShadowLookup on the read-back map; SwiftShader filters f16 texels in f32)");
  let receiverShadowed = 0;
  const viewForCpu = MUTATE === "receiver-far" ? { ...viewCpu, far: viewCpu.far.map((f) => f * 1.01) } : viewCpu;
  points.forEach((p, i) => {
    const depth = MUTATE === "receiver-layer" ? p.depth * 3 : p.depth;
    const cpu = vdbFrameVolumeShadowLookup(viewForCpu, p.world, depth, (layer, s, t) => bilinearLayer(mapData, SIZE, layer, s, t));
    if (cpu < 0.999) receiverShadowed++;
    compareScaled(rstats, [i], [gpuRows[i * 4]!], [cpu], 1e-3, 1);
  });
  result.receiverStats = rstats;

  // (c) the redraw: nothing changed → no piece; the light moved → one piece (the drawn view moves with it)
  enc = device.createCommandEncoder();
  renderer.update(enc, frame(1));
  device.queue.submit([enc.finish()]);
  const same = { ...renderer.redrawStats.volumeShadows };
  enc = device.createCommandEncoder();
  renderer.update(enc, frame(2, 0.05));
  device.queue.submit([enc.finish()]);
  const moved = { ...renderer.redrawStats.volumeShadows };
  result.shadow = { light, layers, shadowed, receiverShadowed, same, moved, cascades: drawn.cascadeCount, enabled: map.enabled };
  renderer.dispose();
  scene.dispose();
}

/**
 * `?probe=emission` (K6.5; `&source=`, `&cascades=`, `&hook=1`): the emission light's kernels on volume B (emission
 * channel; 'planck' colour, or a user WGSL hook) and their twins: (a) the 16 direction texels vs
 * `vdbFrameEmissionDirectionTexel` with the CPU samplers and the LUT colour; (b) the reduce vs `vdbFrameEmissionReduce`
 * on the read-back directions; (c) every probe vs `vdbFrameEmissionProbe` on the read-back emitter; (d) the receiver
 * `vdbEmissionIrradiance` vs `vdbFrameEmissionLookupWith` + `vdbFrameEmissionIrradiance` on the read-back stack; (e) a
 * second identical update draws nothing. `&hook=1` also checks that bad hooks throw VDB_BAD_WGSL naming the line.
 */
async function probeEmission(device: GPUDevice): Promise<void> {
  const kind = params.get("source") === "sparse" ? "sparse" : "dense";
  const cascades = Number(params.get("cascades") ?? 1);
  const hooked = params.get("hook") === "1";
  const [GW, GH, GD] = [24, 16, 12];
  const RES = 8;
  const STEPS = 12;
  const scene = await probeScene(device, { kind, groups: 2, mips: true });
  const settings = { lighting: "volume" as const, width: GW, height: GH, depth: GD, subSteps: 8, temporal: false, jitter: false, multipleScattering: true, srgbBlending: false };
  const renderer = vdbCreateFrameRenderer(device, { settings });
  const HOOK = "// a user colour\nfn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f {\n  return vec3f(t, 1.0 - 0.5 * t, temperature / 3000.0);\n}\n";
  const exponent = 1.39;
  const color = hooked ? { wgsl: HOOK } : ({ spectrum: "soot", exponent } as const);
  const pB: Partial<VDBFrameVolumeParams> = { matrixWorld: compose([0.7, 0.25, -0.3], -0.3, [1.8, 1.5, 1.7]), extinction: 3, absorbance: 0.8, mipmaps: true, emissiveIntensity: 2, temperatureRange: [500, 3000], emissionColor: color };
  renderer.addVolume(scene.slots[0]!, { matrixWorld: compose([0, 0, 0], 0.4, [2, 1.6, 2.2]), extinction: 4, mipmaps: true });
  const b = renderer.addVolume(scene.slots[1]!, pB);
  const hookErrors: string[] = [];
  if (hooked) {
    await renderer.prepare();
    for (const bad of [
      "fn emissionColour(t: f32, temperature: f32) -> vec3f { return vec3f(t); }",
      "fn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f {\n  let k = t;\n  return k;\n}\n",
      "fn vdbUserEmissionColor(t: f32) -> vec3f {\n  return vec3f(t);\n}\n",
    ]) {
      try {
        await vdbValidateEmissionHook(device, bad);
        hookErrors.push("no error");
      } catch (e) {
        hookErrors.push(`${(e as { code?: string }).code}: ${(e as Error).message}`);
      }
    }
  }
  const eyeWorld = lookAt([3.2, 1.6, 4.4], [0.2, 0, 0]);
  const view = vdbMat4Invert(eyeWorld);
  const projection = perspective(0.75, GW / GH, 0.1, 100);
  const emissionLight = { resolution: RES, cascades, steps: STEPS, range: 3, distance: 3 };
  const frame = (frameId: number) => ({ frameId, camera: { view, projection, near: 0.1, far: 100, id: 1 }, viewport: { drawingBuffer: [GW, GH] as [number, number], canvas: [GW, GH] as [number, number] }, lights: PROBE_LIGHTS, emissionLight });
  let enc = device.createCommandEncoder();
  renderer.update(enc, frame(0));
  device.queue.submit([enc.finish()]);
  const probes = renderer.outputs.emissionProbes;
  if (!probes || probes.state.emitters !== 1) throw new Error(`emission probes: ${JSON.stringify(probes?.state)}`);
  const E = vdbFrameRendererEmission(renderer);
  const directions = floats(await readTexture(device, E.directions!, [32, E.slots, 1], 16));
  const emitterTex = floats(await readTexture(device, E.emitters!, [2, E.slots, 1], 16));
  const stack: VDBFrameTexture3d = { data: halves(await readTexture(device, probes.texture, probes.layout.size, 8)), size: probes.layout.size as [number, number, number], channels: 4 };

  // (a) the direction pass
  const r = twinRecord(b, scene.samplers[1]!, { localToView: vdbMat4Multiply(view, b.params.matrixWorld), viewToLocal: vdbMat4Invert(vdbMat4Multiply(view, b.params.matrixWorld)), worldToLocal: vdbMat4Invert(b.params.matrixWorld) }, [0, 0, 0, 0], { first: 0, count: 0 }, 0.8);
  const row = vdbFrameEmissionLutRow(exponent);
  const colorOf = MUTATE === "emission-color" ? null : hooked ? (t: number, temperature: number): VDBFrameVec3 => [t, 1 - 0.5 * t, temperature / 3000] : (_t: number, temperature: number) => vdbFrameEmissionLutColor(row, temperature);
  const bounds = vdbFrameEmitterBounds(b.params.matrixWorld);
  const maxTexels = Math.max(...r.volume.texels);
  const emitter = {
    sampleExtinction: (local: VDBFrameVec3, lod: number) => extinctionAt(r, local, lod),
    sampleEmission: (local: VDBFrameVec3, lod: number) => emissionAt(r, local, lod, colorOf),
    getLod: (size: number) => (r.mipmaps ? Math.max(0, Math.log2(size * maxTexels)) : 0),
    worldToLocal: r.worldToLocal,
    center: bounds.center,
    radius: bounds.radius,
    worldScale: r.worldScale,
    absorbance: r.absorbance,
    steps: MUTATE === "emission-direction" ? STEPS + 1 : STEPS,
  };
  const cpuTexels = Array.from({ length: 16 }, (_, i) => vdbFrameEmissionDirectionTexel(emitter, i));
  const scale = Math.max(...cpuTexels.flatMap((t) => [...t.intensity.slice(0, 3), ...t.center.slice(0, 3).map(Math.abs), t.center[3]]));
  const dstats = newStats("rel 1e-3 of the texel set's largest value (rgba32float direction texels vs vdbFrameEmissionDirectionTexel; SwiftShader's ~12-bit division moves the box segments by ~2.4e-4)");
  const gpuTexels = Array.from({ length: 16 }, (_, i) => ({ intensity: [0, 1, 2, 3].map((k) => directions[i * 4 + k]!) as VDBFrameVec4, center: [0, 1, 2, 3].map((k) => directions[(16 + i) * 4 + k]!) as VDBFrameVec4 }));
  cpuTexels.forEach((t, i) => {
    compareScaled(dstats, [i, 0], gpuTexels[i]!.intensity, t.intensity, 1e-3, scale);
    compareScaled(dstats, [i, 1], gpuTexels[i]!.center, t.center, 1e-3, scale);
  });
  result.stats = dstats;

  // (b) the reduce on the read-back directions
  const reduced = vdbFrameEmissionReduce(gpuTexels);
  if (MUTATE === "emission-reduce") reduced.intensity[0] *= 1.001;
  const restats = newStats("rel 1e-5 (rgba32float emitter vs vdbFrameEmissionReduce on the read-back directions)");
  compareScaled(restats, [0], [0, 1, 2, 3].map((k) => emitterTex[k]!), reduced.intensity, 1e-5, 1e-12);
  compareScaled(restats, [1], [0, 1, 2, 3].map((k) => emitterTex[4 + k]!), reduced.center, 1e-5, 1);
  result.reduceStats = restats;

  // (c) every probe from the read-back emitter
  const light = { intensity: [0, 1, 2, 3].map((k) => emitterTex[k]!) as VDBFrameVec4, center: [0, 1, 2, 3].map((k) => emitterTex[4 + k]!) as VDBFrameVec4 };
  if (MUTATE === "emission-probe") light.intensity[3] *= 1.2;
  const state = probes.state;
  const pstats = newStats("2 f16 ulp (rgba16float probe stack vs vdbFrameEmissionProbe on the read-back emitter)");
  const at = (t: VDBFrameTexture3d, p: readonly number[]) => {
    const i = (p[0]! + (p[1]! + p[2]! * t.size[1]) * t.size[0]) * 4;
    return [t.data[i]!, t.data[i + 1]!, t.data[i + 2]!, t.data[i + 3]!];
  };
  for (let cc = 0; cc < state.count; cc++)
    for (let z = 0; z < RES; z++)
      for (let y = 0; y < RES; y++)
        for (let x = 0; x < RES; x++) {
          const world = [0, 1, 2].map((k) => state.min[cc]![k]! + (([x, y, z][k]! + 0.5) / RES) * state.size[cc]![k]!) as VDBFrameVec3;
          const cpu = vdbFrameEmissionProbe(world, [light]);
          for (let k = 0; k < 3; k++) {
            const o = probes.layout.origins[3 * cc + k]!;
            compare(pstats, [cc, k, x, y, z], at(stack, [o[0] + x, o[1] + y, o[2] + z]), cpu[k]!, 2, 2 ** -20);
          }
        }
  result.probeStats = pstats;

  // (d) the receiver on the read-back stack, at points around the volumes and past the grid
  const gridLayout: VDBFrameGridLayout = { count: state.count, resolution: state.resolution, columns: 1, atlasSize: [1, 1], gridMin: state.min as VDBFrameVec3[], gridSize: state.size as VDBFrameVec3[], gridVoxel: state.size.map((s) => s.map((v) => v / state.resolution) as VDBFrameVec3) };
  const rand = rng(5);
  const rows: number[] = [];
  const pts: { world: VDBFrameVec3; normal: VDBFrameVec3 }[] = [];
  for (let i = 0; i < 512; i++) {
    const spread = i % 4 === 0 ? 30 : 6;
    const world: VDBFrameVec3 = [(rand() - 0.5) * spread, (rand() - 0.5) * spread, (rand() - 0.5) * spread];
    const n0: VDBFrameVec3 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
    const l = Math.hypot(...n0);
    const normal: VDBFrameVec3 = [n0[0] / l, n0[1] / l, n0[2] / l];
    pts.push({ world, normal });
    rows.push(...world, 0, ...normal, 0);
  }
  const gpuRows = await runRowProbe(device, vdbEmissionIrradianceProbeWgsl(), "vdbEmissionIrradianceProbeCs", probes.grid, probes.texture.createView({ dimension: "3d" }), new Float32Array(rows), "3d");
  const cpuE = pts.map((p) => {
    const look = vdbFrameEmissionLookupWith((cc, k, uvw) => vdbFrameStackSample(stack, probes.layout.origins[3 * cc + k]!, state.resolution, uvw), gridLayout, state.center as VDBFrameVec3, p.world);
    return vdbFrameEmissionIrradiance(look.probe, p.normal, MUTATE === "emission-irradiance" ? look.falloff * look.falloff : look.falloff);
  });
  const escale = Math.max(...cpuE.flat());
  const istats = newStats("rel 1e-4 of the largest irradiance (vdbEmissionIrradiance vs vdbFrameEmissionLookupWith on the read-back stack)");
  cpuE.forEach((e, i) => compareScaled(istats, [i], [gpuRows[i * 8]!, gpuRows[i * 8 + 1]!, gpuRows[i * 8 + 2]!], e, 1e-4, escale * 1e-2));
  result.irradianceStats = istats;

  // (e) nothing changed: no direction pass, no probes
  enc = device.createCommandEncoder();
  renderer.update(enc, frame(1));
  device.queue.submit([enc.finish()]);
  result.emission = { state: JSON.parse(JSON.stringify(state)), first: probes.lastUpdate, second: { ...renderer.outputs.emissionProbes!.lastUpdate }, hookErrors, outside: pts.filter((p) => p.world.some((v) => Math.abs(v) > 12)).length };
  renderer.dispose();
  scene.dispose();
}

const COMPILE_HOOK = "fn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f {\n  return vec3f(t, 0.5, temperature * 1e-4);\n}\n";

async function probeCompile(device: GPUDevice): Promise<void> {
  const vol = synthDense([8, 8, 8], 3, { albedo: true, mask: true, emission: true, mips: true });
  const plain = synthDense([8, 8, 8], 4, { mips: false });
  // K6.1d / K6.2: the sparse source (K6.7) with every channel, and the 'frame' lighting kernels
  const fx = await sparseFixture(device);
  const sparse = vdbSparseSourceGpu(fx.pool, { types: ["float"], maxVolumes: 8, label: "k6 compile" });
  sparse.volume({ density: fx.ids.d2, mask: fx.ids.m, albedo: fx.ids.r, emission: fx.ids.e }, { image: fx.images.get(fx.ids.d2)! });
  const specs = [
    ...[vdbDenseSource(device, vol), vdbDenseSource(device, plain), sparse].flatMap((s) => [
      ...(["mixture", "over"] as const).flatMap((mix) => [false, true].map((multi) => vdbFrameGatherSpec(s, { mix, multi }))),
      ...[false, true].map((multi) => vdbFrameGatherSpec(s, { mix: "mixture", multi, lighting: "frame" })),
      ...[false, true].map((multi) => vdbFrameGridDensitySpec(s, { multi })),
    ]),
    vdbFrameTransmittanceSpec(),
    // K6.3: the scattered-light variants and the 'mlp' feature mips
    vdbFrameTransmittanceSpec({ scattering: "mlp" }),
    vdbFrameTransmittanceSpec({ scattering: "multi-scale" }),
    vdbFrameGridMipSpec({ first: true }),
    vdbFrameGridMipSpec({ first: false }),
    vdbFrameResolveSpec({ temporal: false }),
    vdbFrameResolveSpec({ temporal: true }),
    vdbFrameChunkProbeSpec(),
    // K6.2b: the environment kernels
    vdbFrameEnvPrefilterSpec("equirect"),
    vdbFrameEnvPrefilterSpec("cube"),
    vdbFrameEnvBlurSpec(),
    vdbFrameEnvReduceSpec(),
    vdbFrameEnvironmentGridSpec(),
    // K6.5: the shadow map kernel, the emission kernels (dense and sparse sources with emission, with a user hook)
    vdbFrameVolumeShadowSpec(),
    vdbFrameEmissionReduceSpec(),
    vdbFrameEmissionProbeSpec(),
    ...[vdbDenseSource(device, vol), sparse].flatMap((s) => [
      vdbFrameEmissionDirectionSpec(s, {}),
      vdbFrameEmissionDirectionSpec(s, { hook: COMPILE_HOOK }),
      vdbFrameGatherSpec(s, { mix: "mixture", multi: true, hook: COMPILE_HOOK }),
      vdbFrameGatherSpec(s, { mix: "mixture", multi: true, lighting: "frame", hook: COMPILE_HOOK }),
    ]),
  ];
  const errors: string[] = [];
  // `&only=`: compile the specs whose label contains it (diagnostics); per-spec seconds in result.compileSeconds
  const only = params.get("only");
  const seconds: Record<string, number> = {};
  for (const spec of only ? specs.filter((s) => s.label.includes(only)) : specs) {
    const t0 = performance.now();
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
    seconds[spec.label] = (performance.now() - t0) / 1000;
    console.log(`k6 compile ${spec.label}: ${seconds[spec.label]!.toFixed(1)} s`);
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
  result.compile = { variants, errors, seconds };
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
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  if (PROBE === "view") await probeView(device);
  else if (PROBE === "gather") await probeGather(device);
  else if (PROBE === "resolve") await probeResolve(device);
  else if (PROBE === "integrate") await probeIntegrate(device);
  else if (PROBE === "compile") await probeCompile(device);
  else if (PROBE === "frame") await probeFrame(device);
  else if (PROBE === "environment") await probeEnvironment(device);
  else if (PROBE === "shadow") await probeShadow(device);
  else if (PROBE === "emission") await probeEmission(device);
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
