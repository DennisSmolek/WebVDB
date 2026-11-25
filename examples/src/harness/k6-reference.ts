/**
 * K6.R harness: the reference renderer (`vdbCreateReferenceRendererGpu`, core) on raw WebGPU, no three.
 *
 * Modes (`?case=`):
 * - `analytic` (default): the GPU kernel on analytic cases over a constant dense volume (2³ texels of 255: trilinear
 *   reads exactly 1): Beer–Lambert (slab, ball), emission only, single scattering of a directional light (slab, ball;
 *   dual-lobe phase, RGB albedo) against the quadrature, and the white furnace (albedo 1, uniform environment). Each
 *   case compares per pixel and channel within k·σ (σ from the batch means) plus an unbiasedness check of the image.
 *   `?mutate=<name>` patches the WGSL (`VDB_REFERENCE_MUTATIONS`); some case must fail.
 * - `twin`: the GPU kernel against its CPU twin (`vdbReferenceRenderCpu`, the same random stream): the committed CC0
 *   cloud through `vdbDenseSource` (perspective camera, sun + point light, hemisphere + ambient environment,
 *   dual-lobe phase, multiple scattering), and synthetic grids through `vdbSparseSourceGpu` (two overlapping
 *   instances of one grid + a volume with mask, RGB albedo and emission; a spot light, an equirect environment).
 *   Per pixel and channel |GPU − CPU| ≤ k·√(σ²_GPU + σ²_CPU); also the share of values equal to 1e-3 relative (the
 *   shared stream makes most paths identical). `&determinism=1` renders the dense case twice (bitwise equal).
 * - `report`: one K3 scenario from the capture manifest (`?scenario=`, his bunny), rendered by the reference for
 *   `scripts/k6/reference-report.mjs` (`?w=&h=&spp=&batches=&maxBounces=&phase=g,backG,backWeight&budget=<s>`).
 *
 * Results: `window.__VDB_K6_REFERENCE__` (e2e/k6-reference.spec.ts).
 */
import { VDBGridPool, vdbDenseSource } from "@pmndrs/vdb";
import {
  VDB_REFERENCE_MUTATIONS,
  vdbCreateReferenceRendererGpu,
  vdbSparseSourceGpu,
  type VDBReferenceSourceGpu,
  type VDBSparseSourceGpu,
} from "../../../packages/vdb/src/core/internal.js";
import {
  readWvdb,
  vdbBuildFromDense,
  vdbRefBeerLambert,
  vdbRefEmissionOnly,
  vdbRefInvert4,
  vdbRefMultiply4,
  vdbRefSingleScatter,
  vdbReferenceConstantMediumCpu,
  vdbReferenceDenseMediumCpu,
  vdbReferenceLightsFromFrame,
  vdbReferenceRenderCpu,
  vdbReferenceSigma,
  vdbReferenceSparseMediumCpu,
  vdbReferenceStatistics,
  wvdbDenseVolume,
  type VDBRefVec3,
  type VDBReferenceAccum,
  type VDBReferenceFrameLight,
  type VDBReferenceMediumCpu,
  type VDBReferenceScene,
  type VDBReferenceSettings,
  type VDBSparseVolumeCpu,
  type WvdbDenseTexture,
  type WvdbDenseVolume,
} from "../../../packages/vdb/src/tools/internal.js";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

type Settings = Omit<VDBReferenceSettings, "width" | "height">;

interface CaseResult {
  name: string;
  pixels: number;
  values: number;
  over: number;
  maxZ: number;
  meanZ: number;
  passes: boolean;
}

interface TwinResult {
  name: string;
  width: number;
  height: number;
  batches: number;
  spp: number;
  values: number;
  over: number;
  maxZ: number;
  meanZ: number;
  /** Values within 1e-3 relative (+1e-6) of the twin: paths that took the same random decisions. */
  equalShare: number;
  /** Mean relative standard error of L over pixels with signal (GPU). */
  noise: number;
  /** Volume pixels (T < 0.999) and the mean depth of the CPU paths. */
  covered: number;
  meanDepth: number;
  /** Share of CPU paths with ≥ 2 scattering events, and the longest. */
  multipleShare: number;
  maxDepth: number;
  majorantViolations: number;
  passes: boolean;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  mode: string;
  mutate: string | null;
  cases: CaseResult[];
  twins: TwinResult[];
  determinism?: { identical: boolean; values: number };
  report?: Record<string, unknown>;
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K6_REFERENCE__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const MODE = params.get("case") ?? "analytic";
const MUTATE = params.get("mutate");
const K = 4.5;

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, mode: MODE, mutate: MUTATE, cases: [], twins: [], warnings: [], uncaptured: [] };
window.__VDB_K6_REFERENCE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

function patchOf(name: string | null): ((wgsl: string) => string) | undefined {
  if (!name) return undefined;
  const m = VDB_REFERENCE_MUTATIONS[name];
  if (!m) throw new Error(`unknown mutation ${name} (${Object.keys(VDB_REFERENCE_MUTATIONS).join(", ")})`);
  return (wgsl) => {
    if (!wgsl.includes(m[0])) throw new Error(`mutation ${name}: pattern not found`);
    return wgsl.split(m[0]).join(m[1]);
  };
}

// ── scene helpers ─────────────────────────────────────────────────────────────────────────────────────

/** Column-major WebGL perspective (fov y in degrees). */
function perspective(fovY: number, aspect: number, near: number, far: number): number[] {
  const f = 1 / Math.tan((fovY * Math.PI) / 360);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0];
}

/** Camera → world of a camera at `eye` looking at `target` (y up). */
function lookAt(eye: VDBRefVec3, target: VDBRefVec3): number[] {
  const z = norm([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = norm(cross([0, 1, 0], z));
  const y = cross(z, x);
  return [...x, 0, ...y, 0, ...z, 0, ...eye, 1];
}
const cross = (a: number[], b: number[]): VDBRefVec3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
const norm = (a: number[]): VDBRefVec3 => {
  const l = Math.hypot(a[0]!, a[1]!, a[2]!);
  return [a[0]! / l, a[1]! / l, a[2]! / l];
};

/** A dense volume of constant texels (r8unorm 2³; 255 reads 1.0 exactly under trilinear). */
function constantDense(density: number, emission: number | null): WvdbDenseVolume {
  const tex = (v: number): WvdbDenseTexture => ({ data: new Uint8Array(8).fill(v), dimensions: [2, 2, 2], format: "r8unorm", texelBytes: 1, mip: null });
  return {
    kind: "dense",
    normalize: "clamp01",
    lod: 0,
    dimensions: [2, 2, 2],
    bbox: null,
    density: tex(density),
    mask: null,
    albedo: null,
    emission: emission === null ? null : tex(emission),
    densityStats: { sum: (8 * density) / 255, count: 8 },
    densityStatsSource: "texels",
    meanDensity: density / 255,
    maxDensity: density / 255,
    activeTexels: 8,
    gridNames: { density: "density", ...(emission === null ? {} : { emission: "emission" }) },
  };
}

// ── GPU render ────────────────────────────────────────────────────────────────────────────────────────

interface GpuJob {
  source: VDBReferenceSourceGpu;
  sparse?: VDBSparseSourceGpu;
  width: number;
  height: number;
}

async function makeRenderer(device: GPUDevice, job: GpuJob, patch?: (w: string) => string) {
  return vdbCreateReferenceRendererGpu(device, { source: job.source, width: job.width, height: job.height, label: "k6-reference", ...(patch ? { wgslPatch: patch } : {}) });
}

async function renderGpu(device: GPUDevice, ref: Awaited<ReturnType<typeof makeRenderer>>, job: GpuJob, scene: VDBReferenceScene, settings: Settings, batches: number, chunk = 8): Promise<VDBReferenceAccum> {
  ref.setScene(scene, settings);
  let left = batches;
  while (left > 0) {
    const n = Math.min(left, chunk);
    const enc = device.createCommandEncoder();
    if (job.sparse) {
      job.sparse.pool.update(enc);
      job.sparse.update(enc);
    }
    ref.render(enc, n);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    left -= n;
  }
  return ref.readback();
}

// ── checks ────────────────────────────────────────────────────────────────────────────────────────────

function checkAnalytic(name: string, accum: VDBReferenceAccum, expected: (px: number, py: number) => number[]): CaseResult {
  const { mean, stderr } = vdbReferenceStatistics(accum);
  let over = 0;
  let maxZ = 0;
  let sumE = 0;
  let sumS2 = 0;
  let values = 0;
  for (let py = 0; py < accum.height; py++) for (let px = 0; px < accum.width; px++) {
    const want = expected(px, py);
    for (let c = 0; c < 4; c++) {
      const i = (py * accum.width + px) * 4 + c;
      const e = mean[i]! - want[c]!;
      const s = stderr[i]!;
      // 2e-5 absolute: the f32 accumulation of the GPU (a few ulps of the sums).
      if (Math.abs(e) > K * s + 2e-5) over++;
      if (Math.abs(e) > 2e-5) maxZ = Math.max(maxZ, Math.abs(e) / Math.max(s, 1e-12));
      sumE += e;
      sumS2 += s * s;
      values++;
    }
  }
  const meanZ = sumE / Math.sqrt(Math.max(sumS2, 1e-30));
  // Batch-mean σ estimates have few degrees of freedom and rare-event pixels (grazing chords) have heavy tails: allow
  // 0.5 % of values over k·σ; a systematic error shows in the image's mean (|meanZ| ≥ 4) or in many pixels.
  return { name, pixels: accum.width * accum.height, values, over, maxZ, meanZ, passes: over <= Math.max(2, Math.floor(values * 0.005)) && Math.abs(meanZ) < 4 };
}

function checkTwin(name: string, gpu: VDBReferenceAccum, cpu: VDBReferenceAccum & { stats: { samples: number; meanDepth: number; multiple: number; maxDepth: number; majorantViolations: number } }, spp: number): TwinResult {
  const g = vdbReferenceStatistics(gpu);
  const c = vdbReferenceStatistics(cpu);
  let over = 0;
  let maxZ = 0;
  let equal = 0;
  let values = 0;
  let sumE = 0;
  let sumS2 = 0;
  let covered = 0;
  let noiseSum = 0;
  let noiseN = 0;
  for (let p = 0; p < gpu.width * gpu.height; p++) {
    if (c.mean[p * 4 + 3]! < 0.999) covered++;
    const lum = (m: Float64Array, k: number) => 0.2126 * m[k]! + 0.7152 * m[k + 1]! + 0.0722 * m[k + 2]!;
    const y = lum(g.mean, p * 4);
    if (y > 1e-4) {
      noiseSum += lum(g.stderr, p * 4) / y;
      noiseN++;
    }
    for (let ch = 0; ch < 4; ch++) {
      const i = p * 4 + ch;
      const e = g.mean[i]! - c.mean[i]!;
      const s = Math.sqrt(g.stderr[i]! ** 2 + c.stderr[i]! ** 2);
      values++;
      if (Math.abs(e) <= 1e-3 * Math.abs(c.mean[i]!) + 1e-6) equal++;
      if (Math.abs(e) > K * s + 2e-5) over++;
      if (Math.abs(e) > 2e-5) maxZ = Math.max(maxZ, Math.abs(e) / Math.max(s, 1e-12));
      sumE += e;
      sumS2 += s * s;
    }
  }
  const meanZ = sumE / Math.sqrt(Math.max(sumS2, 1e-30));
  return {
    name,
    width: gpu.width,
    height: gpu.height,
    batches: gpu.batches,
    spp,
    values,
    over,
    maxZ,
    meanZ,
    equalShare: equal / values,
    noise: noiseN ? noiseSum / noiseN : 0,
    covered,
    meanDepth: cpu.stats.meanDepth,
    multipleShare: cpu.stats.multiple / Math.max(cpu.stats.samples, 1),
    maxDepth: cpu.stats.maxDepth,
    majorantViolations: cpu.stats.majorantViolations,
    // ≤ 0.1 % of values over k·σ: a path that took one different random decision (f32 vs f64 near a branch) can sit
    // far out in a 16-sample mean; the image as a whole must be unbiased.
    passes: over <= Math.max(1, Math.floor(values * 0.001)) && Math.abs(meanZ) < 4,
  };
}

// ── analytic cases (the unit tests' scenes, 16 × 16) ───────────────────────────────────────────────────

const AW = 16;
const HALF = 1.5;
const SLAB = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const BALL = [2.4, 0, 0, 0, 0, 2.4, 0, 0, 0, 0, 2.4, 0, 0.1, -0.05, 0, 1];

function orthoCamera(): VDBReferenceScene["camera"] {
  const n = 0.1;
  const f = 100;
  const p = [1 / HALF, 0, 0, 0, 0, 1 / HALF, 0, 0, 0, 0, -2 / (f - n), 0, 0, 0, -(f + n) / (f - n), 1];
  return { cameraToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1], projectionInverse: Array.from(vdbRefInvert4(p)), orthographic: true };
}
function orthoRay(px: number, py: number): { o: VDBRefVec3; d: VDBRefVec3 } {
  return { o: [(((px + 0.5) / AW) * 2 - 1) * HALF, ((1 - (py + 0.5) / AW) * 2 - 1) * HALF, 5], d: [0, 0, -1] };
}

async function analytic(device: GPUDevice): Promise<void> {
  const source = vdbDenseSource(device, constantDense(255, 255), { label: "k6-reference constant" });
  const job: GpuJob = { source, width: AW, height: AW };
  const ref = await makeRenderer(device, job, patchOf(MUTATE));
  const B = Number(params.get("batches") ?? 24);
  const base: Settings = { spp: Number(params.get("spp") ?? 8), maxBounces: 1, seed: 5 };
  const run = async (name: string, scene: VDBReferenceScene, settings: Settings, expected: (px: number, py: number) => number[]) => {
    const accum = await renderGpu(device, ref, job, scene, settings, B);
    const r = checkAnalytic(name, accum, expected);
    result.cases.push(r);
    log(`${name}: ${JSON.stringify(r)}`, r.passes ? "ok" : "fail");
  };

  for (const shape of ["box", "sphere"] as const) {
    const m = shape === "box" ? SLAB : BALL;
    const hm = { localToWorld: m, sphere: shape === "sphere", sigma: 1.3 };
    await run(`beer-lambert ${shape}`, { camera: orthoCamera(), volumes: [{ localToWorld: m, sigma: 1.3, majorant: 2, albedo: [0, 0, 0], shape }] }, base, (px, py) => {
      const r = orthoRay(px, py);
      return [0, 0, 0, vdbRefBeerLambert(hm, r.o, r.d)];
    });
  }
  {
    const le: VDBRefVec3 = [0.8, 0.5, 0.2];
    const hm = { localToWorld: BALL, sphere: true, sigma: 0.9 };
    await run(
      "emission only",
      { camera: orthoCamera(), volumes: [{ localToWorld: BALL, sigma: 0.9, majorant: 1, albedo: [0, 0, 0], shape: "sphere", emits: true, emission: { color: le, intensity: 1, temperatureRange: [0, 3000] } }] },
      { ...base, maxBounces: 4 },
      (px, py) => {
        const r = orthoRay(px, py);
        return [...vdbRefEmissionOnly(hm, r.o, r.d, le), vdbRefBeerLambert(hm, r.o, r.d)];
      },
    );
  }
  for (const shape of ["box", "sphere"] as const) {
    const m = shape === "box" ? SLAB : BALL;
    const toLight = norm([0.4, 0.7, 0.6]);
    const albedo: VDBRefVec3 = [0.9, 0.7, 0.5];
    const phase = { g: 0.6, backG: -0.3, backWeight: 0.25 };
    const e: VDBRefVec3 = [1.5, 1.2, 1];
    const hm = { localToWorld: m, sphere: shape === "sphere", sigma: 0.8 };
    await run(
      `single scattering ${shape}`,
      { camera: orthoCamera(), volumes: [{ localToWorld: m, sigma: 0.8, majorant: 1.5, albedo, phase, shape }], lights: [{ kind: "directional", direction: toLight, irradiance: e }] },
      { ...base, spp: base.spp! * 2 },
      (px, py) => {
        const r = orthoRay(px, py);
        return [...vdbRefSingleScatter(hm, r.o, r.d, toLight, e, albedo, phase), vdbRefBeerLambert(hm, r.o, r.d)];
      },
    );
  }
  {
    // White furnace: compare L + T·Le with Le (expected L = Le·(1 − T_exact), T = T_exact: both checked).
    const le: VDBRefVec3 = [0.5, 0.5, 0.5];
    const hm = { localToWorld: BALL, sphere: true, sigma: 1.6 };
    await run(
      "white furnace",
      { camera: orthoCamera(), volumes: [{ localToWorld: BALL, sigma: 1.6, majorant: 1.25, albedo: [1, 1, 1], phase: { g: 0.5 }, shape: "sphere" }], environment: { uniform: le } },
      { ...base, maxBounces: 4096 },
      (px, py) => {
        const r = orthoRay(px, py);
        const t = vdbRefBeerLambert(hm, r.o, r.d);
        return [le[0] * (1 - t), le[1] * (1 - t), le[2] * (1 - t), t];
      },
    );
  }
  ref.dispose();
  source.dispose();
}

// ── GPU vs CPU twin ───────────────────────────────────────────────────────────────────────────────────

async function twinDense(device: GPUDevice): Promise<void> {
  const volume = wvdbDenseVolume(readWvdb(new Uint8Array(await (await fetch(cloudWvdbUrl)).arrayBuffer())), { mipmaps: false });
  const source = vdbDenseSource(device, volume, { label: "k6-reference cloud" });
  const W = Number(params.get("w") ?? 20);
  const H = Number(params.get("h") ?? 16);
  const [lo, hi] = volume.bbox!;
  const size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const center: VDBRefVec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const l2w = [size[0]!, 0, 0, 0, 0, size[1]!, 0, 0, 0, 0, size[2]!, 0, ...center, 1];
  const diag = Math.hypot(...size);
  const eye: VDBRefVec3 = [center[0] + 0.55 * diag, center[1] + 0.3 * diag, center[2] + 0.75 * diag];
  const frame = vdbReferenceLightsFromFrame([
    { kind: "directional", direction: [0.5, 1, 0.4], color: [1, 0.95, 0.85] },
    { kind: "point", position: [center[0] - 0.4 * diag, center[1] + 0.1 * diag, center[2] + 0.3 * diag], color: [0.2, 0.3, 0.6] },
    { kind: "ambient", color: [0.05, 0.05, 0.06] },
    { kind: "hemisphere", sky: [0.3, 0.4, 0.6], ground: [0.1, 0.08, 0.05], up: [0, 1, 0] },
  ]);
  const scene: VDBReferenceScene = {
    camera: { cameraToWorld: lookAt(eye, center), projectionInverse: Array.from(vdbRefInvert4(perspective(40, W / H, 0.01 * diag, 10 * diag))) },
    volumes: [{
      localToWorld: l2w,
      sigma: vdbReferenceSigma({ localToWorld: l2w, extinction: Number(params.get("extinction") ?? 8), normalizeDensity: true, meanNonZero: volume.meanDensity }),
      majorant: volume.maxDensity,
      albedo: [0.95, 0.9, 0.85],
      phase: { g: 0.7, backG: -0.3, backWeight: 0.2 },
    }],
    lights: frame.lights,
    environment: frame.environment,
  };
  const settings: Settings = { spp: Number(params.get("spp") ?? 2), maxBounces: Number(params.get("maxBounces") ?? 32), seed: 17, rrStart: 3 };
  const batches = Number(params.get("batches") ?? 8);
  const job: GpuJob = { source, width: W, height: H };
  const ref = await makeRenderer(device, job, patchOf(MUTATE));
  const gpu = await renderGpu(device, ref, job, scene, settings, batches);
  if (params.get("determinism") === "1") {
    const again = await renderGpu(device, ref, job, scene, settings, batches);
    let identical = true;
    for (let i = 0; i < gpu.sum.length; i++) if (gpu.sum[i] !== again.sum[i] || gpu.sumSq[i] !== again.sumSq[i]) identical = false;
    result.determinism = { identical, values: gpu.sum.length };
    log(`determinism: ${JSON.stringify(result.determinism)}`, identical ? "ok" : "fail");
  }
  const cpu = vdbReferenceRenderCpu(scene, [vdbReferenceDenseMediumCpu(volume)], { ...settings, width: W, height: H }, { batches });
  const r = checkTwin("dense cloud (sun + point + hemisphere + ambient, dual-lobe, MS)", gpu, cpu, settings.spp!);
  result.twins.push(r);
  log(`${r.name}: ${JSON.stringify(r)}`, r.passes ? "ok" : "fail");
  ref.dispose();
  source.dispose();
}

/** A dense field (x-major, z fastest: vdbBuildFromDense's order) as a FLOAT grid at `origin`. */
function grid(dims: VDBRefVec3, origin: VDBRefVec3, fn: (x: number, y: number, z: number) => number): Uint32Array {
  const values = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let x = 0; x < dims[0]; x++) for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) values[(x * dims[1] + y) * dims[2] + z] = fn(x, y, z);
  return vdbBuildFromDense(values, dims, { origin, voxelSize: 0.25 });
}

async function twinSparse(device: GPUDevice): Promise<void> {
  const blob = (x: number, y: number, z: number) => {
    const a = Math.hypot(x - 12, y - 14, z - 10) / 9;
    const b = Math.hypot(x - 22, y - 18, z - 16) / 7;
    const v = Math.max(0, 1 - a) * 1.4 + Math.max(0, 1 - b) * (0.8 + 0.2 * Math.sin(x * 0.7 + z * 0.3));
    return v > 0.02 ? v : 0;
  };
  const S = grid([32, 30, 26], [-13, 5, 70], blob);
  const M = grid([20, 20, 20], [-5, 10, 75], (x, y, z) => 0.2 + 0.8 * Math.abs(Math.sin(x * 0.3 + y * 0.2 - z * 0.1)));
  const R = grid([16, 18, 14], [-13, 5, 70], (x, y) => -0.2 + 1.5 * ((x + y) / 34));
  const G = grid([16, 18, 14], [-13, 5, 70], (_x, y, z) => 0.5 + 0.6 * Math.cos(y * 0.4 + z * 0.2));
  const B = grid([16, 18, 14], [-13, 5, 70], (x, _y, z) => (x * z) / 210);
  const E = grid([24, 24, 24], [0, 10, 80], (x, y, z) => Math.max(0, 1 - Math.hypot(x - 12, y - 12, z - 12) / 12));
  const pool = await VDBGridPool.create(device, { label: "k6-reference" });
  const images = new Map<number, Uint32Array>();
  const add = (image: Uint32Array) => {
    const id = pool.add(image);
    images.set(id, image);
    return id;
  };
  const ids = { s: add(S), m: add(M), r: add(R), g: add(G), b: add(B), e: add(E) };
  pool.flush();
  await pool.settled();
  const source = vdbSparseSourceGpu(pool, { lod: "none", normalize: "clamp01", types: ["float"], maxVolumes: 4, label: "k6-reference sparse" });
  const plain = source.volume({ density: ids.s }, { image: S });
  const rich = source.volume({ density: ids.s, mask: ids.m, albedo: [ids.r, ids.g, ids.b], emission: ids.e }, { image: S });
  const own = (id: number) => {
    const b = pool.info(id).indexBBox;
    return { min: [...b.min] as VDBRefVec3, max: [...b.max] as VDBRefVec3 };
  };
  const cpuOf = (slot: typeof plain, aux: boolean): VDBSparseVolumeCpu => ({
    lod: "none",
    clamp01: true,
    density: { image: S, box: slot.indexBox },
    ...(aux
      ? { mask: { image: M, box: own(ids.m) }, albedo: { images: [R, G, B], box: own(ids.r) }, emission: { image: E, box: own(ids.e) } }
      : {}),
  });
  const W = Number(params.get("w") ?? 18);
  const H = Number(params.get("h") ?? 14);
  const box = (cx: number, cy: number, cz: number, s: number, turn: number) => {
    const c = Math.cos(turn);
    const n = Math.sin(turn);
    return [s * c, 0, -s * n, 0, 0, s, 0, 0, s * n, 0, s * c, 0, cx, cy, cz, 1];
  };
  const A = box(0, 0, 0, 2, 0);
  const A2 = box(0.7, 0.2, -0.3, 1.8, 0.6);
  const C = box(-0.9, -0.4, 0.5, 1.4, -0.3);
  const environment = {
    width: 16,
    height: 8,
    data: new Float32Array(16 * 8 * 4).map((_, i) => {
      const px = Math.floor(i / 4) % 16;
      const py = Math.floor(i / 64);
      const ch = i % 4;
      return ch === 3 ? 1 : 0.15 + 0.6 * (1 - py / 8) * (ch === 2 ? 1 : 0.6) + (px === 3 && py === 2 ? 4 : 0);
    }),
    intensity: 0.8,
  };
  const sigmaOf = (m: number[]) => vdbReferenceSigma({ localToWorld: m, extinction: 6, normalizeDensity: false });
  const scene: VDBReferenceScene = {
    camera: { cameraToWorld: lookAt([3.2, 1.6, 4.2], [0, 0, 0]), projectionInverse: Array.from(vdbRefInvert4(perspective(45, W / H, 0.05, 50))) },
    volumes: [
      { localToWorld: A, sigma: sigmaOf(A), majorant: 1, albedo: [0.9, 0.9, 0.9], phase: { g: 0.5 }, v: plain.v },
      { localToWorld: A2, sigma: sigmaOf(A2), majorant: 1, albedo: [0.8, 0.85, 0.95], phase: { g: -0.2 }, v: plain.v },
      { localToWorld: C, sigma: sigmaOf(C), majorant: 1, albedo: [1, 1, 1], phase: { g: 0.3, backG: -0.4, backWeight: 0.3 }, v: rich.v, emits: true, emission: { color: [1, 0.6, 0.3], intensity: 2, temperatureRange: [500, 3000] }, cutoff: 0.05 },
    ],
    lights: [
      { kind: "spot", position: [2.5, 3, 1], direction: [-2.5, -3, -1], intensity: [3, 2.6, 2.2], cosOuter: Math.cos(0.5), cosInner: Math.cos(0.3), inverseSquare: true },
      { kind: "directional", direction: [-0.3, 0.8, 0.5], irradiance: [0.8, 0.8, 1] },
    ],
    environment: { equirect: { ...environment, rotation: [0, 0, 1, 0, 1, 0, -1, 0, 0] } },
  };
  const settings: Settings = { spp: Number(params.get("spp") ?? 2), maxBounces: Number(params.get("maxBounces") ?? 24), seed: 23, rrStart: 2, pixelFilter: "box" };
  const batches = Number(params.get("batches") ?? 8);
  const job: GpuJob = { source, sparse: source, width: W, height: H };
  const ref = await makeRenderer(device, job, patchOf(MUTATE));
  const gpu = await renderGpu(device, ref, job, scene, settings, batches);
  const media: VDBReferenceMediumCpu[] = [vdbReferenceSparseMediumCpu(cpuOf(plain, false)), vdbReferenceSparseMediumCpu(cpuOf(plain, false)), vdbReferenceSparseMediumCpu(cpuOf(rich, true))];
  const cpu = vdbReferenceRenderCpu(scene, media, { ...settings, width: W, height: H }, { batches });
  const r = checkTwin("sparse (3 volumes, 2 overlapping; mask, RGB albedo, emission; spot + sun; equirect)", gpu, cpu, settings.spp!);
  result.twins.push(r);
  log(`${r.name}: ${JSON.stringify(r)}`, r.passes ? "ok" : "fail");
  ref.dispose();
  source.dispose();
  pool.dispose();
}

// ── report: one K3 scenario from his capture manifest ────────────────────────────────────────────────

interface ManifestScenario {
  name: string;
  metadata: {
    camera: { matrixWorld: number[]; projectionMatrix: number[] };
    lights: { type: string; color: number[]; intensity: number; matrixWorld: number[]; groundColor?: number[] }[];
    volumes: { matrixWorld: number[]; densityScale: number; extinction: number; visible?: boolean; textures: { matrix: number[] }[] }[];
  };
}

const toBase64 = (a: Float32Array) => {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
};

async function report(device: GPUDevice): Promise<void> {
  const name = params.get("scenario") ?? "step1-volume-dir-ms-on-isolated";
  const manifest = (await (await fetch(params.get("manifest") ?? "/fixtures/k3-reference/75e7345/manifest.json")).json()) as { scenarios: ManifestScenario[] };
  const sc = manifest.scenarios.find((s) => s.name === name);
  if (!sc) throw new Error(`scenario ${name} is not in the manifest`);
  const file = params.get("file") ?? "/fixtures/k3/bunny_cloud.wvdb";
  const res = await fetch(file);
  if (!res.ok) throw new Error(`${file}: ${res.status} (copy his bunny_cloud.wvdb into fixtures/k3/)`);
  const volume = wvdbDenseVolume(readWvdb(new Uint8Array(await res.arrayBuffer())), { mipmaps: false });
  const source = vdbDenseSource(device, volume, { label: "k6-reference report" });
  const W = Number(params.get("w") ?? 320);
  const H = Number(params.get("h") ?? 180);
  const md = sc.metadata;
  const vol = md.volumes.find((v) => v.visible !== false) ?? md.volumes[0]!;
  const others = md.volumes.filter((v) => v !== vol && v.visible !== false);
  const phaseArg = (params.get("phase") ?? "0.8,-0.3,0.2").split(",").map(Number);
  const phase = { g: phaseArg[0]!, backG: phaseArg[1] ?? 0, backWeight: phaseArg[2] ?? 0 };
  const volumes = [vol, ...others].map((v) => {
    const l2w = Array.from(vdbRefMultiply4(v.matrixWorld, v.textures[0]!.matrix));
    return {
      localToWorld: l2w,
      // OUR normalisation (k3-frame: normalizeDensity with the texture's mean), his extinction.
      sigma: vdbReferenceSigma({ localToWorld: l2w, extinction: v.extinction, normalizeDensity: true, meanNonZero: volume.meanDensity }),
      majorant: volume.maxDensity,
      albedo: [1, 1, 1] as VDBRefVec3,
      phase,
    };
  });
  const frameLights = md.lights.flatMap((l): VDBReferenceFrameLight[] => {
    const c = l.color.map((x) => x * l.intensity) as VDBRefVec3;
    const p: VDBRefVec3 = [l.matrixWorld[12]!, l.matrixWorld[13]!, l.matrixWorld[14]!];
    if (l.type === "DirectionalLight") return [{ kind: "directional" as const, direction: p, color: c }];
    if (l.type === "AmbientLight") return [{ kind: "ambient" as const, color: c }];
    if (l.type === "HemisphereLight") return [{ kind: "hemisphere" as const, sky: c, ground: (l.groundColor ?? [0, 0, 0]).map((x) => x * l.intensity) as VDBRefVec3, up: p }];
    result.warnings.push(`report: light ${l.type} not mapped`);
    return [];
  });
  const frame = vdbReferenceLightsFromFrame(frameLights);
  const scene: VDBReferenceScene = {
    camera: { cameraToWorld: md.camera.matrixWorld, projectionInverse: Array.from(vdbRefInvert4(md.camera.projectionMatrix)) },
    volumes,
    lights: frame.lights,
    environment: frame.environment,
  };
  const settings: Settings = { spp: Number(params.get("spp") ?? 1), maxBounces: Number(params.get("maxBounces") ?? 512), seed: Number(params.get("seed") ?? 1), rrStart: 4 };
  const job: GpuJob = { source, width: W, height: H };
  const ref = await makeRenderer(device, job);
  ref.setScene(scene, settings);
  const maxBatches = Number(params.get("batches") ?? 256);
  const budget = Number(params.get("budget") ?? 120) * 1000;
  const chunk = Number(params.get("chunk") ?? 1);
  const t0 = performance.now();
  while (ref.batches < maxBatches && (ref.batches < 4 || performance.now() - t0 < budget)) {
    const enc = device.createCommandEncoder();
    ref.render(enc, Math.min(chunk, maxBatches - ref.batches));
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const accum = await ref.readback();
  const { mean, stderr } = vdbReferenceStatistics(accum);
  result.report = {
    scenario: name,
    width: W,
    height: H,
    batches: accum.batches,
    spp: settings.spp,
    maxBounces: settings.maxBounces,
    phase,
    sigma: volumes.map((v) => v.sigma),
    lights: frame.lights,
    environment: frame.environment,
    seconds: (performance.now() - t0) / 1000,
    mean: toBase64(Float32Array.from(mean)),
    stderr: toBase64(Float32Array.from(stderr)),
  };
  log(`report ${name}: ${W}x${H}, ${accum.batches} batches × ${settings.spp} spp in ${((performance.now() - t0) / 1000).toFixed(1)} s`, "ok");
  ref.dispose();
  source.dispose();
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
  if (MODE === "analytic") await analytic(device);
  else if (MODE === "twin") {
    const which = params.get("source") ?? "dense,sparse";
    if (which.includes("dense")) await twinDense(device);
    if (which.includes("sparse")) await twinSparse(device);
  } else if (MODE === "report") await report(device);
  else throw new Error(`unknown case ${MODE}`);
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
