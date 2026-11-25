/**
 * K6.7 harness: the core sparse source (`vdbSparseSourceGpu`) on raw WebGPU, no three. Its `vdbSrc*` WGSL runs in
 * probe compute kernels (the source at group 1, binding 2: a non-zero slot) on fixed-seed inputs and is compared
 * with the CPU twins (`tools/frame/sparse.ts`: `vdbSparseDensityCpu`, `vdbSparseMaskCpu`, `vdbSparseAlbedoCpu`,
 * `vdbSparseEmissionCpu`, `vdbSparseSkipSpanCpu`).
 *
 * Grids: the committed CC0 cloud (`cloud_04_variant_0000.wvdb` as a NanoVDB FLOAT grid), a synthetic blob density
 * with empty 8³ cells (and its Fp8 quantization), a mask, three albedo grids and an emission grid, each on its own
 * index box. Three sources over one pool (`'none'` + clamp01, `'coarse-pyramid'` pyramidMax 16, `'node-average'`),
 * four live volumes each (one removed and re-added: slot reuse; one grid under two boxes: two pyramids).
 *
 * Query:
 * - `?n=` samples per source (default 1024).
 * - `?mutate=` a GPU-side break the probe must catch (the WGSL text patched before compiling): `samplePos` (no half
 *   voxel shift), `slot` (every volume reads slot 0), `origin` (pyramid base ignores its packed origin), `blend`
 *   (wrong level-0 → pyramid blend), `dilate` (coarse HDDA uses the level-0 bound).
 * - `?swap=1`: G16a: a `followImage` source over a coarse FpN preview, then `pool.replace` with the full FLOAT
 *   grid; the shader modules and pipelines created across the swap are counted (must be 0) and the probe repeated
 *   on the new image.
 * Results: `window.__VDB_K6_SPARSE__` (e2e/k6-sparse.spec.ts).
 */
import { VDBGridPool } from "@pmndrs/vdb";
import { vdbSparseSourceGpu, type VDBSparseSourceGpu, type VDBSparseVolumeGpu } from "../../../packages/vdb/src/core/internal.js";
import {
  readWvdb,
  vdbBoxPyramid,
  vdbBuildFromDense,
  vdbPyramidToHalf,
  vdbQuantize,
  vdbResampleGrid,
  vdbSparseAlbedoCpu,
  vdbSparseDensityCpu,
  vdbSparseEmissionCpu,
  vdbSparseLodLevelsCpu,
  vdbSparseMaskCpu,
  vdbSparseSkipSpanCpu,
  wvdbDenseToNvdb,
  wvdbDenseVolume,
  type VDBSparseVolumeCpu,
} from "../../../packages/vdb/src/tools/internal.js";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

type Vec3 = [number, number, number];

interface Stat {
  count: number;
  over: number;
  maxAbs: number;
  maxRel: number;
  firstBad?: unknown;
}

interface SourceResult {
  name: string;
  lod: string;
  volumes: number;
  density: Stat;
  aux: Stat;
  exact: { mismatches: number; count: number; firstBad?: unknown };
  skip: { count: number; mismatches: number; skipped: number; enabled: number; firstBad?: unknown };
  /** CPU check of the GPU spans: density samples before ts must all be exactly 0 (HDDA exactness). */
  exactness: { checked: number; violations: number; firstBad?: unknown };
  atlas: number[] | null;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  mutate: string | null;
  sources: SourceResult[];
  swap?: { builds: number; buildsByCall: Record<string, number>; relayouts: number; texelsBefore: number[]; texelsAfter: number[]; versionMoved: boolean; rebind: boolean; probe: SourceResult };
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K6_SPARSE__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const N = Math.max(64, Number(params.get("n") ?? 1024));
const MUTATE = params.get("mutate");
const SWAP = params.get("swap") === "1";
const SLOT = { group: 1, firstBinding: 2 };
const STAGE_COMPUTE = 0x4;

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, mutate: MUTATE, sources: [], warnings: [], uncaptured: [] };
window.__VDB_K6_SPARSE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

/** mulberry32: fixed-seed inputs. */
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
const f = Math.fround;

/** A dense field (x-major, z fastest: vdbBuildFromDense's order) as a FLOAT grid at `origin`. */
function grid(dims: Vec3, origin: Vec3, fn: (x: number, y: number, z: number) => number): Uint32Array {
  const values = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let x = 0; x < dims[0]; x++) for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) values[(x * dims[1] + y) * dims[2] + z] = fn(x, y, z);
  return vdbBuildFromDense(values, dims, { origin, voxelSize: 0.25 });
}

/** The WGSL patches of `?mutate=` (each must make the probe fail). */
const MUTATIONS: Record<string, [string, string]> = {
  samplePos: ["bmin + uvw * bsize - 0.5", "bmin + uvw * bsize"],
  slot: ["let g = vdbSrcVolumes[v].ids.x;", "let g = vdbSrcVolumes[0u].ids.x;"],
  origin: ["size - 0.5) + vdbSrcVolumes[v].pyrBase.xyz", "size - 0.5)"],
  blend: ["l / L0)", "l / (L0 + 1.0))"],
  dilate: ["dilate = 1u;", "dilate = 0u;"],
};

const PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> probeIn: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> probeOut: array<vec4f>;

@compute @workgroup_size(64)
fn csDensity(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&probeIn) / 3u {
        return;
    }
    let a = probeIn[3u * i];
    let v = bitcast<u32>(probeIn[3u * i + 2u].z);
    probeOut[4u * i] = vec4f(vdbSrcDensity(v, a.xyz, a.w), vdbSrcLodLevels(v), vdbSrcMaxIn(v, vec3f(0.0), vec3f(1.0)), 0.0);
    probeOut[4u * i + 1u] = vec4f(vdbSrcTexels(v), 0.0);
}

@compute @workgroup_size(64)
fn csAux(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&probeIn) / 3u {
        return;
    }
    let a = probeIn[3u * i];
    let v = bitcast<u32>(probeIn[3u * i + 2u].z);
    probeOut[4u * i + 2u] = vec4f(vdbSrcAlbedo(v, a.xyz, a.w), vdbSrcMask(v, a.xyz, a.w));
    probeOut[4u * i + 3u].x = vdbSrcEmission(v, a.xyz, a.w);
}

@compute @workgroup_size(64)
fn csSkip(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&probeIn) / 3u {
        return;
    }
    let a = probeIn[3u * i];
    let b = probeIn[3u * i + 1u];
    let c = probeIn[3u * i + 2u];
    let v = bitcast<u32>(c.z);
    let span = vdbSrcSkipSpan(v, a.xyz, b.xyz, b.w, c.x, a.w, c.y);
    probeOut[4u * i + 3u].y = span.x;
    probeOut[4u * i + 3u].z = span.y;
    probeOut[4u * i + 3u].w = vdbSrcSkipEmpty(v, a.xyz, b.xyz, b.w, c.x);
}
`;

interface Probe {
  pipelines: Record<"csDensity" | "csAux" | "csSkip", GPUComputePipeline>;
  group1Layout: GPUBindGroupLayout;
  group0Layout: GPUBindGroupLayout;
}

async function makeProbe(device: GPUDevice, source: VDBSparseSourceGpu): Promise<Probe> {
  let wgsl = source.wgsl({ slot: SLOT, profile: "march" });
  if (MUTATE) {
    const m = MUTATIONS[MUTATE];
    if (!m) throw new Error(`unknown mutation ${MUTATE}`);
    if (wgsl.includes(m[0])) wgsl = wgsl.split(m[0]).join(m[1]);
  }
  const code = `${source.moduleWgsl(SLOT)}\n${wgsl}\n${PROBE_WGSL}`;
  device.pushErrorScope("validation");
  const group0Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: STAGE_COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: STAGE_COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const group1Layout = device.createBindGroupLayout({ entries: source.layoutEntries(SLOT, STAGE_COMPUTE) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [group0Layout, group1Layout] });
  const module = device.createShaderModule({ code, label: `k6-sparse ${source.lod}` });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) throw new Error(`WGSL (${source.lod}): ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("\n")}`);
  const pipelines = {} as Probe["pipelines"];
  for (const entryPoint of ["csDensity", "csAux", "csSkip"] as const) {
    pipelines[entryPoint] = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint } });
  }
  const err = await device.popErrorScope();
  if (err) throw new Error(`probe pipelines (${source.lod}): ${err.message}`);
  return { pipelines, group0Layout, group1Layout };
}

interface Sample {
  v: number;
  uvw: Vec3;
  lod: number;
  d: Vec3;
  t0: number;
  t1: number;
  cutoff: number;
}

function inputs(seed: number, volumes: readonly { v: number; lodLevels: number }[], n: number): Sample[] {
  const r = rng(seed);
  const out: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const vol = volumes[Math.floor(r() * volumes.length)]!;
    const kind = r();
    // Some samples sit outside the box (clamp to edge); the skip rays start inside it.
    const uvw = [0, 1, 2].map(() => f(kind < 0.15 ? -0.1 + 1.2 * r() : 0.02 + 0.96 * r())) as Vec3;
    const maxLod = vol.lodLevels;
    const pick = r();
    const lod = f(pick < 0.2 ? 0 : pick < 0.35 ? Math.floor(r() * (maxLod + 2)) : pick < 0.55 ? 3 * r() : r() * (maxLod + 1.5));
    let d = [0, 1, 2].map(() => r() * 2 - 1) as Vec3;
    const len = Math.hypot(...d) || 1;
    const scale = 0.5 + r();
    d = d.map((x) => f((x / len) * scale)) as Vec3;
    const o = uvw.map((x) => Math.min(Math.max(x, 0.02), 0.98)) as Vec3;
    // March end: inside the box (the TSL march stops at the box exit), at most 1.5.
    let exit = 1.5;
    for (let a = 0; a < 3; a++) {
      const da = d[a]!;
      const oa = o[a]!;
      if (da > 0) exit = Math.min(exit, (1 - oa) / da);
      else if (da < 0) exit = Math.min(exit, -oa / da);
    }
    const t0 = f(r() * 0.2 * exit);
    const t1 = f(Math.max(t0, exit * (0.6 + 0.4 * r())));
    const cutoff = f([0, 0, 0.05, 0.3][Math.floor(r() * 4)]!);
    out.push({ v: vol.v, uvw: kind < 0.15 ? uvw : o.map(f) as Vec3, lod, d, t0, t1, cutoff });
  }
  return out;
}

async function runProbe(device: GPUDevice, source: VDBSparseSourceGpu, probe: Probe, samples: Sample[]): Promise<Float32Array> {
  const n = samples.length;
  const input = new Float32Array(n * 12);
  const u = new Uint32Array(input.buffer);
  samples.forEach((s, i) => {
    input.set([...s.uvw, s.lod, ...s.d, s.t0, s.t1, s.cutoff], i * 12);
    u[i * 12 + 10] = s.v;
  });
  const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, input);
  const outBytes = n * 64;
  const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  // Host order: the pool's update, then the source's, then the bind group (Traps).
  source.pool.update(encoder);
  source.update(encoder);
  const group0 = device.createBindGroup({ layout: probe.group0Layout, entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
  const group1 = device.createBindGroup({ layout: probe.group1Layout, entries: source.bindEntries(SLOT) });
  const pass = encoder.beginComputePass();
  pass.setBindGroup(0, group0);
  pass.setBindGroup(1, group1);
  for (const p of [probe.pipelines.csDensity, probe.pipelines.csAux, probe.pipelines.csSkip]) {
    pass.setPipeline(p);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
  device.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.destroy();
  inBuf.destroy();
  outBuf.destroy();
  return out;
}

const stat = (): Stat => ({ count: 0, over: 0, maxAbs: 0, maxRel: 0 });
function check(s: Stat, gpu: number, cpu: number, tol: number, detail: () => unknown): void {
  s.count++;
  const d = Math.abs(gpu - cpu);
  const ok = d <= tol || (Number.isNaN(gpu) && Number.isNaN(cpu));
  s.maxAbs = Math.max(s.maxAbs, Number.isFinite(d) ? d : Infinity);
  s.maxRel = Math.max(s.maxRel, tol > 0 ? d / tol : d);
  if (!ok) {
    s.over++;
    s.firstBad ??= { gpu, cpu, tol, ...(detail() as object) };
  }
}

/**
 * Tolerances (justified in docs/archive/handoffs/K6-7.md): level 0 (the pool's f32 trilinear vs the f64 twin, sample
 * positions in f32): 1e-4 (1 + scale); a pyramid tap adds 2⁻⁷ scale: 1 f16 ulp per texel (the GPU pyramid is within 1
 * ulp of the CPU one, K3.2a) plus 8-bit filter weights on hardware (3 lerps). `scale` = the largest |tap|.
 */
const tolOf = (scale: number, filtered: boolean) => 1e-4 * (1 + scale) + (filtered ? 2 ** -7 * scale : 0);

function compare(name: string, source: VDBSparseSourceGpu, cpu: Map<number, VDBSparseVolumeCpu>, samples: Sample[], out: Float32Array): SourceResult {
  const res: SourceResult = {
    name,
    lod: source.lod,
    volumes: cpu.size,
    density: stat(),
    aux: stat(),
    exact: { mismatches: 0, count: 0 },
    skip: { count: 0, mismatches: 0, skipped: 0, enabled: 0 },
    exactness: { checked: 0, violations: 0 },
    atlas: source.atlas ? [source.atlas.width, source.atlas.height, source.atlas.depthOrArrayLayers] : null,
  };
  const slotOf = new Map(source.volumes.map((s) => [s.v, s]));
  samples.forEach((s, i) => {
    const vol = cpu.get(s.v)!;
    const slot = slotOf.get(s.v)!;
    const o = out.subarray(i * 16, i * 16 + 16);
    const where = () => ({ i, v: s.v, uvw: s.uvw, lod: s.lod });
    const dens = vdbSparseDensityCpu(vol, s.uvw, s.lod);
    check(res.density, o[0]!, dens.value, tolOf(dens.scale, dens.filtered), () => ({ ...where(), filtered: dens.filtered }));
    // Exact fields: LOD range, max, texels.
    res.exact.count++;
    const want = [vdbSparseLodLevelsCpu(vol), f(slot.stats.max), ...slot.texels];
    const got = [o[1]!, o[2]!, o[4]!, o[5]!, o[6]!];
    if (want.some((w, k) => w !== got[k])) {
      res.exact.mismatches++;
      res.exact.firstBad ??= { ...where(), want, got };
    }
    const alb = vdbSparseAlbedoCpu(vol, s.uvw);
    for (let c = 0; c < 3; c++) check(res.aux, o[8 + c]!, alb.value[c]!, tolOf(alb.scale, false), () => ({ ...where(), channel: `albedo${c}` }));
    const m = vdbSparseMaskCpu(vol, s.uvw);
    check(res.aux, o[11]!, m.value, tolOf(m.scale, false), () => ({ ...where(), channel: "mask" }));
    const e = vdbSparseEmissionCpu(vol, s.uvw);
    check(res.aux, o[12]!, e.value, tolOf(e.scale, false), () => ({ ...where(), channel: "emission" }));

    // HDDA: the span and the frozen level-0 form against the twin (o = uvw: these samples are inside the box).
    const span = vdbSparseSkipSpanCpu(vol, s.uvw, s.d, s.t0, s.t1, s.lod, s.cutoff);
    const lvl0 = vdbSparseSkipSpanCpu(vol, s.uvw, s.d, s.t0, s.t1, 0, 0);
    const gpuSpan = [o[13]!, o[14]!, o[15]!];
    const cpuSpan = [span[0], span[1], lvl0[0]];
    res.skip.count++;
    if (span[1] < 1e29) res.skip.enabled++;
    if (gpuSpan[0]! > s.t0) res.skip.skipped++;
    if (cpuSpan.some((c, k) => Math.abs(c - gpuSpan[k]!) > 1e-5 * (1 + Math.abs(c)))) {
      res.skip.mismatches++;
      res.skip.firstBad ??= { ...where(), d: s.d, t0: s.t0, t1: s.t1, cutoff: s.cutoff, gpu: gpuSpan, cpu: cpuSpan };
    }
    // Exactness of the GPU's answer: every sample the march would skip (t < ts) reads 0 after the cutoff and mask.
    const ts = Math.min(gpuSpan[0]!, s.t1);
    const inside = s.uvw.every((x) => x >= 0.02 && x <= 0.98);
    if (inside && ts > s.t0) {
      for (let k = 0; k < 8; k++) {
        const t = s.t0 + ((ts - s.t0) * (k + 0.5)) / 8;
        const p = s.uvw.map((x, a) => x + s.d[a]! * t);
        const value = vdbSparseDensityCpu(vol, p, s.lod).value * vdbSparseMaskCpu(vol, p).value;
        res.exactness.checked++;
        if (value > s.cutoff) {
          res.exactness.violations++;
          res.exactness.firstBad ??= { ...where(), t, ts, value, cutoff: s.cutoff };
        }
      }
    }
  });
  return res;
}

/** The CPU side of one volume (the boxes the source uses: the slot's, or each grid's own). */
function cpuVolume(pool: VDBGridPool, source: VDBSparseSourceGpu, slot: VDBSparseVolumeGpu, images: Map<number, Uint32Array>, pyramidMax: number): VDBSparseVolumeCpu {
  const own = (id: number) => {
    const b = pool.info(id).indexBBox;
    return { min: [...b.min] as Vec3, max: [...b.max] as Vec3 };
  };
  const ids = slot.ids;
  const albedo = ids.albedo === undefined ? undefined : Array.isArray(ids.albedo) ? (ids.albedo as number[]) : [ids.albedo as number];
  const density = images.get(ids.density)!;
  return {
    lod: source.lod,
    clamp01: (source as unknown as { normalize: string }).normalize === "clamp01",
    density: { image: density, box: slot.indexBox },
    ...(ids.mask !== undefined ? { mask: { image: images.get(ids.mask)!, box: own(ids.mask) } } : {}),
    ...(albedo ? { albedo: { images: albedo.map((id) => images.get(id)!), box: own(albedo[0]!) } } : {}),
    ...(ids.emission !== undefined ? { emission: { image: images.get(ids.emission)!, box: own(ids.emission) } } : {}),
    ...(source.lod === "coarse-pyramid" ? { pyramid: vdbPyramidToHalf(vdbBoxPyramid(density, { indexBox: slot.indexBox, pyramidMax })) } : {}),
  };
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

  // Grids.
  const cloud = wvdbDenseToNvdb(wvdbDenseVolume(readWvdb(new Uint8Array(await (await fetch(cloudWvdbUrl)).arrayBuffer())), { mipmaps: false }));
  const blob = (x: number, y: number, z: number) => {
    const a = Math.hypot(x - 12, y - 14, z - 10) / 9;
    const b = Math.hypot(x - 29, y - 22, z - 18) / 7;
    const v = Math.max(0, 1 - a) * 1.4 + Math.max(0, 1 - b) * (0.8 + 0.2 * Math.sin(x * 0.7 + z * 0.3));
    return v > 0.02 ? v : 0;
  };
  const S1 = grid([40, 36, 28], [-13, 5, 70], blob);
  const S1q = vdbQuantize(S1, "fp8");
  const M = grid([20, 20, 20], [-5, 10, 75], (x, y, z) => 0.2 + 0.8 * Math.abs(Math.sin(x * 0.3 + y * 0.2 - z * 0.1)));
  const R = grid([16, 18, 14], [-13, 5, 70], (x, y) => -0.2 + 1.5 * ((x + y) / 34));
  const G = grid([16, 18, 14], [-13, 5, 70], (_x, y, z) => 0.5 + 0.6 * Math.cos(y * 0.4 + z * 0.2));
  const B = grid([16, 18, 14], [-13, 5, 70], (x, _y, z) => (x * z) / 210);
  const E = grid([24, 24, 24], [0, 10, 80], (x, y, z) => Math.max(0, 1 - Math.hypot(x - 12, y - 12, z - 12) / 12));

  const pool = await VDBGridPool.create(device, { label: "k6-sparse" });
  const images = new Map<number, Uint32Array>();
  const add = (image: Uint32Array) => {
    const id = pool.add(image);
    images.set(id, image);
    return id;
  };
  const ids = { cloud: add(cloud.image), s1: add(S1), s1q: add(S1q), m: add(M), r: add(R), g: add(G), b: add(B), e: add(E) };
  pool.flush();
  await pool.settled();

  const configs = [
    { name: "none+clamp01", options: { lod: "none" as const, normalize: "clamp01" as const, fastTrilinear: true, cacheAccessor: true }, seed: 11 },
    { name: "coarse-pyramid", options: { lod: "coarse-pyramid" as const, pyramidMax: 16, fastTrilinear: false, cacheAccessor: true }, seed: 22 },
    { name: "node-average", options: { lod: "node-average" as const, fastTrilinear: true, cacheAccessor: false }, seed: 33 },
  ];
  for (const c of configs) {
    const source = vdbSparseSourceGpu(pool, { ...c.options, types: ["float", "fp8"], maxVolumes: 8, label: `k6-sparse ${c.name}` });
    const first = source.volume({ density: ids.cloud }, { indexBox: cloud.textureIndexBox, image: cloud.image });
    source.volume({ density: ids.s1, mask: ids.m, albedo: [ids.r, ids.g, ids.b], emission: ids.e }, { image: S1 });
    source.volume({ density: ids.s1q, albedo: ids.r }, { image: S1q });
    // Slot reuse: remove the first volume and add it again (it gets v = 0 back), plus one grid under a second box.
    first.remove();
    source.volume({ density: ids.cloud }, { indexBox: cloud.textureIndexBox, image: cloud.image });
    source.volume({ density: ids.s1 }, { indexBox: { min: [-5, 10, 75], max: [18, 30, 94] }, image: S1 });
    const probe = await makeProbe(device, source);
    const cpu = new Map(source.volumes.map((s) => [s.v, cpuVolume(pool, source, s, images, c.options.pyramidMax ?? 128)]));
    const samples = inputs(c.seed, source.volumes, N);
    const out = await runProbe(device, source, probe, samples);
    const r = compare(c.name, source, cpu, samples, out);
    result.sources.push(r);
    const ok = r.density.over + r.aux.over + r.exact.mismatches + r.skip.mismatches + r.exactness.violations === 0;
    log(`${c.name}: ${JSON.stringify(r)}`, ok ? "ok" : "fail");
    source.dispose();
  }

  if (SWAP) await swap(device, pool, cloud, images);
  pool.dispose();
}

/** G16a: a coarse-first swap through a followImage source creates no shader module and no pipeline. */
async function swap(device: GPUDevice, pool: VDBGridPool, cloud: ReturnType<typeof wvdbDenseToNvdb>, images: Map<number, Uint32Array>): Promise<void> {
  const preview = vdbResampleGrid(cloud.image, { dimensions: [32, 32, 11], quantize: "fpn" }).image;
  const id = pool.add(preview);
  images.set(id, preview);
  pool.flush();
  await pool.settled();
  const source = vdbSparseSourceGpu(pool, { followImage: true, pyramidMax: 16, types: ["fpn", "float"], label: "k6-sparse swap" });
  const vol = source.volume({ density: id }, { image: preview });
  const probe = await makeProbe(device, source);
  const warm = inputs(44, source.volumes, 128);
  await runProbe(device, source, probe, warm);
  const texelsBefore = [...vol.texels];
  const versionBefore = vol.version;

  // Count every shader module and pipeline the device creates from here on.
  const counted: Record<string, number> = {};
  const d = device as unknown as Record<string, (...a: unknown[]) => unknown>;
  const originals: Record<string, (...a: unknown[]) => unknown> = {};
  for (const name of ["createShaderModule", "createComputePipeline", "createComputePipelineAsync", "createRenderPipeline", "createRenderPipelineAsync"]) {
    originals[name] = d[name]!.bind(device);
    d[name] = (...a: unknown[]) => {
      counted[name] = (counted[name] ?? 0) + 1;
      return originals[name]!(...a);
    };
  }
  pool.replace(id, cloud.image);
  images.set(id, cloud.image);
  const encoder = device.createCommandEncoder();
  pool.update(encoder);
  const u = source.update(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  // The new image's stats: copied in the swap's encoder, mapped by the next update; let them land so the table's max
  // is the new one before the probe reads it.
  const next = device.createCommandEncoder();
  pool.update(next);
  source.update(next);
  device.queue.submit([next.finish()]);
  await source.settled();
  const samples = inputs(55, source.volumes, N);
  const out = await runProbe(device, source, probe, samples);
  for (const [name, fn] of Object.entries(originals)) d[name] = fn;
  const cpu = new Map(source.volumes.map((s) => [s.v, cpuVolume(pool, source, s, images, 16)]));
  const r = compare("swap (followImage, after)", source, cpu, samples, out);
  result.swap = {
    builds: Object.values(counted).reduce((a, b) => a + b, 0),
    buildsByCall: counted,
    relayouts: u.relayouts,
    texelsBefore,
    texelsAfter: [...vol.texels],
    versionMoved: vol.version !== versionBefore,
    rebind: u.rebind,
    probe: r,
  };
  log(`swap: ${JSON.stringify(result.swap)}`, result.swap.builds === 0 && r.density.over === 0 ? "ok" : "fail");
  source.dispose();
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
