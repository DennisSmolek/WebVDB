/**
 * K3.2d sampling microbenchmark: one compute kernel per sampling variant, each thread marching `steps`
 * samples along a random ray through the density grid's index box (`step` voxels apart, clamped to the
 * box as the sparse source clamps), summing them. Timed with compute-pass timestamps (median of
 * `repeat`); every sparse variant's per-thread sums must be bit-identical to `init` (the current
 * `vdbAccessorInit` + `vdbSample` per sample, what `vdbSparseSource` did before K3.2d).
 *
 * Variants:
 * - `noop`: the ray and clamp arithmetic only (the overhead under every variant).
 * - `dense`: one hardware-filtered `textureSampleLevel` of the dense texture the dense source uploads
 *   (r8unorm; cloud04 and bunny are the same voxels, smoke and wdas a 128 bake).
 * - `init` / `fast-init`: a fresh accessor per sample, `vdbSample` / `vdbSampleFast` (`fastTrilinear`).
 * - `reuse` / `fast-reuse`: one accessor per thread, reused along the ray (its node cache).
 * - `private` / `fast-private`: the accessor in a `var<private>` copied in and out per sample (how a TSL
 *   loop can keep one: the sparse source's `cacheAccessor`).
 * - `check`: `vdbSample` and `vdbSampleFast` at the same points in one kernel; counts bit mismatches.
 *
 * Query: `?assets=cloud04,bunny,smoke,wdas`, `?steps=32`, `?strides=0.5,2,8` (voxels), `?threads=262144`,
 * `?repeat=15`, `?variants=` (comma list). Results: `window.__VDB_K3_SAMPLE_BENCH__`.
 */
import { VDBGridPool, vdbModuleWgsl, type VDBGridTypeName } from "@pmndrs/vdb";
import { NvdbFile, buildFromVdb, nvdbToWvdb, parseVdb, readWvdb, vdbToWvdb, wvdbDenseToNvdb, wvdbDenseVolume, type WvdbDenseVolume } from "@pmndrs/vdb/tools";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

type Vec3 = [number, number, number];

interface Row {
  asset: string;
  stride: number;
  variant: string;
  /** Median GPU ms (timestamps) or queue wall ms. */
  ms: number;
  /** ns per sample. */
  nsPerSample: number;
  /** Threads whose sum differs from `init` bit for bit (sparse variants); `check`: sample mismatches. */
  mismatches: number | null;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  timestampQuery?: boolean;
  rows: Row[];
  assets: { name: string; type: string; texels: number[]; denseDims: number[] }[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_SAMPLE_BENCH__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const ASSETS = (params.get("assets") ?? "cloud04,bunny,smoke,wdas").split(",").filter(Boolean);
const STEPS = Number(params.get("steps") ?? 32);
const STRIDES = (params.get("strides") ?? "0.5,2,8").split(",").map(Number);
const THREADS = Number(params.get("threads") ?? 262144);
const REPEAT = Number(params.get("repeat") ?? 15);
const ALL_VARIANTS = ["noop", "dense", "init", "fast-init", "reuse", "fast-reuse", "private", "fast-private", "check"];
const VARIANTS = (params.get("variants") ?? ALL_VARIANTS.join(",")).split(",").filter((v) => ALL_VARIANTS.includes(v));

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, rows: [], assets: [], uncaptured: [] };
window.__VDB_K3_SAMPLE_BENCH__ = result;
const log = (text: string, cls = "") => {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
};

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} (fixtures are git-ignored)`);
  return r.arrayBuffer();
}

/** As k3-timing: the sparse image (+ its texture index box) and the dense volume the dense source uploads. */
async function loadAsset(name: string): Promise<{ image: Uint32Array; dense: WvdbDenseVolume; indexBox?: { min: Vec3; max: Vec3 } }> {
  if (name === "cloud04" || name === "bunny") {
    const url = name === "cloud04" ? cloudWvdbUrl : "/fixtures/k3/bunny_cloud.wvdb";
    const dense = wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(url))), { mipmaps: false });
    const grid = wvdbDenseToNvdb(dense);
    return { image: grid.image, dense, indexBox: grid.textureIndexBox };
  }
  if (name === "smoke") {
    const file = parseVdb(await fetchBytes("/fixtures/vdb-samples/smoke.vdb"));
    const image = buildFromVdb(file.grids.find((g) => g.valueType === "float")!);
    return { image, dense: wvdbDenseVolume(readWvdb(vdbToWvdb(file, { resolution: 128 }).bytes), { mipmaps: false }) };
  }
  if (name === "wdas") {
    const image = NvdbFile.fromArrayBuffer(await fetchBytes("/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb")).gridImage(0);
    return { image, dense: wvdbDenseVolume(readWvdb(nvdbToWvdb(image, { resolution: 128 }).bytes), { mipmaps: false }) };
  }
  throw new Error(`unknown asset ${name}`);
}

const HOST = /* wgsl */ `
struct BenchParams {
    boxMin: vec3f,
    threads: u32,
    boxSize: vec3f,
    steps: u32,
    stride: f32,
    grid: u32,
    seed: u32,
    pad: u32,
}
@group(0) @binding(1) var<storage, read_write> benchOut: array<u32>;
@group(0) @binding(2) var<uniform> P: BenchParams;
@group(0) @binding(3) var benchTex: texture_3d<f32>;
@group(0) @binding(4) var benchSampler: sampler;

var<private> benchAcc: VdbAccessor;
var<private> benchAccGrid: u32 = 0xffffffffu;

fn benchHash(v: u32) -> u32 {
    var x = v * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}
fn benchRand(s: ptr<function, u32>) -> f32 {
    *s = benchHash(*s);
    return f32(*s >> 8u) / 16777216.0;
}
fn benchPoint(o: vec3f, d: vec3f, k: u32) -> vec3f {
    return clamp(o + d * (P.stride * f32(k)), P.boxMin, P.boxMin + P.boxSize - vec3f(1.0));
}
fn benchDense(p: vec3f) -> f32 {
    return textureSampleLevel(benchTex, benchSampler, (p - P.boxMin + vec3f(0.5)) / P.boxSize, 0.0).r;
}
fn benchPrivate(p: vec3f) -> f32 {
    if benchAccGrid != P.grid {
        benchAcc = vdbAccessorInit(P.grid);
        benchAccGrid = P.grid;
    }
    var a = benchAcc;
    let v = vdbSample(&a, p);
    benchAcc = a;
    return v;
}
fn benchPrivateFast(p: vec3f) -> f32 {
    if benchAccGrid != P.grid {
        benchAcc = vdbAccessorInit(P.grid);
        benchAccGrid = P.grid;
    }
    var a = benchAcc;
    let v = vdbSampleFast(&a, p);
    benchAcc = a;
    return v;
}
`;

/** The loop body per variant: `p` is the point, `s` the f32 sum (or `m` the mismatch count for check). */
const BODY: Record<string, string> = {
  noop: "s += p.x * 1e-9;",
  dense: "s += benchDense(p);",
  init: "var a = vdbAccessorInit(P.grid);\n        s += vdbSample(&a, p);",
  "fast-init": "var a = vdbAccessorInit(P.grid);\n        s += vdbSampleFast(&a, p);",
  reuse: "s += vdbSample(&acc, p);",
  "fast-reuse": "s += vdbSampleFast(&acc, p);",
  private: "s += benchPrivate(p);",
  "fast-private": "s += benchPrivateFast(p);",
  check: "let x = vdbSample(&acc, p);\n        let y = vdbSampleFast(&acc2, p);\n        if bitcast<u32>(x) != bitcast<u32>(y) { m += 1u; }\n        s += x;",
};

function kernel(module: string, variant: string): string {
  return `${module}
${HOST}
@compute @workgroup_size(64)
fn benchMain(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if id >= P.threads {
        return;
    }
    var seed = benchHash(id ^ P.seed);
    let o = P.boxMin + vec3f(benchRand(&seed), benchRand(&seed), benchRand(&seed)) * P.boxSize;
    let z = benchRand(&seed) * 2.0 - 1.0;
    let phi = benchRand(&seed) * 6.2831853;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let d = vec3f(r * cos(phi), r * sin(phi), z);
    var acc = vdbAccessorInit(P.grid);
    var acc2 = vdbAccessorInit(P.grid);
    var s = 0.0;
    var m = 0u;
    for (var k = 0u; k < P.steps; k++) {
        let p = benchPoint(o, d, k);
        ${BODY[variant]}
    }
    benchOut[id] = select(bitcast<u32>(s), m, ${variant === "check" ? "true" : "false"});
}
`;
}

async function main(): Promise<void> {
  root.textContent = "";
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(" ");
  const timestamps = adapter.features.has("timestamp-query");
  result.timestampQuery = timestamps;
  const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ["timestamp-query"] : [], requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  log(`${result.adapter}; timestamps ${timestamps}`);

  const query = timestamps ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const resolveRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const out = device.createBuffer({ size: THREADS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const outRead = device.createBuffer({ size: THREADS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });

  const readOut = async (): Promise<Uint32Array> => {
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(out, 0, outRead, 0, THREADS * 4);
    device.queue.submit([e.finish()]);
    await outRead.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(outRead.getMappedRange().slice(0));
    outRead.unmap();
    return v;
  };

  for (const name of ASSETS) {
    const asset = await loadAsset(name);
    const pool = await VDBGridPool.create(device, { label: "k3-sample-bench" });
    const id = pool.add(asset.image);
    pool.flush();
    await pool.settled();
    const ginfo = pool.info(id);
    const types = [...pool.types] as VDBGridTypeName[];
    const box = asset.indexBox ?? { min: ginfo.indexBBox.min as Vec3, max: ginfo.indexBBox.max as Vec3 };
    const size: Vec3 = [box.max[0] - box.min[0] + 1, box.max[1] - box.min[1] + 1, box.max[2] - box.min[2] + 1];
    const dd = asset.dense.dimensions;
    result.assets.push({ name, type: ginfo.type, texels: size, denseDims: [...dd] });
    const tex = device.createTexture({ size: [dd[0], dd[1], dd[2]], dimension: "3d", format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: tex }, asset.dense.density.data, { bytesPerRow: dd[0], rowsPerImage: dd[1] }, [dd[0], dd[1], dd[2]]);
    // The sparse source's module: march + its helpers (+ fastTrilinear).
    const module = vdbModuleWgsl({ slot: { group: 0, firstBinding: 0 }, types, profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      ],
    });
    const bindGroup = device.createBindGroup({
      layout,
      entries: [
        pool.bindGroupEntry({ group: 0, firstBinding: 0 }),
        { binding: 1, resource: { buffer: out } },
        { binding: 2, resource: { buffer: uniform } },
        { binding: 3, resource: tex.createView() },
        { binding: 4, resource: sampler },
      ],
    });
    const pipelines = new Map<string, GPUComputePipeline>();
    for (const v of VARIANTS) {
      device.pushErrorScope("validation");
      const shader = device.createShaderModule({ code: kernel(module, v), label: `bench ${v}` });
      const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module: shader, entryPoint: "benchMain" } });
      const err = await device.popErrorScope();
      if (err) throw new Error(`${v}: ${err.message}`);
      pipelines.set(v, pipeline);
    }

    for (const stride of STRIDES) {
      const u = new ArrayBuffer(48);
      const f = new Float32Array(u);
      const w = new Uint32Array(u);
      f.set(box.min, 0);
      w[3] = THREADS;
      f.set(size, 4);
      w[7] = STEPS;
      f[8] = stride;
      w[9] = id;
      w[10] = 12345;
      device.queue.writeBuffer(uniform, 0, u);
      let reference: Uint32Array | null = null;
      for (const v of VARIANTS) {
        const pipeline = pipelines.get(v)!;
        const run = async (timed: boolean): Promise<number> => {
          const e = device.createCommandEncoder();
          const pass = e.beginComputePass(timed && query ? { timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(Math.ceil(THREADS / 64));
          pass.end();
          if (timed && query) {
            e.resolveQuerySet(query, 0, 2, resolve, 0);
            e.copyBufferToBuffer(resolve, 0, resolveRead, 0, 16);
          }
          const t0 = performance.now();
          device.queue.submit([e.finish()]);
          await device.queue.onSubmittedWorkDone();
          const wall = performance.now() - t0;
          if (!(timed && query)) return wall;
          await resolveRead.mapAsync(GPUMapMode.READ);
          const t = new BigUint64Array(resolveRead.getMappedRange().slice(0));
          resolveRead.unmap();
          return Number(t[1]! - t[0]!) / 1e6;
        };
        for (let i = 0; i < 3; i++) await run(false);
        const times: number[] = [];
        for (let i = 0; i < REPEAT; i++) times.push(await run(true));
        times.sort((a, b) => a - b);
        const ms = times[Math.floor(times.length / 2)]!;
        const values = await readOut();
        let mismatches: number | null = null;
        if (v === "init") reference = values;
        else if (v === "check") mismatches = values.reduce((s, x) => s + x, 0);
        else if (v !== "noop" && v !== "dense" && reference) {
          mismatches = 0;
          for (let i = 0; i < THREADS; i++) if (values[i] !== reference[i]) mismatches++;
        }
        const row: Row = { asset: name, stride, variant: v, ms, nsPerSample: (ms * 1e6) / (THREADS * STEPS), mismatches };
        result.rows.push(row);
        log(`${name} (${ginfo.type}) stride ${stride}: ${v.padEnd(12)} ${ms.toFixed(3)} ms  ${row.nsPerSample.toFixed(3)} ns/sample${mismatches !== null ? `  mismatches ${mismatches}` : ""}`, mismatches ? "fail" : "");
      }
    }
    tex.destroy();
    pool.dispose();
  }
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
