/**
 * K2.3 GPU gate for the core compute tools: `gridStats` and `decodeToAtlas`
 * on a real device against their `tools` CPU twins (`vdbGridStatsCpu`,
 * `vdbDecodeToAtlasCpu` + `atlasFromRaw`), and `vdbApplyValueTransform` round trips.
 *
 * Five grids share one pool (a filler grid first, so ids and image offsets
 * are not 0): box Float, sphere Fp8, torus FpN, sphere Float, and a dense
 * Float grid with negative values (the dump path). Half the calls record into
 * a host encoder (with the pool update), half let the tool submit.
 * Results go to `window.__VDB_COMPUTE__` for e2e/compute.spec.ts.
 */
import { VDBGridPool, vdbApplyValueTransform, vdbDecodeToAtlasGpu, vdbGridStatsGpu } from "@pmndrs/vdb";
import type { VDBAtlasFilter, VDBAtlasFormat, VDBDecodeToAtlasOptions } from "@pmndrs/vdb";
import { NvdbFile, atlasFromRaw, vdbBuildFromDense, vdbDecodeToAtlasCpu, vdbGridStatsCpu, getDim, vdbReadValueF32 } from "../../../packages/vdb/src/tools/internal.js";

interface StatsCheck {
  grid: string;
  fastPath: boolean;
  fastPathTwin: boolean;
  count: number;
  countTwin: number;
  minExact: boolean;
  maxExact: boolean;
  meanAbs: number;
  /** Sum over bins of |gpu - twin|. */
  histogramL1: number;
  histogramSum: number;
}

interface AtlasCheck {
  grid: string;
  label: string;
  dims: number[];
  voxels: number;
  /** Raw f32 vs twin. */
  exact: number;
  maxAbs: number;
  /** Formatted output (`read().data`) vs `atlasFromRaw(twin)`: elements that differ, and by how much. */
  dataDiff: number;
  dataMaxAbs: number;
}

interface TransformCheck {
  grid: string;
  leafSamples: number;
  tileSamples: number;
  leafDoubled: boolean;
  tilesUnchanged: boolean;
  /** Raw nearest atlas of the transformed grid vs the twin on the transformed image. */
  atlasExact: number;
  atlasVoxels: number;
  statsCountMatch: boolean;
  statsMaxExact: boolean;
  versionUnchanged: boolean;
}

interface ComputeReport {
  /** A host-encoder job read before the submit: the rejection code, the submit's validation error, the read after `submitted()`. */
  readBeforeSubmit?: { earlyRead: string; submitValidation: string | null; readAfterSubmittedOk: boolean };
  done: boolean;
  error?: string;
  missing?: string[];
  adapter?: string;
  stats: StatsCheck[];
  atlas: AtlasCheck[];
  transform?: TransformCheck;
  readbackFalseOk?: boolean;
  disposeRejects?: boolean;
  uncaptured: string[];
  lastError?: string;
}

declare global {
  interface Window {
    __VDB_COMPUTE__?: ComputeReport;
  }
}

const FIXTURES = ["box_fog_float", "sphere_fog_fp8", "torus_fog_fpn", "sphere_fog_float"];
const root = document.getElementById("demo-root")!;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

/** A dense Float grid with values in [-1, 1] (root min < 0: the dump path). */
function signedGrid(): Uint32Array {
  const dims: [number, number, number] = [22, 19, 13];
  const values = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let x = 0; x < dims[0]; x++) {
    for (let y = 0; y < dims[1]; y++) {
      for (let z = 0; z < dims[2]; z++) {
        const v = Math.sin(x * 0.37) * Math.cos(y * 0.21 + z * 0.5);
        values[(x * dims[1] + y) * dims[2] + z] = Math.abs(v) < 0.05 ? 0 : v;
      }
    }
  }
  return vdbBuildFromDense(values, dims, { origin: [-7, 3, -12], gridClass: "Unknown", gridName: "signed" });
}

function compareRaw(gpu: Float32Array, twin: Float32Array): { exact: number; maxAbs: number } {
  let exact = 0;
  let maxAbs = 0;
  for (let i = 0; i < gpu.length; i++) {
    if (Object.is(gpu[i], twin[i])) exact++;
    else maxAbs = Math.max(maxAbs, Math.abs(gpu[i]! - twin[i]!));
  }
  return { exact, maxAbs };
}

async function run(): Promise<ComputeReport> {
  const report: ComputeReport = { done: false, stats: [], atlas: [], uncaptured: [] };
  if (!navigator.gpu) return { ...report, done: true, error: "WebGPU unavailable (navigator.gpu undefined)." };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ...report, done: true, error: "No WebGPU adapter." };
  report.adapter = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(" ");
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => report.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const loaded = await Promise.all(
    FIXTURES.map(async (name) => {
      const res = await fetch(`/fixtures/primitives/${name}.nvdb`);
      return res.ok ? NvdbFile.fromArrayBuffer(await res.arrayBuffer()).gridImage(0) : null;
    }),
  );
  const missing = FIXTURES.filter((_, k) => loaded[k] === null);
  if (missing.length > 0) return { ...report, done: true, missing };
  const names = [...FIXTURES, "signed_float"];
  const images = [...(loaded as Uint32Array[]), signedGrid()];

  const pool = await VDBGridPool.create(device, { initialBytes: 4096, label: "compute harness" });
  pool.add(images[2]!, { label: "filler" });
  const ids = images.map((image, k) => pool.add(image, { label: names[k]! }));

  // gridStats: odd grids record into a host encoder together with the pool update.
  for (let k = 0; k < images.length; k++) {
    const histogramBins = k === 1 ? 64 : 256;
    let job;
    if (k % 2 === 1) {
      const encoder = device.createCommandEncoder();
      pool.update(encoder);
      job = vdbGridStatsGpu(device, pool, ids[k]!, { histogramBins, encoder });
      device.queue.submit([encoder.finish()]);
      job.submitted();
    } else {
      pool.flush();
      job = vdbGridStatsGpu(device, pool, ids[k]!, { histogramBins });
    }
    const gpu = await job.read();
    const twin = vdbGridStatsCpu(images[k]!, { histogramBins });
    let l1 = 0;
    for (let b = 0; b < histogramBins; b++) l1 += Math.abs(gpu.histogram[b]! - twin.histogram[b]!);
    report.stats.push({
      grid: names[k]!,
      fastPath: gpu.usedAtomicFastPath,
      fastPathTwin: twin.usedAtomicFastPath,
      count: gpu.activeVoxelCount,
      countTwin: twin.activeVoxelCount,
      minExact: Object.is(gpu.min, twin.min),
      maxExact: Object.is(gpu.max, twin.max),
      meanAbs: Math.abs(gpu.mean - twin.mean),
      histogramL1: l1,
      histogramSum: gpu.histogram.reduce((a, b) => a + b, 0),
    });
  }

  // decodeToAtlas: raw f32 (normalize: false) against the twin, then the formatted outputs.
  const variants: { label: string; filter: VDBAtlasFilter; format: VDBAtlasFormat; normalize: boolean; maxDim: number }[] = [
    { label: "nearest raw", filter: "nearest", format: "float32", normalize: false, maxDim: 256 },
    { label: "trilinear raw", filter: "trilinear", format: "float32", normalize: false, maxDim: 256 },
    { label: "trilinear raw maxDim 40", filter: "trilinear", format: "float32", normalize: false, maxDim: 40 },
    { label: "trilinear uint8 normalized maxDim 96", filter: "trilinear", format: "uint8", normalize: true, maxDim: 96 },
    { label: "nearest float32 normalized maxDim 64", filter: "nearest", format: "float32", normalize: true, maxDim: 64 },
  ];
  for (let k = 0; k < images.length; k++) {
    for (const [v, variant] of variants.entries()) {
      const { label, ...o } = variant;
      const options: VDBDecodeToAtlasOptions = { ...o };
      let job;
      if ((k + v) % 2 === 0) {
        const encoder = device.createCommandEncoder();
        job = vdbDecodeToAtlasGpu(device, pool, ids[k]!, { ...options, encoder });
        device.queue.submit([encoder.finish()]);
        job.submitted();
      } else {
        job = vdbDecodeToAtlasGpu(device, pool, ids[k]!, options);
      }
      const gpu = await job.read();
      const twin = vdbDecodeToAtlasCpu(images[k]!, { filter: o.filter, maxDim: o.maxDim });
      // Raw comparison: re-run unformatted when this variant formats.
      const raw =
        !o.normalize && o.format === "float32"
          ? (gpu.data as Float32Array)
          : ((await vdbDecodeToAtlasGpu(device, pool, ids[k]!, { filter: o.filter, maxDim: o.maxDim, format: "float32", normalize: false }).read()).data as Float32Array);
      const { exact, maxAbs } = compareRaw(raw, twin.raw);
      const twinData = atlasFromRaw(twin.raw, { format: o.format, normalize: o.normalize, range: gpu.range });
      let dataDiff = 0;
      let dataMaxAbs = 0;
      for (let i = 0; i < twinData.length; i++) {
        if (!Object.is(gpu.data[i], twinData[i])) {
          dataDiff++;
          dataMaxAbs = Math.max(dataMaxAbs, Math.abs(gpu.data[i]! - twinData[i]!));
        }
      }
      report.atlas.push({ grid: names[k]!, label, dims: gpu.dims, voxels: twin.raw.length, exact, maxAbs, dataDiff, dataMaxAbs });
    }
  }

  // readback: false keeps the GPU buffer; dispose() frees it; read() after dispose rejects.
  {
    const job = vdbDecodeToAtlasGpu(device, pool, ids[0]!, { readback: false, maxDim: 16 });
    report.readbackFalseOk = job.buffer.size === job.dims[0] * job.dims[1] * job.dims[2] * 4;
    job.dispose();
    report.disposeRejects = await job.read().then(
      () => false,
      () => true,
    );
  }

  // Host encoder: read() before the host's submit must reject and touch nothing
  // (mapping the staging buffer first would fail that submit's validation).
  {
    const encoder = device.createCommandEncoder();
    const job = vdbGridStatsGpu(device, pool, ids[0]!, { encoder });
    const early = await job.read().then(
      () => "resolved",
      (e: unknown) => (e as { code?: string }).code ?? String(e),
    );
    device.pushErrorScope("validation");
    device.queue.submit([encoder.finish()]);
    const submitError = await device.popErrorScope();
    const stats = await job.submitted().read();
    report.readBeforeSubmit = {
      earlyRead: early,
      submitValidation: submitError ? submitError.message : null,
      readAfterSubmittedOk: stats.activeVoxelCount === vdbGridStatsCpu(images[0]!).activeVoxelCount,
    };
  }

  // vdbApplyValueTransform on box Float: leaf values double, tiles stay, in place.
  {
    const k = 0;
    const source = images[k]!;
    const versionBefore = pool.version;
    const transformed = vdbApplyValueTransform(pool, ids[k]!, source, "double");
    pool.flush();
    const versionUnchanged = pool.version === versionBefore;
    let leafSamples = 0;
    let tileSamples = 0;
    let leafDoubled = true;
    let tilesUnchanged = true;
    const info = pool.info(ids[k]!);
    const [lo, hi] = [info.indexBBox.min, info.indexBBox.max];
    for (let n = 0; n < 4000; n++) {
      const ijk: [number, number, number] = [0, 1, 2].map((a) => lo[a]! + ((n * [7919, 104729, 1299709][a]!) % (hi[a]! - lo[a]! + 1))) as [number, number, number];
      const before = vdbReadValueF32(source, ijk);
      const after = vdbReadValueF32(transformed, ijk);
      if (getDim(source, ijk) === 1) {
        leafSamples++;
        if (!Object.is(after, Math.fround(before * 2))) leafDoubled = false;
      } else {
        tileSamples++;
        if (!Object.is(after, before)) tilesUnchanged = false;
      }
    }
    const gpuAtlas = await vdbDecodeToAtlasGpu(device, pool, ids[k]!, { filter: "nearest", format: "float32", normalize: false }).read();
    const twinAtlas = vdbDecodeToAtlasCpu(transformed, { filter: "nearest" });
    const gpuStats = await vdbGridStatsGpu(device, pool, ids[k]!).read();
    const twinStats = vdbGridStatsCpu(transformed);
    report.transform = {
      grid: names[k]!,
      leafSamples,
      tileSamples,
      leafDoubled,
      tilesUnchanged,
      atlasExact: compareRaw(gpuAtlas.data as Float32Array, twinAtlas.raw).exact,
      atlasVoxels: twinAtlas.raw.length,
      statsCountMatch: gpuStats.activeVoxelCount === twinStats.activeVoxelCount,
      statsMaxExact: Object.is(gpuStats.max, twinStats.max),
      versionUnchanged,
    };
  }

  await device.queue.onSubmittedWorkDone();
  if (pool.lastError) report.lastError = pool.lastError.message;
  pool.dispose();
  device.destroy();
  return { ...report, done: true };
}

root.textContent = "";
run()
  .catch((e: unknown): ComputeReport => ({ done: true, error: e instanceof Error ? `${e.name}: ${e.message}\n${e.stack}` : String(e), stats: [], atlas: [], uncaptured: [] }))
  .then((r) => {
    window.__VDB_COMPUTE__ = r;
    if (r.error) log(`error: ${r.error}`, "fail");
    else if (r.missing) log(`fixtures missing: ${r.missing.join(", ")} (pnpm fixtures:bake)`, "fail");
    else {
      log(`adapter: ${r.adapter}`);
      for (const s of r.stats) {
        log(
          `gridStats ${s.grid}: ${s.fastPath ? "atomics" : "dump"}, count ${s.count}/${s.countTwin}, min ${s.minExact ? "exact" : "DIFF"}, ` +
            `max ${s.maxExact ? "exact" : "DIFF"}, mean |Δ| ${s.meanAbs.toExponential(2)}, histogram L1 ${s.histogramL1}`,
        );
      }
      for (const a of r.atlas) {
        log(`decodeToAtlas ${a.grid} ${a.label} ${a.dims.join("x")}: raw ${a.exact}/${a.voxels} exact (max |Δ| ${a.maxAbs.toExponential(2)}), data diff ${a.dataDiff} (max ${a.dataMaxAbs})`);
      }
      log(`transform: ${JSON.stringify(r.transform)}`);
      log(`readback:false ${r.readbackFalseOk}, read after dispose rejects ${r.disposeRejects}, uncaptured ${r.uncaptured.length}`);
    }
  });
