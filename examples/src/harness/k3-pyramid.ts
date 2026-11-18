/**
 * K3.2a harness: the sparse source's coarse pyramid built on the GPU (core `createBoxPyramid`)
 * against its CPU twin (tools `vdbBoxPyramid` + three's `DataUtils.toHalfFloat`, the K3.2 path),
 * texel for texel on the r16f textures, plus build times and bytes.
 *
 * Query:
 * - `?assets=` comma list (default `cloud04,cloud04@32,sphere,sphere8`): `cloud04` (the committed CC0
 *   `.wvdb` as a NanoVDB FLOAT grid), `bunny` (his `.wvdb`, `/fixtures/k3/`), `smoke`
 *   (`/fixtures/vdb-samples/smoke.vdb`, full resolution: fractional box weights), `sphere` / `sphere8`
 *   / `spherefpn` (the Float / Fp8 / FpN primitives: inactive tiles and quantized leaves), `spherels`
 *   (`/fixtures/vdb-samples/sphere.vdb`, a level set: inactive tiles at ±3, so blocks with no active
 *   voxel must read 0 as on the CPU), `wdas` (the
 *   Fp8 quarter cloud). `@N` sets `pyramidMax` (cloud04@32: L₀ = 2, a fractional z axis).
 * - `?repeat=N` (default 3): timed rebuilds per asset (median; GPU timestamps when the adapter has
 *   `timestamp-query`, else queue-synchronised wall time) and CPU builds (median).
 * - `?sequence=1`: replace the grid with another image (pool.replace), check `stale`, rebuild, compare
 *   with the CPU pyramid of the new image over the same index box.
 * - `?three=1`: `vdbSparseSource` with `pyramidBuild: 'gpu'` vs `'cpu'` on cloud04 (the three textures
 *   behind both), then a `pool.replace` picked up by a render (the source's per-render refresh).
 * Results: `window.__VDB_K3_PYRAMID__` (e2e/k3-pyramid.spec.ts, scripts/k3/time-hardware.mjs).
 */
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { VDBGridPool, createBoxPyramid, type VDBBoxPyramidGpu } from "@pmndrs/vdb";
import { VDBFrameRenderer, VDBFrameVolume, VDBGridNode, getGPUTexture, vdbSparseSource } from "@pmndrs/vdb/three";
import { NanoVDBFile, buildFromVdb, parseVdb, readWvdb, vdbBoxPyramid, wvdbDenseToNanoVdb, wvdbDenseVolume } from "@pmndrs/vdb/tools";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

type Vec3 = [number, number, number];

interface Compare {
  texels: number;
  exact: number;
  ulp1: number;
  over: number;
  maxUlp: number;
  firstBad?: { index: number; gpu: number; cpu: number };
}

interface AssetResult {
  name: string;
  type: string;
  indexBox: { min: number[]; max: number[] };
  texels: number[];
  level: number;
  dimensions: number[];
  atlasDimensions: number[] | null;
  atlasLevels: number;
  base: Compare;
  atlas: Compare | null;
  gpuBytes: { textures: number; scratch: number };
  cpuBytesUploaded: number;
  /** Median GPU build ms (timestamps around the compute pass; copies excluded) or null. */
  gpuMs: number | null;
  /** Median wall ms of record + submit + onSubmittedWorkDone (queue-synchronised). */
  gpuWallMs: number;
  /** Median CPU ms: vdbBoxPyramid + the f16 conversion (the K3.2 path, upload excluded). */
  cpuMs: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  timestampQuery?: boolean;
  assets: AssetResult[];
  sequence?: { staleBefore: boolean; rebuilt: boolean; staleAfter: boolean; again: boolean; base: Compare; atlas: Compare | null };
  three?: {
    gpuVsCpu: { base: Compare; atlas: Compare | null };
    buildsAfterCreate: number;
    buildsAfterRender: number;
    buildsAfterReplace: number;
    buildsAfterSecondRender: number;
    replaced: { base: Compare; atlas: Compare | null };
    /** The source's content version before and after the replace (the 'frame' caster signature reads it). */
    versions: [number, number];
    gpuBytes: { pool: number; pyramid: number; pyramidScratch: number };
  };
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_PYRAMID__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const ASSETS = (params.get("assets") ?? "cloud04,cloud04@32,sphere,sphere8").split(",").filter(Boolean);
const REPEAT = Math.max(1, Number(params.get("repeat") ?? 3));
const SEQUENCE = params.get("sequence") === "1";
const THREE_CHECK = params.get("three") === "1";

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, assets: [], warnings: [], uncaptured: [] };
window.__VDB_K3_PYRAMID__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} (fixtures are git-ignored: pnpm fixtures)`);
  return r.arrayBuffer();
}

interface Asset {
  image: Uint32Array;
  indexBox?: { min: Vec3; max: Vec3 };
}

async function loadAsset(name: string): Promise<Asset> {
  if (name === "cloud04" || name === "bunny") {
    const url = name === "cloud04" ? cloudWvdbUrl : "/fixtures/k3/bunny_cloud.wvdb";
    const grid = wvdbDenseToNanoVdb(wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(url))), { mipmaps: false }));
    return { image: grid.image, indexBox: grid.textureIndexBox };
  }
  if (name === "smoke" || name === "spherels") {
    // spherels: the openvdb.org sphere level set (background 3, inactive interior tiles at -3): the active-block rule.
    const file = parseVdb(await fetchBytes(`/fixtures/vdb-samples/${name === "smoke" ? "smoke" : "sphere"}.vdb`));
    return { image: buildFromVdb(file.grids.find((g) => g.valueType === "float")!) };
  }
  const primitive: Record<string, string> = { sphere: "sphere_fog_float", sphere8: "sphere_fog_fp8", spherefpn: "sphere_fog_fpn", torus: "torus_fog_float" };
  if (primitive[name]) return { image: NanoVDBFile.fromArrayBuffer(await fetchBytes(`/fixtures/primitives/${primitive[name]}.nvdb`)).gridImage(0) };
  if (name === "wdas") return { image: NanoVDBFile.fromArrayBuffer(await fetchBytes("/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb")).gridImage(0) };
  throw new Error(`unknown asset ${name}`);
}

const toHalf = (data: Float32Array) => {
  const out = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = THREE.DataUtils.toHalfFloat(data[i]!);
  return out;
};

/** f16 bit patterns → ordered integers (ulp distance = difference). */
const ordered = (h: number) => (h & 0x8000 ? -(h & 0x7fff) : h);

function compare(gpu: Uint16Array, cpu: Uint16Array): Compare {
  if (gpu.length !== cpu.length) throw new Error(`compare: ${gpu.length} GPU texels vs ${cpu.length} CPU`);
  const c: Compare = { texels: cpu.length, exact: 0, ulp1: 0, over: 0, maxUlp: 0 };
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(ordered(gpu[i]!) - ordered(cpu[i]!));
    if (d === 0) c.exact++;
    else if (d === 1) c.ulp1++;
    else {
      c.over++;
      c.firstBad ??= { index: i, gpu: THREE.DataUtils.fromHalfFloat(gpu[i]!), cpu: THREE.DataUtils.fromHalfFloat(cpu[i]!) };
    }
    c.maxUlp = Math.max(c.maxUlp, d);
  }
  return c;
}

/** Reads an r16float 3D texture back (tight rows). */
async function readHalfTexture(device: GPUDevice, texture: GPUTexture): Promise<Uint16Array> {
  const [w, h, d] = [texture.width, texture.height, texture.depthOrArrayLayers];
  const bytesPerRow = Math.ceil((w * 2) / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Uint16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  const out = new Uint16Array(w * h * d);
  const stride = bytesPerRow / 2;
  for (let r = 0; r < h * d; r++) out.set(raw.subarray(r * stride, r * stride + w), r * w);
  return out;
}

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)]! : NaN;
};

/** CPU twin: the K3.2 path (vdbBoxPyramid, then three's f16 conversion). */
function cpuPyramid(image: Uint32Array, indexBox: { min: Vec3; max: Vec3 }, pyramidMax: number | undefined) {
  const t0 = performance.now();
  const p = vdbBoxPyramid(image, { indexBox, ...(pyramidMax !== undefined ? { pyramidMax } : {}) });
  const base = toHalf(p.data);
  const atlas = p.atlas ? toHalf(p.atlas.data) : null;
  return { p, base, atlas, ms: performance.now() - t0 };
}

async function checkAgainstCpu(device: GPUDevice, pyramid: VDBBoxPyramidGpu, cpu: { base: Uint16Array; atlas: Uint16Array | null }) {
  const base = compare(await readHalfTexture(device, pyramid.base), cpu.base);
  const atlas = pyramid.atlas && cpu.atlas ? compare(await readHalfTexture(device, pyramid.atlas), cpu.atlas) : null;
  if (!!pyramid.atlas !== !!cpu.atlas) throw new Error("atlas presence differs between GPU and CPU");
  return { base, atlas };
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
  const timestamps = adapter.features.has("timestamp-query");
  result.timestampQuery = timestamps;
  const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ["timestamp-query"] : [], requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  const querySet = timestamps ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const resolveBuffer = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const readBuffer = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;

  for (const spec of ASSETS) {
    const [name, maxText] = spec.split("@");
    const pyramidMax = maxText ? Number(maxText) : undefined;
    const asset = await loadAsset(name!);
    const pool = await VDBGridPool.create(device, { label: `k3-pyramid ${spec}` });
    const id = pool.add(asset.image);
    pool.flush();
    await pool.settled();
    const gridInfo = pool.info(id);
    const indexBox = asset.indexBox ?? { min: [...gridInfo.indexBBox.min] as Vec3, max: [...gridInfo.indexBBox.max] as Vec3 };

    const cpuRuns: number[] = [];
    let cpu = cpuPyramid(asset.image, indexBox, pyramidMax);
    cpuRuns.push(cpu.ms);
    for (let r = 1; r < REPEAT; r++) {
      cpu = cpuPyramid(asset.image, indexBox, pyramidMax);
      cpuRuns.push(cpu.ms);
    }

    const pyramid = createBoxPyramid(device, pool, id, { indexBox, ...(pyramidMax !== undefined ? { pyramidMax } : {}) });
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder();
    pyramid.record(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const scopeError = await device.popErrorScope();
    const pipelineError = await pyramid.error;
    if (scopeError || pipelineError) throw new Error(`createBoxPyramid ${spec}: ${(scopeError ?? pipelineError)!.message}`);
    const { base, atlas } = await checkAgainstCpu(device, pyramid, cpu);

    // Timed rebuilds (the first build above was the warm-up: pipeline creation).
    const wall: number[] = [];
    const gpu: number[] = [];
    for (let r = 0; r < REPEAT; r++) {
      const e = device.createCommandEncoder();
      const t0 = performance.now();
      pyramid.record(e, querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
      if (querySet) {
        e.resolveQuerySet(querySet, 0, 2, resolveBuffer!, 0);
        e.copyBufferToBuffer(resolveBuffer!, 0, readBuffer!, 0, 16);
      }
      device.queue.submit([e.finish()]);
      await device.queue.onSubmittedWorkDone();
      wall.push(performance.now() - t0);
      if (readBuffer) {
        await readBuffer.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(readBuffer.getMappedRange());
        const ns = Number(t[1]! - t[0]!);
        readBuffer.unmap();
        if (ns > 0) gpu.push(ns / 1e6);
      }
    }
    const { layout } = pyramid;
    const r: AssetResult = {
      name: spec,
      type: gridInfo.type,
      indexBox,
      texels: [...layout.texels],
      level: layout.level,
      dimensions: [...layout.dimensions],
      atlasDimensions: layout.atlas ? [...layout.atlas.dimensions] : null,
      atlasLevels: layout.atlas?.levels.length ?? 0,
      base,
      atlas,
      gpuBytes: { ...pyramid.gpuBytes },
      cpuBytesUploaded: 2 * (cpu.base.length + (cpu.atlas?.length ?? 0)),
      gpuMs: gpu.length ? median(gpu) : null,
      gpuWallMs: median(wall),
      cpuMs: median(cpuRuns),
    };
    result.assets.push(r);
    const ok = base.over === 0 && (atlas?.over ?? 0) === 0;
    log(
      `${spec} (${r.type}): L₀ ${r.level}, base ${r.dimensions.join("x")}, atlas ${r.atlasDimensions?.join("x") ?? "—"}; ` +
        `base ${base.exact}/${base.texels} exact, ${base.ulp1} at 1 ulp, ${base.over} over; ` +
        (atlas ? `atlas ${atlas.exact}/${atlas.texels} exact, ${atlas.ulp1} at 1 ulp, ${atlas.over} over; ` : "") +
        `GPU ${r.gpuMs?.toFixed(3) ?? "—"} ms (wall ${r.gpuWallMs.toFixed(2)} ms), CPU ${r.cpuMs.toFixed(1)} ms; bytes ${JSON.stringify(r.gpuBytes)}`,
      ok ? "ok" : "fail",
    );

    if (SEQUENCE && spec === ASSETS[0]) {
      // A sequence frame: another image in the same grid id; the pyramid keeps its index box.
      const next = await loadAsset(name === "sphere" ? "torus" : "sphere");
      const staleBefore = pyramid.stale;
      pool.replace(id, next.image);
      const e = device.createCommandEncoder();
      pool.update(e);
      const rebuilt = pyramid.update(e);
      device.queue.submit([e.finish()]);
      await device.queue.onSubmittedWorkDone();
      const staleAfter = pyramid.stale;
      const e2 = device.createCommandEncoder();
      const again = pyramid.update(e2);
      device.queue.submit([e2.finish()]);
      const cpuNext = cpuPyramid(next.image, indexBox, pyramidMax);
      result.sequence = { staleBefore, rebuilt, staleAfter, again, ...(await checkAgainstCpu(device, pyramid, cpuNext)) };
      log(`sequence: ${JSON.stringify(result.sequence)}`, rebuilt && !again && result.sequence.base.over === 0 ? "ok" : "fail");
    }
    pyramid.destroy();
    pool.dispose();
  }

  if (THREE_CHECK) await threeCheck(device);
  for (const w of result.warnings) log(`warning: ${w}`, "warn");
}

/** vdbSparseSource with the GPU pyramid vs the CPU one; then a replace picked up by a render. */
async function threeCheck(device: GPUDevice): Promise<void> {
  const cloud = await loadAsset("cloud04");
  const renderer = new WebGPURenderer({ device });
  await renderer.init();
  renderer.setSize(160, 90, false);
  const pool = await VDBGridPool.create(device, { label: "k3-pyramid three" });
  const id = pool.add(cloud.image);
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd"] });
  const common = { images: { density: cloud.image }, indexBox: cloud.indexBox!, pyramidMax: 32 };
  const gpuSource = vdbSparseSource(grids, { density: id }, { ...common, pyramidBuild: "gpu" });
  const cpuSource = vdbSparseSource(grids, { density: id }, { ...common, pyramidBuild: "cpu" });
  await device.queue.onSubmittedWorkDone();
  const g = gpuSource.pyramid!;
  const c = cpuSource.pyramid!;
  const readThree = (t: THREE.Data3DTexture) => readHalfTexture(device, getGPUTexture(renderer, t));
  const gpuVsCpu = {
    base: compare(await readThree(g.map), c.map.image.data as Uint16Array),
    atlas: g.mip && c.mip ? compare(await readThree(g.mip), c.mip.image.data as Uint16Array) : null,
  };
  const buildsAfterCreate = gpuSource.pyramidBuilds;

  // A render that samples the pyramid: no rebuild (nothing changed).
  const frameRenderer = new VDBFrameRenderer();
  Object.assign(frameRenderer.settings, { width: 32, height: 32, depth: 8, subSteps: 8, jitter: false, temporal: false, lighting: "volume" });
  const volume = new VDBFrameVolume(gpuSource, { frameRenderer, shadowSteps: 8, extinction: 40 });
  const scene = new THREE.Scene();
  scene.add(volume);
  const sun = new THREE.DirectionalLight(0xffffff, 1);
  sun.position.set(1, 2, 1);
  scene.add(sun);
  const box = new THREE.Box3(new THREE.Vector3(...gpuSource.bbox[0]), new THREE.Vector3(...gpuSource.bbox[1]));
  const center = box.getCenter(new THREE.Vector3());
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.01, 1e5);
  camera.position.copy(center).add(new THREE.Vector3(1, 0.4, 1.2).multiplyScalar(box.getSize(new THREE.Vector3()).length() * 0.8));
  camera.lookAt(center);
  const rt = new THREE.RenderTarget(160, 90, { type: THREE.HalfFloatType });
  const render = async () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    await device.queue.onSubmittedWorkDone();
  };
  await render();
  const buildsAfterRender = gpuSource.pyramidBuilds;

  // A sequence frame: replace the grid (same box: the cloud with its density halved), update the pool, render.
  const half = wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(cloudWvdbUrl))), { mipmaps: false });
  const halfData = half.density.data as Uint8Array;
  for (let i = 0; i < halfData.length; i++) halfData[i] = halfData[i]! >> 1;
  const next = wvdbDenseToNanoVdb(half).image;
  const versionBefore = gpuSource.version;
  pool.replace(id, next);
  pool.flush();
  grids.update();
  const buildsAfterReplace = gpuSource.pyramidBuilds;
  await render();
  const buildsAfterSecondRender = gpuSource.pyramidBuilds;
  const cpuNext = cpuPyramid(next, cloud.indexBox!, 32);
  const replaced = {
    base: compare(await readThree(g.map), cpuNext.base),
    atlas: g.mip && cpuNext.atlas ? compare(await readThree(g.mip), cpuNext.atlas) : null,
  };
  result.three = { gpuVsCpu, buildsAfterCreate, buildsAfterRender, buildsAfterReplace, buildsAfterSecondRender, replaced, versions: [versionBefore, gpuSource.version], gpuBytes: { ...gpuSource.gpuBytes } };
  log(`three: ${JSON.stringify(result.three)}`, gpuVsCpu.base.over === 0 && replaced.base.over === 0 && buildsAfterSecondRender === 2 ? "ok" : "fail");
  volume.dispose();
  gpuSource.dispose();
  cpuSource.dispose();
  rt.dispose();
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
