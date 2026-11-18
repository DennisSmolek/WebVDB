/**
 * K3.2c timing harness: one froxel-renderer configuration per page load, timed per pass on the GPU
 * (three's render-pass timestamps mapped to pass names, `k3-pass-timer.ts`) and as a queue-synchronised
 * frame (render, then `queue.onSubmittedWorkDone()`: v1's hardware-gate fallback, docs/handoffs/PHASE-3.md).
 * Driven by `scripts/k3/time-hardware.mjs` on installed Chrome; SwiftShader works too (slowly).
 *
 * Query: `?asset=cloud04|bunny|smoke|wdas` (as k3-sparse: cloud04 and bunny are identical voxels on both
 * sources; smoke and wdas sparse at full resolution, dense a `?denseRes=` (128) bake), `?source=dense|sparse`,
 * `?lod=none|node|pyramid` (sparse), `?hdda=1|0`, `?lighting=volume|frame`, `?w=&h=` (1920x1080),
 * `?froxels=WxHxD` (his 512x512x24), `?substeps=` (32), `?shadowSteps=` (32), `?ms=0|1`, `?dist=` (camera
 * distance / box diagonal, 0.75), `?frames=` (30, measured) `?warmup=` (10), `?shadowRes=` (64), `?fast=0|1` and
 * `?cache=0|1` (K3.2d sparse level-0 sampling: `fastTrilinear`, `cacheAccessor`; default the source's defaults).
 *
 * Per measured frame: the froxel passes alone (`renderFroxels`; in 'frame' lighting the signature-cached
 * shadow density and transmittance are skipped in a still scene), the same with the grid passes forced
 * ('frame' only: `gridRedraw`), the scene with the held atlas (integrate + scene) and without the volume
 * (scene), then a whole frame timed on the queue. Medians. Results: `window.__VDB_K3_TIMING__`.
 */
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBFrameRenderer, VDBFrameVolume, VDBGridNode, vdbDenseTextures, vdbSparseSource, type VDBSparseLod, type VDBThreeVolumeSource } from "@pmndrs/vdb/three";
import { NanoVDBFile, buildFromVdb, nanoVdbToWvdb, parseVdb, readWvdb, vdbToWvdb, wvdbDenseToNanoVdb, wvdbDenseVolume, type WvdbDenseVolume } from "@pmndrs/vdb/tools";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";
import { installPassTimer } from "./k3-pass-timer.js";

type Vec3 = [number, number, number];

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  timestampQuery?: boolean;
  config?: Record<string, unknown>;
  /** Median GPU ms per froxel pass (still scene: cached 'frame' grid passes skipped). */
  passesMs?: Record<string, number>;
  /** 'frame' only: median GPU ms per pass with the grid passes forced. */
  gridRedrawPassesMs?: Record<string, number>;
  froxelsMs?: number | null;
  gridRedrawMs?: number | null;
  integrateMs?: number | null;
  sceneMs?: number | null;
  /** froxels + integrate + scene (GPU, still scene). */
  gpuFrameMs?: number | null;
  /** Median wall ms of a whole frame, render to queue done (CPU encode + GPU). */
  frameWallMs?: number;
  /** Mean wall ms of `frames` back-to-back frames (throughput). */
  frameThroughputMs?: number;
  gpuBytes?: { source: number; pyramid: number; pyramidScratch: number };
  lodLevels?: number;
  setupMs?: number;
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_TIMING__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const ASSET = params.get("asset") ?? "cloud04";
const SOURCE = (params.get("source") ?? "sparse") as "dense" | "sparse";
const LOD_OF: Record<string, VDBSparseLod> = { none: "none", node: "node-average", pyramid: "coarse-pyramid" };
const LOD = LOD_OF[params.get("lod") ?? "pyramid"];
const HDDA = params.get("hdda") !== "0";
const LIGHTING = (params.get("lighting") ?? "volume") as "volume" | "frame";
const W = Number(params.get("w") ?? 1920);
const H = Number(params.get("h") ?? 1080);
const FROXELS = (params.get("froxels") ?? "512x512x24").split("x").map(Number);
const SUBSTEPS = Number(params.get("substeps") ?? 32);
const SHADOW_STEPS = Number(params.get("shadowSteps") ?? 32);
const FRAMES = Number(params.get("frames") ?? 30);
const WARMUP = Number(params.get("warmup") ?? 10);
const DIST = Number(params.get("dist") ?? 0.75);
const DENSE_RES = Number(params.get("denseRes") ?? 128);
// K3.2d sparse level-0 sampling: `?fast=0|1` (vdbSampleFast; default: the source's default), `?cache=0|1` (one accessor per invocation).
const FAST = params.has("fast") ? params.get("fast") !== "0" : undefined;
const CACHE = params.has("cache") ? params.get("cache") !== "0" : undefined;
const BACKGROUND = new THREE.Color(0x598eff);

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K3_TIMING__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} (fixtures are git-ignored)`);
  return r.arrayBuffer();
}

interface Asset {
  image: Uint32Array | null;
  dense: WvdbDenseVolume | null;
  indexBox?: { min: Vec3; max: Vec3 };
  bbox?: [Vec3, Vec3];
}

async function loadAsset(): Promise<Asset> {
  const sparse = SOURCE === "sparse";
  if (ASSET === "cloud04" || ASSET === "bunny") {
    const url = ASSET === "cloud04" ? cloudWvdbUrl : "/fixtures/k3/bunny_cloud.wvdb";
    const dense = wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(url))), { mipmaps: true });
    const grid = sparse ? wvdbDenseToNanoVdb(dense) : null;
    return { image: grid?.image ?? null, dense, ...(grid ? { indexBox: grid.textureIndexBox } : {}), ...(dense.bbox ? { bbox: dense.bbox as [Vec3, Vec3] } : {}) };
  }
  if (ASSET === "smoke") {
    const file = parseVdb(await fetchBytes("/fixtures/vdb-samples/smoke.vdb"));
    if (sparse) return { image: buildFromVdb(file.grids.find((g) => g.valueType === "float")!), dense: null };
    const dense = wvdbDenseVolume(readWvdb(vdbToWvdb(file, { resolution: DENSE_RES }).bytes), { mipmaps: true });
    return { image: null, dense, ...(dense.bbox ? { bbox: dense.bbox as [Vec3, Vec3] } : {}) };
  }
  if (ASSET === "wdas") {
    const image = NanoVDBFile.fromArrayBuffer(await fetchBytes("/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb")).gridImage(0);
    if (sparse) return { image, dense: null };
    const dense = wvdbDenseVolume(readWvdb(nanoVdbToWvdb(image, { resolution: DENSE_RES }).bytes), { mipmaps: true });
    return { image: null, dense, ...(dense.bbox ? { bbox: dense.bbox as [Vec3, Vec3] } : {}) };
  }
  throw new Error(`unknown ?asset=${ASSET}`);
}

const median = (v: (number | null | undefined)[]): number | null => {
  const s = v.filter((x): x is number => typeof x === "number" && Number.isFinite(x)).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)]! : null;
};
const sum = (m: Record<string, number>) => Object.values(m).reduce((s, v) => s + v, 0);

async function main(): Promise<void> {
  root.textContent = "";
  const t0 = performance.now();
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(" ");
  const timestamps = adapter.features.has("timestamp-query");
  result.timestampQuery = timestamps;
  const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ["timestamp-query"] : [], requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const renderer = new WebGPURenderer({ device, trackTimestamp: timestamps });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);

  const asset = await loadAsset();
  const frameRenderer = new VDBFrameRenderer();
  Object.assign(frameRenderer.settings, {
    width: FROXELS[0],
    height: FROXELS[1],
    depth: FROXELS[2],
    subSteps: SUBSTEPS,
    jitter: false,
    temporal: false,
    lighting: LIGHTING,
    multipleScattering: params.get("ms") !== "0",
    shadowResolution: Number(params.get("shadowRes") ?? 64),
  });

  let source: VDBThreeVolumeSource;
  let gpuBytes = { source: 0, pyramid: 0, pyramidScratch: 0 };
  if (SOURCE === "sparse") {
    if (!asset.image) throw new Error("no sparse image");
    const pool = await VDBGridPool.create(device, { label: "k3-timing" });
    const id = pool.add(asset.image);
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
    const s = vdbSparseSource(grids, { density: id }, {
      lod: LOD!,
      ...(FAST !== undefined ? { fastTrilinear: FAST } : {}),
      ...(CACHE !== undefined ? { cacheAccessor: CACHE } : {}),
      images: { density: asset.image },
      ...(asset.dense ? { stats: { meanNonZero: asset.dense.meanDensity, max: asset.dense.maxDensity, activeVoxels: asset.dense.activeTexels } } : {}),
      ...(asset.indexBox ? { indexBox: asset.indexBox } : {}),
      ...(asset.bbox ? { bbox: asset.bbox } : {}),
    });
    source = s;
    gpuBytes = { source: s.gpuBytes.pool, pyramid: s.gpuBytes.pyramid, pyramidScratch: s.gpuBytes.pyramidScratch };
  } else {
    if (!asset.dense) throw new Error("no dense volume");
    source = vdbDenseTextures(asset.dense);
    const t = asset.dense.density;
    gpuBytes = { source: t.data.byteLength + (t.mip?.data.byteLength ?? 0), pyramid: 0, pyramidScratch: 0 };
  }
  result.gpuBytes = gpuBytes;
  result.lodLevels = source.lodLevels;
  result.config = { asset: ASSET, source: SOURCE, lod: SOURCE === "sparse" ? LOD : "dense mips", hdda: HDDA, ...(SOURCE === "sparse" ? { fastTrilinear: (source as { fastTrilinear?: boolean }).fastTrilinear, cacheAccessor: (source as { cacheAccessor?: boolean }).cacheAccessor } : {}), lighting: LIGHTING, width: W, height: H, froxels: FROXELS, subSteps: SUBSTEPS, shadowSteps: SHADOW_STEPS, frames: FRAMES, warmup: WARMUP, dist: DIST, settings: { ...frameRenderer.settings } };

  const b = asset.bbox ?? (source as { bbox?: [Vec3, Vec3] }).bbox;
  if (!b) throw new Error("no volume box");
  const bbox = new THREE.Box3(new THREE.Vector3(...b[0]), new THREE.Vector3(...b[1]));
  const center = bbox.getCenter(new THREE.Vector3());
  const diag = bbox.getSize(new THREE.Vector3()).length();
  const camera = new THREE.PerspectiveCamera(60, W / H, diag * 0.01, diag * 40);
  camera.position.copy(center).add(new THREE.Vector3(0.9, 0.35, 1.1).normalize().multiplyScalar(diag * DIST));
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);

  const volume = new VDBFrameVolume(source, {
    frameRenderer,
    bbox: bbox.clone(),
    baseColor: "#ffffff",
    absorbance: 1.0,
    densityScale: 1.0,
    normalizeDensity: true,
    opacity: 1.0,
    extinction: Number(params.get("extinction") ?? 40),
    phase: "dual-lobe",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    shadowSteps: SHADOW_STEPS,
    mipmaps: true,
    hdda: HDDA,
  });
  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(volume);
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(100.0, 200.0, 100.0).add(center);
  sun.target.position.copy(center);
  scene.add(sun, sun.target);
  scene.updateMatrixWorld(true);
  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });

  const timer = timestamps ? installPassTimer(renderer, frameRenderer) : null;
  const frame = () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
  };
  // First frames: shader builds, the pyramid, the 'frame' grid passes.
  for (let i = 0; i < 2; i++) {
    frame();
    await device.queue.onSubmittedWorkDone();
  }
  await timer?.collect();
  result.setupMs = Math.round(performance.now() - t0);

  const passes: Record<string, number>[] = [];
  const redraw: Record<string, number>[] = [];
  const integrate: (number | null)[] = [];
  const sceneOnly: (number | null)[] = [];
  const wall: number[] = [];
  for (let f = 0; f < WARMUP + FRAMES; f++) {
    const keep = f >= WARMUP;
    if (timer) {
      frameRenderer.renderFroxels(renderer, scene, camera);
      const p = await timer.collect();
      if (LIGHTING === "frame") {
        frameRenderer.shadowDrawnVersion = -1;
        frameRenderer.transmittanceSignature = [];
        frameRenderer.environmentGridSignature = [];
        frameRenderer.renderFroxels(renderer, scene, camera);
        const r = await timer.collect();
        if (keep) redraw.push(r);
      }
      frameRenderer.holdFroxels = true;
      frame();
      frameRenderer.holdFroxels = false;
      const held = sum(await timer.collect());
      volume.visible = false;
      frame();
      volume.visible = true;
      const bare = sum(await timer.collect());
      if (keep) {
        passes.push(p);
        integrate.push(held - bare);
        sceneOnly.push(bare);
      }
    }
    const w0 = performance.now();
    frame();
    await device.queue.onSubmittedWorkDone();
    if (keep) wall.push(performance.now() - w0);
    await timer?.collect();
  }
  // Throughput: frames back to back, one wait.
  const tp0 = performance.now();
  for (let f = 0; f < FRAMES; f++) frame();
  await device.queue.onSubmittedWorkDone();
  result.frameThroughputMs = (performance.now() - tp0) / FRAMES;
  await timer?.collect();

  const medians = (rows: Record<string, number>[]) => {
    const names = new Set(rows.flatMap((r) => Object.keys(r)));
    const out: Record<string, number> = {};
    for (const n of names) out[n] = median(rows.map((r) => r[n] ?? 0))!;
    return out;
  };
  if (timer) {
    result.passesMs = medians(passes);
    result.froxelsMs = median(passes.map(sum));
    if (LIGHTING === "frame") {
      result.gridRedrawPassesMs = medians(redraw);
      result.gridRedrawMs = median(redraw.map(sum));
    }
    result.integrateMs = median(integrate);
    result.sceneMs = median(sceneOnly);
    result.gpuFrameMs = result.froxelsMs !== null && result.integrateMs !== null && result.sceneMs !== null ? result.froxelsMs + result.integrateMs + result.sceneMs : null;
  }
  result.frameWallMs = median(wall)!;
  log(`${result.adapter}: ${JSON.stringify({ ...result, config: undefined, warnings: undefined })}`, "ok");
  timer?.dispose();
  for (const w of result.warnings) log(`warning: ${w}`, "warn");
  volume.dispose();
  source.dispose();
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
