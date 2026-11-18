/**
 * K3.2 harness (R1 §6 Step 2): the froxel renderer ('volume' lighting) on the sparse VDBGridPool
 * source next to the dense `.wvdb` source, with the LOD strategies and HDDA skipping as variants.
 *
 * Assets (`?asset=`):
 * - `cloud04` (default): the committed CC0 `cloud_04_variant_0000.wvdb`; dense = his 8-bit texels,
 *   sparse = the same texels converted to a NanoVDB FLOAT grid (`wvdbDenseToNanoVdb`, voxel-exact),
 *   so dense vs sparse `none` differ only in the sampler (the identical-voxel A/B).
 * - `bunny`: his `bunny_cloud.wvdb` from `/fixtures/k3/` (not committed), same conversion.
 * - `smoke`: `/fixtures/vdb-samples/smoke.vdb` full resolution on sparse; dense = the 128³ `.wvdb`
 *   bake (`vdbToWvdb`, his point sampling). Different data: report only.
 * - `wdas`: `/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb` (sparse only, unless `dense` is listed:
 *   then a 128³ bake of it).
 *
 * Variants (`?variants=`, comma separated): `dense`, `none`, `node`, `pyramid` (sparse LOD
 * strategies), each sparse one optionally with `-nohdda` (no empty-space skipping in the shadow
 * march). Each renders the same camera into a linear HalfFloat target → sRGB → RGBA8.
 *
 * Other query: `?w=&h=`, `?froxels=WxHxD`, `?substeps=`, `?shadowSteps=`, `?dist=` (camera distance
 * multiplier), `?extinction=`, `?orbit=N&orbitDeg=d` (N more frames, the camera turning d degrees
 * about the volume each: flicker metrics), `?probe=1` (GPU vs CPU twins of `vdbNodeAverage`, the
 * node-average tap and the HDDA query), `?timing=1` (froxel-pass GPU ms per variant; provisional, D25),
 * `?stats=dense|grid` (identical-voxel assets: normalize by the dense mean, default, or the grid's own),
 * `?clamp01=1`, `?pyramidMax=`.
 * Results: `window.__VDB_K3_SPARSE__` (e2e/k3-sparse.spec.ts).
 */
import * as THREE from "three";
import { NodeMaterial, WebGPURenderer } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBFrameRenderer, VDBFrameVolume, VDBGridNode, vdbDenseTextures, vdbSparseSource, type VDBThreeVolumeSource, type VDBSparseLod } from "@pmndrs/vdb/three";
import {
  NanoVDBFile,
  buildFromVdb,
  nanoVdbToWvdb,
  nodeAverage,
  parseVdb,
  readWvdb,
  vdbFrameNodeAverageSample,
  vdbFrameSkipEmpty,
  vdbSrgbOetf,
  vdbToWvdb,
  wvdbDenseToNanoVdb,
  wvdbDenseVolume,
  type WvdbDenseVolume,
} from "@pmndrs/vdb/tools";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;
const { Fn, screenCoordinate, vec4, uniform, uint, uvec2, wgslFn } = TSL as unknown as Record<string, N>;

interface VariantResult {
  name: string;
  rgba: string;
  nonBackground: number;
  inPageIdentical: boolean;
  gpuBytes: { source: number; pyramid: number };
  lodLevels: number;
  flicker?: { frames: number; meanDiff: number; temporalNoise: number; meanDiffs: number[] };
  froxelsMs?: number | null;
  /** Wall-clock ms of the variant (shader builds + two frames + orbit); not a GPU timing. */
  wallMs?: number;
}

interface Probe {
  count: number;
  over: number;
  maxAbs: number;
  skipMismatch: number;
  firstBad?: unknown;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  asset?: string;
  width?: number;
  height?: number;
  froxels?: number[];
  grid?: { indexBox: { min: number[]; max: number[] }; texels: number[]; meanNonZero: number; gridMean?: number; denseMean?: number; poolBytes: number; imageBytes: number; type: string };
  dense?: { dimensions: number[]; meanDensity: number; textureBytes: number };
  variants: VariantResult[];
  probe?: Probe;
  timingsNote?: string;
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_SPARSE__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const ASSET = params.get("asset") ?? "cloud04";
const VARIANTS = (params.get("variants") ?? "dense,none").split(",").filter(Boolean);
const W = Number(params.get("w") ?? 320);
const H = Number(params.get("h") ?? 180);
const FROXELS = (params.get("froxels") ?? "128x128x24").split("x").map(Number);
const SUBSTEPS = Number(params.get("substeps") ?? 32);
const SHADOW_STEPS = Number(params.get("shadowSteps") ?? 32);
const DIST = Number(params.get("dist") ?? 1);
const ORBIT = Number(params.get("orbit") ?? 0);
const ORBIT_DEG = Number(params.get("orbitDeg") ?? 0.5);
const PROBE = params.get("probe") === "1";
const TIMING = params.get("timing") === "1";
const STATS = params.get("stats") ?? "dense";
const CLAMP01 = params.get("clamp01") === "1";
const PYRAMID_MAX = params.has("pyramidMax") ? Number(params.get("pyramidMax")) : undefined;
const DENSE_RES = Number(params.get("denseRes") ?? 128);
const BACKGROUND = new THREE.Color(0x598eff);

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, variants: [], warnings: [], uncaptured: [] };
window.__VDB_K3_SPARSE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

function toBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** three r187 WebGPU readback: rows top first, padded to 256 bytes (CLAUDE.md trap); strip the padding. */
function tightRows<T extends Uint16Array | Float32Array | Uint8Array>(pixels: T, w: number, h: number, channels: number): T {
  const tight = w * channels;
  const stride = h > 1 ? (pixels.length - tight) / (h - 1) : tight;
  if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${pixels.length} elements for ${w}x${h}`);
  const out = new (pixels.constructor as new (n: number) => T)(w * h * channels);
  for (let y = 0; y < h; y++) out.set(pixels.subarray(y * stride, y * stride + tight) as never, y * tight);
  return out;
}

const toByte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);

async function readDisplay(renderer: WebGPURenderer, rt: THREE.RenderTarget): Promise<Uint8Array> {
  const raw = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, W, H)) as unknown as Uint16Array;
  const rows = tightRows(raw, W, H, 4);
  const out = new Uint8Array(rows.length);
  for (let i = 0; i < rows.length; i += 4) {
    for (let c = 0; c < 3; c++) out[i + c] = toByte(vdbSrgbOetf(THREE.DataUtils.fromHalfFloat(rows[i + c]!)));
    out[i + 3] = 255;
  }
  return out;
}

/** Mean |a − b| over RGB (8-bit units). */
function meanAbs(a: Uint8Array, b: Uint8Array): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) s += Math.abs(a[i + c]! - b[i + c]!);
    n += 3;
  }
  return s / n;
}

/** Mean |b − (a + c) / 2| over RGB: the frame's departure from its neighbours' average (smooth motion cancels; aliasing does not). */
function temporalNoise(a: Uint8Array, b: Uint8Array, c: Uint8Array): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let k = 0; k < 3; k++) s += Math.abs(b[i + k]! - (a[i + k]! + c[i + k]!) / 2);
    n += 3;
  }
  return s / n;
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} (fixtures are git-ignored: pnpm fixtures, or copy his .wvdb into fixtures/k3/)`);
  return r.arrayBuffer();
}

interface Asset {
  image: Uint32Array;
  dense: WvdbDenseVolume | null;
  indexBox?: { min: [number, number, number]; max: [number, number, number] };
  bbox?: [[number, number, number], [number, number, number]];
  identical: boolean;
}

async function loadAsset(needDense: boolean): Promise<Asset> {
  if (ASSET === "cloud04" || ASSET === "bunny") {
    const url = ASSET === "cloud04" ? cloudWvdbUrl : "/fixtures/k3/bunny_cloud.wvdb";
    const dense = wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(url))), { mipmaps: true });
    const grid = wvdbDenseToNanoVdb(dense);
    return { image: grid.image, dense, indexBox: grid.textureIndexBox, ...(dense.bbox ? { bbox: dense.bbox as [[number, number, number], [number, number, number]] } : {}), identical: true };
  }
  if (ASSET === "smoke") {
    const buffer = await fetchBytes("/fixtures/vdb-samples/smoke.vdb");
    const file = parseVdb(buffer);
    const grid = file.grids.find((g) => g.valueType === "float")!;
    const image = buildFromVdb(grid);
    const dense = needDense ? wvdbDenseVolume(readWvdb(vdbToWvdb(file, { resolution: DENSE_RES }).bytes), { mipmaps: true }) : null;
    return { image, dense, identical: false };
  }
  if (ASSET === "wdas") {
    const image = NanoVDBFile.fromArrayBuffer(await fetchBytes("/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb")).gridImage(0);
    const dense = needDense ? wvdbDenseVolume(readWvdb(nanoVdbToWvdb(image, { resolution: DENSE_RES }).bytes), { mipmaps: true }) : null;
    return { image, dense, identical: false };
  }
  throw new Error(`unknown ?asset=${ASSET}; use cloud04, bunny, smoke, wdas`);
}

const LOD_OF: Record<string, VDBSparseLod> = { none: "none", node: "node-average", pyramid: "coarse-pyramid" };

async function main(): Promise<void> {
  root.textContent = "";
  result.asset = ASSET;
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };

  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const features: GPUFeatureName[] = TIMING && adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const needDense = VARIANTS.includes("dense");
  const asset = await loadAsset(needDense);

  const renderer = new WebGPURenderer({ device, trackTimestamp: TIMING && features.length > 0 });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);

  const pool = await VDBGridPool.create(device, { label: "k3-sparse" });
  const id = pool.add(asset.image);
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage"] });
  const gridInfo = pool.info(id);

  const frameRenderer = new VDBFrameRenderer();
  Object.assign(frameRenderer.settings, {
    width: FROXELS[0],
    height: FROXELS[1],
    depth: FROXELS[2],
    subSteps: SUBSTEPS,
    jitter: false,
    temporal: false,
    lighting: "volume",
    multipleScattering: params.get("ms") !== "0",
  });
  result.width = W;
  result.height = H;
  result.froxels = [...FROXELS];

  const denseSource = asset.dense ? vdbDenseTextures(asset.dense) : null;
  if (asset.dense) {
    const t = asset.dense.density;
    result.dense = { dimensions: [...asset.dense.dimensions], meanDensity: asset.dense.meanDensity, textureBytes: t.data.byteLength + (t.mip?.data.byteLength ?? 0) };
  }
  const statsOverride =
    asset.identical && STATS === "dense" && asset.dense ? { meanNonZero: asset.dense.meanDensity, max: asset.dense.maxDensity, activeVoxels: asset.dense.activeTexels } : undefined;

  const sources = new Map<string, VDBThreeVolumeSource>();
  const sourceFor = (lod: VDBSparseLod): VDBThreeVolumeSource => {
    let s = sources.get(lod);
    if (!s) {
      s = vdbSparseSource(
        grids,
        { density: id },
        {
          lod,
          images: { density: asset.image },
          ...(statsOverride ? { stats: statsOverride } : {}),
          ...(asset.indexBox ? { indexBox: asset.indexBox } : {}),
          ...(asset.bbox ? { bbox: asset.bbox } : {}),
          ...(CLAMP01 ? { normalize: "clamp01" as const } : {}),
          ...(PYRAMID_MAX !== undefined ? { pyramidMax: PYRAMID_MAX } : {}),
        },
      );
      sources.set(lod, s);
    }
    return s;
  };
  const anySparse = sourceFor("none") as ReturnType<typeof vdbSparseSource>;
  result.grid = {
    indexBox: anySparse.indexBox,
    texels: [...anySparse.texels],
    meanNonZero: anySparse.stats.meanNonZero,
    ...(statsOverride ? { gridMean: (vdbSparseSource(grids, { density: id }, { lod: "none", images: { density: asset.image }, ...(asset.indexBox ? { indexBox: asset.indexBox } : {}) }).stats.meanNonZero), denseMean: asset.dense!.meanDensity } : {}),
    poolBytes: gridInfo.capacity,
    imageBytes: gridInfo.imageBytes,
    type: gridInfo.type,
  };

  // Camera from the volume box (the sparse box; the dense one is the same for identical-voxel assets).
  const b = asset.bbox ?? anySparse.bbox;
  const bbox = new THREE.Box3(new THREE.Vector3(...b[0]), new THREE.Vector3(...b[1]));
  const center = bbox.getCenter(new THREE.Vector3());
  const diag = bbox.getSize(new THREE.Vector3()).length();
  const camera = new THREE.PerspectiveCamera(60, W / H, diag * 0.01, diag * 40);
  const eyeDir = new THREE.Vector3(0.9, 0.35, 1.1).normalize();
  const placeCamera = (deg: number) => {
    const dir = eyeDir.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(deg));
    camera.position.copy(center).add(dir.multiplyScalar(diag * 0.75 * DIST));
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
  };
  placeCamera(0);

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const bgBytes = (() => {
    const c = BACKGROUND.clone().convertLinearToSRGB();
    return [c.r, c.g, c.b].map((v) => Math.round(v * 255));
  })();

  const volumeProps = {
    frameRenderer,
    baseColor: "#ffffff",
    absorbance: 1.0,
    densityScale: 1.0,
    normalizeDensity: true,
    opacity: 1.0,
    extinction: Number(params.get("extinction") ?? 40),
    phase: "dual-lobe" as const,
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    shadowSteps: SHADOW_STEPS,
    mipmaps: true,
  };

  for (const variant of VARIANTS) {
    const wall0 = performance.now();
    const [kind, flag] = variant.split("-");
    let source: VDBThreeVolumeSource;
    if (kind === "dense") {
      if (!denseSource) throw new Error("no dense volume for this asset");
      source = denseSource;
    } else {
      const lod = LOD_OF[kind!];
      if (!lod) throw new Error(`unknown variant ${variant}; use dense, none, node, pyramid (+ -nohdda)`);
      source = sourceFor(lod);
    }
    const volume = new VDBFrameVolume(source, { ...volumeProps, hdda: flag !== "nohdda", ...(asset.bbox || kind === "dense" ? { bbox: bbox.clone() } : {}) });
    const scene = new THREE.Scene();
    scene.background = BACKGROUND.clone();
    scene.add(volume);
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(100.0, 200.0, 100.0).add(center);
    sun.target.position.copy(center);
    scene.add(sun, sun.target);
    scene.updateMatrixWorld(true);

    const frame = async (): Promise<Uint8Array> => {
      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      return readDisplay(renderer, rt);
    };
    placeCamera(0);
    const first = await frame();
    const second = await frame();
    let nonBackground = 0;
    for (let i = 0; i < first.length; i += 4) {
      if (Math.abs(first[i]! - bgBytes[0]!) + Math.abs(first[i + 1]! - bgBytes[1]!) + Math.abs(first[i + 2]! - bgBytes[2]!) > 6) nonBackground++;
    }
    const sparse = source.kind === "sparse" ? (source as ReturnType<typeof vdbSparseSource>) : null;
    const r: VariantResult = {
      name: variant,
      rgba: toBase64(first),
      nonBackground,
      inPageIdentical: first.every((v, i) => v === second[i]),
      gpuBytes: sparse ? { source: sparse.gpuBytes.pool, pyramid: sparse.gpuBytes.pyramid } : { source: result.dense!.textureBytes, pyramid: 0 },
      lodLevels: source.lodLevels,
    };

    if (ORBIT > 0) {
      const frames: Uint8Array[] = [first];
      for (let f = 1; f <= ORBIT; f++) {
        placeCamera(f * ORBIT_DEG);
        frames.push(await frame());
      }
      placeCamera(0);
      const diffs: number[] = [];
      for (let f = 1; f < frames.length; f++) diffs.push(meanAbs(frames[f - 1]!, frames[f]!));
      let noise = 0;
      for (let f = 1; f + 1 < frames.length; f++) noise += temporalNoise(frames[f - 1]!, frames[f]!, frames[f + 1]!);
      r.flicker = {
        frames: frames.length,
        meanDiff: diffs.reduce((s, v) => s + v, 0) / Math.max(1, diffs.length),
        temporalNoise: noise / Math.max(1, frames.length - 2),
        meanDiffs: diffs,
      };
    }

    if (TIMING && features.length > 0) {
      const times: number[] = [];
      const frames = Number(params.get("timingFrames") ?? 15);
      for (let f = 0; f < frames + 3; f++) {
        frameRenderer.renderFroxels(renderer, scene, camera);
        const ms = await renderer.resolveTimestampsAsync("render");
        if (f >= 3 && typeof ms === "number") times.push(ms);
      }
      times.sort((a, c) => a - c);
      r.froxelsMs = times.length ? times[Math.floor(times.length / 2)]! : null;
      result.timingsNote = `D25 provisional: ${result.adapter}, median of ${frames} froxel passes after 3 warm-up, not a quiet machine; never gate on it`;
    }

    r.wallMs = Math.round(performance.now() - wall0);
    result.variants.push(r);
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    canvas.title = variant;
    canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(first), W, H), 0, 0);
    root.append(canvas);
    log(
      `${variant}: ${nonBackground} volume px, repeat identical ${r.inPageIdentical}, lodLevels ${r.lodLevels}, GPU bytes ${JSON.stringify(r.gpuBytes)}` +
        (r.flicker ? `, flicker mean ${r.flicker.meanDiff.toFixed(3)} noise ${r.flicker.temporalNoise.toFixed(3)}` : "") +
        (r.froxelsMs !== undefined ? `, froxels ${r.froxelsMs?.toFixed(2)} ms (provisional)` : ""),
      r.inPageIdentical ? "ok" : "fail",
    );
    volume.dispose();
  }

  if (PROBE) {
    result.probe = await probe(renderer, grids, id, asset.image, anySparse.indexBox);
    log(`probe: ${result.probe.count} values, max |GPU − CPU| ${result.probe.maxAbs.toExponential(2)}, ${result.probe.over} over, ${result.probe.skipMismatch} HDDA mismatches`, result.probe.over === 0 ? "ok" : "fail");
  }
  for (const w of result.warnings) log(`warning: ${w}`, "warn");
  for (const s of sources.values()) s.dispose();
  denseSource?.dispose();
}

// --- GPU probe of the sparse WGSL against the CPU twins -------------------------------------------

/** PCG-style u32 hash, the same in WGSL and JS. */
function hash(x: number): number {
  const v = (Math.imul(x >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul((v >>> ((v >>> 28) + 4)) ^ v, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

const PROBE_WGSL = /* wgsl */ `fn k3SparseProbe(g: u32, px: vec2u, lo: vec3f, size: vec3f) -> vec4f {
    let i = px.x + px.y * 4096u;
    let h0 = k3Hash(i * 7u + 1u);
    let h1 = k3Hash(i * 7u + 2u);
    let h2 = k3Hash(i * 7u + 3u);
    let r = vec3f(f32(h0 & 4095u), f32(h1 & 4095u), f32(h2 & 4095u)) / 4096.0;
    let p = (lo - vec3f(1.0)) + r * (size + vec3f(2.0));
    let si = vec3u(size);
    let ijk = vec3i(lo) - vec3i(1) + vec3i(vec3u(k3Hash(i * 7u + 4u) % (si.x + 2u), k3Hash(i * 7u + 5u) % (si.y + 2u), k3Hash(i * 7u + 6u) % (si.z + 2u)));
    var acc = vdbAccessorInit(g);
    if px.y == 0u {
        return vec4f(vdbNodeAverage(&acc, ijk, 8u), vdbNodeAverage(&acc, ijk, 128u), vdbFrameNodeAverage(g, p, 8u), vdbFrameNodeAverage(g, p, 128u));
    }
    let hd = k3Hash(i * 7u + 7u);
    let d = (vec3f(f32(hd & 1023u), f32((hd >> 10u) & 1023u), f32((hd >> 20u) & 1023u)) / 1024.0 - vec3f(0.5)) * 2.0;
    let t1 = 2.0 * max(size.x, max(size.y, size.z)) / max(length(d), 0.25);
    let a = vdbFrameSkipEmpty(g, p, d, 0.0, t1, 0.0, 0u);
    let b = vdbFrameSkipEmpty(g, p, d, 0.0, t1, 0.0, 1u);
    return vec4f(a, b);
}`;
const HASH_WGSL = /* wgsl */ `fn k3Hash(x: u32) -> u32 {
    let v = x * 747796405u + 2891336453u;
    let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}`;

async function probe(renderer: WebGPURenderer, grids: VDBGridNode, id: number, image: Uint32Array, box: { min: number[]; max: number[] }): Promise<Probe> {
  // The source's own WGSL helpers (the same text the inject material compiles).
  const helperSource = vdbSparseSource(grids, { density: id }, { lod: "node-average", stats: { meanNonZero: 1, max: 1, activeVoxels: 1 } });
  const nodeAverageFn = helperSource.nodeAverageFn();
  const skipFn = helperSource.skipEmptyFn();
  const hashFn = wgslFn(HASH_WGSL);
  const probeFn = wgslFn(PROBE_WGSL, [grids.module, hashFn, nodeAverageFn, skipFn]);
  const PW = 256;
  const PH = 2;
  const lo = new THREE.Vector3(...box.min);
  const size = new THREE.Vector3(box.max[0]! - box.min[0]! + 1, box.max[1]! - box.min[1]! + 1, box.max[2]! - box.min[2]! + 1);
  const material = new NodeMaterial();
  material.fragmentNode = Fn(() => {
    const px = uvec2(screenCoordinate.x.floor(), screenCoordinate.y.floor());
    return probeFn(uint(id), px, uniform(lo), uniform(size));
  })();
  material.vertexNode = vec4(TSL.positionGeometry.xy, 0.0, 1.0);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(quad);
  const target = new THREE.RenderTarget(PW, PH, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  renderer.setRenderTarget(target);
  renderer.render(scene, new THREE.OrthographicCamera());
  renderer.setRenderTarget(null);
  const gpu = tightRows((await renderer.readRenderTargetPixelsAsync(target, 0, 0, PW, PH)) as unknown as Float32Array, PW, PH, 4);
  const f = Math.fround;
  let over = 0;
  let maxAbs = 0;
  let count = 0;
  let skipMismatch = 0;
  let firstBad: unknown;
  for (let y = 0; y < PH; y++) {
    for (let x = 0; x < PW; x++) {
      const i = (x + y * 4096) >>> 0;
      const h0 = hash(i * 7 + 1);
      const h1 = hash(i * 7 + 2);
      const h2 = hash(i * 7 + 3);
      const r = [(h0 & 4095) / 4096, (h1 & 4095) / 4096, (h2 & 4095) / 4096];
      const p = [0, 1, 2].map((a) => f(f(box.min[a]! - 1) + f(r[a]! * f([size.x, size.y, size.z][a]! + 2)))) as [number, number, number];
      const si = [size.x, size.y, size.z];
      const ijk = [0, 1, 2].map((a) => box.min[a]! - 1 + (hash(i * 7 + 4 + a) % (si[a]! + 2))) as [number, number, number];
      const g = Array.from(gpu.subarray((y * PW + x) * 4, (y * PW + x) * 4 + 4));
      let cpu: number[];
      if (y === 0) {
        cpu = [nodeAverage(image, ijk, 8), nodeAverage(image, ijk, 128), vdbFrameNodeAverageSample(image, p, 8), vdbFrameNodeAverageSample(image, p, 128)];
      } else {
        const hd = hash(i * 7 + 7);
        const d = [hd & 1023, (hd >>> 10) & 1023, (hd >>> 20) & 1023].map((v) => f(f(f(v / 1024) - 0.5) * 2)) as [number, number, number];
        const t1 = f(f(2 * Math.max(...si)) / Math.max(f(Math.hypot(...d)), 0.25));
        cpu = [...vdbFrameSkipEmpty(image, p, d, 0, t1, 0, false), ...vdbFrameSkipEmpty(image, p, d, 0, t1, 0, true)];
      }
      for (let c = 0; c < 4; c++) {
        const d = Math.abs(g[c]! - cpu[c]!);
        count++;
        const tol = 1e-4 * Math.max(1, Math.abs(cpu[c]!));
        if (y === 1) {
          // HDDA decisions: a cell boundary rounding differently moves ts/te by one cell. Count, don't fail.
          if (d > tol) skipMismatch++;
          continue;
        }
        maxAbs = Math.max(maxAbs, d);
        if (d > tol) {
          over++;
          firstBad ??= { x, y, c, gpu: g[c], cpu: cpu[c], p, ijk };
        }
      }
    }
  }
  target.dispose();
  material.dispose();
  return { count, over, maxAbs, skipMismatch, ...(firstBad ? { firstBad } : {}) };
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
