/**
 * K6.R1 harness: one scene of the reference report's own set (`k6-scenes.ts`: the WDAS cloud, the smoke, absorbing and
 * thin bunnies, a point light) through `./three` `VDBFrameRenderer`, read back for scripts/k6/reference-report.mjs.
 * The reference page (`k6-reference.html?case=scene`) renders the same definition.
 *
 * Query: `?scene=<name>` (K6_SCENES), `?lighting=volume|frame`, `?ms=0|1|two-sided|octaves|neural` (K6.3, core only:
 * `mlp`, `multi-scale` with `?msEdge=zero|clamp`, `?msLevels=`),
 * `?backend=tsl|core` (K6.1/K6.2 internal A/B), `?froxelMix=mixture|over`, `?w=&h=` (default 1280 × 720), `?froxels=WxHxD`
 * (default his 512x512x24), `?lod=` (sparse sources: default 'coarse-pyramid', D28), `?shadowRes=`, `?shadowSteps=`.
 * Jitter and temporal reprojection off (deterministic); no MSAA (no meshes).
 *
 * Results: `window.__VDB_K6_SCENE__` = { done, error?, adapter, width, height, rgba (base64 RGBA8, sRGB, his background
 * 0x598eff), settings, backend, froxelMix, warnings, uncaptured }.
 */
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { VDBFrameRenderer, VDBFrameVolume, VDBGridNode, vdbDenseTextures, vdbSparseSource, type VDBFrameMultipleScattering, type VDBSparseLod, type VDBThreeVolumeSource } from "@pmndrs/vdb/three";
import { VDBGridPool } from "@pmndrs/vdb";
import { K6_SCENES, k6LoadAsset, k6SceneFrame, k6SparseBox, type K6Lighting } from "./k6-scenes.js";

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  scene?: string;
  width?: number;
  height?: number;
  rgba?: string;
  settings?: Record<string, unknown>;
  backend?: string;
  froxelMix?: string;
  sourceKind?: "dense" | "sparse";
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K6_SCENE__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const NAME = params.get("scene") ?? "wdas";
const W = Number(params.get("w") ?? 1280);
const H = Number(params.get("h") ?? 720);
const FROXELS = (params.get("froxels") ?? "512x512x24").split("x").map(Number);
const LIGHTING = (params.get("lighting") ?? "frame") as K6Lighting;
const parseMs = (v: string): VDBFrameMultipleScattering => (v === "0" || v === "false" ? false : v === "1" || v === "true" ? true : (v as VDBFrameMultipleScattering));
const MS = parseMs(params.get("ms") ?? "1");
const BACKGROUND = new THREE.Color(0x598eff);

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K6_SCENE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

async function readImage(renderer: WebGPURenderer, rt: THREE.RenderTarget): Promise<string> {
  const raw = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, W, H)) as unknown as Uint16Array;
  const stride = H > 1 ? (raw.length - W * 4) / (H - 1) : W * 4;
  const out = new Uint8Array(W * H * 4);
  const oetf = (x: number) => (x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
  const byte = (v: number) => Math.round(Math.min(1, Math.max(0, oetf(v))) * 255);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * stride + x * 4;
      const o = (y * W + x) * 4;
      for (let c = 0; c < 3; c++) out[o + c] = byte(THREE.DataUtils.fromHalfFloat(raw[i + c]!));
      out[o + 3] = 255;
    }
  let s = "";
  for (let i = 0; i < out.length; i += 0x8000) s += String.fromCharCode(...out.subarray(i, i + 0x8000));
  return btoa(s);
}

async function main(): Promise<void> {
  root.textContent = "";
  const def = K6_SCENES[NAME];
  if (!def) throw new Error(`unknown ?scene=${NAME}; known: ${Object.keys(K6_SCENES).join(", ")}`);
  result.scene = NAME;
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const renderer = new WebGPURenderer({ device });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);

  const asset = await k6LoadAsset(def.asset);
  const frameRenderer = new VDBFrameRenderer();
  frameRenderer.backend = params.get("backend") === "core" ? "core" : "tsl";
  frameRenderer.froxelMix = params.get("froxelMix") === "over" ? "over" : "mixture";
  // K6.3: the core's 'multi-scale' settings (the V75 edge A/B)
  if (params.has("msEdge")) frameRenderer.coreSettings.multiScaleEdge = params.get("msEdge") === "clamp" ? "clamp" : "zero";
  if (params.has("msLevels")) frameRenderer.coreSettings.multiScaleLevels = Number(params.get("msLevels"));
  // K6.Q1: the shadowed ambient (?ao=1, ?aoPairs=, ?aoSteps=)
  if (params.has("ao")) frameRenderer.coreSettings.ambientOcclusion = params.get("ao") === "1";
  if (params.has("aoPairs")) frameRenderer.coreSettings.ambientOcclusionPairs = Number(params.get("aoPairs"));
  if (params.has("aoSteps")) frameRenderer.coreSettings.ambientOcclusionSteps = Number(params.get("aoSteps"));
  result.backend = frameRenderer.backend;
  result.froxelMix = frameRenderer.froxelMix;
  Object.assign(frameRenderer.settings, {
    width: FROXELS[0],
    height: FROXELS[1],
    depth: FROXELS[2],
    jitter: false,
    temporal: false,
    lighting: LIGHTING,
    multipleScattering: MS,
    shadowResolution: Number(params.get("shadowRes") ?? 64),
    phase: "dual-lobe",
    phaseG: def.phase[0],
    phaseBackG: def.phase[1],
    phaseBackWeight: def.phase[2],
  });

  let source: VDBThreeVolumeSource;
  if (asset.image) {
    const pool = await VDBGridPool.create(device, { label: "k6-scene" });
    const id = pool.add(asset.image);
    pool.flush();
    await pool.settled();
    k6SparseBox(asset, pool.info(id));
    const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
    source = vdbSparseSource(grids, { density: id }, {
      lod: (params.get("lod") ?? "coarse-pyramid") as VDBSparseLod,
      images: { density: asset.image },
      stats: asset.stats,
      bbox: asset.bbox,
    });
    result.sourceKind = "sparse";
  } else {
    source = vdbDenseTextures(asset.dense!);
    result.sourceKind = "dense";
  }
  result.settings = { ...frameRenderer.settings };
  const frame = k6SceneFrame(def, asset, W / H, LIGHTING);
  const bbox = new THREE.Box3(new THREE.Vector3(...asset.bbox[0]), new THREE.Vector3(...asset.bbox[1]));
  const volume = new VDBFrameVolume(source, {
    frameRenderer,
    bbox,
    baseColor: new THREE.Color().setRGB(def.albedo[0], def.albedo[1], def.albedo[2], THREE.LinearSRGBColorSpace),
    angstromExponent: 0.0,
    absorbance: 1.0,
    densityScale: 1.0,
    normalizeDensity: true,
    opacity: 1.0,
    extinction: def.extinction,
    phase: "dual-lobe",
    phaseG: def.phase[0],
    phaseBackG: def.phase[1],
    phaseBackWeight: def.phase[2],
    shadowSteps: Number(params.get("shadowSteps") ?? 32),
    mipmaps: true,
  });
  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(volume);
  for (const l of frame.three) {
    const color = new THREE.Color().setRGB(l.color[0], l.color[1], l.color[2], THREE.LinearSRGBColorSpace);
    if (l.kind === "sun") {
      const sun = new THREE.DirectionalLight(color, l.intensity);
      sun.position.set(...l.position);
      sun.target.position.set(...frame.center);
      scene.add(sun, sun.target);
    } else if (l.kind === "ambient") scene.add(new THREE.AmbientLight(color, l.intensity));
    else if (l.kind === "hemisphere") {
      const ground = new THREE.Color().setRGB(l.ground![0], l.ground![1], l.ground![2], THREE.LinearSRGBColorSpace);
      const h = new THREE.HemisphereLight(color, ground, l.intensity);
      h.position.set(...l.position);
      scene.add(h);
    } else if (l.kind === "probe") {
      const sh = new THREE.SphericalHarmonics3().fromArray(l.sh!);
      scene.add(new THREE.LightProbe(sh, l.intensity));
    } else {
      const p = new THREE.PointLight(color, l.intensity, 0, l.decay);
      p.position.set(...l.position);
      scene.add(p);
    }
  }
  const camera = new THREE.PerspectiveCamera(frame.fov, W / H, frame.near, frame.far);
  camera.matrixAutoUpdate = false;
  camera.matrixWorld.fromArray(frame.cameraToWorld);
  camera.matrix.copy(camera.matrixWorld);
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  camera.projectionMatrix.fromArray(frame.projection);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  scene.updateMatrixWorld(true);

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const draw = () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
  };
  // First frames: shader builds, the sparse pyramid, the 'frame' grid passes (redrawn only on change).
  for (let i = 0; i < 4; i++) {
    draw();
    await device.queue.onSubmittedWorkDone();
  }
  result.width = W;
  result.height = H;
  result.rgba = await readImage(renderer, rt);
  log(`${NAME} ${LIGHTING} ms ${String(MS)} backend ${frameRenderer.backend} (${frameRenderer.froxelMix}): ${W}x${H}`, "ok");
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
