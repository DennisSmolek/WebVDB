/**
 * K3.1 harness (R1 §6 Step 1): our port of mjurczyk's froxel pipeline with `'volume'` lighting,
 * on the dense `.wvdb` source, rendered offscreen and read back.
 *
 * Scenes (`?scene=`):
 * - `smoke` (default): the committed CC0 `cloud_04_variant_0000.wvdb`, one directional light, a
 *   small froxel grid; the e2e scene (SwiftShader-sized).
 * - his parity scenes, by the names of `scripts/k3/capture-reference.mjs` (`step1-volume-dir-ms-off`,
 *   `step1-volume-dir-ms-on`, `step1-lighting-volume`, `step1-lights-directional`,
 *   `step1-multiple-scattering-off`, `step0-bunny`, `step0-smoke`): his `frame.js` scene rebuilt (his
 *   `bunny_cloud.wvdb` / `smoke.wvdb` from `/fixtures/k3/`, not committed: CC BY-SA), his camera,
 *   volume placement, phase and light; or, when `?manifest=<url>` (default
 *   `/fixtures/k3-reference/75e7345/manifest.json`) has the scenario, the captured camera and volume
 *   matrices. 1280x720, his froxel defaults (512x512x24, 32 sub-steps). `parityExpected` says whether
 *   the Step 1 port can match his capture (see docs/handoffs/K3-1.md: environment and 'frame' lighting).
 *
 * Pipeline: scene → linear HalfFloat RenderTarget (as his canvas frame-buffer target; no MSAA) →
 * readback → sRGB OETF on the CPU (three's formula, NoToneMapping) → RGBA8. Jitter and temporal
 * off: deterministic. Also: a probe pass of the TSL chunks against the CPU twins (`vdbFrame*`,
 * tools/frame) and, with `?twin=1`, the integrate pass against `vdbFrameIntegrate` from the read-back
 * froxel atlas (froxel grid = screen size, so each pixel reads one froxel column unfiltered).
 *
 * Other query: `?w=&h=`, `?froxels=WxHxD`, `?substeps=`, `?shadowSteps=`, `?ms=0|1`, `?mips=0`,
 * `?occluder=1` (an opaque box in front of the volume centre), `?novolume=1`, `?timing=1` (per-pass
 * GPU ms: scene, froxels (inject), integrate; `?timingFrames=300&timingWarmup=60` for medians; provisional per
 * D25), `?file=`, `?debug=1` (raw chunk-probe values in the result).
 * Results: `window.__VDB_K3_FRAME__` (e2e/k3-frame.spec.ts writes PNGs to artifacts/k3/ours/).
 */
import * as THREE from "three";
import { NodeMaterial, WebGPURenderer } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBFrameRenderer, VDBFrameVolume, vdbDenseTextures, vdbFrameChunks } from "@pmndrs/vdb/three";
import {
  readWvdb,
  vdbFrameDistanceFade,
  vdbFrameHenyeyGreenstein,
  vdbFrameIntegrate,
  vdbFrameNoise,
  vdbFramePhaseWeight,
  vdbFrameShadowAbsorption,
  vdbFrameShadowDepth,
  vdbFrameShadowStep,
  vdbFrameShadowTransmittance,
  vdbFrameShadowedLight,
  vdbFrameSliceDepth,
  vdbFrameSrgbBlend,
  vdbSrgbOetf,
  wvdbDenseVolume,
  type VDBFrameFroxelSample,
} from "@pmndrs/vdb/tools";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;
const { Fn, float, screenCoordinate, vec2, vec3, vec4, If, select } = TSL as unknown as Record<string, N>;

interface SceneDef {
  name: string;
  file: string;
  /** Lights of his `lightSetup` we can render ('dir' only in Step 1). */
  lightSetup: "dir" | "environment";
  lighting: "volume" | "frame";
  multipleScattering: boolean;
  /** True when the Step 1 port can match his capture of this scene. */
  parityExpected: boolean;
  why?: string;
  smoke?: boolean;
}

// His frame.js scene list (capture-reference.mjs SCENARIOS). `multipleScattering` 'neural' (his default) is the
// same as true with 'volume' lighting (the network runs only in 'frame' lighting).
const HIS_SCENES: Record<string, SceneDef> = {
  // The Step 1 parity scenes: his page with `isolate` (capture-reference.mjs): directional light only, no environment
  // light on the volume, no skybox/ground/props, background 0x598eff.
  "step1-volume-dir-ms-off-isolated": { name: "step1-volume-dir-ms-off-isolated", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "dir", lighting: "volume", multipleScattering: false, parityExpected: true },
  "step1-volume-dir-ms-on-isolated": { name: "step1-volume-dir-ms-on-isolated", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "dir", lighting: "volume", multipleScattering: true, parityExpected: true },
  "step1-volume-dir-ms-off": {
    name: "step1-volume-dir-ms-off",
    file: "/fixtures/k3/bunny_cloud.wvdb",
    lightSetup: "dir",
    lighting: "volume",
    multipleScattering: false,
    parityExpected: false,
    why: "his 'dir' setup keeps scene.environment, which lights the volume too (K3.4), and shows the skybox and ground",
  },
  "step1-volume-dir-ms-on": {
    name: "step1-volume-dir-ms-on",
    file: "/fixtures/k3/bunny_cloud.wvdb",
    lightSetup: "dir",
    lighting: "volume",
    multipleScattering: true,
    parityExpected: false,
    why: "his 'dir' setup keeps scene.environment, which lights the volume too (K3.4), and shows the skybox and ground",
  },
  "step1-lighting-volume": {
    name: "step1-lighting-volume",
    file: "/fixtures/k3/bunny_cloud.wvdb",
    lightSetup: "environment",
    lighting: "volume",
    multipleScattering: true,
    parityExpected: false,
    why: "his 'environment' light setup lights the volume from the HDR environment (K3.4); rendered unlit here",
  },
  "step1-lights-directional": { name: "step1-lights-directional", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "dir", lighting: "frame", multipleScattering: true, parityExpected: false, why: "'frame' lighting (K3.3); rendered with 'volume' lighting" },
  "step1-multiple-scattering-off": { name: "step1-multiple-scattering-off", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "environment", lighting: "frame", multipleScattering: false, parityExpected: false, why: "'frame' lighting + environment (K3.3/K3.4)" },
  "step0-bunny": { name: "step0-bunny", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "environment", lighting: "frame", multipleScattering: true, parityExpected: false, why: "'frame' lighting + environment (K3.3/K3.4)" },
  "step0-smoke": { name: "step0-smoke", file: "/fixtures/k3/smoke.wvdb", lightSetup: "environment", lighting: "frame", multipleScattering: true, parityExpected: false, why: "'frame' lighting + environment (K3.3/K3.4)" },
};

interface ChunkProbe {
  count: number;
  maxAbs: number;
  maxRel: number;
  over: number;
  firstBad?: { row: number; x: number; channel: number; gpu: number; cpu: number };
}

interface TwinCheck {
  pixels: number;
  covered: number;
  maxAbs: number;
  over: number;
  firstBad?: { x: number; y: number; gpu: number[]; cpu: number[] };
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  scene?: string;
  parityExpected?: boolean;
  parityNote?: string;
  source?: "manifest" | "frame.js";
  file?: string;
  volume?: { dimensions: number[]; lod: number; meanDensity: number; mipLevels: number; statsSource: string };
  width?: number;
  height?: number;
  froxels?: number[];
  rgba?: string;
  /** Screen rectangle of the volume box in pixels (top-left origin): x0, y0, x1, y1. */
  volumeRect?: number[];
  inPageIdentical?: boolean;
  nonBackground?: number;
  chunkProbe?: ChunkProbe;
  twin?: TwinCheck;
  timings?: { provisional: true; note: string; sceneMs: number | null; froxelsMs: number | null; fullMs: number | null; integrateMs: number | null };
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_FRAME__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const SCENE = params.get("scene") ?? "smoke";
const def: SceneDef =
  SCENE === "smoke"
    ? { name: "smoke", file: cloudWvdbUrl, lightSetup: "dir", lighting: "volume", multipleScattering: true, parityExpected: false, why: "our own test scene", smoke: true }
    : HIS_SCENES[SCENE]!;
const W = Number(params.get("w") ?? (def?.smoke ? 320 : 1280));
const H = Number(params.get("h") ?? (def?.smoke ? 180 : 720));
const FROXELS = (params.get("froxels") ?? (def?.smoke ? "128x128x24" : "512x512x24")).split("x").map(Number);
const SUBSTEPS = Number(params.get("substeps") ?? 32);
const SHADOW_STEPS = Number(params.get("shadowSteps") ?? 32);
const MIPS = params.get("mips") !== "0";
const TIMING = params.get("timing") === "1";
const TWIN = params.get("twin") === "1";
const OCCLUDER = params.get("occluder") === "1";
const NOVOLUME = params.get("novolume") === "1";
const MANIFEST = params.get("manifest") ?? "/fixtures/k3-reference/75e7345/manifest.json";
// His examples page: scene.background 0x598eff until the environment loads (main.jsx L41).
const BACKGROUND = new THREE.Color(0x598eff);

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K3_FRAME__ = result;

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

/**
 * Readback of a render target as tight rows. three r187 WebGPU returns the rows top first (row 0 is
 * `screenCoordinate.y` 0), padded to 256 bytes; the chunk probe below checks the orientation.
 */
function tightRows<T extends Uint16Array | Float32Array | Uint8Array>(pixels: T, w: number, h: number, channels: number): T {
  const tight = w * channels;
  const stride = h > 1 ? (pixels.length - tight) / (h - 1) : tight;
  if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${pixels.length} elements for ${w}x${h}`);
  const out = new (pixels.constructor as new (n: number) => T)(w * h * channels);
  for (let y = 0; y < h; y++) out.set(pixels.subarray(y * stride, y * stride + tight) as never, y * tight);
  return out;
}

async function readHalf(renderer: WebGPURenderer, rt: THREE.RenderTarget, w: number, h: number): Promise<Float32Array> {
  const raw = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, w, h)) as unknown as Uint16Array;
  const rows = tightRows(raw, w, h, 4);
  const out = new Float32Array(rows.length);
  for (let i = 0; i < rows.length; i++) out[i] = THREE.DataUtils.fromHalfFloat(rows[i]!);
  return out;
}

const toByte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);

/** Linear (top row first) → display RGBA8 as his canvas output pass: NoToneMapping, sRGB OETF, 8-bit. */
function toDisplay(linear: Float32Array): Uint8Array {
  const out = new Uint8Array(linear.length);
  for (let i = 0; i < linear.length; i += 4) {
    out[i] = toByte(vdbSrgbOetf(linear[i]!));
    out[i + 1] = toByte(vdbSrgbOetf(linear[i + 1]!));
    out[i + 2] = toByte(vdbSrgbOetf(linear[i + 2]!));
    out[i + 3] = 255;
  }
  return out;
}

interface ManifestScenario {
  name: string;
  metadata?: { camera?: { matrixWorld: number[]; projectionMatrix: number[]; near: number; far: number; fov: number; aspect: number }; volumes?: { matrixWorld: number[] }[] };
}

async function loadManifest(name: string): Promise<ManifestScenario | null> {
  try {
    const r = await fetch(MANIFEST);
    if (!r.ok) return null;
    const m = (await r.json()) as { scenarios?: ManifestScenario[] };
    return m.scenarios?.find((s) => s.name === name) ?? null;
  } catch {
    return null;
  }
}

async function main(): Promise<void> {
  root.textContent = "";
  if (!def) throw new Error(`unknown ?scene=${SCENE}; known: smoke, ${Object.keys(HIS_SCENES).join(", ")}`);
  result.scene = def.name;
  result.parityExpected = def.parityExpected;
  if (def.why) result.parityNote = def.why;
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
  const device = await adapter.requestDevice({ requiredFeatures: features });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  // --- the dense volume (his bytes) ----------------------------------------------------------
  const file = params.get("file") ?? def.file;
  const response = await fetch(file);
  if (!response.ok) throw new Error(`fetch ${file}: ${response.status}${def.smoke ? "" : " (his .wvdb files are not committed: copy them from his clone into fixtures/k3/)"}`);
  const wvdb = readWvdb(new Uint8Array(await response.arrayBuffer()));
  const dense = wvdbDenseVolume(wvdb, { mipmaps: MIPS });
  result.file = file;
  result.volume = {
    dimensions: [...dense.dimensions],
    lod: dense.lod,
    meanDensity: dense.meanDensity,
    mipLevels: dense.density.mip?.levels ?? 0,
    statsSource: dense.densityStatsSource,
  };

  // --- renderer and scene -------------------------------------------------------------------
  const renderer = new WebGPURenderer({ device, trackTimestamp: TIMING && features.length > 0 });
  await renderer.init();
  // NOTE The froxel grid is capped by the drawing buffer size (layout.ts getFroxelSize), as on his canvas.
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);

  const frameRenderer = new VDBFrameRenderer();
  Object.assign(frameRenderer.settings, {
    width: FROXELS[0],
    height: FROXELS[1],
    depth: FROXELS[2],
    subSteps: SUBSTEPS,
    jitter: false,
    temporal: false,
    lighting: "volume",
    multipleScattering: params.has("ms") ? params.get("ms") === "1" : def.multipleScattering,
  });
  result.froxels = [...FROXELS];

  const source = vdbDenseTextures(dense);
  const bbox = new THREE.Box3(new THREE.Vector3(...(dense.bbox?.[0] ?? [-0.5, -0.5, -0.5])), new THREE.Vector3(...(dense.bbox?.[1] ?? [0.5, 0.5, 0.5])));
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());

  // His frame.js demo params (L93–116, utils/phase.js phaseParams): dual-lobe 0.8 / -0.3 / 0.2, extinction 100.
  const volume = new VDBFrameVolume(source, {
    frameRenderer,
    bbox,
    baseColor: "#ffffff",
    absorbance: 1.0,
    densityScale: 1.0,
    normalizeDensity: true,
    opacity: 1.0,
    extinction: def.smoke ? 40 : 100,
    phase: "dual-lobe",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    shadowSteps: SHADOW_STEPS,
    mipmaps: MIPS,
  });

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 10000.0);
  const manifest = def.smoke ? null : await loadManifest(def.name);

  if (manifest?.metadata?.camera && manifest.metadata.volumes?.[0]) {
    result.source = "manifest";
    const c = manifest.metadata.camera;
    camera.matrixAutoUpdate = false;
    camera.matrixWorld.fromArray(c.matrixWorld);
    camera.matrix.copy(camera.matrixWorld);
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    camera.near = c.near;
    camera.far = c.far;
    camera.projectionMatrix.fromArray(c.projectionMatrix);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    volume.matrixAutoUpdate = false;
    volume.matrix.fromArray(manifest.metadata.volumes[0].matrixWorld);
    volume.matrixWorld.copy(volume.matrix);
  } else if (def.smoke) {
    result.source = "frame.js";
    const diag = size.length();
    camera.near = diag * 0.01;
    camera.far = diag * 20;
    camera.position.copy(center).add(new THREE.Vector3(0.9, 0.35, 1.1).normalize().multiplyScalar(diag * 0.75));
    camera.lookAt(center);
    camera.updateProjectionMatrix();
  } else {
    // His frame.js: the volume box is centred on the stage at groundLevel + height / 2 (L141–165, L351–364, L453–457);
    // main.jsx: camera (-100, -11, 80) looking at the orbit target (origin), fov 60, near 0.01, far 10000.
    result.source = "frame.js";
    const groundLevel = -50.0;
    const isSmoke = def.file.includes("smoke");
    const scale = isSmoke ? 100.0 / Math.max(size.x, size.y, size.z) : 1.0;
    volume.position.copy(center).negate().multiplyScalar(scale);
    volume.scale.setScalar(scale);
    volume.position.y += groundLevel + (size.y * scale) / 2.0;
    camera.position.set(-100.0, -11.0, 80.0);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }
  if (!NOVOLUME) scene.add(volume);

  if (def.lightSetup === "dir") {
    // His sun: DirectionalLight(0xffffff, 1.0) at (100, 200, 100), target at the origin (frame.js L191–194).
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(100.0, 200.0, 100.0);
    scene.add(sun);
    scene.add(sun.target);
  }

  const volumeCenter = new THREE.Vector3();
  camera.updateMatrixWorld(true);
  scene.updateMatrixWorld(true);
  volume.updateMatrixWorld(true);
  const boxWorld = new THREE.Box3().copy(bbox).applyMatrix4(volume.matrixWorld);
  boxWorld.getCenter(volumeCenter);

  if (OCCLUDER) {
    // An opaque box between the camera and the volume centre, half way.
    const camPos = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const occ = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff3366 }));
    occ.position.lerpVectors(camPos, volumeCenter, 0.5);
    occ.scale.setScalar(boxWorld.getSize(new THREE.Vector3()).length() * 0.08);
    occ.lookAt(camPos);
    scene.add(occ);
  }
  scene.updateMatrixWorld(true);

  // Screen rect of the volume box (top-left origin), for the compare mask.
  {
    const vp = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(i & 1 ? bbox.max.x : bbox.min.x, i & 2 ? bbox.max.y : bbox.min.y, i & 4 ? bbox.max.z : bbox.min.z).applyMatrix4(volume.matrixWorld);
      const v = new THREE.Vector4(p.x, p.y, p.z, 1).applyMatrix4(vp);
      const sx = ((v.x / v.w) * 0.5 + 0.5) * W;
      const sy = (1 - ((v.y / v.w) * 0.5 + 0.5)) * H;
      x0 = Math.min(x0, sx);
      x1 = Math.max(x1, sx);
      y0 = Math.min(y0, sy);
      y1 = Math.max(y1, sy);
    }
    result.volumeRect = [Math.max(0, Math.floor(x0)), Math.max(0, Math.floor(y0)), Math.min(W, Math.ceil(x1)), Math.min(H, Math.ceil(y1))];
  }

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });

  async function frame(): Promise<Float32Array> {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    return readHalf(renderer, rt, W, H);
  }

  const firstLinear = await frame();
  const first = toDisplay(firstLinear);
  const second = toDisplay(await frame());
  result.inPageIdentical = first.length === second.length && first.every((v, i) => v === second[i]);
  const bg = BACKGROUND.clone().convertLinearToSRGB();
  const bgBytes = [bg.r, bg.g, bg.b].map((v) => Math.round(v * 255));
  let nonBackground = 0;
  for (let i = 0; i < first.length; i += 4) {
    if (Math.abs(first[i]! - bgBytes[0]!) + Math.abs(first[i + 1]! - bgBytes[1]!) + Math.abs(first[i + 2]! - bgBytes[2]!) > 6) nonBackground++;
  }
  result.width = W;
  result.height = H;
  result.rgba = toBase64(first);
  result.nonBackground = nonBackground;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(first), W, H), 0, 0);
  root.append(canvas);
  log(
    `${def.name} (${result.source}): ${file.split("/").pop()} LOD ${dense.lod} ${dense.dimensions.join("x")}, mean ${dense.meanDensity.toFixed(4)}; ${W}x${H}, froxels ${FROXELS.join("x")}; ${nonBackground} volume px; repeat identical: ${result.inPageIdentical}`,
    result.inPageIdentical ? "ok" : "fail",
  );
  if (!def.parityExpected) log(`parity with his capture not expected: ${def.why}`, "warn");

  // --- integrate pass vs its CPU twin (froxel grid = screen: one unfiltered froxel column per pixel) ---
  if (TWIN) result.twin = await integrateTwin(renderer, frameRenderer, firstLinear);

  // --- TSL chunks vs the CPU twins --------------------------------------------------------------
  result.chunkProbe = await probeChunks(renderer);
  log(
    `chunk probe: ${result.chunkProbe.count} values, max |GPU - CPU twin| ${result.chunkProbe.maxAbs.toExponential(2)} (max rel ${result.chunkProbe.maxRel.toExponential(2)}), ${result.chunkProbe.over} over`,
    result.chunkProbe.over === 0 ? "ok" : "fail",
  );
  if (result.twin) {
    log(`integrate twin: ${result.twin.pixels} pixels (${result.twin.covered} with volume), max |GPU - CPU| ${result.twin.maxAbs.toExponential(2)}, ${result.twin.over} over`, result.twin.over === 0 ? "ok" : "fail");
  }

  // --- per-pass timing (D25: provisional) ---------------------------------------------------------
  if (TIMING && features.length > 0) {
    const resolve = async () => {
      const ms = await renderer.resolveTimestampsAsync("render");
      return typeof ms === "number" ? ms : null;
    };
    const median = (v: (number | null)[]) => {
      const x = v.filter((n): n is number => n !== null).sort((a, b) => a - b);
      return x.length ? x[Math.floor(x.length / 2)]! : null;
    };
    const frames = Number(params.get("timingFrames") ?? 1);
    const warmup = Number(params.get("timingWarmup") ?? 0);
    const samples = { scene: [] as (number | null)[], froxels: [] as (number | null)[], integrate: [] as (number | null)[], full: [] as (number | null)[] };
    await resolve();
    for (let f = 0; f < warmup + frames; f++) {
      // Scene alone (volume hidden); the froxel (inject) passes alone; scene + integrate over the held atlas; the full frame.
      volume.visible = false;
      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);
      const sceneMs = await resolve();
      volume.visible = true;
      frameRenderer.renderFroxels(renderer, scene, camera);
      const froxelsMs = await resolve();
      frameRenderer.holdFroxels = true;
      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);
      frameRenderer.holdFroxels = false;
      const heldMs = await resolve();
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      const fullMs = await resolve();
      if (f < warmup) continue;
      samples.scene.push(sceneMs);
      samples.froxels.push(froxelsMs);
      samples.integrate.push(heldMs !== null && sceneMs !== null ? heldMs - sceneMs : null);
      samples.full.push(fullMs);
    }
    result.timings = {
      provisional: true,
      note: `D25: not from a quiet machine; median of ${frames} frame(s) after ${warmup} warm-up; never gate on it. full is three's figure for the whole render (nested froxel passes may not be included)`,
      sceneMs: median(samples.scene),
      froxelsMs: median(samples.froxels),
      integrateMs: median(samples.integrate),
      fullMs: median(samples.full),
    };
    log(`timings (provisional, D25): ${JSON.stringify(result.timings)}`, "warn");
  } else {
    result.timings = { provisional: true, note: "timing off (?timing=1 needs timestamp-query); D25", sceneMs: null, froxelsMs: null, fullMs: null, integrateMs: null };
  }
  for (const w of result.warnings) log(`warning: ${w}`, "warn");
  source.dispose();
}

/**
 * The integrate pass against `vdbFrameIntegrate` + `vdbFrameSrgbBlend`. Needs froxels = screen size
 * (each pixel reads its own froxel column at texel centres) and no occluder (empty depth).
 */
async function integrateTwin(renderer: WebGPURenderer, frameRenderer: VDBFrameRenderer, linear: Float32Array): Promise<TwinCheck> {
  const atlasTarget = frameRenderer.atlasTarget!;
  const aw = atlasTarget.width;
  const ah = atlasTarget.height;
  const atlas = await readHalf(renderer, atlasTarget, aw, ah);
  const fu = frameRenderer.froxelUniforms;
  const grid = [fu.froxelGrid.value.x, fu.froxelGrid.value.y, fu.froxelGrid.value.z];
  if (grid[0] !== W || grid[1] !== H) throw new Error(`?twin=1 needs ?froxels=${W}x${H}xD (got ${grid.join("x")})`);
  const columns = fu.froxelColumns.value as number;
  const range: [number, number] = [fu.froxelRange.value.x, fu.froxelRange.value.y];
  const sr = frameRenderer.integrateMaterial.frameSliceRange.value;
  const bg = BACKGROUND.clone();
  const tol = 4e-3;
  let maxAbs = 0;
  let over = 0;
  let covered = 0;
  let firstBad: TwinCheck["firstBad"];
  const step = 3;
  let pixels = 0;
  for (let y = 3; y < H; y += step) {
    for (let x = 3; x < W; x += step) {
      pixels++;
      // Screen pixel (x, y top-left) → bottom-left froxel column (x, H - 1 - y).
      const fy = H - 1 - y;
      const froxels: VDBFrameFroxelSample[] = [];
      for (let s = sr.x; s < sr.y; s++) {
        const tx = (s % columns) * grid[0]! + x;
        const ty = Math.floor(s / columns) * grid[1]! + fy;
        // Atlas rows are bottom-left in the shaders' convention; the readback is top row first.
        const row = ah - 1 - ty;
        const o = (row * aw + tx) * 4;
        froxels.push([atlas[o]!, atlas[o + 1]!, atlas[o + 2]!, atlas[o + 3]!]);
      }
      const r = vdbFrameIntegrate(froxels, [sr.x, sr.y], range, grid[2]!, 1e20);
      const o = (y * W + x) * 4;
      const gpu = [linear[o]!, linear[o + 1]!, linear[o + 2]!];
      const cpu = r ? vdbFrameSrgbBlend([bg.r, bg.g, bg.b], r.radiance, r.alpha) : [bg.r, bg.g, bg.b];
      if (r) covered++;
      for (let c = 0; c < 3; c++) {
        // Half-float output: relative tolerance on top of the absolute one.
        const d = Math.abs(gpu[c]! - cpu[c]!);
        maxAbs = Math.max(maxAbs, d);
        if (d > tol + cpu[c]! * 2e-3) {
          over++;
          firstBad ??= { x, y, gpu, cpu };
        }
      }
    }
  }
  return { pixels, covered, maxAbs, over, ...(firstBad ? { firstBad } : {}) };
}

/** Evaluates the TSL chunks in a fragment pass (RGBA32F, one value per channel) and compares with the CPU twins. */
async function probeChunks(renderer: WebGPURenderer): Promise<ChunkProbe> {
  const PW = 64;
  const PH = 4;
  const C = vdbFrameChunks;
  const froxel = C.createFroxelUniforms();
  const range: [number, number] = [0.5, 300.0];
  froxel.froxelRange.value.set(range[0], range[1]);
  froxel.frameFadeRadius.value = 50.0;
  const jitterOn = 0.3;
  const albedo: [number, number, number] = [0.9, 0.8, 0.7];
  const ms = 0.42;
  const phase: [number, number, number, number] = [0.8, -0.3, 0.2, 0];

  const material = new NodeMaterial();
  material.fragmentNode = Fn(() => {
    const px = screenCoordinate.x.floor();
    const row = screenCoordinate.y.floor().toVar();
    const a = px.add(0.5).div(PW).toVar();
    const out = vec4(0.0).toVar();
    const cosT = a.mul(2.0).sub(1.0).toVar();
    const sinT = float(1.0).sub(cosT.mul(cosT)).max(0.0).sqrt();
    const ray = vec3(0.0, 0.0, 1.0);
    const light = vec3(sinT, 0.0, cosT);
    const phaseU = vec4(...phase);
    const e = a.mul(4.0).toVar();
    const alb = vec3(...albedo);
    const msU = float(ms);
    If(row.lessThan(0.5), () => {
      out.assign(vec4(C.frameHenyeyGreenstein(cosT, float(0.8)), C.frameHenyeyGreenstein(cosT, float(-0.3)), C.frameHenyeyGreenstein(cosT, float(0.0)), C.framePhaseWeight(phaseU, ray, light)));
    })
      .ElseIf(row.lessThan(1.5), () => {
        out.assign(
          vec4(
            C.frameSliceDepth(froxel, a),
            C.frameNoise(float(-1.0), vec2(px.mul(7.0).add(3.0), row.mul(13.0).add(5.0))),
            C.frameNoise(float(jitterOn), vec2(px.mul(7.0).add(3.0), row.mul(13.0).add(5.0))),
            C.frameDistanceFade(froxel, vec3(0.0, 0.0, a.mul(-60.0))),
          ),
        );
      })
      .ElseIf(row.lessThan(2.5), () => {
        const absorption = C.frameShadowAbsorption(e, alb).toVar();
        const s = C.frameShadowStep(e, absorption, msU).toVar();
        const depths = s.mul(3.0).toVar();
        out.assign(vec4(s.y, C.frameShadowDepth(depths, msU), C.frameShadowTransmittance(depths, msU).y, absorption));
      })
      .Else(() => {
        const shadow = vec2(e.negate().exp(), e.mul(-0.5).exp()).toVar();
        const s = C.frameShadowStep(e, C.frameShadowAbsorption(e, alb), float(0.0)).mul(2.0).toVar();
        out.assign(
          vec4(
            C.frameShadowedLight(phaseU, vec3(1.0, 1.0, 0.0), ray, light, shadow).x,
            C.frameShadowDepth(s, float(0.0)),
            C.frameShadowTransmittance(s, float(0.0)).x,
            select(a.greaterThan(0.5), C.framePhaseColor(phaseU, vec3(1.0, 1.0, 0.0), ray, light).y, float(0.0)),
          ),
        );
      });
    return out;
  })();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  material.vertexNode = vec4(TSL.positionGeometry.xy, 0.0, 1.0);
  quad.frustumCulled = false;
  const probeScene = new THREE.Scene();
  probeScene.add(quad);
  const target = new THREE.RenderTarget(PW, PH, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  renderer.setRenderTarget(target);
  renderer.render(probeScene, new THREE.OrthographicCamera());
  renderer.setRenderTarget(null);
  const raw = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, PW, PH)) as unknown as Float32Array;
  // Readback rows are top first, as screenCoordinate rows: out row 0 = probe row 0.
  const gpu = tightRows(raw, PW, PH, 4);

  let maxAbs = 0;
  let maxRel = 0;
  let over = 0;
  let count = 0;
  let firstBad: ChunkProbe["firstBad"];
  for (let row = 0; row < PH; row++) {
    for (let x = 0; x < PW; x++) {
      const a = (x + 0.5) / PW;
      const cosT = a * 2 - 1;
      const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
      const ray: [number, number, number] = [0, 0, 1];
      const light: [number, number, number] = [sinT, 0, cosT];
      const e = a * 4;
      const pixel: [number, number] = [x * 7 + 3, row * 13 + 5];
      let cpu: number[];
      if (row === 0) {
        cpu = [vdbFrameHenyeyGreenstein(cosT, 0.8), vdbFrameHenyeyGreenstein(cosT, -0.3), vdbFrameHenyeyGreenstein(cosT, 0), vdbFramePhaseWeight(phase, ray, light)];
      } else if (row === 1) {
        cpu = [vdbFrameSliceDepth(range, a), vdbFrameNoise(-1, pixel), vdbFrameNoise(jitterOn, pixel), vdbFrameDistanceFade(50, [0, 0, -a * 60])];
      } else if (row === 2) {
        const absorption = vdbFrameShadowAbsorption(e, albedo);
        const s = vdbFrameShadowStep(e, absorption, ms);
        const depths: [number, number, number] = [s[0] * 3, s[1] * 3, s[2] * 3];
        cpu = [s[1], vdbFrameShadowDepth(depths, ms), vdbFrameShadowTransmittance(depths, ms)[1], absorption];
      } else {
        const shadow: [number, number] = [Math.exp(-e), Math.exp(-e * 0.5)];
        const s0 = vdbFrameShadowStep(e, vdbFrameShadowAbsorption(e, albedo), 0);
        const s: [number, number, number] = [s0[0] * 2, s0[1] * 2, s0[2] * 2];
        cpu = [vdbFrameShadowedLight(phase, ray, light, shadow), vdbFrameShadowDepth(s, 0), vdbFrameShadowTransmittance(s, 0)[0], a > 0.5 ? vdbFramePhaseWeight(phase, ray, light) : 0];
      }
      for (let c = 0; c < 4; c++) {
        const g = gpu[(row * PW + x) * 4 + c]!;
        const d = Math.abs(g - cpu[c]!);
        const rel = d / Math.max(1, Math.abs(cpu[c]!));
        count++;
        maxAbs = Math.max(maxAbs, d);
        maxRel = Math.max(maxRel, rel);
        // f32 vs f64; IGN noise amplifies the fract() argument error (×52.98), so 2e-3 there.
        const tol = row === 1 && c === 2 ? 2e-3 : 2e-4;
        if (rel > tol) {
          over++;
          firstBad ??= { row, x, channel: c, gpu: g, cpu: cpu[c]! };
        }
      }
    }
  }
  target.dispose();
  material.dispose();
  if (params.get("debug") === "1") (result as unknown as Record<string, unknown>).probeGpu = Array.from(gpu);
  return { count, maxAbs, maxRel, over, ...(firstBad ? { firstBad } : {}) };
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
