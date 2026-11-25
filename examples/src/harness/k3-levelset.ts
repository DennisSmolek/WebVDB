/**
 * K3.7 harness (R1 §6 Step 7): level-set surfaces traced on the sparse pool (`VDBLevelSetMaterial`) and point
 * grids as sprites (`VDBPoints`), rendered offscreen and read back.
 *
 * Scenes (`?scene=`):
 * - `teapot` (default), `sphere`, `bunny`, `venus`: the level set of `/fixtures/vdb-samples/<model>.vdb`
 *   (`venus`: `/fixtures/k3/venusstatue.vdb`, copied from his clone), placed as his `levelset.js` places it
 *   (longest side 100 units, box centred at the origin), his default camera (-100, -11, 80) looking at the
 *   origin, fov 60. Renders (a) the probe passes, world normal and world hit position into a float target,
 *   compared per pixel with the CPU twin (`tools` `levelSetHit` / `levelSetNormal`) with `?twin=1`: silhouette
 *   IoU, hit distance, normal-angle error; `sphere` also against the analytic normal; (b) the shaded image
 *   (his 'sunHemi' lights, white, roughness 0.5), twice (in-page determinism).
 * - `points`: `sphere_points.vdb` through `vdbPointRenderData` as `VDBPoints` (his placement), white on black;
 *   checks the drawn count and that every point's projected centre is covered and every covered pixel lies
 *   within its sprite radius of a point. `points-color`: a synthetic 8x8 point grid with distinct colours;
 *   the pixel at each centre must be its colour.
 * - His Step 7 parity scenes, by the names of `scripts/k3/capture-reference.mjs`: `step7-*-normals`
 *   (`isolate: 'normals'`: view-space normal * 0.5 + 0.5 as the unlit colour, black background, sRGB output),
 *   `step7-sphere-points-isolated` (white points on black), and the `*-asis` scenes (INFO: his 'environment'
 *   lights are not reproduced; ours use his sun + hemisphere). 1280x720; the camera and object matrices come
 *   from `?manifest=` (default `/fixtures/k3-reference/75e7345/manifest.json`) when it has the scenario.
 *
 * K6.6 (`?backend=core`): the level-set scenes render through the core's raw pass (`vdbCreateLevelSetPassGpu`,
 * G-buffer + shade, no three objects) and are compared with the K6.6 surface twins (`levelSetPixelUv`,
 * `levelSetViewRay`, `levelSetSurface`): the same Stats plus NDC depth and shade; `&vsThree=1` also renders
 * three's `VDBLevelSetMaterial` probe on the same device and compares the two (IoU, hits, normals). The point
 * scenes render through `vdbCreatePointPassGpu` (analytic disc: strays judged against the disc). `sprite-probe`:
 * the sprite WGSL in a compute kernel vs the f32 twin. `?mutate=` breaks one twin term (pixel-corner,
 * ray-ortho, normal-view, depth-linear, shade-sign; sprite-half, sprite-min, sprite-w): the gate must fail.
 *
 * Query: `?w=&h=`, `?twin=1`, `?aniso=1` (a rotated, non-uniformly scaled instance transform on top of his placement), `?timing=N` (N shaded frames, median submit-to-done ms; provisional, D25),
 * `?maxSamples=`, `?normalStep=`, `?file=` (another level set or point `.vdb` under /fixtures/).
 * Results: `window.__VDB_K3_LEVELSET__` (e2e/k3-levelset.spec.ts writes PNGs to artifacts/k3/ours-step7/).
 */
import * as THREE from "three";
import { Mesh, WebGPURenderer } from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBGridNode, VDBLevelSetMaterial, VDBPoints, vdbProxyGeometry, type VDBTslNode } from "@pmndrs/vdb/three";
import {
  buildFromVdb,
  levelSetHit,
  levelSetNdcDepth,
  levelSetNormal,
  levelSetPixelUv,
  levelSetSurface,
  levelSetViewRay,
  parseVdb,
  vdbPointRenderData,
  vdbSrgbOetf,
  type VdbGrid,
  type VdbPointDataGrid,
} from "../../../packages/vdb/src/tools/internal.js";
import { vdbPointPixelScale } from "../../../packages/vdb/src/core/internal.js";
import { renderLevelSetCore, renderPointsCore, spriteProbe, type SpriteMutation } from "./k6-surface-core.js";

type V3 = [number, number, number];

interface Stats {
  pixels: number;
  gpuHits: number;
  cpuHits: number;
  both: number;
  iou: number;
  /** Hit distance GPU vs CPU, in index voxels. */
  hitMeanVoxels: number;
  hitMaxVoxels: number;
  /** Normal angle GPU vs CPU twin, degrees, over pixels both hit. */
  normalMeanDeg: number;
  normalP99Deg: number;
  normalMaxDeg: number;
  /** Same, with the 1-pixel silhouette rim removed (both masks eroded). */
  normalInteriorMaxDeg: number;
  /** sphere: GPU normal vs the analytic sphere normal. */
  analytic?: { meanDeg: number; maxDeg: number; radiusSpreadVoxels: number };
  twinMs: number;
  avgSamples: number;
  /** K6.6 core: GPU frag_depth vs the twin's NDC depth of the GPU hit, max |difference| (pixels both hit). */
  depthMaxAbs?: number;
  /** K6.6 core: shade output vs color · (ambient + light · max(n · −L, 0)) of the GPU normal, max |difference|. */
  shadeMaxAbs?: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  scene?: string;
  kind?: "levelset" | "points";
  parityExpected?: boolean;
  source?: "manifest" | "levelset.js";
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  nonBackground?: number;
  grid?: { name: string; indexBBox: { min: number[]; max: number[] }; voxelSize: number; leaves?: number; imageBytes: number };
  twin?: Stats;
  points?: { count: number; totalCount: number; spacing: number; covered: number; centresCovered: number; centresTotal: number; stray: number };
  colors?: { checked: number; bad: number; firstBad?: unknown };
  timing?: { frames: number; medianMs: number; minMs: number; note: string };
  backend?: "three" | "core";
  mutate?: string;
  /** K6.6 `&vsThree=1`: the core pass vs three's material on the same device. */
  vsThree?: { coreHits: number; threeHits: number; iou: number; hitMaxVoxels: number; normalMaxDeg: number };
  sprite?: { compared: number; maxUlp: number; over: number; bound: number; first?: unknown };
  pixelScale?: { core: number; three: number };
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_LEVELSET__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const SCENE = params.get("scene") ?? "teapot";
const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K3_LEVELSET__ = result;
const MANIFEST = params.get("manifest") ?? "/fixtures/k3-reference/75e7345/manifest.json";

interface SceneDef {
  name: string;
  kind: "levelset" | "points" | "points-color" | "sprite";
  file: string;
  output: "probe" | "normals" | "shaded" | "points";
  width: number;
  height: number;
  parityExpected: boolean;
}

const MODELS: Record<string, string> = {
  teapot: "/fixtures/vdb-samples/utahteapot.vdb",
  sphere: "/fixtures/vdb-samples/sphere.vdb",
  bunny: "/fixtures/vdb-samples/bunny.vdb",
  venus: "/fixtures/k3/venusstatue.vdb",
};
const E2E: Record<string, SceneDef> = {
  ...Object.fromEntries(Object.entries(MODELS).map(([k, file]) => [k, { name: k, kind: "levelset", file, output: "probe", width: 384, height: 216, parityExpected: false } as SceneDef])),
  points: { name: "points", kind: "points", file: "/fixtures/vdb-samples/sphere_points.vdb", output: "points", width: 384, height: 216, parityExpected: false },
  "points-color": { name: "points-color", kind: "points-color", file: "", output: "points", width: 128, height: 128, parityExpected: false },
  "sprite-probe": { name: "sprite-probe", kind: "sprite", file: "", output: "probe", width: 1, height: 1, parityExpected: false },
};
const PARITY: Record<string, SceneDef> = {
  "step7-utah-teapot-surface-normals": { name: "", kind: "levelset", file: MODELS.teapot!, output: "normals", width: 1280, height: 720, parityExpected: true },
  "step7-venus-statue-surface-normals": { name: "", kind: "levelset", file: MODELS.venus!, output: "normals", width: 1280, height: 720, parityExpected: true },
  "step7-utah-teapot-surface-asis": { name: "", kind: "levelset", file: MODELS.teapot!, output: "shaded", width: 1280, height: 720, parityExpected: false },
  "step7-venus-statue-surface-asis": { name: "", kind: "levelset", file: MODELS.venus!, output: "shaded", width: 1280, height: 720, parityExpected: false },
  "step7-utah-teapot-mesh-asis": { name: "", kind: "levelset", file: MODELS.teapot!, output: "shaded", width: 1280, height: 720, parityExpected: false },
  "step7-sphere-points-isolated": { name: "", kind: "points", file: "/fixtures/vdb-samples/sphere_points.vdb", output: "points", width: 1280, height: 720, parityExpected: true },
  "step7-sphere-points-asis": { name: "", kind: "points", file: "/fixtures/vdb-samples/sphere_points.vdb", output: "shaded", width: 1280, height: 720, parityExpected: false },
};
for (const [k, v] of Object.entries(PARITY)) v.name = k;
const def = E2E[SCENE] ?? PARITY[SCENE];
const W = Number(params.get("w") ?? def?.width ?? 192);
const H = Number(params.get("h") ?? def?.height ?? 108);
const TWIN = params.get("twin") === "1";
const TIMING = Number(params.get("timing") ?? 0);
const BACKEND: "three" | "core" = params.get("backend") === "core" ? "core" : "three";
const MUTATE = params.get("mutate") ?? "none";
const VS_THREE = params.get("vsThree") === "1";
result.backend = BACKEND;
result.mutate = MUTATE;
/** His examples page background (main.jsx). */
const HIS_BACKGROUND = new THREE.Color(0x598eff);
/** Added to the probe pass's hit positions (three's output clamps negative colours). */
const HIT_OFFSET = 1000;
/** His `levelset.js` / `points.js` displaySize. */
const DISPLAY_SIZE = 100;

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

/** three r187 readbacks are top row first, rows padded to 256 bytes. */
function tightRows<T extends Uint16Array | Float32Array>(pixels: T, w: number, h: number, channels: number): T {
  const tight = w * channels;
  const stride = h > 1 ? (pixels.length - tight) / (h - 1) : tight;
  if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${pixels.length} elements for ${w}x${h}`);
  const out = new (pixels.constructor as new (n: number) => T)(w * h * channels);
  for (let y = 0; y < h; y++) out.set(pixels.subarray(y * stride, y * stride + tight) as never, y * tight);
  return out;
}

async function readFloat(renderer: WebGPURenderer, rt: THREE.RenderTarget): Promise<Float32Array> {
  return tightRows((await renderer.readRenderTargetPixelsAsync(rt, 0, 0, rt.width, rt.height)) as unknown as Float32Array, rt.width, rt.height, 4);
}

async function readHalf(renderer: WebGPURenderer, rt: THREE.RenderTarget): Promise<Float32Array> {
  const rows = tightRows((await renderer.readRenderTargetPixelsAsync(rt, 0, 0, rt.width, rt.height)) as unknown as Uint16Array, rt.width, rt.height, 4);
  const out = new Float32Array(rows.length);
  for (let i = 0; i < rows.length; i++) out[i] = THREE.DataUtils.fromHalfFloat(rows[i]!);
  return out;
}

const toByte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);

/** Linear → display RGBA8, as his canvas output: NoToneMapping, sRGB OETF, 8-bit. */
function toDisplay(linear: Float32Array): Uint8Array {
  const out = new Uint8Array(linear.length);
  for (let i = 0; i < linear.length; i += 4) {
    for (let c = 0; c < 3; c++) out[i + c] = toByte(vdbSrgbOetf(Math.max(0, linear[i + c]!)));
    out[i + 3] = 255;
  }
  return out;
}

interface ManifestScenario {
  name: string;
  metadata?: {
    camera?: { matrixWorld: number[]; projectionMatrix: number[]; near: number; far: number };
    levelSets?: { matrixWorld: number[] }[];
    points?: { matrixWorld: number[]; pointCount?: number }[];
  };
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

/** His main.jsx camera: (-100, -11, 80) looking at the orbit target (origin), fov 60, near 0.01, far 10000. */
function hisCamera(manifest: ManifestScenario | null): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 10000);
  const c = manifest?.metadata?.camera;
  if (c) {
    result.source = "manifest";
    camera.matrixAutoUpdate = false;
    camera.matrixWorld.fromArray(c.matrixWorld);
    camera.matrix.copy(camera.matrixWorld);
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    camera.near = c.near;
    camera.far = c.far;
    camera.projectionMatrix.fromArray(c.projectionMatrix);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  } else {
    result.source = "levelset.js";
    camera.position.set(-100, -11, 80);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld(true);
  }
  return camera;
}

/** His addObject: scale the box to DISPLAY_SIZE on its longest side, centred at the origin (object = grid world). */
function hisPlacement(boxMin: V3, boxMax: V3): THREE.Matrix4 {
  const size = boxMax.map((v, k) => v - boxMin[k]!) as V3;
  const scale = DISPLAY_SIZE / Math.max(...size);
  const center = boxMin.map((v, k) => (v + boxMax[k]!) / 2) as V3;
  return new THREE.Matrix4().makeScale(scale, scale, scale).setPosition(-center[0] * scale, -center[1] * scale, -center[2] * scale);
}

/** Index box corners through a 4x4 row-major index-to-world matrix (his getPreciseWorldBbox: voxel centres). */
function worldBox(min: readonly number[], max: readonly number[], m: readonly number[]): { min: V3; max: V3 } {
  const lo: V3 = [Infinity, Infinity, Infinity];
  const hi: V3 = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < 8; k++) {
    const p = [k & 1 ? max[0]! : min[0]!, k & 2 ? max[1]! : min[1]!, k & 4 ? max[2]! : min[2]!];
    for (let r = 0; r < 3; r++) {
      const v = m[r * 4]! * p[0]! + m[r * 4 + 1]! * p[1]! + m[r * 4 + 2]! * p[2]! + m[r * 4 + 3]!;
      lo[r] = Math.min(lo[r]!, v);
      hi[r] = Math.max(hi[r]!, v);
    }
  }
  return { min: lo, max: hi };
}

async function fetchVdb(file: string) {
  const response = await fetch(file);
  if (!response.ok) throw new Error(`fetch ${file}: ${response.status} (fixtures are git-ignored; venusstatue.vdb comes from his clone into fixtures/k3/)`);
  return parseVdb(await response.arrayBuffer());
}

function pixelRay(camera: THREE.Camera, x: number, y: number): { o: V3; d: V3 } {
  const ndc = new THREE.Vector3(((x + 0.5) / W) * 2 - 1, 1 - ((y + 0.5) / H) * 2, 0.5).unproject(camera);
  const o = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
  const d = ndc.sub(o).normalize();
  return { o: [o.x, o.y, o.z], d: [d.x, d.y, d.z] };
}

const affine = (m: ArrayLike<number>, p: readonly number[], w: number): V3 => [
  m[0]! * p[0]! + m[1]! * p[1]! + m[2]! * p[2]! + m[3]! * w,
  m[4]! * p[0]! + m[5]! * p[1]! + m[6]! * p[2]! + m[7]! * w,
  m[8]! * p[0]! + m[9]! * p[1]! + m[10]! * p[2]! + m[11]! * w,
];
const angleDeg = (a: readonly number[], b: readonly number[]) => {
  const la = Math.hypot(a[0]!, a[1]!, a[2]!);
  const lb = Math.hypot(b[0]!, b[1]!, b[2]!);
  const c = (a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!) / (la * lb);
  return (Math.acos(Math.min(1, Math.max(-1, c))) * 180) / Math.PI;
};
const erode = (mask: Uint8Array): Uint8Array => {
  const out = new Uint8Array(mask.length);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      out[i] = mask[i]! & mask[i - 1]! & mask[i + 1]! & mask[i - W]! & mask[i + W]!;
    }
  }
  return out;
};

async function renderTo(renderer: WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, rt: THREE.RenderTarget): Promise<void> {
  renderer.setRenderTarget(rt);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
}

async function levelSetScene(renderer: WebGPURenderer, device: GPUDevice): Promise<Uint8Array> {
  const file = await fetchVdb(params.get("file") ?? def!.file);
  const grid = file.grids.find((g: VdbGrid) => g.valueType === "float");
  if (!grid) throw new Error(`${def!.file} has no float grid`);
  const image = buildFromVdb(grid);
  const pool = await VDBGridPool.create(device, { label: "k3-levelset" });
  const id = pool.add(image);
  let info = pool.info(id);
  const manifest = def!.output === "probe" ? null : await loadManifest(def!.name);
  const camera = hisCamera(manifest);
  const placed = manifest?.metadata?.levelSets?.[0]?.matrixWorld;
  const box = worldBox(info.indexBBox.min, info.indexBBox.max, grid.transform.matrix);
  const objectToWorld = placed ? new THREE.Matrix4().fromArray(placed) : hisPlacement(box.min, box.max);
  if (params.get("aniso") === "1") {
    // A rotated, non-uniformly scaled instance: the world normal must be W^T times the index gradient.
    objectToWorld.multiply(new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(1, 1, 0).normalize(), 0.6)).multiply(new THREE.Matrix4().makeScale(1, 1.6, 0.7));
  }
  pool.setTransform(id, objectToWorld.elements);
  pool.flush();
  info = pool.info(id);
  result.grid = { name: grid.name, indexBBox: { min: [...info.indexBBox.min], max: [...info.indexBBox.max] }, voxelSize: info.voxelSize[0], imageBytes: info.imageBytes };
  if (BACKEND === "core") return coreLevelSetScene(renderer, device, pool, id, image, info, camera, objectToWorld);

  const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["levelSet"] });
  grids.update();
  const extra = {
    ...(params.get("maxSamples") ? { maxSamples: Number(params.get("maxSamples")) } : {}),
    ...(params.get("normalStep") ? { normalStep: Number(params.get("normalStep")) } : {}),
  };
  const geometry = vdbProxyGeometry(info, { pad: 1 });

  // Unlit output: no light in their scene, so a lit three material's outgoing light is 0 + the emissive term.
  const unlit = (color: (m: VDBLevelSetMaterial) => VDBTslNode) => {
    const m = new VDBLevelSetMaterial(renderer, grids, id, { ...extra, color: 0x000000, toneMapped: false });
    m.emissiveNode = color(m);
    return m;
  };
  const shaded = new VDBLevelSetMaterial(renderer, grids, id, { color: 0xffffff, roughness: 0.5, ...extra });

  const lit = new THREE.Scene();
  lit.background = HIS_BACKGROUND.clone();
  lit.add(new Mesh(geometry, shaded));
  // His levelset.js 'sunHemi' setup.
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(100, 200, 100);
  lit.add(sun, new THREE.HemisphereLight(0x88aaff, 0x443322, 0.5));

  const half = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  let display: Uint8Array;
  if (def!.output === "normals") {
    const scene = new THREE.Scene();
    scene.add(new Mesh(geometry, unlit((m) => m.normalViewNode.mul(0.5).add(0.5))));
    renderer.setClearColor(0x000000, 1);
    await renderTo(renderer, scene, camera, half);
    display = toDisplay(await readHalf(renderer, half));
    await renderTo(renderer, scene, camera, half);
    const again = toDisplay(await readHalf(renderer, half));
    result.inPageIdentical = display.every((v, i) => v === again[i]);
  } else {
    await renderTo(renderer, lit, camera, half);
    display = toDisplay(await readHalf(renderer, half));
    await renderTo(renderer, lit, camera, half);
    const again = toDisplay(await readHalf(renderer, half));
    result.inPageIdentical = display.every((v, i) => v === again[i]);
  }

  if (def!.output === "probe" && TWIN) {
    const float = new THREE.RenderTarget(W, H, { type: THREE.FloatType, depthBuffer: true, samples: 0 });
    renderer.setClearColor(0x000000, 0);
    const normalScene = new THREE.Scene();
    // three's output keeps colours >= 0: normals as n * 0.5 + 0.5, hits offset by +HIT_OFFSET (his placement is within +-100).
    normalScene.add(new Mesh(geometry, unlit((m) => m.normalWorldNode.mul(0.5).add(0.5))));
    await renderTo(renderer, normalScene, camera, float);
    const gpuNormal = await readFloat(renderer, float);
    for (let i = 0; i < gpuNormal.length; i += 4) for (let c = 0; c < 3; c++) gpuNormal[i + c] = gpuNormal[i + c]! * 2 - 1;
    const hitScene = new THREE.Scene();
    hitScene.add(new Mesh(geometry, unlit((m) => m.hitWorldNode.add(HIT_OFFSET))));
    await renderTo(renderer, hitScene, camera, float);
    const gpuHit = await readFloat(renderer, float);
    for (let i = 0; i < gpuHit.length; i += 4) for (let c = 0; c < 3; c++) gpuHit[i + c] = gpuHit[i + c]! - HIT_OFFSET;
    result.twin = compareTwin(image, info, camera, gpuNormal, gpuHit, shaded.sdfToIndex.value, objectToWorld);
  }

  if (TIMING > 0) {
    const times: number[] = [];
    for (let i = 0; i < TIMING; i++) {
      const t0 = performance.now();
      renderer.setRenderTarget(half);
      renderer.render(lit, camera);
      renderer.setRenderTarget(null);
      await device.queue.onSubmittedWorkDone();
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    result.timing = { frames: TIMING, medianMs: times[times.length >> 1]!, minMs: times[0]!, note: `${W}x${H} shaded frame, submit to onSubmittedWorkDone; provisional (D25)` };
  }
  return display;
}

/** three's probe (world normal, world hit) of the same grid, for `&vsThree=1` (the current path on the same device). */
async function threeProbe(renderer: WebGPURenderer, pool: VDBGridPool, id: number, camera: THREE.Camera): Promise<{ normal: Float32Array; hit: Float32Array }> {
  const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["levelSet"] });
  grids.update();
  const geometry = vdbProxyGeometry(pool.info(id), { pad: 1 });
  const unlit = (color: (m: VDBLevelSetMaterial) => VDBTslNode) => {
    const m = new VDBLevelSetMaterial(renderer, grids, id, { color: 0x000000, toneMapped: false });
    m.emissiveNode = color(m);
    return m;
  };
  const float = new THREE.RenderTarget(W, H, { type: THREE.FloatType, depthBuffer: true, samples: 0 });
  renderer.setClearColor(0x000000, 0);
  const normalScene = new THREE.Scene();
  normalScene.add(new Mesh(geometry, unlit((m) => m.normalWorldNode.mul(0.5).add(0.5))));
  await renderTo(renderer, normalScene, camera, float);
  const normal = await readFloat(renderer, float);
  for (let i = 0; i < normal.length; i += 4) for (let c = 0; c < 3; c++) normal[i + c] = normal[i + c]! * 2 - 1;
  const hitScene = new THREE.Scene();
  hitScene.add(new Mesh(geometry, unlit((m) => m.hitWorldNode.add(HIT_OFFSET))));
  await renderTo(renderer, hitScene, camera, float);
  const hit = await readFloat(renderer, float);
  for (let i = 0; i < hit.length; i += 4) for (let c = 0; c < 3; c++) hit[i + c] = hit[i + c]! - HIT_OFFSET;
  return { normal, hit };
}

/** His 'sunHemi' lights as the raw pass's one light + ambient (INFO scenes only). */
const SUN_DIRECTION: [number, number, number] = [-100, -200, -100];
const HEMI_AMBIENT = (() => {
  const sky = new THREE.Color(0x88aaff);
  const ground = new THREE.Color(0x443322);
  return [0, 1, 2].map((k) => 0.5 * 0.5 * ([sky.r, sky.g, sky.b][k]! + [ground.r, ground.g, ground.b][k]!)) as [number, number, number];
})();
/** The probe's shade check: a coloured albedo under a coloured light and ambient. */
const PROBE_COLOR: [number, number, number] = [0.9, 0.7, 0.5];
const PROBE_LIGHT = { direction: [-0.4, -0.8, -0.45] as [number, number, number], color: [1, 0.95, 0.9] as [number, number, number] };
const PROBE_AMBIENT: [number, number, number] = [0.15, 0.2, 0.25];

/**
 * K6.6 `?backend=core`: the level set through the core's raw pass (G-buffer and shade pipelines, no three
 * objects), compared with the K6.6 surface twins; `&vsThree=1` also against three's material.
 */
async function coreLevelSetScene(
  renderer: WebGPURenderer,
  device: GPUDevice,
  pool: VDBGridPool,
  id: number,
  image: Uint32Array,
  info: ReturnType<VDBGridPool["info"]>,
  camera: THREE.PerspectiveCamera,
  objectToWorld: THREE.Matrix4,
): Promise<Uint8Array> {
  // three's renderer switches a camera to WebGPU's depth range before it renders; the raw pass needs the same.
  if (camera.coordinateSystem !== THREE.WebGPUCoordinateSystem) {
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
  }
  const sdfToIndex = 1 / info.voxelSize[0]!;
  const frame = { view: camera.matrixWorldInverse.elements, projection: camera.projectionMatrix.elements, viewport: [0, 0, W, H] as const };
  const draw = {
    grid: id,
    sdfToIndex,
    ...(params.get("maxSamples") ? { maxSamples: Number(params.get("maxSamples")) } : {}),
    ...(params.get("normalStep") ? { normalStep: Number(params.get("normalStep")) } : {}),
  };
  const g = await renderLevelSetCore(device, pool, "gbuffer", frame, [draw], W, H);
  const again = await renderLevelSetCore(device, pool, "gbuffer", frame, [draw], W, H);
  result.inPageIdentical = g.color.every((v, i) => Object.is(v, again.color[i])) && g.depth.every((v, i) => v === again.depth[i]);

  // The G-buffer as the three probe's arrays: (normal, mask) and (world hit, 1), the hit from the pixel's ray + t.
  const projectionInverse = camera.projectionMatrixInverse.elements;
  const cameraWorld = camera.matrixWorld.elements;
  const viewProjection = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
  const gpuNormal = new Float32Array(W * H * 4);
  const gpuHit = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const n = g.color.subarray(i * 4, i * 4 + 3);
      if (!(Math.hypot(n[0]!, n[1]!, n[2]!) > 0.5)) continue;
      const ray = levelSetViewRay(levelSetPixelUv([x + 0.5, y + 0.5], [0, 0, W, H]), projectionInverse, cameraWorld, false);
      const t = g.color[i * 4 + 3]!;
      gpuNormal.set([n[0]!, n[1]!, n[2]!, 1], i * 4);
      gpuHit.set([ray.origin[0] + ray.dir[0] * t, ray.origin[1] + ray.dir[1] * t, ray.origin[2] + ray.dir[2] * t, 1], i * 4);
    }
  }

  let display: Uint8Array;
  if (def!.output === "normals") {
    // His isolate: 'normals' encoding: view-space normal * 0.5 + 0.5, black where nothing was hit.
    const linear = new Float32Array(W * H * 4);
    const v = new THREE.Vector3();
    for (let i = 0; i < W * H; i++) {
      linear[i * 4 + 3] = 1;
      if (!gpuNormal[i * 4 + 3]) continue;
      v.fromArray(gpuNormal, i * 4).transformDirection(camera.matrixWorldInverse);
      linear.set([v.x * 0.5 + 0.5, v.y * 0.5 + 0.5, v.z * 0.5 + 0.5], i * 4);
    }
    display = toDisplay(linear);
  } else {
    const asis = def!.output === "shaded";
    const lit = { ...frame, light: asis ? { direction: SUN_DIRECTION, color: [1, 1, 1] as [number, number, number] } : PROBE_LIGHT, ambient: asis ? HEMI_AMBIENT : PROBE_AMBIENT };
    const shade = await renderLevelSetCore(device, pool, "shade", lit, [{ ...draw, color: asis ? [1, 1, 1] : PROBE_COLOR }], W, H);
    let shadeMax = 0;
    if (!asis) {
      // The shade output against the Lambert + ambient of the G-buffer's normal (the same trace, another pipeline).
      const L = new THREE.Vector3(...PROBE_LIGHT.direction).normalize();
      for (let i = 0; i < W * H; i++) {
        if (!gpuNormal[i * 4 + 3] || shade.color[i * 4 + 3]! !== 1) continue;
        const ndl = Math.max(0, -(gpuNormal[i * 4]! * L.x + gpuNormal[i * 4 + 1]! * L.y + gpuNormal[i * 4 + 2]! * L.z) * (MUTATE === "shade-sign" ? -1 : 1));
        for (let c = 0; c < 3; c++) {
          const expected = PROBE_COLOR[c]! * (PROBE_AMBIENT[c]! + PROBE_LIGHT.color[c]! * ndl);
          shadeMax = Math.max(shadeMax, Math.abs(shade.color[i * 4 + c]! - expected));
        }
      }
    }
    const linear = shade.color;
    for (let i = 0; i < W * H; i++) {
      if (linear[i * 4 + 3] === 1) continue;
      linear.set(asis ? [HIS_BACKGROUND.r, HIS_BACKGROUND.g, HIS_BACKGROUND.b, 1] : [0, 0, 0, 1], i * 4);
    }
    display = toDisplay(linear);
    if (def!.output === "probe" && TWIN) {
      const stats = compareTwin(image, info, camera, gpuNormal, gpuHit, sdfToIndex, objectToWorld, true);
      // frag_depth vs the twin's NDC depth of the GPU's own hit (isolates the depth formula from the hit noise).
      const vp = new THREE.Matrix4().fromArray(viewProjection);
      let depthMax = 0;
      for (let i = 0; i < W * H; i++) {
        if (!gpuNormal[i * 4 + 3]) continue;
        const p = gpuHit.subarray(i * 4, i * 4 + 3);
        const twin = MUTATE === "depth-linear" ? new THREE.Vector4(p[0]!, p[1]!, p[2]!, 1).applyMatrix4(vp).z : levelSetNdcDepth([...p], viewProjection);
        depthMax = Math.max(depthMax, Math.abs(g.depth[i]! - twin));
      }
      result.twin = { ...stats, depthMaxAbs: depthMax, shadeMaxAbs: shadeMax };
    }
  }

  if (VS_THREE) {
    const three = await threeProbe(renderer, pool, id, camera);
    const worldPerVoxel = info.voxelSize[0]! * objectToWorld.getMaxScaleOnAxis();
    let coreHits = 0;
    let threeHits = 0;
    let both = 0;
    let union = 0;
    let hitMax = 0;
    let normalMax = 0;
    for (let i = 0; i < W * H; i++) {
      const a = gpuNormal[i * 4 + 3]! > 0.5;
      const b = three.normal[i * 4 + 3]! > 0.5;
      coreHits += a ? 1 : 0;
      threeHits += b ? 1 : 0;
      if (a || b) union++;
      if (!(a && b)) continue;
      both++;
      hitMax = Math.max(hitMax, Math.hypot(gpuHit[i * 4]! - three.hit[i * 4]!, gpuHit[i * 4 + 1]! - three.hit[i * 4 + 1]!, gpuHit[i * 4 + 2]! - three.hit[i * 4 + 2]!) / worldPerVoxel);
      normalMax = Math.max(normalMax, angleDeg([...gpuNormal.subarray(i * 4, i * 4 + 3)], [...three.normal.subarray(i * 4, i * 4 + 3)]));
    }
    result.vsThree = { coreHits, threeHits, iou: union ? both / union : 1, hitMaxVoxels: hitMax, normalMaxDeg: normalMax };
  }
  return display;
}

function compareTwin(
  image: Uint32Array,
  info: { worldToIndex: Float64Array; voxelSize: readonly number[] },
  camera: THREE.Camera,
  gpuNormal: Float32Array,
  gpuHit: Float32Array,
  sdfToIndex: number,
  objectToWorld: THREE.Matrix4,
  core = false,
): Stats {
  const t0 = performance.now();
  // K6.6 core: the surface twins (pixel uv, view ray, surface), with ?mutate= breaking one term.
  const projectionInverse = camera.projectionMatrixInverse.elements;
  const cameraWorld = camera.matrixWorld.elements;
  const viewProjection = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
  const gpuMask = new Uint8Array(W * H);
  const cpuMask = new Uint8Array(W * H);
  const cpuNormal = new Float32Array(W * H * 3);
  const cpuHit = new Float32Array(W * H * 3);
  let samples = 0;
  // World units per index voxel (his placement scales the grid uniformly).
  const worldPerVoxel = info.voxelSize[0]! * objectToWorld.getMaxScaleOnAxis();
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      gpuMask[i] = gpuNormal[i * 4 + 3]! > 0.5 ? 1 : 0;
      if (core) {
        const uv = levelSetPixelUv(MUTATE === "pixel-corner" ? [x, y] : [x + 0.5, y + 0.5], [0, 0, W, H]);
        const ray = levelSetViewRay(uv, projectionInverse, cameraWorld, MUTATE === "ray-ortho");
        const s = levelSetSurface(image, ray, info.worldToIndex, viewProjection, {
          sdfToIndex,
          maxSamples: Number(params.get("maxSamples") ?? 512),
          normalStep: Number(params.get("normalStep") ?? 1),
        });
        samples += s.samples;
        if (!s.hit) continue;
        cpuMask[i] = 1;
        cpuHit.set(s.world, i * 3);
        const n = MUTATE === "normal-view" ? new THREE.Vector3(...s.normal).transformDirection(camera.matrixWorldInverse).toArray() : s.normal;
        cpuNormal.set(n, i * 3);
        continue;
      }
      const { o, d } = pixelRay(camera, x, y);
      const oi = affine(info.worldToIndex, o, 1);
      const di = affine(info.worldToIndex, d, 0);
      const h = levelSetHit(image, oi, di, [0, 1e30], { sdfToIndex, maxSamples: Number(params.get("maxSamples") ?? 512) });
      samples += h.samples;
      if (!h.hit) continue;
      cpuMask[i] = 1;
      const p = [o[0] + d[0] * h.t, o[1] + d[1] * h.t, o[2] + d[2] * h.t];
      cpuHit.set(p, i * 3);
      const n = levelSetNormal(image, affine(info.worldToIndex, p, 1), info.worldToIndex, Number(params.get("normalStep") ?? 1));
      cpuNormal.set(n, i * 3);
    }
  }
  const twinMs = performance.now() - t0;
  if (params.get("debug")) {
    const c = (Math.floor(H / 2) * W + Math.floor(W / 2));
    (result as unknown as Record<string, unknown>).debug = {
      gpuNormal: [...gpuNormal.subarray(c * 4, c * 4 + 4)],
      gpuHit: [...gpuHit.subarray(c * 4, c * 4 + 4)],
      cpuNormal: [...cpuNormal.subarray(c * 3, c * 3 + 3)],
      cpuHit: [...cpuHit.subarray(c * 3, c * 3 + 3)],
      pixel: [Math.floor(W / 2) + 0.5, Math.floor(H / 2) + 0.5],
      gpuProjected: (() => {
        const q = project(camera, new THREE.Vector3(gpuHit[c * 4]!, gpuHit[c * 4 + 1]!, gpuHit[c * 4 + 2]!));
        return [q.x, q.y];
      })(),
      cpuProjected: (() => {
        const q = project(camera, new THREE.Vector3(cpuHit[c * 3]!, cpuHit[c * 3 + 1]!, cpuHit[c * 3 + 2]!));
        return [q.x, q.y];
      })(),
    };
  }
  let gpuHits = 0;
  let cpuHits = 0;
  let both = 0;
  let union = 0;
  let hitSum = 0;
  let hitMax = 0;
  const angles: number[] = [];
  const ge = erode(gpuMask);
  const ce = erode(cpuMask);
  let interiorMax = 0;
  for (let i = 0; i < W * H; i++) {
    gpuHits += gpuMask[i]!;
    cpuHits += cpuMask[i]!;
    if (gpuMask[i] || cpuMask[i]) union++;
    if (!(gpuMask[i] && cpuMask[i])) continue;
    both++;
    const g = [gpuHit[i * 4]!, gpuHit[i * 4 + 1]!, gpuHit[i * 4 + 2]!];
    const c = [cpuHit[i * 3]!, cpuHit[i * 3 + 1]!, cpuHit[i * 3 + 2]!];
    const dv = Math.hypot(g[0]! - c[0]!, g[1]! - c[1]!, g[2]! - c[2]!) / worldPerVoxel;
    hitSum += dv;
    hitMax = Math.max(hitMax, dv);
    const a = angleDeg([gpuNormal[i * 4]!, gpuNormal[i * 4 + 1]!, gpuNormal[i * 4 + 2]!], [cpuNormal[i * 3]!, cpuNormal[i * 3 + 1]!, cpuNormal[i * 3 + 2]!]);
    angles.push(a);
    if (ge[i] && ce[i]) interiorMax = Math.max(interiorMax, a);
  }
  angles.sort((a, b) => a - b);
  const stats: Stats = {
    pixels: W * H,
    gpuHits,
    cpuHits,
    both,
    iou: union ? both / union : 1,
    hitMeanVoxels: both ? hitSum / both : 0,
    hitMaxVoxels: hitMax,
    normalMeanDeg: angles.length ? angles.reduce((s, a) => s + a, 0) / angles.length : 0,
    normalP99Deg: angles.length ? angles[Math.min(angles.length - 1, Math.floor(angles.length * 0.99))]! : 0,
    normalMaxDeg: angles.length ? angles[angles.length - 1]! : 0,
    normalInteriorMaxDeg: interiorMax,
    twinMs,
    avgSamples: samples / (W * H),
  };
  if (def!.name === "sphere") {
    // ls_sphere is centred at index (0, 0, 0) (its box is symmetric): the analytic index normal is q / |q| at the
    // index hit q, and the world normal is W^T q (W = world-to-index linear part), under any instance transform.
    const m = info.worldToIndex;
    let sum = 0;
    let max = 0;
    let rMin = Infinity;
    let rMax = -Infinity;
    for (let i = 0; i < W * H; i++) {
      if (!gpuMask[i]) continue;
      const q = affine(m, [gpuHit[i * 4]!, gpuHit[i * 4 + 1]!, gpuHit[i * 4 + 2]!], 1);
      const expected = [0, 1, 2].map((k) => m[k]! * q[0] + m[4 + k]! * q[1] + m[8 + k]! * q[2]);
      const a = angleDeg([gpuNormal[i * 4]!, gpuNormal[i * 4 + 1]!, gpuNormal[i * 4 + 2]!], expected);
      sum += a;
      max = Math.max(max, a);
      const r = Math.hypot(q[0], q[1], q[2]);
      rMin = Math.min(rMin, r);
      rMax = Math.max(rMax, r);
    }
    stats.analytic = { meanDeg: gpuHits ? sum / gpuHits : 0, maxDeg: max, radiusSpreadVoxels: rMax - rMin };
  }
  return stats;
}

/** Projects a world point to (x, y) pixel coordinates (top row 0) and view depth. */
function project(camera: THREE.Camera, p: THREE.Vector3): { x: number; y: number; depth: number } {
  const view = p.clone().applyMatrix4(camera.matrixWorldInverse);
  const ndc = p.clone().project(camera);
  return { x: ((ndc.x + 1) / 2) * W, y: ((1 - ndc.y) / 2) * H, depth: -view.z };
}

async function pointsScene(renderer: WebGPURenderer, device: GPUDevice): Promise<Uint8Array> {
  const manifest = def!.kind === "points" && def!.output !== "points" ? await loadManifest(def!.name) : def!.parityExpected ? await loadManifest(def!.name) : null;
  let points: VDBPoints;
  let camera: THREE.PerspectiveCamera;
  if (def!.kind === "points-color") {
    // 8x8 points on a plane, 1 unit apart, distinct colours; a camera 12 units in front.
    const n = 8;
    const positions = new Float32Array(n * n * 3);
    const colors = new Float32Array(n * n * 3);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const k = j * n + i;
        positions.set([i - (n - 1) / 2, j - (n - 1) / 2, 0], k * 3);
        colors.set([(i + 1) / n, (j + 1) / n, ((i + j) % 3) / 2], k * 3);
      }
    }
    points = new VDBPoints({ positions, colors, spacing: 1 }, { size: 0.5 });
    camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 100);
    camera.position.set(0, 0, 12);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld(true);
    result.source = "levelset.js";
  } else {
    const file = await fetchVdb(params.get("file") ?? def!.file);
    const grid = file.grids.find((g: VdbGrid) => g.valueType === "ptdataidx32") as VdbPointDataGrid | undefined;
    if (!grid) throw new Error(`${def!.file} has no point data grid`);
    const data = vdbPointRenderData(grid);
    points = new VDBPoints(data);
    const placed = manifest?.metadata?.points?.[0]?.matrixWorld;
    const box = worldBox(grid.indexBBox!.min, grid.indexBBox!.max, grid.transform.matrix);
    const m = placed ? new THREE.Matrix4().fromArray(placed) : hisPlacement(box.min, box.max);
    points.matrixAutoUpdate = false;
    points.matrix.copy(m);
    camera = hisCamera(manifest);
    result.points = { count: data.count, totalCount: data.totalCount, spacing: data.spacing, covered: 0, centresCovered: 0, centresTotal: 0, stray: 0 };
  }
  const scene = new THREE.Scene();
  const asis = def!.output === "shaded";
  if (asis) scene.background = HIS_BACKGROUND.clone();
  scene.add(points);
  scene.updateMatrixWorld(true);
  (points.material as unknown as { toneMapped: boolean }).toneMapped = false;
  renderer.setClearColor(0x000000, 1);
  let linear: Float32Array;
  let drawn = (points as unknown as { count: number }).count;
  if (BACKEND === "core") {
    // K6.6: the raw sprite pass (core WGSL), the same points, camera and placement; the pixel scale from the
    // projection (vdbPointPixelScale), checked against VDBPoints' own.
    if (camera.coordinateSystem !== THREE.WebGPUCoordinateSystem) {
      camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
      camera.updateProjectionMatrix();
    }
    const pixelScale = vdbPointPixelScale(points.diameter, camera.projectionMatrix.elements, H, points.matrixWorld.elements);
    points.updatePixelScale(renderer as never, camera);
    result.pixelScale = { core: pixelScale, three: points.pixelScale.value };
    points.pixelScale.value = pixelScale;
    const data = { positions: points.positionAttribute.array as Float32Array, colors: (points.colorAttribute?.array as Float32Array | undefined) ?? null, count: points.pointCount };
    const frame = { view: camera.matrixWorldInverse.elements, projection: camera.projectionMatrix.elements, model: points.matrixWorld.elements, viewport: [0, 0, W, H] as const, pixelScale };
    const first = await renderPointsCore(device, data, frame, W, H);
    const second = await renderPointsCore(device, data, frame, W, H);
    linear = first.rgba;
    drawn = first.count;
    result.inPageIdentical = linear.every((v, i) => Object.is(v, second.rgba[i]));
    if (asis) {
      for (let i = 0; i < W * H; i++) if (!(linear[i * 4]! + linear[i * 4 + 1]! + linear[i * 4 + 2]! > 0)) linear.set([HIS_BACKGROUND.r, HIS_BACKGROUND.g, HIS_BACKGROUND.b, 1], i * 4);
    }
  } else {
    const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
    await renderTo(renderer, scene, camera, rt);
    linear = await readHalf(renderer, rt);
    await renderTo(renderer, scene, camera, rt);
    const again = await readHalf(renderer, rt);
    result.inPageIdentical = linear.every((v, i) => v === again[i]);
  }
  if (asis) return toDisplay(linear);

  const covered = (x: number, y: number) => {
    const i = (y * W + x) * 4;
    return linear[i]! + linear[i + 1]! + linear[i + 2]! > 0.05;
  };
  const pos = points.positionAttribute.array as Float32Array;
  const p = new THREE.Vector3();
  if (def!.kind === "points-color") {
    const colors = points.colorAttribute!.array as Float32Array;
    const out = { checked: 0, bad: 0 } as NonNullable<Result["colors"]>;
    for (let k = 0; k < points.pointCount; k++) {
      p.fromArray(pos, k * 3).applyMatrix4(points.matrixWorld);
      const q = project(camera, p);
      const x = Math.floor(q.x);
      const y = Math.floor(q.y);
      const i = (y * W + x) * 4;
      out.checked++;
      const err = Math.max(...[0, 1, 2].map((c) => Math.abs(linear[i + c]! - colors[k * 3 + c]!)));
      if (err > 2e-3) {
        out.bad++;
        out.firstBad ??= { k, x, y, gpu: [linear[i], linear[i + 1], linear[i + 2]], expected: [...colors.subarray(k * 3, k * 3 + 3)] };
      }
    }
    result.colors = out;
  } else {
    // Each sprite is a quad of max(pixelScale / depth, 1) px around its projected centre with the alpha-tested disc
    // map: every pixel whose centre lies inside the disc (shrunk by a margin) must be covered, and every covered
    // pixel's centre must lie inside some sprite's quad (grown by the margin): no strays.
    const near = new Uint8Array(W * H);
    let centresCovered = 0;
    let centresTotal = 0;
    for (let k = 0; k < points.pointCount; k++) {
      p.fromArray(pos, k * 3).applyMatrix4(points.matrixWorld);
      const q = project(camera, p);
      if (q.depth <= 0 || q.x < -8 || q.y < -8 || q.x >= W + 8 || q.y >= H + 8) continue;
      const r = Math.max(points.pixelScale.value / q.depth, 1) / 2;
      for (let y = Math.max(0, Math.floor(q.y - r - 1)); y <= Math.min(H - 1, Math.ceil(q.y + r + 1)); y++) {
        for (let x = Math.max(0, Math.floor(q.x - r - 1)); x <= Math.min(W - 1, Math.ceil(q.x + r + 1)); x++) {
          const dist = Math.hypot(x + 0.5 - q.x, y + 0.5 - q.y);
          // The disc map is mipmapped (his texture defaults), so a sprite of a few pixels reads a blurred disc
          // whose alpha-0.5 edge moves toward the quad: strays are judged against the sprite's square quad.
          // K6.6 core: the raw pass cuts an analytic disc, so strays are judged against the disc itself; the margin
          // covers the rasterizer's vertex snapping (SwiftShader: measured edges up to ~0.03 px off the f64 disc).
          const margin = BACKEND === "core" ? 0.05 : Math.max(0.05, (2 * r) / 64);
          const inside = BACKEND === "core" ? dist <= r + margin : Math.max(Math.abs(x + 0.5 - q.x), Math.abs(y + 0.5 - q.y)) <= r + margin;
          if (inside) near[y * W + x] = 1;
          if (dist <= r - margin) {
            centresTotal++;
            if (covered(x, y)) centresCovered++;
          }
        }
      }
    }
    let coveredCount = 0;
    let stray = 0;
    let firstStray: { x: number; y: number; nearest: number; r: number } | undefined;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (!covered(x, y)) continue;
        coveredCount++;
        if (near[y * W + x]) continue;
        stray++;
        if (!firstStray) {
          let best = { nearest: Infinity, r: 0 };
          for (let k = 0; k < points.pointCount; k++) {
            const q = project(camera, p.fromArray(pos, k * 3).applyMatrix4(points.matrixWorld));
            const dist = Math.hypot(x + 0.5 - q.x, y + 0.5 - q.y);
            if (dist < best.nearest) best = { nearest: dist, r: Math.max(points.pixelScale.value / q.depth, 1) / 2 };
          }
          firstStray = { x, y, ...best };
        }
      }
    }
    Object.assign(result.points!, { covered: coveredCount, centresCovered, centresTotal, stray, firstStray, drawn });
  }
  return toDisplay(linear);
}

async function main(): Promise<void> {
  root.textContent = "";
  if (!def) throw new Error(`unknown ?scene=${SCENE}; known: ${[...Object.keys(E2E), ...Object.keys(PARITY)].join(", ")}`);
  result.scene = def.name;
  result.kind = def.kind === "levelset" ? "levelset" : "points";
  result.parityExpected = def.parityExpected;
  if (!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  const renderer = new WebGPURenderer({ device });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  if (def.kind === "sprite") {
    result.sprite = await spriteProbe(device, MUTATE as SpriteMutation);
    result.width = 1;
    result.height = 1;
    log(`sprite probe${MUTATE !== "none" ? ` (mutate ${MUTATE})` : ""}: ${JSON.stringify(result.sprite)}`, result.sprite.over === 0 ? "ok" : "fail");
    return;
  }
  const display = def.kind === "levelset" ? await levelSetScene(renderer, device) : await pointsScene(renderer, device);
  result.width = W;
  result.height = H;
  result.rgba = toBase64(display);
  let nonBackground = 0;
  const bg = def.output === "shaded" ? [HIS_BACKGROUND.r, HIS_BACKGROUND.g, HIS_BACKGROUND.b].map((v) => toByte(vdbSrgbOetf(v))) : [0, 0, 0];
  for (let i = 0; i < display.length; i += 4) if (display[i] !== bg[0] || display[i + 1] !== bg[1] || display[i + 2] !== bg[2]) nonBackground++;
  result.nonBackground = nonBackground;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(display), W, H), 0, 0);
  root.append(canvas);
  log(`${def.name}: ${nonBackground} px; ${JSON.stringify(result.twin ?? result.points ?? result.colors ?? {})}`, "ok");
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
