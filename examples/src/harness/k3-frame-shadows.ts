/**
 * K3.6 harness (R1 §6 Step 6): mjurczyk's volume shadows on meshes (`settings.castShadows`) and the emission light
 * (`VDBFrameEmissionLight`) on three r187, checked against the CPU twins (`@pmndrs/vdb/tools` tools/frame/shadows.ts,
 * emission.ts, blackbody.ts) from read-back render targets.
 *
 * `?case=shadow` (default): the committed CC0 cloud above a ground plane (`ShadowMaterial` opacity 0.5, or
 * `?ground=standard`), a sun (`?light=spot`: a spot light) with `castShadow`. Frames: shadows on, shadows off
 * (`castShadows` false), the volume alone (ground hidden: its screen coverage). Checks, `?twins=1`:
 * - `mapTwin`: the light's Beer shadow map vs `vdbFrameShadowMapTexel` marched over the read-back shadow grid;
 * - `groundTwin`: every visible ground pixel's transmittance (from the ShadowMaterial alpha) vs `vdbFrameShadowReceiver`
 *   on the read-back map (the exact chain), and vs an independent CPU-integrated transmittance (a 512-step march toward
 *   the light through the read-back grid);
 * - `nowhereElse`: pixels the CPU puts outside every shadow are byte-identical to the castShadows-off frame.
 * `?source=sparse` (the same texels as a NanoVDB grid), `?lighting=frame`, `?cascades=N`, `?mapSize=`, `?steps=`.
 *
 * `?case=emission`: the committed CC0 ground explosion frame 10 (EmberGen; his explosion settings: Z-up, 100 units,
 * flames ÷ 6 as emission, 0–3000 K, his 'planck' spectrum through `vdbFrameBlackbodyEmission`, intensity 15) over a
 * white `MeshLambertMaterial` ground lit only by a `VDBFrameEmissionLight` (intensity 25). Checks, `?twins=1`: the
 * direction pass vs `vdbFrameEmissionDirectionTexel` (CPU samplers on the dense texels), the reduce pass, the probe pass
 * vs `vdbFrameEmissionProbe`, and every ground pixel vs `vdbFrameEmissionLookup` → E(n) × colour / π (Lambert);
 * the falloff along a line on the ground. `?emit=0` hides the light. `?source=sparse`, `?probes=`, `?spectrum=white`,
 * `?emitter=blob` (a smooth synthetic Gaussian emitter: the strict per-direction twin; SwiftShader's ~12-bit division
 * makes the explosion's sharp flames alias by several % per direction, so it gates on the 16-direction mean).
 *
 * `?timing=N` (hardware): median ms of N forced volume shadow map redraws (pass timer; D25: provisional).
 * Results: `window.__VDB_K3_SHADOWS__` (e2e/k3-frame-shadows.spec.ts).
 */
import * as THREE from "three";
import { NodeMaterial, WebGPURenderer } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBFrameEmissionLight, VDBFrameRenderer, VDBFrameVolume, VDBGridNode, vdbDenseTextures, vdbFrameBlackbodyEmission, vdbSparseSource, vdbVolumeNodes, type VDBFrameShadowEntry, type VDBThreeVolumeSource } from "../../../packages/vdb/src/three/internal.js";
import {
  readWvdb,
  sampleWvdbDenseTexture,
  vdbFrameAtlasBilinear,
  vdbFrameBlackbodyColor,
  vdbFrameEmissionDirectionTexel,
  vdbFrameEmissionIrradiance,
  vdbFrameEmissionLookup,
  vdbFrameEmissionProbe,
  vdbFrameEmissionReduce,
  vdbFrameGridCell,
  vdbFrameGridUvw,
  vdbFrameSampleGrid,
  vdbFrameShadowCameraDepth,
  vdbFrameShadowCascadeIndex,
  vdbFrameShadowCascadeUv,
  vdbFrameShadowMapTexel,
  vdbFrameShadowPassRay,
  vdbFrameShadowReceiver,
  wvdbDenseToNvdb,
  wvdbDenseVolume,
  writeWvdb,
  vdbSrgbOetf,
  type VDBFrameAtlas,
  type VDBFrameGridLayout,
  type WvdbDenseVolume,
} from "../../../packages/vdb/src/tools/internal.js";
import { accumulate, finish, gridLayout, newStats, readAtlas, type TwinStats } from "./k3-frame-lighting.js";
import { installPassTimer } from "./k3-pass-timer.js";
import cloudUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";
import explosionUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/ground_explosion_0010.wvdb?url";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;
type V3 = [number, number, number];
type V4 = [number, number, number, number];

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  case?: string;
  width?: number;
  height?: number;
  rgba?: string;
  /** The castShadows-off frame (shadow case) or the light-off frame (emission), display RGBA8. */
  rgbaOff?: string;
  inPageIdentical?: boolean;
  warnings: string[];
  uncaptured: string[];
  settings?: Record<string, unknown>;
  sourceKind?: string;
  // shadow case
  entry?: { size: number; cascades: number; breaks: number[]; far: number[]; drawn: number; waiting: number } | null;
  mapTwin?: TwinStats & { texels: number; shadowed: number; maxDepth: number };
  groundTwin?: TwinStats & { ground: number; shadowed: number; meanT: number; minT: number; strictOver: number; maxStrict: number };
  independent?: { compared: number; meanAbs: number; maxAbs: number; within05: number; within10: number };
  nowhereElse?: { unshadowed: number; changed: number; shadowed: number; darker: number };
  shadowStats?: { groundPixels: number; covered: number; shadowedPixels: number; meanDarkening: number };
  mapTau?: number[];
  debugRgba?: string;
  // emission case
  emitters?: number;
  probeGrid?: { resolution: number; count: number; min: number[][]; size: number[][]; center: number[] };
  directionTwin?: TwinStats & { texels: number };
  sourceProbe?: TwinStats & { lod: TwinStats };
  directionMean?: { cpu: number[]; gpu: number[]; maxRel: number; centerDistance: number };
  reduceTwin?: TwinStats;
  probeTwin?: TwinStats & { probes: number };
  irradianceTwin?: TwinStats & { ground: number; lit: number };
  falloff?: { distance: number; measured: number; twin: number; pointLight: number }[];
  emitterLight?: { intensity: number[]; center: number[] };
  probeValues?: number[];
  lightOff?: { groundPixels: number; nonBlack: number };
  timing?: { provisional: true; note: string; frames: number; mapMs: number | null; perPieceMs: number | null; samples: number[] } | null;
}

declare global {
  interface Window {
    __VDB_K3_SHADOWS__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const CASE = (params.get("case") ?? "shadow") as "shadow" | "emission";
const W = Number(params.get("w") ?? 160);
const H = Number(params.get("h") ?? 120);
const SOURCE_KIND = (params.get("source") ?? "dense") as "dense" | "sparse";
const LIGHTING = (params.get("lighting") ?? "volume") as "volume" | "frame";
const LIGHT = (params.get("light") ?? "dir") as "dir" | "spot";
const GROUND = (params.get("ground") ?? "shadow") as "shadow" | "standard";
const CAST = params.get("cast") !== "0";
const CASCADES = Number(params.get("cascades") ?? 1);
const MAP_SIZE = Number(params.get("mapSize") ?? 512);
const STEPS = Number(params.get("steps") ?? 32);
const TWINS = params.get("twins") === "1";
const EMIT = params.get("emit") !== "0";
const PROBES = Number(params.get("probes") ?? 16);
const SPECTRUM = params.get("spectrum") ?? "planck";
const TIMING = Number(params.get("timing") ?? 0);
const MIPS = params.get("mips") !== "0";
const BLOB = params.get("emitter") === "blob";
const BACKGROUND = new THREE.Color(CASE === "emission" ? 0x000000 : 0x598eff);
const SHADOW_OPACITY = 0.5;

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K3_SHADOWS__ = result;

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

/** Linear frame, rows top first (from readAtlas's bottom-first rows). */
async function readLinear(renderer: WebGPURenderer, rt: THREE.RenderTarget): Promise<Float32Array> {
  const a = await readAtlas(renderer, rt, 4);
  const out = new Float32Array(a.data.length);
  const row = W * 4;
  for (let y = 0; y < H; y++) out.set((a.data as Float32Array).subarray((H - 1 - y) * row, (H - y) * row), y * row);
  return out;
}

const toByte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);

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

/** The world point of pixel (x, y top) on the plane y = planeY, or null. */
function groundPoint(camera: THREE.PerspectiveCamera, x: number, y: number, planeY: number, half: number, sub: [number, number] = [0.5, 0.5]): V3 | null {
  const ndc = new THREE.Vector3(((x + sub[0]) / W) * 2 - 1, 1 - ((y + sub[1]) / H) * 2, 0.5).unproject(camera);
  const origin = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
  const dir = ndc.sub(origin).normalize();
  if (Math.abs(dir.y) < 1e-9) return null;
  const t = (planeY - origin.y) / dir.y;
  if (!(t > 0)) return null;
  const p = origin.addScaledVector(dir, t);
  if (Math.abs(p.x) > half || Math.abs(p.z) > half) return null;
  return [p.x, p.y, p.z];
}

const v3 = (v: THREE.Vector3): V3 => [v.x, v.y, v.z];

async function makeSource(renderer: WebGPURenderer, device: GPUDevice, dense: WvdbDenseVolume, emission: boolean): Promise<VDBThreeVolumeSource> {
  if (SOURCE_KIND !== "sparse") return vdbDenseTextures(dense);
  // The same texels as NanoVDB FLOAT grids (voxel for voxel), the K3.2 default LOD; the emission grid shares the texture box
  const grid = wvdbDenseToNvdb(dense);
  const pool = await VDBGridPool.create(device, { label: "k3-shadows" });
  const id = pool.add(grid.image);
  const flames = emission ? wvdbDenseToNvdb(dense, { channel: "emission" }) : null;
  const emissionId = flames ? pool.add(flames.image) : undefined;
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "fastTrilinear"] });
  return vdbSparseSource(grids, emissionId === undefined ? { density: id } : { density: id, emission: emissionId }, {
    lod: "coarse-pyramid",
    images: { density: grid.image },
    indexBox: grid.textureIndexBox,
    ...(flames ? { auxIndexBoxes: { emission: flames.textureIndexBox } } : {}),
    stats: { meanNonZero: dense.meanDensity, max: dense.maxDensity, activeVoxels: dense.activeTexels },
    ...(dense.bbox ? { bbox: dense.bbox as [V3, V3] } : {}),
  });
}

// ── shadow case ──────────────────────────────────────────────────────────────────────────────────────

/** The light's Beer shadow map vs the CPU pass over the read-back grid (every `step`-th texel of each drawn tile). */
async function shadowMapTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, entry: VDBFrameShadowEntry, step: number): Promise<NonNullable<Result["mapTwin"]>> {
  const map = await readAtlas(renderer, entry.target, 4);
  const grid = await readAtlas(renderer, fr.shadowDensityTarget!, 2);
  const layout = gridLayout(fr);
  const last = layout.count - 1;
  const min = layout.gridMin[last]!;
  const size = layout.gridSize[last]!;
  const box = { min, max: [min[0] + size[0], min[1] + size[1], min[2] + size[2]] as V3 };
  const extinctionAt = (p: V3) => vdbFrameSampleGrid(grid, layout, last, vdbFrameGridUvw(layout, last, p))[0]!;
  const steps = fr.volumeShadows.getSteps();
  const s = entry.size;
  const stats = Object.assign(newStats(), { texels: 0, shadowed: 0, maxDepth: 0 });
  const tiles = entry.cascades
    ? entry.cascades.slice(0, entry.drawn.cascadeCount).map((c, i) => ({ camera: c.camera as THREE.Camera, offset: i * s, ray: { origin: v3(c.drawn.origin), forward: v3(entry.drawn.forward), perspective: 0, far: c.drawn.far } }))
    : [{ camera: entry.light.shadow.camera as THREE.Camera, offset: 0, ray: { origin: v3(entry.drawn.origin), forward: v3(entry.drawn.forward), perspective: entry.drawn.perspective, far: entry.drawn.far } }];
  for (const tile of tiles) {
    const cam = tile.camera as THREE.PerspectiveCamera;
    for (let ty = 0; ty < s; ty += step) {
      for (let tx = 0; tx < s; tx += step) {
        const ndc: [number, number] = [((tx + 0.5) / s) * 2 - 1, ((ty + 0.5) / s) * 2 - 1];
        const r = vdbFrameShadowPassRay(cam.projectionMatrix.elements, cam.projectionMatrixInverse.elements, cam.matrixWorld.elements, ndc);
        const cpu = vdbFrameShadowMapTexel(extinctionAt, r.origin, r.direction, box, steps, tile.ray).map;
        const i = (ty * map.width + tile.offset + tx) * 4;
        const gpu = [map.data[i]!, map.data[i + 1]!, map.data[i + 2]!, map.data[i + 3]!];
        stats.texels++;
        if (cpu[3] > 1e-4) stats.shadowed++;
        stats.maxDepth = Math.max(stats.maxDepth, cpu[3]);
        // Half-float storage: 2^-11 relative; f32 marching on the GPU
        for (const c of [0, 1, 3]) accumulate(stats, gpu[c]!, cpu[c]!, 2e-3, 2e-3, { tile: tile.offset / s, tx, ty, c, cpu, gpu });
      }
    }
  }
  return finish(stats) as NonNullable<Result["mapTwin"]>;
}

let lastDiag: unknown = null;
const TEXEL_JITTER: [number, number][] = [[0.06, 0], [-0.06, 0], [0, 0.06], [0, -0.06], [0.06, 0.06], [-0.06, -0.06], [0.06, -0.06], [-0.06, 0.06]];

/** The volume transmittance a receiver at `p` reads (the exact chain from the read-back map), or null outside it. */
function receiverTransmittance(entry: VDBFrameShadowEntry, map: VDBFrameAtlas, camera: THREE.Camera, p: V3, offset: [number, number] = [0, 0]): number | null {
  const s = entry.size;
  const pv = new THREE.Vector3(...p);
  let uv: [number, number];
  let ray: { origin: V3; forward: V3; perspective: number; far: number };
  if (entry.cascades) {
    const viewDepth = -pv.clone().applyMatrix4(camera.matrixWorldInverse).z;
    const count = entry.drawn.cascadeCount;
    const breaks = entry.drawn.cascadeBreaks.toArray();
    const index = vdbFrameShadowCascadeIndex(viewDepth, breaks, count);
    const cascade = entry.cascades[index]!;
    const clip = new THREE.Vector4(p[0], p[1], p[2], 1).applyMatrix4(cascade.drawn.viewProjection);
    const lookup = vdbFrameShadowCascadeUv([clip.x, clip.y], index, s);
    if (!(viewDepth <= breaks[count - 1]!) || !lookup.insideTile) return null;
    uv = lookup.uv;
    ray = { origin: v3(cascade.drawn.origin), forward: v3(entry.drawn.forward), perspective: 0, far: cascade.drawn.far };
  } else {
    const shadowCamera = entry.light.shadow.camera;
    const clip = new THREE.Vector4(p[0], p[1], p[2], 1).applyMatrix4(new THREE.Matrix4().multiplyMatrices(shadowCamera.projectionMatrix, shadowCamera.matrixWorldInverse));
    uv = [(clip.x / clip.w) * 0.5 + 0.5, 0.5 - (clip.y / clip.w) * 0.5];
    if (uv[0] < 0 || uv[0] > 1 || uv[1] < 0 || uv[1] > 1) return null;
    ray = { origin: v3(entry.drawn.origin), forward: v3(entry.drawn.forward), perspective: entry.drawn.perspective, far: entry.drawn.far };
  }
  const tx = uv[0] * map.width + offset[0];
  const ty = (1 - uv[1]) * map.height + offset[1];
  const texel = vdbFrameAtlasBilinear(map, tx, ty) as V4;
  const x0 = Math.floor(tx - 0.5);
  const y0 = Math.floor(ty - 0.5);
  lastDiag = { tx, ty, texel: [...texel], depth: vdbFrameShadowCameraDepth(ray, p), far: ray.far, taus: [[x0, y0], [x0 + 1, y0], [x0, y0 + 1], [x0 + 1, y0 + 1]].map(([a, b]) => [0, 1, 3].map((c) => map.data[(b! * map.width + a!) * 4 + c]!)) };
  return vdbFrameShadowReceiver(texel, vdbFrameShadowCameraDepth(ray, p), ray.far) as number;
}

async function runShadow(renderer: WebGPURenderer, device: GPUDevice): Promise<void> {
  const wvdb = readWvdb(new Uint8Array(await (await fetch(cloudUrl)).arrayBuffer()));
  const dense = wvdbDenseVolume(wvdb, { mipmaps: MIPS });
  const source = await makeSource(renderer, device, dense, false);

  const fr = new VDBFrameRenderer();
  Object.assign(fr.settings, {
    width: 96,
    height: 64,
    depth: 24,
    temporal: false,
    jitter: false,
    lighting: LIGHTING,
    multipleScattering: true,
    castShadows: CAST,
    castShadowMapSize: MAP_SIZE,
    castShadowCascades: CASCADES,
    castShadowSteps: STEPS,
  });
  result.settings = { ...fr.settings };

  // The cloud at 100 units (his demo size), its box bottom 20 above the ground (y = 0)
  const bbox = new THREE.Box3(new THREE.Vector3(...(dense.bbox?.[0] ?? [-0.5, -0.5, -0.5])), new THREE.Vector3(...(dense.bbox?.[1] ?? [0.5, 0.5, 0.5])));
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  const scale = 100 / Math.max(size.x, size.y, size.z);
  const volume = new VDBFrameVolume(source, { frameRenderer: fr, bbox, extinction: 60, shadowSteps: 24, mipmaps: MIPS });
  volume.scale.setScalar(scale);
  volume.position.copy(center).multiplyScalar(-scale).add(new THREE.Vector3(0, 20 + (size.y * scale) / 2, 0));

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(volume);

  const GROUND_HALF = 300;
  const ground = new THREE.Mesh(
    // Subdivided: SwiftShader interpolates varyings across a large triangle with ~1e-4 relative error (K3.6)
    new THREE.PlaneGeometry(GROUND_HALF * 2, GROUND_HALF * 2, Number(params.get("groundSegments") ?? 64), Number(params.get("groundSegments") ?? 64)),
    GROUND === "standard" ? new THREE.MeshStandardMaterial({ color: 0xbbbbbb, roughness: 1, metalness: 0 }) : new THREE.ShadowMaterial({ opacity: SHADOW_OPACITY }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  let light: THREE.DirectionalLight | THREE.SpotLight;
  if (LIGHT === "spot") {
    const spot = new THREE.SpotLight(0xffffff, 40000, 0, 0.6, 0.0, 2);
    spot.position.set(-60, 260, -110);
    spot.target.position.set(30, 0, 60);
    spot.shadow.camera.near = 10;
    spot.shadow.camera.far = 600;
    light = spot;
  } else {
    // The sun low enough that the shadow falls beside the volume; its three shadow camera covers the ground (his fitSunShadow)
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(-90, 240, -150);
    sun.target.position.set(0, 0, 0);
    Object.assign(sun.shadow.camera, { left: -450, right: 450, top: 450, bottom: -450, near: -1000, far: 1000 });
    light = sun;
  }
  light.castShadow = true;
  light.shadow.mapSize.setScalar(512);
  light.shadow.camera.updateProjectionMatrix();
  scene.add(light, light.target);
  if (GROUND === "standard") scene.add(new THREE.AmbientLight(0xffffff, 0.2));

  const camera = new THREE.PerspectiveCamera(55, W / H, 1, 5000);
  camera.position.set(70, 190, 220);
  camera.lookAt(25, 10, 45);
  camera.updateProjectionMatrix();
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const frame = async () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    return readLinear(renderer, rt);
  };

  // The first frame puts the filter on the light (in the proxy's onBeforeRender, after the ground drew); the receivers
  // build with it on the next frame, whose first receiver draws the maps (budget 1: one piece per render).
  for (let i = 0; i < 2 + CASCADES; i++) await frame();
  const on = await frame();
  const onAgain = await frame();
  const display = toDisplay(on);
  const displayAgain = toDisplay(onAgain);
  result.inPageIdentical = display.every((v, i) => v === displayAgain[i]);
  result.width = W;
  result.height = H;
  result.rgba = toBase64(display);

  const entry = fr.volumeShadows.entries.get(light) ?? null;
  result.entry = entry
    ? {
        size: entry.size,
        cascades: entry.drawn.cascadeCount,
        breaks: entry.drawn.cascadeBreaks.toArray(),
        far: entry.cascades ? entry.cascades.slice(0, entry.drawn.cascadeCount).map((c) => c.drawn.far) : [entry.drawn.far],
        drawn: fr.volumeShadows.lastUpdate.drawn,
        waiting: fr.volumeShadows.lastUpdate.waiting,
      }
    : null;

  // The volume's screen coverage: the volume alone (ground hidden) against the background
  ground.visible = false;
  const alone = await frame();
  ground.visible = true;
  const bgLinear = [BACKGROUND.r, BACKGROUND.g, BACKGROUND.b];
  const covered = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (Math.abs(alone[i * 4]! - bgLinear[0]!) + Math.abs(alone[i * 4 + 1]! - bgLinear[1]!) + Math.abs(alone[i * 4 + 2]! - bgLinear[2]!) > 1e-4) covered[i] = 1;
  }
  // Dilate by 2 pixels (the froxel composite's soft edge)
  const coveredWide = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      if (covered[y * W + x]) for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (x + dx >= 0 && x + dx < W && y + dy >= 0 && y + dy < H) coveredWide[(y + dy) * W + x + dx] = 1;

  // castShadows off: the shadow-only reference (three's own shadow of the light stays: no mesh casts, so 1)
  fr.settings.castShadows = false;
  await frame();
  const off = await frame();
  result.rgbaOff = toBase64(toDisplay(off));

  // Shadow stats over the visible ground (ShadowMaterial: out = background × (1 − opacity × (1 − T)))
  let groundPixels = 0;
  let shadowedPixels = 0;
  let darkening = 0;
  const points: (V3 | null)[] = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const p = coveredWide[i] ? null : groundPoint(camera, x, y, 0, GROUND_HALF - 1);
      points.push(p);
      if (!p) continue;
      groundPixels++;
      const d = off[i * 4 + 1]! - on[i * 4 + 1]!;
      if (d > 1e-3) shadowedPixels++;
      darkening += d;
    }
  }
  result.shadowStats = { groundPixels, covered: covered.reduce((a, b) => a + b, 0), shadowedPixels, meanDarkening: darkening / Math.max(1, groundPixels) };

  if (TWINS && entry && CAST) {
    // Read while the maps hold the shadows-on frame (castShadows off froze them: no update ran; entries are freed though)
    log("twins need the maps: rendering shadows on again", "warn");
  }
  if (TWINS && CAST) {
    fr.settings.castShadows = true;
    for (let i = 0; i < 2 + CASCADES; i++) await frame();
    const again = await frame();
    const e = fr.volumeShadows.entries.get(light)!;
    result.mapTwin = await shadowMapTwin(renderer, fr, e, Number(params.get("twinStep") ?? 2));
    log(`map twin: ${JSON.stringify(result.mapTwin)}`, result.mapTwin.over === 0 ? "ok" : "fail");

    const map = await readAtlas(renderer, e.target, 4);
    const grid = await readAtlas(renderer, fr.shadowDensityTarget!, 2);
    const layout = gridLayout(fr);
    const last = layout.count - 1;
    const gMin = layout.gridMin[last]!;
    const gSize = layout.gridSize[last]!;
    const toLight = LIGHT === "spot" ? null : new THREE.Vector3().subVectors(light.position, light.target.position).normalize();
    const independentT = (p: V3): number => {
      // Fine march from p toward the light through the last cascade (an independent CPU-integrated transmittance)
      const dir = toLight ? toLight.clone() : new THREE.Vector3().subVectors(light.position, new THREE.Vector3(...p)).normalize();
      const maxT = toLight ? Infinity : light.position.distanceTo(new THREE.Vector3(...p));
      let t0 = 0;
      let t1 = maxT;
      for (let k = 0; k < 3; k++) {
        const o = p[k]!;
        const d = dir.getComponent(k);
        const a = (gMin[k]! - o) / d;
        const b = (gMin[k]! + gSize[k]! - o) / d;
        t0 = Math.max(t0, Math.min(a, b));
        t1 = Math.min(t1, Math.max(a, b));
      }
      if (!(t1 > t0)) return 1;
      const n = 512;
      const dt = (t1 - t0) / n;
      let tau = 0;
      for (let j = 0; j < n; j++) {
        const t = t0 + (j + 0.5) * dt;
        tau += vdbFrameSampleGrid(grid, layout, last, vdbFrameGridUvw(layout, last, [p[0] + dir.x * t, p[1] + dir.y * t, p[2] + dir.z * t]))[0]! * dt;
      }
      return Math.exp(-tau);
    };

    const stats = Object.assign(newStats(), { ground: 0, shadowed: 0, meanT: 0, minT: 1, strictOver: 0, maxStrict: 0 });
    const ind = { compared: 0, meanAbs: 0, maxAbs: 0, within05: 0, within10: 0 };
    const nowhere = { unshadowed: 0, changed: 0, shadowed: 0, darker: 0 };
    let sumT = 0;
    // Debug image: red = twin over, green = volume coverage (dilated), blue = CPU shadow (1 − T)
    const dbg = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      dbg[i * 4 + 1] = coveredWide[i] ? 120 : 0;
      dbg[i * 4 + 3] = 255;
    }
    for (let i = 0; i < W * H; i++) {
      const p = points[i];
      if (!p) continue;
      const cpu = receiverTransmittance(e, map, camera, p);
      if (cpu === null) continue;
      stats.ground++;
      if (GROUND === "shadow") {
        // T from the ShadowMaterial alpha: out = bg (1 − opacity (1 − T)), per channel; green has the most bits
        const gpu = 1 - (1 - again[i * 4 + 1]! / bgLinear[1]!) / SHADOW_OPACITY;
        // SwiftShader rasterizes positionWorld with ~1e-4 relative error (the map read moves by a few hundredths of a texel; a
        // finer ground mesh changes which pixels): off by more than 4e-3, the GPU value must lie within the CPU values of
        // map reads ±0.06 texel away (the position tolerance), and those pixels are counted.
        let expected = cpu;
        if (Math.abs(gpu - cpu) > 4e-3) {
          stats.strictOver++;
          const alts = TEXEL_JITTER.map((o) => receiverTransmittance(e, map, camera, p, o) ?? 1);
          const lo = Math.min(cpu, ...alts);
          const hi = Math.max(cpu, ...alts);
          if (gpu >= lo - 4e-3 && gpu <= hi + 4e-3) expected = gpu;
          else dbg[i * 4] = 255;
        }
        accumulate(stats, gpu, expected, 4e-3, 0, { i, p, x: i % W, y: Math.floor(i / W), cpu, diag: expected !== gpu ? JSON.parse(JSON.stringify(lastDiag)) : null });
        stats.maxStrict = Math.max(stats.maxStrict, Math.abs(gpu - cpu));
        sumT += gpu;
        stats.minT = Math.min(stats.minT, gpu);
        const it = independentT(p);
        const d = Math.abs(gpu - it);
        ind.compared++;
        ind.meanAbs += d;
        ind.maxAbs = Math.max(ind.maxAbs, d);
        if (d <= 0.05) ind.within05++;
        if (d <= 0.1) ind.within10++;
      }
      // Nowhere else: where the CPU chain gives no shadow, the frame equals the shadows-off frame exactly
      if (cpu >= 0.99999) {
        nowhere.unshadowed++;
        if (again[i * 4] !== off[i * 4] || again[i * 4 + 1] !== off[i * 4 + 1] || again[i * 4 + 2] !== off[i * 4 + 2]) nowhere.changed++;
      } else if (cpu < 0.98) {
        nowhere.shadowed++;
        if (again[i * 4 + 1]! < off[i * 4 + 1]!) nowhere.darker++;
      }
      if (cpu < 0.999) stats.shadowed++;
      dbg[i * 4 + 2] = Math.round((1 - cpu) * 255);
    }
    stats.meanT = sumT / Math.max(1, stats.ground);
    result.debugRgba = toBase64(dbg);
    result.groundTwin = finish(stats) as NonNullable<Result["groundTwin"]>;
    ind.meanAbs /= Math.max(1, ind.compared);
    ind.within05 /= Math.max(1, ind.compared);
    ind.within10 /= Math.max(1, ind.compared);
    result.independent = ind;
    result.nowhereElse = nowhere;

    // The map's total optical depth and shadowed texels (the same grid gives the same map whatever the lighting mode)
    let tauSum = 0;
    let tauCount = 0;
    for (let k = 3; k < map.data.length; k += 4) if (map.data[k]! > 1e-4) (tauSum += map.data[k]!), tauCount++;
    result.mapTau = [tauSum, tauCount];
    log(`ground twin: ${JSON.stringify(result.groundTwin)}; independent: ${JSON.stringify(ind)}; nowhere else: ${JSON.stringify(nowhere)}`, result.groundTwin.over === 0 && nowhere.changed === 0 ? "ok" : "fail");
  }

  // Timing (D25: provisional): forced map redraws (signatures cleared), the volumeShadows pass alone
  if (TIMING > 0) {
    const timer = installPassTimer(renderer, fr);
    const samples: number[] = [];
    fr.settings.castShadows = true;
    for (let i = 0; i < 3; i++) await frame();
    await timer.collect();
    for (let f = 0; f < TIMING + 5; f++) {
      const e = fr.volumeShadows.entries.get(light)!;
      e.item.signature.length = 0;
      e.cascades?.forEach((c) => (c.signature.length = 0));
      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      const ms = await timer.collect();
      if (f >= 5 && ms.volumeShadows !== undefined) samples.push(ms.volumeShadows);
    }
    timer.dispose();
    samples.sort((a, b) => a - b);
    const median = samples.length ? samples[Math.floor(samples.length / 2)]! : null;
    const pieces = Math.max(1, fr.volumeShadows.lastUpdate.drawn);
    result.timing = { provisional: true, note: "three render-pass timestamps of the volume shadow map pass (forced redraw of every piece); D25", frames: samples.length, mapMs: median, perPieceMs: median === null ? null : median / pieces, samples };
    log(`timing: ${JSON.stringify(result.timing)}`);
  }

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(display), W, H), 0, 0);
  root.append(canvas);
  log(`shadow (${LIGHT}, ${SOURCE_KIND}, '${LIGHTING}' lighting): ${JSON.stringify(result.shadowStats)}; entry ${JSON.stringify(result.entry)}; repeat identical ${result.inPageIdentical}`, result.inPageIdentical ? "ok" : "fail");
}

// ── emission case ────────────────────────────────────────────────────────────────────────────────────

/** A smooth synthetic emitter as a `.wvdb`: Gaussian density (σ n/5) and flames (σ n/8, off-centre), 32³, 100 units. */
function blobWvdb(): Uint8Array {
  const n = 32;
  const density = new Float32Array(n * n * n);
  const flames = new Float32Array(n * n * n);
  for (let z = 0; z < n; z++)
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const i = x + n * (y + n * z);
        const c = (n - 1) / 2;
        density[i] = 0.8 * Math.exp(-((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) / (2 * (n / 5) ** 2));
        flames[i] = 0.9 * Math.exp(-((x - c - 3) ** 2 + (y - c + 2) ** 2 + (z - c) ** 2) / (2 * (n / 8) ** 2));
      }
  const bbox: [[number, number, number], [number, number, number]] = [
    [-50, -50, -50],
    [50, 50, 50],
  ];
  return writeWvdb(
    [
      { name: "density", class: "fog volume", background: 0, bbox, finest: { dimensions: [n, n, n], values: density } },
      { name: "flames", class: "fog volume", background: 0, bbox, finest: { dimensions: [n, n, n], values: flames } },
    ],
    { lods: 1 },
  ).bytes;
}

/**
 * The volume source's samplers on the GPU (the direction pass's `sampleExtinction` / `sampleEmission`) vs the CPU
 * samplers of the direction twin, at 64 × 64 local positions (a 64 × 16 × 4 lattice), extinction at lod 0 and 1.5.
 */
async function probeSource(renderer: WebGPURenderer, volume: VDBFrameVolume, hook: ReturnType<typeof vdbFrameBlackbodyEmission> | undefined, cpu: { sampleExtinction(l: V3, lod: number): number; sampleEmission(l: V3): V3 }): Promise<TwinStats & { lod: TwinStats }> {
  const { Fn, float, screenCoordinate, vec3, vec4, positionGeometry } = TSL as unknown as Record<string, N>;
  const nodes = vdbVolumeNodes(volume.source, volume.sourceUniforms, { useMipmaps: volume.mipmaps && volume.source.lodLevels > 0, ...(hook ? { emissionColor: hook } : {}) });
  const PW = 64;
  const local = (px: number, row: number): V3 => [(px + 0.5) / PW - 0.5, ((row % 16) + 0.5) / 16 - 0.5, (Math.floor(row / 16) + 0.5) / 4 - 0.5];
  const material = new NodeMaterial();
  material.fragmentNode = Fn(() => {
    const px = screenCoordinate.x.floor();
    const row = screenCoordinate.y.floor();
    const l = vec3(px.add(0.5).div(PW).sub(0.5), row.mod(16).add(0.5).div(16).sub(0.5), row.div(16).floor().add(0.5).div(4).sub(0.5)).toVar();
    const e = nodes.sampleEmission(l, float(0.0));
    return vec4(nodes.sampleExtinction(l, float(0.0)), e.x, e.y, nodes.sampleExtinction(l, float(1.5)));
  })();
  material.vertexNode = vec4(positionGeometry.xy, 0.0, 1.0);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(quad);
  const rt = new THREE.RenderTarget(PW, PW, { type: THREE.FloatType, depthBuffer: false });
  renderer.setRenderTarget(rt);
  renderer.render(scene, new THREE.OrthographicCamera());
  renderer.setRenderTarget(null);
  const a = await readAtlas(renderer, rt, 4);
  const stats = newStats();
  // Extinction at lod 1.5: dense mips vs the sparse coarse pyramid differ by design (K3.2); reported apart
  const lodStats = newStats();
  for (let yb = 0; yb < PW; yb++) {
    const row = PW - 1 - yb;
    for (let px = 0; px < PW; px++) {
      const l = local(px, row);
      const em = cpu.sampleEmission(l);
      const want = [cpu.sampleExtinction(l, 0), em[0], em[1], cpu.sampleExtinction(l, 1.5)];
      for (let c = 0; c < 4; c++) accumulate(c === 3 ? lodStats : stats, a.data[(yb * PW + px) * 4 + c]!, want[c]!, 1e-4, 1e-3, { px, row, c, l });
    }
  }
  rt.dispose();
  material.dispose();
  return Object.assign(finish(stats), { lod: finish(lodStats) });
}

async function runEmission(renderer: WebGPURenderer, device: GPUDevice): Promise<void> {
  // `?emitter=blob`: a smooth synthetic emitter (Gaussian density and flames, 32³): the strict direction-pass twin. SwiftShader's
  // division is ~12 bits (box segments move by ~2e-4), and the explosion's sharp flames under 8² rays × 24 steps amplify
  // that to several % per direction (its mean over the 16 directions stays within ~1.5%).
  const wvdb = readWvdb(BLOB ? blobWvdb() : new Uint8Array(await (await fetch(explosionUrl)).arrayBuffer()));
  // His explosion: the flames grid as emission, ÷ 6 (his emissiveScale), the density grid
  const dense = wvdbDenseVolume(wvdb, { emission: "flames", scales: { flames: BLOB ? 1.0 : 6.0 }, mipmaps: MIPS });
  const source = await makeSource(renderer, device, dense, true);

  const fr = new VDBFrameRenderer();
  Object.assign(fr.settings, { width: 96, height: 64, depth: 24, temporal: false, jitter: false, lighting: "volume", multipleScattering: true });
  result.settings = { ...fr.settings };

  const bbox = new THREE.Box3(new THREE.Vector3(...dense.bbox![0]), new THREE.Vector3(...dense.bbox![1]));
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  const scale = 100 / Math.max(size.x, size.y, size.z);
  const hook = SPECTRUM === "white" ? undefined : vdbFrameBlackbodyEmission("planck");
  const volume = new VDBFrameVolume(source, {
    frameRenderer: fr,
    bbox,
    baseColor: "#555555",
    // His explosion: normalized extinction 100 (optically very thick: the first non-empty sample takes almost all; `?extinction=`)
    extinction: Number(params.get("extinction") ?? (BLOB ? 20 : 100)),
    temperatureRange: [0, 3000],
    // His explosion 'planck' 15; the blob is dimmer (smooth, mostly thin flames): × 1000 keeps its probes out of half-float subnormals
    emissiveIntensity: Number(params.get("emissive") ?? (BLOB ? 15000 : 15)),
    shadowSteps: 24,
    mipmaps: MIPS,
    ...(hook ? { emissionColor: hook } : {}),
  });
  // Z-up, 100 units, the box centred on the stage; the stage stands on the ground at y = −50 (his frame.js)
  volume.scale.setScalar(scale);
  volume.position.copy(center).multiplyScalar(-scale);
  const up = new THREE.Group();
  up.rotation.x = -Math.PI / 2;
  up.add(volume);
  const stage = new THREE.Group();
  const height = size.z * scale;
  const groundY = -50;
  stage.position.y = groundY + height / 2;
  stage.add(up);

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(stage);

  const GROUND_HALF = 400;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(GROUND_HALF * 2, GROUND_HALF * 2, 64, 64), new THREE.MeshLambertMaterial({ color: 0xffffff }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = groundY;
  scene.add(ground);

  const light = new VDBFrameEmissionLight({ intensity: 25, resolution: PROBES, cascades: Number(params.get("emissionCascades") ?? 1) });
  light.visible = EMIT;
  scene.add(light);

  const camera = new THREE.PerspectiveCamera(55, W / H, 1, 5000);
  camera.position.set(0, 160, 330);
  camera.lookAt(0, -50, 0);
  camera.updateProjectionMatrix();
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const frame = async () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    return readLinear(renderer, rt);
  };

  await frame();
  const on = await frame();
  const onAgain = await frame();
  const display = toDisplay(on);
  result.inPageIdentical = display.every((v, i) => v === toDisplay(onAgain)[i]);
  result.width = W;
  result.height = H;
  result.rgba = toBase64(display);
  result.emitters = light.emitterCount;
  result.probeGrid = {
    resolution: light.probeResolution,
    count: light.cascadeCount,
    min: light.cascadeMin.slice(0, light.cascadeCount).map((v) => v.toArray()),
    size: light.cascadeSize.slice(0, light.cascadeCount).map((v) => v.toArray()),
    center: light.probeCenter.toArray(),
  };

  const groundPoints: (V3 | null)[] = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) groundPoints.push(groundPoint(camera, x, y, groundY, GROUND_HALF - 1));

  if (TWINS && EMIT) {
    const slots = light.slotCount;
    const dirA = await readAtlas(renderer, light.directionTarget, 4, 0);
    const dirB = await readAtlas(renderer, light.directionTarget, 4, 1);
    const emA = await readAtlas(renderer, light.emitterTarget, 4, 0);
    const emB = await readAtlas(renderer, light.emitterTarget, 4, 1);
    const probes = [await readAtlas(renderer, light.probeTarget, 4, 0), await readAtlas(renderer, light.probeTarget, 4, 1), await readAtlas(renderer, light.probeTarget, 4, 2)] as [VDBFrameAtlas, VDBFrameAtlas, VDBFrameAtlas];
    // rows bottom first: slot s is row s
    const texelOf = (a: VDBFrameAtlas, x: number, row: number): V4 => [0, 1, 2, 3].map((c) => a.data[(row * a.width + x) * 4 + c]!) as V4;

    // Direction pass vs the CPU twin (dense texels; the sparse source samples the same voxels at lod 0, its pyramid above)
    const emitter = light.emitters[0]!;
    const worldScale = emitter.matrixWorld.getMaxScaleOnAxis();
    const densityScale = volume.getDensityScale();
    const maxTexels = Math.max(...dense.dimensions);
    const range = volume.temperatureRange;
    const gridUv = (l: V3): V3 => [Math.min(1, Math.max(0, l[0] + 0.5)), Math.min(1, Math.max(0, l[1] + 0.5)), Math.min(1, Math.max(0, l[2] + 0.5))];
    const cpuEmitter = {
      sampleExtinction: (l: V3, lod: number) => {
        const v = sampleWvdbDenseTexture(dense.density, gridUv(l), MIPS ? lod : 0);
        return (v > volume.densityCutoff ? v : 0) * densityScale * volume.extinction / worldScale;
      },
      sampleEmission: (l: V3): V3 => {
        const v = sampleWvdbDenseTexture(dense.emission!, gridUv(l), 0);
        const T = range.x + (range.y - range.x) * v;
        if (!(T > 0)) return [0, 0, 0];
        const power = Math.pow(T / Math.max(range.y, 1), 4) * volume.emissiveIntensity;
        const c = hook ? vdbFrameBlackbodyColor(T, 0) : ([1, 1, 1] as V3);
        return [c[0] * power, c[1] * power, c[2] * power];
      },
      getLod: (localSize: number) => Math.max(0, Math.log2(localSize * maxTexels)),
      worldToLocal: emitter.worldToLocal.elements,
      center: v3(emitter.center),
      radius: emitter.radius,
      worldScale,
      absorbance: volume.absorbance,
      steps: light.getSteps(),
    };
    const dStats = Object.assign(newStats(), { texels: 0 });
    const cpuTexels = [];
    for (let d = 0; d < 16; d++) {
      const cpu = vdbFrameEmissionDirectionTexel(cpuEmitter, d);
      cpuTexels.push(cpu);
      const a = texelOf(dirA, d, 0);
      const b = texelOf(dirB, d, 0);
      dStats.texels++;
      const scaleI = Math.max(1e-6, Math.max(cpu.intensity[0], cpu.intensity[1], cpu.intensity[2]));
      const tolRel = SOURCE_KIND === "sparse" ? 0.02 : 4e-3;
      for (let c = 0; c < 4; c++) accumulate(dStats, a[c]!, cpu.intensity[c]!, scaleI * tolRel, tolRel, { d, out: 0, c });
      const scaleC = Math.max(1e-6, Math.abs(cpu.center[3]));
      for (let c = 0; c < 4; c++) accumulate(dStats, b[c]! / scaleC, cpu.center[c]! / scaleC, 0.5 * tolRel * 100, tolRel, { d, out: 1, c });
    }
    result.directionTwin = finish(dStats) as NonNullable<Result["directionTwin"]>;
    const cpuReduced = vdbFrameEmissionReduce(cpuTexels);
    result.sourceProbe = await probeSource(renderer, volume, hook, cpuEmitter);

    // Reduce pass vs the twin on the GPU direction texels
    const gpuTexels = Array.from({ length: 16 }, (_, d) => ({ intensity: texelOf(dirA, d, 0), center: texelOf(dirB, d, 0) }));
    const reduced = vdbFrameEmissionReduce(gpuTexels);
    const rStats = newStats();
    const gpuIntensity = texelOf(emA, 0, 0);
    const gpuCenter = texelOf(emB, 0, 0);
    for (let c = 0; c < 4; c++) accumulate(rStats, gpuIntensity[c]!, reduced.intensity[c]!, 1e-6, 1e-5, { out: 0, c });
    for (let c = 0; c < 4; c++) accumulate(rStats, gpuCenter[c]!, reduced.center[c]!, 1e-3, 1e-5, { out: 1, c });
    result.reduceTwin = finish(rStats);
    result.emitterLight = { intensity: [...gpuIntensity], center: [...gpuCenter] };
    // The mean over the 16 directions (what the probes read) vs the CPU twin's
    result.directionMean = {
      cpu: [...cpuReduced.intensity],
      gpu: [...gpuIntensity],
      maxRel: Math.max(...[0, 1, 2].map((c) => Math.abs(gpuIntensity[c]! - cpuReduced.intensity[c]!) / Math.max(1e-9, Math.max(...cpuReduced.intensity.slice(0, 3))))),
      centerDistance: Math.hypot(gpuCenter[0]! - cpuReduced.center[0], gpuCenter[1]! - cpuReduced.center[1], gpuCenter[2]! - cpuReduced.center[2]),
    };

    // Probe pass vs the twin (each probe from the GPU emitter texels; half-float targets)
    const layout: VDBFrameGridLayout = {
      count: light.cascadeCount,
      resolution: light.probeResolution,
      columns: 12,
      atlasSize: [384, 384],
      gridMin: light.cascadeMin.map(v3),
      gridSize: light.cascadeSize.map(v3),
      gridVoxel: light.cascadeSize.map((s) => v3(s.clone().divideScalar(light.probeResolution))),
    };
    const emitters = Array.from({ length: light.emitterCount }, (_, s) => ({ intensity: texelOf(emA, 0, s), center: texelOf(emB, 0, s) }));
    const pStats = Object.assign(newStats(), { probes: 0 });
    const res = layout.resolution;
    for (let fy = 0; fy < probes[0].height; fy++) {
      for (let fx = 0; fx < res * 12; fx++) {
        const cell = vdbFrameGridCell(res, 12, [fx + 0.5, fy + 0.5]);
        if (cell.cascade >= layout.count) continue;
        const m = layout.gridMin[cell.cascade]!;
        const s = layout.gridSize[cell.cascade]!;
        const probe: V3 = [m[0] + cell.gridUvw[0] * s[0], m[1] + cell.gridUvw[1] * s[1], m[2] + cell.gridUvw[2] * s[2]];
        const cpu = vdbFrameEmissionProbe(probe, emitters);
        pStats.probes++;
        for (let k = 0; k < 3; k++) {
          const g = texelOf(probes[k]!, fx, fy);
          const scaleP = Math.max(1e-9, Math.abs(cpu[0]![0]), Math.abs(cpu[1]![0]), Math.abs(cpu[2]![0])) * 2;
          for (let c = 0; c < 4; c++) accumulate(pStats, g[c]! / scaleP, cpu[k]![c]! / scaleP, 2e-3, 0, { fx, fy, k, c });
        }
      }
    }
    result.probeTwin = finish(pStats) as NonNullable<Result["probeTwin"]>;
    result.probeValues = [...texelOf(probes[0], Math.floor(res / 2), Math.floor(res / 2))];

    // Lit ground (Lambert: colour × E / π) vs the probe read twin on the read-back atlases
    const center3 = v3(light.probeCenter);
    const color = light.color.clone().multiplyScalar(light.intensity);
    const iStats = Object.assign(newStats(), { ground: 0, lit: 0 });
    let maxE = 0;
    const expectedAt = (p: V3): V3 => {
      const { probe, falloff } = vdbFrameEmissionLookup(probes, layout, center3, p);
      const e = vdbFrameEmissionIrradiance(probe, [0, 1, 0], falloff);
      return [(e[0] * color.r) / Math.PI, (e[1] * color.g) / Math.PI, (e[2] * color.b) / Math.PI];
    };
    for (let i = 0; i < W * H; i++) {
      const p = groundPoints[i];
      if (!p) continue;
      // The volume in front of the ground (the froxel composite) changes these pixels: compare where it does not reach
      const exp = expectedAt(p);
      maxE = Math.max(maxE, exp[0], exp[1], exp[2]);
    }
    const volumeOff = await (async () => {
      // The volume's proxy hidden: no froxel composite over the ground, the light still sees the emitter (traverseVisible)
      volume.proxy.visible = false;
      const f = await frame();
      volume.proxy.visible = true;
      return f;
    })();
    for (let i = 0; i < W * H; i++) {
      const p = groundPoints[i];
      if (!p) continue;
      iStats.ground++;
      const exp = expectedAt(p);
      if (exp[0] + exp[1] + exp[2] > maxE * 1e-3) iStats.lit++;
      for (let c = 0; c < 3; c++) accumulate(iStats, volumeOff[i * 4 + c]!, exp[c]!, maxE * 2e-3, 1e-2, { i, p, c });
    }
    result.irradianceTwin = finish(iStats) as NonNullable<Result["irradianceTwin"]>;

    // Falloff along the ground from below the emission centre outward (+x): measured (volume hidden) vs twin, and the
    // point-light estimate of the emitter as one L1 point light at its emission centre
    const c3 = new THREE.Vector3(...gpuCenter.slice(0, 3));
    result.falloff = [];
    for (const distance of [20, 60, 120, 200, 300]) {
      const p: V3 = [c3.x + distance, groundY, c3.z];
      // the pixel nearest to p
      const s = new THREE.Vector3(...p).project(camera);
      const px = Math.floor((s.x * 0.5 + 0.5) * W);
      const py = Math.floor((0.5 - s.y * 0.5) * H);
      if (px < 0 || px >= W || py < 0 || py >= H) continue;
      const gp = groundPoints[py * W + px];
      if (!gp) continue;
      const o = (py * W + px) * 4;
      const measured = volumeOff[o]! + volumeOff[o + 1]! + volumeOff[o + 2]!;
      const e3 = expectedAt(gp);
      const twin = e3[0] + e3[1] + e3[2];
      const pl = vdbFrameEmissionIrradiance(vdbFrameEmissionProbe(gp, emitters), [0, 1, 0]);
      const point = (pl[0] * color.r + pl[1] * color.g + pl[2] * color.b) / Math.PI;
      result.falloff.push({ distance: Math.hypot(gp[0] - c3.x, gp[2] - c3.z), measured, twin, pointLight: point });
    }
    log(`direction twin ${JSON.stringify(result.directionTwin)}; reduce ${JSON.stringify(result.reduceTwin)}; probe ${JSON.stringify(result.probeTwin)}; irradiance ${JSON.stringify(result.irradianceTwin)}; falloff ${JSON.stringify(result.falloff)}`,
      [result.directionTwin, result.reduceTwin, result.probeTwin, result.irradianceTwin].every((s) => s.over === 0) ? "ok" : "fail");
  }

  // The light off: the ground is black (no other light), the probes are not read
  light.visible = false;
  await frame();
  const off = await frame();
  result.rgbaOff = toBase64(toDisplay(off));
  volume.visible = false;
  const offNoVolume = await frame();
  volume.visible = true;
  let groundPixels = 0;
  let nonBlack = 0;
  for (let i = 0; i < W * H; i++) {
    if (!groundPoints[i]) continue;
    groundPixels++;
    if (offNoVolume[i * 4]! !== 0 || offNoVolume[i * 4 + 1]! !== 0 || offNoVolume[i * 4 + 2]! !== 0) nonBlack++;
  }
  result.lightOff = { groundPixels, nonBlack };
  light.visible = EMIT;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(display), W, H), 0, 0);
  root.append(canvas);
  log(`emission (${SOURCE_KIND}): ${light.emitterCount} emitter(s), probes ${JSON.stringify(result.probeGrid)}; light off ${JSON.stringify(result.lightOff)}; repeat identical ${result.inPageIdentical}`, result.inPageIdentical ? "ok" : "fail");
}

async function main(): Promise<void> {
  root.textContent = "";
  result.case = CASE;
  result.sourceKind = SOURCE_KIND;
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const features: GPUFeatureName[] = TIMING > 0 && adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures: features, ...(SOURCE_KIND === "sparse" ? { requiredLimits: VDBGridPool.requiredLimits(adapter) } : {}) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const renderer = new WebGPURenderer({ device, trackTimestamp: features.length > 0 });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  renderer.shadowMap.enabled = true;

  if (CASE === "emission") await runEmission(renderer, device);
  else await runShadow(renderer, device);
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
