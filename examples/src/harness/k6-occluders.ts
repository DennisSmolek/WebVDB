/**
 * K6.5b gate harness: mesh → volume shadows through `backend: 'core'` on three r187. A box between a light and the
 * cloud (his `bunny_cloud.wvdb` from `fixtures/k3/` when present, else the committed CC0 cloud) casts into three's
 * shadow map; the core reads that map (`VDBFrameInput.occluders`) and the cloud behind the box darkens. The box is on
 * layer 1 only (the light's shadow camera sees it, the view camera does not), so the image shows the cloud alone; a
 * small lit sphere in a corner receives shadows (three makes a light's shadow node only for a lit receiver).
 *
 * Frames (jitter and temporal off): `on` (occlusion), `off` (`frameRenderer.occlusion = false`), `noCaster` (occlusion
 * on, the box's `castShadow` off: a map with nothing in it), and the lag check: the box moves and one frame is drawn
 * (`moved`), then more frames (`settled`) — equal when the core reads this frame's map (no lag).
 *
 * `mask` (per pixel): 0 the ray misses the volume box; 1 every sample of the ray inside the volume box is in the
 * eroded box's shadow (expect darker); 2 no sample is in the dilated box's shadow (expect unchanged); 3 between.
 *
 * Query: `?light=sun|spot|point`, `?lighting=frame|volume`, `?volume=bunny|cloud`, `?w=&h=`.
 * Results: `window.__VDB_K6_OCCLUDERS__` (e2e/k6-occluders.spec.ts).
 */
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { VDBFrameRenderer, VDBFrameVolume, vdbDenseTextures } from "../../../packages/vdb/src/three/internal.js";
import type { VDBFrameInput } from "../../../packages/vdb/src/core/internal.js";
import { readWvdb, sampleWvdbDenseTexture, vdbSrgbOetf, wvdbDenseVolume } from "../../../packages/vdb/src/tools/internal.js";
import { readAtlas } from "./k3-frame-lighting.js";
import cloudUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

type V3 = [number, number, number];

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  volume?: string;
  width?: number;
  height?: number;
  on?: string;
  off?: string;
  noCaster?: string;
  moved?: string;
  settled?: string;
  mask?: string;
  backend?: { core: boolean; occluders: number[]; layers: number[] };
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K6_OCCLUDERS__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const LIGHT = (params.get("light") ?? "sun") as "sun" | "spot" | "point";
const LIGHTING = (params.get("lighting") ?? "frame") as "frame" | "volume";
const VOLUME = params.get("volume") ?? "bunny";
const W = Number(params.get("w") ?? 256);
const H = Number(params.get("h") ?? 160);
const BACKGROUND = new THREE.Color(0x598eff);
const EXTINCTION = 40;

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K6_OCCLUDERS__ = result;

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

/** The frame as sRGB bytes (three's OETF, no tone mapping), rows top first. */
async function readDisplay(renderer: WebGPURenderer, rt: THREE.RenderTarget): Promise<Uint8Array> {
  const a = await readAtlas(renderer, rt, 4);
  const out = new Uint8Array(W * H * 4);
  const byte = (v: number) => Math.round(Math.min(1, Math.max(0, vdbSrgbOetf(v))) * 255);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const src = ((H - 1 - y) * W + x) * 4;
      const dst = (y * W + x) * 4;
      for (let c = 0; c < 3; c++) out[dst + c] = byte(a.data[src + c]!);
      out[dst + 3] = 255;
    }
  return out;
}

/** Slab test of a ray against an axis-aligned box: [t0, t1] or null. */
function slab(o: V3, d: V3, min: V3, max: V3): [number, number] | null {
  let t0 = 0;
  let t1 = Infinity;
  for (let a = 0; a < 3; a++) {
    if (Math.abs(d[a]!) < 1e-12) {
      if (o[a]! < min[a]! || o[a]! > max[a]!) return null;
      continue;
    }
    const ta = (min[a]! - o[a]!) / d[a]!;
    const tb = (max[a]! - o[a]!) / d[a]!;
    t0 = Math.max(t0, Math.min(ta, tb));
    t1 = Math.min(t1, Math.max(ta, tb));
  }
  return t0 <= t1 ? [t0, t1] : null;
}

async function loadVolume(): Promise<{ name: string; bytes: Uint8Array }> {
  if (VOLUME === "bunny") {
    const r = await fetch("/fixtures/k3/bunny_cloud.wvdb");
    if (r.ok && !(r.headers.get("content-type") ?? "").includes("text/html")) return { name: "bunny_cloud.wvdb", bytes: new Uint8Array(await r.arrayBuffer()) };
  }
  return { name: "cloud_04_variant_0000.wvdb", bytes: new Uint8Array(await (await fetch(cloudUrl)).arrayBuffer()) };
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
  result.adapter = [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  const renderer = new WebGPURenderer({ device });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  renderer.shadowMap.enabled = true;

  const file = await loadVolume();
  result.volume = file.name;
  const dense = wvdbDenseVolume(readWvdb(file.bytes), { mipmaps: true });
  const fr = new VDBFrameRenderer();
  fr.backend = "core";
  Object.assign(fr.settings, { width: W / 2, height: H / 2, depth: 32, temporal: false, jitter: false, lighting: LIGHTING, multipleScattering: true, shadowResolution: 64 });

  // The cloud: its longest side 2 units, centred at (0, 1.5, 0)
  const bbox = new THREE.Box3(new THREE.Vector3(...(dense.bbox?.[0] ?? [-0.5, -0.5, -0.5])), new THREE.Vector3(...(dense.bbox?.[1] ?? [0.5, 0.5, 0.5])));
  const size = bbox.getSize(new THREE.Vector3());
  const center = new THREE.Vector3(0, 1.5, 0);
  const scale = 2 / Math.max(size.x, size.y, size.z);
  const volume = new VDBFrameVolume(vdbDenseTextures(dense), { frameRenderer: fr, bbox, extinction: EXTINCTION, shadowSteps: 24, mipmaps: true });
  volume.scale.setScalar(scale);
  volume.position.copy(bbox.getCenter(new THREE.Vector3())).multiplyScalar(-scale).add(center);
  const worldMin: V3 = [bbox.min.x * scale + volume.position.x, bbox.min.y * scale + volume.position.y, bbox.min.z * scale + volume.position.z];
  const worldMax: V3 = [bbox.max.x * scale + volume.position.x, bbox.max.y * scale + volume.position.y, bbox.max.z * scale + volume.position.z];

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(volume);
  scene.add(new THREE.AmbientLight(0xffffff, 0.25));

  // The light, toward the light from the cloud's centre along L; the box 2.3 units along L, between the light and the cloud
  const L = new THREE.Vector3(0.4, 1, 0.3).normalize();
  let light: THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight;
  let boxSize = 0.8;
  if (LIGHT === "spot") {
    const spot = new THREE.SpotLight(0xffffff, 3 * 25, 0, 0.45, 0.1, 2);
    spot.position.copy(center).addScaledVector(L, 5);
    spot.target.position.copy(center);
    spot.shadow.camera.near = 0.5;
    spot.shadow.camera.far = 20;
    light = spot;
    boxSize = 0.45;
  } else if (LIGHT === "point") {
    const point = new THREE.PointLight(0xffffff, 3 * 12, 0, 2);
    point.position.copy(center).addScaledVector(L, 3.5);
    point.shadow.camera.near = 0.1;
    point.shadow.camera.far = 20;
    light = point;
    boxSize = 0.3;
  } else {
    const sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.position.copy(center).addScaledVector(L, 10);
    sun.target.position.copy(center);
    Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 30 });
    light = sun;
  }
  light.castShadow = true;
  light.shadow.mapSize.setScalar(512);
  light.shadow.camera.layers.enable(1);
  (light.shadow.camera as THREE.Camera & { updateProjectionMatrix(): void }).updateProjectionMatrix();
  scene.add(light);
  if ("target" in light) scene.add(light.target);

  const box = new THREE.Mesh(new THREE.BoxGeometry(boxSize, boxSize, boxSize), new THREE.MeshBasicMaterial({ color: 0xff3366 }));
  box.position.copy(center).addScaledVector(L, 2.3);
  box.castShadow = true;
  box.layers.set(1);
  scene.add(box);

  // A lit receiver in the corner: three makes the light's shadow node (and map) for a material that receives shadows
  const receiver = new THREE.Mesh(new THREE.SphereGeometry(0.15, 16, 8), new THREE.MeshStandardMaterial({ color: 0xcccccc }));
  receiver.receiveShadow = true;

  const camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 100);
  camera.position.set(0, 1.8, 4.6);
  camera.lookAt(center);
  camera.updateMatrixWorld(true);
  const corner = new THREE.Vector3(-0.85, -0.8, 0.5).unproject(camera).sub(camera.position).normalize();
  receiver.position.copy(camera.position).addScaledVector(corner, 5);
  scene.add(receiver);
  scene.updateMatrixWorld(true);

  // What the core gets: wrap its update once it exists (after the first render)
  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const seen: VDBFrameInput[] = [];
  let wrapped = false;
  const frame = async () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    if (!wrapped && fr.core) {
      const core = fr.core;
      const update = core.update.bind(core);
      core.update = (encoder, input, options) => {
        seen.push(input);
        return update(encoder, input, options);
      };
      wrapped = true;
    }
    return readDisplay(renderer, rt);
  };
  for (let i = 0; i < 3; i++) await frame();
  const on = await frame();
  const last = seen[seen.length - 1];
  result.backend = { core: !!fr.core, occluders: (last?.occluders ?? []).map((o) => Number(o.light)), layers: (last?.occluders ?? []).map((o) => o.viewProjection.length) };

  fr.occlusion = false;
  await frame();
  const off = await frame();
  fr.occlusion = true;
  box.castShadow = false;
  await frame();
  const noCaster = await frame();
  box.castShadow = true;

  // Lag: move the box sideways and draw one frame, then settle
  const home = box.position.clone();
  box.position.add(new THREE.Vector3(0.25, 0, -0.1));
  box.updateMatrixWorld(true);
  const moved = await frame();
  await frame();
  const settled = await frame();
  box.position.copy(home);
  box.updateMatrixWorld(true);

  // The mask: samples along each pixel ray inside the volume box, tested against the box's shadow (eroded / dilated)
  const lightPosition = light.position.clone();
  const directional = LIGHT === "sun";
  const half = boxSize / 2;
  const erode = Math.min(0.1, boxSize / 4);
  const dilate = 0.25;
  const inShadow = (p: V3, grow: number) => {
    const min: V3 = [home.x - half - grow, home.y - half - grow, home.z - half - grow];
    const max: V3 = [home.x + half + grow, home.y + half + grow, home.z + half + grow];
    if (directional) return !!slab(p, [L.x, L.y, L.z], min, max);
    const d: V3 = [lightPosition.x - p[0], lightPosition.y - p[1], lightPosition.z - p[2]];
    const hit = slab(p, d, min, max);
    return !!hit && hit[0] <= 1;
  };
  const mask = new Uint8Array(W * H);
  const origin: V3 = [camera.position.x, camera.position.y, camera.position.z];
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const p = new THREE.Vector3(((x + 0.5) / W) * 2 - 1, 1 - ((y + 0.5) / H) * 2, 0.5).unproject(camera).sub(camera.position).normalize();
      const seg = slab(origin, [p.x, p.y, p.z], worldMin, worldMax);
      if (!seg) continue;
      // The cloud's visibility weights along the ray (a CPU march of the density: what the pixel shows)
      let any = false;
      let transmittance = 1;
      let weight = 0;
      let shadowed = 0;
      const n = 96;
      const dt = (seg[1] - seg[0]) / n;
      for (let k = 0; k < n; k++) {
        const t = seg[0] + (k + 0.5) * dt;
        const q: V3 = [origin[0] + p.x * t, origin[1] + p.y * t, origin[2] + p.z * t];
        if (inShadow(q, dilate)) any = true;
        const uvw: V3 = [0, 1, 2].map((a) => (q[a]! - worldMin[a]!) / (worldMax[a]! - worldMin[a]!)) as V3;
        const sigma = (sampleWvdbDenseTexture(dense.density, uvw, 0) / Math.max(dense.meanDensity, 1e-6)) * (EXTINCTION / 2); // per world unit: the box’s longest side is 2
        const w = transmittance * (1 - Math.exp(-sigma * dt));
        transmittance *= Math.exp(-sigma * dt);
        weight += w;
        if (inShadow(q, -erode)) shadowed += w;
      }
      // inside: an opaque enough pixel whose visible light is ≥ 90 % in the eroded shadow; outside: no sample near the shadow
      mask[y * W + x] = weight > 0.3 && shadowed >= 0.9 * weight ? 1 : !any ? 2 : 3;
    }

  result.width = W;
  result.height = H;
  result.on = toBase64(on);
  result.off = toBase64(off);
  result.noCaster = toBase64(noCaster);
  result.moved = toBase64(moved);
  result.settled = toBase64(settled);
  result.mask = toBase64(mask);
  log(`volume ${file.name}, light ${LIGHT}, '${LIGHTING}' lighting: core ${result.backend.core}, occluders ${JSON.stringify(result.backend.occluders)}`, "ok");
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
