/**
 * Demo 04 — atlas-fallback: the compute toolset end to end (SPEC §3.3/§5;
 * K2.3 port onto the engine-free core + three adapter).
 *
 * `vdbDecodeToAtlas` (core compute on a `VDBGridPool`, through the three
 * wrapper) turns `sphere_fog_fp8.nvdb` into a dense `Data3DTexture`, rendered
 * with three's OWN stock `VolumeNodeMaterial` (not the sparse raymarch) — the
 * mobile/compat fallback path for a GPU that doesn't need (or can't do) the
 * full storage-buffer traversal every fragment. `vdbGridStats` and
 * `vdbApplyValueTransform` run against a second (Float) fixture and are checked
 * against the `tools` CPU twins, publishing verification booleans.
 *
 * ## Atlas sampling: the stock 3D texture (K2.5, DIVERGES V8 closed)
 *
 * The atlas is sampled with TSL `texture3D()` on the wrapper's
 * `Data3DTexture`, on hardware (R9, 2026-07-12) and in the e2e. v1 packed the
 * bytes into a storage buffer and sampled them by hand, because the sandbox's
 * Dawn rejected the 3D texture upload under three r185; with three r187 and
 * headless shell 1243 the stock path works on SwiftShader too, and renders the
 * golden bit for bit (K2.3B, 2026-11-18), so the workaround is gone.
 *
 * ## Why a second fixture for stats/transform
 *
 * `vdbApplyValueTransform` is Float-only (re-encoding quantized leaves is out of scope)
 * and the atlas fixture (`sphere_fog_fp8`) is Fp8, so `box_fog_float.nvdb` —
 * small (81 x 41 x 61 index bbox, every voxel active, ~200K voxels) —
 * exercises `vdbGridStats` (against `vdbGridStatsCpu` and the sidecar) and
 * `vdbApplyValueTransform` (on the CPU since K2.3, staged with `pool.replace`).
 *
 * ## Presentation & determinism
 *
 * Default presentation is the canvas (K2.5, DIVERGES V7). `?presentation=
 * readback` keeps the offscreen-RT -> 2D-canvas blit the e2e needs (canvas
 * presentation drops the Dawn instance under headless SwiftShader), with the
 * same swizzle/adapter-keepalive shims as demo 02 and the `?test=1`
 * deterministic mode (fixed pose, one settled render).
 * `window.__DEMO04__ = { ready, statsOk, transformOk, error? }` is the e2e
 * read channel.
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Fn, texture3D, uniform } from "three/tsl";
import { VolumeNodeMaterial, WebGPURenderer } from "three/webgpu";
import type { Node } from "three/webgpu";
import { VDBGridPool, vdbApplyValueTransform } from "@pmndrs/vdb";
import { NvdbFile, vdbGridStatsCpu, vdbReadValueF32 } from "@pmndrs/vdb/tools";
// `getDim` (leaf voxel vs tile) is a CPU twin of the WGSL march helper, not public API (D32): this check imports it
// from the tools internals by path.
import { getDim } from "../../../../packages/vdb/src/tools/internal.js";
import { getDevice, vdbDecodeToAtlas, vdbGridStats } from "@pmndrs/vdb/three";

interface Demo04State {
  ready: boolean;
  statsOk: boolean;
  transformOk: boolean;
  fixture?: string;
  statsFixture?: string;
  gridType?: string;
  adapter?: string;
  presentation?: string;
  textureMode?: string;
  swizzleShim?: boolean;
  keepAliveShim?: boolean;
  resolution?: string;
  atlasMaxDim?: number;
  atlasFilter?: string;
  atlasFormat?: string;
  atlasNormalize?: boolean;
  error?: string;
}

declare global {
  interface Window {
    __DEMO04__?: Demo04State;
  }
}

interface Sample {
  ijk: [number, number, number];
  value: number;
  active: boolean;
}
interface Sidecar {
  grid: { name: string; activeVoxelCount: number };
  samples: Sample[];
}

const ATLAS_FIXTURE = fixtureUrl("primitives/sphere_fog_fp8.nvdb");
const STATS_FIXTURE_NVDB = fixtureUrl("primitives/box_fog_float.nvdb");
const STATS_FIXTURE_SIDECAR = fixtureUrl("primitives/box_fog_float.sidecar.json");
const ATLAS_MAX_DIM = 96;
const ATLAS_FILTER = "trilinear";
const ATLAS_FORMAT = "uint8";
const ATLAS_NORMALIZE = true;

const STATS_VALUE_EPS = 1e-3; // FP8 atlas isn't involved here, but keep parity with demo 01/harness tolerances
const TRANSFORM_VALUE_EPS = 1e-4;
const MAX_TRANSFORM_SAMPLES = 50;

/** Same r185–r187 x SwiftShader compat shim as demo 02 (see its module header). */
function installSwizzleCompatShim(): void {
  if (typeof GPUTexture === "undefined") return;
  const proto = GPUTexture.prototype as unknown as {
    createView: (d?: GPUTextureViewDescriptor) => GPUTextureView;
    __swizzleShim?: boolean;
  };
  if (proto.__swizzleShim) return;
  const orig = proto.createView;
  proto.createView = function (this: GPUTexture, descriptor?: GPUTextureViewDescriptor): GPUTextureView {
    if (descriptor && typeof (descriptor as { swizzle?: unknown }).swizzle === "string") {
      const { swizzle: _drop, ...rest } = descriptor as GPUTextureViewDescriptor & { swizzle?: unknown };
      return orig.call(this, rest);
    }
    return orig.call(this, descriptor);
  };
  proto.__swizzleShim = true;
}

const _pinnedAdapters: unknown[] = [];
function installAdapterKeepAliveShim(): void {
  if (typeof navigator === "undefined" || !navigator.gpu) return;
  const gpu = navigator.gpu as GPU & { __keepAliveShim?: boolean };
  if (gpu.__keepAliveShim) return;
  const orig = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (options?: GPURequestAdapterOptions): Promise<GPUAdapter | null> => {
    const adapter = await orig(options);
    if (adapter) _pinnedAdapters.push(adapter);
    return adapter;
  };
  gpu.__keepAliveShim = true;
}

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
const PRESENTATION = params.get("presentation") === "readback" ? "readback" : "direct";
/** Reported in the state; the storage-buffer atlas path (`?texture=storage`) was removed in K2.5. */
const TEXTURE_MODE = "3d";
const SWIZZLE_SHIM = params.get("swizzle") !== "off";
const KEEP_ALIVE_SHIM = params.get("keepAlive") !== "off";

const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const panelEl = document.querySelector<HTMLDivElement>("#panel")!;
const appEl = document.querySelector<HTMLDivElement>("#app")!;

function fail(message: string): void {
  errEl.textContent = message;
  statsEl.textContent = "failed";
  window.__DEMO04__ = { ready: false, statsOk: false, transformOk: false, error: message };
}

async function fetchOk(url: string): Promise<ArrayBuffer | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.arrayBuffer();
  } catch {
    return null;
  }
}

async function run(): Promise<void> {
  if (!navigator.gpu) {
    fail("WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.");
    return;
  }
  if (SWIZZLE_SHIM) installSwizzleCompatShim();
  if (KEEP_ALIVE_SHIM) installAdapterKeepAliveShim();

  const [atlasBuf, statsBuf, sidecarRes] = await Promise.all([
    fetchOk(ATLAS_FIXTURE),
    fetchOk(STATS_FIXTURE_NVDB),
    fetch(STATS_FIXTURE_SIDECAR).catch(() => null),
  ]);
  if (!atlasBuf) {
    fail(`Missing atlas fixture: ${ATLAS_FIXTURE}`);
    return;
  }
  if (!statsBuf) {
    fail(`Missing stats fixture: ${STATS_FIXTURE_NVDB}`);
    return;
  }
  if (!sidecarRes || !sidecarRes.ok) {
    fail(`Missing sidecar: ${STATS_FIXTURE_SIDECAR}`);
    return;
  }
  const sidecar = (await sidecarRes.json()) as Sidecar;

  const atlasImage = NvdbFile.fromArrayBuffer(atlasBuf).gridImage(0);
  const statsImage = NvdbFile.fromArrayBuffer(statsBuf).gridImage(0);

  // Device first (D8): the host owns the device and passes it to three, so the
  // pool on it outlives the renderer (CLAUDE.md Traps).
  const gpuAdapter = await navigator.gpu.requestAdapter();
  if (!gpuAdapter) {
    fail("No WebGPU adapter.");
    return;
  }
  const device = await gpuAdapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(gpuAdapter) });
  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const renderer = new WebGPURenderer({ device, antialias: false, ...(directCanvas ? { canvas: directCanvas } : {}) });
  await renderer.init();
  const adapter = [gpuAdapter.info?.vendor, gpuAdapter.info?.architecture].filter(Boolean).join(" ") || "unreported adapter";

  const pool = await VDBGridPool.create(getDevice(renderer), { label: "demo 04" });
  const atlasId = pool.add(atlasImage, { label: "atlas" });
  const statsId = pool.add(statsImage, { label: "stats" });
  pool.flush();
  const atlasInfo = pool.info(atlasId);

  // -------------------------------------------------------------------------
  // gridStats: GPU result vs. the CPU twin (vdbGridStatsCpu) and the sidecar.
  // -------------------------------------------------------------------------
  const gpuStats = await vdbGridStats(renderer, pool, statsId);
  const cpuStats = vdbGridStatsCpu(statsImage);
  const statsOk =
    gpuStats.activeVoxelCount === sidecar.grid.activeVoxelCount &&
    gpuStats.activeVoxelCount === cpuStats.activeVoxelCount &&
    Math.abs(gpuStats.min - cpuStats.min) <= STATS_VALUE_EPS &&
    Math.abs(gpuStats.max - cpuStats.max) <= STATS_VALUE_EPS &&
    Math.abs(gpuStats.mean - cpuStats.mean) <= STATS_VALUE_EPS &&
    gpuStats.histogram.reduce((a, b) => a + b, 0) === gpuStats.activeVoxelCount;

  // -------------------------------------------------------------------------
  // vdbApplyValueTransform "double" (CPU, staged with pool.replace), then check the
  // result at up to 50 active sidecar coords. `box_fog_float`'s solid
  // interior is constant LOWER-level tiles even though every bbox voxel is
  // active, so per sample:
  //   - leaf voxel (getDim 1): the value doubles;
  //   - tile value (getDim > 1): unchanged (shared tile storage is skipped).
  // Both branches are asserted, so the leaf-vs-tile rule is enforced, not just
  // documented. Then the GPU reads the replaced grid: gridStats' max must
  // match the twin on the transformed image (the pool wrote it in place).
  // -------------------------------------------------------------------------
  const versionBefore = pool.version;
  const transformedImage = vdbApplyValueTransform(pool, statsId, statsImage, "double");
  pool.flush();
  const activeSamples = sidecar.samples.filter((s) => s.active).slice(0, MAX_TRANSFORM_SAMPLES);
  let sawLeaf = false;
  let transformOk = activeSamples.length > 0 && pool.version === versionBefore;
  for (const s of activeSamples) {
    const got = vdbReadValueF32(transformedImage, s.ijk);
    if (getDim(statsImage, s.ijk) === 1) {
      sawLeaf = true;
      if (Math.abs(got - 2 * s.value) > TRANSFORM_VALUE_EPS) transformOk = false;
    } else if (Math.abs(got - s.value) > TRANSFORM_VALUE_EPS) {
      transformOk = false;
    }
  }
  // A vacuously-true result (no leaf samples, nothing exercised) doesn't count.
  if (!sawLeaf) transformOk = false;
  const transformedStats = await vdbGridStats(renderer, pool, statsId);
  if (transformedStats.max !== vdbGridStatsCpu(transformedImage).max) transformOk = false;

  // -------------------------------------------------------------------------
  // decodeToAtlas -> VolumeNodeMaterial (the fallback path). The wrapper
  // returns the bytes and a ready `Data3DTexture` (RedFormat, uint8, linear).
  // -------------------------------------------------------------------------
  const atlas = await vdbDecodeToAtlas(renderer, pool, atlasId, {
    maxDim: ATLAS_MAX_DIM,
    filter: ATLAS_FILTER,
    format: ATLAS_FORMAT,
    normalize: ATLAS_NORMALIZE,
  });
  const [dx, dy, dz] = atlas.dims;
  const dataTexture = atlas.texture;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);

  const worldBox = new THREE.Box3(new THREE.Vector3(...atlasInfo.worldBBox.min), new THREE.Vector3(...atlasInfo.worldBBox.max));
  const center = worldBox.getCenter(new THREE.Vector3());
  const size = worldBox.getSize(new THREE.Vector3());
  const diag = size.length();

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, diag * 100);

  // Map world position -> the box's normalized [0,1]^3 texture space. The
  // proxy geometry sits exactly on worldBox with no extra mesh transform
  // (the proxy box below), so this is a plain
  // translate + per-axis inverse-scale, no matrix inverse needed.
  // K2.6: atlas texel i is voxel min + i, so the texture spans the voxel-centred
  // box [min - 0.5, max + 0.5] (index space): the world cell box shifted by
  // half a voxel. The proxy sits on the same box.
  const halfVoxel = new THREE.Vector3(...atlasInfo.voxelSize).multiplyScalar(0.5);
  const boxMin = uniform(worldBox.min.clone().sub(halfVoxel));
  const boxSizeInv = uniform(new THREE.Vector3(1 / size.x, 1 / size.y, 1 / size.z));
  const tintColor = uniform(new THREE.Vector3(0.85, 0.9, 1.0));
  const densityGain = uniform(0.9);

  const material = new VolumeNodeMaterial();
  material.steps = 48;
  // `scatteringEmissiveNode` is a real, runtime-supported `VolumeNodeMaterial`
  // property (`VolumetricLightingModel.start()` reads it directly, adding its
  // result independent of any scene lights) — used here instead of
  // `scatteringNode` (which only contributes via the lighting pipeline's
  // `direct()` callback, i.e. only when a real `PointLight`/`SpotLight` with
  // a `.distance` is present in the scene) so this demo needs no light setup
  // at all, just the density texture. `@types/three@0.185.1` hasn't caught up
  // with this field yet (only `scatteringNode` is typed), hence the cast.
  (material as VolumeNodeMaterial & { scatteringEmissiveNode?: (p: { positionRay: Node<"vec3"> }) => Node }).scatteringEmissiveNode = Fn(
    ({ positionRay }: { positionRay: Node<"vec3"> }) => {
      const uvw = positionRay.sub(boxMin).mul(boxSizeInv);
      const density = texture3D(dataTexture, uvw).r;
      return tintColor.mul(density).mul(densityGain);
    },
  );

  const proxy = new THREE.BoxGeometry(size.x, size.y, size.z);
  proxy.translate(center.x - halfVoxel.x, center.y - halfVoxel.y, center.z - halfVoxel.z);
  const mesh = new THREE.Mesh(proxy, material);
  scene.add(mesh);

  // `NodeMaterial.setupLighting()` (three's own WebGPU node-material pipeline)
  // skips the ENTIRE lighting-model codepath — meaning `VolumetricLighting
  // Model.start()`/`finish()`, and therefore `scatteringEmissiveNode` above —
  // unless the scene has at least one light (`lightsNode.getScope().
  // hasLights`, see NodeMaterial.js). `scatteringEmissiveNode` itself needs
  // no actual light contribution (it adds to `stepLight` unconditionally,
  // independent of any light's `direct()` callback), so a real light merely
  // needs to be PRESENT; zero intensity keeps it a no-op otherwise.
  scene.add(new THREE.AmbientLight(0xffffff, 0));

  const display = directCanvas ?? document.createElement("canvas");
  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  const ctx = PRESENTATION === "readback" ? display.getContext("2d") : null;

  const controls = new OrbitControls(camera, display);
  controls.target.copy(center);
  controls.enableDamping = !TEST_MODE;

  const rt = new THREE.RenderTarget(2, 2);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
  let flip = new Uint8ClampedArray(0);
  let vw = 0;
  let vh = 0;

  function setViewport(w: number, h: number): void {
    vw = Math.max(1, w);
    vh = Math.max(1, h);
    if (PRESENTATION === "direct") renderer.setSize(vw, vh, false);
    else {
      display.width = vw;
      display.height = vh;
      rt.setSize(vw, vh);
      flip = new Uint8ClampedArray(vw * vh * 4);
    }
    camera.aspect = vw / vh;
    camera.updateProjectionMatrix();
  }

  function placeCamera(azimuth: number, elevation: number, radiusFactor: number): void {
    const r = diag * radiusFactor;
    camera.position.set(
      center.x + r * Math.cos(elevation) * Math.cos(azimuth),
      center.y + r * Math.sin(elevation),
      center.z + r * Math.cos(elevation) * Math.sin(azimuth),
    );
    camera.lookAt(center);
    controls.update();
  }

  async function presentFrame(): Promise<void> {
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      await renderer.renderAsync(scene, camera);
      return;
    }

    (renderer as WebGPURenderer).setRenderTarget(rt);
    await (renderer as WebGPURenderer).renderAsync(scene, camera);
    const pixels = (await (renderer as WebGPURenderer).readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
    (renderer as WebGPURenderer).setRenderTarget(null);
    const stride = vw * 4;
    for (let y = 0; y < vh; y++) {
      const src = (vh - 1 - y) * stride;
      flip.set(pixels.subarray(src, src + stride), y * stride);
    }
    ctx!.putImageData(new ImageData(flip, vw, vh), 0, 0);
  }

  statsEl.textContent =
    `atlas: ${dx}x${dy}x${dz} (${atlas.range.min.toFixed(3)}..${atlas.range.max.toFixed(3)})\n` +
    `adapter: ${adapter}\n` +
    `texture: ${TEXTURE_MODE}  ·  presentation: ${PRESENTATION}  ·  ` +
    `swizzle: ${SWIZZLE_SHIM ? "on" : "off"}  ·  keepAlive: ${KEEP_ALIVE_SHIM ? "on" : "off"}\n` +
    `gridStats: min ${gpuStats.min.toFixed(4)} max ${gpuStats.max.toFixed(4)} mean ${gpuStats.mean.toFixed(4)} ` +
    `active ${gpuStats.activeVoxelCount} (fast path: ${gpuStats.usedAtomicFastPath})\n` +
    `statsOk: ${statsOk}  transformOk: ${transformOk}`;

  if (TEST_MODE) {
    panelEl.classList.add("test-hidden");
    setViewport(640, 480);
    placeCamera(Math.PI * 0.25, Math.PI * 0.16, 1.7);
    try {
      await presentFrame();
      await presentFrame();
      window.__DEMO04__ = {
        ready: true,
        statsOk,
        transformOk,
        fixture: ATLAS_FIXTURE,
        statsFixture: STATS_FIXTURE_NVDB,
        gridType: atlasInfo.type,
        adapter,
        presentation: PRESENTATION,
        textureMode: TEXTURE_MODE,
        swizzleShim: SWIZZLE_SHIM,
        keepAliveShim: KEEP_ALIVE_SHIM,
        resolution: "640x480",
        atlasMaxDim: ATLAS_MAX_DIM,
        atlasFilter: ATLAS_FILTER,
        atlasFormat: ATLAS_FORMAT,
        atlasNormalize: ATLAS_NORMALIZE,
      };
    } catch (err) {
      fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
    }
    return;
  }

  setViewport(window.innerWidth, window.innerHeight);
  placeCamera(Math.PI * 0.25, Math.PI * 0.18, 1.8);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));

  let published = false;
  let busy = false;
  const loop = (): void => {
    requestAnimationFrame(loop);
    if (busy || window.__DEMO04__?.error) return;
    busy = true;
    controls.update();
    presentFrame()
      .then(() => {
        busy = false;
        if (!published) {
          published = true;
          window.__DEMO04__ = {
            ready: true,
            statsOk,
            transformOk,
            fixture: ATLAS_FIXTURE,
            statsFixture: STATS_FIXTURE_NVDB,
            gridType: atlasInfo.type,
            adapter,
            presentation: PRESENTATION,
            textureMode: TEXTURE_MODE,
            swizzleShim: SWIZZLE_SHIM,
            keepAliveShim: KEEP_ALIVE_SHIM,
            resolution: `${vw}x${vh}`,
            atlasMaxDim: ATLAS_MAX_DIM,
            atlasFilter: ATLAS_FILTER,
            atlasFormat: ATLAS_FORMAT,
            atlasNormalize: ATLAS_NORMALIZE,
          };
        }
      })
      .catch((err: unknown) => {
        busy = false;
        fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
      });
  };
  requestAnimationFrame(loop);
}

run().catch((err: unknown) => {
  fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
});
