/**
 * Demo 07 — builder: in-browser NanoVDB authoring round-trip (Phase 5/8 demo
 * gate half 2, docs/PLAN.md — "dense -> TS-build -> render round-trip").
 *
 * Pipeline (SPEC §3.3/§4, all pure-TS `vdb-web-tools`):
 *
 *   generate a deterministic 96^3 dense fog field
 *     -> `buildFromDenseDetailed` (FLOAT NanoVDB grid image)
 *     -> optional `quantize(..., "fp8")`
 *     -> `writeNvdb([image])` (a real `.nvdb` file, in memory)
 *     -> `NanoVDBFile.fromArrayBuffer` (re-parse — the SAME loader demo 02/06
 *        use for real files, proving this isn't a special-cased shortcut)
 *     -> `NanoVDBGrid` -> `NanoVDBVolumeMaterial` render (demo 02's
 *        offscreen-RT + shims path).
 *
 * ## Verifying the round trip in-page
 *
 * `?test=1` forces a fixed seed AND disables quantization (the probe check
 * below only makes sense against an un-quantized FLOAT image — Fp8 is lossy
 * by design), then asserts two things and publishes them on
 * `window.__DEMO07__`:
 *
 *   (a) `roundTripOk` — the re-parsed file's `GridMetadata` (voxel count,
 *       index/world bbox, voxel size) agrees with what `buildFromDenseDetailed`
 *       itself computed from the same build. Note `quantize()` preserves
 *       topology/metadata/transform (see its module doc), so this equality
 *       would hold even with quantization on; the interactive quantize
 *       toggle exercises that path too, just not under the strict test-mode
 *       assertions.
 *   (b) `probesOk` — 20 deterministic, active (`value !== background`)
 *       coordinates, read back from the re-parsed image via the harness's
 *       browser-clean `readValueCpu` (see `cpu-reference.ts`'s module doc for
 *       why not `nanovdb-wgsl`'s own `cpu/*`), match the SOURCE dense array's
 *       f32 values exactly (`===`) — FLOAT is a lossless bit-for-bit
 *       representation through this entire pipeline, so exact equality (not
 *       a tolerance) is the correct bar.
 *
 * Presentation is demo 02's recipe verbatim: offscreen `RenderTarget` ->
 * `readRenderTargetPixelsAsync` -> 2D-canvas blit (canvas *presentation*
 * drops the Dawn instance under this sandbox's SwiftShader — see
 * docs/handoffs/PHASE-3.md), plus the same r185 swizzle-string shim and
 * adapter keep-alive shim.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { NanoVDBFile } from "nanovdb-wgsl";
import pnanovdbSource from "nanovdb-wgsl/pnanovdb.wgsl?raw";
import { NanoVDBGrid, NanoVDBVolumeMaterial, createVolumeRenderer } from "three-nanovdb";
import type { WebGPURenderer } from "three/webgpu";
import { buildFromDenseDetailed, quantize, writeNvdb } from "vdb-web-tools";
import type { BuiltGrid } from "vdb-web-tools";

import { readValueCpu } from "../../harness/cpu-reference";
import { parseWgslConstants } from "../../harness/wgsl-constants";
import type { ParsedWgslConstants } from "../../harness/wgsl-constants";

interface Demo07State {
  ready: boolean;
  roundTripOk?: boolean;
  probesOk?: boolean;
  error?: string;
}

declare global {
  interface Window {
    __DEMO07__?: Demo07State;
  }
}

// ---------------------------------------------------------------------------
// Compatibility shims (identical to demo 02/04/06 — see their module headers).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Deterministic procedural dense field: a fog sphere + two sine lobes.
// ---------------------------------------------------------------------------

const DIM = 96;
const DEFAULT_SEED = 1234;
const TEST_SEED = 1234; // fixed regardless of any interactive seed state

/** Small deterministic PRNG (mulberry32) — same seed always yields the same lobe phases. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A radial falloff (sphere-ish fog) modulated by two sine lobes, clamped to [0,1]. Deterministic given `seed`. */
function generateDense(dim: number, seed: number): Float32Array {
  const values = new Float32Array(dim * dim * dim);
  const rand = mulberry32(seed);
  const phase0 = rand() * Math.PI * 2;
  const phase1 = rand() * Math.PI * 2;
  const phase2 = rand() * Math.PI * 2;
  const c = (dim - 1) / 2;
  const r = dim * 0.34;

  for (let x = 0; x < dim; x++) {
    const dx = x - c;
    for (let y = 0; y < dim; y++) {
      const dy = y - c;
      for (let z = 0; z < dim; z++) {
        const dz = z - c;
        const dist = Math.hypot(dx, dy, dz);
        const base = Math.max(0, 1 - dist / r);
        const lobe1 = 0.18 * Math.sin(dx * 0.22 + phase0) * Math.sin(dy * 0.22 + phase1);
        const lobe2 = 0.12 * Math.sin(dz * 0.28 + phase2);
        const v = Math.min(1, Math.max(0, base + base * (lobe1 + lobe2)));
        values[(x * dim + y) * dim + z] = Math.fround(v);
      }
    }
  }
  return values;
}

/** 20 deterministic, active (nonzero) probe coords + their source dense values. */
function pickProbes(
  values: Float32Array,
  dim: number,
  count: number,
): Array<{ ijk: [number, number, number]; value: number }> {
  const probes: Array<{ ijk: [number, number, number]; value: number }> = [];
  const step = Math.max(1, Math.floor(dim / 8));
  for (let x = 0; x < dim && probes.length < count; x += step) {
    for (let y = 0; y < dim && probes.length < count; y += step) {
      for (let z = 0; z < dim && probes.length < count; z += step) {
        const idx = (x * dim + y) * dim + z;
        const v = values[idx]!;
        if (v !== 0) probes.push({ ijk: [x, y, z], value: v });
      }
    }
  }
  return probes;
}

// ---------------------------------------------------------------------------
// Build -> quantize? -> writeNvdb -> re-parse
// ---------------------------------------------------------------------------

interface BuiltPipeline {
  built: BuiltGrid;
  values: Float32Array;
  fileBuffer: ArrayBuffer;
  reparsedGrid: NanoVDBGrid;
}

function runPipeline(seed: number, quantizeToFp8: boolean): BuiltPipeline {
  const values = generateDense(DIM, seed);
  const built = buildFromDenseDetailed(values, [DIM, DIM, DIM], {
    gridName: "builder_demo",
    background: 0,
  });
  const image = quantizeToFp8 ? quantize(built.image, "fp8") : built.image;
  const fileBuffer = writeNvdb([image]);
  const reparsedGrid = NanoVDBGrid.fromFile(NanoVDBFile.fromArrayBuffer(fileBuffer), 0);
  return { built, values, fileBuffer, reparsedGrid };
}

/** Compares the re-parsed file's metadata against what `buildFromDenseDetailed` itself computed. */
function checkRoundTrip(built: BuiltGrid, reparsedGrid: NanoVDBGrid): boolean {
  const m = reparsedGrid.metadata;
  if (m.voxelCount !== built.voxelCount) return false;
  for (let a = 0; a < 3; a++) {
    if (m.indexBBox.min[a] !== built.indexBBox.min[a]) return false;
    if (m.indexBBox.max[a] !== built.indexBBox.max[a]) return false;
    if (Math.abs(m.worldBBox.min[a]! - built.worldBBox.min[a]!) > 1e-9) return false;
    if (Math.abs(m.worldBBox.max[a]! - built.worldBBox.max[a]!) > 1e-9) return false;
    if (Math.abs(m.voxelSize[a]! - built.voxelSize) > 1e-12) return false;
  }
  return true;
}

/** 20 CPU probes on the re-parsed image must match the source dense values exactly. */
function checkProbes(
  reparsedGrid: NanoVDBGrid,
  wc: ParsedWgslConstants,
  probes: ReadonlyArray<{ ijk: [number, number, number]; value: number }>,
): boolean {
  if (probes.length === 0) return false;
  for (const p of probes) {
    const r = readValueCpu(reparsedGrid.image, p.ijk, reparsedGrid.gridTypeId, wc);
    if (!r.active || r.value !== p.value) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// DOM / params
// ---------------------------------------------------------------------------

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";

const appEl = document.querySelector<HTMLDivElement>("#app")!;
const panelEl = document.querySelector<HTMLDivElement>("#panel")!;
const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const verifyEl = document.querySelector<HTMLDivElement>("#verify")!;
const regenBtn = document.querySelector<HTMLButtonElement>("#regen")!;
const quantizeToggle = document.querySelector<HTMLInputElement>("#quantize-toggle")!;
const downloadBtn = document.querySelector<HTMLButtonElement>("#download")!;

function fail(message: string): void {
  errEl.textContent = message;
  statsEl.textContent = "failed";
  window.__DEMO07__ = { ready: false, error: message };
}

// ---------------------------------------------------------------------------
// Renderer bootstrap (device-first, created once and reused across rebuilds)
// ---------------------------------------------------------------------------

let renderer: WebGPURenderer;
let device: GPUDevice;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let display: HTMLCanvasElement;
let ctx2d: CanvasRenderingContext2D;
let rt: THREE.RenderTarget;
let flip = new Uint8ClampedArray(0);
let vw = 0;
let vh = 0;

let currentMesh: THREE.Mesh | undefined;
let currentMeshMaterial: NanoVDBVolumeMaterial | undefined;
let currentFileBuffer: ArrayBuffer | undefined;
let downloadUrl: string | undefined;

async function initRenderer(): Promise<void> {
  ({ renderer, device } = await createVolumeRenderer({}));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14101f);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);

  display = document.createElement("canvas");
  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  ctx2d = display.getContext("2d")!;

  controls = new OrbitControls(camera, display);
  controls.enableDamping = !TEST_MODE;

  rt = new THREE.RenderTarget(2, 2);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
}

function setViewport(w: number, h: number): void {
  vw = Math.max(1, w);
  vh = Math.max(1, h);
  display.width = vw;
  display.height = vh;
  rt.setSize(vw, vh);
  flip = new Uint8ClampedArray(vw * vh * 4);
  camera.aspect = vw / vh;
  camera.updateProjectionMatrix();
}

function placeCamera(center: THREE.Vector3, diag: number): void {
  camera.near = Math.max(0.001, diag * 0.001);
  camera.far = diag * 100;
  const r = diag * 1.8;
  camera.position.set(center.x + r * 0.65, center.y + r * 0.45, center.z + r * 0.65);
  camera.updateProjectionMatrix();
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
}

async function presentFrame(): Promise<void> {
  renderer.setRenderTarget(rt);
  await renderer.renderAsync(scene, camera);
  const pixels = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
  renderer.setRenderTarget(null);
  const stride = vw * 4;
  for (let y = 0; y < vh; y++) {
    const src = (vh - 1 - y) * stride;
    flip.set(pixels.subarray(src, src + stride), y * stride);
  }
  ctx2d.putImageData(new ImageData(flip, vw, vh), 0, 0);
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

let seed = DEFAULT_SEED;

async function rebuildAndRender(opts: { seed: number; quantizeToFp8: boolean; verify: boolean }): Promise<void> {
  statsEl.textContent = "building…";
  errEl.textContent = "";

  const { built, values, fileBuffer, reparsedGrid } = runPipeline(opts.seed, opts.quantizeToFp8);
  currentFileBuffer = fileBuffer;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([fileBuffer], { type: "application/octet-stream" }));

  let roundTripOk: boolean | undefined;
  let probesOk: boolean | undefined;
  if (opts.verify) {
    const wc = parseWgslConstants(pnanovdbSource);
    roundTripOk = checkRoundTrip(built, reparsedGrid);
    const probes = pickProbes(values, DIM, 20);
    probesOk = checkProbes(reparsedGrid, wc, probes);
    verifyEl.innerHTML =
      `<div class="${roundTripOk ? "ok" : "bad"}">round trip: ${roundTripOk ? "OK" : "MISMATCH"}</div>` +
      `<div class="${probesOk ? "ok" : "bad"}">${probes.length} probes: ${probesOk ? "OK" : "MISMATCH"}</div>`;
  }

  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.geometry.dispose();
    currentMeshMaterial?.dispose();
  }
  const material = new NanoVDBVolumeMaterial({ grid: reparsedGrid, pnanovdbSource, jitter: !TEST_MODE });
  const mesh = new THREE.Mesh(reparsedGrid.proxyGeometry(), material);
  scene.add(mesh);
  currentMesh = mesh;
  currentMeshMaterial = material;

  const worldBox = reparsedGrid.worldBBox();
  const center = worldBox.getCenter(new THREE.Vector3());
  const size = worldBox.getSize(new THREE.Vector3());
  const diag = Math.max(size.length(), 1e-3);
  placeCamera(center, diag);

  statsEl.textContent =
    `seed ${opts.seed}  ·  ${opts.quantizeToFp8 ? "Fp8" : "Float"}  ·  ` +
    `voxels ${built.voxelCount.toLocaleString()}  ·  leaves ${built.nodeCounts.leaf.toLocaleString()}\n` +
    `.nvdb size: ${(fileBuffer.byteLength / 1024).toFixed(1)} KB`;

  if (TEST_MODE) {
    setViewport(640, 480);
    await presentFrame();
    await presentFrame();
    const state: Demo07State = { ready: true };
    if (roundTripOk !== undefined) state.roundTripOk = roundTripOk;
    if (probesOk !== undefined) state.probesOk = probesOk;
    window.__DEMO07__ = state;
  }
}

// ---------------------------------------------------------------------------
// Interactive wiring
// ---------------------------------------------------------------------------

let rebuilding = false;
async function triggerRebuild(): Promise<void> {
  if (rebuilding) return;
  rebuilding = true;
  try {
    await rebuildAndRender({ seed, quantizeToFp8: quantizeToggle.checked, verify: false });
  } catch (err) {
    fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
  } finally {
    rebuilding = false;
  }
}

function wireInteractiveControls(): void {
  regenBtn.addEventListener("click", () => {
    seed += 1;
    void triggerRebuild();
  });
  quantizeToggle.addEventListener("change", () => void triggerRebuild());
  downloadBtn.addEventListener("click", () => {
    if (!downloadUrl || !currentFileBuffer) return;
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = "builder_demo.nvdb";
    a.click();
  });
}

let frameBusy = false;
function startRenderLoop(): void {
  const loop = (): void => {
    requestAnimationFrame(loop);
    if (frameBusy || !currentMesh || window.__DEMO07__?.error) return;
    frameBusy = true;
    controls.update();
    presentFrame()
      .then(() => {
        frameBusy = false;
      })
      .catch((err: unknown) => {
        frameBusy = false;
        fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
      });
  };
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function run(): Promise<void> {
  if (!navigator.gpu) {
    fail("WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.");
    return;
  }
  installSwizzleCompatShim();
  installAdapterKeepAliveShim();
  await initRenderer();

  if (TEST_MODE) {
    panelEl.classList.add("test-hidden");
    await rebuildAndRender({ seed: TEST_SEED, quantizeToFp8: false, verify: true });
    return;
  }

  wireInteractiveControls();
  setViewport(window.innerWidth, window.innerHeight);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));
  await rebuildAndRender({ seed, quantizeToFp8: quantizeToggle.checked, verify: true });
  startRenderLoop();
}

run().catch((err: unknown) => {
  fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
});
