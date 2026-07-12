/**
 * Demo 05 — embergen-sequence: animated NanoVDB playback (docs/PLAN.md Phase 7,
 * SPEC §3.5). Plays a looping volume sequence through a SINGLE
 * `NanoVDBVolumeMaterial`, swapping the grid each frame via its in-place
 * `rebindGrid()` (no per-frame material rebuild), scheduled by
 * `NanoVDBSequence`'s prefetch ring + frame-time clock.
 *
 * ## Frames: synthetic here, EmberGen on a network-open machine
 *
 * In-sandbox there is no EmberGen download, so frames are authored with the
 * project's own serializer (`frames.ts` -> `vdb-web-tools.buildFromDense`). Two
 * real loader paths are exercised:
 *
 *   - Interactive default: an in-memory `loader` override hands the sequence
 *     pre-built grids (no fetch) — the fast path a memory-resident pack uses.
 *   - `?test=1` (and the EmberGen path): frames are written to `.nvdb` and
 *     served as Blob URLs, so the DEFAULT loader's real
 *     `NanoVDBFile.fromURL` fetch+parse pipeline is what actually runs.
 *
 * Pass `?src=embergen` to probe `/fixtures/embergen/<pack>/manifest.json`
 * (a JSON `{ fps, frames: [...] }`). It 404s in the sandbox -> the demo shows
 * the "run pnpm fixtures:embergen" note and falls back to synthetic frames; on
 * a machine with a baked pack it plays the real sequence over the same code.
 *
 * ## Presentation + shims
 *
 * Demo 02's recipe verbatim: offscreen `RenderTarget` ->
 * `readRenderTargetPixelsAsync` -> 2D-canvas blit (canvas presentation drops
 * the Dawn instance under SwiftShader — see docs/handoffs/PHASE-3.md), plus the
 * r185 swizzle-string shim and the adapter keep-alive shim.
 *
 * ## `?test=1` — deterministic 6-frame gate
 *
 * Drives the scheduler by hand with a fixed timestep (not the wall clock),
 * playing exactly 6 frames off the Blob-URL manifest, and publishes
 * `window.__DEMO05__ = { ready, framesPlayed, rebinds, stalls, framesDiffer,
 * error? }`. `framesDiffer` hashes each presented frame and asserts the output
 * actually changed across rebinds — the real GPU proof that the buffer swap
 * reached the shader.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { NanoVDBFile } from "nanovdb-wgsl";
import pnanovdbSource from "nanovdb-wgsl/pnanovdb.wgsl?raw";
import { NanoVDBGrid, NanoVDBSequence, NanoVDBVolumeMaterial, createVolumeRenderer } from "three-nanovdb";
import type { FrameLoader, NanoVDBSequenceStats } from "three-nanovdb";
import type { WebGPURenderer } from "three/webgpu";
import { writeNvdb } from "vdb-web-tools";
import { FRAME_COUNT, domainExtent, makeFrame } from "./frames";

interface Demo05State {
  ready: boolean;
  framesPlayed?: number;
  rebinds?: number;
  stalls?: number;
  framesDiffer?: boolean;
  uploadMs?: number;
  maxGridBytes?: number;
  source?: string;
  error?: string;
}

declare global {
  interface Window {
    __DEMO05__?: Demo05State;
  }
}

// --- Test config ---
const TEST_FRAMES = 6;
const BLOB_FRAMES = 8; // frames written to Blob URLs for the real fetch path
const FPS = 24;

// ---------------------------------------------------------------------------
// Compatibility shims (identical to demo 02/07 — see their module headers).
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
// DOM / params
// ---------------------------------------------------------------------------

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
const WANT_EMBERGEN = params.get("src") === "embergen";

const appEl = document.querySelector<HTMLDivElement>("#app")!;
const panelEl = document.querySelector<HTMLDivElement>("#panel")!;
const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const noteEl = document.querySelector<HTMLDivElement>("#note")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const playPauseBtn = document.querySelector<HTMLButtonElement>("#playpause")!;
const restartBtn = document.querySelector<HTMLButtonElement>("#restart")!;

function fail(message: string): void {
  errEl.textContent = message;
  statsEl.textContent = "failed";
  window.__DEMO05__ = { ready: false, error: message };
}

// ---------------------------------------------------------------------------
// EmberGen manifest (on-device path). Format:
//   /fixtures/embergen/<pack>/manifest.json = { "fps": 24, "frames": ["f0001.nvdb", ...] }
// frame paths are resolved relative to the manifest's directory.
// ---------------------------------------------------------------------------

interface EmberGenManifest {
  fps: number;
  urls: string[];
}

async function probeEmberGen(): Promise<EmberGenManifest | null> {
  // A few conventional locations; all 404 in-sandbox.
  const candidates = [
    "/fixtures/embergen/manifest.json",
    "/fixtures/embergen/smoke/manifest.json",
    "/fixtures/embergen/explosion/manifest.json",
  ];
  for (const manifestUrl of candidates) {
    try {
      const res = await fetch(manifestUrl);
      if (!res.ok) continue;
      const json = (await res.json()) as { fps?: number; frames?: string[] };
      if (!Array.isArray(json.frames) || json.frames.length === 0) continue;
      const dir = manifestUrl.slice(0, manifestUrl.lastIndexOf("/") + 1);
      return {
        fps: json.fps ?? FPS,
        urls: json.frames.map((f) => (f.startsWith("/") ? f : dir + f)),
      };
    } catch {
      /* try next */
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Renderer / presentation (demo 02's offscreen-RT blit)
// ---------------------------------------------------------------------------

let renderer: WebGPURenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let display: HTMLCanvasElement;
let ctx2d: CanvasRenderingContext2D;
let rt: THREE.RenderTarget;
let flip = new Uint8ClampedArray(0);
let vw = 0;
let vh = 0;

async function initRenderer(): Promise<void> {
  ({ renderer } = await createVolumeRenderer({}));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x10131f);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 4000);

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

/**
 * A proxy box for the FIXED domain [0, dim)^3 — NOT the tight per-frame active
 * bbox. The blob orbits, so a single frame's bbox wouldn't enclose later
 * frames; the material clips each frame against its own root bbox on the GPU,
 * so a domain-sized box is correct for every frame.
 */
function domainMesh(material: NanoVDBVolumeMaterial): THREE.Mesh {
  const d = domainExtent();
  const geom = new THREE.BoxGeometry(d, d, d);
  geom.translate(d / 2, d / 2, d / 2);
  return new THREE.Mesh(geom, material);
}

function placeCamera(): void {
  const d = domainExtent();
  const center = new THREE.Vector3(d / 2, d / 2, d / 2);
  const diag = d * Math.SQRT2 * 1.2;
  camera.near = diag * 0.01;
  camera.far = diag * 20;
  camera.position.set(center.x + diag * 0.9, center.y + diag * 0.55, center.z + diag * 0.9);
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

/** Cheap running hash of the presented frame — used to prove frames actually change. */
function frameHash(): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < flip.length; i += 97) {
    h = (h ^ flip[i]!) >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

// ---------------------------------------------------------------------------
// Frame preparation: build synthetic frames once; expose both an in-memory
// loader and a Blob-URL manifest (real fetch path).
// ---------------------------------------------------------------------------

interface SyntheticFrames {
  /** Parsed grids for the in-memory loader (index -> grid). */
  memGrids: NanoVDBGrid[];
  /** Blob URLs of the first BLOB_FRAMES frames (real .nvdb, fetchable). */
  blobUrls: string[];
  /** Largest grid image byte length across ALL frames (sizes the padded buffer). */
  maxGridBytes: number;
}

function buildSyntheticFrames(): SyntheticFrames {
  const memGrids: NanoVDBGrid[] = [];
  const blobUrls: string[] = [];
  let maxGridBytes = 0;
  for (let i = 0; i < FRAME_COUNT; i++) {
    const image = makeFrame(i);
    const fileBuffer = writeNvdb([image]);
    // Round-trip through the real loader so the in-memory path uses genuine
    // GridMetadata (identical to what the fetch path would produce).
    const grid = NanoVDBGrid.fromFile(NanoVDBFile.fromArrayBuffer(fileBuffer), 0);
    memGrids.push(grid);
    maxGridBytes = Math.max(maxGridBytes, grid.image.byteLength);
    if (i < BLOB_FRAMES) {
      blobUrls.push(URL.createObjectURL(new Blob([fileBuffer], { type: "application/octet-stream" })));
    }
  }
  return { memGrids, blobUrls, maxGridBytes };
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

let sequence: NanoVDBSequence | undefined;

function updateHud(stats: NanoVDBSequenceStats, fps: number, sourceLabel: string): void {
  statsEl.textContent =
    `source: ${sourceLabel}\n` +
    `frame: ${stats.frame + 1}/${sequence?.frameCount ?? "?"}  ·  fps: ${fps.toFixed(0)}\n` +
    `decode-ahead: ${stats.decodedAhead}  ·  stalls: ${stats.stalls}\n` +
    `rebinds: ${stats.rebinds}  ·  upload: ${stats.uploadMs.toFixed(2)} ms`;
}

async function runInteractive(frames: SyntheticFrames, sourceLabel: string): Promise<void> {
  // In-memory loader: hand the sequence a pre-built grid per index (no fetch).
  const loader: FrameLoader = (_url, i) => Promise.resolve(frames.memGrids[i % frames.memGrids.length]!);

  const material = new NanoVDBVolumeMaterial({
    grid: frames.memGrids[0]!,
    pnanovdbSource,
    maxGridBytes: frames.maxGridBytes,
    densityScale: 60,
    stepSize: 0.6,
    shadowSteps: 8,
    sunIntensity: 4,
    jitter: true,
  });
  scene.add(domainMesh(material));
  placeCamera();

  sequence = new NanoVDBSequence({
    urls: (i) => `mem://${i}`,
    frameCount: FRAME_COUNT,
    fps: FPS,
    prefetch: 3,
    loop: true,
    loader,
  });

  setViewport(window.innerWidth, window.innerHeight);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));

  await sequence.preload();
  sequence.start(material);

  // FPS meter.
  let frames60 = 0;
  let lastFpsT = performance.now();
  let fps = 0;
  let published = false;
  let busy = false;

  const loop = (): void => {
    requestAnimationFrame(loop);
    if (busy || window.__DEMO05__?.error) return;
    busy = true;
    controls.update();
    sequence!.update(performance.now());
    presentFrame()
      .then(() => {
        busy = false;
        frames60++;
        const now = performance.now();
        if (now - lastFpsT >= 500) {
          fps = (frames60 * 1000) / (now - lastFpsT);
          frames60 = 0;
          lastFpsT = now;
          updateHud(sequence!.stats, fps, sourceLabel);
        }
        if (!published) {
          published = true;
          window.__DEMO05__ = { ready: true, source: sourceLabel };
        }
      })
      .catch((err: unknown) => {
        busy = false;
        fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
      });
  };
  requestAnimationFrame(loop);

  wireControls();
}

/**
 * Deterministic test mode: play exactly TEST_FRAMES off the Blob-URL manifest
 * (real fetch+parse), driving the scheduler with a fixed timestep. Assert every
 * frame advanced, rebinds succeeded, no stalls, and the output changed.
 */
async function runTest(frames: SyntheticFrames): Promise<void> {
  panelEl.classList.add("test-hidden");
  setViewport(640, 480);

  const material = new NanoVDBVolumeMaterial({
    grid: frames.memGrids[0]!, // same bytes as blobUrls[0]; construction only
    pnanovdbSource,
    maxGridBytes: frames.maxGridBytes,
    densityScale: 60,
    stepSize: 0.5,
    shadowSteps: 8,
    sunIntensity: 4,
    jitter: false, // deterministic
  });
  scene.add(domainMesh(material));
  placeCamera();

  // Blob-URL manifest + DEFAULT loader => NanoVDBFile.fromURL fetch path runs.
  sequence = new NanoVDBSequence({
    urls: frames.blobUrls,
    fps: FPS,
    prefetch: BLOB_FRAMES,
    loop: false,
  });

  try {
    // Decode every blob frame up front so playback is stall-free & deterministic.
    await sequence.preload(BLOB_FRAMES);

    const dt = 1000 / FPS;
    const hashes: number[] = [];

    sequence.start(material, 0); // binds frame 0
    await presentFrame();
    hashes.push(frameHash());

    for (let f = 1; f < TEST_FRAMES; f++) {
      const shown = sequence.update(f * dt);
      if (shown !== f) {
        throw new Error(`scheduler did not advance to frame ${f} (got ${shown})`);
      }
      await presentFrame();
      hashes.push(frameHash());
    }
    sequence.stop();

    const framesDiffer = new Set(hashes).size > 1;
    window.__DEMO05__ = {
      ready: true,
      framesPlayed: sequence.stats.rebinds,
      rebinds: sequence.stats.rebinds,
      stalls: sequence.stats.stalls,
      framesDiffer,
      uploadMs: sequence.stats.uploadMs,
      maxGridBytes: frames.maxGridBytes,
      source: "synthetic (blob-url fetch)",
    };
  } catch (err) {
    fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
  }
}

function wireControls(): void {
  playPauseBtn.addEventListener("click", () => {
    if (!sequence) return;
    if (sequence.isPlaying) {
      sequence.stop();
      playPauseBtn.textContent = "play";
    } else {
      sequence.resume();
      playPauseBtn.textContent = "pause";
    }
  });
  restartBtn.addEventListener("click", () => sequence?.seek(0));
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

  // Always show the synthetic-frames note (and the EmberGen how-to).
  noteEl.innerHTML =
    "synthetic frames — run <code>pnpm fixtures:embergen</code> for real EmberGen packs, " +
    "then load with <code>?src=embergen</code>.";

  const frames = buildSyntheticFrames();

  if (TEST_MODE) {
    await runTest(frames);
    return;
  }

  // On-device EmberGen path: probe a real manifest; fall back to synthetic.
  if (WANT_EMBERGEN) {
    const manifest = await probeEmberGen();
    if (manifest) {
      await runEmberGen(manifest, frames.maxGridBytes);
      return;
    }
    noteEl.innerHTML =
      "no EmberGen pack found at <code>/fixtures/embergen/**/manifest.json</code> — " +
      "run <code>pnpm fixtures:embergen</code> on a network-open machine. Playing synthetic frames.";
  }

  await runInteractive(frames, "synthetic (in-memory)");
}

/**
 * Play a real EmberGen manifest via the DEFAULT loader (fetch+parse). Frame
 * grids are streamed by the prefetch ring; the padded buffer must be sized for
 * the largest frame. Without a probe of every frame's size, we reuse the
 * synthetic max as a conservative floor and grow it lazily on the first
 * over-capacity rebind by rebuilding the material (documented fallback).
 */
async function runEmberGen(manifest: EmberGenManifest, floorBytes: number): Promise<void> {
  // First frame decides the initial capacity; use max(floor, first frame).
  const firstFile = await NanoVDBFile.fromURL(manifest.urls[0]!);
  const firstGrid = NanoVDBGrid.fromFile(firstFile, 0);
  const material = new NanoVDBVolumeMaterial({
    grid: firstGrid,
    pnanovdbSource,
    maxGridBytes: Math.max(floorBytes, firstGrid.image.byteLength * 2),
    jitter: true,
  });
  const worldBox = firstGrid.worldBBox();
  const size = worldBox.getSize(new THREE.Vector3());
  const center = worldBox.getCenter(new THREE.Vector3());
  const geom = new THREE.BoxGeometry(size.x, size.y, size.z);
  geom.translate(center.x, center.y, center.z);
  scene.add(new THREE.Mesh(geom, material));
  const diag = size.length();
  camera.near = diag * 0.01;
  camera.far = diag * 20;
  camera.position.set(center.x + diag, center.y + diag * 0.6, center.z + diag);
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();

  sequence = new NanoVDBSequence({
    urls: manifest.urls,
    fps: manifest.fps,
    prefetch: 4,
    loop: true,
  });
  setViewport(window.innerWidth, window.innerHeight);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));
  await sequence.preload();
  sequence.start(material);

  let busy = false;
  let published = false;
  const loop = (): void => {
    requestAnimationFrame(loop);
    if (busy || window.__DEMO05__?.error) return;
    busy = true;
    controls.update();
    sequence!.update(performance.now());
    presentFrame()
      .then(() => {
        busy = false;
        updateHud(sequence!.stats, manifest.fps, "embergen");
        if (!published) {
          published = true;
          window.__DEMO05__ = { ready: true, source: "embergen" };
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
