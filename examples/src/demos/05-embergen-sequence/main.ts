/**
 * Demo 05 — embergen-sequence: animated NanoVDB playback (docs/archive/v1/PLAN.md Phase 7,
 * SPEC §3.5) on the K2 three adapter. Plays a looping volume sequence through
 * ONE pool grid and ONE `VDBVolumeMaterial`: `VDBSequence` swaps each frame in
 * with `pool.replace(id, frame)`, scheduled by its prefetch ring + frame-time
 * clock.
 *
 * ## The pool contract
 *
 * The grid is added with `capacity` = the largest frame, so every `replace`
 * writes in place: no move, no `pool.version` change, no `VDBGridNode` rebind,
 * no new bind group. Each rendered frame runs `sequence.update(now)` (may
 * `replace`), `pool.flush()` (writes it), `grids.update()` (rebinds only after
 * a growth), then three renders.
 *
 * ## Frames: synthetic here, EmberGen on a network-open machine
 *
 * In-sandbox there is no EmberGen download, so frames are authored with the
 * project's own serializer (`frames.ts` -> `@pmndrs/vdb/tools.vdbBuildFromDense`). Two
 * real loader paths are exercised:
 *
 *   - Interactive default: an in-memory `loader` override hands the sequence
 *     pre-built frame images (no fetch) — the fast path a memory-resident pack uses.
 *   - `?test=1`: frames are written to `.nvdb` and served as Blob URLs, read by
 *     the default loader (`NvdbFile.fromURL` fetch+parse).
 *   - EmberGen manifests resolve real fixture URLs.
 *
 * Pass `?src=embergen` to probe `/fixtures/embergen/<pack>/manifest.json`
 * (a JSON `{ kind: "temporal-sequence", fps, frames: [...] }`). If unavailable, the demo shows
 * the "run pnpm fixtures:embergen" note and falls back to synthetic frames; on
 * a machine with a baked pack it plays the real sequence over the same code.
 *
 * ## Presentation
 *
 * Default (`?presentation=direct`): three renders into the page's canvas.
 * `?presentation=readback` renders offscreen, reads back and draws on a 2D
 * canvas; the e2e spec uses it (headless SwiftShader has no compositor,
 * DIVERGES V7) and `?test=1` requires it (it hashes the read-back pixels).
 * `?validate=cycle` preloads a real manifest, advances a fixed clock through
 * every frame and back to frame 0, then publishes the full-cycle evidence.
 *
 * ## `?test=1` — deterministic 6-frame gate
 *
 * Drives the scheduler by hand with a fixed timestep (not the wall clock),
 * playing exactly 6 frames off the Blob-URL manifest, and publishes
 * `window.__DEMO05__ = { ready, framesPlayed, swaps, stalls, framesDiffer,
 * versionChanges, nodeRebinds, error? }`. `framesDiffer` hashes each presented
 * frame and asserts the output changed across swaps: the GPU proof that the
 * in-place `replace` reached the shader; `versionChanges` and `nodeRebinds`
 * are 0 when the reserved capacity held every frame.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { NvdbFile } from "@pmndrs/vdb/tools";
import { VDBGridNode, VDBSequence, VDBVolumeMaterial, getDevice } from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import type { VDBFrameLoader, VDBSequenceStats } from "@pmndrs/vdb/three";
import type { WebGPURenderer } from "three/webgpu";
import { writeNvdb } from "@pmndrs/vdb/tools";
import { FRAME_COUNT, domainExtent, makeFrame } from "./frames";

interface Demo05State {
  ready: boolean;
  framesPlayed?: number;
  swaps?: number;
  /** `pool.version` changes during playback (0: every frame fit the reserved capacity). */
  versionChanges?: number;
  /** `grids.update()` calls that rebound during playback (0 with no version change). */
  nodeRebinds?: number;
  stalls?: number;
  framesDiffer?: boolean;
  frameCount?: number;
  framesValidated?: number;
  cyclesCompleted?: number;
  finalFrame?: number;
  validationMode?: string;
  replaceMs?: number;
  maxFrameBytes?: number;
  source?: string;
  adapter?: string;
  presentation?: string;
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
// DOM / params
// ---------------------------------------------------------------------------

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
const WANT_EMBERGEN = params.get("src") === "embergen" || params.has("manifest");
const VALIDATE_CYCLE = params.get("validate") === "cycle";
const PRESENTATION = params.get("presentation") === "readback" ? "readback" : "direct";

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
//   /fixtures/embergen/<pack>/manifest.json =
//     { "kind": "temporal-sequence", "fps": 24, "frames": ["f0001.nvdb", ...] }
// frame paths are resolved relative to the manifest's directory.
// ---------------------------------------------------------------------------

interface EmberGenManifest {
  fps: number;
  urls: string[];
  manifestUrl: string;
}

async function probeEmberGen(): Promise<EmberGenManifest | null> {
  const requestedManifest = params.get("manifest");
  const candidates = requestedManifest ? [requestedManifest] : [];

  // Sequence baking writes an index so the browser does not need directory
  // listing support from the fixture server.
  try {
    const response = await fetch("/fixtures/embergen/index.json");
    if (response.ok) {
      const index = (await response.json()) as { manifests?: unknown };
      if (Array.isArray(index.manifests)) {
        for (const manifest of index.manifests) {
          if (typeof manifest === "string") candidates.push(`/fixtures/embergen/${manifest}`);
        }
      }
    }
  } catch {
    // Conventional paths below remain useful for manually prepared packs.
  }

  candidates.push(
    "/fixtures/embergen/manifest.json",
    "/fixtures/embergen/smoke/manifest.json",
    "/fixtures/embergen/explosion/manifest.json",
  );

  for (const candidate of [...new Set(candidates)]) {
    const manifestUrl = new URL(candidate, location.href).href;
    try {
      const res = await fetch(manifestUrl);
      if (!res.ok) continue;
      const json = (await res.json()) as { kind?: unknown; fps?: unknown; frames?: unknown };
      if (json.kind !== "temporal-sequence")
        throw new Error(`manifest must declare kind "temporal-sequence": ${manifestUrl}`);
      if (!Number.isFinite(json.fps) || Number(json.fps) <= 0)
        throw new Error(`manifest fps must be positive: ${manifestUrl}`);
      if (!Array.isArray(json.frames) || json.frames.length === 0)
        throw new Error(`manifest frames must be a non-empty array: ${manifestUrl}`);
      if (!json.frames.every((frame) => typeof frame === "string" && frame.length > 0))
        throw new Error(`manifest contains an invalid frame path: ${manifestUrl}`);

      return {
        fps: Number(json.fps),
        urls: json.frames.map((frame) => new URL(frame, manifestUrl).href),
        manifestUrl,
      };
    } catch (error) {
      if (requestedManifest)
        throw new Error(
          `Could not load requested EmberGen manifest ${manifestUrl}: ${error instanceof Error ? error.message : String(error)}`,
        );
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Renderer, pool, presentation
// ---------------------------------------------------------------------------

let renderer: WebGPURenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let display: HTMLCanvasElement;
let ctx2d: CanvasRenderingContext2D | null;
let rt: THREE.RenderTarget;
let flip = new Uint8ClampedArray(0);
let vw = 0;
let vh = 0;
let adapter = "unreported adapter";
let pool: VDBGridPool;
let grids: VDBGridNode | undefined;
/** `grids.update()` calls that rebound (a pool growth moved the buffer). */
let nodeRebinds = 0;

async function initRenderer(): Promise<void> {
  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const created = await createDeviceRenderer(directCanvas ? { canvas: directCanvas } : {});
  renderer = created.renderer;
  adapter = created.adapterName || adapter;
  pool = await VDBGridPool.create(getDevice(renderer), { label: "demo 05 pool" });
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x10131f);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 4000);

  display = directCanvas ?? document.createElement("canvas");
  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  ctx2d = PRESENTATION === "readback" ? display.getContext("2d") : null;

  controls = new OrbitControls(camera, display);
  controls.enableDamping = !TEST_MODE;

  rt = new THREE.RenderTarget(2, 2);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
}

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

/**
 * A proxy box for the FIXED domain [0, dim)^3 — NOT the tight per-frame active
 * bbox. The blob orbits, so a single frame's bbox wouldn't enclose later
 * frames; the material clips each frame against its own record bbox on the
 * GPU, so a domain-sized box is correct for every frame.
 */
function domainMesh(material: VDBVolumeMaterial): THREE.Mesh {
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

/**
 * One frame: write what `sequence.update` staged (`pool.flush`, submitted
 * before three's render), rebind the node if the pool grew, then render.
 */
async function presentFrame(): Promise<void> {
  pool.flush();
  if (grids?.update()) nodeRebinds++;

  if (PRESENTATION === "direct") {
    renderer.setRenderTarget(null);
    await renderer.renderAsync(scene, camera);
    return;
  }

  renderer.setRenderTarget(rt);
  await renderer.renderAsync(scene, camera);
  const pixels = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
  renderer.setRenderTarget(null);
  // Bottom-up rows, padded to WebGPU's 256-byte row alignment (see demo 02).
  const packedStride = vw * 4;
  const sourceStride = vh > 1 ? (pixels.byteLength - packedStride) / (vh - 1) : packedStride;
  for (let y = 0; y < vh; y++) {
    const src = (vh - 1 - y) * sourceStride;
    flip.set(pixels.subarray(src, src + packedStride), y * packedStride);
  }
  ctx2d!.putImageData(new ImageData(flip, vw, vh), 0, 0);
}

/**
 * Adds the sequence's grid with room for its largest frame, and makes the
 * pool node + material. One VDBGridNode per pool (one module per shader).
 */
async function makeVolume(
  first: Uint32Array,
  maxFrameBytes: number,
  params: ConstructorParameters<typeof VDBVolumeMaterial>[3],
): Promise<{ id: number; material: VDBVolumeMaterial }> {
  const id = pool.add(first, { capacity: maxFrameBytes, label: "sequence" });
  pool.flush();
  await pool.settled();
  grids = new VDBGridNode(renderer, pool, { helpers: ["rayClipSupport"] });
  return { id, material: new VDBVolumeMaterial(renderer, grids, id, params) };
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
  /** Parsed frame images for the in-memory loader (index -> image). */
  memImages: Uint32Array[];
  /** Blob URLs of the first BLOB_FRAMES frames (real .nvdb, fetchable). */
  blobUrls: string[];
  /** Largest frame image byte length across ALL frames (the pool capacity to reserve). */
  maxFrameBytes: number;
}

function buildSyntheticFrames(): SyntheticFrames {
  const memImages: Uint32Array[] = [];
  const blobUrls: string[] = [];
  let maxFrameBytes = 0;
  for (let i = 0; i < FRAME_COUNT; i++) {
    const fileBuffer = writeNvdb([makeFrame(i)]);
    // Round-trip through the real loader so the in-memory path uses the same
    // bytes the fetch path produces.
    const image = NvdbFile.fromArrayBuffer(fileBuffer).gridImage(0);
    memImages.push(image);
    maxFrameBytes = Math.max(maxFrameBytes, image.byteLength);
    if (i < BLOB_FRAMES) {
      blobUrls.push(URL.createObjectURL(new Blob([fileBuffer], { type: "application/octet-stream" })));
    }
  }
  return { memImages, blobUrls, maxFrameBytes };
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

let sequence: VDBSequence | undefined;

function updateHud(stats: VDBSequenceStats, fps: number, sourceLabel: string): void {
  statsEl.textContent =
    `source: ${sourceLabel}\n` +
    `frame: ${stats.frame + 1}/${sequence?.frameCount ?? "?"}  ·  fps: ${fps.toFixed(0)}\n` +
    `decode-ahead: ${stats.decodedAhead}  ·  stalls: ${stats.stalls}\n` +
    `swaps: ${stats.swaps}  ·  replace: ${stats.replaceMs.toFixed(2)} ms  ·  pool v${pool.version}`;
}

async function runInteractive(frames: SyntheticFrames, sourceLabel: string): Promise<void> {
  // In-memory loader: hand the sequence a pre-built frame image per index (no fetch).
  const loader: VDBFrameLoader = (_url, i) => Promise.resolve(frames.memImages[i % frames.memImages.length]!);

  const { id, material } = await makeVolume(frames.memImages[0]!, frames.maxFrameBytes, {
    densityScale: 60,
    stepSize: 0.6,
    shadowSteps: 8,
    sunIntensity: 4,
    jitter: true,
  });
  scene.add(domainMesh(material));
  placeCamera();

  sequence = new VDBSequence({
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
  sequence.start({ pool, id });

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
          window.__DEMO05__ = {
            ready: true,
            source: sourceLabel,
            adapter,
            presentation: PRESENTATION,
          };
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
 * frame advanced, every swap stayed in place, no stalls, and the output changed.
 */
async function runTest(frames: SyntheticFrames): Promise<void> {
  panelEl.classList.add("test-hidden");
  setViewport(640, 480);

  // Same bytes as blobUrls[0]; the sequence replaces it with the fetched frame 0.
  const { id, material } = await makeVolume(frames.memImages[0]!, frames.maxFrameBytes, {
    densityScale: 60,
    stepSize: 0.5,
    shadowSteps: 8,
    sunIntensity: 4,
    jitter: false, // deterministic
  });
  scene.add(domainMesh(material));
  placeCamera();

  // Blob-URL manifest + DEFAULT loader => NvdbFile.fromURL fetch path runs.
  sequence = new VDBSequence({
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

    const versionAtStart = pool.version;
    nodeRebinds = 0;
    let versionChanges = 0;
    let lastVersion = versionAtStart;
    const trackVersion = (): void => {
      if (pool.version !== lastVersion) versionChanges++;
      lastVersion = pool.version;
    };

    sequence.start({ pool, id }, 0); // replaces with frame 0
    await presentFrame();
    trackVersion();
    hashes.push(frameHash());

    for (let f = 1; f < TEST_FRAMES; f++) {
      const shown = sequence.update(f * dt);
      if (shown !== f) {
        throw new Error(`scheduler did not advance to frame ${f} (got ${shown})`);
      }
      await presentFrame();
      trackVersion();
      hashes.push(frameHash());
    }
    sequence.stop();

    const framesDiffer = new Set(hashes).size > 1;
    window.__DEMO05__ = {
      ready: true,
      framesPlayed: sequence.stats.swaps,
      swaps: sequence.stats.swaps,
      stalls: sequence.stats.stalls,
      framesDiffer,
      versionChanges,
      nodeRebinds,
      replaceMs: sequence.stats.replaceMs,
      maxFrameBytes: frames.maxFrameBytes,
      source: "synthetic (blob-url fetch)",
      adapter,
      presentation: PRESENTATION,
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
  if (TEST_MODE && PRESENTATION !== "readback")
    throw new Error("test=1 requires presentation=readback for frame hash verification");
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
      await runEmberGen(manifest);
      return;
    }
    noteEl.innerHTML =
      "no EmberGen pack found at <code>/fixtures/embergen/**/manifest.json</code> — " +
      "run <code>pnpm fixtures:embergen</code> on a network-open machine. Playing synthetic frames.";
  }

  await runInteractive(frames, "synthetic (in-memory)");
}

/**
 * Play a real EmberGen manifest. Every frame is fetched and validated up front
 * (type, bounds, the largest size for the pool capacity), then handed to the
 * sequence from memory.
 */
async function runEmberGen(manifest: EmberGenManifest): Promise<void> {
  //* Validate the complete sequence before playback ===========================

  const images: Uint32Array[] = [];
  const bounds = new THREE.Box3();
  let maxFrameBytes = 0;
  let expectedGridType: string | undefined;

  for (let index = 0; index < manifest.urls.length; index++) {
    const url = manifest.urls[index]!;
    const file = await NvdbFile.fromURL(url);
    const meta = file.grids[0];
    if (!meta) throw new Error(`EmberGen frame ${index} has no grid`);
    if (expectedGridType === undefined) expectedGridType = meta.gridType;
    else if (meta.gridType !== expectedGridType)
      throw new Error(`EmberGen frame ${index} grid type ${meta.gridType} does not match ${expectedGridType}`);

    const frameBounds = new THREE.Box3(new THREE.Vector3(...meta.worldBBox.min), new THREE.Vector3(...meta.worldBBox.max));
    if (frameBounds.isEmpty() || !frameBounds.min.toArray().every(Number.isFinite) || !frameBounds.max.toArray().every(Number.isFinite))
      throw new Error(`EmberGen frame ${index} has invalid world bounds`);

    bounds.union(frameBounds);
    const image = file.gridImage(0);
    maxFrameBytes = Math.max(maxFrameBytes, image.byteLength);
    images.push(image);
  }

  if (images.length !== manifest.urls.length || images.length === 0)
    throw new Error(`EmberGen preflight loaded ${images.length}/${manifest.urls.length} frames`);

  const { id, material } = await makeVolume(images[0]!, maxFrameBytes, { jitter: true });
  if (pool.info(id).capacity < maxFrameBytes)
    throw new Error(`pool capacity ${pool.info(id).capacity} is smaller than sequence maximum ${maxFrameBytes}`);

  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
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

  const observedFrames: number[] = [];
  sequence = new VDBSequence({
    urls: manifest.urls,
    fps: manifest.fps,
    prefetch: VALIDATE_CYCLE ? manifest.urls.length : 4,
    loop: true,
    loader: (_url, index) => Promise.resolve(images[index]!),
    onFrame: (stats) => observedFrames.push(stats.frame),
    onError: (error, index) => fail(`EmberGen frame ${index} failed: ${String(error)}`),
  });
  setViewport(window.innerWidth, window.innerHeight);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));

  if (VALIDATE_CYCLE) {
    await sequence.preload(manifest.urls.length);
    sequence.start({ pool, id }, 0);
    await presentFrame();

    const frameDuration = 1000 / manifest.fps;
    for (let frame = 1; frame <= manifest.urls.length; frame++) {
      sequence.update((frame + 0.1) * frameDuration);
      await presentFrame();
    }

    const expectedFrames = [
      ...Array.from({ length: manifest.urls.length }, (_, index) => index),
      0,
    ];
    if (observedFrames.some((frame, index) => frame !== expectedFrames[index]))
      throw new Error(
        `EmberGen cycle progression mismatch: ${observedFrames.join(",")} ` +
          `!= ${expectedFrames.join(",")}`,
      );
    if (observedFrames.length !== expectedFrames.length)
      throw new Error(
        `EmberGen cycle rendered ${observedFrames.length}/${expectedFrames.length} expected frames`,
      );
    if (sequence.stats.stalls !== 0 || sequence.stats.frame !== 0)
      throw new Error(
        `EmberGen cycle ended at frame ${sequence.stats.frame} with ${sequence.stats.stalls} stalls`,
      );

    updateHud(sequence.stats, manifest.fps, `embergen (${manifest.urls.length} frames, cycle validated)`);
    window.__DEMO05__ = {
      ready: true,
      source: manifest.manifestUrl,
      frameCount: manifest.urls.length,
      framesValidated: images.length,
      framesPlayed: sequence.stats.swaps,
      swaps: sequence.stats.swaps,
      stalls: sequence.stats.stalls,
      cyclesCompleted: 1,
      finalFrame: sequence.stats.frame,
      validationMode: "full-cycle",
      maxFrameBytes,
      nodeRebinds,
      adapter,
      presentation: PRESENTATION,
    };
    return;
  }

  await sequence.preload();
  sequence.start({ pool, id });

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
        updateHud(sequence!.stats, manifest.fps, `embergen (${manifest.urls.length} frames)`);
        if (!published) {
          published = true;
          window.__DEMO05__ = {
            ready: true,
            source: manifest.manifestUrl,
            frameCount: manifest.urls.length,
            framesValidated: images.length,
            framesPlayed: sequence!.stats.swaps,
            swaps: sequence!.stats.swaps,
            stalls: sequence!.stats.stalls,
            finalFrame: sequence!.stats.frame,
            validationMode: "bounded-playback-smoke",
            maxFrameBytes,
            adapter,
            presentation: PRESENTATION,
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
