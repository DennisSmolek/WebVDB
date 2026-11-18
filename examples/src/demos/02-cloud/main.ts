/**
 * Demo 02 — cloud: the WebVDB main goal (SPEC §5, docs/PLAN.md Phase 3).
 *
 * Fragment-raymarches a NanoVDB fog volume with `NanoVDBVolumeMaterial`:
 * device-first renderer bootstrap, an orbit camera, a sun-lit cloud look, and
 * hand-rolled range controls for the live material params. Publishes
 * `window.__DEMO02__` for the e2e gate.
 *
 * ## Presentation
 *
 * The volume is rendered into an offscreen `RenderTarget`, read back, and drawn
 * to an ordinary 2D `<canvas>`. This is deliberate: on headless SwiftShader
 * (CI, and this sandbox) there is no GPU compositor, so configuring a WebGPU
 * *canvas context* drops the Dawn instance ("A valid external Instance
 * reference no longer exists") and nothing ever presents — whereas offscreen
 * render + `readRenderTargetPixelsAsync` works exactly as on real hardware
 * (demo 01's compute path never hit this). The blit path is portable across
 * both, at the cost of one GPU→CPU copy per frame. Stable-Chrome hardware and
 * performance checks use `?presentation=direct`; CI pins `presentation=readback`.
 * See docs/handoffs/PHASE-3.
 *
 * ## Deterministic test mode (`?test=1`)
 *
 * Fixed camera pose (selectable with `?pose=N`), jitter OFF, fixed params —
 * renders exactly one settled frame, then stops. Screenshots are byte-stable.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { NanoVDBFile } from "nanovdb-wgsl";
import pnanovdbSource from "nanovdb-wgsl/pnanovdb.wgsl?raw";
import { NanoVDBGrid, NanoVDBVolumeMaterial, createVolumeRenderer } from "three-nanovdb";
import type { WebGPURenderer } from "three/webgpu";

interface Demo02State {
  ready: boolean;
  error?: string;
  fixture?: string;
  gridType?: string;
  adapter?: string;
  presentation?: string;
  swizzleShim?: boolean;
  keepAliveShim?: boolean;
  resolution?: string;
  stepSize?: number;
  shadowSteps?: number;
  sampleBudgetCap?: number;
  fps?: number;
  frameMs?: number;
  gpuMs?: number;
  timestampQuery?: boolean;
  gpuTimestampValid?: boolean;
  shadowBaselineFrameMs?: number;
  shadowDeltaFrameMs?: number;
  shadowBaselineGpuMs?: number;
  shadowDeltaGpuMs?: number;
}

declare global {
  interface Window {
    __DEMO02__?: Demo02State;
  }
}

/**
 * Compatibility shim (three r185 × this Chromium/SwiftShader build).
 *
 * three's `GPUTextureViewDescriptor` unconditionally sets `swizzle: 'rgba'`
 * (a string) on EVERY texture-view, assuming browsers without the
 * `texture-component-swizzle` feature ignore it. This build's WebGPU IDL knows
 * `swizzle` as a *dictionary*, so the string fails conversion and
 * `createView()` throws. `'rgba'` is the identity swizzle, so dropping it is a
 * semantic no-op. Patched at the app edge (the renderer bootstrap is out of
 * this lane).
 */
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

/**
 * Pin every requested adapter (device-first hygiene). `createVolumeRenderer`
 * (decision D4) drops its local adapter reference on return; on Dawn a collected
 * adapter can invalidate the device. Cheap insurance kept at the app edge.
 */
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

// Fixed camera presets for deterministic test mode: [azimuth, elevation, radiusFactor].
const TEST_POSES: ReadonlyArray<readonly [number, number, number]> = [
  [Math.PI * 0.25, Math.PI * 0.16, 1.7],
  [Math.PI * 0.85, Math.PI * 0.34, 2.0],
];

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
const PERF_MODE = params.get("perf") === "1";
const PRESENTATION = params.get("presentation") === "direct" ? "direct" : "readback";
const SWIZZLE_SHIM = params.get("swizzle") !== "off";
const KEEP_ALIVE_SHIM = params.get("keepAlive") !== "off";
const FIXTURE = params.get("fixture") ?? (TEST_MODE ? "primitive" : "auto");
const QUALITY_STEP_SIZE = Number(params.get("stepSize") ?? "0.75");
const QUALITY_SHADOW_STEPS = Number(params.get("shadowSteps") ?? "12");
const SAMPLE_BUDGET_CAP = Math.max(1, Number(params.get("sampleBudgetCap") ?? "8192") | 0);
const POSE_INDEX = Math.max(0, Math.min(TEST_POSES.length - 1, Number(params.get("pose") ?? "0") | 0));

function fixtureCandidates(): string[] {
  if (FIXTURE === "primitive") return ["/fixtures/primitives/sphere_fog_fp8.nvdb"];
  if (FIXTURE === "wdas") return ["/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb"];
  if (FIXTURE !== "auto") throw new Error(`fixture must be primitive, wdas, or auto (got ${FIXTURE})`);
  return [
    "/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb",
    "/fixtures/primitives/sphere_fog_fp8.nvdb",
  ];
}

const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const controlsEl = document.querySelector<HTMLDivElement>("#controls")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const panelEl = document.querySelector<HTMLDivElement>("#panel")!;
const appEl = document.querySelector<HTMLDivElement>("#app")!;

function fail(message: string): void {
  errEl.textContent = message;
  statsEl.textContent = "failed";
  window.__DEMO02__ = { ready: false, error: message };
}

async function probeFixture(): Promise<{ url: string; buffer: ArrayBuffer } | null> {
  for (const url of fixtureCandidates()) {
    try {
      const res = await fetch(url);
      if (res.ok) return { url, buffer: await res.arrayBuffer() };
    } catch {
      /* try next */
    }
  }
  return null;
}

/** A labelled range control that live-updates a scalar material uniform. */
function addRange(label: string, uni: { value: number }, min: number, max: number, step: number): void {
  const row = document.createElement("div");
  row.className = "ctl";
  const id = `ctl-${label.replace(/\s+/g, "-")}`;
  const out = document.createElement("output");
  const fmt = (v: number): string => v.toFixed(step < 1 ? 2 : 0);
  out.textContent = fmt(uni.value);
  const input = document.createElement("input");
  input.type = "range";
  input.id = id;
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(uni.value);
  input.addEventListener("input", () => {
    uni.value = Number(input.value);
    out.textContent = fmt(uni.value);
  });
  const lab = document.createElement("label");
  lab.htmlFor = id;
  lab.textContent = label;
  row.append(lab, input, out);
  controlsEl.appendChild(row);
}

async function run(): Promise<void> {
  if (!navigator.gpu) {
    fail("WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.");
    return;
  }
  if (!Number.isFinite(QUALITY_STEP_SIZE) || QUALITY_STEP_SIZE <= 0)
    throw new Error(`stepSize must be positive, got ${QUALITY_STEP_SIZE}`);
  if (!Number.isFinite(QUALITY_SHADOW_STEPS) || QUALITY_SHADOW_STEPS < 0)
    throw new Error(`shadowSteps must be non-negative, got ${QUALITY_SHADOW_STEPS}`);
  if (SWIZZLE_SHIM) installSwizzleCompatShim();
  if (KEEP_ALIVE_SHIM) installAdapterKeepAliveShim();

  const hit = await probeFixture();
  if (!hit) {
    fail(`No fixture found. Tried:\n${fixtureCandidates().join("\n")}`);
    return;
  }

  const file = NanoVDBFile.fromArrayBuffer(hit.buffer);
  const grid = NanoVDBGrid.fromFile(file, 0);
  const fixtureName = hit.url.split("/").pop() ?? hit.url;

  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;

  // Device-first renderer (D4). Hardware runs can present its canvas directly;
  // deterministic SwiftShader runs retain the readback path.
  let renderer: WebGPURenderer;
  let adapter = "unreported adapter";
  const created = await createVolumeRenderer({
    gridBytes: grid.byteLength,
    ...(directCanvas ? { canvas: directCanvas } : {}),
    trackTimestamp: PERF_MODE,
  });
  renderer = created.renderer;
  if (created.report.adapterInfo)
    adapter = [created.report.adapterInfo.vendor, created.report.adapterInfo.architecture]
      .filter(Boolean)
      .join(" ");

  const display = directCanvas ?? document.createElement("canvas");
  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  const ctx = PRESENTATION === "readback" ? display.getContext("2d") : null;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x223049);

  const worldBox = grid.worldBBox();
  const center = worldBox.getCenter(new THREE.Vector3());
  const size = worldBox.getSize(new THREE.Vector3());
  const diag = size.length();

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, diag * 100);

  const material = new NanoVDBVolumeMaterial({
    grid,
    pnanovdbSource,
    jitter: !TEST_MODE,
    stepSize: QUALITY_STEP_SIZE,
    shadowSteps: QUALITY_SHADOW_STEPS,
    sampleBudgetCap: SAMPLE_BUDGET_CAP,
  });
  const mesh = new THREE.Mesh(grid.proxyGeometry(), material);
  scene.add(mesh);

  const controls = new OrbitControls(camera, display);
  controls.target.copy(center);
  controls.enableDamping = !TEST_MODE;

  // Offscreen render target. sRGB texture so three applies the linear->sRGB
  // output encode when writing to it (RTs are otherwise kept linear, which
  // would read back dark) — this matches what a canvas present would show.
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

  /** Render the scene to the offscreen RT, read it back, and blit to the 2D canvas. */
  async function presentFrame(): Promise<void> {
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      return;
    }

    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    const pixels = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
    renderer.setRenderTarget(null);

    // GPU readback is bottom-up and three r185 preserves WebGPU's 256-byte
    // row alignment. Derive that padded source stride from the returned byte
    // length, then copy only the tightly-packed RGBA8 pixels into ImageData.
    const packedStride = vw * 4;
    const sourceStride = vh > 1 ? (pixels.byteLength - packedStride) / (vh - 1) : packedStride;
    if (!Number.isInteger(sourceStride) || sourceStride < packedStride)
      throw new Error(
        `Unexpected readback layout: ${pixels.byteLength} bytes for ${vw}x${vh} RGBA8 pixels`,
      );

    for (let y = 0; y < vh; y++) {
      const src = (vh - 1 - y) * sourceStride;
      flip.set(pixels.subarray(src, src + packedStride), y * packedStride);
    }
    ctx!.putImageData(new ImageData(flip, vw, vh), 0, 0);
  }

  if (PERF_MODE) {
    if (PRESENTATION !== "direct")
      throw new Error("perf=1 requires presentation=direct so readback cost is excluded");
    if (FIXTURE !== "wdas")
      throw new Error("perf=1 requires fixture=wdas so the recorded gate cannot silently use a primitive");

    panelEl.classList.add("test-hidden");
    setViewport(1920, 1080);
    placeCamera(Math.PI * 0.25, Math.PI * 0.18, 1.8);

    const timestampQuery = created.report.grantedFeatures.includes("timestamp-query");
    const measure = async (frameCount: number): Promise<{ fps: number; frameMs: number; gpuMs?: number }> => {
      const started = performance.now();
      for (let frame = 0; frame < frameCount; frame++) await presentFrame();
      await created.device.queue.onSubmittedWorkDone();
      const frameMs = (performance.now() - started) / frameCount;
      const rawGpuMs = timestampQuery ? await renderer.resolveTimestampsAsync("render") : undefined;
      const gpuMs =
        rawGpuMs !== undefined && Number.isFinite(rawGpuMs) && rawGpuMs > 0 && rawGpuMs <= frameMs * 4
          ? rawGpuMs
          : undefined;
      return { fps: 1000 / frameMs, frameMs, ...(gpuMs !== undefined ? { gpuMs } : {}) };
    };
    const measurePreset = async (shadowSteps: number): Promise<{ fps: number; frameMs: number; gpuMs?: number }> => {
      material.shadowSteps.value = shadowSteps;
      for (let frame = 0; frame < 60; frame++) await presentFrame();
      return measure(180);
    };
    const average = (
      first: { frameMs: number; gpuMs?: number },
      second: { frameMs: number; gpuMs?: number },
    ): { fps: number; frameMs: number; gpuMs?: number } => {
      const frameMs = (first.frameMs + second.frameMs) / 2;
      const gpuMs =
        first.gpuMs !== undefined && second.gpuMs !== undefined
          ? (first.gpuMs + second.gpuMs) / 2
          : undefined;
      return { fps: 1000 / frameMs, frameMs, ...(gpuMs !== undefined ? { gpuMs } : {}) };
    };

    for (let frame = 0; frame < 60; frame++) await presentFrame();
    const configuredShadowSteps = material.shadowSteps.value;
    const baselineFirst = await measurePreset(0);
    const measuredFirst = await measurePreset(configuredShadowSteps);
    const baselineSecond = await measurePreset(0);
    const measuredSecond = await measurePreset(configuredShadowSteps);
    const baseline = average(baselineFirst, baselineSecond);
    const measured = average(measuredFirst, measuredSecond);
    const shadowDeltaFrameMs = measured.frameMs - baseline.frameMs;
    const shadowDeltaGpuMs =
      measured.gpuMs !== undefined && baseline.gpuMs !== undefined
        ? measured.gpuMs - baseline.gpuMs
        : undefined;

    statsEl.textContent =
      `fixture: ${fixtureName} (${grid.metadata.gridType})\n` +
      `adapter: ${adapter}\n` +
      `resolution: 1920x1080  ·  presentation: direct\n` +
      `stepSize: ${material.stepSize.value}  ·  shadowSteps: ${material.shadowSteps.value}  ·  ` +
      `sampleBudgetCap: ${SAMPLE_BUDGET_CAP}\n` +
      `fps: ${measured.fps.toFixed(2)}  ·  frame: ${measured.frameMs.toFixed(2)} ms  ·  ` +
      `gpu: ${measured.gpuMs?.toFixed(2) ?? "unavailable"} ms\n` +
      `shadow=0 frame: ${baseline.frameMs.toFixed(2)} ms  ·  frame delta: ${shadowDeltaFrameMs.toFixed(2)} ms\n` +
      `shadow=0 gpu: ${baseline.gpuMs?.toFixed(2) ?? "unavailable"} ms  ·  ` +
      `gpu delta: ${shadowDeltaGpuMs?.toFixed(2) ?? "unavailable"} ms`;
    window.__DEMO02__ = {
      ready: true,
      fixture: fixtureName,
      gridType: grid.metadata.gridType,
      adapter,
      presentation: PRESENTATION,
      swizzleShim: SWIZZLE_SHIM,
      keepAliveShim: KEEP_ALIVE_SHIM,
      resolution: "1920x1080",
      stepSize: material.stepSize.value,
      shadowSteps: material.shadowSteps.value,
      sampleBudgetCap: SAMPLE_BUDGET_CAP,
      fps: measured.fps,
      frameMs: measured.frameMs,
      timestampQuery,
      gpuTimestampValid: !timestampQuery || measured.gpuMs !== undefined,
      shadowBaselineFrameMs: baseline.frameMs,
      shadowDeltaFrameMs,
      ...(measured.gpuMs !== undefined ? { gpuMs: measured.gpuMs } : {}),
      ...(baseline.gpuMs !== undefined ? { shadowBaselineGpuMs: baseline.gpuMs } : {}),
      ...(shadowDeltaGpuMs !== undefined ? { shadowDeltaGpuMs } : {}),
    };
    return;
  }

  if (TEST_MODE) {
    panelEl.classList.add("test-hidden");
    setViewport(640, 480);
    material.densityScale.value = 45;
    material.stepSize.value = 0.45;
    material.maxSteps.value = 640;
    material.sunDirection.value.set(0.7, 0.5, 0.45).normalize();
    material.sunIntensity.value = 5;
    material.anisotropy.value = 0.5;
    material.shadowSteps.value = 16;
    material.shadowDensity.value = 1;
    material.ambient.value = 0.22;
    material.jitter.value = 0;
    const [az, el, rf] = TEST_POSES[POSE_INDEX]!;
    placeCamera(az, el, rf);
    try {
      // First render compiles the (large) shader; second settles a clean frame.
      await presentFrame();
      await presentFrame();
      window.__DEMO02__ = {
        ready: true,
        fixture: fixtureName,
        gridType: grid.metadata.gridType,
        adapter,
        presentation: PRESENTATION,
        swizzleShim: SWIZZLE_SHIM,
        keepAliveShim: KEEP_ALIVE_SHIM,
        resolution: "640x480",
        stepSize: material.stepSize.value,
        shadowSteps: material.shadowSteps.value,
        sampleBudgetCap: SAMPLE_BUDGET_CAP,
      };
    } catch (err) {
      fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
    }
    return;
  }

  // Interactive.
  addRange("density", material.densityScale, 1, 200, 1);
  addRange("step", material.stepSize, 0.1, 3, 0.05);
  addRange("maxSteps", material.maxSteps, 32, 1024, 1);
  addRange("sun", material.sunIntensity, 0, 10, 0.1);
  addRange("anisotropy", material.anisotropy, -0.9, 0.9, 0.05);
  addRange("shadowSteps", material.shadowSteps, 0, 64, 1);
  addRange("ambient", material.ambient, 0, 1, 0.01);

  setViewport(window.innerWidth, window.innerHeight);
  placeCamera(Math.PI * 0.25, Math.PI * 0.18, 1.8);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));

  let frames = 0;
  let lastFpsT = performance.now();
  let fps = 0;
  let published = false;
  let busy = false;

  const loop = (): void => {
    requestAnimationFrame(loop);
    if (busy || window.__DEMO02__?.error) return;
    busy = true;
    controls.update();
    presentFrame()
      .then(() => {
        busy = false;
        frames++;
        const now = performance.now();
        if (now - lastFpsT >= 500) {
          fps = (frames * 1000) / (now - lastFpsT);
          frames = 0;
          lastFpsT = now;
          statsEl.textContent =
            `fixture: ${fixtureName} (${grid.metadata.gridType})\n` +
            `sampler: ${material.samplerFn}\n` +
            `fps: ${fps.toFixed(0)}  ·  step ${material.stepSize.value.toFixed(2)}  ·  ` +
            `maxSteps ${material.maxSteps.value.toFixed(0)}`;
        }
        if (!published) {
          published = true;
          window.__DEMO02__ = {
            ready: true,
            fixture: fixtureName,
            gridType: grid.metadata.gridType,
            adapter,
            presentation: PRESENTATION,
            swizzleShim: SWIZZLE_SHIM,
            keepAliveShim: KEEP_ALIVE_SHIM,
            resolution: `${vw}x${vh}`,
            stepSize: material.stepSize.value,
            shadowSteps: material.shadowSteps.value,
            sampleBudgetCap: SAMPLE_BUDGET_CAP,
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
