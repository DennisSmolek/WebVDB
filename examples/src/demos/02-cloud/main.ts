/**
 * Demo 02 — cloud: the WebVDB main goal (SPEC §5, docs/archive/v1/PLAN.md Phase 3), on
 * the K2 three adapter.
 *
 * Fragment-raymarches a NanoVDB fog volume: the grid goes into a
 * `VDBGridPool` on the renderer's device, a `VDBGridNode` puts the pool into
 * three's node system, and `VDBVolumeMaterial` marches grid `id` of it. An
 * orbit camera, a sun-lit cloud look, and range controls for the live material
 * uniforms. Publishes `window.__DEMO02__` for the e2e gate.
 *
 * ## Presentation
 *
 * Default (`?presentation=direct`): three renders into the page's canvas, as
 * on any hardware WebGPU browser. `?presentation=readback` renders into an
 * offscreen `RenderTarget`, reads it back and draws it on a 2D canvas. The e2e
 * spec uses it because headless SwiftShader has no compositor: configuring a
 * WebGPU canvas context there drops the Dawn instance ("A valid external
 * Instance reference no longer exists"; DIVERGES V7).
 *
 * ## Deterministic test mode (`?test=1`)
 *
 * Fixed camera pose (selectable with `?pose=N`), jitter OFF, fixed params —
 * renders exactly one settled frame, then stops. Screenshots are byte-stable.
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { NvdbFile } from "@pmndrs/vdb/tools";
import { VDBGridNode, VDBVolumeMaterial, getDevice, vdbProxyGeometry } from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import type { WebGPURenderer } from "three/webgpu";

interface Demo02State {
  ready: boolean;
  error?: string;
  fixture?: string;
  gridType?: string;
  adapter?: string;
  presentation?: string;
  poolVersion?: number;
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

// Fixed camera presets for deterministic test mode: [azimuth, elevation, radiusFactor].
const TEST_POSES: ReadonlyArray<readonly [number, number, number]> = [
  [Math.PI * 0.25, Math.PI * 0.16, 1.7],
  [Math.PI * 0.85, Math.PI * 0.34, 2.0],
];

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
const PERF_MODE = params.get("perf") === "1";
const PRESENTATION = params.get("presentation") === "readback" ? "readback" : "direct";
const FIXTURE = params.get("fixture") ?? (TEST_MODE ? "primitive" : "auto");
const QUALITY_STEP_SIZE = Number(params.get("stepSize") ?? "0.75");
const QUALITY_SHADOW_STEPS = Number(params.get("shadowSteps") ?? "12");
const SAMPLE_BUDGET_CAP = Math.max(1, Number(params.get("sampleBudgetCap") ?? "8192") | 0);
const POSE_INDEX = Math.max(0, Math.min(TEST_POSES.length - 1, Number(params.get("pose") ?? "0") | 0));

function fixtureCandidates(): string[] {
  if (FIXTURE === "primitive") return [fixtureUrl("primitives/sphere_fog_fp8.nvdb")];
  if (FIXTURE === "wdas") return [fixtureUrl("wdas/wdas_cloud_quarter_fp8.nvdb")];
  if (FIXTURE !== "auto") throw new Error(`fixture must be primitive, wdas, or auto (got ${FIXTURE})`);
  return [
    fixtureUrl("wdas/wdas_cloud_quarter_fp8.nvdb"),
    fixtureUrl("primitives/sphere_fog_fp8.nvdb"),
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

  const hit = await probeFixture();
  if (!hit) {
    fail(`No fixture found. Tried:\n${fixtureCandidates().join("\n")}`);
    return;
  }

  const file = NvdbFile.fromArrayBuffer(hit.buffer);
  const image = file.gridImage(0);
  const gridType = file.grids[0]!.gridType;
  const fixtureName = hit.url.split("/").pop() ?? hit.url;

  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;

  // Device-first renderer (D4). Hardware runs can present its canvas directly;
  // deterministic SwiftShader runs retain the readback path.
  let renderer: WebGPURenderer;
  let adapter = "unreported adapter";
  const created = await createDeviceRenderer({
    ...(directCanvas ? { canvas: directCanvas } : {}),
    trackTimestamp: PERF_MODE,
  });
  renderer = created.renderer;
  if (created.adapterName) adapter = created.adapterName;

  const display = directCanvas ?? document.createElement("canvas");
  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  const ctx = PRESENTATION === "readback" ? display.getContext("2d") : null;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x223049);

  // The pool lives on the renderer's device; one VDBGridNode per pool, with the
  // helper the material's ray clip needs.
  const pool = await VDBGridPool.create(getDevice(renderer));
  const id = pool.add(image);
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { helpers: ["rayClipSupport"] });
  const info = pool.info(id);

  const worldBox = new THREE.Box3(new THREE.Vector3(...info.worldBBox.min), new THREE.Vector3(...info.worldBBox.max));
  const center = worldBox.getCenter(new THREE.Vector3());
  const size = worldBox.getSize(new THREE.Vector3());
  const diag = size.length();

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, diag * 100);

  const material = new VDBVolumeMaterial(renderer, grids, id, {
    jitter: !TEST_MODE,
    stepSize: QUALITY_STEP_SIZE,
    shadowSteps: QUALITY_SHADOW_STEPS,
    sampleBudgetCap: SAMPLE_BUDGET_CAP,
  });
  const mesh = new THREE.Mesh(vdbProxyGeometry(info), material);
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
    // The pool contract: the pool's update is submitted (flush) before the
    // VDBGridNode rebinds and three renders. A static grid needs no update,
    // but grids.update() is cheap and keeps the loop correct if grids change.
    grids.update();
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      return;
    }

    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    const pixels = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
    renderer.setRenderTarget(null);

    // GPU readback is bottom-up and three r185–r187 preserves WebGPU's 256-byte
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

    const timestampQuery = created.features.includes("timestamp-query");
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
      `fixture: ${fixtureName} (${gridType})\n` +
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
      gridType: gridType,
      adapter,
      presentation: PRESENTATION,
      poolVersion: pool.version,
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
        gridType: gridType,
        adapter,
        presentation: PRESENTATION,
        poolVersion: pool.version,
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
            `fixture: ${fixtureName} (${gridType})\n` +
            `pool: ${(pool.byteLength / 1048576).toFixed(1)} MiB · v${pool.version}\n` +
            `fps: ${fps.toFixed(0)}  ·  step ${material.stepSize.value.toFixed(2)}  ·  ` +
            `maxSteps ${material.maxSteps.value.toFixed(0)}`;
        }
        if (!published) {
          published = true;
          window.__DEMO02__ = {
            ready: true,
            fixture: fixtureName,
            gridType: gridType,
            adapter,
            presentation: PRESENTATION,
            poolVersion: pool.version,
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
