/**
 * Demo 08 — webgpu-smoke: the OpenVDB smoke rendered with NO engine (task
 * K1.7, docs/dev/design/K1-core.md §7). Only `@pmndrs/vdb` (the core: pool + WGSL
 * generator) and `@pmndrs/vdb/tools` (the .vdb parser); no three.js
 * (examples/test/demo-08-imports.test.ts checks the module graph).
 *
 * It is the reference for the engine contract (D8). The host:
 *   1. owns the GPUDevice, created with `VDBGridPool.requiredLimits(adapter)`;
 *   2. generates one WGSL module for its shader (`vdbModuleWgsl`) and declares
 *      the pool binding in its own layout (`vdbModuleEntries`);
 *   3. per frame, records `pool.update(encoder)` into ITS encoder, before the
 *      passes that read the pool, and submits that encoder (the pool's
 *      contract: submit each update's encoder before the second update after it);
 *   4. rebuilds its bind group whenever `pool.version` changes (the pool's
 *      buffer is a new object after a growth);
 *   5. awaits `pool.settled()` to learn that a growth's allocation held, and
 *      shows `pool.lastError` otherwise;
 *   6. calls `pool.dispose()` on teardown. The device stays the host's.
 *
 * Presentation: `?presentation=direct` (default) draws into the canvas's
 * WebGPU context. `?presentation=readback` renders to a texture, copies it to
 * a buffer and draws it on a 2D canvas: headless SwiftShader has no
 * compositor (docs/archive/handoffs/PHASE-3.md), so the e2e spec uses it.
 * `?test=1` pins the camera and renders only on demand.
 * State for e2e/demo-08.spec.ts: `window.__DEMO08__`.
 */
import { VDBGridPool, vdbCheckModule, vdbModuleEntries, vdbModuleWgsl } from "@pmndrs/vdb";
import type { VDBSlotModuleOptions } from "@pmndrs/vdb";
import { NvdbFile, buildFromVdb, parseVdb } from "@pmndrs/vdb/tools";
import { FRAME_FLOATS, HOST_WGSL } from "./shader.js";

interface Demo08State {
  ready: boolean;
  error?: string;
  adapter?: string;
  presentation?: string;
  /** Bytes of the generated `vdbModuleWgsl` text, and of the whole shader. */
  wgslBytes?: number;
  shaderBytes?: number;
  frames: number;
  version: number;
  revision: number;
  gridCount: number;
  poolBytes: number;
  bindGroupBuilds: number;
  /** Mean ms from submit to `onSubmittedWorkDone` over the last frames. */
  frameMs?: number;
  /** Uncaptured WebGPU errors (validation, out-of-memory). */
  gpuErrors: string[];
  secondGrid: boolean;
}

declare global {
  interface Window {
    __DEMO08__?: Demo08State;
  }
}

const SMOKE_URL = "/fixtures/vdb-samples/smoke.vdb";
const SECOND_URL = "/fixtures/primitives/torus_fog_fp8.nvdb";

const params = new URLSearchParams(location.search);
const testMode = params.get("test") === "1";
const presentation = params.get("presentation") === "readback" ? "readback" : "direct";

const canvas = document.querySelector<HTMLCanvasElement>("#view")!;
const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const secondButton = document.querySelector<HTMLButtonElement>("#second-grid")!;
const spinBox = document.querySelector<HTMLInputElement>("#spin")!;

const state: Demo08State = {
  ready: false,
  frames: 0,
  version: 0,
  revision: 0,
  gridCount: 0,
  poolBytes: 0,
  bindGroupBuilds: 0,
  gpuErrors: [],
  secondGrid: false,
  presentation,
};
window.__DEMO08__ = state;

function fail(error: unknown): void {
  state.error = error instanceof Error ? error.message : String(error);
  errEl.textContent = state.error;
  console.error(error);
}

async function main(): Promise<void> {
  //* 1. The host's device, with the limits the pool may grow into.
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter.");
  state.adapter = `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description}`.trim();
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    state.gpuErrors.push(message);
    errEl.textContent = message;
  });

  //* 2. Load the smoke: .vdb -> parseVdb -> buildFromVdb -> NanoVDB Float image.
  statsEl.textContent = "loading smoke.vdb…";
  const response = await fetch(SMOKE_URL);
  if (!response.ok) throw new Error(`${SMOKE_URL}: ${response.status}. Run pnpm fixtures:vdb-samples.`);
  const vdb = parseVdb(await response.arrayBuffer());
  const smoke = buildFromVdb(vdb.grids[0]!);

  //* 3. A pool on the host's device; add() stages, the next update() uploads.
  const pool = await VDBGridPool.create(device, { label: "demo-08 pool" });
  const smokeId = pool.add(smoke, { label: "smoke" });

  //* 4. One generated module for the shader. Several grids share it by id, so
  //     compile every type the app will add (the torus is Fp8), and check the
  //     pool against it whenever grids are added. The two opt-in helpers give
  //     the march its trilinear-safe skip bound and the HDDA cell's exit t.
  const moduleOptions: VDBSlotModuleOptions = {
    slot: { group: 0, firstBinding: 0 },
    types: ["float", "fp8"],
    profile: "march",
    helpers: ["cellBoundTrilinear", "hddaCellEnd"],
  };
  vdbCheckModule(pool, moduleOptions);
  const vdbWgsl = vdbModuleWgsl(moduleOptions);
  const code = vdbWgsl + HOST_WGSL;
  state.wgslBytes = new TextEncoder().encode(vdbWgsl).byteLength;
  state.shaderBytes = new TextEncoder().encode(code).byteLength;

  //* 5. The host's pipeline: its layout holds the pool entry and its uniforms.
  const format = presentation === "direct" ? navigator.gpu.getPreferredCanvasFormat() : "rgba8unorm";
  const bindGroupLayout = device.createBindGroupLayout({
    label: "demo-08",
    entries: [
      ...vdbModuleEntries(moduleOptions.slot), // default visibility: COMPUTE | FRAGMENT
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });
  device.pushErrorScope("validation");
  const shader = device.createShaderModule({ label: "demo-08 march", code });
  const pipeline = await device.createRenderPipelineAsync({
    label: "demo-08 march",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shader, entryPoint: "vs" },
    fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
  });
  const compileError = await device.popErrorScope();
  if (compileError) throw new Error(`pipeline: ${compileError.message}`);
  const frameBuffer = device.createBuffer({ label: "demo-08 frame", size: FRAME_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // The bind group points at pool.buffer, so it is stale whenever the version moves.
  let bindGroup: GPUBindGroup | null = null;
  let boundVersion = -1;
  function bindGroupFor(p: VDBGridPool): GPUBindGroup {
    if (bindGroup === null || boundVersion !== p.version) {
      bindGroup = device.createBindGroup({
        label: `demo-08 v${p.version}`,
        layout: bindGroupLayout,
        entries: [p.bindGroupEntry(moduleOptions.slot), { binding: 1, resource: { buffer: frameBuffer } }],
      });
      boundVersion = p.version;
      state.bindGroupBuilds++;
    }
    return bindGroup;
  }

  //* Presentation targets.
  const presenter = presentation === "direct" ? directPresenter(device, format) : readbackPresenter(device);

  //* Camera: orbit around the scene's bounds (pool.info gives world boxes at once).
  const camera = { yaw: 0.6, pitch: 0.18, distance: 1, target: [0, 0, 0] as number[] };
  function frameBounds(ids: number[]): void {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const id of ids) {
      const box = pool.info(id).worldBBox;
      for (let a = 0; a < 3; a++) {
        min[a] = Math.min(min[a]!, box.min[a]!);
        max[a] = Math.max(max[a]!, box.max[a]!);
      }
    }
    camera.target = min.map((v, a) => (v + max[a]!) / 2);
    camera.distance = 1.25 * Math.hypot(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!);
  }
  frameBounds([smokeId]);

  function writeFrame(width: number, height: number): void {
    const { yaw, pitch, distance, target } = camera;
    const dir = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
    const eye = target.map((v, a) => v + dir[a]! * distance);
    const forward = dir.map((v) => -v);
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const data = new Float32Array(FRAME_FLOATS);
    data.set([...eye, Math.tan((40 * Math.PI) / 360)], 0);
    data.set([...right, width / height], 4);
    data.set([...up, 0.35], 8); // density: extinction per world unit at value 1
    data.set([...forward, 1.0], 12); // stepVoxels
    data.set([width, height, 0, 0], 16);
    device.queue.writeBuffer(frameBuffer, 0, data);
  }

  //* The frame. One encoder: pool.update, then the pass that reads the pool.
  let inFlight = false;
  let dirty = true;
  let disposed = false;
  let spinAngle = 0;
  let secondId = -1;
  const frameTimes: number[] = [];

  async function renderFrame(): Promise<void> {
    inFlight = true;
    dirty = false;
    const { width, height } = presenter.resize(canvas);
    writeFrame(width, height);

    const encoder = device.createCommandEncoder({ label: "demo-08 frame" });
    pool.update(encoder); // uploads staged grids/records; may grow -> new buffer, version + 1
    const group = bindGroupFor(pool); // after update: the pass must bind the current buffer
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: presenter.view(), loadOp: "clear", clearValue: [0, 0, 0, 1], storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    presenter.afterPass(encoder);
    const t0 = performance.now();
    device.queue.submit([encoder.finish()]); // submitted before the next update
    await device.queue.onSubmittedWorkDone();
    frameTimes.push(performance.now() - t0);
    if (frameTimes.length > 20) frameTimes.shift();
    await presenter.present(canvas);

    state.frames++;
    state.version = pool.version;
    state.revision = pool.revision;
    state.gridCount = pool.gridCount;
    state.poolBytes = pool.byteLength;
    state.frameMs = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    if (pool.lastError) errEl.textContent = `pool: ${pool.lastError.message}`;
    showStats();
    inFlight = false;
  }

  function showStats(): void {
    const mib = (n: number) => `${(n / 2 ** 20).toFixed(1)} MiB`;
    statsEl.textContent =
      `adapter   ${state.adapter || "?"}\n` +
      `module    march · float+fp8 · ${state.wgslBytes!.toLocaleString()} B WGSL\n` +
      `shader    ${state.shaderBytes!.toLocaleString()} B total\n` +
      `pool      ${pool.gridCount} grids · ${mib(pool.byteLength)} · version ${pool.version} · revision ${pool.revision}\n` +
      `bind grp  rebuilt ${state.bindGroupBuilds}×\n` +
      `frame     ${state.frameMs!.toFixed(1)} ms (submit → done) · ${presentation}`;
  }

  // The first frame grows the pool (1 MiB -> fits the 6.7 MiB smoke); confirm it held.
  await renderFrame();
  await pool.settled();

  //* The second grid: add() stages it; the next update grows the buffer, the
  //  version moves and bindGroupFor rebuilds. setTransform places it.
  async function addSecondGrid(): Promise<void> {
    secondButton.disabled = true;
    const res = await fetch(SECOND_URL);
    if (!res.ok) throw new Error(`${SECOND_URL}: ${res.status}. Run pnpm fixtures:bake.`);
    const image = NvdbFile.fromArrayBuffer(await res.arrayBuffer()).gridImage(0);
    secondId = pool.add(image, { label: "torus", transform: torusTransform(pool.info(smokeId).worldBBox, 0) });
    vdbCheckModule(pool, moduleOptions); // the pool now holds fp8: the module must decode it
    frameBounds([smokeId, secondId]);
    state.secondGrid = true;
    spinBox.disabled = false;
    dirty = true;
    await whenIdle();
    await renderFrame();
    await pool.settled(); // throws VDB_POOL_LIMIT if the growth's allocation failed (rolled back)
  }

  function whenIdle(): Promise<void> {
    return new Promise((resolve) => {
      const check = () => (inFlight ? requestAnimationFrame(check) : resolve());
      check();
    });
  }

  secondButton.addEventListener("click", () => {
    state.ready = false;
    addSecondGrid().then(() => (state.ready = true), fail);
  });
  secondButton.disabled = false;

  //* Orbit: drag to rotate, wheel to zoom (disabled in test mode).
  if (!testMode) {
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
    });
    canvas.addEventListener("pointerup", (e) => {
      canvas.releasePointerCapture(e.pointerId);
      canvas.style.cursor = "grab";
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!canvas.hasPointerCapture(e.pointerId)) return;
      camera.yaw -= e.movementX * 0.008;
      camera.pitch = Math.max(-1.4, Math.min(1.4, camera.pitch + e.movementY * 0.008));
      dirty = true;
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        camera.distance *= Math.exp(e.deltaY * 0.001);
        dirty = true;
      },
      { passive: false },
    );
    window.addEventListener("resize", () => (dirty = true));
  }

  // Render on demand; "spin" animates the torus with setTransform only: the
  // record changes (revision + 1), the image and the buffer don't (same version).
  function loop(): void {
    if (disposed) return;
    requestAnimationFrame(loop);
    if (inFlight) return;
    if (spinBox.checked && secondId >= 0) {
      spinAngle += 0.02;
      pool.setTransform(secondId, torusTransform(pool.info(smokeId).worldBBox, spinAngle));
      dirty = true;
    }
    if (dirty) renderFrame().catch(fail);
  }
  if (!testMode) requestAnimationFrame(loop);

  //* Teardown: let submitted work finish, then free what the pool allocated.
  window.addEventListener("pagehide", () => {
    disposed = true;
    void device.queue.onSubmittedWorkDone().then(() => {
      pool.dispose();
      presenter.dispose();
      frameBuffer.destroy();
      device.destroy(); // the host created the device, so the host destroys it
    });
  });

  state.ready = true;
}

/**
 * Object-to-world for the torus (4x4 column-major, the WebGPU/three
 * convention): scaled to 0.3, spun `angle` about its own axis (y), tilted 60°
 * about x towards the camera, and placed to the right of the smoke.
 */
function torusTransform(smoke: { min: number[]; max: number[] }, angle: number): number[] {
  const s = 0.3;
  const tilt = (60 * Math.PI) / 180;
  const [ca, sa] = [Math.cos(tilt), Math.sin(tilt)];
  const [cb, sb] = [Math.cos(angle), Math.sin(angle)];
  // R = Rx(tilt) * Ry(angle), rows
  const r = [
    [cb, 0, sb],
    [sa * sb, ca, -sa * cb],
    [-ca * sb, sa, ca * cb],
  ];
  const tx = smoke.max[0]! + 32;
  const ty = smoke.min[1]! + 0.3 * (smoke.max[1]! - smoke.min[1]!);
  const tz = (smoke.min[2]! + smoke.max[2]!) / 2;
  // column-major: m[col * 4 + row]
  return [
    s * r[0]![0]!, s * r[1]![0]!, s * r[2]![0]!, 0,
    s * r[0]![1]!, s * r[1]![1]!, s * r[2]![1]!, 0,
    s * r[0]![2]!, s * r[1]![2]!, s * r[2]![2]!, 0,
    tx, ty, tz, 1,
  ];
}

interface Presenter {
  resize(canvas: HTMLCanvasElement): { width: number; height: number };
  view(): GPUTextureView;
  afterPass(encoder: GPUCommandEncoder): void;
  present(canvas: HTMLCanvasElement): Promise<void>;
  dispose(): void;
}

function canvasSize(canvas: HTMLCanvasElement): { width: number; height: number } {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height };
}

/** Draw straight into the canvas's WebGPU context. */
function directPresenter(device: GPUDevice, format: GPUTextureFormat): Presenter {
  const context = canvas.getContext("webgpu")!;
  context.configure({ device, format, alphaMode: "opaque" });
  return {
    resize: canvasSize,
    view: () => context.getCurrentTexture().createView(),
    afterPass: () => {},
    present: async () => {},
    dispose: () => context.unconfigure(),
  };
}

/** Render to a texture, copy it back, draw it on a 2D canvas (headless SwiftShader). */
function readbackPresenter(device: GPUDevice): Presenter {
  const ctx2d = canvas.getContext("2d")!;
  let texture: GPUTexture | null = null;
  let readback: GPUBuffer | null = null;
  let bytesPerRow = 0;
  return {
    resize(c) {
      const size = canvasSize(c);
      if (!texture || texture.width !== size.width || texture.height !== size.height) {
        texture?.destroy();
        readback?.destroy();
        texture = device.createTexture({
          label: "demo-08 offscreen",
          size: [size.width, size.height],
          format: "rgba8unorm",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        bytesPerRow = Math.ceil((size.width * 4) / 256) * 256;
        readback = device.createBuffer({ label: "demo-08 readback", size: bytesPerRow * size.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      }
      return size;
    },
    view: () => texture!.createView(),
    afterPass(encoder) {
      encoder.copyTextureToBuffer({ texture: texture! }, { buffer: readback!, bytesPerRow }, [texture!.width, texture!.height]);
    },
    async present() {
      const { width, height } = texture!;
      await readback!.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(readback!.getMappedRange());
      const pixels = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y++) pixels.set(src.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
      readback!.unmap();
      ctx2d.putImageData(new ImageData(pixels, width, height), 0, 0);
    },
    dispose() {
      texture?.destroy();
      readback?.destroy();
    },
  };
}

function cross(a: number[], b: number[]): number[] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
}

function normalize(v: number[]): number[] {
  const n = Math.hypot(v[0]!, v[1]!, v[2]!);
  return v.map((x) => x / n);
}

main().catch(fail);
