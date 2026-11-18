/**
 * Demo 09 — coarse-first (K4.20): stream one `.nvdb` that holds a small preview grid and then the full grid
 * (docs/design/K4-5-coarse-first.md, K4.16), and watch the volume go from preview to full with a byte counter.
 *
 * - The file is baked in the page (`vdbBakeCoarseFirst` on `fixtures/vdb-samples/smoke.vdb`: `[density.lod48, density]`,
 *   box previews, FpN, codec ZIP; K4.19 proposes trilinear as the default, docs/handoffs/K4-19.md), made a `Blob`, and fetched back as a `Response` so the loader
 *   sees what a network delivers. `?file=<url>` streams a file you baked yourself instead
 *   (`node scripts/bake-coarse-first.mjs --out fixtures/smoke.coarse.nvdb`, then `?file=/fixtures/smoke.coarse.nvdb`).
 * - The body is throttled to a fixed link speed (`?speed=<KiB/s>`, default 256) by a pull-driven stream, so the
 *   preview (about 12 KiB of ZIP bytes) lands in a few frames and the full grid a few seconds later.
 * - `vdbStreamCoarseFirst` adds the preview to the pool as soon as its bytes are in and replaces it in place with the
 *   full grid. The stream is held after each grid until that grid has been drawn, so the transition is always shown.
 * - The material integrates per index voxel, so while a preview shows, `densityScale` is multiplied by the event's
 *   `voxelRatio`; the proxy geometry is re-made from `pool.info(id)` at each swap. Both are done by
 *   `vdbCoarseFirstApply` (the `bind` option's work, called here because the swap handler is async).
 *
 * ## Deterministic test mode (`?test=1`)
 *
 * Fixed camera and parameters, jitter off, unthrottled 1 KiB chunks, the panel hidden, readback presentation, and
 * the stream held until each frame is read back. Publishes `window.__DEMO09__` for the e2e gate.
 *
 * `?presentation=readback` renders to an offscreen target and draws it on a 2D canvas (headless SwiftShader has no
 * compositor, DIVERGES V7); the default draws into the page's WebGPU canvas.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { vdbBakeCoarseFirst, parseVdb } from "@pmndrs/vdb/tools";
import { VDBGridNode, VDBVolumeMaterial, getDevice, vdbCoarseFirstApply, vdbStreamCoarseFirst, type VDBCoarseFirstEvent } from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import type { WebGPURenderer } from "three/webgpu";

interface PhaseRecord {
  phase: "preview" | "full";
  grid: string;
  /** `.nvdb` bytes decoded when the grid was added or swapped in. */
  fileBytes: number;
  /** Bytes transferred (the wire) when it was shown. */
  transferBytes: number;
  voxelRatio: number;
  densityScale: number;
  /** Pixels that differ from the background in the frame drawn for this phase. */
  nonBackground: number;
}

interface Demo09State {
  ready: boolean;
  error?: string;
  adapter?: string;
  presentation?: string;
  source?: string;
  totalBytes?: number;
  /** The stream reader's own expected total (K4.20), derived from the file's FileMetaData. */
  streamTotalBytes?: number | null;
  phases?: PhaseRecord[];
  sameId?: boolean;
  inPlace?: boolean;
  transferred?: number;
  done?: boolean;
}

declare global {
  interface Window {
    __DEMO09__?: Demo09State;
  }
}

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
const PRESENTATION = params.get("presentation") === "readback" || TEST_MODE ? "readback" : "direct";
const SPEED_KIB = Math.max(1, Number(params.get("speed") ?? "256"));
const FILE = params.get("file");
const SOURCE_VDB = "/fixtures/vdb-samples/smoke.vdb";
const DENSITY = 4;
const BACKGROUND = 0x223049;
/** Interactive: how long each phase stays on screen before the stream may continue, ms. */
const HOLD_MS = TEST_MODE ? 0 : 900;

const phaseEl = document.querySelector<HTMLDivElement>("#phase")!;
const barEl = document.querySelector<HTMLDivElement>("#bar > div")!;
const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const panelEl = document.querySelector<HTMLDivElement>("#panel")!;
const appEl = document.querySelector<HTMLDivElement>("#app")!;
const speedEl = document.querySelector<HTMLSelectElement>("#speed")!;
const restartEl = document.querySelector<HTMLButtonElement>("#restart")!;

speedEl.value = [...speedEl.options].some((o) => Number(o.value) === SPEED_KIB) ? String(SPEED_KIB) : speedEl.value;
speedEl.addEventListener("change", () => {
  const next = new URLSearchParams(location.search);
  next.set("speed", speedEl.value);
  location.search = next.toString();
});
restartEl.addEventListener("click", () => location.reload());

const state: Demo09State = { ready: false };
window.__DEMO09__ = state;

function fail(message: string): void {
  errEl.textContent = message;
  phaseEl.textContent = "failed";
  phaseEl.dataset.phase = "error";
  state.error = message;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const kib = (n: number) => `${(n / 1024).toFixed(1)} KiB`;

/** The baked file, or the one `?file=` names, as a `Response` with its total size. */
async function openSource(): Promise<{ response: Response; total: number; label: string }> {
  if (FILE) {
    const response = await fetch(FILE);
    if (!response.ok) throw new Error(`${FILE} -> ${response.status}`);
    return { response, total: Number(response.headers.get("content-length") ?? 0), label: FILE };
  }
  const resp = await fetch(SOURCE_VDB);
  if (!resp.ok) throw new Error(`${SOURCE_VDB} -> ${resp.status} (pnpm fixtures:vdb-samples)`);
  statsEl.textContent = "baking [density.lod48, density] in the page…";
  await sleep(0); // let the status paint before the bake blocks the main thread
  const baked = vdbBakeCoarseFirst(parseVdb(await resp.arrayBuffer()), { previewResolution: 48, filter: "box", quantize: "fpn" });
  const url = URL.createObjectURL(new Blob([baked as BlobPart], { type: "application/octet-stream" }));
  const response = await fetch(url);
  return { response, total: baked.length, label: `smoke.vdb baked in page (${kib(baked.length)}, ZIP)` };
}

/** Re-streams a response body at a fixed link speed, one pull at a time; `gate` can hold the next chunk. */
function throttle(
  response: Response,
  opts: { chunk: number; bytesPerSec: number; gate: () => Promise<void>; onBytes: (sent: number) => void },
): Response {
  const upstream = response.body!.getReader();
  let pending = new Uint8Array(0);
  let sent = 0;
  const body = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        await opts.gate();
        while (pending.length < opts.chunk) {
          const { done, value } = await upstream.read();
          if (done) break;
          const merged = new Uint8Array(pending.length + value.length);
          merged.set(pending);
          merged.set(value, pending.length);
          pending = merged;
        }
        if (pending.length === 0) {
          controller.close();
          return;
        }
        const chunk = pending.slice(0, opts.chunk);
        pending = pending.slice(chunk.length);
        await sleep(opts.bytesPerSec > 0 ? (chunk.length / opts.bytesPerSec) * 1000 : 0);
        sent += chunk.length;
        opts.onBytes(sent);
        controller.enqueue(chunk);
      },
      cancel: () => upstream.cancel(),
    },
    { highWaterMark: 0 },
  );
  return new Response(body);
}

async function run(): Promise<void> {
  if (!navigator.gpu) {
    fail("WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.");
    return;
  }
  if (TEST_MODE) panelEl.classList.add("test-hidden");

  const source = await openSource();
  state.source = source.label;
  state.totalBytes = source.total;

  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const created = await createDeviceRenderer({ ...(directCanvas ? { canvas: directCanvas } : {}) });
  const renderer: WebGPURenderer = created.renderer;
  const display = directCanvas ?? document.createElement("canvas");
  display.style.width = "100%";
  display.style.height = "100%";
  appEl.appendChild(display);
  const ctx = PRESENTATION === "readback" ? display.getContext("2d") : null;

  const scene = new THREE.Scene();
  const bg = new THREE.Color(BACKGROUND);
  scene.background = bg;
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
  const controls = new OrbitControls(camera, display);
  controls.enableDamping = !TEST_MODE;

  const rt = new THREE.RenderTarget(2, 2);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
  const bgSrgb = bg.clone().convertLinearToSRGB();
  const bgBytes = [bgSrgb.r, bgSrgb.g, bgSrgb.b].map((c) => Math.round(c * 255));
  let flip = new Uint8ClampedArray(0);
  let vw = 0;
  let vh = 0;
  let nonBackground = 0;

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

  const pool = await VDBGridPool.create(getDevice(renderer));
  let grids: VDBGridNode | null = null;
  let material: VDBVolumeMaterial | null = null;
  let mesh: THREE.Mesh | null = null;
  let id = -1;

  // One render at a time: the swap work and the interactive loop both draw.
  let chain: Promise<unknown> = Promise.resolve();
  const serial = <T>(fn: () => Promise<T>): Promise<T> => {
    const next = chain.then(fn, fn);
    chain = next.catch(() => undefined);
    return next;
  };

  async function presentFrame(): Promise<number> {
    if (!grids || !mesh) return 0;
    grids.update();
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      return 0;
    }
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    const pixels = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
    renderer.setRenderTarget(null);
    const packed = vw * 4;
    const stride = vh > 1 ? (pixels.byteLength - packed) / (vh - 1) : packed;
    let count = 0;
    for (let y = 0; y < vh; y++) {
      const row = pixels.subarray((vh - 1 - y) * stride, (vh - 1 - y) * stride + packed);
      flip.set(row, y * packed);
      for (let i = 0; i < packed; i += 4) {
        if (Math.abs(row[i]! - bgBytes[0]!) + Math.abs(row[i + 1]! - bgBytes[1]!) + Math.abs(row[i + 2]! - bgBytes[2]!) > 6) count++;
      }
    }
    ctx!.putImageData(new ImageData(flip, vw, vh), 0, 0);
    return count;
  }

  if (TEST_MODE) setViewport(480, 360);
  else {
    setViewport(window.innerWidth, window.innerHeight);
    window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));
  }

  // Phase bookkeeping and the on-screen counter.
  const phases: PhaseRecord[] = [];
  let transferred = 0;
  // K4.20: the reader's own total (the grids' FileMetaData, once they have arrived) drives the counter; the
  // Response total is only the fallback for the first chunks.
  let streamTotal: number | null = null;
  const showCounter = (): void => {
    const total = streamTotal ?? source.total;
    const pct = total > 0 ? Math.min(100, (transferred / total) * 100) : 0;
    barEl.style.width = `${pct}%`;
    const first = phases[0];
    const full = phases[1];
    statsEl.textContent =
      `received ${kib(transferred)}${total ? ` of ${kib(total)} (${pct.toFixed(0)}%)` : ""}\n` +
      `link ${SPEED_KIB} KiB/s${TEST_MODE ? " (test: unthrottled)" : ""}\n` +
      (first ? `preview drawn after ${kib(first.transferBytes)}\n` : "") +
      (full ? `full grid in after ${kib(full.transferBytes)}\n` : "") +
      `${source.label}`;
    state.transferred = transferred;
  };

  let fresh = true;
  let versionAfterPreview = -1;
  /** Make the grid `e` visible: flush the pool, (re)build the scene pieces, draw it, record the phase. */
  async function showGrid(e: VDBCoarseFirstEvent): Promise<void> {
    pool.flush();
    await pool.settled();
    if (!e.final) versionAfterPreview = pool.version;
    id = e.id;
    const first = fresh;
    if (first) {
      fresh = false;
      grids = new VDBGridNode(renderer, pool, { helpers: ["rayClipSupport"] });
      material = new VDBVolumeMaterial(renderer, grids, id, { jitter: !TEST_MODE, densityScale: DENSITY });
      mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material); // the proxy geometry is made below, per swap
      scene.add(mesh);
      const info = pool.info(id);
      const box = new THREE.Box3(new THREE.Vector3(...info.worldBBox.min), new THREE.Vector3(...info.worldBBox.max));
      const center = box.getCenter(new THREE.Vector3());
      const diag = box.getSize(new THREE.Vector3()).length();
      camera.near = diag * 0.01;
      camera.far = diag * 10;
      camera.position.set(center.x + diag * 0.9, center.y + diag * 0.35, center.z + diag * 1.1);
      camera.lookAt(center);
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
      if (TEST_MODE) {
        material.sunDirection.value.set(0.7, 0.5, 0.45).normalize();
        material.stepSize.value = 0.45;
      }
    }
    // Per swap: the proxy box from pool.info(id), the grid node, and densityScale x voxelRatio (the march steps in
    // index voxels, so a preview with k-times larger voxels has 1/k the optical depth).
    vdbCoarseFirstApply(pool, e, { grids: grids!, mesh: mesh!, material: material!, densityScale: DENSITY });
    const densityScale = material!.densityScale.value;
    const phase = e.final ? "full" : "preview";
    phaseEl.dataset.phase = phase;
    phaseEl.textContent = e.final ? "FULL" : `PREVIEW · ${e.grid.metadata.name} · ${e.voxelRatio.toFixed(1)}× coarser`;
    // Compile on the first frame, then a settled one.
    nonBackground = await serial(presentFrame);
    if (first) nonBackground = await serial(presentFrame);
    phases.push({
      phase,
      grid: e.grid.metadata.name,
      fileBytes: e.bytesReceived,
      transferBytes: transferred,
      voxelRatio: e.voxelRatio,
      densityScale,
      nonBackground,
    });
    showCounter();
  }

  // The stream is held until the grid just shown has been drawn (and, interactively, has stayed up for HOLD_MS).
  let held: Promise<void> | null = null;
  const speed = SPEED_KIB * 1024;
  const throttled = throttle(source.response, {
    chunk: TEST_MODE ? 1024 : Math.max(1024, Math.round(speed / 200)),
    bytesPerSec: TEST_MODE ? 0 : speed,
    gate: async () => {
      if (held) await held;
    },
    onBytes: (sent) => {
      transferred = sent;
      showCounter();
    },
  });

  // Interactive draw loop (test mode draws only on swaps).
  if (!TEST_MODE) {
    let inFlight = false; // never queue behind a slow frame: the swap work needs its turn in `serial`
    const loop = (): void => {
      requestAnimationFrame(loop);
      if (!grids || state.error || inFlight) return;
      inFlight = true;
      controls.update();
      serial(presentFrame)
        .catch((err: unknown) => fail(err instanceof Error ? err.message : String(err)))
        .finally(() => {
          inFlight = false;
        });
    };
    requestAnimationFrame(loop);
  }

  const result = await vdbStreamCoarseFirst(pool, throttled, {
    onProgress: (p) => {
      streamTotal = p.bytesTotal;
      state.streamTotalBytes = p.bytesTotal;
    },
    onGrid: (e) => {
      held = (async () => {
        await showGrid(e);
        if (HOLD_MS > 0) await sleep(HOLD_MS);
      })();
      held.catch((err: unknown) => fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
    },
  });
  await held;
  await serial(async () => undefined);

  state.phases = phases;
  state.sameId = result.shown.every((s) => s.id === result.id);
  state.inPlace = pool.version === versionAfterPreview;
  state.adapter = created.adapterName;
  state.presentation = PRESENTATION;
  state.done = true;
  state.ready = true;
  phaseEl.textContent = "FULL · done";
  showCounter();
}

run().catch((err: unknown) => {
  fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
});
