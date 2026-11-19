/**
 * The page shell of gallery demos 13 and 14 (shared design spec of demos 11–16): a full-bleed canvas, a caption card
 * top-left (number · title, one sentence, "← all demos", a dim stats line, a status line with a progress bar, a
 * note for fallbacks), a small control strip bottom-right, gentle auto camera motion that stops on interaction, and
 * the e2e plumbing (readback presentation, a frame summary, a coverage probe).
 *
 * Query: `?presentation=readback` (an offscreen HalfFloat target read back to a 2D canvas and tone mapped on the CPU,
 * for headless SwiftShader e2e, as demo 10), `?still=1` (no camera motion), `?panel=0` (no UI), `?gallery=0` (don't
 * load the git-ignored `fixtures/gallery/` assets: the procedural fallbacks).
 *
 * Frame times are submit-to-done wall times of one frame on this page: provisional (D25), not a benchmark.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { createDeviceRenderer } from "./device-renderer.js";
import { Panel } from "./10-frame-renderer/panel.js";

export interface ShowcaseState {
  ready: boolean;
  loading: boolean;
  error?: string;
  frames: number;
  frameMs?: number;
  adapter?: string;
  presentation: "direct" | "readback";
  /** Whether the page may load `fixtures/gallery/` assets (`?gallery=0` turns them off). */
  galleryAssets: boolean;
  uncaptured: string[];
  /** Demo-specific facts for the e2e spec. */
  info: Record<string, unknown>;
}

export interface ShowcaseSnapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  /** 32 × 18 grey thumbnail (0–255), row-major, top row first. */
  thumb: number[];
}

export interface ShowcaseApi {
  capture(): ShowcaseSnapshot | null;
  /** Readback only: renders once without the demo's content and counts the pixels that change (sum |d| > 12). */
  coverage(): Promise<number>;
}

export const params = new URLSearchParams(location.search);
export const PRESENTATION: "direct" | "readback" = params.get("presentation") === "readback" ? "readback" : "direct";
export const STILL = params.get("still") === "1";
export const PANEL = params.get("panel") !== "0";
export const GALLERY_ASSETS = params.get("gallery") !== "0";

const STYLE = /* css */ `
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; overflow: hidden; background: #0d0f14; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
#app { position: fixed; inset: 0; }
#app canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.sc-card {
  position: fixed; margin: 16px; padding: 12px 14px; color: #e8ecf6; font-size: 13px; line-height: 1.4;
  background: rgba(14, 17, 26, 0.66); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 10px;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
}
#caption { top: 0; left: 0; width: 360px; max-width: calc(100vw - 32px); }
#caption h1 { margin: 0 0 4px; font-size: 14px; font-weight: 600; letter-spacing: 0.01em; }
#caption h1 .num { opacity: 0.55; font-weight: 500; }
#caption p { margin: 0 0 6px; opacity: 0.88; }
#caption a { color: #a9bcff; text-decoration: none; }
#caption a:hover { text-decoration: underline; }
#sc-stats { margin-top: 8px; font-size: 11.5px; line-height: 1.45; opacity: 0.58; font-variant-numeric: tabular-nums; white-space: pre-line; }
#sc-status { margin-top: 8px; font-size: 12px; opacity: 0.9; }
#sc-status:empty, #sc-status[hidden] { display: none; }
#sc-bar { height: 3px; margin-top: 5px; border-radius: 2px; background: rgba(255, 255, 255, 0.12); overflow: hidden; }
#sc-bar > div { height: 100%; width: 0%; background: #a9bcff; transition: width 120ms linear; }
#sc-bar.indeterminate > div { width: 30%; animation: sc-slide 1.1s ease-in-out infinite; }
@keyframes sc-slide { from { margin-left: -30%; } to { margin-left: 100%; } }
#sc-note { margin-top: 8px; font-size: 11.5px; padding: 5px 7px; border-radius: 6px; background: rgba(255, 200, 110, 0.10); color: #f3d7a6; white-space: pre-line; }
#sc-note:empty { display: none; }
#sc-error { margin-top: 8px; font-size: 12px; color: #ff9f9f; white-space: pre-wrap; max-height: 40vh; overflow: auto; }
#sc-error:empty { display: none; }
#controls { right: 0; bottom: 0; width: 290px; max-width: calc(100vw - 32px); padding: 10px 12px; }
#controls .ctl { display: grid; grid-template-columns: 92px 1fr 34px; align-items: center; gap: 8px; margin: 3px 0; font-size: 12.5px; }
#controls .ctl.wide { grid-template-columns: 92px 1fr; }
#controls .ctl label { opacity: 0.72; }
#controls .ctl output { text-align: right; font-variant-numeric: tabular-nums; opacity: 0.8; font-size: 11.5px; }
#controls .ctl.disabled { opacity: 0.4; }
#controls .ctl[hidden] { display: none; }
#controls select { width: 100%; font: inherit; font-size: 12.5px; color: #e8ecf6; background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(255, 255, 255, 0.14); border-radius: 6px; padding: 3px 6px; }
#controls select option { background: #1a1e2a; }
#controls input[type="range"] { width: 100%; accent-color: #a9bcff; }
#controls input[type="checkbox"] { accent-color: #a9bcff; }
@media (max-width: 640px) { #caption { width: auto; right: 0; } #controls { width: auto; left: 0; } }
`;

export interface ShowcaseUi {
  readonly app: HTMLDivElement;
  readonly controls: Panel;
  readonly state: ShowcaseState;
  stats(text: string): void;
  /** A status line; `progress` 0–1 fills the bar, `null` makes it indeterminate, `undefined` hides it. */
  status(text: string, progress?: number | null): void;
  note(text: string): void;
  fail(message: string): void;
}

/** Builds the caption card and the control strip, and the e2e state object at `window[stateKey]`. */
export function mountShowcase(cfg: { number: string; title: string; blurb: string; stateKey: string }): ShowcaseUi {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.append(style);
  const app = document.createElement("div");
  app.id = "app";
  const caption = document.createElement("div");
  caption.id = "caption";
  caption.className = "sc-card";
  caption.dataset.testid = "caption";
  caption.innerHTML =
    `<h1><span class="num">${cfg.number} ·</span> ${cfg.title}</h1><p>${cfg.blurb}</p><a href="/">&larr; all demos</a>` +
    `<div id="sc-status" data-testid="status" hidden><span></span><div id="sc-bar"><div></div></div></div>` +
    `<div id="sc-stats" data-testid="stats">starting WebGPU…</div><div id="sc-note" data-testid="note"></div><div id="sc-error" data-testid="error"></div>`;
  const strip = document.createElement("div");
  strip.id = "controls";
  strip.className = "sc-card";
  strip.dataset.testid = "controls";
  document.body.append(app, caption, strip);
  if (!PANEL) {
    caption.style.display = "none";
    strip.style.display = "none";
  }
  const state: ShowcaseState = { ready: false, loading: true, frames: 0, presentation: PRESENTATION, galleryAssets: GALLERY_ASSETS, uncaptured: [], info: {} };
  (window as unknown as Record<string, unknown>)[cfg.stateKey] = state;

  const statsEl = caption.querySelector<HTMLDivElement>("#sc-stats")!;
  const statusEl = caption.querySelector<HTMLDivElement>("#sc-status")!;
  const statusText = statusEl.querySelector("span")!;
  const bar = caption.querySelector<HTMLDivElement>("#sc-bar")!;
  const noteEl = caption.querySelector<HTMLDivElement>("#sc-note")!;
  const errEl = caption.querySelector<HTMLDivElement>("#sc-error")!;
  const ui: ShowcaseUi = {
    app,
    controls: new Panel(strip),
    state,
    stats: (t) => {
      statsEl.textContent = t;
    },
    status: (t, progress) => {
      statusEl.hidden = t === "";
      statusText.textContent = t;
      bar.hidden = progress === undefined;
      bar.classList.toggle("indeterminate", progress === null);
      (bar.firstElementChild as HTMLDivElement).style.width = progress == null ? "" : `${Math.round(Math.min(1, Math.max(0, progress)) * 100)}%`;
    },
    note: (t) => {
      noteEl.textContent = t;
    },
    fail: (message) => {
      errEl.textContent = message;
      state.error = message;
      state.loading = false;
      if (!PANEL) caption.style.display = "";
    },
  };
  const errorText = (err: unknown): string => (err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
  window.addEventListener("error", (e) => ui.fail(errorText(e.error ?? e.message)));
  window.addEventListener("unhandledrejection", (e) => ui.fail(errorText(e.reason)));
  return ui;
}

export const errorText = (err: unknown): string => (err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));

/** True when a HEAD request for the URL succeeds. */
export async function probe(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { method: "HEAD" });
    return r.ok;
  } catch {
    return false;
  }
}

/** Fetches a URL with byte progress (`onProgress(received, total)`; total 0 when the server sends no length). */
export async function fetchWithProgress(url: string, onProgress: (received: number, total: number) => void): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok || !r.body) throw new Error(`fetch ${url}: ${r.status}`);
  const total = Number(r.headers.get("content-length") ?? 0);
  const reader = r.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.byteLength;
  }
  return out.buffer;
}

export const MiB = (bytes: number): string =>
  bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : `${(bytes / 1048576).toFixed(bytes < 10 * 1048576 ? 2 : 1)} MiB`;

// --- Tone mapping of the read-back frame (three's ACESFilmic, then sRGB), so readback matches direct ------------

function aces(r: number, g: number, b: number, exposure: number, out: number[]): void {
  const k = exposure / 0.6;
  r *= k;
  g *= k;
  b *= k;
  const ir = 0.59719 * r + 0.35458 * g + 0.04823 * b;
  const ig = 0.076 * r + 0.90834 * g + 0.01566 * b;
  const ib = 0.0284 * r + 0.13383 * g + 0.83777 * b;
  const fit = (v: number): number => (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.432951) + 0.238081);
  const fr = fit(ir);
  const fg = fit(ig);
  const fb = fit(ib);
  out[0] = 1.60475 * fr - 0.53108 * fg - 0.07367 * fb;
  out[1] = -0.10208 * fr + 1.10813 * fg - 0.00605 * fb;
  out[2] = -0.00327 * fr - 0.07276 * fg + 1.07602 * fb;
}
const srgbByte = (v: number): number => {
  const c = v <= 0 ? 0 : v >= 1 ? 1 : v < 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return Math.round(c * 255);
};

export interface Viewer {
  renderer: THREE.WebGPURenderer;
  device: GPUDevice;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  orbit: OrbitControls;
  /** The demo's own objects: hidden by the coverage probe. */
  content: THREE.Group;
  /** Auto camera motion (sway or orbit around `orbit.target`); off with `?still=1` and after any interaction. */
  motion: { enabled: boolean; rebase(): void };
  /** Starts the render loop; `tick` runs before each frame. */
  start(tick?: (now: number) => void, statsLine?: () => string): void;
  api: ShowcaseApi;
}

/**
 * The device-first renderer (`device-renderer.ts`), the canvas (direct, or a 2D canvas fed by readback), an
 * `OrbitControls` camera with gentle auto motion, and the loop that keeps `ui.state` current.
 */
export async function createViewer(
  ui: ShowcaseUi,
  options: { fov?: number; near?: number; far?: number; exposure?: number; motion?: "sway" | "orbit"; swayDegrees?: number; speed?: number; pixelRatio?: number },
): Promise<Viewer> {
  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const created = await createDeviceRenderer(directCanvas ? { canvas: directCanvas } : {});
  const { renderer, device } = created;
  device.addEventListener("uncapturederror", (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    ui.state.uncaptured.push(message);
    console.error(`WebGPU: ${message}`);
  });
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") ui.fail(`WebGPU device lost: ${info.message}`);
  });
  ui.state.adapter = created.adapterName || "unreported adapter";
  const exposure = options.exposure ?? 1;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = exposure;

  const display = directCanvas ?? document.createElement("canvas");
  ui.app.append(display);
  const ctx2d = PRESENTATION === "readback" ? display.getContext("2d") : null;

  const scene = new THREE.Scene();
  const content = new THREE.Group();
  scene.add(content);
  const camera = new THREE.PerspectiveCamera(options.fov ?? 40, 1, options.near ?? 0.05, options.far ?? 500);
  const orbit = new OrbitControls(camera, display);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.dampingFactor = 0.08;

  // Auto motion: sway the azimuth around where the camera was put (or orbit), until the user touches the view.
  const motionMode = options.motion ?? "sway";
  const sway = THREE.MathUtils.degToRad(options.swayDegrees ?? 18);
  const speed = options.speed ?? 0.12;
  const base = new THREE.Spherical();
  let motionStart = performance.now();
  const motion = {
    enabled: !STILL,
    rebase(): void {
      base.setFromVector3(camera.position.clone().sub(orbit.target));
      motionStart = performance.now();
    },
  };
  orbit.addEventListener("start", () => {
    motion.enabled = false;
  });
  motion.rebase();
  const offset = new THREE.Vector3();
  const applyMotion = (now: number): void => {
    if (!motion.enabled) return;
    const t = (now - motionStart) / 1000;
    const s = base.clone();
    // ease in from the rest pose so a rebase never jumps
    s.theta = motionMode === "orbit" ? base.theta + t * speed : base.theta + sway * Math.sin(t * speed * 2 * Math.PI * 0.25);
    offset.setFromSpherical(s);
    camera.position.copy(orbit.target).add(offset);
    camera.lookAt(orbit.target);
  };

  const rt = new THREE.RenderTarget(2, 2, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  let vw = 0;
  let vh = 0;
  let lastFrame: Uint8ClampedArray<ArrayBuffer> | null = null;
  const pixelRatio = options.pixelRatio ?? Math.min(2, window.devicePixelRatio || 1);
  function setViewport(): void {
    vw = Math.max(1, ui.app.clientWidth || window.innerWidth);
    vh = Math.max(1, ui.app.clientHeight || window.innerHeight);
    if (PRESENTATION === "direct") {
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(vw, vh, false);
    } else {
      display.width = vw;
      display.height = vh;
      rt.setSize(vw, vh);
    }
    camera.aspect = vw / vh;
    camera.updateProjectionMatrix();
  }
  setViewport();
  window.addEventListener("resize", setViewport);

  const tone = [0, 0, 0];
  async function present(): Promise<Uint8ClampedArray<ArrayBuffer> | null> {
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      await device.queue.onSubmittedWorkDone();
      return null;
    }
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const raw = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as unknown as Uint16Array;
    const tight = vw * 4;
    const stride = vh > 1 ? (raw.length - tight) / (vh - 1) : tight;
    if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${raw.length} halves for ${vw}x${vh}`);
    const out = new Uint8ClampedArray(vw * vh * 4);
    const half = THREE.DataUtils.fromHalfFloat;
    for (let y = 0; y < vh; y++) {
      for (let x = 0; x < vw; x++) {
        const s = y * stride + x * 4;
        const d = (y * vw + x) * 4;
        aces(half(raw[s]!), half(raw[s + 1]!), half(raw[s + 2]!), renderer.toneMappingExposure, tone);
        out[d] = srgbByte(tone[0]!);
        out[d + 1] = srgbByte(tone[1]!);
        out[d + 2] = srgbByte(tone[2]!);
        out[d + 3] = 255;
      }
    }
    return out;
  }

  function capture(): ShowcaseSnapshot | null {
    if (!lastFrame) return null;
    const TW = 32;
    const TH = 18;
    const thumb = new Array<number>(TW * TH).fill(0);
    const counts = new Array<number>(TW * TH).fill(0);
    const mean: [number, number, number] = [0, 0, 0];
    for (let y = 0; y < vh; y++) {
      for (let x = 0; x < vw; x++) {
        const i = (y * vw + x) * 4;
        const r = lastFrame[i]!;
        const g = lastFrame[i + 1]!;
        const b = lastFrame[i + 2]!;
        mean[0] += r;
        mean[1] += g;
        mean[2] += b;
        const t = Math.min(TH - 1, Math.floor((y * TH) / vh)) * TW + Math.min(TW - 1, Math.floor((x * TW) / vw));
        thumb[t]! += (r + g + b) / 3;
        counts[t]!++;
      }
    }
    const n = vw * vh;
    return { width: vw, height: vh, mean: mean.map((v) => v / n) as [number, number, number], thumb: thumb.map((v, i) => v / Math.max(1, counts[i]!)) };
  }

  let busy = false;
  async function coverage(): Promise<number> {
    if (PRESENTATION !== "readback") throw new Error("coverage() needs ?presentation=readback");
    while (busy) await new Promise((r) => setTimeout(r, 5));
    busy = true;
    try {
      const withContent = await present();
      content.visible = false;
      const without = await present();
      content.visible = true;
      let n = 0;
      for (let i = 0; i < withContent!.length; i += 4) {
        const d = Math.abs(withContent![i]! - without![i]!) + Math.abs(withContent![i + 1]! - without![i + 1]!) + Math.abs(withContent![i + 2]! - without![i + 2]!);
        if (d > 12) n++;
      }
      return n;
    } finally {
      content.visible = true;
      busy = false;
    }
  }

  const api: ShowcaseApi = { capture, coverage };
  let frameMs = 0;
  let lastStats = 0;

  function start(tick?: (now: number) => void, statsLine?: () => string): void {
    const loop = (now: number): void => {
      requestAnimationFrame(loop);
      if (busy || ui.state.error) return;
      busy = true;
      applyMotion(now);
      orbit.update();
      try {
        tick?.(now);
      } catch (err) {
        busy = false;
        ui.fail(errorText(err));
        return;
      }
      const t0 = performance.now();
      present()
        .then((frame) => {
          if (frame) {
            lastFrame = frame;
            ctx2d!.putImageData(new ImageData(frame, vw, vh), 0, 0);
          }
          const ms = performance.now() - t0;
          frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
          ui.state.frames++;
          ui.state.frameMs = frameMs;
          if (now - lastStats > 400 || ui.state.frames <= 3) {
            lastStats = now;
            const px = PRESENTATION === "direct" ? `${Math.round(vw * pixelRatio)}×${Math.round(vh * pixelRatio)}` : `${vw}×${vh}`;
            ui.stats(`${ui.state.adapter} · ${px} · ${frameMs.toFixed(1)} ms/frame (provisional)` + (statsLine ? `\n${statsLine()}` : ""));
          }
        })
        .catch((err: unknown) => ui.fail(errorText(err)))
        .finally(() => {
          busy = false;
        });
    };
    requestAnimationFrame(loop);
  }

  return { renderer, device, scene, camera, orbit, content, motion, start, api };
}
