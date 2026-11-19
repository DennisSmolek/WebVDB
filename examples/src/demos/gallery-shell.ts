/**
 * The gallery shell for demos 11 and 15 (the shared gallery spec): a full-bleed canvas, a caption card top-left
 * (number, title, one sentence, "all demos" link, a dim stats line, notes and errors), a small control strip
 * bottom-right, the query switches every gallery page takes, and the two presentations:
 *
 * - `direct` (default): three renders into the page canvas (tone mapped).
 * - `readback` (`?presentation=readback`, for headless SwiftShader e2e, as demos 08 and 10): a HalfFloat target with
 *   depth is read back, sRGB-encoded on the CPU and drawn on a 2D canvas; `capture()` summarizes the last frame.
 *
 * Query: `?panel=0` hides the UI (clean screenshots), `?still=1` stops camera motion, `?gallery=0` acts as if the
 * git-ignored `fixtures/gallery/` assets were absent (the e2e runs both ways), `?quality=low|medium|high`.
 * Engine-neutral apart from three: no package import here.
 */
import * as THREE from "three/webgpu";

export const params = new URLSearchParams(location.search);
export const PRESENTATION: "direct" | "readback" = params.get("presentation") === "readback" ? "readback" : "direct";
export const STILL = params.get("still") === "1" || PRESENTATION === "readback";
export const SHOW_PANEL = params.get("panel") !== "0";
/** `?gallery=0`: behave as if `fixtures/gallery/` were missing. */
export const GALLERY_ENABLED = params.get("gallery") !== "0";
export const GALLERY_DIR = "/fixtures/gallery/";

export type Quality = "low" | "medium" | "high";
export function qualityParam(fallback: Quality): Quality {
  const q = params.get("quality");
  return q === "low" || q === "medium" || q === "high" ? q : fallback;
}

/** HEAD-probe a file (the dev server serves `fixtures/` and answers 404 for a missing one). */
export async function probe(url: string): Promise<boolean> {
  if (url.startsWith(GALLERY_DIR) && !GALLERY_ENABLED) return false;
  try {
    return (await fetch(url, { method: "HEAD" })).ok;
  } catch {
    return false;
  }
}

export async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

export const formatBytes = (bytes: number): string =>
  bytes >= 1048576 ? `${(bytes / 1048576).toFixed(bytes < 10 * 1048576 ? 2 : 1)} MiB` : `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KiB`;

const CSS = `
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; overflow: hidden; background: #0b1020; }
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
#app { position: fixed; inset: 0; }
#app canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.g-card {
  position: fixed; margin: 12px; padding: 12px 14px;
  background: rgba(12, 16, 32, 0.72); color: #e6ebff; border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.12); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  font-size: 13px; line-height: 1.4;
}
#g-caption { top: 0; left: 0; width: 340px; max-width: calc(100vw - 24px); }
#g-caption h1 { font-size: 14px; font-weight: 600; margin: 0 0 4px; letter-spacing: 0.01em; }
#g-caption p { margin: 0 0 6px; opacity: 0.88; }
#g-caption a { color: #9db4ff; text-decoration: none; }
#g-caption a:hover { text-decoration: underline; }
#g-stats { font-size: 11.5px; opacity: 0.55; font-variant-numeric: tabular-nums; white-space: pre-line; margin-top: 6px; }
#g-status { font-size: 12px; opacity: 0.75; font-style: italic; margin-top: 4px; }
#g-status:empty { display: none; }
#g-note { margin-top: 8px; padding: 6px 8px; border-radius: 6px; background: rgba(255, 190, 90, 0.12); color: #ffd69b; font-size: 12px; white-space: pre-line; }
#g-note:empty { display: none; }
#g-error { margin-top: 8px; color: #ff9b9b; font-size: 12px; white-space: pre-wrap; max-height: 40vh; overflow: auto; }
#g-error:empty { display: none; }
#g-controls { right: 0; bottom: 0; width: 280px; max-width: calc(100vw - 24px); padding: 10px 12px; }
.g-ctl { display: grid; grid-template-columns: 92px 1fr 44px; align-items: center; gap: 8px; margin: 3px 0; }
.g-ctl.wide { grid-template-columns: 92px 1fr; }
.g-ctl label { opacity: 0.8; font-size: 12.5px; }
.g-ctl output { text-align: right; font-variant-numeric: tabular-nums; opacity: 0.85; font-size: 12px; }
.g-ctl select, .g-ctl input[type="range"] { width: 100%; font: inherit; font-size: 12.5px; }
.g-ctl select { background: rgba(255,255,255,0.08); color: inherit; border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; padding: 2px 4px; }
.g-ctl select option { background: #1a2040; }
.g-ctl input[type="range"] { accent-color: #9db4ff; }
.g-hidden { display: none !important; }
@media (max-width: 640px) { #g-caption { width: auto; right: 0; } #g-controls { width: auto; left: 0; } }
`;

export interface SelectOption<T extends string> {
  value: T;
  label: string;
}

export interface ControlRow<T> {
  readonly el: HTMLDivElement;
  readonly input: HTMLInputElement | HTMLSelectElement;
  set(value: T): void;
}

/** Caption card, control strip and the page's error and status lines. */
export class GalleryShell {
  readonly app: HTMLDivElement;
  private readonly statsEl: HTMLDivElement;
  private readonly statusEl: HTMLDivElement;
  private readonly noteEl: HTMLDivElement;
  private readonly errorEl: HTMLDivElement;
  private readonly controlsEl: HTMLDivElement;
  /** Set by `fail()`; the loop stops drawing. */
  error: string | undefined;
  onError: ((message: string) => void) | null = null;

  constructor(opts: { number: string; title: string; blurb: string }) {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.append(style);
    document.title = `${opts.number} · ${opts.title} · @pmndrs/vdb gallery`;

    this.app = document.createElement("div");
    this.app.id = "app";
    const caption = document.createElement("div");
    caption.id = "g-caption";
    caption.className = "g-card";
    caption.dataset.testid = "caption";
    const h1 = document.createElement("h1");
    h1.textContent = `${opts.number} · ${opts.title}`;
    const p = document.createElement("p");
    p.textContent = opts.blurb;
    const back = document.createElement("a");
    back.href = "/";
    back.textContent = "← all demos";
    this.statsEl = div("g-stats", "stats");
    this.statsEl.textContent = "starting…";
    this.statusEl = div("g-status", "status");
    this.noteEl = div("g-note", "note");
    this.errorEl = div("g-error", "error");
    caption.append(h1, p, back, this.statsEl, this.statusEl, this.noteEl, this.errorEl);

    this.controlsEl = div("g-controls", "controls");
    this.controlsEl.className = "g-card";
    document.body.append(this.app, caption, this.controlsEl);
    if (!SHOW_PANEL) {
      caption.classList.add("g-hidden");
      this.controlsEl.classList.add("g-hidden");
    }

    const text = (err: unknown): string => (err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
    window.addEventListener("error", (e) => this.fail(text(e.error ?? e.message)));
    window.addEventListener("unhandledrejection", (e) => this.fail(text(e.reason)));
  }

  stats(text: string): void {
    this.statsEl.textContent = text;
  }

  status(text: string): void {
    this.statusEl.textContent = text;
  }

  note(text: string): void {
    this.noteEl.textContent = text;
  }

  fail(message: unknown): void {
    const m = message instanceof Error ? `${message.message}\n${message.stack ?? ""}` : String(message);
    this.errorEl.textContent = m;
    this.error = m;
    this.status("");
    this.onError?.(m);
  }

  select<T extends string>(key: string, label: string, options: readonly SelectOption<T>[], value: T, onChange: (v: T) => void): ControlRow<T> {
    const select = document.createElement("select");
    for (const o of options) {
      const el = document.createElement("option");
      el.value = o.value;
      el.textContent = o.label;
      select.append(el);
    }
    select.value = value;
    select.addEventListener("change", () => onChange(select.value as T));
    return { ...this.row(key, label, select, null), set: (v: T) => void (select.value = v) };
  }

  range(key: string, label: string, min: number, max: number, step: number, value: number, onChange: (v: number) => void, format: (v: number) => string = (v) => v.toFixed(step < 1 ? 2 : 0)): ControlRow<number> {
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    const out = document.createElement("output");
    out.textContent = format(value);
    input.addEventListener("input", () => {
      out.textContent = format(Number(input.value));
      onChange(Number(input.value));
    });
    return {
      ...this.row(key, label, input, out),
      set: (v: number) => {
        input.value = String(v);
        out.textContent = format(v);
      },
    };
  }

  private row(key: string, label: string, input: HTMLInputElement | HTMLSelectElement, out: HTMLOutputElement | null): Omit<ControlRow<never>, "set"> {
    const el = document.createElement("div");
    el.className = out ? "g-ctl" : "g-ctl wide";
    input.id = `ctl-${key}`;
    const lab = document.createElement("label");
    lab.htmlFor = input.id;
    lab.textContent = label;
    el.append(lab, input);
    if (out) el.append(out);
    this.controlsEl.append(el);
    return { el, input };
  }
}

function div(id: string, testid: string): HTMLDivElement {
  const el = document.createElement("div");
  el.id = id;
  el.dataset.testid = testid;
  return el;
}

// --- Presentation -----------------------------------------------------------------------------------------

/** A summary of the last read-back frame (e2e). */
export interface FrameSnapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  /** 32 × 18 grey thumbnail (0–255), row-major, top row first. */
  thumb: number[];
  /** max − min of the thumbnail: a flat frame (nothing drawn) has none. */
  spread: number;
}

/** sRGB encode (three's formula, NoToneMapping) of a linear value to a byte. */
const srgbByte = (v: number): number => {
  const c = v <= 0 ? 0 : v >= 1 ? 1 : v < 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return Math.round(c * 255);
};

/**
 * The page's canvas and viewport in either presentation. `direct`: the renderer's own canvas. `readback`: three
 * renders into `target()`s and `readTarget()` turns one into sRGB bytes; `show()` draws bytes on the 2D canvas.
 */
export class Presenter {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx2d: CanvasRenderingContext2D | null;
  width = 1;
  height = 1;
  pixelRatio = 1;
  private last: Uint8ClampedArray | null = null;
  private readonly targets: THREE.RenderTarget[] = [];

  constructor(
    readonly app: HTMLElement,
    readonly renderer: THREE.WebGPURenderer,
    directCanvas: HTMLCanvasElement | undefined,
  ) {
    this.canvas = directCanvas ?? document.createElement("canvas");
    this.canvas.id = "view";
    app.append(this.canvas);
    this.ctx2d = PRESENTATION === "readback" ? this.canvas.getContext("2d") : null;
  }

  /** A HalfFloat target with depth (the froxel integrate pass reads the scene depth), sized with the viewport. */
  target(): THREE.RenderTarget {
    const rt = new THREE.RenderTarget(2, 2, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
    rt.setSize(this.pixelWidth, this.pixelHeight);
    this.targets.push(rt);
    return rt;
  }

  get pixelWidth(): number {
    return PRESENTATION === "direct" ? Math.round(this.width * this.pixelRatio) : this.width;
  }

  get pixelHeight(): number {
    return PRESENTATION === "direct" ? Math.round(this.height * this.pixelRatio) : this.height;
  }

  /** Fit the viewport to the page; returns the aspect ratio. */
  resize(pixelRatio: number, camera: THREE.PerspectiveCamera): void {
    this.width = Math.max(1, this.app.clientWidth || window.innerWidth);
    this.height = Math.max(1, this.app.clientHeight || window.innerHeight);
    this.pixelRatio = pixelRatio;
    if (PRESENTATION === "direct") {
      this.renderer.setPixelRatio(pixelRatio);
      this.renderer.setSize(this.width, this.height, false);
    } else {
      this.canvas.width = this.width;
      this.canvas.height = this.height;
    }
    for (const rt of this.targets) rt.setSize(this.pixelWidth, this.pixelHeight);
    camera.aspect = this.width / this.height;
    camera.updateProjectionMatrix();
  }

  /** Read a HalfFloat target back as sRGB RGBA bytes (rows top first, padding dropped). */
  async readTarget(rt: THREE.RenderTarget): Promise<Uint8ClampedArray> {
    const w = rt.width;
    const h = rt.height;
    const raw = (await this.renderer.readRenderTargetPixelsAsync(rt, 0, 0, w, h)) as unknown as Uint16Array;
    const tight = w * 4;
    const stride = h > 1 ? (raw.length - tight) / (h - 1) : tight;
    if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${raw.length} halves for ${w}x${h}`);
    const out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = y * stride + x * 4;
        const d = (y * w + x) * 4;
        out[d] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s]!));
        out[d + 1] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s + 1]!));
        out[d + 2] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s + 2]!));
        out[d + 3] = 255;
      }
    }
    return out;
  }

  /** Draw read-back bytes (the viewport's size) on the 2D canvas. */
  show(bytes: Uint8ClampedArray): void {
    this.ctx2d!.putImageData(new ImageData(new Uint8ClampedArray(bytes), this.width, this.height), 0, 0);
    this.last = bytes;
  }

  capture(): FrameSnapshot | null {
    const f = this.last;
    if (!f) return null;
    const w = this.width;
    const h = this.height;
    const TW = 32;
    const TH = 18;
    const thumb = new Array<number>(TW * TH).fill(0);
    const counts = new Array<number>(TW * TH).fill(0);
    const mean: [number, number, number] = [0, 0, 0];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        mean[0] += f[i]!;
        mean[1] += f[i + 1]!;
        mean[2] += f[i + 2]!;
        const t = Math.min(TH - 1, Math.floor((y * TH) / h)) * TW + Math.min(TW - 1, Math.floor((x * TW) / w));
        thumb[t]! += (f[i]! + f[i + 1]! + f[i + 2]!) / 3;
        counts[t]!++;
      }
    }
    const n = w * h;
    const t = thumb.map((v, i) => v / Math.max(1, counts[i]!));
    return { width: w, height: h, mean: mean.map((v) => v / n) as [number, number, number], thumb: t, spread: Math.max(...t) - Math.min(...t) };
  }
}

/**
 * Slow auto-orbit that stops for good on the first user interaction (`?still=1`: never starts). Call `update()`
 * every frame.
 */
export function autoOrbit(controls: { autoRotate: boolean; autoRotateSpeed: number; addEventListener(type: "start", cb: () => void): void }, speed: number): void {
  controls.autoRotate = !STILL;
  controls.autoRotateSpeed = speed;
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
  });
}
