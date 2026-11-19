/**
 * The gallery page shell shared by demos 12 and 16 (the gallery design spec): a full-bleed canvas, a caption
 * card top-left (title, one sentence, "← all demos", a dim stats line), a small control strip bottom-right,
 * gentle auto camera motion that stops on interaction, and the readback presentation the headless e2e uses.
 *
 * Query (every gallery page): `?presentation=readback` (an offscreen HalfFloat target read back to a 2D canvas, as
 * demo 10, for headless SwiftShader), `?panel=0` (no caption or controls), `?still=1` (no camera motion),
 * `?gallery=0` (ignore the git-ignored `fixtures/gallery/` assets, as if they were missing).
 *
 * Plain DOM and three only: no package import, so the shell is the same for every adapter demo.
 */
import * as THREE from "three/webgpu";
import type { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { UltraHDRLoader } from "three/addons/loaders/UltraHDRLoader.js";

export const params = new URLSearchParams(location.search);
export type Presentation = "direct" | "readback";
export const PRESENTATION: Presentation = params.get("presentation") === "readback" ? "readback" : "direct";
/** No camera motion (deterministic e2e frames). */
export const STILL = params.get("still") === "1";
/** Use the git-ignored gallery assets when present (`?gallery=0`: behave as if they were missing). */
export const GALLERY_ASSETS = params.get("gallery") !== "0";
export const PANEL = params.get("panel") !== "0";

const CSS = /* css */ `
:root { color-scheme: dark; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; overflow: hidden; background: #07090f; color: #e8ecf6; }
#app { position: fixed; inset: 0; }
#app canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.g-card {
  position: fixed; top: 0; left: 0; margin: 14px; padding: 12px 14px 11px;
  width: 340px; max-width: calc(100vw - 28px);
  background: rgba(10, 13, 22, 0.62); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 10px;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  font-size: 13.5px; line-height: 1.42; transition: box-shadow 0.25s ease, border-color 0.25s ease;
}
.g-card h1 { margin: 0 0 4px; font-size: 15px; font-weight: 650; letter-spacing: 0.01em; }
.g-card h1 .n { opacity: 0.55; font-weight: 500; margin-right: 2px; }
.g-card p { margin: 0 0 6px; opacity: 0.86; }
.g-card a { color: #a9bcff; text-decoration: none; font-size: 13px; }
.g-card a:hover { text-decoration: underline; }
.g-stats { margin-top: 7px; font-size: 12px; opacity: 0.52; font-variant-numeric: tabular-nums; white-space: pre-line; }
.g-status { margin-top: 6px; font-size: 12.5px; opacity: 0.8; font-style: italic; }
.g-status:empty, .g-note:empty, .g-err:empty { display: none; }
.g-note { margin-top: 7px; padding: 5px 8px; border-radius: 6px; font-size: 12px; background: rgba(255, 190, 90, 0.10); color: #ffd69b; white-space: pre-line; }
.g-err { margin-top: 7px; color: #ff9b9b; font-size: 12px; white-space: pre-wrap; max-height: 40vh; overflow: auto; }
.g-strip {
  position: fixed; right: 0; bottom: 0; margin: 14px; padding: 8px 10px;
  display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; justify-content: flex-end;
  max-width: calc(100vw - 28px);
  background: rgba(10, 13, 22, 0.62); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 10px;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); font-size: 13px;
}
.g-ctl { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.g-ctl label { opacity: 0.72; }
.g-ctl output { min-width: 3.2em; text-align: right; font-variant-numeric: tabular-nums; opacity: 0.85; }
.g-ctl input[type="range"] { width: 120px; accent-color: #a9bcff; }
.g-ctl input[type="checkbox"] { accent-color: #a9bcff; margin: 0; }
.g-ctl select, .g-ctl button {
  font: inherit; color: inherit; background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 6px; padding: 2px 8px; cursor: pointer;
}
.g-ctl select option { background: #141824; }
.g-ctl button:hover, .g-ctl select:hover { background: rgba(255, 255, 255, 0.14); }
.g-hidden { display: none !important; }
.g-card.g-flash-preview { animation: g-flash-preview 1.6s ease-out; }
.g-card.g-flash-full { animation: g-flash-full 1.8s ease-out; }
@keyframes g-flash-preview { 0% { box-shadow: 0 0 0 2px #ffd479, 0 0 28px rgba(255, 212, 121, 0.55); border-color: #ffd479; } 100% { box-shadow: 0 0 0 0 transparent; } }
@keyframes g-flash-full { 0% { box-shadow: 0 0 0 2px #7be3a1, 0 0 32px rgba(123, 227, 161, 0.6); border-color: #7be3a1; } 100% { box-shadow: 0 0 0 0 transparent; } }
.g-toast {
  position: fixed; left: 50%; top: 18px; transform: translateX(-50%); padding: 6px 14px; border-radius: 999px;
  background: rgba(10, 13, 22, 0.7); border: 1px solid rgba(255, 255, 255, 0.14); font-size: 13.5px; font-weight: 600;
  letter-spacing: 0.02em; opacity: 0; pointer-events: none; transition: opacity 0.5s ease;
}
.g-toast.show { opacity: 1; transition: opacity 0.15s ease; }
@media (max-width: 560px) { .g-card { width: auto; right: 0; } .g-ctl input[type="range"] { width: 90px; } }
`;

/** One labelled control in the strip; the input's id is `ctl-<key>` (the e2e drives it). */
export interface Control<T> {
  readonly el: HTMLElement;
  readonly input: HTMLInputElement | HTMLSelectElement | HTMLButtonElement;
  set(value: T): void;
  show(visible: boolean): void;
}

export interface ShellOptions {
  /** "12" */
  number: string;
  /** "Explosion" */
  title: string;
  /** One sentence: what it shows and why it matters. */
  blurb: string;
}

/** The caption card and the control strip. */
export class GalleryShell {
  readonly app: HTMLDivElement;
  readonly card: HTMLDivElement;
  readonly strip: HTMLDivElement;
  readonly stats: HTMLDivElement;
  readonly status: HTMLDivElement;
  readonly note: HTMLDivElement;
  readonly error: HTMLDivElement;
  /** Extra content between the blurb and the stats line (demo 16's byte counter and timeline). */
  readonly extra: HTMLDivElement;
  private failed: ((message: string) => void) | null = null;

  constructor(options: ShellOptions) {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.append(style);
    document.title = `${options.number} · ${options.title} · @pmndrs/vdb`;

    this.app = document.createElement("div");
    this.app.id = "app";
    this.card = document.createElement("div");
    this.card.className = "g-card";
    this.card.dataset.testid = "caption";
    const h1 = document.createElement("h1");
    h1.innerHTML = `<span class="n">${options.number} ·</span> `;
    h1.append(options.title);
    const p = document.createElement("p");
    p.textContent = options.blurb;
    const back = document.createElement("a");
    back.href = "/";
    back.textContent = "← all demos";
    this.extra = document.createElement("div");
    this.status = div("g-status", "status");
    this.note = div("g-note", "note");
    this.stats = div("g-stats", "stats");
    this.stats.textContent = "starting WebGPU…";
    this.error = div("g-err", "error");
    this.card.append(h1, p, back, this.extra, this.status, this.note, this.stats, this.error);

    this.strip = document.createElement("div");
    this.strip.className = "g-strip";
    this.strip.dataset.testid = "controls";
    if (!PANEL) {
      this.card.classList.add("g-hidden");
      this.strip.classList.add("g-hidden");
    }
    document.body.append(this.app, this.card, this.strip);
  }

  /** Errors: shown in the card (even with `?panel=0`) and handed to `onFail`. */
  onFail(handler: (message: string) => void): void {
    this.failed = handler;
    const text = (err: unknown): string => (err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
    window.addEventListener("error", (e) => this.fail(text(e.error ?? e.message)));
    window.addEventListener("unhandledrejection", (e) => this.fail(text(e.reason)));
  }

  /** Flash the caption card (`'preview'` amber, `'full'` green) and show a short toast at the top centre. */
  flash(kind: "preview" | "full", toast: string): void {
    const cls = `g-flash-${kind}`;
    this.card.classList.remove("g-flash-preview", "g-flash-full");
    void this.card.offsetWidth; // restart the animation
    this.card.classList.add(cls);
    if (!PANEL) return;
    let el = document.querySelector<HTMLDivElement>(".g-toast");
    if (!el) {
      el = document.createElement("div");
      el.className = "g-toast";
      el.dataset.testid = "toast";
      document.body.append(el);
    }
    el.textContent = toast;
    el.style.color = kind === "full" ? "#7be3a1" : "#ffd479";
    el.classList.add("show");
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => el.classList.remove("show"), 2200);
  }
  private toastTimer: ReturnType<typeof setTimeout> | undefined;

  fail(message: string): void {
    this.error.textContent = message;
    this.card.classList.remove("g-hidden");
    this.status.textContent = "";
    this.failed?.(message);
  }

  button(key: string, label: string, onClick: () => void): Control<string> {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", onClick);
    return this.control(key, null, button, null, (v) => (button.textContent = v));
  }

  range(key: string, label: string, min: number, max: number, step: number, value: number, onInput: (v: number) => void, format: (v: number) => string = (v) => v.toFixed(step < 1 ? 2 : 0)): Control<number> {
    const input = document.createElement("input");
    input.type = "range";
    Object.assign(input, { min: String(min), max: String(max), step: String(step), value: String(value) });
    const out = document.createElement("output");
    out.textContent = format(value);
    input.addEventListener("input", () => {
      out.textContent = format(Number(input.value));
      onInput(Number(input.value));
    });
    return this.control(key, label, input, out, (v) => {
      input.value = String(v);
      out.textContent = format(v);
    });
  }

  select<T extends string>(key: string, label: string, options: readonly { value: T; label: string }[], value: T, onChange: (v: T) => void): Control<T> {
    const select = document.createElement("select");
    for (const o of options) {
      const el = document.createElement("option");
      el.value = o.value;
      el.textContent = o.label;
      select.append(el);
    }
    select.value = value;
    select.addEventListener("change", () => onChange(select.value as T));
    return this.control(key, label, select, null, (v) => (select.value = v));
  }

  checkbox(key: string, label: string, value: boolean, onChange: (v: boolean) => void): Control<boolean> {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = value;
    input.addEventListener("change", () => onChange(input.checked));
    return this.control(key, label, input, null, (v) => (input.checked = v), true);
  }

  private control<T>(key: string, label: string | null, input: Control<T>["input"], out: HTMLOutputElement | null, set: (v: T) => void, labelAfter = false): Control<T> {
    const el = document.createElement("div");
    el.className = "g-ctl";
    input.id = `ctl-${key}`;
    if (label) {
      const lab = document.createElement("label");
      lab.htmlFor = input.id;
      lab.textContent = label;
      if (labelAfter) el.append(input, lab);
      else el.append(lab, input);
    } else el.append(input);
    if (out) el.append(out);
    this.strip.append(el);
    return { el, input, set, show: (v) => el.classList.toggle("g-hidden", !v) };
  }
}

function div(className: string, testid: string): HTMLDivElement {
  const d = document.createElement("div");
  d.className = className;
  d.dataset.testid = testid;
  return d;
}

/** HEAD-probe a URL (gallery assets are optional; `?gallery=0` treats every gallery URL as missing). */
export async function probe(url: string): Promise<boolean> {
  if (!GALLERY_ASSETS && url.includes("/fixtures/gallery/")) return false;
  try {
    return (await fetch(url, { method: "HEAD" })).ok;
  } catch {
    return false;
  }
}

/** An UltraHDR JPEG sky (his examples' maps) as an equirect environment and background. */
export async function loadHdrSky(url: string): Promise<THREE.Texture> {
  const texture = await new UltraHDRLoader().loadAsync(url);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  return texture;
}

/**
 * A procedural equirect sky (the fallback when no HDR map is present): a vertical gradient from `zenith` through
 * `horizon` to `ground`, with an optional warm glow band at the horizon. Linear-light values, so it lights as well.
 */
export function proceduralSky(stops: { zenith: string; horizon: string; ground: string; glow?: string }): THREE.DataTexture {
  const W = 256;
  const H = 128;
  const data = new Float32Array(W * H * 4);
  const c = (s: string) => new THREE.Color(s); // sRGB hex -> linear working colour
  const zenith = c(stops.zenith);
  const horizon = c(stops.horizon);
  const ground = c(stops.ground);
  const glow = stops.glow ? c(stops.glow) : null;
  const col = new THREE.Color();
  for (let y = 0; y < H; y++) {
    // row 0 = top of the texture (flipY false below): v = 1 at the zenith
    const v = 1 - (y + 0.5) / H;
    const el = (v - 0.5) * Math.PI; // elevation
    if (el >= 0) col.copy(horizon).lerp(zenith, Math.pow(Math.sin(el), 0.55));
    else col.copy(horizon).lerp(ground, Math.min(1, Math.pow(-Math.sin(el), 0.35) * 1.2));
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      let r = col.r;
      let g = col.g;
      let b = col.b;
      if (glow && el > -0.05) {
        const az = (x / W) * Math.PI * 2;
        const band = Math.exp(-Math.pow(el / 0.12, 2)) * (0.55 + 0.45 * Math.max(0, Math.cos(az - 0.6)));
        r += glow.r * band;
        g += glow.g * band;
        b += glow.b * band;
      }
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = b;
      data[o + 3] = 1;
    }
  }
  const texture = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}

/**
 * The mean colour of an equirect map's horizon band (the rows within ±`band` of the middle), linear: a fog colour
 * that blends a ground plane into the sky. Reads a `DataTexture`'s float or half-float texels; null otherwise.
 */
export function horizonColor(texture: THREE.Texture, band = 0.03): THREE.Color | null {
  const image = texture.image as { data?: ArrayLike<number>; width?: number; height?: number } | undefined;
  const data = image?.data;
  const w = image?.width ?? 0;
  const h = image?.height ?? 0;
  if (!data || !w || !h || data.length < w * h * 4) return null;
  const half = data instanceof Uint16Array;
  const read = (i: number) => (half ? THREE.DataUtils.fromHalfFloat(data[i]!) : data[i]!);
  const mean = (y0: number, y1: number): THREE.Color => {
    const sum: [number, number, number] = [0, 0, 0];
    let n = 0;
    for (let y = Math.max(0, y0); y < Math.min(h, y1); y++) {
      for (let x = 0; x < w; x += 2) {
        const o = (y * w + x) * 4;
        sum[0] += read(o);
        sum[1] += read(o + 1);
        sum[2] += read(o + 2);
        n++;
      }
    }
    return new THREE.Color(sum[0]! / Math.max(1, n), sum[1]! / Math.max(1, n), sum[2]! / Math.max(1, n));
  };
  // The darker of the bands just above and just below the horizon (the map's ground side, whichever way up it is).
  const mid = Math.round(h / 2);
  const a = mean(mid - Math.ceil(h * band), mid);
  const b = mean(mid, mid + Math.ceil(h * band));
  const lum = (c: THREE.Color) => c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
  return lum(a) <= lum(b) ? a : b;
}

/** Gentle auto-rotation that stops for good on the first user interaction (`?still=1`: never starts). */
export function autoOrbit(orbit: OrbitControls, speed = 0.35): void {
  orbit.autoRotate = !STILL;
  orbit.autoRotateSpeed = speed;
  orbit.addEventListener("start", () => {
    orbit.autoRotate = false;
  });
}

/** A summary of the last read-back frame (readback presentation), for the e2e. */
export interface Snapshot {
  width: number;
  height: number;
  /** Pixels farther than 6 (sum of |d|) from `background` (sRGB bytes), when given. */
  nonBackground: number;
  mean: [number, number, number];
  /** 32 × 18 grey thumbnail (0–255), row-major, top row first. */
  thumb: number[];
  /** 96 × 54 grey thumbnail (0–255): finer detail, for changes a 32 × 18 average hides (a preview vs its full grid). */
  fine: number[];
}

const srgbByte = (v: number): number => {
  const c = v <= 0 ? 0 : v >= 1 ? 1 : v < 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return Math.round(c * 255);
};

/**
 * Draws a scene to the page: `direct` into the renderer's canvas, `readback` into an offscreen linear HalfFloat target
 * with depth (the froxel integrate pass reads the scene depth), read back (rows top first, padded to 256 bytes; Traps),
 * sRGB-encoded on the CPU (no tone mapping) and drawn on a 2D canvas.
 */
export class Presenter {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx2d: CanvasRenderingContext2D | null;
  private readonly rt = new THREE.RenderTarget(2, 2, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  private lastFrame: Uint8ClampedArray | null = null;
  width = 1;
  height = 1;

  constructor(
    private readonly renderer: THREE.WebGPURenderer,
    private readonly device: GPUDevice,
    canvas: HTMLCanvasElement | undefined,
    private readonly app: HTMLElement,
  ) {
    this.canvas = canvas ?? document.createElement("canvas");
    this.canvas.id = "view";
    app.appendChild(this.canvas);
    this.ctx2d = PRESENTATION === "readback" ? this.canvas.getContext("2d") : null;
  }

  /** Size to the page at a pixel ratio (direct only); updates the camera aspect. */
  resize(camera: THREE.PerspectiveCamera, pixelRatio: number): void {
    this.width = Math.max(1, this.app.clientWidth || window.innerWidth);
    this.height = Math.max(1, this.app.clientHeight || window.innerHeight);
    if (PRESENTATION === "direct") {
      this.renderer.setPixelRatio(pixelRatio);
      this.renderer.setSize(this.width, this.height, false);
    } else {
      this.canvas.width = this.width;
      this.canvas.height = this.height;
      this.rt.setSize(this.width, this.height);
    }
    camera.aspect = this.width / this.height;
    camera.updateProjectionMatrix();
  }

  async present(scene: THREE.Scene, camera: THREE.Camera): Promise<void> {
    const { renderer } = this;
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      await this.device.queue.onSubmittedWorkDone();
      return;
    }
    const vw = this.width;
    const vh = this.height;
    renderer.setRenderTarget(this.rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const raw = (await renderer.readRenderTargetPixelsAsync(this.rt, 0, 0, vw, vh)) as unknown as Uint16Array;
    const tight = vw * 4;
    const stride = vh > 1 ? (raw.length - tight) / (vh - 1) : tight;
    if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${raw.length} halves for ${vw}x${vh}`);
    const out = new Uint8ClampedArray(vw * vh * 4);
    for (let y = 0; y < vh; y++) {
      for (let x = 0; x < vw; x++) {
        const s = y * stride + x * 4;
        const d = (y * vw + x) * 4;
        out[d] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s]!));
        out[d + 1] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s + 1]!));
        out[d + 2] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s + 2]!));
        out[d + 3] = 255;
      }
    }
    this.ctx2d!.putImageData(new ImageData(out, vw, vh), 0, 0);
    this.lastFrame = out;
  }

  /** The last read-back frame (readback presentation), or null. */
  capture(background?: readonly [number, number, number]): Snapshot | null {
    const frame = this.lastFrame;
    if (!frame) return null;
    const w = this.width;
    const h = this.height;
    const TW = 32;
    const TH = 18;
    const thumb = new Array<number>(TW * TH).fill(0);
    const counts = new Array<number>(TW * TH).fill(0);
    const FW = 96;
    const FH = 54;
    const fine = new Array<number>(FW * FH).fill(0);
    const fineCounts = new Array<number>(FW * FH).fill(0);
    const mean: [number, number, number] = [0, 0, 0];
    let nonBackground = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const r = frame[i]!;
        const g = frame[i + 1]!;
        const b = frame[i + 2]!;
        if (background && Math.abs(r - background[0]) + Math.abs(g - background[1]) + Math.abs(b - background[2]) > 6) nonBackground++;
        mean[0] += r;
        mean[1] += g;
        mean[2] += b;
        const t = Math.min(TH - 1, Math.floor((y * TH) / h)) * TW + Math.min(TW - 1, Math.floor((x * TW) / w));
        thumb[t]! += (r + g + b) / 3;
        counts[t]!++;
        const f = Math.min(FH - 1, Math.floor((y * FH) / h)) * FW + Math.min(FW - 1, Math.floor((x * FW) / w));
        fine[f]! += (r + g + b) / 3;
        fineCounts[f]!++;
      }
    }
    const n = w * h;
    return { width: w, height: h, nonBackground, mean: mean.map((v) => v / n) as [number, number, number], thumb: thumb.map((v, i) => v / Math.max(1, counts[i]!)), fine: fine.map((v, i) => v / Math.max(1, fineCounts[i]!)) };
  }
}

/** The froxel quality presets of the gallery (demo 10's table; `low` is for the headless e2e). */
export type Quality = "low" | "medium" | "high";
export const QUALITY: Record<Quality, { width: number; height: number; depth: number; subSteps: number; shadowSteps: number; shadowResolution: number; emission: number; pixelRatio: number; label: string }> = {
  low: { width: 128, height: 128, depth: 16, subSteps: 8, shadowSteps: 8, shadowResolution: 32, emission: 8, pixelRatio: 1, label: "low" },
  medium: { width: 256, height: 256, depth: 24, subSteps: 16, shadowSteps: 16, shadowResolution: 64, emission: 16, pixelRatio: 1, label: "medium" },
  high: { width: 512, height: 512, depth: 24, subSteps: 32, shadowSteps: 32, shadowResolution: 64, emission: 32, pixelRatio: Math.min(2, window.devicePixelRatio || 1), label: "high" },
};
/** `?quality=`, else medium (low in readback, for the headless e2e). */
export function initialQuality(): Quality {
  const q = params.get("quality");
  return q === "low" || q === "medium" || q === "high" ? q : PRESENTATION === "readback" ? "low" : "medium";
}

export const MiB = (bytes: number): string => `${(bytes / 1048576).toFixed(bytes < 10 * 1048576 ? 2 : 1)} MiB`;
export const KiB = (bytes: number): string => `${(bytes / 1024).toFixed(1)} KiB`;
