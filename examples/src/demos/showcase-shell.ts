/**
 * The one page shell of gallery demos 11-16 (the shared design spec): a full-bleed canvas, a caption card top-left
 * (number · title, one sentence, "← all demos", a slot for demo widgets, a status line with a progress bar, a dim
 * stats line, a note for fallbacks, errors), a small control strip bottom-right (selects, ranges, checkboxes,
 * buttons), a flash and toast for state changes, gentle auto camera motion that stops on interaction, and the e2e
 * plumbing: the readback presentation (an offscreen HalfFloat target read back to a 2D canvas and tone mapped on
 * the CPU, as three's own output transform is skipped there), a frame summary, a coverage probe.
 *
 * Query (every gallery page): `?presentation=readback` (headless SwiftShader e2e, as demo 10), `?still=1` (no
 * camera motion), `?panel=0` (no caption or controls), `?gallery=0` (don't load the git-ignored `fixtures/gallery/`
 * assets: the procedural fallbacks), `?debug=1` (the demo may expose its objects on `window`).
 *
 * Demos 11, 12, 15 and 16 build their own renderer and use `mountShowcase` + `Presenter`; demos 13 and 14 take the
 * whole `createViewer` (renderer, camera, orbit, loop). Frame times are submit-to-done wall times of one frame on
 * this page: provisional (D25), not a benchmark.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { createDeviceRenderer } from "./device-renderer.js";

export const params = new URLSearchParams(location.search);
export type Presentation = "direct" | "readback";
export const PRESENTATION: Presentation = params.get("presentation") === "readback" ? "readback" : "direct";
export const STILL = params.get("still") === "1";
export const PANEL = params.get("panel") !== "0";
/** `?gallery=0`: behave as if the git-ignored `fixtures/gallery/` assets were missing. */
export const GALLERY_ASSETS = params.get("gallery") !== "0";
export const DEBUG = params.get("debug") === "1";
export const GALLERY_DIR = "/fixtures/gallery/";

export type Quality = "low" | "medium" | "high";
export function qualityParam(fallback: Quality): Quality {
  const q = params.get("quality");
  return q === "low" || q === "medium" || q === "high" ? q : fallback;
}
/** The froxel quality presets of demos 12 and 16 (demo 10's table; `low` is for the headless e2e). */
export const QUALITY: Record<Quality, { width: number; height: number; depth: number; subSteps: number; shadowSteps: number; shadowResolution: number; emission: number; pixelRatio: number; label: string }> = {
  low: { width: 128, height: 128, depth: 16, subSteps: 8, shadowSteps: 8, shadowResolution: 32, emission: 8, pixelRatio: 1, label: "low" },
  medium: { width: 256, height: 256, depth: 24, subSteps: 16, shadowSteps: 16, shadowResolution: 64, emission: 16, pixelRatio: 1, label: "medium" },
  high: { width: 512, height: 512, depth: 24, subSteps: 32, shadowSteps: 32, shadowResolution: 64, emission: 32, pixelRatio: Math.min(2, window.devicePixelRatio || 1), label: "high" },
};
/** `?quality=`, else medium (low in readback, for the headless e2e). */
export const initialQuality = (): Quality => qualityParam(PRESENTATION === "readback" ? "low" : "medium");

export interface ShowcaseState {
  ready: boolean;
  loading: boolean;
  error?: string;
  frames: number;
  frameMs?: number;
  adapter?: string;
  presentation: Presentation;
  /** Whether the page may load `fixtures/gallery/` assets (`?gallery=0` turns them off). */
  galleryAssets: boolean;
  uncaptured: string[];
  /** Demo-specific facts for the e2e spec. */
  info: Record<string, unknown>;
}

/** A summary of the last read-back frame (readback presentation), for the e2e. */
export interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  /** 32 × 18 grey thumbnail (0–255), row-major, top row first. */
  thumb: number[];
  /** 96 × 54 grey thumbnail (0–255): finer detail, for changes a 32 × 18 average hides (a preview vs its full grid). */
  fine: number[];
  /** max − min of the thumbnail: a flat frame (nothing drawn) has none. */
  spread: number;
  /** Pixels farther than 6 (sum of |d|) from `capture(background)` (sRGB bytes); 0 without one. */
  nonBackground: number;
}
export type ShowcaseSnapshot = Snapshot;
export type FrameSnapshot = Snapshot;

export interface ShowcaseApi {
  capture(): Snapshot | null;
  /** Readback only: renders once without the demo's content and counts the pixels that change (sum |d| > 12). */
  coverage(): Promise<number>;
}

const STYLE = /* css */ `
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; overflow: hidden; background: #0d0f14; color: #e8ecf6; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
#app { position: fixed; inset: 0; }
#app canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.sc-card {
  position: fixed; margin: 16px; padding: 12px 14px; color: #e8ecf6; font-size: 13px; line-height: 1.4;
  background: rgba(14, 17, 26, 0.66); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 10px;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
  transition: box-shadow 0.25s ease, border-color 0.25s ease;
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
#controls select, #controls button { font: inherit; font-size: 12.5px; color: #e8ecf6; background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(255, 255, 255, 0.14); border-radius: 6px; padding: 3px 6px; }
#controls select { width: 100%; }
#controls button { cursor: pointer; padding: 2px 8px; }
#controls button:hover, #controls select:hover { background: rgba(255, 255, 255, 0.14); }
#controls select option { background: #1a1e2a; }
#controls input[type="range"] { width: 100%; accent-color: #a9bcff; }
#controls input[type="checkbox"] { accent-color: #a9bcff; justify-self: start; }
/* strip "row" (demos 12 and 16): a horizontal strip of many small controls */
#controls.row { width: auto; display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; justify-content: flex-end; padding: 8px 10px; }
#controls.row .ctl, #controls.row .ctl.wide { display: flex; gap: 6px; margin: 0; white-space: nowrap; }
#controls.row .ctl input[type="range"] { width: 120px; }
#controls.row .ctl select { width: auto; }
#controls.row .ctl output { min-width: 3.2em; }
#controls.row .ctl.check input { order: -1; }
#controls .ctl[hidden] { display: none; }
.sc-card.sc-flash-preview { animation: sc-flash-preview 1.6s ease-out; }
.sc-card.sc-flash-full { animation: sc-flash-full 1.8s ease-out; }
@keyframes sc-flash-preview { 0% { box-shadow: 0 0 0 2px #ffd479, 0 0 28px rgba(255, 212, 121, 0.55); border-color: #ffd479; } 100% { box-shadow: 0 0 0 0 transparent; } }
@keyframes sc-flash-full { 0% { box-shadow: 0 0 0 2px #7be3a1, 0 0 32px rgba(123, 227, 161, 0.6); border-color: #7be3a1; } 100% { box-shadow: 0 0 0 0 transparent; } }
.sc-toast {
  position: fixed; left: 50%; top: 18px; transform: translateX(-50%); padding: 6px 14px; border-radius: 999px;
  background: rgba(10, 13, 22, 0.7); border: 1px solid rgba(255, 255, 255, 0.14); font-size: 13.5px; font-weight: 600;
  letter-spacing: 0.02em; opacity: 0; pointer-events: none; transition: opacity 0.5s ease;
}
.sc-toast.show { opacity: 1; transition: opacity 0.15s ease; }
@media (max-width: 640px) { #caption { width: auto; right: 0; } #controls:not(.row) { width: auto; left: 0; } }
`;

export interface SelectOption<T extends string> {
  value: T;
  label: string;
}

/** One control row in the strip; the input's id is `ctl-<key>` (the e2e drives it). `set` never fires the callback. */
export interface Control<T> {
  readonly el: HTMLDivElement;
  readonly input: HTMLInputElement | HTMLSelectElement | HTMLButtonElement;
  set(value: T): void;
  show(visible: boolean): void;
  enable(enabled: boolean): void;
}
export type SelectControl<T extends string> = Control<T> & { readonly input: HTMLSelectElement; options(next: readonly SelectOption<T>[]): void };

/** The control strip: labelled `<select>`, range, checkbox and button rows. */
export class Controls {
  constructor(private readonly root: HTMLElement) {}

  select<T extends string>(key: string, label: string, options: readonly SelectOption<T>[], value: T, onChange: (v: T) => void): SelectControl<T> {
    const select = document.createElement("select");
    const fill = (opts: readonly SelectOption<T>[]): void => {
      select.replaceChildren(
        ...opts.map((o) => {
          const el = document.createElement("option");
          el.value = o.value;
          el.textContent = o.label;
          return el;
        }),
      );
    };
    fill(options);
    select.value = value;
    select.addEventListener("change", () => onChange(select.value as T));
    return {
      ...this.row(key, label, select, null, true),
      input: select,
      set: (v: T) => {
        select.value = v;
      },
      options: (next) => {
        const current = select.value;
        fill(next);
        select.value = next.some((o) => o.value === current) ? current : (next[0]?.value ?? "");
      },
    };
  }

  range(key: string, label: string, min: number, max: number, step: number, value: number, onChange: (v: number) => void, format?: (v: number) => string): Control<number> {
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    const out = document.createElement("output");
    const fmt = format ?? ((v: number) => v.toFixed(step < 1 ? 2 : 0));
    out.textContent = fmt(value);
    input.addEventListener("input", () => {
      out.textContent = fmt(Number(input.value));
      onChange(Number(input.value));
    });
    return {
      ...this.row(key, label, input, out, false),
      set: (v: number) => {
        input.value = String(v);
        out.textContent = fmt(v);
      },
    };
  }

  checkbox(key: string, label: string, value: boolean, onChange: (v: boolean) => void): Control<boolean> {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = value;
    input.addEventListener("change", () => onChange(input.checked));
    const row = this.row(key, label, input, null, true);
    row.el.classList.add("check");
    return {
      ...row,
      set: (v: boolean) => {
        input.checked = v;
      },
    };
  }

  /** A button (no label); `set` changes its text. */
  button(key: string, label: string, onClick: () => void): Control<string> {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", onClick);
    return { ...this.row(key, null, button, null, true), set: (v: string) => void (button.textContent = v) };
  }

  private row(key: string, label: string | null, input: HTMLInputElement | HTMLSelectElement | HTMLButtonElement, out: HTMLOutputElement | null, wide: boolean): Omit<Control<never>, "set"> {
    const el = document.createElement("div");
    el.className = wide ? "ctl wide" : "ctl";
    input.id = `ctl-${key}`;
    if (label) {
      const lab = document.createElement("label");
      lab.htmlFor = input.id;
      lab.textContent = label;
      el.append(lab);
    }
    el.append(input);
    if (out) el.append(out);
    this.root.append(el);
    return {
      el,
      input,
      show: (visible: boolean) => {
        el.hidden = !visible;
      },
      enable: (enabled: boolean) => {
        input.disabled = !enabled;
        el.classList.toggle("disabled", !enabled);
      },
    };
  }
}

export const errorText = (err: unknown): string => (err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));

export interface ShowcaseUi {
  readonly app: HTMLDivElement;
  /** The caption card and the control strip. */
  readonly card: HTMLDivElement;
  readonly strip: HTMLDivElement;
  readonly controls: Controls;
  /** A slot in the caption between the back link and the status line, for demo widgets (demo 16's byte counter and timeline). */
  readonly extra: HTMLDivElement;
  /** The e2e state object (at `window[stateKey]` when the demo asked for one; the gallery demos keep their own). */
  readonly state: ShowcaseState;
  /** Set by `fail()`; render loops stop drawing. */
  readonly error: string | undefined;
  stats(text: string): void;
  /** A status line; `progress` 0–1 fills the bar, `null` makes it indeterminate, `undefined` hides the bar; `""` hides the line. */
  status(text: string, progress?: number | null): void;
  note(text: string): void;
  /** Shows the error in the card (even with `?panel=0`), stops the loops and calls the `onFail` handler. */
  fail(message: unknown): void;
  onFail(handler: (message: string) => void): void;
  /** Flashes the caption card (`'preview'` amber, `'full'` green) and shows a short toast at the top centre. */
  flash(kind: "preview" | "full", toast: string): void;
  select<T extends string>(key: string, label: string, options: readonly SelectOption<T>[], value: T, onChange: (v: T) => void): SelectControl<T>;
  range(key: string, label: string, min: number, max: number, step: number, value: number, onChange: (v: number) => void, format?: (v: number) => string): Control<number>;
  checkbox(key: string, label: string, value: boolean, onChange: (v: boolean) => void): Control<boolean>;
  button(key: string, label: string, onClick: () => void): Control<string>;
}

/** Builds the caption card and the control strip (and the e2e state at `window[stateKey]`), and hooks page errors. */
export function mountShowcase(cfg: { number: string; title: string; blurb: string; stateKey?: string; strip?: "column" | "row" }): ShowcaseUi {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.append(style);
  document.title = `${cfg.number} · ${cfg.title.replace(/<[^>]*>/g, "")} · @pmndrs/vdb`;
  const app = document.createElement("div");
  app.id = "app";
  const caption = document.createElement("div");
  caption.id = "caption";
  caption.className = "sc-card";
  caption.dataset.testid = "caption";
  caption.innerHTML =
    `<h1><span class="num">${cfg.number} ·</span> ${cfg.title}</h1><p>${cfg.blurb}</p><a href="/">&larr; all demos</a><div id="sc-extra"></div>` +
    `<div id="sc-status" data-testid="status" hidden><span></span><div id="sc-bar"><div></div></div></div>` +
    `<div id="sc-stats" data-testid="stats">starting WebGPU…</div><div id="sc-note" data-testid="note"></div><div id="sc-error" data-testid="error"></div>`;
  const strip = document.createElement("div");
  strip.id = "controls";
  strip.className = cfg.strip === "row" ? "sc-card row" : "sc-card";
  strip.dataset.testid = "controls";
  document.body.append(app, caption, strip);
  if (!PANEL) {
    caption.style.display = "none";
    strip.style.display = "none";
  }
  const state: ShowcaseState = { ready: false, loading: true, frames: 0, presentation: PRESENTATION, galleryAssets: GALLERY_ASSETS, uncaptured: [], info: {} };
  if (cfg.stateKey) (window as unknown as Record<string, unknown>)[cfg.stateKey] = state;

  const statsEl = caption.querySelector<HTMLDivElement>("#sc-stats")!;
  const statusEl = caption.querySelector<HTMLDivElement>("#sc-status")!;
  const statusText = statusEl.querySelector("span")!;
  const bar = caption.querySelector<HTMLDivElement>("#sc-bar")!;
  const noteEl = caption.querySelector<HTMLDivElement>("#sc-note")!;
  const errEl = caption.querySelector<HTMLDivElement>("#sc-error")!;
  const extra = caption.querySelector<HTMLDivElement>("#sc-extra")!;
  const controls = new Controls(strip);
  let failed: ((message: string) => void) | null = null;
  let toastTimer: ReturnType<typeof setTimeout> | undefined;
  const ui = {
    app,
    card: caption,
    strip,
    controls,
    extra,
    state,
    error: undefined as string | undefined,
    stats: (t: string) => {
      statsEl.textContent = t;
    },
    status: (t: string, progress?: number | null) => {
      statusEl.hidden = t === "";
      statusText.textContent = t;
      bar.hidden = progress === undefined;
      bar.classList.toggle("indeterminate", progress === null);
      (bar.firstElementChild as HTMLDivElement).style.width = progress == null ? "" : `${Math.round(Math.min(1, Math.max(0, progress)) * 100)}%`;
    },
    note: (t: string) => {
      noteEl.textContent = t;
    },
    fail: (message: unknown) => {
      const text = errorText(message);
      errEl.textContent = text;
      ui.error = text;
      state.error = text;
      state.loading = false;
      caption.style.display = "";
      ui.status("");
      failed?.(text);
    },
    onFail: (handler: (message: string) => void) => {
      failed = handler;
    },
    flash: (kind: "preview" | "full", toast: string) => {
      caption.classList.remove("sc-flash-preview", "sc-flash-full");
      void caption.offsetWidth; // restart the animation
      caption.classList.add(`sc-flash-${kind}`);
      if (!PANEL) return;
      let el = document.querySelector<HTMLDivElement>(".sc-toast");
      if (!el) {
        el = document.createElement("div");
        el.className = "sc-toast";
        el.dataset.testid = "toast";
        document.body.append(el);
      }
      el.textContent = toast;
      el.style.color = kind === "full" ? "#7be3a1" : "#ffd479";
      el.classList.add("show");
      clearTimeout(toastTimer);
      const shown = el;
      toastTimer = setTimeout(() => shown.classList.remove("show"), 2200);
    },
    select: controls.select.bind(controls),
    range: controls.range.bind(controls),
    checkbox: controls.checkbox.bind(controls),
    button: controls.button.bind(controls),
  } satisfies ShowcaseUi;
  window.addEventListener("error", (e) => ui.fail(e.error ?? e.message));
  window.addEventListener("unhandledrejection", (e) => ui.fail(e.reason));
  return ui;
}

/** True when a HEAD request for the URL succeeds (`?gallery=0` treats every `fixtures/gallery/` URL as missing). */
export async function probe(url: string): Promise<boolean> {
  if (!GALLERY_ASSETS && url.includes(GALLERY_DIR)) return false;
  try {
    const r = await fetch(url, { method: "HEAD" });
    return r.ok;
  } catch {
    return false;
  }
}

export async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
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
export const KiB = (bytes: number): string => `${(bytes / 1024).toFixed(1)} KiB`;
/** MiB above 1 MiB, KiB below (demos 11 and 15). */
export const formatBytes = (bytes: number): string =>
  bytes >= 1048576 ? `${(bytes / 1048576).toFixed(bytes < 10 * 1048576 ? 2 : 1)} MiB` : `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KiB`;

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

/**
 * The page's canvas and viewport in either presentation. `direct`: the renderer's own canvas. `readback`: three
 * renders into an offscreen linear HalfFloat target with depth (the froxel integrate pass reads the scene depth;
 * `present()` makes one, `target()` more), `readTarget()` turns one into sRGB bytes (rows top first, padding
 * dropped; ACES tone mapped with the renderer's exposure when `renderer.toneMapping` is ACESFilmic, since three
 * skips its output transform when rendering to a target) and `show()` draws bytes on the 2D canvas.
 */
export class Presenter {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx2d: CanvasRenderingContext2D | null;
  width = 1;
  height = 1;
  pixelRatio = 1;
  private last: Uint8ClampedArray<ArrayBuffer> | null = null;
  private readonly targets: THREE.RenderTarget[] = [];
  private main: THREE.RenderTarget | null = null;

  constructor(
    readonly app: HTMLElement,
    readonly renderer: THREE.WebGPURenderer,
    directCanvas?: HTMLCanvasElement,
    /** Needed by `present()` in the direct presentation (waits for the submit). */
    private readonly device?: GPUDevice,
  ) {
    this.canvas = directCanvas ?? document.createElement("canvas");
    this.canvas.id = "view";
    app.append(this.canvas);
    this.ctx2d = PRESENTATION === "readback" ? this.canvas.getContext("2d") : null;
  }

  /** A HalfFloat target with depth, sized with the viewport. */
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

  /** Fits the viewport to the page and updates the camera aspect. */
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

  /** Reads a HalfFloat target back as sRGB RGBA bytes. */
  async readTarget(rt: THREE.RenderTarget): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const w = rt.width;
    const h = rt.height;
    const raw = (await this.renderer.readRenderTargetPixelsAsync(rt, 0, 0, w, h)) as unknown as Uint16Array;
    const tight = w * 4;
    const stride = h > 1 ? (raw.length - tight) / (h - 1) : tight;
    if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${raw.length} halves for ${w}x${h}`);
    const mapped = this.renderer.toneMapping === THREE.ACESFilmicToneMapping;
    const exposure = this.renderer.toneMappingExposure;
    const tone = [0, 0, 0];
    const out = new Uint8ClampedArray(w * h * 4);
    const half = THREE.DataUtils.fromHalfFloat;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = y * stride + x * 4;
        const d = (y * w + x) * 4;
        tone[0] = half(raw[s]!);
        tone[1] = half(raw[s + 1]!);
        tone[2] = half(raw[s + 2]!);
        if (mapped) aces(tone[0], tone[1], tone[2], exposure, tone);
        out[d] = srgbByte(tone[0]!);
        out[d + 1] = srgbByte(tone[1]!);
        out[d + 2] = srgbByte(tone[2]!);
        out[d + 3] = 255;
      }
    }
    return out;
  }

  /** Draws read-back bytes (the viewport's size) on the 2D canvas and keeps them for `capture()`. */
  show(bytes: Uint8ClampedArray<ArrayBuffer>): void {
    this.ctx2d!.putImageData(new ImageData(bytes, this.width, this.height), 0, 0);
    this.last = bytes;
  }

  /**
   * Renders the scene: direct, into the canvas (resolves when the GPU is done); readback, into the page's own
   * target, read back, shown (unless `show: false`) and returned.
   */
  async present(scene: THREE.Scene, camera: THREE.Camera, options: { show?: boolean } = {}): Promise<Uint8ClampedArray<ArrayBuffer> | null> {
    const { renderer } = this;
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      await this.device?.queue.onSubmittedWorkDone();
      return null;
    }
    this.main ??= this.target();
    renderer.setRenderTarget(this.main);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const bytes = await this.readTarget(this.main);
    if (options.show !== false) this.show(bytes);
    return bytes;
  }

  /** A summary of the last shown read-back frame, or null. `background` (sRGB bytes) counts the pixels that differ. */
  capture(background?: readonly [number, number, number]): Snapshot | null {
    const f = this.last;
    if (!f) return null;
    const w = this.width;
    const h = this.height;
    const TW = 32;
    const TH = 18;
    const FW = 96;
    const FH = 54;
    const thumb = new Array<number>(TW * TH).fill(0);
    const counts = new Array<number>(TW * TH).fill(0);
    const fine = new Array<number>(FW * FH).fill(0);
    const fineCounts = new Array<number>(FW * FH).fill(0);
    const mean: [number, number, number] = [0, 0, 0];
    let nonBackground = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const r = f[i]!;
        const g = f[i + 1]!;
        const b = f[i + 2]!;
        if (background && Math.abs(r - background[0]) + Math.abs(g - background[1]) + Math.abs(b - background[2]) > 6) nonBackground++;
        mean[0] += r;
        mean[1] += g;
        mean[2] += b;
        const grey = (r + g + b) / 3;
        const t = Math.min(TH - 1, Math.floor((y * TH) / h)) * TW + Math.min(TW - 1, Math.floor((x * TW) / w));
        thumb[t]! += grey;
        counts[t]!++;
        const q = Math.min(FH - 1, Math.floor((y * FH) / h)) * FW + Math.min(FW - 1, Math.floor((x * FW) / w));
        fine[q]! += grey;
        fineCounts[q]!++;
      }
    }
    const n = w * h;
    const t = thumb.map((v, i) => v / Math.max(1, counts[i]!));
    return {
      width: w,
      height: h,
      mean: mean.map((v) => v / n) as [number, number, number],
      thumb: t,
      fine: fine.map((v, i) => v / Math.max(1, fineCounts[i]!)),
      spread: Math.max(...t) - Math.min(...t),
      nonBackground,
    };
  }
}

/** Slow auto-orbit that stops for good on the first user interaction (never starts with `?still=1` or in readback). Call `update()` every frame. */
export function autoOrbit(controls: { autoRotate: boolean; autoRotateSpeed: number; addEventListener(type: "start", cb: () => void): void }, speed = 0.35): void {
  controls.autoRotate = !STILL && PRESENTATION === "direct";
  controls.autoRotateSpeed = speed;
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
  });
}

export interface Viewer {
  renderer: THREE.WebGPURenderer;
  device: GPUDevice;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  orbit: OrbitControls;
  presenter: Presenter;
  /** The demo's own objects: hidden by the coverage probe. */
  content: THREE.Group;
  /** Auto camera motion (sway or orbit around `orbit.target`); off with `?still=1` and after any interaction. */
  motion: { enabled: boolean; rebase(): void };
  /** Starts the render loop; `tick` runs before each frame. */
  start(tick?: (now: number) => void, statsLine?: () => string): void;
  api: ShowcaseApi;
}

/**
 * The device-first renderer (`device-renderer.ts`), the presenter, an `OrbitControls` camera with gentle auto
 * motion, and the loop that keeps `ui.state` current (demos 13 and 14).
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
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = options.exposure ?? 1;

  const presenter = new Presenter(ui.app, renderer, directCanvas, device);
  const scene = new THREE.Scene();
  const content = new THREE.Group();
  scene.add(content);
  const camera = new THREE.PerspectiveCamera(options.fov ?? 40, 1, options.near ?? 0.05, options.far ?? 500);
  const orbit = new OrbitControls(camera, presenter.canvas);
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

  const pixelRatio = options.pixelRatio ?? Math.min(2, window.devicePixelRatio || 1);
  const setViewport = (): void => presenter.resize(pixelRatio, camera);
  setViewport();
  window.addEventListener("resize", setViewport);

  let busy = false;
  async function coverage(): Promise<number> {
    if (PRESENTATION !== "readback") throw new Error("coverage() needs ?presentation=readback");
    while (busy) await new Promise((r) => setTimeout(r, 5));
    busy = true;
    try {
      const withContent = await presenter.present(scene, camera, { show: false });
      content.visible = false;
      const without = await presenter.present(scene, camera, { show: false });
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

  const api: ShowcaseApi = { capture: () => presenter.capture(), coverage };
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
        ui.fail(err);
        return;
      }
      const t0 = performance.now();
      presenter
        .present(scene, camera)
        .then(() => {
          const ms = performance.now() - t0;
          frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
          ui.state.frames++;
          ui.state.frameMs = frameMs;
          if (now - lastStats > 400 || ui.state.frames <= 3) {
            lastStats = now;
            ui.stats(`${ui.state.adapter} · ${presenter.pixelWidth}×${presenter.pixelHeight} · ${frameMs.toFixed(1)} ms/frame (provisional)` + (statsLine ? `\n${statsLine()}` : ""));
          }
        })
        .catch((err: unknown) => ui.fail(err))
        .finally(() => {
          busy = false;
        });
    };
    requestAnimationFrame(loop);
  }

  return { renderer, device, scene, camera, orbit, presenter, content, motion, start, api };
}
