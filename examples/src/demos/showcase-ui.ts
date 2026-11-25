/**
 * The engine-free half of the gallery page shell (`showcase-shell.ts` re-exports all of it): query flags, the caption
 * card and control strip (`mountShowcase`, `Controls`), fetch helpers, byte formats, the ACES + sRGB curve of the
 * read-back presentation and the frame snapshot. No three.js here, so the raw-WebGPU demo 17 shares the gallery's look
 * (examples/test/demo-08-imports.test.ts keeps that demo's module graph free of three).
 */
import { fixtureUrl, HOME_URL, HOSTED } from "./assets.js";

export const params = new URLSearchParams(location.search);
export type Presentation = "direct" | "readback";
export const PRESENTATION: Presentation = params.get("presentation") === "readback" ? "readback" : "direct";
export const STILL = params.get("still") === "1";
export const PANEL = params.get("panel") !== "0";
/** `?gallery=0`: behave as if the git-ignored `fixtures/gallery/` assets were missing. */
export const GALLERY_ASSETS = params.get("gallery") !== "0";
export const DEBUG = params.get("debug") === "1";
export const GALLERY_DIR = fixtureUrl("gallery/");

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
    `<h1><span class="num">${cfg.number} ·</span> ${cfg.title}</h1><p>${cfg.blurb}</p><a href="${HOME_URL}">&larr; all demos</a> · <a href="${HOME_URL}attribution.html">credits</a> · <a href="https://pmndrs.github.io/vdb/">docs</a><div id="sc-extra"></div>` +
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
      // Notes name missing local fixtures (gallery assets, WDAS): not for visitors of the hosted site.
      if (HOSTED) return;
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

export function aces(r: number, g: number, b: number, exposure: number, out: number[]): void {
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
export const srgbByte = (v: number): number => {
  const c = v <= 0 ? 0 : v >= 1 ? 1 : v < 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return Math.round(c * 255);
};

/**
 * A summary of read-back sRGB bytes (`width × height` RGBA, top row first): the mean colour, two grey thumbnails and
 * the pixels farther than 6 (sum of |d|) from `background`. `Presenter.capture` and the raw-WebGPU demo 17 use it.
 */
export function snapshotOf(f: ArrayLike<number>, w: number, h: number, background?: readonly [number, number, number]): Snapshot {
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
