/**
 * Import this FIRST in every demo entry (`import "../webgpu-gate.js";`). Top-level await keeps the importing
 * module from evaluating when WebGPU is missing: the page shows a card with a link back instead of a blank
 * canvas. With WebGPU it costs one `requestAdapter()` call.
 */
import { HOME_URL } from "./assets.js";
import { checkWebGPU, NO_WEBGPU_MESSAGE } from "./webgpu-check.js";

const result = await checkWebGPU(typeof navigator === "undefined" ? undefined : (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu);

if (!result.ok) {
  const style = document.createElement("style");
  style.textContent = `
.no-webgpu { position: fixed; inset: 0; display: grid; place-items: center; background: #0d0f14; color: #e8ecf6; z-index: 2147483647;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.no-webgpu > div { max-width: 28rem; margin: 1rem; padding: 1.25rem 1.5rem; border: 1px solid rgba(255,255,255,0.14); border-radius: 12px; background: rgba(255,255,255,0.05); line-height: 1.5; }
.no-webgpu h1 { margin: 0 0 0.5rem; font-size: 1.1rem; }
.no-webgpu p { margin: 0 0 0.6rem; }
.no-webgpu small { opacity: 0.6; display: block; margin-bottom: 0.75rem; }
.no-webgpu a { color: #a9bcff; }`;
  document.head.append(style);
  const card = document.createElement("div");
  card.className = "no-webgpu";
  card.dataset.testid = "no-webgpu";
  card.dataset.reason = result.reason;
  const inner = document.createElement("div");
  const h = document.createElement("h1");
  h.textContent = "WebGPU is not available";
  const p = document.createElement("p");
  p.textContent = NO_WEBGPU_MESSAGE;
  const small = document.createElement("small");
  small.textContent = result.detail;
  const a = document.createElement("a");
  a.href = HOME_URL;
  a.textContent = "← all demos";
  inner.append(h, p, small, a);
  card.append(inner);
  document.body.append(card);
  // Never resolve: the demo that imported this gate must not run (no blank canvas, no cascade of errors).
  await new Promise<never>(() => {});
}

export {};
