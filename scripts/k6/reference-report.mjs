// K6.R / K6.R1: quality report of the froxel renderer (TSL path and core frames, `./three` `backend: 'core'`) and his
// K3 captures against the reference renderer (a brute-force volumetric path tracer, core/reference; design
// K6-core-renderer.md §7.4). REPORT ONLY: nothing gates on it, and no timing is measured.
//
//   node scripts/k6/reference-report.mjs                    # starts its own Vite server, Chrome on the real GPU (headed)
//   options: --groups=volume,frame,...  (default: all; see GROUP_DEFS) --w=160 --h=90 (reference size; frames render at
//            --scale× and are box-downsampled in linear light) --scale=8 (his 1280 × 720 at 160 × 90)
//            --target=0.03 (convergence stop: mean relative standard error of Y over the volume pixels)
//            --budget=300 (seconds: safety cap per reference render) --check=32 --spp=1 --maxBounces=512
//            --port=5461 --base=http://127.0.0.1:5173 (reuse a server) --headless (installed Chrome without a window;
//            check the adapter it reports) --swiftshader (bundled Chromium + SwiftShader: script debugging, never a result)
//
// Needs (local, git-ignored): fixtures/k3/bunny_cloud.wvdb, fixtures/k3-reference/75e7345/ (his captures + manifest),
// fixtures/wdas/wdas_cloud_quarter_fp8.nvdb, fixtures/vdb-samples/smoke.vdb. Groups whose fixtures are missing are skipped.
// Writes artifacts/k6-reference/<group>/*.png (reference, frames, his captures, heat maps) and report.{json,md}.
//
// Per comparison (a frame vs the reference with multiple scattering, and vs single scattering):
//   PSNR (sRGB 8-bit, whole frame), pixels > 8/255, SSIM (K6.R1: luma, 11 × 11 Gaussian; whole frame and the volume's
//   rectangle), and over the volume pixels (reference alpha > 0.02): RMSE of the linear composite, median and mean
//   relative luminance error |Y_A − Y_ref| / Y_ref (Y_ref > 1e-3), and the luminance bias Σ Y_A / Σ Y_ref − 1, Y the
//   luminance of the volume's radiance recovered from the composite by inverting his sRGB blend at the reference's
//   coverage (K6.R1; K6.R subtracted the background linearly, which goes negative where a volume is darker than it).
//   Silhouette band (K6.R1, V80): pixels within one pixel of the reference's alpha = 0.02 contour (both sides): the
//   composite's error Σ|A − R| / Σ|R − background| there (lower is better) and its RMSE. Overlap (K6.R1, V79): pixels
//   where two volumes each reach optical depth 0.1: bias and median error there. The reference's own noise is listed:
//   errors below it are not resolved.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { decodePng, encodePng } from "../k3/png.mjs";
import { diffHeatmap } from "../k3/metrics.mjs";
import { silhouetteMask, ssim } from "./metrics.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.join(ROOT, "artifacts", "k6-reference");
const CAPTURES = path.join(ROOT, "fixtures", "k3-reference", "75e7345");
const arg = (name, fallback) => {
  const hit = process.argv.find((a) => a === `--${name}` || a.startsWith(`--${name}=`));
  if (!hit) return fallback;
  return hit.includes("=") ? hit.slice(hit.indexOf("=") + 1) : true;
};
const SWIFT = arg("swiftshader", false) === true;
const W = Number(arg("w", SWIFT ? 32 : 160));
const H = Number(arg("h", SWIFT ? 18 : 90));
const SCALE = Number(arg("scale", SWIFT ? 4 : 8));
const TARGET = Number(arg("target", SWIFT ? 0.5 : 0.03));
const BUDGET = Number(arg("budget", SWIFT ? 10 : 300));
const CHECK = Number(arg("check", 32));
const SPP = Number(arg("spp", 1));
const MAX_BOUNCES = Number(arg("maxBounces", 512));
const PORT = Number(arg("port", 5461));
let BASE = arg("base", "");

const need = {
  bunny: path.join(ROOT, "fixtures", "k3", "bunny_cloud.wvdb"),
  captures: path.join(CAPTURES, "manifest.json"),
  wdas: path.join(ROOT, "fixtures", "wdas", "wdas_cloud_quarter_fp8.nvdb"),
  smoke: path.join(ROOT, "fixtures", "vdb-samples", "smoke.vdb"),
};
if (!existsSync(need.bunny)) {
  console.error("needs fixtures/k3/bunny_cloud.wvdb (copy it from the main checkout)");
  process.exit(2);
}
const MANIFEST = existsSync(need.captures) ? JSON.parse(readFileSync(need.captures, "utf8")) : null;

// --- the groups ------------------------------------------------------------------------------------------------
// 'his' groups: a K3 scenario of his capture manifest (his bunny, camera, box, sun), frames from k3-frame, his captures.
// 'scene' groups (K6.R1): a scene of examples/src/harness/k6-scenes.ts, frames from k6-scene, no captures.
// Rows: { label, page, query, lighting, capture?, vsSingle? } or { label, slot: true } (a row with no frame yet).
// 'volume' lighting uses the volume's phase (k3-frame: his frame.js dual-lobe 0.8 / −0.3 / 0.2); 'frame' lighting the
// renderer's settings phase (read from the frame's `settings`; k3-frame sets 0.8 / −0.3 / 0.2 too).
const VOLUME_PHASE = [0.8, -0.3, 0.2];
// K6.3: the core's own MS modes ('multi-scale' both edge modes, the A/B for V75; the core's two-sided / octaves / neural)
const coreMsRows = (make, base) => [
  make("core ms two-sided", `${base}&ms=two-sided&backend=core`, { lighting: "frame" }),
  make("core ms octaves", `${base}&ms=octaves&backend=core`, { lighting: "frame" }),
  make("core ms neural (generic MLP, his weights)", `${base}&ms=neural&backend=core`, { lighting: "frame" }),
  make("core ms multi-scale (edge zero)", `${base}&ms=multi-scale&msEdge=zero&backend=core`, { lighting: "frame" }),
  make("core ms multi-scale (edge clamp, his)", `${base}&ms=multi-scale&msEdge=clamp&backend=core`, { lighting: "frame" }),
];
const k3 = (label, query, extra = {}) => ({ label, page: "k3-frame", query, ...extra });
const k6 = (label, query, extra = {}) => ({ label, page: "k6-scene", query, ...extra });
// The renderer rows of a scene group: both lightings, TSL and core, every multiple-scattering mode each path has.
const sceneRows = (scene) => [
  k6("volume TSL ms off", `scene=${scene}&lighting=volume&ms=0`, { lighting: "volume", vsSingle: true }),
  k6("volume core ms off", `scene=${scene}&lighting=volume&ms=0&backend=core`, { lighting: "volume", vsSingle: true }),
  k6("volume TSL ms on", `scene=${scene}&lighting=volume&ms=1`, { lighting: "volume" }),
  k6("volume core ms on", `scene=${scene}&lighting=volume&ms=1&backend=core`, { lighting: "volume" }),
  k6("frame TSL ms off", `scene=${scene}&lighting=frame&ms=0`, { lighting: "frame", vsSingle: true }),
  k6("frame core ms off", `scene=${scene}&lighting=frame&ms=0&backend=core`, { lighting: "frame", vsSingle: true }),
  k6("frame TSL ms diffusion", `scene=${scene}&lighting=frame&ms=1`, { lighting: "frame" }),
  k6("frame core ms diffusion", `scene=${scene}&lighting=frame&ms=1&backend=core`, { lighting: "frame" }),
  k6("frame TSL ms two-sided", `scene=${scene}&lighting=frame&ms=two-sided`, { lighting: "frame" }),
  k6("frame TSL ms octaves", `scene=${scene}&lighting=frame&ms=octaves`, { lighting: "frame" }),
  k6("frame TSL ms neural", `scene=${scene}&lighting=frame&ms=neural`, { lighting: "frame" }),
  ...coreMsRows((label, query, extra) => k6(`frame ${label}`, query, extra), `scene=${scene}&lighting=frame`),
];
const GROUP_DEFS = {
  volume: {
    kind: "his",
    title: "'volume' lighting, his sun only (Step 1 isolated)",
    scenario: "step1-volume-dir-ms-on-isolated",
    lighting: "volume",
    rows: [
      k3("TSL ms off", "scene=step1-volume-dir-ms-off-isolated", { lighting: "volume", capture: "step1-volume-dir-ms-off-isolated", vsSingle: true }),
      k3("core ms off", "scene=step1-volume-dir-ms-off-isolated&backend=core", { lighting: "volume", vsSingle: true }),
      k3("TSL ms on (diffusion)", "scene=step1-volume-dir-ms-on-isolated", { lighting: "volume", capture: "step1-volume-dir-ms-on-isolated" }),
      k3("core ms on (diffusion)", "scene=step1-volume-dir-ms-on-isolated&backend=core", { lighting: "volume" }),
    ],
  },
  frame: {
    kind: "his",
    title: "'frame' lighting, his sun only (Step 3 default isolated, every multipleScattering mode)",
    scenario: "step3-default-dir-isolated",
    lighting: "frame",
    rows: [
      k3("TSL ms off", "scene=step3-default-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
      k3("core ms off", "scene=step3-default-dir-isolated&ms=0&backend=core", { lighting: "frame", vsSingle: true }),
      k3("TSL ms diffusion", "scene=step3-default-dir-isolated&ms=1", { lighting: "frame", capture: "step3-default-dir-isolated" }),
      k3("core ms diffusion", "scene=step3-default-dir-isolated&ms=1&backend=core", { lighting: "frame" }),
      k3("TSL ms two-sided", "scene=step3-default-dir-isolated&ms=two-sided", { lighting: "frame", capture: "step4-multiple-scattering-two-sided-dir-isolated" }),
      k3("TSL ms octaves", "scene=step3-default-dir-isolated&ms=octaves", { lighting: "frame", capture: "step4-multiple-scattering-octaves-dir-isolated" }),
      k3("TSL ms neural (his v0.26 default)", "scene=step3-default-dir-isolated&ms=neural", { lighting: "frame", capture: "step4-default-neural-dir-isolated" }),
      ...coreMsRows(k3, "scene=step3-default-dir-isolated"),
    ],
  },
  two: {
    kind: "his",
    title: "two overlapping bunnies, 'frame' lighting (density sums: order-independent in both paths, V90)",
    scenario: "step3-two-bunnies-dir-isolated",
    lighting: "frame",
    rows: [
      k3("TSL diffusion", "scene=step3-two-bunnies-dir-isolated", { lighting: "frame", capture: "step3-two-bunnies-dir-isolated" }),
      k3("core diffusion", "scene=step3-two-bunnies-dir-isolated&backend=core", { lighting: "frame" }),
      k3("TSL ms off", "scene=step3-two-bunnies-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
      k3("core ms off", "scene=step3-two-bunnies-dir-isolated&ms=0&backend=core", { lighting: "frame", vsSingle: true }),
    ],
  },
  mix: {
    kind: "his",
    title: "V79: two overlapping bunnies, 'volume' lighting: core froxelMix 'mixture' vs 'over' (and the TSL path's draw order)",
    scenario: "step3-two-bunnies-dir-isolated",
    lighting: "volume",
    rows: [
      k3("TSL ms off (draw order)", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=0", { lighting: "volume", vsSingle: true }),
      k3("core ms off, mixture", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=0&backend=core&froxelMix=mixture", { lighting: "volume", vsSingle: true }),
      k3("core ms off, over", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=0&backend=core&froxelMix=over", { lighting: "volume", vsSingle: true }),
      k3("TSL ms on (draw order)", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=1", { lighting: "volume" }),
      k3("core ms on, mixture", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=1&backend=core&froxelMix=mixture", { lighting: "volume" }),
      k3("core ms on, over", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=1&backend=core&froxelMix=over", { lighting: "volume" }),
    ],
  },
  silhouette: {
    kind: "his",
    title: "V80: silhouettes, 'volume' lighting ms off, reference single scattering at 4× the pixels (frames 2× down)",
    scenario: "step1-volume-dir-ms-off-isolated",
    lighting: "volume",
    size: [640, 360],
    scale: 2,
    singleOnly: true,
    rows: [
      k3("TSL ms off (per-slice)", "scene=step1-volume-dir-ms-off-isolated", { lighting: "volume", capture: "step1-volume-dir-ms-off-isolated", vsSingle: true }),
      k3("core ms off (prefix)", "scene=step1-volume-dir-ms-off-isolated&backend=core", { lighting: "volume", vsSingle: true }),
    ],
  },
  mie: {
    kind: "his",
    title: "Lorenz-Mie 'frame' phase (his table), sun only",
    scenario: "step4-frame-phase-lorenz-mie-dir-isolated",
    lighting: "frame",
    mie: "mono",
    rows: [
      k3("TSL ms neural", "scene=step4-frame-phase-lorenz-mie-dir-isolated", { lighting: "frame", capture: "step4-frame-phase-lorenz-mie-dir-isolated" }),
      k3("TSL ms diffusion", "scene=step4-frame-phase-lorenz-mie-dir-isolated&ms=1", { lighting: "frame" }),
      k3("TSL ms off", "scene=step4-frame-phase-lorenz-mie-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
    ],
  },
  "mie-rgb": {
    kind: "his",
    title: "Mie RGB 'frame' phase (his tables), sun only",
    scenario: "step4-frame-phase-lorenz-mie-rgb-dir-isolated",
    lighting: "frame",
    mie: "rgb",
    rows: [
      k3("TSL ms neural", "scene=step4-frame-phase-lorenz-mie-rgb-dir-isolated", { lighting: "frame", capture: "step4-frame-phase-lorenz-mie-rgb-dir-isolated" }),
      k3("TSL ms diffusion", "scene=step4-frame-phase-lorenz-mie-rgb-dir-isolated&ms=1", { lighting: "frame" }),
      k3("TSL ms off", "scene=step4-frame-phase-lorenz-mie-rgb-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
    ],
  },
  env: {
    kind: "his",
    title: "V96: his sky-midday environment only, 'frame' lighting: 'outward' irradiance (core SH L2 vs TSL PMREM roughness 1 × π) and 'directional'",
    scenario: "step3-environment-sky-midday-env-isolated",
    lighting: "frame",
    env: "/fixtures/k3/sky-midday-HDR.jpg",
    rows: [
      k3("TSL outward, ms off", "scene=step3-environment-sky-midday-env-isolated&envMode=outward&ms=0", { lighting: "frame", vsSingle: true }),
      k3("core outward, ms off", "scene=step3-environment-sky-midday-env-isolated&envMode=outward&ms=0&backend=core", { lighting: "frame", vsSingle: true }),
      k3("TSL outward, ms diffusion", "scene=step3-environment-sky-midday-env-isolated&envMode=outward&ms=1", { lighting: "frame" }),
      k3("core outward, ms diffusion", "scene=step3-environment-sky-midday-env-isolated&envMode=outward&ms=1&backend=core", { lighting: "frame" }),
      k3("TSL directional, ms diffusion", "scene=step3-environment-sky-midday-env-isolated&envMode=directional&ms=1", { lighting: "frame", capture: "step3-environment-sky-midday-env-isolated" }),
      k3("core directional, ms diffusion", "scene=step3-environment-sky-midday-env-isolated&envMode=directional&ms=1&backend=core", { lighting: "frame" }),
    ],
  },
  ...Object.fromEntries(
    [
      ["wdas", "WDAS cloud (sparse, Fp8), sun", "wdas"],
      ["wdas-ambient", "WDAS cloud, sun + ambient", "wdas"],
      ["smoke", "openvdb.org smoke (sparse), sun", "smoke"],
      ["smoke-ambient", "smoke, sun + ambient", "smoke"],
      ["bunny-absorbing", "absorbing bunny (albedo 0.5), sun", "bunny"],
      ["bunny-absorbing-ambient", "absorbing bunny, sun + ambient", "bunny"],
      ["bunny-thin", "thin bunny (extinction 5), sun", "bunny"],
      ["bunny-thin-ambient", "thin bunny, sun + ambient", "bunny"],
      ["bunny-point", "bunny, point light (falloff per lighting mode)", "bunny", true],
    ].map(([scene, title, asset, falloff]) => [scene, { kind: "scene", title, scene, asset, falloff: !!falloff, rows: sceneRows(scene) }]),
  ),
};
const GROUPS = String(arg("groups", Object.keys(GROUP_DEFS).join(","))).split(",");

// --- colour ------------------------------------------------------------------------------------------------------
const eotf = (x) => (x < 0.04045 ? x * 0.0773993808 : Math.pow(x * 0.9478672986 + 0.0521327014, 2.4));
const oetf = (x) => (x < 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 0.41666) - 0.055);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const BG_SRGB = [0x59 / 255, 0x8e / 255, 0xff / 255];
const BG = BG_SRGB.map(eotf);
const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** His srgbBlending composite (tools vdbFrameSrgbBlend, clamp on): linear out. */
function srgbBlend(bg, radiance, alpha) {
  return [0, 1, 2].map((c) => {
    const b = clamp01(oetf(bg[c]));
    const v = clamp01(oetf(radiance[c] / alpha));
    return eotf(b + (v - b) * alpha);
  });
}

/** Linear float RGB image → RGBA8 sRGB. */
function toRgba8(img) {
  const data = new Uint8Array(img.width * img.height * 4);
  for (let i = 0; i < img.width * img.height; i++) {
    for (let c = 0; c < 3; c++) data[i * 4 + c] = Math.round(clamp01(oetf(Math.max(img.rgb[i * 3 + c], 0))) * 255);
    data[i * 4 + 3] = 255;
  }
  return { width: img.width, height: img.height, data };
}

/** RGBA8 sRGB (w × h) → linear RGB, box-downsampled by `k` in linear light. */
function fromRgba8(rgba, w, h, k) {
  const ow = Math.floor(w / k);
  const oh = Math.floor(h / k);
  const rgb = new Float64Array(ow * oh * 3);
  for (let y = 0; y < oh * k; y++) for (let x = 0; x < ow * k; x++) {
    const o = (Math.floor(y / k) * ow + Math.floor(x / k)) * 3;
    for (let c = 0; c < 3; c++) rgb[o + c] += eotf(rgba[(y * w + x) * 4 + c] / 255) / (k * k);
  }
  return { width: ow, height: oh, rgb };
}

/** The reference's mean (L, T) → composited linear RGB (his sRGB blend over the background), alpha, noise, masks. */
function referenceImage(r) {
  const mean = new Float32Array(Buffer.from(r.mean, "base64").buffer.slice(0));
  const n = r.width * r.height;
  const rgb = new Float64Array(n * 3);
  const alpha = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const L = [mean[i * 4], mean[i * 4 + 1], mean[i * 4 + 2]];
    const a = clamp01(1 - mean[i * 4 + 3]);
    alpha[i] = a;
    rgb.set(a > 1e-6 ? srgbBlend(BG, L, a) : BG, i * 3);
  }
  const overlap = r.overlap ? new Uint8Array(Buffer.from(r.overlap, "base64")) : null;
  const twoOrMore = overlap ? overlap.map((b) => (b & (b - 1) ? 1 : 0)) : null;
  return { width: r.width, height: r.height, rgb, alpha, noise: r.noise?.mean ?? 0, noiseP90: r.noise?.p90 ?? 0, volumePixels: r.noise?.pixels ?? 0, silhouette: silhouetteMask(alpha, r.width, r.height), overlap: twoOrMore };
}

/** His sRGB blend inverted: the radiance L (linear, ≤ α per channel) whose composite over the background at α is pixel i. */
function unblend(rgb, i, alpha) {
  return [0, 1, 2].map((c) => {
    const b = clamp01(oetf(BG[c]));
    const v = clamp01((clamp01(oetf(Math.max(rgb[i * 3 + c], 0))) - b * (1 - alpha)) / alpha);
    return alpha * eotf(v);
  });
}

/** Pixel rectangle of the volume (reference alpha > 0.02) grown by `pad`, as a mask. */
function rectMask(ref, pad = 6) {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (let y = 0; y < ref.height; y++) for (let x = 0; x < ref.width; x++) {
    if (ref.alpha[y * ref.width + x] > 0.02) {
      x0 = Math.min(x0, x);
      y0 = Math.min(y0, y);
      x1 = Math.max(x1, x);
      y1 = Math.max(y1, y);
    }
  }
  const m = new Uint8Array(ref.width * ref.height);
  for (let y = Math.max(0, y0 - pad); y <= Math.min(ref.height - 1, y1 + pad); y++) for (let x = Math.max(0, x0 - pad); x <= Math.min(ref.width - 1, x1 + pad); x++) m[y * ref.width + x] = 1;
  return m;
}

/** Luminance errors of `a` vs `ref` over the pixels where `inMask(i)`: bias, median and mean relative, RMSE linear. */
function regionStats(a, ref, inMask) {
  let sq = 0;
  let count = 0;
  let absErr = 0;
  let contrast = 0;
  let yA = 0;
  let yR = 0;
  const rels = [];
  for (let i = 0; i < ref.width * ref.height; i++) {
    if (!inMask(i)) continue;
    count++;
    for (let c = 0; c < 3; c++) {
      sq += (a.rgb[i * 3 + c] - ref.rgb[i * 3 + c]) ** 2;
      absErr += Math.abs(a.rgb[i * 3 + c] - ref.rgb[i * 3 + c]);
      contrast += Math.abs(ref.rgb[i * 3 + c] - BG[c]);
    }
    if (!(ref.alpha[i] > 0.02)) continue;
    // Luminance of the volume's radiance L: his sRGB blend inverted at the reference's coverage (K6.R1; the linear
    // "composite minus (1 − α)·background" of K6.R goes negative for volumes darker than the background).
    const ya = lum(...unblend(a.rgb, i, ref.alpha[i]));
    const yr = lum(...unblend(ref.rgb, i, ref.alpha[i]));
    yA += ya;
    yR += yr;
    if (yr > 1e-3) rels.push(Math.abs(ya - yr) / yr);
  }
  rels.sort((x, y) => x - y);
  if (!count) return { pixels: 0, rmseLinear: null, medianRelLum: null, meanRelLum: null, lumBias: null, relError: null };
  return {
    pixels: count,
    // composite error relative to the reference's contrast against the background (defined on both sides of an edge)
    relError: contrast > 0 ? absErr / contrast : null,
    rmseLinear: count ? Math.sqrt(sq / (count * 3)) : 0,
    medianRelLum: rels.length ? rels[Math.floor(rels.length / 2)] : 0,
    meanRelLum: rels.length ? rels.reduce((s, v) => s + v, 0) / rels.length : 0,
    lumBias: yR > 0 ? yA / yR - 1 : null,
  };
}

function compare(a, ref) {
  if (a.width !== ref.width || a.height !== ref.height) throw new Error(`size ${a.width}x${a.height} vs ${ref.width}x${ref.height}`);
  const n = ref.width * ref.height;
  const A8 = toRgba8(a);
  const R8 = toRgba8(ref);
  let mse8 = 0;
  let over8 = 0;
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let c = 0; c < 3; c++) {
      const d = A8.data[i * 4 + c] - R8.data[i * 4 + c];
      mse8 += d * d;
      m = Math.max(m, Math.abs(d));
    }
    if (m > 8) over8++;
  }
  mse8 /= n * 3;
  const vol = regionStats(a, ref, (i) => ref.alpha[i] > 0.02);
  const sil = regionStats(a, ref, (i) => ref.silhouette[i] === 1);
  const ovl = ref.overlap ? regionStats(a, ref, (i) => ref.overlap[i] === 1) : null;
  return {
    psnr: mse8 === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse8),
    over8: over8 / n,
    ssim: ssim(A8, R8).ssim,
    ssimRect: ssim(A8, R8, rectMask(ref)).ssim,
    volumePixels: vol.pixels,
    rmseLinear: vol.rmseLinear,
    meanRelLum: vol.meanRelLum,
    medianRelLum: vol.medianRelLum,
    lumBias: vol.lumBias,
    silhouette: { pixels: sil.pixels, relError: sil.relError, rmseLinear: sil.rmseLinear, lumBias: sil.lumBias },
    overlap: ovl && { pixels: ovl.pixels, lumBias: ovl.lumBias, medianRelLum: ovl.medianRelLum, rmseLinear: ovl.rmseLinear },
  };
}

// --- server and browser ------------------------------------------------------------------------------------------
async function startServer() {
  if (BASE) return null;
  BASE = `http://127.0.0.1:${PORT}`;
  const child = spawn("pnpm", ["--filter", "@webvdb/examples", "dev", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"], { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(BASE)).ok) return child;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  child.kill();
  throw new Error(`Vite did not start on ${BASE}`);
}

async function runPage(browser, page, query, state) {
  const p = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  await p.goto(`${BASE}/src/harness/${page}.html?${query}`);
  await p.waitForFunction((s) => window[s]?.done === true, state, { timeout: 3_600_000, polling: 1000 });
  const r = await p.evaluate((s) => window[s], state);
  await p.close();
  if (r.error) throw new Error(`${page}?${query}: ${String(r.error).split("\n")[0]}`);
  return { ...r, pageErrors: errors };
}

const STATE = { "k3-frame": "__VDB_K3_FRAME__", "k6-scene": "__VDB_K6_SCENE__" };
const f = (x, d = 2) => (x === null || x === undefined ? "—" : Number.isFinite(x) ? x.toFixed(d) : String(x));
const pct = (x, d = 1) => (x === null || x === undefined ? "—" : `${f(x * 100, d)} %`);

function groupUsable(g) {
  if (g.kind === "his") return MANIFEST && MANIFEST.scenarios.some((s) => s.name === g.scenario) ? null : `scenario ${g.scenario} not in the capture manifest`;
  if (g.asset === "wdas" && !existsSync(need.wdas)) return "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb missing";
  if (g.asset === "smoke" && !existsSync(need.smoke)) return "fixtures/vdb-samples/smoke.vdb missing";
  return null;
}

const server = await startServer();
const browser = await chromium.launch({
  ...(SWIFT ? {} : { channel: "chrome" }),
  // Headed: macOS Chrome picks the hardware adapter only with a window server session (scripts/k3/time-hardware.mjs).
  headless: SWIFT || arg("headless", false) === true,
  args: ["--enable-unsafe-webgpu", ...(SWIFT ? ["--enable-features=Vulkan", "--enable-unsafe-swiftshader"] : [])],
});
const report = { at: new Date().toISOString(), browser: browser.version(), size: [W, H], scale: SCALE, target: TARGET, budgetSeconds: BUDGET, maxBounces: MAX_BOUNCES, groups: [] };
mkdirSync(OUT, { recursive: true });
try {
  for (const key of GROUPS) {
    const g = GROUP_DEFS[key];
    if (!g) throw new Error(`unknown group ${key}`);
    const skip = groupUsable(g);
    if (skip) {
      console.log(`== ${key}: skipped (${skip})`);
      continue;
    }
    const [gw, gh] = g.size ?? [W, H];
    const scale = g.scale ?? SCALE;
    const dir = path.join(OUT, key);
    mkdirSync(dir, { recursive: true });
    console.log(`== ${key}: ${g.title}`);
    // Frames first ('frame' lighting: their settings give the his-group reference its phase).
    const frames = [];
    for (const t of g.rows) {
      if (t.slot) {
        frames.push({ ...t });
        continue;
      }
      process.stdout.write(`  ${t.label} … `);
      const r = await runPage(browser, t.page, `${t.query}&w=${gw * scale}&h=${gh * scale}`, STATE[t.page]);
      const img = fromRgba8(new Uint8Array(Buffer.from(r.rgba, "base64")), r.width, r.height, scale);
      frames.push({ ...t, img, settings: r.settings, adapter: r.adapter, backend: r.backend, froxelMix: r.froxelMix });
      console.log(`${r.width}x${r.height} ${r.backend ?? ""} (${r.adapter})`);
    }
    // References: per lighting mapping when the scene has point lights, else one; multiple and single scattering.
    const lightings = g.kind === "scene" && g.falloff ? ["volume", "frame"] : [g.lighting ?? "frame"];
    const refs = {};
    for (const lighting of lightings) {
      let query;
      if (g.kind === "his") {
        const s = frames.find((t) => t.settings && t.lighting === "frame")?.settings;
        const phase = g.lighting === "volume" ? VOLUME_PHASE : s?.phase === "dual-lobe" ? [s.phaseG, s.phaseBackG, s.phaseBackWeight] : [s?.phaseG ?? 0.8, 0, 0];
        if (g.lighting === "frame" && s && !["henyey-greenstein", "dual-lobe", "isotropic", "lorenz-mie", "lorenz-mie-rgb"].includes(s.phase)) console.warn(`  phase ${s.phase}: the reference uses HG g = ${phase[0]}`);
        query = `case=report&scenario=${g.scenario}&lighting=${lighting}&phase=${phase.join(",")}${g.mie ? `&mie=${g.mie}` : ""}${g.env ? `&env=${g.env}` : ""}`;
      } else query = `case=scene&scene=${g.scene}&lighting=${lighting}`;
      const common = `&w=${gw}&h=${gh}&spp=${SPP}&target=${TARGET}&check=${CHECK}&budget=${BUDGET}`;
      const entry = {};
      for (const [kind, bounces] of g.singleOnly ? [["single", 1]] : [["full", MAX_BOUNCES], ["single", 1]]) {
        process.stdout.write(`  reference ${lighting} ${kind === "full" ? `(multiple scattering, ≤ ${MAX_BOUNCES} bounces)` : "(single scattering)"} … `);
        const r = await runPage(browser, "k6-reference", `${query}${common}&maxBounces=${bounces}`, "__VDB_K6_REFERENCE__");
        const img = referenceImage(r.report);
        entry[kind] = { img, report: { ...r.report, mean: undefined, stderr: undefined, overlap: undefined }, adapter: r.adapter };
        console.log(`${r.report.batches} batches in ${r.report.seconds.toFixed(0)} s, noise ${(img.noise * 100).toFixed(2)} % (p90 ${(img.noiseP90 * 100).toFixed(1)} %, ${r.report.stop}) (${r.adapter})`);
        writeFileSync(path.join(dir, `reference-${lighting}-${kind}.png`), encodePng(toRgba8(img)));
      }
      refs[lighting] = entry;
    }
    const rows = [];
    const add = (label, img, kind, lighting, extra = {}) => {
      const ref = refs[lightings.includes(lighting) ? lighting : lightings[0]];
      const file = label.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
      writeFileSync(path.join(dir, `${file}.png`), encodePng(toRgba8(img)));
      const vsFull = ref.full ? compare(img, ref.full.img) : null;
      const vsSingle = compare(img, ref.single.img);
      writeFileSync(path.join(dir, `${file}.err.png`), encodePng(diffHeatmap(toRgba8(img), toRgba8((ref.full ?? ref.single).img))));
      rows.push({ label, kind, lighting, vsFull, vsSingle, ...extra });
    };
    for (const t of frames) {
      if (t.slot) {
        rows.push({ label: t.label, kind: "slot", vsFull: null, vsSingle: null });
        continue;
      }
      add(t.label, t.img, t.backend === "core" ? "core" : "tsl", t.lighting, { vsSingleExpected: !!t.vsSingle });
      if (t.capture && existsSync(path.join(CAPTURES, `${t.capture}.png`))) {
        const png = decodePng(readFileSync(path.join(CAPTURES, `${t.capture}.png`)));
        add(`his ${t.capture}`, fromRgba8(png.data, png.width, png.height, Math.round(png.width / gw)), "his", t.lighting, { vsSingleExpected: !!t.vsSingle });
      }
    }
    for (const l of lightings) if (refs[l].full) add(`reference single scattering (${l})`, refs[l].single.img, "reference", l);
    const group = {
      key,
      kind: g.kind,
      title: g.title,
      scenario: g.scenario ?? g.scene,
      size: [gw, gh],
      scale,
      references: Object.fromEntries(Object.entries(refs).map(([l, e]) => [l, { full: e.full?.report ?? null, single: e.single.report, noise: { full: e.full?.img.noise ?? null, single: e.single.img.noise } }])),
      rows,
    };
    report.groups.push(group);
    group.md = markdown(group);
    for (const r of rows) {
      if (!r.vsSingle) {
        console.log(`  ${r.label.padEnd(56)} (slot)`);
        continue;
      }
      const v = r.vsFull ?? r.vsSingle;
      console.log(`  ${r.label.padEnd(56)} vs ${r.vsFull ? "MS" : "SS"}: bias ${pct(v.lumBias)}  median ${pct(v.medianRelLum)}  SSIM ${f(v.ssimRect, 3)}  sil. err ${pct(v.silhouette.relError)} | vs SS: bias ${pct(r.vsSingle.lumBias)}  sil. err ${pct(r.vsSingle.silhouette.relError)}`);
    }
  }
} finally {
  await browser.close();
  server?.kill();
}

function markdown(g) {
  const md = [];
  const refLine = Object.entries(g.references)
    .map(([l, r]) => `${l} mapping: ${r.full ? `multiple scattering ${r.full.batches} × ${SPP} spp (noise ${pct(r.noise.full, 2)}${r.full.converged ? "" : `, stopped by ${r.full.stop}`}), ` : ""}single ${r.single.batches} × ${SPP} spp (noise ${pct(r.noise.single, 2)})`)
    .join("; ");
  md.push(`## ${g.key}: ${g.title}`, "", `\`${g.scenario}\`, reference ${g.size[0]}×${g.size[1]} (frames ${g.scale}× down): ${refLine}.`, "");
  md.push("| Image | vs MS: lum. bias | median rel. err | SSIM (rect) | PSNR dB | silhouette err | vs SS: lum. bias | median rel. err | silhouette err |" + (g.rows.some((r) => r.vsSingle?.overlap) ? " overlap bias (MS / SS) |" : ""), "|---|---|---|---|---|---|---|---|---|" + (g.rows.some((r) => r.vsSingle?.overlap) ? "---|" : ""));
  for (const r of g.rows) {
    if (!r.vsSingle) {
      md.push(`| ${r.label} | — | — | — | — | — | — | — | — |`);
      continue;
    }
    const a = r.vsFull;
    const s = r.vsSingle;
    const ovl = s.overlap ? ` ${a?.overlap ? pct(a.overlap.lumBias) : "—"} / ${pct(s.overlap.lumBias)} |` : "";
    md.push(`| ${r.label} | ${a ? pct(a.lumBias) : "—"} | ${a ? pct(a.medianRelLum) : "—"} | ${a ? f(a.ssimRect, 3) : "—"} | ${a ? f(a.psnr) : "—"} | ${a ? pct(a.silhouette.relError) : "—"} | ${pct(s.lumBias)} | ${pct(s.medianRelLum)} | ${pct(s.silhouette.relError)} |${ovl}`);
  }
  md.push("");
  return md;
}

const header = [
  "# K6.R / K6.R1 reference report",
  "",
  `${report.at}, Chrome ${report.browser}${SWIFT ? " (SwiftShader: script debugging only, not a result)" : ""}. Report only: nothing gates on these numbers; no timings.`,
  "Reference: brute-force volumetric path tracing (core/reference) at level 0, box-filtered pixels, per-cell majorants; stopped at",
  `a mean relative standard error of Y of ${pct(TARGET)} over the volume pixels (cap ${BUDGET} s). Frames at ${SCALE}× the size (his 1280 × 720 at`,
  "160 × 90), box-downsampled in linear light; his captures likewise. All composited with his sRGB blend over 0x598eff.",
  "Columns: luminance bias Σ Y / Σ Y_ref − 1 over the volume pixels (the energy error), median relative luminance error",
  "per pixel (floor ≈ the reference noise), SSIM of the volume rectangle, PSNR of the whole frame, the silhouette band's",
  "error Σ|A − R| / Σ|R − background| (V80), and vs single scattering the same (rows marked ms off compare there).",
  "",
];
// A partial run (--groups) keeps the other groups of the previous report.
const previous = existsSync(path.join(OUT, "report.json")) ? JSON.parse(readFileSync(path.join(OUT, "report.json"), "utf8")) : null;
const kept = (previous?.groups ?? []).filter((g) => !report.groups.some((n) => n.key === g.key) && GROUP_DEFS[g.key]);
report.groups = [...kept, ...report.groups].sort((a, b) => Object.keys(GROUP_DEFS).indexOf(a.key) - Object.keys(GROUP_DEFS).indexOf(b.key));
writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 2));
writeFileSync(path.join(OUT, "report.md"), [...header, ...report.groups.flatMap((g) => [...(g.md ?? []), ...(kept.includes(g) ? [`(from the run of ${previous.at})`, ""] : [])])].join("\n"));
console.log(`wrote ${path.relative(ROOT, path.join(OUT, "report.md"))}`);
