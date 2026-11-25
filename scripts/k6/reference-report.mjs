// K6.R: quality report of the TSL froxel renderer (and his K3 captures) against the reference renderer (a brute-force
// volumetric path tracer, core/reference; design K6-core-renderer.md §7.4). REPORT ONLY: nothing gates on it.
//
//   node scripts/k6/reference-report.mjs                    # starts its own Vite server, Chrome on the real GPU (headed)
//   options: --groups=volume,frame,two  --w=320 --h=180 (reference size; the TSL frame renders at --scale× and is
//            box-downsampled in linear light) --scale=4 --budget=120 (seconds of reference per render) --batches=4096
//            --spp=1 --maxBounces=512 --port=5461 --base=http://127.0.0.1:5173 (reuse a server)
//            --swiftshader (bundled Chromium + SwiftShader, for debugging the script itself: tiny, never a result)
//            --headless (installed Chrome without a window; check the adapter it reports: it may fall back)
//
// Needs fixtures/k3/bunny_cloud.wvdb and fixtures/k3-reference/75e7345/ (his captures + manifest). Writes
// artifacts/k6-reference/<group>/*.png (reference, TSL, his, heat maps) and artifacts/k6-reference/report.{json,md}.
//
// Per comparison (A vs the reference with multiple scattering, and vs single scattering where it says so):
//   PSNR (sRGB 8-bit, whole frame), pixels > 8/255, and over the volume pixels (reference alpha > 0.02): RMSE of the
//   linear radiance, median and mean relative luminance error |Y_A − Y_ref| / Y_ref (pixels with Y_ref > 1e-3), and the
//   luminance bias Σ Y_A / Σ Y_ref − 1 (Y of the composited frame minus the background's share). The reference's own
//   noise (mean relative standard error of Y over the volume pixels) is listed: errors below it are not resolved.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { decodePng, encodePng } from "../k3/png.mjs";
import { diffHeatmap } from "../k3/metrics.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.join(ROOT, "artifacts", "k6-reference");
const CAPTURES = path.join(ROOT, "fixtures", "k3-reference", "75e7345");
const arg = (name, fallback) => {
  const hit = process.argv.find((a) => a === `--${name}` || a.startsWith(`--${name}=`));
  if (!hit) return fallback;
  return hit.includes("=") ? hit.slice(hit.indexOf("=") + 1) : true;
};
const SWIFT = arg("swiftshader", false) === true;
const W = Number(arg("w", SWIFT ? 64 : 320));
const H = Number(arg("h", SWIFT ? 36 : 180));
const SCALE = Number(arg("scale", 4));
const BUDGET = Number(arg("budget", SWIFT ? 5 : 120));
const BATCHES = Number(arg("batches", SWIFT ? 4 : 4096));
const SPP = Number(arg("spp", 1));
const MAX_BOUNCES = Number(arg("maxBounces", 512));
const PORT = Number(arg("port", 5461));
const GROUPS = String(arg("groups", "volume,frame,two")).split(",");
let BASE = arg("base", "");

if (!existsSync(path.join(ROOT, "fixtures", "k3", "bunny_cloud.wvdb")) || !existsSync(path.join(CAPTURES, "manifest.json"))) {
  console.error("needs fixtures/k3/bunny_cloud.wvdb and fixtures/k3-reference/75e7345/manifest.json (copy them from the main checkout)");
  process.exit(2);
}
const MANIFEST = JSON.parse(readFileSync(path.join(CAPTURES, "manifest.json"), "utf8"));

// --- the groups: one reference geometry each, several TSL renders and captures compared with it ---------------------
// 'volume' lighting uses the volume's phase (k3-frame: his frame.js dual-lobe 0.8 / −0.3 / 0.2); 'frame' lighting the
// renderer's settings phase (read from the TSL run's `settings`; k3-frame sets his demos' dual-lobe 0.8 / −0.3 / 0.2 too).
// k3-frame's Step 3 scenes default to multipleScattering true (diffusion), as his captures of them; 'neural' needs ?ms=neural.
const VOLUME_PHASE = [0.8, -0.3, 0.2];
const GROUP_DEFS = {
  volume: {
    title: "'volume' lighting, his sun only (Step 1 isolated)",
    scenario: "step1-volume-dir-ms-on-isolated",
    lighting: "volume",
    tsl: [
      { label: "TSL ms off", query: "scene=step1-volume-dir-ms-off-isolated", capture: "step1-volume-dir-ms-off-isolated", vsSingle: true },
      { label: "TSL ms on (diffusion)", query: "scene=step1-volume-dir-ms-on-isolated", capture: "step1-volume-dir-ms-on-isolated" },
    ],
  },
  frame: {
    title: "'frame' lighting, his sun only (Step 3 default isolated, every multipleScattering mode)",
    scenario: "step3-default-dir-isolated",
    lighting: "frame",
    tsl: [
      { label: "TSL ms off", query: "scene=step3-default-dir-isolated&ms=0", vsSingle: true },
      { label: "TSL ms diffusion", query: "scene=step3-default-dir-isolated&ms=1", capture: "step3-default-dir-isolated" },
      { label: "TSL ms two-sided", query: "scene=step3-default-dir-isolated&ms=two-sided", capture: "step4-multiple-scattering-two-sided-dir-isolated" },
      { label: "TSL ms octaves", query: "scene=step3-default-dir-isolated&ms=octaves", capture: "step4-multiple-scattering-octaves-dir-isolated" },
      { label: "TSL ms neural (his v0.26 default)", query: "scene=step3-default-dir-isolated&ms=neural", capture: "step4-default-neural-dir-isolated" },
    ],
  },
  two: {
    title: "two overlapping bunnies, 'frame' lighting (draw-order 'over' mixing in the TSL path)",
    scenario: "step3-two-bunnies-dir-isolated",
    lighting: "frame",
    tsl: [{ label: "TSL diffusion, over", query: "scene=step3-two-bunnies-dir-isolated", capture: "step3-two-bunnies-dir-isolated" }],
  },
};

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

/** The reference's mean (L, T) → composited linear RGB (his sRGB blend over the background), alpha and noise. */
function referenceImage(r) {
  const mean = new Float32Array(Buffer.from(r.mean, "base64").buffer.slice(0));
  const stderr = new Float32Array(Buffer.from(r.stderr, "base64").buffer.slice(0));
  const n = r.width * r.height;
  const rgb = new Float64Array(n * 3);
  const alpha = new Float64Array(n);
  let noise = 0;
  let noiseN = 0;
  for (let i = 0; i < n; i++) {
    const L = [mean[i * 4], mean[i * 4 + 1], mean[i * 4 + 2]];
    const a = clamp01(1 - mean[i * 4 + 3]);
    alpha[i] = a;
    const c = a > 1e-6 ? srgbBlend(BG, L, a) : BG;
    rgb.set(c, i * 3);
    const y = lum(...L);
    if (a > 0.02 && y > 1e-3) {
      noise += lum(stderr[i * 4], stderr[i * 4 + 1], stderr[i * 4 + 2]) / y;
      noiseN++;
    }
  }
  return { width: r.width, height: r.height, rgb, alpha, noise: noiseN ? noise / noiseN : 0, volumePixels: noiseN };
}

function compare(a, ref) {
  const n = ref.width * ref.height;
  if (a.width !== ref.width || a.height !== ref.height) throw new Error(`size ${a.width}x${a.height} vs ${ref.width}x${ref.height}`);
  const A8 = toRgba8(a);
  const R8 = toRgba8(ref);
  let mse8 = 0;
  let over8 = 0;
  let sq = 0;
  let vol = 0;
  let relSum = 0;
  let relN = 0;
  const rels = [];
  let yA = 0;
  let yR = 0;
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let c = 0; c < 3; c++) {
      const d = A8.data[i * 4 + c] - R8.data[i * 4 + c];
      mse8 += d * d;
      m = Math.max(m, Math.abs(d));
    }
    if (m > 8) over8++;
    if (ref.alpha[i] > 0.02) {
      vol++;
      for (let c = 0; c < 3; c++) sq += (a.rgb[i * 3 + c] - ref.rgb[i * 3 + c]) ** 2;
      // Luminance of the volume's contribution: the composite minus the background's remaining share.
      const t = 1 - ref.alpha[i];
      const ya = lum(a.rgb[i * 3], a.rgb[i * 3 + 1], a.rgb[i * 3 + 2]) - t * lum(...BG);
      const yr = lum(ref.rgb[i * 3], ref.rgb[i * 3 + 1], ref.rgb[i * 3 + 2]) - t * lum(...BG);
      yA += ya;
      yR += yr;
      if (yr > 1e-3) {
        relSum += Math.abs(ya - yr) / yr;
        rels.push(Math.abs(ya - yr) / yr);
        relN++;
      }
    }
  }
  mse8 /= n * 3;
  return {
    psnr: mse8 === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse8),
    over8: over8 / n,
    volumePixels: vol,
    rmseLinear: vol ? Math.sqrt(sq / (vol * 3)) : 0,
    meanRelLum: relN ? relSum / relN : 0,
    medianRelLum: rels.length ? rels.sort((x, y) => x - y)[Math.floor(rels.length / 2)] : 0,
    lumBias: yR > 0 ? yA / yR - 1 : 0,
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
  await p.waitForFunction((s) => window[s]?.done === true, state, { timeout: 1_800_000 });
  const r = await p.evaluate((s) => window[s], state);
  await p.close();
  if (r.error) throw new Error(`${page}?${query}: ${String(r.error).split("\n")[0]}`);
  return { ...r, pageErrors: errors };
}

const server = await startServer();
const browser = await chromium.launch({
  ...(SWIFT ? {} : { channel: "chrome" }),
  // Headed: macOS Chrome picks the hardware adapter only with a window server session (scripts/k3/time-hardware.mjs).
  headless: SWIFT || arg("headless", false) === true,
  args: ["--enable-unsafe-webgpu", ...(SWIFT ? ["--enable-features=Vulkan", "--enable-unsafe-swiftshader"] : [])],
});
const report = { at: new Date().toISOString(), browser: browser.version(), size: [W, H], scale: SCALE, budgetSeconds: BUDGET, maxBounces: MAX_BOUNCES, groups: [] };
mkdirSync(OUT, { recursive: true });
try {
  for (const key of GROUPS) {
    const g = GROUP_DEFS[key];
    if (!g) throw new Error(`unknown group ${key}`);
    const dir = path.join(OUT, key);
    mkdirSync(dir, { recursive: true });
    console.log(`== ${key}: ${g.title}`);
    // TSL renders first ('frame' lighting: their settings give the reference its phase).
    const tsl = [];
    for (const t of g.tsl) {
      process.stdout.write(`  ${t.label} … `);
      const r = await runPage(browser, "k3-frame", `${t.query}&w=${W * SCALE}&h=${H * SCALE}`, "__VDB_K3_FRAME__");
      const img = fromRgba8(new Uint8Array(Buffer.from(r.rgba, "base64")), r.width, r.height, SCALE);
      tsl.push({ ...t, img, settings: r.settings, adapter: r.adapter });
      console.log(`${r.width}x${r.height} (${r.adapter})`);
    }
    const s = tsl.find((t) => t.settings)?.settings;
    const phase = g.lighting === "volume" ? VOLUME_PHASE : s?.phase === "dual-lobe" ? [s.phaseG, s.phaseBackG, s.phaseBackWeight] : [s?.phaseG ?? 0.8, 0, 0];
    if (g.lighting === "frame" && s && !["henyey-greenstein", "dual-lobe", "isotropic"].includes(s.phase)) console.warn(`  phase ${s.phase}: the reference uses HG g = ${phase[0]}`);
    const refQuery = (maxBounces) => `case=report&scenario=${g.scenario}&w=${W}&h=${H}&spp=${SPP}&batches=${BATCHES}&budget=${BUDGET}&maxBounces=${maxBounces}&phase=${phase.join(",")}`;
    process.stdout.write(`  reference (multiple scattering, ≤ ${MAX_BOUNCES} bounces) … `);
    const refFull = await runPage(browser, "k6-reference", refQuery(MAX_BOUNCES), "__VDB_K6_REFERENCE__");
    const full = referenceImage(refFull.report);
    console.log(`${refFull.report.batches} batches in ${refFull.report.seconds.toFixed(0)} s, noise ${(full.noise * 100).toFixed(2)} % (${refFull.adapter})`);
    process.stdout.write("  reference (single scattering) … ");
    const refSingle = await runPage(browser, "k6-reference", refQuery(1), "__VDB_K6_REFERENCE__");
    const single = referenceImage(refSingle.report);
    console.log(`${refSingle.report.batches} batches in ${refSingle.report.seconds.toFixed(0)} s, noise ${(single.noise * 100).toFixed(2)} %`);
    writeFileSync(path.join(dir, "reference.png"), encodePng(toRgba8(full)));
    writeFileSync(path.join(dir, "reference-single.png"), encodePng(toRgba8(single)));
    const rows = [];
    const add = (label, img, kind) => {
      const file = label.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
      writeFileSync(path.join(dir, `${file}.png`), encodePng(toRgba8(img)));
      const vsFull = compare(img, full);
      writeFileSync(path.join(dir, `${file}.err.png`), encodePng(diffHeatmap(toRgba8(img), toRgba8(full))));
      const vsSingle = compare(img, single);
      rows.push({ label, kind, vsFull, vsSingle });
    };
    for (const t of tsl) {
      add(t.label, t.img, "tsl");
      if (t.capture && existsSync(path.join(CAPTURES, `${t.capture}.png`))) {
        const png = decodePng(readFileSync(path.join(CAPTURES, `${t.capture}.png`)));
        add(`his ${t.capture}`, fromRgba8(png.data, png.width, png.height, Math.round(png.width / W)), "his");
      }
    }
    add("reference single scattering", single, "reference");
    const group = { key, title: g.title, scenario: g.scenario, phase, reference: { full: { ...refFull.report, mean: undefined, stderr: undefined, noise: full.noise, volumePixels: full.volumePixels }, single: { ...refSingle.report, mean: undefined, stderr: undefined, noise: single.noise } }, rows };
    report.groups.push(group);
    const md = (group.md = []);
    md.push(`## ${key}: ${g.title}`, "", `Scenario \`${g.scenario}\`, phase (g, back g, back weight) = ${phase.join(", ")}. Reference ${W}×${H}: multiple scattering ${refFull.report.batches} × ${SPP} spp (noise ${(full.noise * 100).toFixed(2)} %), single scattering ${refSingle.report.batches} × ${SPP} spp (noise ${(single.noise * 100).toFixed(2)} %).`, "");
    md.push("| Image | vs reference (MS): PSNR dB | > 8/255 | RMSE lin | median rel. lum. err | mean rel. lum. err | lum. bias | vs single scattering: median rel. lum. err | lum. bias |", "|---|---|---|---|---|---|---|---|---|");
    const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : String(x));
    for (const r of rows) md.push(`| ${r.label} | ${f(r.vsFull.psnr)} | ${f(r.vsFull.over8 * 100, 1)} % | ${f(r.vsFull.rmseLinear, 4)} | ${f(r.vsFull.medianRelLum * 100, 1)} % | ${f(r.vsFull.meanRelLum * 100, 1)} % | ${f(r.vsFull.lumBias * 100, 1)} % | ${f(r.vsSingle.medianRelLum * 100, 1)} % | ${f(r.vsSingle.lumBias * 100, 1)} % |`);
    md.push("");
    for (const r of rows) console.log(`  ${r.label.padEnd(48)} vs MS: PSNR ${f(r.vsFull.psnr)}  median rel ${f(r.vsFull.medianRelLum * 100, 1)} %  bias ${f(r.vsFull.lumBias * 100, 1)} % | vs SS: median rel ${f(r.vsSingle.medianRelLum * 100, 1)} %  bias ${f(r.vsSingle.lumBias * 100, 1)} %`);
  }
} finally {
  await browser.close();
  server?.kill();
}
const header = [
  "# K6.R reference report",
  "",
  `${report.at}, Chrome ${report.browser}${SWIFT ? " (SwiftShader: script debugging only, not a result)" : ""}. Report only: nothing gates on these numbers.`,
  "Reference: brute-force volumetric path tracing (core/reference), his bunny at level 0, box-filtered pixels; TSL frames at",
  `${SCALE}× the size, box-downsampled in linear light; his captures likewise. All composited with his sRGB blend over 0x598eff.`,
  "",
];
// A partial run (--groups) keeps the other groups of the previous report.
const previous = existsSync(path.join(OUT, "report.json")) ? JSON.parse(readFileSync(path.join(OUT, "report.json"), "utf8")) : null;
const kept = (previous?.groups ?? []).filter((g) => !report.groups.some((n) => n.key === g.key));
report.groups = [...kept, ...report.groups].sort((a, b) => Object.keys(GROUP_DEFS).indexOf(a.key) - Object.keys(GROUP_DEFS).indexOf(b.key));
writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 2));
writeFileSync(path.join(OUT, "report.md"), [...header, ...report.groups.flatMap((g) => [...(g.md ?? []), ...(kept.includes(g) ? [`(from the run of ${previous.at})`, ""] : [])])].join("\n"));
console.log(`wrote ${path.relative(ROOT, path.join(OUT, "report.md"))}`);
