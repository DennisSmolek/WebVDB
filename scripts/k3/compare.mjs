#!/usr/bin/env node
// K3 parity comparison (R1 §6): our render vs a reference capture.
//
//   node scripts/k3/compare.mjs <ours.png> <reference.png> [options]
//   node scripts/k3/compare.mjs --ours <dir> --reference <dir> [options]   # every PNG name in both
//
// Options:
//   --bar strict|jpeg-smoke   strict: PSNR >= 40 dB and <= 0.5% pixels > 8/255 (default; Step 1+ on identical
//                             dense data). jpeg-smoke: his loose check, <= 1% pixels > 48/255.
//   --threshold N --max-fraction F --min-psnr DB   override the bar's numbers
//   --heatmap <file|dir>      write the diff heatmap PNG(s)
//   --timings <json>          GPU timing hook (stub): print per-pass ms from a JSON written by a harness
//                             ({ "<pass>": ms, ... } or { "passes": {...} }). Always labelled provisional:
//                             no timing gates until a quiet machine (D25).
//   --json                    print a JSON report instead of text
//   --mask none|volume        volume: compare only inside the volume's screen rectangle, read from the
//                             `<name>.json` sidecar next to our PNG (`volumeRect`, written by the K3.1 harness).
//                             Default: volume when the sidecar has a rect, else none.
//   --scenes a,b              directory mode: only these names
//   --gate expected|all       expected (default): a pair whose sidecar says `parityExpected: false` (the Step 1
//                             port cannot match that scene yet, e.g. environment or 'frame' lighting) is
//                             reported as INFO and does not set the exit code. all: every pair gates.
//
// K3.1 one-shot (after the owner's K3.0a capture exists in fixtures/k3-reference/75e7345/):
//   pnpm k3:parity     # renders our parity scenes (e2e/k3-frame.spec.ts, K3_FRAME_PARITY=1) then runs this
//                      # with --ours artifacts/k3/ours --reference fixtures/k3-reference/75e7345 --heatmap artifacts/k3/diff
//
// Exit code: 0 when every gating pair passes, 1 when any fails, 2 on a usage or input error.
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { PARITY_BARS, compareImages, diffHeatmap } from "./metrics.mjs";
import { decodePng, encodePng } from "./png.mjs";

function usage(message) {
  if (message) console.error(`compare: ${message}`);
  console.error("usage: node scripts/k3/compare.mjs <ours.png> <reference.png> [--bar strict|jpeg-smoke] [--heatmap out.png] [--timings t.json] [--json]");
  console.error("       node scripts/k3/compare.mjs --ours <dir> --reference <dir> [--heatmap <dir>] ...");
  process.exit(2);
}

const args = process.argv.slice(2);
const positional = [];
const opt = {};
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === "--json") opt.json = true;
  else if (a.startsWith("--")) {
    const v = args[++i];
    if (v === undefined) usage(`${a} needs a value`);
    opt[a.slice(2)] = v;
  } else positional.push(a);
}

const barName = opt.bar ?? "strict";
const base = barName === "strict" ? PARITY_BARS.strict : barName === "jpeg-smoke" ? PARITY_BARS.jpegSmoke : usage(`unknown bar ${barName}`);
const bar = {
  minPsnr: opt["min-psnr"] !== undefined ? Number(opt["min-psnr"]) : base.minPsnr,
  threshold: opt.threshold !== undefined ? Number(opt.threshold) : base.threshold,
  maxFraction: opt["max-fraction"] !== undefined ? Number(opt["max-fraction"]) : base.maxFraction,
};

/** @type {{ name: string, ours: string, reference: string }[]} */
let pairs;
if (opt.ours || opt.reference) {
  if (!opt.ours || !opt.reference) usage("--ours and --reference go together");
  for (const d of [opt.ours, opt.reference]) if (!existsSync(d) || !statSync(d).isDirectory()) usage(`not a directory: ${d}`);
  const refs = new Set(readdirSync(opt.reference).filter((f) => f.endsWith(".png")));
  const only = opt.scenes ? new Set(String(opt.scenes).split(",")) : null;
  pairs = readdirSync(opt.ours)
    .filter((f) => f.endsWith(".png") && refs.has(f) && (!only || only.has(f.replace(/\.png$/, ""))))
    .sort()
    .map((f) => ({ name: f.replace(/\.png$/, ""), ours: path.join(opt.ours, f), reference: path.join(opt.reference, f) }));
  if (!pairs.length) usage(`no PNG names in both ${opt.ours} and ${opt.reference}`);
} else {
  if (positional.length !== 2) usage();
  for (const f of positional) if (!existsSync(f)) usage(`missing file: ${f}`);
  pairs = [{ name: path.basename(positional[0]).replace(/\.png$/, ""), ours: positional[0], reference: positional[1] }];
}

const maskMode = opt.mask ?? "auto";
if (!["auto", "none", "volume"].includes(maskMode)) usage(`unknown --mask ${maskMode}`);
const gateMode = opt.gate ?? "expected";
if (!["expected", "all"].includes(gateMode)) usage(`unknown --gate ${gateMode}`);

/** The K3.1 harness sidecar next to our PNG (`<name>.json`), or null. */
function sidecarOf(oursPng) {
  const f = oursPng.replace(/\.png$/, ".json");
  if (!existsSync(f)) return null;
  try {
    return JSON.parse(readFileSync(f, "utf8"));
  } catch {
    return null;
  }
}

/** Crop to [x0, y0, x1) x [y0, y1) (top-left origin). */
function crop(img, [x0, y0, x1, y1]) {
  const w = Math.max(0, Math.min(img.width, x1) - x0);
  const h = Math.max(0, Math.min(img.height, y1) - y0);
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) data.set(img.data.subarray(((y0 + y) * img.width + x0) * 4, ((y0 + y) * img.width + x0 + w) * 4), y * w * 4);
  return { width: w, height: h, data };
}

const results = [];
for (const p of pairs) {
  let a;
  let b;
  try {
    a = decodePng(readFileSync(p.ours));
    b = decodePng(readFileSync(p.reference));
  } catch (e) {
    usage(`${p.name}: ${e.message}`);
  }
  const sidecar = sidecarOf(p.ours);
  const rect = sidecar?.volumeRect;
  let mask = null;
  if (maskMode === "volume" || (maskMode === "auto" && Array.isArray(rect))) {
    if (!Array.isArray(rect) || rect.length !== 4) usage(`${p.name}: --mask volume needs volumeRect in ${p.ours.replace(/\.png$/, ".json")}`);
    if (a.width !== b.width || a.height !== b.height) usage(`${p.name}: size mismatch ${a.width}x${a.height} vs ${b.width}x${b.height}`);
    mask = rect;
    a = crop(a, rect);
    b = crop(b, rect);
  }
  const gates = gateMode === "all" || sidecar?.parityExpected !== false;
  let r;
  try {
    r = compareImages(a, b, bar);
  } catch (e) {
    usage(`${p.name}: ${e.message}`);
  }
  if (opt.heatmap) {
    const out = pairs.length > 1 || (existsSync(opt.heatmap) && statSync(opt.heatmap).isDirectory()) ? path.join(opt.heatmap, `${p.name}-diff.png`) : opt.heatmap;
    mkdirSync(path.dirname(out), { recursive: true });
    writeFileSync(out, encodePng(diffHeatmap(a, b, { threshold: bar.threshold })));
    r.heatmap = out;
  }
  results.push({ name: p.name, ours: p.ours, reference: p.reference, mask, gates, ...(sidecar?.parityNote ? { note: sidecar.parityNote } : {}), ...r });
}

let timings = null;
if (opt.timings) {
  // GPU timing hook (stub). Timings never gate before a quiet machine (D25); they are reported, labelled provisional.
  const raw = JSON.parse(readFileSync(opt.timings, "utf8"));
  timings = { provisional: true, note: "D25: not from a quiet machine; do not decide on these", passes: raw.passes ?? raw };
}

const pass = results.every((r) => r.pass || !r.gates);
if (opt.json) {
  console.log(JSON.stringify({ bar: { name: barName, ...bar }, pass, results, timings }, (k, v) => (v === Infinity ? "Infinity" : v), 2));
} else {
  console.log(`bar ${barName}: PSNR >= ${bar.minPsnr} dB, <= ${(bar.maxFraction * 100).toFixed(2)}% pixels > ${bar.threshold}/255`);
  for (const r of results) {
    const psnrText = r.psnr === Infinity ? "inf (identical)" : `${r.psnr.toFixed(2)} dB`;
    console.log(
      `${!r.gates ? "INFO" : r.pass ? "PASS" : "FAIL"} ${r.name}${r.mask ? ` [volume rect ${r.mask.join(",")}]` : ""}: PSNR ${psnrText}, ${r.above}/${r.pixels} px (${(r.fraction * 100).toFixed(3)}%) > ${r.threshold}/255, max diff ${r.maxDiff}` +
        (r.heatmap ? `, heatmap ${r.heatmap}` : "") +
        (r.failures.length ? `\n     ${r.failures.join("; ")}` : "") +
        (r.note ? `\n     not gating: ${r.note}` : ""),
    );
  }
  if (timings) {
    console.log("GPU timings (PROVISIONAL, D25):");
    for (const [k, v] of Object.entries(timings.passes)) console.log(`  ${k}: ${v === null ? "n/a" : `${Number(v).toFixed(3)} ms`}`);
  }
}
process.exit(pass ? 0 : 1);
