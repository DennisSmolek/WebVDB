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
//
// Exit code: 0 when every pair passes, 1 when any fails, 2 on a usage or input error.
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
  pairs = readdirSync(opt.ours)
    .filter((f) => f.endsWith(".png") && refs.has(f))
    .sort()
    .map((f) => ({ name: f.replace(/\.png$/, ""), ours: path.join(opt.ours, f), reference: path.join(opt.reference, f) }));
  if (!pairs.length) usage(`no PNG names in both ${opt.ours} and ${opt.reference}`);
} else {
  if (positional.length !== 2) usage();
  for (const f of positional) if (!existsSync(f)) usage(`missing file: ${f}`);
  pairs = [{ name: path.basename(positional[0]).replace(/\.png$/, ""), ours: positional[0], reference: positional[1] }];
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
  results.push({ name: p.name, ours: p.ours, reference: p.reference, ...r });
}

let timings = null;
if (opt.timings) {
  // GPU timing hook (stub). Timings never gate before a quiet machine (D25); they are reported, labelled provisional.
  const raw = JSON.parse(readFileSync(opt.timings, "utf8"));
  timings = { provisional: true, note: "D25: not from a quiet machine; do not decide on these", passes: raw.passes ?? raw };
}

const pass = results.every((r) => r.pass);
if (opt.json) {
  console.log(JSON.stringify({ bar: { name: barName, ...bar }, pass, results, timings }, (k, v) => (v === Infinity ? "Infinity" : v), 2));
} else {
  console.log(`bar ${barName}: PSNR >= ${bar.minPsnr} dB, <= ${(bar.maxFraction * 100).toFixed(2)}% pixels > ${bar.threshold}/255`);
  for (const r of results) {
    const psnrText = r.psnr === Infinity ? "inf (identical)" : `${r.psnr.toFixed(2)} dB`;
    console.log(
      `${r.pass ? "PASS" : "FAIL"} ${r.name}: PSNR ${psnrText}, ${r.above}/${r.pixels} px (${(r.fraction * 100).toFixed(3)}%) > ${r.threshold}/255, max diff ${r.maxDiff}` +
        (r.heatmap ? `, heatmap ${r.heatmap}` : "") +
        (r.failures.length ? `\n     ${r.failures.join("; ")}` : ""),
    );
  }
  if (timings) {
    console.log("GPU timings (PROVISIONAL, D25):");
    for (const [k, v] of Object.entries(timings.passes)) console.log(`  ${k}: ${v === null ? "n/a" : `${Number(v).toFixed(3)} ms`}`);
  }
}
process.exit(pass ? 0 : 1);
