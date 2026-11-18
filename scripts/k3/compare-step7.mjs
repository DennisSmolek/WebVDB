#!/usr/bin/env node
// K3.7 parity comparison (R1 §6 Step 7): our level-set and points renders vs his Step 7 captures.
//
//   node scripts/k3/compare-step7.mjs --ours artifacts/k3/ours-step7 --reference fixtures/k3-reference/75e7345 [--json]
//
// Metrics (both images display-referred sRGB 8-bit, his canvas and our harness's sRGB OETF of a linear target):
// - `*-normals` (his `isolate: 'normals'`: unlit view-space normal * 0.5 + 0.5 on black): silhouette = pixels with a
//   channel > 8 (a unit normal's colour is never that dark); silhouette IoU; normal-angle error (decode: sRGB EOTF,
//   n = 2c - 1, normalized) over pixels inside both silhouettes, the 1-pixel rim removed: mean, p99, max.
//   Gate (R1 Step 7): IoU >= 0.99 for the GPU-surface scenes. The normal error has no bar in R1; it is reported,
//   with a provisional bar (mean <= 5 degrees) printed as PROVISIONAL, not gating. `mesh-normals` (his marching-cubes
//   mesh vs our trace) is INFO.
// - `step7-sphere-points-isolated` (white points on black): covered-pixel IoU (INFO: 1-pixel sprites shift with
//   sub-pixel differences) and the point count, his manifest's VDBPoints.pointCount vs our sidecar (gate: equal).
// - `*-asis`: INFO only (his 'environment' lights are not ported), PSNR.
// Exit code: 0 when every gate passes (or there is no Step 7 capture yet: SKIPPED), 1 when one fails, 2 on usage.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { psnr } from "./metrics.mjs";
import { decodePng } from "./png.mjs";

export const STEP7_BARS = Object.freeze({ iou: 0.99, provisionalNormalMeanDeg: 5, silhouetteThreshold: 8 });

const eotf = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/** Silhouette mask of an image on black: any channel above `threshold` (0-255). */
export function silhouette(img, threshold = STEP7_BARS.silhouetteThreshold) {
  const mask = new Uint8Array(img.width * img.height);
  for (let i = 0; i < mask.length; i++) {
    const d = img.data;
    mask[i] = d[i * 4] > threshold || d[i * 4 + 1] > threshold || d[i * 4 + 2] > threshold ? 1 : 0;
  }
  return mask;
}

/** Intersection over union of two masks. */
export function maskIou(a, b) {
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter++;
    if (a[i] || b[i]) union++;
  }
  return { iou: union ? inter / union : 1, intersection: inter, union, a: a.reduce((s, v) => s + v, 0), b: b.reduce((s, v) => s + v, 0) };
}

function erode(mask, w, h) {
  const out = new Uint8Array(mask.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out[i] = mask[i] & mask[i - 1] & mask[i + 1] & mask[i - w] & mask[i + w];
    }
  }
  return out;
}

/** The unit normal a pixel encodes (sRGB of n * 0.5 + 0.5). */
export function decodeNormal(data, i) {
  const n = [0, 1, 2].map((c) => eotf(data[i * 4 + c] / 255) * 2 - 1);
  const l = Math.hypot(n[0], n[1], n[2]) || 1;
  return n.map((v) => v / l);
}

/** Silhouette IoU and normal-angle error (degrees) of two normal images. */
export function compareNormals(ours, his) {
  if (ours.width !== his.width || ours.height !== his.height) throw new Error(`size mismatch ${ours.width}x${ours.height} vs ${his.width}x${his.height}`);
  const a = silhouette(ours);
  const b = silhouette(his);
  const iou = maskIou(a, b);
  const ea = erode(a, ours.width, ours.height);
  const eb = erode(b, his.width, his.height);
  const angles = [];
  for (let i = 0; i < ea.length; i++) {
    if (!(ea[i] && eb[i])) continue;
    const p = decodeNormal(ours.data, i);
    const q = decodeNormal(his.data, i);
    angles.push((Math.acos(Math.min(1, Math.max(-1, p[0] * q[0] + p[1] * q[1] + p[2] * q[2]))) * 180) / Math.PI);
  }
  angles.sort((x, y) => x - y);
  const pick = (f) => (angles.length ? angles[Math.min(angles.length - 1, Math.floor(angles.length * f))] : 0);
  return {
    ...iou,
    normal: { pixels: angles.length, meanDeg: angles.length ? angles.reduce((s, v) => s + v, 0) / angles.length : 0, medianDeg: pick(0.5), p99Deg: pick(0.99), maxDeg: angles.length ? angles[angles.length - 1] : 0 },
  };
}

function main() {
  const args = process.argv.slice(2);
  const opt = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--json") opt.json = true;
    else if (a.startsWith("--") && args[i + 1] !== undefined) opt[a.slice(2)] = args[++i];
    else {
      console.error(`compare-step7: unexpected argument ${a}`);
      process.exit(2);
    }
  }
  const oursDir = opt.ours ?? "artifacts/k3/ours-step7";
  const refDir = opt.reference ?? "fixtures/k3-reference/75e7345";
  const step7 = existsSync(refDir) ? readdirSync(refDir).filter((f) => /^step7-.*\.png$/.test(f) && !f.endsWith(".asis.png")) : [];
  if (step7.length === 0) {
    console.log(`compare-step7: SKIPPED, no step7-*.png captures in ${refDir} yet (run the Step 7 capture, see docs/handoffs/K3-7.md).`);
    return 0;
  }
  if (!existsSync(oursDir)) {
    console.error(`compare-step7: ${oursDir} is missing; run K3_FRAME_PARITY=1 playwright test e2e/k3-levelset.spec.ts -g "parity render"`);
    return 2;
  }
  const manifestPath = path.join(refDir, "manifest.json");
  const manifest = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, "utf8")) : { scenarios: [] };
  const results = [];
  let failed = false;
  for (const file of step7.sort()) {
    const name = file.replace(/\.png$/, "");
    const oursFile = path.join(oursDir, file);
    if (!existsSync(oursFile)) {
      results.push({ name, status: "MISSING", note: `no ${oursFile}` });
      failed = true;
      continue;
    }
    const ours = decodePng(readFileSync(oursFile));
    const his = decodePng(readFileSync(path.join(refDir, file)));
    const sidecarFile = oursFile.replace(/\.png$/, ".json");
    const sidecar = existsSync(sidecarFile) ? JSON.parse(readFileSync(sidecarFile, "utf8")) : null;
    const entry = manifest.scenarios?.find((s) => s.name === name);
    if (name.endsWith("-normals")) {
      const r = compareNormals(ours, his);
      const gates = name.includes("-surface-");
      const pass = r.iou >= STEP7_BARS.iou;
      if (gates && !pass) failed = true;
      results.push({
        name,
        status: gates ? (pass ? "PASS" : "FAIL") : "INFO",
        iou: r.iou,
        pixels: { ours: r.a, his: r.b },
        normal: r.normal,
        provisional: { normalMeanDeg: r.normal.meanDeg <= STEP7_BARS.provisionalNormalMeanDeg ? "within" : "over", bar: STEP7_BARS.provisionalNormalMeanDeg },
      });
    } else if (name === "step7-sphere-points-isolated") {
      const iou = maskIou(silhouette(ours, 64), silhouette(his, 64));
      const hisCount = entry?.metadata?.points?.[0]?.pointCount ?? null;
      const ourCount = sidecar?.points?.count ?? null;
      const pass = hisCount !== null && ourCount === hisCount;
      if (!pass) failed = true;
      results.push({ name, status: pass ? "PASS" : "FAIL", count: { ours: ourCount, his: hisCount }, coveredIou: iou.iou, pixels: { ours: iou.a, his: iou.b } });
    } else {
      results.push({ name, status: "INFO", psnr: psnr(ours, his) });
    }
  }
  if (opt.json) console.log(JSON.stringify(results, null, 2));
  else {
    for (const r of results) {
      const parts = [`${r.status.padEnd(7)} ${r.name}`];
      if (r.iou !== undefined) parts.push(`IoU ${r.iou.toFixed(4)} (ours ${r.pixels.ours} px, his ${r.pixels.his} px)`);
      if (r.normal) parts.push(`normal mean ${r.normal.meanDeg.toFixed(2)}° median ${r.normal.medianDeg.toFixed(2)}° p99 ${r.normal.p99Deg.toFixed(2)}° max ${r.normal.maxDeg.toFixed(1)}° (PROVISIONAL bar mean <= ${r.provisional.bar}°: ${r.provisional.normalMeanDeg})`);
      if (r.count) parts.push(`points ours ${r.count.ours} his ${r.count.his}; covered IoU ${r.coveredIou.toFixed(4)} (INFO)`);
      if (r.psnr !== undefined) parts.push(`PSNR ${r.psnr.toFixed(2)} dB (INFO: his 'environment' lights are not ported)`);
      if (r.note) parts.push(r.note);
      console.log(parts.join("  "));
    }
  }
  return failed ? 1 : 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main();
