import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.2 (R1 §6 Step 2): the sparse VDBGridPool source behind the VDBVolumeSource seam
// (examples/src/harness/k3-sparse.ts). Gates: the identical-voxel A/B (the committed CC0 cloud's
// .wvdb texels converted to a NanoVDB FLOAT grid: dense vs sparse 'none' differ only in the sampler)
// at PSNR >= 38 dB; HDDA skipping changes nothing (identical within float noise); the K3.2d level-0 sampler
// is bit-identical to the K3.2 one (fresh accessor + vdbSample per sample); every LOD strategy
// renders deterministically without validation errors; the sparse WGSL helpers match their CPU twins.
// The report-only runs (smoke.vdb, WDAS, his bunny, flicker per LOD strategy, timings) are opt-in:
// K3_SPARSE_REPORT=1 (see docs/archive/handoffs/K3-2.md).
const PAGE_URL = "/src/harness/k3-sparse.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OURS = path.join(repoRoot, "artifacts", "k3", "ours");
const REPORT = process.env.K3_SPARSE_REPORT === "1";
/** The report runs' size: node-average is slow on SwiftShader. */
const SMALL = "&w=240&h=135&froxels=96x96x16&substeps=16&shadowSteps=16";

interface Variant {
  name: string;
  rgba: string;
  nonBackground: number;
  inPageIdentical: boolean;
  gpuBytes: { source: number; pyramid: number };
  lodLevels: number;
  flicker?: { frames: number; meanDiff: number; temporalNoise: number };
  froxelsMs?: number | null;
  wallMs?: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  width?: number;
  height?: number;
  grid?: unknown;
  dense?: unknown;
  variants: Variant[];
  probe?: { count: number; over: number; maxAbs: number; skipMismatch: number; firstBad?: unknown };
  timingsNote?: string;
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string, timeout = 400_000): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_SPARSE__?: { done: boolean } }).__VDB_K3_SPARSE__?.done === true, undefined, { timeout });
  return (await page.evaluate(() => (window as { __VDB_K3_SPARSE__?: unknown }).__VDB_K3_SPARSE__)) as Result;
}

const image = (r: Result, v: Variant) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(v.rgba, "base64")) });
const detailOf = (r: Result) => JSON.stringify({ ...r, variants: r.variants.map((v) => ({ ...v, rgba: `${v.rgba.length} b64 chars` })) }, null, 2);
const byName = (r: Result, name: string) => r.variants.find((v) => v.name === name)!;

function save(name: string, r: Result, v: Variant): Uint8Array {
  const png = encodePng(image(r, v)) as Uint8Array;
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, `${name}.png`), png);
  return png;
}

/** One page run: no errors, every variant renders and repeats; HDDA on = off; determinism across loads. */
async function gate(page: Page, q: string, lods: string[], tag: string): Promise<Result> {
  const first = await run(page, q);
  const detail = detailOf(first);
  expect(first.error, detail).toBeUndefined();
  expect(first.uncaptured, detail).toEqual([]);
  expect(first.warnings, detail).toEqual([]);
  for (const v of first.variants) {
    expect(v.inPageIdentical, `${v.name}\n${detail}`).toBe(true);
    expect(v.nonBackground, `${v.name}\n${detail}`).toBeGreaterThan(1_000);
  }
  const dense = byName(first, "dense");
  for (const lod of lods) {
    // HDDA skipping: the skipped samples are exactly 0, so the image is the same within float noise.
    const cmp = compareImages(image(first, byName(first, lod)), image(first, byName(first, `${lod}-nohdda`)));
    console.log(`[k3-sparse] ${tag} ${lod}: HDDA on vs off PSNR ${cmp.psnr === Infinity ? "inf" : cmp.psnr.toFixed(2)}, max channel diff ${cmp.maxDiff}, ${cmp.above} px > 8/255`);
    expect(cmp.psnr, `${lod} ${JSON.stringify(cmp)}`).toBeGreaterThanOrEqual(60);
    expect(cmp.above).toBe(0);
    // K3.2d: the fast level-0 sampler (vdbSampleFast, one accessor per invocation) is bit-identical to the K3.2 one.
    const taps = first.variants.find((v) => v.name === `${lod}-nohdda-taps`);
    if (taps) {
      const same = compareImages(image(first, byName(first, `${lod}-nohdda`)), image(first, taps));
      console.log(`[k3-sparse] ${tag} ${lod}: K3.2d sampler vs K3.2 taps PSNR ${same.psnr === Infinity ? "inf" : same.psnr.toFixed(2)}, max channel diff ${same.maxDiff}`);
      expect(same.maxDiff, `${lod} fast vs taps ${JSON.stringify(same)}`).toBe(0);
    }
    const vs = compareImages(image(first, dense), image(first, byName(first, lod)));
    console.log(`[k3-sparse] ${tag} dense vs sparse '${lod}': PSNR ${vs.psnr.toFixed(2)} dB, ${vs.above} px > 8/255`);
  }
  const second = await run(page, q);
  expect(second.error).toBeUndefined();
  for (const v of first.variants) {
    const cmp = compareImages(image(first, v), image(second, byName(second, v.name)));
    expect(cmp.psnr, `${v.name} ${JSON.stringify(cmp)}`).toBe(Infinity);
  }
  for (const v of first.variants) save(`k3-sparse-${tag}-${v.name}`, first, v);
  return first;
}

test("K3.2 sparse source: identical voxels match dense (PSNR >= 38 dB), HDDA changes nothing, K3.2d sampler = K3.2 taps, deterministic, WGSL twins", async ({ page }) => {
  test.setTimeout(600_000);
  const first = await gate(page, "?asset=cloud04&variants=dense,none,none-nohdda,none-nohdda-taps,pyramid,pyramid-nohdda,pyramid-nohdda-taps&probe=1", ["none", "pyramid"], "cloud04");
  const ab = compareImages(image(first, byName(first, "dense")), image(first, byName(first, "none")));
  console.log(`[k3-sparse] ${first.adapter}: dense vs sparse 'none' (identical voxels) PSNR ${ab.psnr.toFixed(2)} dB, ${ab.above} px > 8/255`);
  expect(ab.psnr, JSON.stringify(ab)).toBeGreaterThanOrEqual(38);
  const p = first.probe!;
  const detail = detailOf(first);
  expect(p.count, detail).toBe(2048);
  expect(p.over, detail).toBe(0);
  // HDDA cell decisions could round differently on a cell face; a handful at most (0 on SwiftShader).
  expect(p.skipMismatch, detail).toBeLessThanOrEqual(8);
});

test("K3.2 sparse source, 'node-average' LOD: renders, HDDA changes nothing, deterministic (small froxel grid)", async ({ page }) => {
  test.setTimeout(600_000);
  // node-average reads 8 leaf averages per coarse tap (~20x the level-0 cost on SwiftShader): a smaller grid keeps the e2e short.
  await gate(page, "?asset=cloud04&variants=dense,node,node-nohdda&w=160&h=90&froxels=64x64x16&substeps=16&shadowSteps=16", ["node"], "cloud04-small");
});

test("K3.2 report: smoke.vdb / WDAS / bunny, flicker per LOD strategy (opt-in)", async ({ page }) => {
  test.skip(!REPORT, "report-only runs: K3_SPARSE_REPORT=1");
  test.setTimeout(3_600_000);
  const rows: Record<string, unknown>[] = [];
  const runs: [string, string][] = [
    ["bunny", "?asset=bunny&variants=dense,none,node,pyramid"],
    ["smoke", `?asset=smoke&variants=dense,none,node,pyramid,none-nohdda,pyramid-nohdda&orbit=8&orbitDeg=0.5&dist=2.5${SMALL}`],
    ["wdas", `?asset=wdas&variants=none,node,pyramid,none-nohdda,pyramid-nohdda&orbit=8&orbitDeg=0.5&dist=1.5${SMALL}`],
    ["cloud04-orbit", `?asset=cloud04&variants=dense,none,node,pyramid&orbit=8&orbitDeg=0.5&dist=3${SMALL}`],
  ];
  for (const [name, q] of runs) {
    const r = await run(page, `${q}${process.env.K3_SPARSE_TIMING === "1" ? "&timing=1" : ""}`, 3_000_000);
    if (r.error) {
      rows.push({ run: name, error: r.error.split("\n")[0] });
      continue;
    }
    const dense = r.variants.find((v) => v.name === "dense");
    for (const v of r.variants) {
      const vsDense = dense && v !== dense ? compareImages(image(r, dense), image(r, v)).psnr : null;
      const noh = r.variants.find((x) => x.name === `${v.name}-nohdda`);
      rows.push({
        run: name,
        variant: v.name,
        nonBackground: v.nonBackground,
        psnrVsDense: vsDense,
        psnrVsNoHdda: noh ? compareImages(image(r, v), image(r, noh)).psnr : null,
        gpuBytes: v.gpuBytes,
        lodLevels: v.lodLevels,
        flicker: v.flicker ? { meanDiff: v.flicker.meanDiff, temporalNoise: v.flicker.temporalNoise } : null,
        froxelsMs: v.froxelsMs ?? null,
        wallMs: v.wallMs ?? null,
        uncaptured: r.uncaptured.length,
      });
      save(`k3-sparse-${name}-${v.name}`, r, v);
    }
    rows.push({ run: name, grid: r.grid, dense: r.dense, adapter: r.adapter, timingsNote: r.timingsNote, warnings: r.warnings });
    mkdirSync(OURS, { recursive: true });
    writeFileSync(path.join(OURS, "k3-sparse-report.json"), JSON.stringify(rows, null, 2));
  }
  console.log(JSON.stringify(rows, null, 2));
});
