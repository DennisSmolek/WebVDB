import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages, diffHeatmap } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K6.1 (design §7.1): the core frame kernels on raw WebGPU (examples/src/harness/k6-core.ts), each against its
// CPU twin (tools/frame) within 2 f16 ulp; every probe's mutation must fail; every pipeline variant compiles;
// a second identical update builds nothing; two loads are deterministic.
const PAGE_URL = "/src/harness/k6-core.html";

interface Stats {
  compared: number;
  covered: number;
  over: number;
  maxUlp: number;
  maxAbs: number;
  bound: string;
  firstBad?: unknown;
}
interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  stats?: Stats;
  renderStats?: Stats;
  gridStats?: Stats;
  transmittanceStats?: Stats;
  frame?: { cascades: number; redraw: unknown };
  builds?: { first: number; second: number };
  compile?: { variants: number; errors: string[] };
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K6_CORE__?: { done: boolean } }).__VDB_K6_CORE__?.done === true, undefined, { timeout: 300_000 });
  return (await page.evaluate(() => (window as { __VDB_K6_CORE__?: unknown }).__VDB_K6_CORE__)) as Result;
}

const detail = (r: Result) => JSON.stringify(r, null, 2);

test("K6.1 every frame pipeline variant compiles", async ({ page }) => {
  test.setTimeout(300_000);
  const errors = collectErrors(page);
  const r = await run(page, "?probe=compile");
  expect(r.error, detail(r)).toBeUndefined();
  expect(r.compile!.errors, detail(r)).toEqual([]);
  expect(r.compile!.variants).toBeGreaterThanOrEqual(15);
  expect(r.uncaptured).toEqual([]);
  expect(errors).toEqual([]);
});

const PROBES: { name: string; query: string; minCovered: number; mutations: string[] }[] = [
  { name: "view chunks", query: "?probe=view", minCovered: 512, mutations: ["coverage"] },
  { name: "gather (mixture, one group)", query: "?probe=gather&mix=mixture&groups=1", minCovered: 300, mutations: ["mix", "phase", "extinction", "fade"] },
  { name: "gather (over, one group)", query: "?probe=gather&mix=over&groups=1", minCovered: 300, mutations: ["mix"] },
  { name: "gather (mixture, two groups)", query: "?probe=gather&mix=mixture&groups=2", minCovered: 300, mutations: ["mix"] },
  { name: "gather (over, two groups, no mips)", query: "?probe=gather&mix=over&groups=2&mips=0", minCovered: 300, mutations: ["extinction"] },
  { name: "resolve", query: "?probe=resolve&temporal=0", minCovered: 512, mutations: ["scan-stop"] },
  { name: "resolve (temporal)", query: "?probe=resolve&temporal=1", minCovered: 512, mutations: ["temporal-clamp", "scan-stop"] },
  { name: "integrate", query: "?probe=integrate", minCovered: 512, mutations: ["coverage", "taps"] },
  // K6.1d: the gather on the core sparse source (K6.7), volume A skipping empty space (hdda), B wrapped (no skip)
  { name: "gather sparse (lod none, one group, hdda)", query: "?probe=gather&source=sparse&lod=none&groups=1", minCovered: 200, mutations: ["skip", "extinction", "phase"] },
  { name: "gather sparse (node-average, two groups)", query: "?probe=gather&source=sparse&lod=node-average&groups=2", minCovered: 300, mutations: ["mix"] },
  { name: "gather sparse (coarse-pyramid, one group)", query: "?probe=gather&source=sparse&lod=coarse-pyramid&groups=1", minCovered: 300, mutations: ["extinction"] },
];

for (const p of PROBES) {
  test(`K6.1 probe: ${p.name} matches its CPU twin; mutations fail`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.stats!.compared, detail(r)).toBeGreaterThanOrEqual(512);
    expect(r.stats!.covered, detail(r)).toBeGreaterThanOrEqual(p.minCovered);
    expect(r.stats!.over, detail(r)).toBeLessThanOrEqual(Math.floor(r.stats!.compared * 0.001));
    if (r.renderStats) {
      expect(r.renderStats.over, detail(r)).toBe(0);
      expect(r.renderStats.covered, detail(r)).toBeGreaterThan(30);
    }
    console.log(`[k6-core] ${p.name} on ${r.adapter}: ${r.stats!.compared} texels, ${r.stats!.covered} non-zero, max ${r.stats!.maxUlp.toFixed(2)} f16 ulp, ${r.stats!.over} over`);
    for (const m of p.mutations) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad.stats!.over, `mutation ${m} must fail: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad.stats!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.2 (design §7.1): 'frame' lighting, each kernel against its twin from the read-back input of the kernel before it:
// the shadow grid (vdbFrameGridDensityVoxel), the transmittance (vdbFrameTransmittanceVoxel3d on the read-back grid) and
// the fused gather (his density + lighting twins on the read-back transmittance). `mutations` maps a twin term to the
// stats it must break. A second identical update skips the grid and the transmittance (his redraw signatures).
type StatsKey = "stats" | "gridStats" | "transmittanceStats";
const FRAME_PROBES: { name: string; query: string; cascades: number; mutations: Record<string, StatsKey> }[] = [
  {
    name: "dense, one group, one cascade",
    query: "?probe=frame",
    cascades: 1,
    mutations: { "grid-absorbance": "gridStats", "grid-lod": "gridStats", diffusion: "transmittanceStats", slots: "stats", sky: "stats", extinction: "stats" },
  },
  { name: "dense, two groups (partials), two cascades", query: "?probe=frame&groups=2&cascades=2", cascades: 2, mutations: { slots: "stats" } },
  { name: "sparse, one group", query: "?probe=frame&source=sparse", cascades: 1, mutations: { "grid-absorbance": "gridStats", diffusion: "transmittanceStats" } },
  { name: "sparse, two groups, single scattering", query: "?probe=frame&source=sparse&groups=2&ms=0", cascades: 1, mutations: { slots: "stats" } },
];

for (const p of FRAME_PROBES) {
  test(`K6.2 probe: 'frame' lighting (${p.name}) matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.frame!.cascades, detail(r)).toBe(p.cascades);
    const redraw = r.frame!.redraw as { shadowDensity: { drawn: number; skipped: number }; transmittance: { drawn: number; skipped: number } };
    expect(redraw.shadowDensity, detail(r)).toEqual({ drawn: 1, skipped: 1 });
    expect(redraw.transmittance, detail(r)).toEqual({ drawn: 1, skipped: 1 });
    for (const key of ["gridStats", "transmittanceStats", "stats"] as const) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "stats" ? 200 : 250);
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] frame ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

test("K6.1 a second identical update builds no pipeline or module; two loads agree", async ({ page }) => {
  test.setTimeout(600_000);
  const r = await run(page, "?probe=gather&swap=1");
  expect(r.error, detail(r)).toBeUndefined();
  expect(r.builds!.first, detail(r)).toBeGreaterThan(0);
  expect(r.builds!.second, detail(r)).toBe(0);
  const again = await run(page, "?probe=gather&swap=1");
  expect(again.stats, detail(again)).toEqual(r.stats);
});

// ---------------------------------------------------------------------------------------------------------------
// Gate 2 (design §7.2): his K3 Step 1 scenes through the k3-frame harness with `?backend=core` vs `?backend=tsl` on
// the same device: PSNR ≥ 45 dB and ≤ 0.2 % pixels > 4/255, with froxelMix 'mixture' and 'over'. The scenes need his
// CC BY-SA bunny (fixtures/k3/bunny_cloud.wvdb); full size (1280x720, 512x512x24): opt in with K6_CORE_AB=1.
// 'frame'-lighting scenes render with ?lighting=volume (K6.1 has 'volume' lighting only). PNGs, diff heatmaps and a
// JSON per scene go to artifacts/k6/ab/.
// ---------------------------------------------------------------------------------------------------------------
const AB_SCENES = [
  "step1-lighting-volume",
  "step1-lights-directional",
  "step1-multiple-scattering-off",
  "step1-volume-dir-ms-off",
  "step1-volume-dir-ms-off-isolated",
  "step1-volume-dir-ms-on",
  "step1-volume-dir-ms-on-isolated",
];
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const AB_DIR = path.join(REPO, "artifacts", "k6", "ab");

interface K3Result {
  done: boolean;
  error?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  uncaptured: string[];
  warnings: string[];
}

async function k3(page: Page, query: string): Promise<K3Result> {
  await page.goto(`/src/harness/k3-frame.html${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 600_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as K3Result;
}
const k3Image = (r: K3Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });
const k3Detail = (r: K3Result) => JSON.stringify({ ...r, rgba: undefined });

for (const scene of AB_SCENES) {
  for (const mix of ["mixture", "over"] as const) {
    test(`K6.1 A/B core vs TSL: ${scene} (${mix})`, async ({ page }) => {
      test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1 (needs fixtures/k3/bunny_cloud.wvdb)");
      test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing");
      test.setTimeout(1_200_000);
      const base = `?scene=${scene}&lighting=volume`;
      const tsl = await k3(page, `${base}&backend=tsl`);
      const core = await k3(page, `${base}&backend=core&froxelMix=${mix}`);
      expect(tsl.error, k3Detail(tsl)).toBeUndefined();
      expect(core.error, k3Detail(core)).toBeUndefined();
      expect(core.uncaptured, k3Detail(core)).toEqual([]);
      expect(core.inPageIdentical, k3Detail(core)).toBe(true);
      const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
      mkdirSync(AB_DIR, { recursive: true });
      writeFileSync(path.join(AB_DIR, `${scene}-tsl.png`), encodePng(k3Image(tsl)));
      writeFileSync(path.join(AB_DIR, `${scene}-core-${mix}.png`), encodePng(k3Image(core)));
      writeFileSync(path.join(AB_DIR, `${scene}-core-${mix}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
      writeFileSync(path.join(AB_DIR, `${scene}-${mix}.json`), JSON.stringify({ scene, mix, ...cmp, coreWarnings: core.warnings }, null, 2));
      if (mix === "mixture") {
        // Gate 3 (his captures): scripts/k3/compare.mjs --ours artifacts/k6/ours-{core,tsl} --reference fixtures/k3-reference/75e7345
        for (const [dir, r] of [["ours-core", core], ["ours-tsl", tsl]] as const) {
          mkdirSync(path.join(REPO, "artifacts", "k6", dir), { recursive: true });
          writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.png`), encodePng(k3Image(r)));
          writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
        }
      }
      console.log(`[k6-ab] ${scene} ${mix}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
      expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
    });
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K6.2 gate 2 (design §7.2): his K3 Step 3 'frame' lighting scenes through `?backend=core` vs `?backend=tsl` (same
// device, same bar: PSNR ≥ 45 dB, ≤ 0.2 % pixels > 4/255), dense and (K6.1d) sparse sources. The environment scene
// (`*-env-isolated`) needs K6.2b's GPU environment and is not here. Opt in with K6_CORE_AB=1; PNGs, heatmaps and JSON
// go to artifacts/k6/ab-frame/, and the mixture images to artifacts/k6/ours-{core,tsl}/ for scripts/k3/compare.mjs.
// ---------------------------------------------------------------------------------------------------------------
const AB_FRAME = [
  { scene: "step3-default-dir-isolated", source: "dense" },
  { scene: "step3-shadow-cascades-4-dir-isolated", source: "dense" },
  { scene: "step3-shadow-grid-high-dir-isolated", source: "dense" },
  { scene: "step3-two-bunnies-dir-isolated", source: "dense" },
  { scene: "step3-default-dir-isolated", source: "sparse" },
  { scene: "step3-two-bunnies-dir-isolated", source: "sparse" },
  { scene: "step1-volume-dir-ms-on-isolated", source: "sparse" },
] as const;
const AB_FRAME_DIR = path.join(REPO, "artifacts", "k6", "ab-frame");

for (const { scene, source } of AB_FRAME) {
  test(`K6.2 A/B core vs TSL: ${scene} (${source})`, async ({ page }) => {
    test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1 (needs fixtures/k3/bunny_cloud.wvdb)");
    test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing");
    test.setTimeout(1_800_000);
    const base = `?scene=${scene}&source=${source}`;
    const tsl = await k3(page, `${base}&backend=tsl`);
    const core = await k3(page, `${base}&backend=core`);
    expect(tsl.error, k3Detail(tsl)).toBeUndefined();
    expect(core.error, k3Detail(core)).toBeUndefined();
    expect(core.uncaptured, k3Detail(core)).toEqual([]);
    expect(core.inPageIdentical, k3Detail(core)).toBe(true);
    const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
    const name = `${scene}-${source}`;
    mkdirSync(AB_FRAME_DIR, { recursive: true });
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-tsl.png`), encodePng(k3Image(tsl)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-core.png`), encodePng(k3Image(core)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}.json`), JSON.stringify({ scene, source, ...cmp, coreWarnings: core.warnings }, null, 2));
    if (source === "dense") {
      for (const [dir, r] of [["ours-core", core], ["ours-tsl", tsl]] as const) {
        mkdirSync(path.join(REPO, "artifacts", "k6", dir), { recursive: true });
        writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.png`), encodePng(k3Image(r)));
        writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
      }
    }
    console.log(`[k6-ab-frame] ${name}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
    expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  });
}

// K6.1d (design §7.2): the WDAS cloud (fixtures/wdas/wdas_cloud_quarter_fp8.nvdb, Fp8, sparse at full resolution) through
// examples/src/harness/k3-timing.html `?image=1`, core vs TSL on the same device, 'volume' and 'frame' lighting, the
// sparse source ('coarse-pyramid', HDDA on), 1280x720 with his 512x512x24 froxels (at 640x360 with 160² froxels the
// silhouette band of V80 alone is 0.3 % of the pixels). Opt in with K6_CORE_AB=1; skipped without the fixture.
for (const lighting of ["volume", "frame"] as const) {
  test(`K6.1d A/B core vs TSL: WDAS cloud, sparse, ${lighting} lighting`, async ({ page }) => {
    test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1");
    test.skip(!existsSync(path.join(REPO, "fixtures", "wdas", "wdas_cloud_quarter_fp8.nvdb")), "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb missing");
    test.setTimeout(1_800_000);
    const shot = async (backend: string) => {
      await page.goto(`/src/harness/k3-timing.html?asset=wdas&source=sparse&lod=pyramid&hdda=1&lighting=${lighting}&w=1280&h=720&froxels=512x512x24&image=1&backend=${backend}`);
      await page.waitForFunction(() => (window as { __VDB_K3_TIMING__?: { done: boolean } }).__VDB_K3_TIMING__?.done === true, undefined, { timeout: 1_800_000 });
      return (await page.evaluate(() => (window as { __VDB_K3_TIMING__?: unknown }).__VDB_K3_TIMING__)) as K3Result;
    };
    const tsl = await shot("tsl");
    const core = await shot("core");
    expect(tsl.error, k3Detail(tsl)).toBeUndefined();
    expect(core.error, k3Detail(core)).toBeUndefined();
    expect(core.uncaptured, k3Detail(core)).toEqual([]);
    const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
    const name = `wdas-sparse-${lighting}`;
    mkdirSync(AB_FRAME_DIR, { recursive: true });
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-tsl.png`), encodePng(k3Image(tsl)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-core.png`), encodePng(k3Image(core)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}.json`), JSON.stringify({ lighting, ...cmp, coreWarnings: core.warnings }, null, 2));
    console.log(`[k6-ab-wdas] ${name}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
    expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  });
}
