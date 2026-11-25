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
      expect(r.renderStats.covered, detail(r)).toBeGreaterThan(50);
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
