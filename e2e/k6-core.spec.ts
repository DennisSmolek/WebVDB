import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

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
