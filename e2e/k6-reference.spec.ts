import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

// K6.R: the reference renderer (core `vdbCreateReferenceRendererGpu`) on raw WebGPU (examples/src/harness/k6-reference.ts):
// - analytic: Beer–Lambert (slab, ball), emission only, single scattering (slab, ball) against the quadrature, white
//   furnace, each per pixel within 4.5σ and unbiased over the image; every WGSL mutation fails some case;
// - twin: the GPU kernel against its CPU twin (same random stream) on the committed cloud (dense source) and on
//   synthetic grids (sparse source, 3 volumes); the dense render is bitwise deterministic;
// - K6.R1 features twin: per-cell majorants from the node stats, Mie RGB and a 3-lobe phase table, the blackbody colour
//   table, a point light with three's falloff, a spot light, a light-probe environment; each new WGSL mutation breaks it.
// Committed and synthetic data only: no fixtures, no skips. The high-sample report (scripts/k6/reference-report.mjs) is local.
const PAGE_URL = "/src/harness/k6-reference.html";

interface CaseResult {
  name: string;
  over: number;
  meanZ: number;
  passes: boolean;
}

interface TwinResult extends CaseResult {
  equalShare: number;
  covered: number;
  meanDepth: number;
  multipleShare: number;
  maxDepth: number;
  majorantViolations: number;
}

interface Result {
  done: boolean;
  error?: string;
  cases: CaseResult[];
  twins: TwinResult[];
  determinism?: { identical: boolean; values: number };
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string, timeout = 300_000): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K6_REFERENCE__?: { done: boolean } }).__VDB_K6_REFERENCE__?.done === true, undefined, { timeout });
  const r = (await page.evaluate(() => (window as { __VDB_K6_REFERENCE__?: unknown }).__VDB_K6_REFERENCE__)) as Result;
  // K6R_LOG=1: print the numbers (the handoff quotes them).
  if (process.env.K6R_LOG) console.log(`${query}\n${JSON.stringify({ cases: r.cases, twins: r.twins, determinism: r.determinism })}`);
  return r;
}

test.describe.configure({ mode: "serial", timeout: 400_000 });

test("K6.R reference kernel = analytic cases (Beer–Lambert, emission, single scattering, white furnace)", async ({ page }) => {
  const errors = collectErrors(page);
  const r = await run(page, "?case=analytic");
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.cases.map((c) => c.name), detail).toEqual([
    "beer-lambert box",
    "beer-lambert sphere",
    "emission only",
    "single scattering box",
    "single scattering sphere",
    "white furnace",
  ]);
  for (const c of r.cases) expect(c.passes, `${c.name}\n${detail}`).toBe(true);
  expect(errors, detail).toEqual([]);
});

for (const mutate of ["ratio-weight", "delta-accept", "phase-norm", "nee-transmittance", "emission-albedo", "escape-depth"]) {
  test(`K6.R fail-before: mutation '${mutate}' fails an analytic case`, async ({ page }) => {
    const r = await run(page, `?case=analytic&mutate=${mutate}`);
    const detail = JSON.stringify(r, null, 2);
    expect(r.error, detail).toBeUndefined();
    expect(r.cases.length, detail).toBe(6);
    expect(r.cases.some((c) => !c.passes), detail).toBe(true);
  });
}

test("K6.R reference kernel = CPU twin (dense cloud, sparse 3 volumes, K6.R1 features) and deterministic", async ({ page }) => {
  const errors = collectErrors(page);
  const r = await run(page, "?case=twin&determinism=1");
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.determinism?.identical, detail).toBe(true);
  expect(r.twins.length, detail).toBe(3);
  for (const t of r.twins) {
    expect(t.passes, `${t.name}\n${detail}`).toBe(true);
    // The scenes exercise the medium (volume pixels, multiple scattering) and the shared stream (most values equal).
    expect(t.covered, `${t.name}\n${detail}`).toBeGreaterThan(20);
    expect(t.multipleShare, `${t.name}\n${detail}`).toBeGreaterThan(0.05);
    expect(t.maxDepth, `${t.name}\n${detail}`).toBeGreaterThan(4);
    expect(t.majorantViolations, `${t.name}\n${detail}`).toBe(0);
    expect(t.equalShare, `${t.name}\n${detail}`).toBeGreaterThan(0.5);
  }
  expect(errors, detail).toEqual([]);
});

test("K6.R fail-before: a mutated kernel no longer matches its CPU twin", async ({ page }) => {
  const r = await run(page, "?case=twin&source=dense&mutate=phase-norm");
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.twins[0]?.passes, detail).toBe(false);
});

test("K6.R1 phase variant of the features twin (environment through the 3-lobe table) = CPU twin", async ({ page }) => {
  const r = await run(page, "?case=twin&source=features&variant=phase&batches=32");
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.twins[0]?.passes, detail).toBe(true);
});

for (const mutate of ["majorant-cell", "phase-table-weight", "falloff"]) {
  test(`K6.R1 fail-before: mutation '${mutate}' breaks the features twin`, async ({ page }) => {
    const r = await run(page, `?case=twin&source=features${mutate === "phase-table-weight" ? "&batches=64&variant=phase" : "&batches=32"}&mutate=${mutate}`);
    const detail = JSON.stringify(r, null, 2);
    expect(r.error, detail).toBeUndefined();
    expect(r.twins[0]?.passes, detail).toBe(false);
  });
}
