import { expect, test, type Page } from "@playwright/test";

// K7.3: 'multi-scale' multiple scattering on raw WebGPU (examples/src/harness/k7-multi-scale.ts). The core mip
// build (vdbCreateMultiScaleMips) = its CPU twin (vdbFrameMultiScaleMips + round to nearest half), every texel of
// every mip within 1 f16 ulp (the f32 GPU sums vs the f64 CPU sums can only move a rounding by one step), on
// six seeded cases (edge zero / clamp, a stacked-grid region, channel selection into rgba16f and r16f, an
// rg32float source). The WGSL lookup (vdbFrameMultiScaleLight / Sample) = its twin over the same f16 texels on
// 2 × 1024 seeded queries. Tolerance: SwiftShader filters in f32 (Traps), so the gate is 1e-4 absolute on the
// scattered light and 1e-4 relative on samples; hardware that quantizes filter weights to 8 bits needs ~1e-2.
// A resize between frames keeps the old view valid for the frame in flight (the retire rule).
const PAGE_URL = "/src/harness/k7-multi-scale.html";

interface Compare {
  texels: number;
  exact: number;
  ulp1: number;
  over: number;
  maxUlp: number;
}

interface Result {
  done: boolean;
  error?: string;
  builds: { name: string; format: string; dimensions: number[][]; compare: Compare; peak: number }[];
  lookups: { name: string; queries: number; light: { maxAbs: number; maxRel: number; nonZero: number }; sample: { maxRel: number; outside: number } }[];
  retire?: { version: number; errors: string[] };
  uncaptured: string[];
}

async function run(page: Page): Promise<Result> {
  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __VDB_K7_MULTI_SCALE__?: { done: boolean } }).__VDB_K7_MULTI_SCALE__?.done === true, undefined, { timeout: 240_000 });
  return (await page.evaluate(() => (window as { __VDB_K7_MULTI_SCALE__?: unknown }).__VDB_K7_MULTI_SCALE__)) as Result;
}

test.describe.configure({ timeout: 300_000 });

test("K7.3 multi-scale: GPU mips = CPU twin (≤ 1 f16 ulp), GPU lookup = CPU twin, retire keeps the old view", async ({ page }) => {
  const r = await run(page);
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.builds.map((b) => [b.name, b.format]), detail).toEqual([
    ["rg-zero", "rg16float"],
    ["rg-clamp", "rg16float"],
    ["stacked", "rg16float"],
    ["rgba-select", "rgba16float"],
    ["r-single", "r16float"],
    ["rg32-source", "rg16float"],
  ]);
  for (const b of r.builds) {
    expect(b.compare.over, `${b.name}\n${detail}`).toBe(0);
    expect(b.compare.maxUlp, `${b.name}\n${detail}`).toBeLessThanOrEqual(1);
    expect(b.compare.exact + b.compare.ulp1, `${b.name}\n${detail}`).toBe(b.compare.texels);
    // Round to nearest even moves a half only at a near-tie: 1-ulp texels are rare (a truncating pack makes ~half).
    expect(b.compare.ulp1, `${b.name}\n${detail}`).toBeLessThanOrEqual(b.compare.texels * 0.01);
    expect(b.compare.texels, `${b.name}\n${detail}`).toBeGreaterThanOrEqual(512);
    expect(b.peak, `${b.name}: the chain is not empty\n${detail}`).toBeGreaterThan(0.1);
  }
  // Odd, non-cubic sizes halve exactly: 37x24x50 at 6 levels → level 1 32x16x32.
  expect(r.builds[0]!.dimensions[0], detail).toEqual([32, 16, 32]);
  expect(r.lookups.map((l) => l.name), detail).toEqual(["lookup rg-zero", "lookup stacked"]);
  for (const l of r.lookups) {
    expect(l.queries, detail).toBe(1024);
    expect(l.light.maxAbs, `${l.name}\n${detail}`).toBeLessThan(1e-4);
    expect(l.sample.maxRel, `${l.name}\n${detail}`).toBeLessThan(1e-4);
    // The queries exercise the method: most of them see scattered light, some samples fall outside the box.
    expect(l.light.nonZero, `${l.name}\n${detail}`).toBeGreaterThan(l.queries / 4);
    expect(l.sample.outside, `${l.name}\n${detail}`).toBeGreaterThan(0);
  }
  expect(r.retire, detail).toEqual({ version: 1, errors: [] });
});
