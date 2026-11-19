import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

/**
 * Gallery demo 15 — full resolution vs 128³. Structural (no golden): with and without the gallery assets (`?gallery=0`),
 * the split view loads in readback presentation, both sides report their numbers (the worker's bake and error), the
 * frame is non-flat with no WebGPU validation error, and the divider and the dense resolution change the image.
 * smoke.vdb keeps SwiftShader fast; the WDAS cloud is the page's default on hardware.
 */
const DEMO = "/src/demos/15-full-res-vs-dense/index.html";
const BASE_QUERY = "presentation=readback&still=1&quality=low&temporal=0&asset=smoke";

interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  thumb: number[];
  spread: number;
}

test.use({ viewport: { width: 480, height: 320 } });
// Two full renders per frame on SwiftShader (one a 24 M-voxel or 1 M-voxel sparse grid): slow, more so beside other workers.
test.setTimeout(540_000);

async function open(page: Page, query: string): Promise<NonNullable<Window["__DEMO15__"]>> {
  await page.goto(`${DEMO}?${BASE_QUERY}&${query}`);
  await page.waitForFunction(() => window.__DEMO15__?.error !== undefined || (window.__DEMO15__?.ready === true && window.__DEMO15__.frames >= 3), undefined, { timeout: 480_000 });
  return (await page.evaluate(() => window.__DEMO15__))!;
}

async function settle(page: Page, n = 3): Promise<Snapshot> {
  await page.waitForFunction(() => window.__DEMO15__!.error !== undefined || (window.__DEMO15__!.ready && !window.__DEMO15__!.loading), undefined, { timeout: 480_000 });
  const from = await page.evaluate(() => window.__DEMO15__!.frames);
  await page.waitForFunction(([f, k]) => window.__DEMO15__!.error !== undefined || window.__DEMO15__!.frames >= f + k, [from, n] as const, { timeout: 480_000 });
  const state = await page.evaluate(() => window.__DEMO15__!);
  expect(state.error, JSON.stringify(state)).toBeUndefined();
  return (await page.evaluate(() => window.__DEMO15_API__!.capture()))!;
}

const thumbDiff = (a: Snapshot, b: Snapshot): number => a.thumb.reduce((s, v, i) => s + Math.abs(v - b.thumb[i]!), 0) / a.thumb.length;
const noAsset = (state: NonNullable<Window["__DEMO15__"]>): boolean => !!state.error?.startsWith("Neither smoke.vdb nor the WDAS grid");

test.describe("demo 15 — full res vs 128³", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const gallery of [true, false]) {
    test(`loads both sides ${gallery ? "with" : "without"} the gallery assets, no validation errors`, async ({ page }) => {
      const errors = collectErrors(page);
      const state = await open(page, gallery ? "" : "gallery=0");
      test.skip(noAsset(state), "no fixtures (pnpm fixtures:vdb-samples or pnpm fixtures:wdas)");
      const detail = () => JSON.stringify({ state, errors }, null, 2);
      expect(state.error, detail()).toBeUndefined();
      expect(state.presentation).toBe("readback");
      if (!gallery) expect(state.hdr, detail()).toBe(false);
      expect(state.resolution).toBe(128);
      expect(state.sparse?.gpuBytes, detail()).toBeGreaterThan(0);
      // The dense side is far smaller on the GPU than the full-resolution grid.
      expect(state.dense!.gpuBytes!, detail()).toBeLessThan(state.sparse!.gpuBytes!);
      const shot = (await page.evaluate(() => window.__DEMO15_API__!.capture()))!;
      expect(shot.width * shot.height).toBe(480 * 320);
      expect(shot.spread, detail()).toBeGreaterThan(20);
      // The worker's numbers arrive after the first frames: the dense bake's error is far above the full grid's.
      await page.waitForFunction(() => window.__DEMO15__!.dense?.errorPercent !== undefined && window.__DEMO15__!.sparse?.errorPercent !== undefined, undefined, { timeout: 480_000 });
      const measured = (await page.evaluate(() => window.__DEMO15__))!;
      expect(measured.sparse!.errorPercent!, JSON.stringify(measured)).toBeLessThan(1);
      expect(measured.dense!.errorPercent!, JSON.stringify(measured)).toBeGreaterThan(5);
      await expect(page.getByTestId("side-dense")).toContainText("%");
      expect(measured.uncaptured, detail()).toEqual([]);
      expect(errors, detail()).toEqual([]);
    });
  }

  test("the divider and the dense resolution change the image", async ({ page }) => {
    const errors = collectErrors(page);
    const state = await open(page, "gallery=0");
    test.skip(noAsset(state), "no fixtures");
    const detail = () => JSON.stringify({ state, errors }, null, 2);
    const base = await settle(page, 1);
    const changes: Record<string, number> = {};
    await page.locator("#ctl-divider").fill("0.15");
    const narrow = await settle(page);
    changes.divider = thumbDiff(base, narrow);
    expect((await page.evaluate(() => window.__DEMO15__!.divider))).toBeCloseTo(0.15, 2);
    await page.selectOption("#ctl-res", "64");
    const coarse = await settle(page);
    changes.res64 = thumbDiff(narrow, coarse);
    const after = (await page.evaluate(() => window.__DEMO15__))!;
    expect(after.resolution).toBe(64);
    // 64 texels on the longest side.
    expect(Math.max(...after.dense!.resolution.match(/\d+/g)!.map(Number)), JSON.stringify(after)).toBe(64);
    console.log(`demo 15 control diffs: ${JSON.stringify(changes)}`);
    expect(changes.divider, JSON.stringify(changes)).toBeGreaterThan(0.2);
    expect(changes.res64, JSON.stringify(changes)).toBeGreaterThan(0.05);
    expect(after.uncaptured, detail()).toEqual([]);
    expect(errors, detail()).toEqual([]);
  });
});
