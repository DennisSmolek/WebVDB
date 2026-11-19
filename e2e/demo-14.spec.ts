import { expect, test, type Page } from "@playwright/test";

/**
 * Demo 14 — point grids. The light `sphere_points.vdb` (9,802 points) keeps the runtime low; the 9.5 M-point
 * waterfall is a hardware showcase and isn't fetched here. The sphere must decode in the worker, draw in readback
 * presentation with no WebGPU validation error, and colour, size and density changes must change the image.
 * Demo 14 loads nothing from `fixtures/gallery/`; the test runs with `?gallery=0` and without it.
 */
const DEMO = "/src/demos/14-points/index.html";
const BASE = "presentation=readback&still=1&model=sphere";

interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  thumb: number[];
}

test.use({ viewport: { width: 480, height: 320 } });
test.setTimeout(240_000);

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

async function settle(page: Page, n = 3): Promise<Snapshot> {
  const from = await page.evaluate(() => window.__DEMO14__!.frames);
  await page.waitForFunction(([f, k]) => window.__DEMO14__!.error !== undefined || window.__DEMO14__!.frames >= f + k, [from, n] as const, { timeout: 120_000 });
  expect((await page.evaluate(() => window.__DEMO14__!)).error).toBeUndefined();
  return (await page.evaluate(() => window.__DEMO14_API__!.capture()))!;
}

const thumbDiff = (a: Snapshot, b: Snapshot): number => a.thumb.reduce((s, v, i) => s + Math.abs(v - b.thumb[i]!), 0) / a.thumb.length;

test.describe("demo 14 — point grids", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const gallery of ["gallery=0", "gallery=1"]) {
    test(`sphere_points (${gallery}): decodes, draws, controls change the image, no validation errors`, async ({ page }) => {
      const errors = collectErrors(page);
      const probe = await page.request.head("/fixtures/vdb-samples/sphere_points.vdb");
      test.skip(!probe.ok(), "fixtures/vdb-samples/sphere_points.vdb missing (copied from a mjurczyk/openvdb clone)");
      await page.goto(`${DEMO}?${BASE}&${gallery}`);
      await page.waitForFunction(() => window.__DEMO14__?.error !== undefined || window.__DEMO14__?.ready === true, undefined, { timeout: 120_000 });
      const state = (await page.evaluate(() => window.__DEMO14__))!;
      const detail = () => JSON.stringify({ state, errors }, null, 2);
      expect(state.error, detail()).toBeUndefined();
      expect(state.presentation).toBe("readback");
      expect(state.info.model).toBe("sphere");
      expect(state.info.count, detail()).toBe(9_802);
      expect(state.info.drawn, detail()).toBe(9_802);

      const base = await settle(page, 1);
      expect(base.width * base.height).toBe(480 * 320);
      const covered = await page.evaluate(() => window.__DEMO14_API__!.coverage());
      expect(covered, detail()).toBeGreaterThan(480 * 320 * 0.1);
      expect(covered, detail()).toBeLessThan(480 * 320 * 0.9);

      await page.selectOption("#ctl-color", "uniform");
      const uniform = await settle(page);
      await page.locator("#ctl-size").fill("2.5");
      const big = await settle(page);
      const coveredBig = await page.evaluate(() => window.__DEMO14_API__!.coverage());
      await page.selectOption("#ctl-density", "10");
      const sparse = await settle(page);
      const drawn10 = (await page.evaluate(() => window.__DEMO14__!.info.drawn)) as number;
      const coveredSparse = await page.evaluate(() => window.__DEMO14_API__!.coverage());
      const d = { color: thumbDiff(base, uniform), size: thumbDiff(uniform, big), density: thumbDiff(big, sparse), covered, coveredBig, coveredSparse, drawn10 };
      console.log(`demo 14: ${JSON.stringify(d)}`);
      expect(d.color, detail()).toBeGreaterThan(1);
      expect(d.size, detail()).toBeGreaterThan(1);
      expect(d.density, detail()).toBeGreaterThan(1);
      expect(coveredBig, detail()).toBeGreaterThan(covered);
      // Every 10th point (981 of 9,802), each drawn larger by cbrt(10) so the sphere still reads.
      expect(drawn10).toBe(981);
      expect(coveredSparse, detail()).toBeGreaterThan(480 * 320 * 0.1);

      const after = (await page.evaluate(() => window.__DEMO14__))!;
      expect(after.uncaptured, detail()).toEqual([]);
      expect(errors, detail()).toEqual([]);
    });
  }
});

declare global {
  interface Window {
    __DEMO14__?: { ready: boolean; error?: string; frames: number; presentation: string; uncaptured: string[]; info: Record<string, unknown> };
    __DEMO14_API__?: { capture(): Snapshot | null; coverage(): Promise<number> };
  }
}
