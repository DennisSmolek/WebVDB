import { expect, test, type Page } from "@playwright/test";

/**
 * Gallery demo 11 — sky clouds. Structural (no golden): with and without the git-ignored `fixtures/gallery/`
 * assets (`?gallery=0` acts as if they were absent), the page loads in readback presentation, draws a non-flat frame
 * with no WebGPU validation error, and a sky control changes the image. Skips when no cloud is present at all.
 */
const DEMO = "/src/demos/11-sky-clouds/index.html";
const BASE_QUERY = "presentation=readback&still=1&quality=low&temporal=0";

interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  thumb: number[];
  spread: number;
}

test.use({ viewport: { width: 480, height: 320 } });
// Four volumes (one 24 M-voxel sparse grid) compile and march slowly on SwiftShader, more so beside other workers.
test.setTimeout(540_000);

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

async function open(page: Page, query: string): Promise<NonNullable<Window["__DEMO11__"]>> {
  await page.goto(`${DEMO}?${BASE_QUERY}&${query}`);
  await page.waitForFunction(() => window.__DEMO11__?.error !== undefined || (window.__DEMO11__?.ready === true && window.__DEMO11__.frames >= 3), undefined, { timeout: 480_000 });
  return (await page.evaluate(() => window.__DEMO11__))!;
}

async function settle(page: Page, n = 3): Promise<Snapshot> {
  const from = await page.evaluate(() => window.__DEMO11__!.frames);
  await page.waitForFunction(([f, k]) => window.__DEMO11__!.error !== undefined || window.__DEMO11__!.frames >= f + k, [from, n] as const, { timeout: 480_000 });
  const state = await page.evaluate(() => window.__DEMO11__!);
  expect(state.error, JSON.stringify(state)).toBeUndefined();
  return (await page.evaluate(() => window.__DEMO11_API__!.capture()))!;
}

const thumbDiff = (a: Snapshot, b: Snapshot): number => a.thumb.reduce((s, v, i) => s + Math.abs(v - b.thumb[i]!), 0) / a.thumb.length;
const noCloud = (state: NonNullable<Window["__DEMO11__"]>): boolean => !!state.error?.startsWith("No cloud to draw");

test.describe("demo 11 — sky clouds", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const gallery of [true, false]) {
    test(`loads and draws ${gallery ? "with" : "without"} the gallery assets, no validation errors`, async ({ page }) => {
      const errors = collectErrors(page);
      const state = await open(page, gallery ? "" : "gallery=0");
      test.skip(noCloud(state), "no cloud fixtures (pnpm fixtures:wdas, fixtures/gallery/CloudPackVDB_wvdb/)");
      const detail = JSON.stringify({ state, errors }, null, 2);
      expect(state.error, detail).toBeUndefined();
      expect(state.presentation, detail).toBe("readback");
      if (!gallery) {
        expect(state.sky, detail).toBe("procedural");
        expect(state.assets?.cloudPack, detail).toBe(0);
        expect(state.assets?.hdr, detail).toEqual([]);
        await expect(page.getByTestId("note"), detail).toContainText("procedural sky");
      }
      expect(state.volumes, detail).toBeGreaterThan(0);
      const shot = (await page.evaluate(() => window.__DEMO11_API__!.capture()))!;
      expect(shot.width * shot.height, detail).toBe(480 * 320);
      // Sky and clouds: not a flat frame.
      expect(shot.spread, detail).toBeGreaterThan(20);
      const after = (await page.evaluate(() => window.__DEMO11__))!;
      expect(after.uncaptured, detail).toEqual([]);
      expect(errors, detail).toEqual([]);
    });
  }

  test("sky controls change the image", async ({ page }) => {
    const errors = collectErrors(page);
    const state = await open(page, "gallery=0");
    test.skip(noCloud(state) || !state.assets?.wdas, "needs the WDAS grid (pnpm fixtures:wdas, then Fp8)");
    const detail = () => JSON.stringify({ state, errors }, null, 2);
    const base = await settle(page, 1);
    const changes: Record<string, number> = {};
    let previous = base;
    const step = async (name: string, act: () => Promise<void>, min: number): Promise<void> => {
      await act();
      const next = await settle(page);
      changes[name] = thumbDiff(previous, next);
      expect(changes[name], `${name}: ${JSON.stringify(changes)}\n${detail()}`).toBeGreaterThan(min);
      previous = next;
    };
    // The sun sinks: warmer, dimmer light and (procedural) a dusk sky.
    await step("sun elevation 8°", () => page.locator("#ctl-elevation").fill("8"), 1);
    await step("time of day: overcast", () => page.selectOption("#ctl-time", "cloudy"), 1);
    await step("scattering: single only", () => page.selectOption("#ctl-ms", "off"), 0.05);
    console.log(`demo 11 control diffs: ${JSON.stringify(changes)}`);
    expect((await page.evaluate(() => window.__DEMO11__))!.uncaptured, detail()).toEqual([]);
    expect(errors, detail()).toEqual([]);
  });
});
