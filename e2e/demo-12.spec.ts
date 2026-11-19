import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

/**
 * Gallery demo 12 — the EmberGen explosion as a sequence in one dense source, with the emission light on the ground
 * and props. Readback presentation, no camera motion, paused: the page loads and draws a non-trivial frame with no
 * WebGPU validation error, and scrubbing the sequence changes the image. Runs with the git-ignored gallery assets
 * (HDR sky, car.glb) and without them (`?gallery=0`: procedural sky, primitive props), so it passes either way.
 */
const DEMO = "/src/demos/12-explosion/index.html";
const BASE = "presentation=readback&still=1&play=0&quality=low&frame=30";

interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  thumb: number[];
}

test.use({ viewport: { width: 480, height: 320 } });
test.setTimeout(300_000);

const thumbDiff = (a: Snapshot, b: Snapshot): number => a.thumb.reduce((s, v, i) => s + Math.abs(v - b.thumb[i]!), 0) / a.thumb.length;
const thumbSpread = (s: Snapshot): number => Math.max(...s.thumb) - Math.min(...s.thumb);

/** Wait until the sequence shows `frame` and `n` more frames were drawn, then summarize the last one. */
async function settleOn(page: Page, frame: number, n = 3): Promise<Snapshot> {
  await page.waitForFunction((f) => window.__DEMO12__?.error !== undefined || (window.__DEMO12__?.ready === true && window.__DEMO12__.frame === f), frame, { timeout: 240_000 });
  const from = await page.evaluate(() => window.__DEMO12__!.frames);
  await page.waitForFunction(([f, k]) => window.__DEMO12__!.error !== undefined || window.__DEMO12__!.frames >= f + k, [from, n] as const, { timeout: 240_000 });
  const state = await page.evaluate(() => window.__DEMO12__!);
  expect(state.error, JSON.stringify(state)).toBeUndefined();
  return (await page.evaluate(() => window.__DEMO12_API__!.capture()))!;
}

test.describe("demo 12 — gallery: explosion", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const gallery of [true, false]) {
    test(`draws the explosion and a frame scrub changes the image (gallery assets ${gallery ? "on" : "off"})`, async ({ page }) => {
      const frames = await page.request.get("/fixtures/k3/ground_explosion/frames.json");
      test.skip(!frames.ok(), "fixtures/k3/ground_explosion/ missing (EmberGen, CC0: copy it from a mjurczyk/openvdb clone)");
      const carPresent = (await page.request.head("/fixtures/gallery/car.glb")).ok();
      const errors = collectErrors(page);
      await page.goto(`${DEMO}?${BASE}${gallery ? "" : "&gallery=0"}`);
      const first = await settleOn(page, 30);
      const state = (await page.evaluate(() => window.__DEMO12__))!;
      const detail = () => JSON.stringify({ state, errors }, null, 2);
      expect(state.presentation, detail()).toBe("readback");
      expect(state.frameCount, detail()).toBe(200);
      if (!gallery) {
        expect(state.props, detail()).toBe("primitives");
        expect(state.sky, detail()).toBe("procedural");
        expect(state.missing, detail()).toEqual(expect.arrayContaining(["car.glb", "sky-night-HDR.jpg"]));
        await expect(page.getByTestId("note")).toContainText("car.glb");
      } else if (carPresent) {
        expect(state.props, detail()).toBe("car.glb");
      }

      // Something non-trivial: a full-size frame with structure (sky, ground, the volume, its glow), not flat or black.
      expect(first.width * first.height, detail()).toBe(480 * 320);
      expect(thumbSpread(first), detail()).toBeGreaterThan(20);
      expect(Math.max(...first.mean), detail()).toBeGreaterThan(4);

      // Scrubbing swaps another frame into the same source: the image changes.
      await page.locator("#ctl-frame").fill("70");
      const later = await settleOn(page, 70);
      const d = thumbDiff(first, later);
      console.log(`demo 12 (gallery ${gallery ? "on" : "off"}): frame 30 -> 70 thumb diff ${d.toFixed(2)}`);
      expect(d, detail()).toBeGreaterThan(1);

      const after = (await page.evaluate(() => window.__DEMO12__))!;
      expect(after.uncaptured, detail()).toEqual([]);
      expect(errors, detail()).toEqual([]);
    });
  }
});
