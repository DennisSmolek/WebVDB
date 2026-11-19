import { expect, test, type Page } from "@playwright/test";

/**
 * Demo 10 — the froxel renderer showcase. Too many settings for a golden: each available scene must load
 * in readback presentation and draw something non-trivial with no WebGPU validation error, and live
 * controls must change the image. Scenes whose fixtures are absent are skipped (the page names the
 * fixture script for them; `state.missing`).
 */
const DEMO = "/src/demos/10-frame-renderer/index.html";
// The panel stays: the controls test drives its inputs, and captures read the frame, not the page.
const BASE_QUERY = "presentation=readback&quality=low&play=0";
const SCENES = ["wdas", "smoke", "bunny", "explosion", "teapot", "points"] as const;

interface Snapshot {
  width: number;
  height: number;
  nonBackground: number;
  mean: [number, number, number];
  thumb: number[];
}

test.use({ viewport: { width: 480, height: 320 } });
test.setTimeout(300_000);

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

async function open(page: Page, query: string): Promise<NonNullable<Window["__DEMO10__"]>> {
  await page.goto(`${DEMO}?${BASE_QUERY}&${query}`);
  await page.waitForFunction(() => window.__DEMO10__?.error !== undefined || (window.__DEMO10__?.ready === true && window.__DEMO10__.frames >= 3), undefined, { timeout: 240_000 });
  return (await page.evaluate(() => window.__DEMO10__))!;
}

/** Wait for `n` more frames after a change (and the rebuild, if any), then summarize the last one. */
async function settle(page: Page, n = 3): Promise<Snapshot> {
  const from = await page.evaluate(() => window.__DEMO10__!.frames);
  await page.waitForFunction(
    ([f, k]) => window.__DEMO10__!.error !== undefined || (window.__DEMO10__!.ready && !window.__DEMO10__!.loading && window.__DEMO10__!.frames >= f + k),
    [from, n] as const,
    { timeout: 240_000 },
  );
  const state = await page.evaluate(() => window.__DEMO10__!);
  expect(state.error, JSON.stringify(state)).toBeUndefined();
  return (await page.evaluate(() => window.__DEMO10_API__!.capture()))!;
}

/** Mean absolute difference of two grey thumbnails (0–255). */
const thumbDiff = (a: Snapshot, b: Snapshot): number => a.thumb.reduce((s, v, i) => s + Math.abs(v - b.thumb[i]!), 0) / a.thumb.length;

/** Spread of a thumbnail: a flat image (nothing drawn, or all black) has none. */
const thumbSpread = (s: Snapshot): number => Math.max(...s.thumb) - Math.min(...s.thumb);

test.describe("demo 10 — froxel renderer showcase", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const id of SCENES) {
    test(`scene ${id}: loads and draws, no validation errors`, async ({ page }) => {
      const errors = collectErrors(page);
      const state = await open(page, `scene=${id}&stage=0`);
      const detail = JSON.stringify({ state, errors }, null, 2);
      test.skip(!state.available?.includes(id), `fixtures missing for ${id}: ${state.missing?.find((m) => m.id === id)?.hint ?? "?"}`);
      expect(state.error, detail).toBeUndefined();
      expect(state.scene, detail).toBe(id);
      expect(state.presentation, detail).toBe("readback");
      expect(state.info, detail).toBeTruthy();

      const shot = (await page.evaluate(() => window.__DEMO10_API__!.capture()))!;
      expect(shot.width * shot.height, detail).toBe(480 * 320);
      // The object covers a few thousand pixels against the plain sky (no stage), and is not flat.
      expect(shot.nonBackground, detail).toBeGreaterThan(2_000);
      expect(shot.nonBackground, detail).toBeLessThan(shot.width * shot.height * 0.8);
      expect(thumbSpread(shot), detail).toBeGreaterThan(20);

      const after = await page.evaluate(() => window.__DEMO10__!);
      expect(after.uncaptured, detail).toEqual([]);
      expect(errors, detail).toEqual([]);
    });
  }

  test("live controls change the image", async ({ page }) => {
    const errors = collectErrors(page);
    // smoke has both source kinds; else the first froxel scene there is
    let state = await open(page, "scene=smoke&stage=0");
    const froxel = ["smoke", "bunny", "wdas", "explosion"].find((s) => state.available?.includes(s));
    test.skip(!froxel, "no froxel scene's fixtures are present");
    if (state.scene !== froxel) state = await open(page, `scene=${froxel}&stage=0`);
    const detail = () => JSON.stringify({ state, errors }, null, 2);
    expect(state.error, detail()).toBeUndefined();

    const base = await settle(page, 1);
    const changes: Record<string, number> = {};
    let previous = base;
    const step = async (name: string, act: () => Promise<void>, min: number): Promise<void> => {
      await act();
      const next = await settle(page);
      changes[name] = thumbDiff(previous, next);
      expect(changes[name], `${name}: ${JSON.stringify(changes)}\n${detail()}`).toBeGreaterThan(min);
      expect(next.nonBackground, `${name}\n${detail()}`).toBeGreaterThan(1_000);
      previous = next;
    };

    // Lighting path, scattering and density are settings / props read every frame. Density on smoke's sparse
    // source also guards the demo's workaround for TASKS K5.9a (densityScale needs updateSource() there).
    await step("lighting volume", () => page.selectOption("#ctl-lighting", "volume"), 0.5);
    await step("multiple scattering off", () => page.selectOption("#ctl-ms", "off"), 0.5);
    await step("density x3", () => page.locator("#ctl-density").fill("3"), 0.3);
    await step("phase isotropic", () => page.selectOption("#ctl-phase", "isotropic"), 0.2);
    // A source rebuild (a new volume) where the scene has both kinds.
    state = (await page.evaluate(() => window.__DEMO10__))!;
    if (state.scene === "smoke" || state.scene === "bunny") {
      const other = state.sourceKind === "dense" ? "sparse" : "dense";
      await step(`source ${other}`, () => page.selectOption("#ctl-source", other), 0.1);
      expect((await page.evaluate(() => window.__DEMO10__))!.sourceKind).toBe(other);
    }
    // The stage, then the volume's shadows on it ('frame' lighting draws the shadow grid they come from).
    await page.selectOption("#ctl-lighting", "frame");
    await step("stage on", () => page.locator("#ctl-stage").check(), 5);
    await step("mesh shadows on", () => page.locator("#ctl-meshShadows").check(), 0.2);

    state = (await page.evaluate(() => window.__DEMO10__))!;
    console.log(`demo 10 control diffs (${state.scene}): ${JSON.stringify(changes)}`);
    expect(state.uncaptured, detail()).toEqual([]);
    expect(errors, detail()).toEqual([]);
  });
});
