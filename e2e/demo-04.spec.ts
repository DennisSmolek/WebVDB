import { expect, test } from "@playwright/test";

/**
 * Demo 04 — atlas-fallback: the Phase 4 gate (SPEC §3.3/§5, docs/PLAN.md).
 *
 * Loads `sphere_fog_fp8.nvdb` (atlas texture path) and `box_fog_float.nvdb`
 * (gridStats/valueTransform verification path), asserts
 * `window.__DEMO04__.{ready,statsOk,transformOk}`, and pins one golden
 * screenshot of the atlas render (deterministic `?test=1` mode, fixed pose,
 * one settled frame — same recipe as `demo-02.spec.ts`).
 *
 * SwiftShader compiles the shader + runs the compute passes slowly (cold
 * compile ~10-20s plus a ~200K-voxel dense-bbox gridStats/valueTransform
 * pass), so timeouts are generous.
 */

const ATLAS_FIXTURE = "/fixtures/primitives/sphere_fog_fp8.nvdb";
const STATS_FIXTURE = "/fixtures/primitives/box_fog_float.nvdb";
const STATS_SIDECAR = "/fixtures/primitives/box_fog_float.sidecar.json";
const DEMO_URL = "/src/demos/04-atlas-fallback/index.html";

test.use({ viewport: { width: 640, height: 480 } });
test.setTimeout(180_000);

test.describe("demo 04 — atlas-fallback (compute toolset)", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("gridStats/valueTransform verify against CPU truth, and the atlas render matches the golden", async ({
    page,
  }) => {
    const [atlas, stats, sidecar] = await Promise.all([
      page.request.get(ATLAS_FIXTURE),
      page.request.get(STATS_FIXTURE),
      page.request.get(STATS_SIDECAR),
    ]);
    test.skip(
      !atlas.ok() || !stats.ok() || !sidecar.ok(),
      `fixtures missing (atlas ${atlas.status()}, stats ${stats.status()}, sidecar ${sidecar.status()})`,
    );

    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto(`${DEMO_URL}?test=1`);
    await page.waitForFunction(() => window.__DEMO04__ !== undefined, undefined, { timeout: 150_000 });

    const state = await page.evaluate(() => window.__DEMO04__);
    const detail = JSON.stringify({ state, consoleErrors }, null, 2);

    expect(state, `window.__DEMO04__ should be set:\n${detail}`).toBeTruthy();
    expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
    expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
    expect(state!.statsOk, `gridStats should match the CPU/sidecar truth:\n${detail}`).toBe(true);
    expect(state!.transformOk, `valueTransform should match 2x the sidecar values:\n${detail}`).toBe(true);

    await expect(page).toHaveScreenshot("demo-04-atlas.png", { maxDiffPixelRatio: 0.02 });
  });
});
