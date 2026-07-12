import { expect, test } from "@playwright/test";

/**
 * Demo 07 — builder: the other half of the Phase 5/8 demo gate ("dense ->
 * TS-build -> render round-trip", docs/PLAN.md). Everything is generated
 * in-page (no external fixture): a deterministic 96^3 dense field ->
 * `buildFromDense` -> `writeNvdb` -> re-parse via `NanoVDBFile` -> render,
 * with the round-trip proven in-page (`window.__DEMO07__`) before the
 * screenshot is taken.
 *
 * SwiftShader compiles the full pnanovdb library shader slowly (~10-20s
 * cold) on top of the dense-build + quantize-free verification pass, so
 * timeouts are generous.
 */
const DEMO_URL = "/src/demos/07-builder/index.html";

test.use({ viewport: { width: 640, height: 480 } });
test.setTimeout(120_000);

test.describe("demo 07 — builder", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("dense -> buildFromDense -> writeNvdb -> re-parse round trip, matches the golden", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto(`${DEMO_URL}?test=1`);
    await page.waitForFunction(() => window.__DEMO07__ !== undefined, undefined, { timeout: 100_000 });

    const state = await page.evaluate(() => window.__DEMO07__);
    const detail = JSON.stringify({ state, consoleErrors }, null, 2);

    expect(state, `window.__DEMO07__ should be set:\n${detail}`).toBeTruthy();
    expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
    expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
    expect(state!.roundTripOk, `re-parsed metadata should match the built grid:\n${detail}`).toBe(true);
    expect(state!.probesOk, `20 CPU probes should match the source dense values exactly:\n${detail}`).toBe(true);

    await expect(page).toHaveScreenshot("demo-07-builder.png", { maxDiffPixelRatio: 0.02 });
  });
});
