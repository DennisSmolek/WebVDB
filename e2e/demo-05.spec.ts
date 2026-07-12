import { expect, test } from "@playwright/test";

/**
 * Demo 05 — embergen-sequence: the Phase 7 gate (docs/PLAN.md, SPEC §3.5).
 *
 * Everything is generated in-page (no external fixture): synthetic animated fog
 * frames authored with `vdb-web-tools.buildFromDense`, written to `.nvdb`, and
 * served as Blob URLs so the real `NanoVDBFile.fromURL` fetch+parse path runs.
 * In `?test=1` the sequence scheduler is driven by a fixed timestep to play
 * exactly 6 frames, each swapped into ONE material via `rebindGrid`. The demo
 * asserts, in-page, that every frame advanced, all rebinds succeeded, there
 * were no stalls, and the rendered output actually changed across frames
 * (`framesDiffer`) — the real GPU proof that the buffer swap reached the shader.
 *
 * The golden is a single settled frame (the 6th), SwiftShader-specific and
 * byte-stable across runs (deterministic frames, fixed pose, jitter off);
 * regenerate on a matching runner with `--update-snapshots`.
 *
 * SwiftShader compiles the full pnanovdb library shader slowly (~10-20s cold)
 * on top of building + serializing 24 frames, so timeouts are generous.
 */
const DEMO_URL = "/src/demos/05-embergen-sequence/index.html";

test.use({ viewport: { width: 640, height: 480 } });
test.setTimeout(180_000);

test.describe("demo 05 — embergen-sequence", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("plays 6 frames stall-free with per-frame rebind, matches the golden", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto(`${DEMO_URL}?test=1`);
    await page.waitForFunction(() => window.__DEMO05__ !== undefined, undefined, { timeout: 150_000 });

    const state = await page.evaluate(() => window.__DEMO05__);
    const detail = JSON.stringify({ state, consoleErrors }, null, 2);

    expect(state, `window.__DEMO05__ should be set:\n${detail}`).toBeTruthy();
    expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
    expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
    expect(state!.framesPlayed, `should play exactly 6 frames:\n${detail}`).toBe(6);
    expect(state!.rebinds, `should rebind once per frame:\n${detail}`).toBe(6);
    expect(state!.stalls, `should play stall-free (frames preloaded):\n${detail}`).toBe(0);
    expect(state!.framesDiffer, `rendered output must change across rebinds:\n${detail}`).toBe(true);

    await expect(page).toHaveScreenshot("demo-05-sequence.png", { maxDiffPixelRatio: 0.02 });
  });
});
