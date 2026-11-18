import { expect, test } from "@playwright/test";

/**
 * Demo 05 — embergen-sequence: the Phase 7 gate (docs/PLAN.md, SPEC §3.5).
 *
 * Everything is generated in-page (no external fixture): synthetic animated fog
 * frames authored with `@pmndrs/vdb/tools.vdbBuildFromDense`, written to `.nvdb`, and
 * served as Blob URLs so the real `NvdbFile.fromURL` fetch+parse path runs.
 * In `?test=1` the `VDBSequence` scheduler is driven by a fixed timestep to play
 * exactly 6 frames, each swapped into ONE pool grid via `pool.replace` (the grid
 * was added with `capacity` for the largest frame). The demo reports that every
 * frame advanced, there were no stalls, the rendered output changed across
 * frames (`framesDiffer`: the in-place write reached the shader), and the pool
 * never moved the grid (`versionChanges` 0, so the `VDBGridNode` never rebound).
 *
 * The golden is a single settled frame (the 6th), SwiftShader-specific and
 * byte-stable across runs (deterministic frames, fixed pose, jitter off);
 * regenerate on a matching runner with `--update-snapshots`.
 *
 * `presentation=readback`: headless SwiftShader can't present a WebGPU canvas
 * (DIVERGES V7), and test mode hashes the read-back pixels.
 *
 * SwiftShader compiles the shader slowly (~10-20s cold) on top of building +
 * serializing 24 frames, so timeouts are generous.
 */
const DEMO_URL = "/src/demos/05-embergen-sequence/index.html";

test.use({ viewport: { width: 640, height: 480 } });
test.setTimeout(180_000);

test.describe("demo 05 — embergen-sequence", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("plays 6 frames stall-free with in-place pool.replace, matches the golden", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto(`${DEMO_URL}?test=1&presentation=readback`);
    await page.waitForFunction(() => window.__DEMO05__ !== undefined, undefined, { timeout: 150_000 });

    const state = await page.evaluate(() => window.__DEMO05__);
    const detail = JSON.stringify({ state, consoleErrors }, null, 2);

    expect(state, `window.__DEMO05__ should be set:\n${detail}`).toBeTruthy();
    expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
    expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
    expect(state!.framesPlayed, `should play exactly 6 frames:\n${detail}`).toBe(6);
    expect(state!.swaps, `should pool.replace once per frame:\n${detail}`).toBe(6);
    expect(state!.versionChanges, `replace must stay in place (capacity reserved):\n${detail}`).toBe(0);
    expect(state!.nodeRebinds, `no growth, so the VDBGridNode never rebinds:\n${detail}`).toBe(0);
    expect(state!.stalls, `should play stall-free (frames preloaded):\n${detail}`).toBe(0);
    expect(state!.framesDiffer, `rendered output must change across rebinds:\n${detail}`).toBe(true);
    expect(state!.presentation, `CI must retain deterministic readback:\n${detail}`).toBe("readback");
    expect(state!.adapter, `canonical goldens require SwiftShader:\n${detail}`).toMatch(/swiftshader/i);

    await expect(page).toHaveScreenshot("demo-05-sequence.png", { maxDiffPixelRatio: 0.02 });
  });
});
