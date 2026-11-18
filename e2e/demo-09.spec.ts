import { expect, test } from "@playwright/test";

/**
 * Demo 09 — coarse-first streaming (K4.20): smoke.vdb is baked in the page to `[density.lod48, density]` (ZIP),
 * streamed through `vdbStreamCoarseFirst` into a real pool, shown as a preview and then swapped in place for the
 * full grid. In `?test=1` the stream is held after each grid until its frame is read back, so both phases are drawn.
 *
 * The golden is the final frame (the full grid, 480x360, jitter off). It was generated on macOS headless Chromium /
 * SwiftShader; like every golden here it must be confirmed (or regenerated with --update-snapshots) on the Linux CI
 * runner, which is what `snapshotPathTemplate` names (`*-chromium-linux.png`).
 */
const SOURCE = "/fixtures/vdb-samples/smoke.vdb";
const DEMO_URL = "/src/demos/09-coarse-first/index.html";
const GATE = Math.floor(14.8 * 1024); // his smoke.wvdb needs 15,155 B before LOD0 can draw

test.use({ viewport: { width: 480, height: 360 } });
test.setTimeout(240_000);

test.describe("demo 09 — coarse-first streaming", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("preview first, then the full grid in place; the final frame matches the golden", async ({ page }) => {
    const fixture = await page.request.get(SOURCE);
    test.skip(!fixture.ok(), `fixtures missing (${SOURCE} -> ${fixture.status()})`);

    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    await page.goto(`${DEMO_URL}?test=1`);
    await page.waitForFunction(() => window.__DEMO09__ !== undefined && (window.__DEMO09__.ready || window.__DEMO09__.error !== undefined), undefined, {
      timeout: 200_000,
    });
    const state = await page.evaluate(() => window.__DEMO09__);
    const detail = JSON.stringify({ state, consoleErrors }, null, 2);
    expect(state!.error, detail).toBeUndefined();
    expect(state!.ready, detail).toBe(true);
    expect(state!.adapter, `canonical goldens require SwiftShader:\n${detail}`).toMatch(/swiftshader/i);
    expect(state!.presentation, detail).toBe("readback");

    // Two phases, in order, the preview first.
    const phases = state!.phases!;
    expect(phases.map((p) => [p.phase, p.grid]), detail).toEqual([
      ["preview", "density.lod48"],
      ["full", "density"],
    ]);
    const [preview, full] = phases as [(typeof phases)[number], (typeof phases)[number]];

    // First paint within his 14.8 KiB (ZIP inside the file, no HTTP compression; whole 1 KiB chunks).
    expect(preview.transferBytes, detail).toBeGreaterThan(0);
    expect(preview.transferBytes, detail).toBeLessThanOrEqual(GATE);
    expect(full.transferBytes, detail).toBeGreaterThan(preview.transferBytes * 10);
    expect(full.transferBytes, detail).toBeLessThanOrEqual(state!.totalBytes!);
    expect(preview.voxelRatio, detail).toBeGreaterThan(1);
    expect(preview.densityScale, detail).toBeCloseTo(4 * preview.voxelRatio, 6);
    expect(full.voxelRatio, detail).toBe(1);
    expect(full.densityScale, detail).toBe(4);

    // Both phases draw the cloud, and the swap was in place (same id, no pool growth).
    expect(preview.nonBackground, detail).toBeGreaterThan(2_000);
    expect(full.nonBackground, detail).toBeGreaterThan(2_000);
    expect(state!.sameId, detail).toBe(true);
    expect(state!.inPlace, detail).toBe(true);

    // The counter reached the end of the file.
    expect(state!.transferred, detail).toBe(state!.totalBytes);
    await expect(page.getByTestId("phase")).toHaveText("FULL · done");
    expect(consoleErrors, detail).toEqual([]);

    // Golden: the final frame (the panel is hidden in test mode, the 2D canvas is the whole page).
    await expect(page).toHaveScreenshot("demo-09-final.png", { maxDiffPixelRatio: 0.02 });
  });
});
