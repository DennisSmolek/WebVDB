import { expect, test } from "@playwright/test";

/**
 * Demo 06 — explorer: half of the Phase 5/8 demo gate ("drag-drop a .vdb and
 * render it", docs/PLAN.md). Two deterministic navigations exercise both
 * load routes — `?sample=nvdb` (NanoVDBFile) and `?sample=vdb` (parseVdb ->
 * buildFromVdb) — asserting the metadata/inspect/histogram/slice panels all
 * populated (`window.__DEMO06__`). One golden image pins the nvdb render
 * (leaf-bbox wireframe over the raymarched volume).
 *
 * SwiftShader compiles the full pnanovdb library shader slowly (~10-20s
 * cold), and this demo adds a `gridStats` compute pass plus a CPU tree walk
 * (leaf-bbox enumeration) and a CPU slice scan on top of demo 02's baseline
 * — so timeouts are generous.
 */
const DEMO_URL = "/src/demos/06-explorer/index.html";
const NVDB_FIXTURE = "/fixtures/primitives/sphere_fog_fp8.nvdb";
const VDB_FIXTURE = "/fixtures/vdb-samples/smoke.vdb";

test.use({ viewport: { width: 640, height: 480 } });
test.setTimeout(120_000);

test.describe("demo 06 — explorer", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("nvdb sample: ready with populated panels, matches the golden", async ({ page }) => {
    const fixture = await page.request.get(NVDB_FIXTURE);
    test.skip(!fixture.ok(), `fixtures missing (${NVDB_FIXTURE} -> ${fixture.status()})`);

    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto(`${DEMO_URL}?test=1&sample=nvdb`);
    await page.waitForFunction(() => window.__DEMO06__ !== undefined, undefined, { timeout: 100_000 });

    const state = await page.evaluate(() => window.__DEMO06__);
    const detail = JSON.stringify({ state, consoleErrors }, null, 2);

    expect(state, `window.__DEMO06__ should be set:\n${detail}`).toBeTruthy();
    expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
    expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
    expect(state!.source, `expected the .nvdb route:\n${detail}`).toBe("nvdb");
    expect(state!.voxelCount ?? 0, `voxelCount should be > 0:\n${detail}`).toBeGreaterThan(0);
    expect(state!.histogramNonEmpty, `histogram should be non-empty:\n${detail}`).toBe(true);
    expect(state!.sliceNonEmpty, `slice should be non-empty:\n${detail}`).toBe(true);

    await expect(page).toHaveScreenshot("demo-06-nvdb.png", { maxDiffPixelRatio: 0.02 });
  });

  test("vdb sample: ready with populated panels (.vdb -> parseVdb -> buildFromVdb route)", async ({ page }) => {
    const fixture = await page.request.get(VDB_FIXTURE);
    test.skip(!fixture.ok(), `fixtures missing (${VDB_FIXTURE} -> ${fixture.status()})`);

    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto(`${DEMO_URL}?test=1&sample=vdb`);
    await page.waitForFunction(() => window.__DEMO06__ !== undefined, undefined, { timeout: 100_000 });

    const state = await page.evaluate(() => window.__DEMO06__);
    const detail = JSON.stringify({ state, consoleErrors }, null, 2);

    expect(state, `window.__DEMO06__ should be set:\n${detail}`).toBeTruthy();
    expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
    expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
    expect(state!.source, `expected the .vdb route:\n${detail}`).toBe("vdb");
    expect(state!.voxelCount ?? 0, `voxelCount should be > 0:\n${detail}`).toBeGreaterThan(0);
    expect(state!.histogramNonEmpty, `histogram should be non-empty:\n${detail}`).toBe(true);
    expect(state!.sliceNonEmpty, `slice should be non-empty:\n${detail}`).toBe(true);
  });
});
