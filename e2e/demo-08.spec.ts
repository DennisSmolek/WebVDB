import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

/**
 * Demo 08 — webgpu-smoke (K1.7): the OpenVDB smoke through raw WebGPU and the
 * `@pmndrs/vdb` core only (no three.js). The e2e gate (docs/dev/design/K1-core.md
 * §8 row K1.7):
 *   - the first frame matches the golden (`presentation=readback`: headless
 *     SwiftShader has no compositor for a WebGPU canvas context);
 *   - the "add second grid" button stages a torus, the next `pool.update`
 *     grows the buffer, `version` moves, the host rebuilds its bind group,
 *     and the frame still renders (second golden) with no WebGPU errors.
 *
 * Goldens are SwiftShader-canonical (`-linux` suffix by the repo template);
 * regenerate on a matching runner with `--update-snapshots`.
 */

const SMOKE = "/fixtures/vdb-samples/smoke.vdb";
const TORUS = "/fixtures/primitives/torus_fog_fp8.nvdb";
const DEMO_URL = "/src/demos/08-webgpu-smoke/index.html?test=1&presentation=readback";

test.use({ viewport: { width: 640, height: 480 } });
test.setTimeout(240_000);

async function state(page: Page) {
  return page.evaluate(() => window.__DEMO08__!);
}

/** The panel overlaps the canvas and prints live timings; keep it out of the goldens. */
const HIDE_PANEL = "#panel { visibility: hidden !important; }";

test.describe("demo 08 — raw WebGPU smoke", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("renders the smoke, then a second grid bumps version and still renders", async ({ page }) => {
    for (const url of [SMOKE, TORUS]) {
      const res = await page.request.get(url);
      test.skip(!res.ok(), `fixtures missing (${url} -> ${res.status()})`);
    }
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto(DEMO_URL);
    await page.waitForFunction(() => window.__DEMO08__?.ready === true || window.__DEMO08__?.error !== undefined, undefined, {
      timeout: 180_000,
    });
    const first = await state(page);
    const detail = () => JSON.stringify({ first, consoleErrors }, null, 2);
    expect(first.error, `demo reported an error:\n${detail()}`).toBeUndefined();
    expect(first.adapter, `canonical goldens require SwiftShader:\n${detail()}`).toMatch(/swiftshader/i);
    expect(first.frames).toBe(1);
    expect(first.gridCount).toBe(1);
    expect(first.version, "the first update grows the 1 MiB pool to fit the smoke").toBeGreaterThan(0);
    expect(first.bindGroupBuilds).toBe(1);
    expect(first.wgslBytes, "the page reports its generated WGSL size").toBeGreaterThan(10_000);
    expect(first.gpuErrors).toEqual([]);
    await expect(page.locator("#stats")).toContainText(`${first.wgslBytes!.toLocaleString("en-US")} B WGSL`);
    const hide = await page.addStyleTag({ content: HIDE_PANEL });
    await expect(page.locator("#view")).toHaveScreenshot("demo-08-smoke.png", { maxDiffPixelRatio: 0.02 });

    await hide.evaluate((el) => el.remove());
    await page.click("#second-grid");
    await page.waitForFunction(
      () => (window.__DEMO08__?.ready === true && window.__DEMO08__.secondGrid) || window.__DEMO08__?.error !== undefined,
      undefined,
      { timeout: 180_000 },
    );
    const second = await state(page);
    const detail2 = JSON.stringify({ first, second, consoleErrors }, null, 2);
    expect(second.error, `demo reported an error:\n${detail2}`).toBeUndefined();
    expect(second.gridCount).toBe(2);
    expect(second.version, `adding the torus grows the pool:\n${detail2}`).toBeGreaterThan(first.version);
    expect(second.bindGroupBuilds, "the host rebuilt its bind group for the new buffer").toBe(first.bindGroupBuilds + 1);
    expect(second.frames).toBe(2);
    expect(second.gpuErrors, `no validation errors after the growth:\n${detail2}`).toEqual([]);
    await page.addStyleTag({ content: HIDE_PANEL });
    await expect(page.locator("#view")).toHaveScreenshot("demo-08-two-grids.png", { maxDiffPixelRatio: 0.02 });
  });
});
