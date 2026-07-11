import { expect, test } from "@playwright/test";

/**
 * Demo 02 — cloud: the Phase 3 golden-image gate (SPEC §5/§6, docs/PLAN.md).
 *
 * Loads the baked Fp8 fog-sphere fixture, fragment-raymarches it with
 * `NanoVDBVolumeMaterial` in deterministic test mode (fixed pose + params,
 * jitter off, one settled frame), and compares against a Playwright-managed
 * golden. PLAN's SSIM image gate is satisfied here by Playwright's pixel-diff
 * equivalent (`maxDiffPixelRatio`) — recorded as an intentional substitution in
 * the Phase 3 handoff.
 *
 * Two poses are checked: two camera angles catch more regressions than one.
 *
 * The goldens are SwiftShader-specific (software WebGPU renders deterministically
 * but not bit-identically to other adapters); regenerate on a matching runner
 * with `--update-snapshots`. First creation happens on the first run of this
 * spec.
 *
 * SwiftShader compiles the full pnanovdb library shader slowly (~10-20s cold),
 * so timeouts are generous.
 */

const PRIMITIVE_FIXTURE = "/fixtures/primitives/sphere_fog_fp8.nvdb";
const DEMO_URL = "/src/demos/02-cloud/index.html";

// The blit path renders a fixed 640x480 offscreen frame; a matching viewport
// makes the 2D display canvas 1:1 so the full-page screenshot is crisp.
test.use({ viewport: { width: 640, height: 480 } });

// Cold shader compile + two settled frames + snapshot stability retries.
test.setTimeout(180_000);

async function loadPose(page: import("@playwright/test").Page, pose: number): Promise<void> {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto(`${DEMO_URL}?test=1&pose=${pose}`);
  await page.waitForFunction(() => window.__DEMO02__ !== undefined, undefined, { timeout: 120_000 });

  const state = await page.evaluate(() => window.__DEMO02__);
  const detail = JSON.stringify({ state, consoleErrors }, null, 2);
  expect(state, `window.__DEMO02__ should be set:\n${detail}`).toBeTruthy();
  expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
  expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
}

test.describe("demo 02 — cloud golden image", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("pose 0 matches the golden", async ({ page }) => {
    const fixture = await page.request.get(PRIMITIVE_FIXTURE);
    test.skip(!fixture.ok(), `fixtures missing (${PRIMITIVE_FIXTURE} -> ${fixture.status()})`);

    await loadPose(page, 0);
    await expect(page).toHaveScreenshot("demo-02-sphere.png", { maxDiffPixelRatio: 0.02 });
  });

  test("pose 1 matches the golden", async ({ page }) => {
    const fixture = await page.request.get(PRIMITIVE_FIXTURE);
    test.skip(!fixture.ok(), `fixtures missing (${PRIMITIVE_FIXTURE} -> ${fixture.status()})`);

    await loadPose(page, 1);
    await expect(page).toHaveScreenshot("demo-02-sphere-pose1.png", { maxDiffPixelRatio: 0.02 });
  });
});
