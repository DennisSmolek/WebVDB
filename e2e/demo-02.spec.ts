import { expect, test } from "@playwright/test";

/**
 * Demo 02 — cloud: the Phase 3 golden-image gate (SPEC §5/§6, docs/archive/v1/PLAN.md).
 *
 * Loads the baked Fp8 fog-sphere fixture into a `VDBGridPool`, fragment-raymarches it with
 * `VDBVolumeMaterial` (through a `VDBGridNode`) in deterministic test mode (fixed pose + params,
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
 * `presentation=readback` because headless SwiftShader can't present a WebGPU
 * canvas (DIVERGES V7); the demo's default is direct canvas presentation.
 *
 * SwiftShader compiles the shader slowly (~10-20s cold), so timeouts are generous.
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

  await page.goto(`${DEMO_URL}?test=1&fixture=primitive&presentation=readback&pose=${pose}`);
  await page.waitForFunction(() => window.__DEMO02__ !== undefined, undefined, { timeout: 120_000 });

  const state = await page.evaluate(() => window.__DEMO02__);
  const detail = JSON.stringify({ state, consoleErrors }, null, 2);
  expect(state, `window.__DEMO02__ should be set:\n${detail}`).toBeTruthy();
  expect(state!.error, `demo reported an error:\n${detail}`).toBeUndefined();
  expect(state!.ready, `demo should reach ready:\n${detail}`).toBe(true);
  expect(state!.fixture, `CI must use the primitive fixture:\n${detail}`).toBe("sphere_fog_fp8.nvdb");
  expect(state!.presentation, `CI must retain deterministic readback:\n${detail}`).toBe("readback");
  expect(state!.adapter, `canonical goldens require SwiftShader:\n${detail}`).toMatch(/swiftshader/i);
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

  test("readback handles rows whose byte width needs WebGPU padding", async ({ page }) => {
    const fixture = await page.request.get(PRIMITIVE_FIXTURE);
    test.skip(!fixture.ok(), `fixtures missing (${PRIMITIVE_FIXTURE} -> ${fixture.status()})`);

    // 641 RGBA8 pixels need 252 bytes of WebGPU row padding. The old blit
    // treated those padded bytes as pixels and smeared the volume across the canvas.
    await page.setViewportSize({ width: 641, height: 480 });
    await page.goto(`${DEMO_URL}?fixture=primitive&presentation=readback`);
    await page.waitForFunction(() => window.__DEMO02__?.ready === true, undefined, { timeout: 120_000 });

    const bounds = await page.evaluate(() => {
      const canvas = document.querySelector<HTMLCanvasElement>("#view")!;
      const context = canvas.getContext("2d")!;
      const { data, width, height } = context.getImageData(0, 0, canvas.width, canvas.height);
      const background = [data[0]!, data[1]!, data[2]!] as const;
      let minX = width;
      let maxX = -1;

      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
          const offset = (y * width + x) * 4;
          const difference =
            Math.abs(data[offset]! - background[0]) +
            Math.abs(data[offset + 1]! - background[1]) +
            Math.abs(data[offset + 2]! - background[2]);
          if (difference <= 18) continue;
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
        }

      return { canvasWidth: width, contentWidth: maxX - minX + 1 };
    });

    expect(bounds.contentWidth).toBeGreaterThan(0);
    expect(bounds.contentWidth).toBeLessThan(bounds.canvasWidth * 0.65);
  });
});
