import { expect, test } from "@playwright/test";

// Demo 03 — gpu-read: the educational "read a NanoVDB volume on the GPU"
// example (docs/PLAN.md Phase 3, docs/SPEC.md §5 row 03). A tiny raw-WebGPU
// compute pass reads 8 voxels via a hand-written WGSL traversal and must
// reproduce the native sidecar's value + active flag for every one of them.
//
// SwiftShader (software WebGPU) compiles + runs slowly, so we wait generously.
const DEMO_URL = "/src/demos/03-gpu-read/index.html";
const SIDECAR_URL = "/fixtures/primitives/sphere_fog_float.sidecar.json";

test("demo 03 GPU read matches native ground truth (8/8)", async ({ page }) => {
  // Fixtures are git-ignored; skip cleanly if this machine doesn't have them.
  const sidecar = await page.request.get(SIDECAR_URL);
  test.skip(!sidecar.ok(), `fixtures missing (${SIDECAR_URL} -> ${sidecar.status()})`);

  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto(DEMO_URL);

  // Wait for the demo to publish its result (SwiftShader compile + dispatch).
  await page.waitForFunction(() => window.__DEMO03__ !== undefined, undefined, {
    timeout: 60_000,
  });

  const result = await page.evaluate(() => window.__DEMO03__);
  expect(result, "window.__DEMO03__ should be set").toBeTruthy();

  // Surface diagnostics in the assertion message on failure.
  const detail = JSON.stringify({ result, consoleErrors }, null, 2);

  expect(result!.error, `demo reported an error:\n${detail}`).toBeFalsy();
  expect(result!.total, `expected at least 8 voxels:\n${detail}`).toBeGreaterThanOrEqual(8);
  expect(result!.matched, `all voxels must match:\n${detail}`).toBe(result!.total);
});
