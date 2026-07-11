import { expect, test } from "@playwright/test";

// Demo 01 — hello-nvdb: the Phase 1 GPU-read gate. A compute pass runs the
// hand-written WGSL NanoVDB traversal over 73 probe coords and must reproduce
// the native sidecar's value (|Δ| ≤ 1e-5) and active flag exactly.
//
// SwiftShader (software WebGPU) compiles + runs slowly, so we wait generously.
const DEMO_URL = "/src/demos/01-hello-nvdb/index.html";
const SIDECAR_URL = "/fixtures/primitives/sphere_fog_float.sidecar.json";

test("demo 01 GPU probe matches native ground truth (73/73)", async ({ page }) => {
  // Fixtures are git-ignored; skip cleanly if this machine doesn't have them.
  const sidecar = await page.request.get(SIDECAR_URL);
  test.skip(!sidecar.ok(), `fixtures missing (${SIDECAR_URL} -> ${sidecar.status()})`);

  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto(DEMO_URL);

  // Wait for the demo to publish its result (SwiftShader compile+dispatch).
  await page.waitForFunction(() => window.__DEMO01__ !== undefined, undefined, {
    timeout: 60_000,
  });

  const result = await page.evaluate(() => window.__DEMO01__);
  expect(result, "window.__DEMO01__ should be set").toBeTruthy();

  // Surface diagnostics in the assertion message on failure.
  const detail = JSON.stringify(
    { binding: result!.binding, error: result!.error, mismatches: result!.mismatches, consoleErrors },
    null,
    2,
  );

  expect(result!.error, `demo reported an error:\n${detail}`).toBeFalsy();
  expect(result!.total, `expected 73 probes:\n${detail}`).toBe(73);
  expect(result!.failed, `all probes must match:\n${detail}`).toBe(0);
  expect(result!.passed).toBe(73);
});
