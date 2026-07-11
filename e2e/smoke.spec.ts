import { expect, test } from "@playwright/test";

// Phase 0 smoke: the Vite app serves, TS imports resolve across workspace
// packages, and the WebGPU capability probe completes (with or without an
// adapter — headless CI often has none).
test("examples app boots and reports WebGPU capability", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("WebVDB Examples");

  const report = page.getByTestId("webgpu-report");
  // main.ts imported constants from nanovdb-wgsl — proves workspace wiring.
  await expect(report).toContainText("workspace ok");
  await expect(report).toContainText("Float, Fp8, FpN");
  // The probe always sets data-webgpu, whatever the environment offers.
  await expect(report).toHaveAttribute(
    "data-webgpu",
    /^(ok|unavailable|no-adapter|error)$/,
  );
});
