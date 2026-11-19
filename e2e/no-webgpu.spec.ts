import { expect, test } from "@playwright/test";

/**
 * Every demo shows a card (not a blank canvas) when `navigator.gpu` is missing or `requestAdapter()` returns null:
 * `examples/src/demos/webgpu-gate.ts` (imported first by each demo entry). No GPU needed.
 */
const DEMOS = [
  "01-hello-nvdb", "02-cloud", "03-gpu-read", "04-atlas-fallback", "05-embergen-sequence", "06-explorer", "07-builder",
  "08-webgpu-smoke", "09-coarse-first", "10-frame-renderer", "11-sky-clouds", "12-explosion", "13-level-sets",
  "14-points", "15-full-res-vs-dense", "16-streaming",
];

test.describe("no WebGPU", () => {
  for (const demo of DEMOS) {
    test(`${demo}: no navigator.gpu`, async ({ page }) => {
      await page.addInitScript(() => Object.defineProperty(navigator, "gpu", { value: undefined, configurable: true }));
      await page.goto(`/src/demos/${demo}/index.html`);
      const card = page.getByTestId("no-webgpu");
      await expect(card).toBeVisible();
      await expect(card).toContainText("needs WebGPU");
      expect(await card.getAttribute("data-reason")).toBe("no-api");
      expect(await card.locator("a").getAttribute("href")).toBe("/");
    });
  }

  test("null adapter", async ({ page }) => {
    await page.addInitScript(() => Object.defineProperty(navigator, "gpu", { value: { requestAdapter: async () => null }, configurable: true }));
    await page.goto("/src/demos/11-sky-clouds/index.html");
    await expect(page.getByTestId("no-webgpu")).toHaveAttribute("data-reason", "no-adapter");
  });
});
