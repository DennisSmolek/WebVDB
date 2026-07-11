import { defineConfig, devices } from "@playwright/test";

// E2E runs against the examples app. Later phases add WebGPU-dependent
// suites (traversal probes, golden-image SSIM) that need
// --enable-unsafe-webgpu; the Phase 0 smoke passes without an adapter.
export default defineConfig({
  testDir: "e2e",
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          // Escape hatch for environments with a pre-installed Chromium
          // that doesn't match this Playwright version's pinned build.
          ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
            ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH }
            : {}),
          // WebGPU everywhere: hardware where available, SwiftShader
          // (software Vulkan) in headless/CI. Verified working in CI-like
          // sandboxes; note WebGPU needs a secure context, so tests must
          // navigate to the served page before touching navigator.gpu.
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--enable-unsafe-swiftshader",
          ],
        },
      },
    },
  ],
  webServer: {
    command: "pnpm --filter @webvdb/examples dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
