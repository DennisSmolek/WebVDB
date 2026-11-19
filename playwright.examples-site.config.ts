import { defineConfig, devices } from "@playwright/test";

// The built-site smoke (`pnpm examples:smoke`, scripts/examples-smoke.mjs): no dev server, no fixture global setup.
// The spec is skipped unless WEBVDB_EXAMPLES_SITE_URL is set (so the normal e2e run needs no build).
export default defineConfig({
  testDir: "e2e",
  testMatch: "examples-site.spec.ts",
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {},
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-unsafe-swiftshader"],
        },
      },
    },
  ],
});
