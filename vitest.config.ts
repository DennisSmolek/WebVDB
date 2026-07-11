import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Each package is its own Vitest project; add "examples" here once it
    // grows unit tests (its coverage today is the Playwright smoke).
    projects: ["packages/*"],
  },
});
