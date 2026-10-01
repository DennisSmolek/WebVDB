import { defineConfig } from "vitest/config";

// Local config for the PicoVDB evaluation spike. The repo-root
// vitest.config.ts only runs `projects: ["packages/*"]`, so this suite never
// joins `pnpm test`. Run it explicitly:
//   pnpm vitest run --root experiments/picovdb-spike
export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // The teapot round-trip reads ~7M voxels twice (f32 + u8).
    testTimeout: 300_000,
    hookTimeout: 300_000,
  },
});
