import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: [
      {
        find: /^nanovdb-wgsl$/,
        replacement: fileURLToPath(
          new URL("./packages/nanovdb-wgsl/src/index.ts", import.meta.url),
        ),
      },
      {
        find: /^three-nanovdb$/,
        replacement: fileURLToPath(
          new URL("./packages/three-nanovdb/src/index.ts", import.meta.url),
        ),
      },
      {
        find: /^vdb-web-tools$/,
        replacement: fileURLToPath(
          new URL("./packages/vdb-web-tools/src/index.ts", import.meta.url),
        ),
      },
    ],
  },
  test: {
    // Each package is its own Vitest project; add "examples" here once it
    // grows unit tests (its coverage today is the Playwright smoke).
    projects: ["packages/*"],
  },
});
