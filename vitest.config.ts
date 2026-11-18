import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: [
      {
        find: /^@pmndrs\/vdb\/tools$/,
        replacement: fileURLToPath(new URL("./packages/vdb/src/tools/index.ts", import.meta.url)),
      },
      {
        find: /^@pmndrs\/vdb\/three$/,
        replacement: fileURLToPath(new URL("./packages/vdb/src/three/index.ts", import.meta.url)),
      },
      {
        find: /^@pmndrs\/vdb$/,
        replacement: fileURLToPath(new URL("./packages/vdb/src/core/index.ts", import.meta.url)),
      },
    ],
  },
  test: {
    // Each package is its own Vitest project; "examples" holds node-side
    // checks of the demos (the demos themselves run in Playwright).
    projects: ["packages/*", "examples"],
  },
});
