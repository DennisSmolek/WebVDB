import { defineConfig } from "vite";

export default defineConfig({
  // WGSL sources import as strings; `?raw` works out of the box, this
  // covers bare .wgsl imports once demos start including the module.
  assetsInclude: ["**/*.wgsl", "**/*.nvdb"],
  server: {
    port: 5173,
    strictPort: true,
  },
});
