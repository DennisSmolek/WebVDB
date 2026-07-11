import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { defineConfig } from "vite";

// Repo-root `fixtures/` dir (this config lives in `examples/`).
const fixturesDir = fileURLToPath(new URL("../fixtures/", import.meta.url));

/**
 * Dev-only: serve the git-ignored repo-root `fixtures/` directory at
 * `/fixtures/*`. Demos (and the Playwright suite) fetch `.nvdb` grids and
 * their `.sidecar.json` ground-truth files from there. Kept intentionally
 * tiny and dev-server-scoped — production builds don't ship fixtures.
 */
function serveFixtures(): Plugin {
  return {
    name: "webvdb-serve-fixtures",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/fixtures", async (req, res, next) => {
        try {
          // Strip query string and leading slash, block path traversal.
          const rel = decodeURIComponent((req.url ?? "").split("?")[0]!).replace(/^\/+/, "");
          if (rel.includes("..")) {
            res.statusCode = 400;
            res.end("bad request");
            return;
          }
          const data = await readFile(fixturesDir + rel);
          res.setHeader(
            "Content-Type",
            rel.endsWith(".json") ? "application/json" : "application/octet-stream",
          );
          res.end(data);
        } catch {
          // Missing file -> 404 so tests can `test.skip()` cleanly.
          res.statusCode = 404;
          res.end("not found");
          next();
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [serveFixtures()],
  // WGSL sources import as strings; `?raw` works out of the box, this
  // covers bare .wgsl imports once demos start including the module.
  assetsInclude: ["**/*.wgsl", "**/*.nvdb"],
  server: {
    port: 5173,
    strictPort: true,
  },
});
