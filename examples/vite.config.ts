import { readdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { defineConfig } from "vite";
// @ts-expect-error plain .mjs without types: the hosted asset set and the attribution rows (RELEASING.md section 4)
import { renderAttribution } from "../scripts/examples-assets.mjs";

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

/** Fills `<!-- ASSET_LIST -->` in attribution.html from examples/assets.manifest.json (dev and build). */
function attributionPage(): Plugin {
  return {
    name: "webvdb-attribution",
    transformIndexHtml: {
      order: "pre",
      handler(html, ctx) {
        return ctx.filename.endsWith("attribution.html") ? html.replace("<!-- ASSET_LIST -->", () => renderAttribution() as string) : html;
      },
    },
  };
}

/** Every page of the multi-page build: the gallery home, the attribution page and each demo's index.html. */
function pages(): Record<string, string> {
  const root = fileURLToPath(new URL("./", import.meta.url));
  const input: Record<string, string> = { main: `${root}index.html`, attribution: `${root}attribution.html` };
  const demos = `${root}src/demos/`;
  for (const d of readdirSync(demos, { withFileTypes: true })) {
    if (d.isDirectory() && /^\d\d-/.test(d.name)) input[d.name] = `${demos}${d.name}/index.html`;
  }
  return input;
}

// Hosted: docs at https://pmndrs.github.io/vdb/, examples under /vdb/examples/ (override with VDB_EXAMPLES_BASE).
// `pnpm dev` and the Playwright dev server keep base "/" and the serveFixtures plugin.
export default defineConfig(({ command }) => ({
  base: command === "build" ? (process.env["VDB_EXAMPLES_BASE"] ?? "/vdb/examples/") : "/",
  plugins: [serveFixtures(), attributionPage()],
  build: {
    // Demos use top-level await (webgpu-gate) and module workers.
    target: "esnext",
    rollupOptions: { input: pages() },
  },
  worker: { format: "es" },
  resolve: {
    alias: [
      // Most specific first: first matching alias wins.
      {
        find: /^@pmndrs\/vdb\/pnanovdb\.wgsl/,
        replacement: fileURLToPath(new URL("../packages/vdb/vendor/pnanovdb.wgsl", import.meta.url)),
      },
      {
        find: /^@pmndrs\/vdb\/tools$/,
        replacement: fileURLToPath(new URL("../packages/vdb/src/tools/index.ts", import.meta.url)),
      },
      {
        find: /^@pmndrs\/vdb\/three$/,
        replacement: fileURLToPath(new URL("../packages/vdb/src/three/index.ts", import.meta.url)),
      },
      {
        find: /^@pmndrs\/vdb$/,
        replacement: fileURLToPath(new URL("../packages/vdb/src/core/index.ts", import.meta.url)),
      },
    ],
  },
  // WGSL sources import as strings; `?raw` works out of the box, this
  // covers bare .wgsl imports once demos start including the module.
  assetsInclude: ["**/*.wgsl", "**/*.nvdb"],
  server: {
    port: 5173,
    strictPort: true,
  },
}));
