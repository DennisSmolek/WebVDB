#!/usr/bin/env node
// Smoke test of the BUILT examples site (RELEASING.md section 4): serves the build under /vdb/examples/ with a
// tiny static server (like GitHub Pages: no rewrites, 404 for a missing file) and runs
// e2e/examples-site.spec.ts against it in headless Chromium (SwiftShader flags from the Playwright config).
//
//   pnpm examples:build && pnpm examples:smoke [--dir examples/dist] [--port 4173] [-- <playwright args>]
//
// --dir is the built examples directory (CI: out/examples). Needs a Chromium: `npx playwright install chromium`,
// or PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH.

import { spawn } from "node:child_process";
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : fallback;
};
const dir = path.resolve(repo, flag("--dir", "examples/dist"));
const port = Number(flag("--port", "4173"));
const rest = argv.includes("--") ? argv.slice(argv.indexOf("--") + 1) : [];
const MOUNT = "/vdb/examples/";

if (!existsSync(path.join(dir, "index.html"))) {
  console.error(`examples-smoke: no build at ${dir} (run pnpm examples:build first)`);
  process.exit(2);
}

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".wasm": "application/wasm",
  ".glb": "model/gltf-binary",
  ".txt": "text/plain; charset=utf-8",
};

const server = createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  let rel = decodeURIComponent(url.pathname);
  if (!rel.startsWith(MOUNT) || rel.includes("..")) {
    res.statusCode = 404;
    res.end("not found");
    return;
  }
  rel = rel.slice(MOUNT.length);
  let file = path.join(dir, rel);
  if (existsSync(file) && statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!existsSync(file)) {
    res.statusCode = 404;
    res.end("not found");
    return;
  }
  res.setHeader("Content-Type", TYPES[path.extname(file)] ?? "application/octet-stream");
  res.setHeader("Content-Length", statSync(file).size);
  if (req.method === "HEAD") {
    res.end();
    return;
  }
  createReadStream(file).pipe(res);
});

await new Promise((resolve) => server.listen(port, "127.0.0.1", resolve));
const site = `http://127.0.0.1:${port}${MOUNT.replace(/\/$/, "")}`;
console.log(`examples-smoke: serving ${path.relative(repo, dir)} at ${site}/`);

const child = spawn("npx", ["playwright", "test", "-c", "playwright.examples-site.config.ts", ...rest], {
  cwd: repo,
  stdio: "inherit",
  env: { ...process.env, WEBVDB_EXAMPLES_SITE_URL: site },
});
child.on("exit", (code) => {
  server.close();
  process.exit(code ?? 1);
});
