#!/usr/bin/env node
// Fetch the Walt Disney Animation Studios cloud dataset and extract the
// quarter-resolution VDB — our stretch fixture (docs/DECISIONS.md D4).
//
// License: CC-BY-SA 3.0 — © Walt Disney Animation Studios.
// https://disneyanimation.com/data-sets/  (do not commit; fixtures are
// git-ignored and fetched per-machine by this script.)
//
// Usage:
//   node scripts/fetch-wdas-cloud.mjs                # quarter (default)
//   WDAS_VARIANTS=quarter,eighth node scripts/fetch-wdas-cloud.mjs
//   WDAS_URL=https://... node scripts/fetch-wdas-cloud.mjs   # URL override

import { spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { download, exists } from "./lib/download.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = path.join(ROOT, "fixtures", "wdas");
const DOWNLOADS = path.join(ROOT, "fixtures", "downloads");

const ZIP_URL =
  process.env.WDAS_URL ??
  "https://disneyanimation.com/media/technology/datasets/wdas_cloud.zip";
const VARIANTS = (process.env.WDAS_VARIANTS ?? "quarter").split(",").map((s) => s.trim());

console.error("WDAS cloud fixture (CC-BY-SA 3.0, © Walt Disney Animation Studios)");

const wanted = VARIANTS.map((v) => `wdas_cloud_${v}.vdb`);
const missing = [];
for (const name of wanted) {
  if (await exists(path.join(FIXTURES, name))) {
    console.error(`  ✓ already present: fixtures/wdas/${name}`);
  } else {
    missing.push(name);
  }
}
if (missing.length === 0) {
  console.error("  nothing to do.");
  process.exit(0);
}

const zipPath = path.join(DOWNLOADS, "wdas_cloud.zip");
try {
  await download(ZIP_URL, zipPath);
} catch (err) {
  console.error(`\n  ✗ download failed: ${err.message}`);
  console.error(
    [
      "",
      "  Manual fallback:",
      "    1. Visit https://disneyanimation.com/data-sets/ and download the",
      "       'Cloud Data Set' zip (the URL occasionally moves).",
      `    2. Save it as ${path.relative(ROOT, zipPath)}`,
      "    3. Re-run this script — it will pick the zip up and extract.",
      "  Or point this script at the real URL: WDAS_URL=<url> pnpm fixtures:wdas",
    ].join("\n"),
  );
  process.exit(1);
}

await mkdir(FIXTURES, { recursive: true });
const unzip = spawnSync("unzip", ["-o", "-j", zipPath, ...wanted, "-d", FIXTURES], {
  stdio: "inherit",
});
if (unzip.error?.code === "ENOENT") {
  console.error(
    `  ✗ 'unzip' not found. Extract manually: unzip -j ${path.relative(ROOT, zipPath)} ${wanted.join(" ")} -d fixtures/wdas/`,
  );
  process.exit(1);
}
if (unzip.status !== 0) {
  console.error(
    "  ✗ extraction failed — the zip's member names may have changed; list them with `unzip -l` and extract the variant you need into fixtures/wdas/.",
  );
  process.exit(unzip.status ?? 1);
}
console.error(`  ✓ done — ${wanted.join(", ")} in fixtures/wdas/`);
console.error(
  "  Next (Phase 1+): bake to .nvdb with `nanovdb_convert --fp8` — see fixtures/README.md.",
);
