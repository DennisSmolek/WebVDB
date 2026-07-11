#!/usr/bin/env node
// Fetch JangaFX's free EmberGen VDB animation packs — our animated-sequence
// test corpus (docs/FEASIBILITY.md §8). JangaFX distributes these free packs
// from their site; check the page for current license terms.
//
// The direct asset URLs aren't stable, so this script scrapes the download
// page for .zip/.vdb links. Override discovery entirely with:
//   EMBERGEN_URLS="https://...a.zip,https://...b.zip" pnpm fixtures:embergen
//
// Assets land in fixtures/embergen/ (git-ignored, fetched per-machine).

import { spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { download, scrapeLinks } from "./lib/download.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = path.join(ROOT, "fixtures", "embergen");
const DOWNLOADS = path.join(ROOT, "fixtures", "downloads");

const PAGE = "https://jangafx.com/software/embergen/download/free-vdb-animations";

console.error("EmberGen free VDB pack (JangaFX)");

let urls = (process.env.EMBERGEN_URLS ?? "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

if (urls.length === 0) {
  try {
    urls = await scrapeLinks(PAGE, /vdb[^"']*\.(zip|7z)$|\.vdb$/i);
  } catch (err) {
    console.error(`  ✗ could not read ${PAGE}: ${err.message}`);
  }
  if (urls.length === 0) {
    console.error(
      [
        "",
        "  No direct links found (the page layout may have changed, or it",
        "  gates downloads behind a form).",
        "  Manual fallback:",
        `    1. Visit ${PAGE}`,
        "    2. Download one or more VDB packs (e.g. the smoke/explosion packs).",
        "    3. Unpack the .vdb frames into fixtures/embergen/<pack-name>/",
        "  Or pass direct URLs: EMBERGEN_URLS=<url1,url2> pnpm fixtures:embergen",
      ].join("\n"),
    );
    process.exit(1);
  }
  console.error(`  found ${urls.length} candidate pack link(s) on the page`);
}

await mkdir(FIXTURES, { recursive: true });
let failures = 0;
for (const url of urls) {
  const name = decodeURIComponent(new URL(url).pathname.split("/").pop() ?? "pack.zip");
  try {
    if (name.toLowerCase().endsWith(".vdb")) {
      await download(url, path.join(FIXTURES, name));
      continue;
    }
    const archive = await download(url, path.join(DOWNLOADS, name));
    const destDir = path.join(FIXTURES, name.replace(/\.(zip|7z)$/i, ""));
    await mkdir(destDir, { recursive: true });
    const unzip = spawnSync("unzip", ["-o", archive, "-d", destDir], { stdio: "ignore" });
    if (unzip.status === 0) {
      console.error(`  ✓ extracted → fixtures/embergen/${path.basename(destDir)}/`);
    } else {
      console.error(
        `  ⚠ saved ${path.relative(ROOT, archive)} but could not extract (need 'unzip'); unpack it into fixtures/embergen/ manually.`,
      );
    }
  } catch (err) {
    failures += 1;
    console.error(`  ✗ ${name}: ${err.message}`);
  }
}
process.exit(failures > 0 && failures === urls.length ? 1 : 0);
