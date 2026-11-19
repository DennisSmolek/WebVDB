#!/usr/bin/env node
// Fetch JangaFX's free EmberGen VDB animation packs — our animated-sequence
// test corpus (docs/archive/v1/FEASIBILITY.md §8). JangaFX distributes these free packs
// from their site; check the page for current license terms.
//
// The direct asset URLs aren't stable, so this script scrapes the download
// page for .zip/.vdb links. Override discovery entirely with:
//   EMBERGEN_URLS="https://...a.zip,https://...b.zip" pnpm fixtures:embergen
//
// Assets land in fixtures/embergen/ (git-ignored, fetched per-machine).

import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, open, readdir, rename, rm } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { download, scrapeLinks } from "./lib/download.mjs";
import { assertSafeArchiveMembers, validateExtractedTree } from "./lib/safe-archive.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = path.join(ROOT, "fixtures", "embergen");
const DOWNLOADS = path.join(ROOT, "fixtures", "downloads");

const PAGE = "https://jangafx.com/software/embergen/download/free-vdb-animations";

console.error("EmberGen free VDB pack (JangaFX)");

//* Archive helpers ============================================================

async function archiveType(file) {
  const handle = await open(file, "r");
  try {
    const magic = Buffer.alloc(8);
    await handle.read(magic, 0, magic.length, 0);
    if (magic[0] === 0x50 && magic[1] === 0x4b) return "zip";
    if (magic.subarray(0, 6).equals(Buffer.from([0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c]))) return "7z";
  } finally {
    await handle.close();
  }
  throw new Error(`download is neither a ZIP nor 7z archive: ${path.relative(ROOT, file)}`);
}

async function vdbFrames(dir) {
  const frames = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) frames.push(...(await vdbFrames(fullPath)));
    else if (entry.isFile() && entry.name.toLowerCase().endsWith(".vdb")) frames.push(fullPath);
  }
  return frames.sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
}

function extractArchive(type, archive, destDir) {
  if (type === "zip") return spawnSync("unzip", ["-o", archive, "-d", destDir], { stdio: "inherit" });

  for (const command of ["7zz", "7z", "unar"]) {
    const args = command === "unar" ? ["-f", "-o", destDir, archive] : ["x", "-y", `-o${destDir}`, archive];
    const result = spawnSync(command, args, { stdio: "inherit" });
    if (!result.error || result.error.code !== "ENOENT") return result;
  }
  return { status: 1, error: new Error("no 7z extractor found (install 7zz, 7z, or unar)") };
}

async function resolveDownloadUrl(url) {
  if (!/mediafire\.com\/file\//i.test(url)) return url;
  const links = await scrapeLinks(url, /download[^/]*\.mediafire\.com\/.*\.(zip|7z)(?:[?#]|$)/i);
  if (links.length === 0) throw new Error(`could not resolve MediaFire download page: ${url}`);
  return links[0];
}

//* URL discovery ==============================================================

let urls = (process.env.EMBERGEN_URLS ?? "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

if (urls.length === 0) {
  try {
    urls = await scrapeLinks(PAGE, /mediafire\.com\/file\/.*\.(zip|7z)\/file|\.vdb(?:[?#]|$)/i);
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
  const allUrls = urls;
  const preferred = process.env.EMBERGEN_PACK ?? "SmallCampfireVDB";
  if (process.env.EMBERGEN_ALL !== "1")
    urls = allUrls.filter((url) => decodeURIComponent(url).includes(preferred)).slice(0, 1);
  if (urls.length === 0) urls = allUrls.slice(0, 1);
  console.error(
    `  found ${allUrls.length} pack link(s); fetching ${urls.length} (${process.env.EMBERGEN_ALL === "1" ? "all" : preferred})`,
  );
}

await mkdir(FIXTURES, { recursive: true });
let failures = 0;
for (const url of urls) {
  let name = "pack";
  let tempDir;
  try {
    const resolvedUrl = await resolveDownloadUrl(url);
    const pathname = decodeURIComponent(new URL(resolvedUrl).pathname);
    name = path.basename(pathname) || "pack.zip";
    const packName = name.replace(/\.(zip|7z|vdb)$/i, "");
    const destDir = path.join(FIXTURES, packName);

    if (name.toLowerCase().endsWith(".vdb")) {
      await mkdir(destDir, { recursive: true });
      await download(resolvedUrl, path.join(destDir, name));
      const frames = await vdbFrames(destDir);
      if (frames.length === 0) throw new Error(`direct download produced no .vdb frame in ${destDir}`);
      console.error(`  ✓ pack: ${path.relative(ROOT, destDir)}/ (${frames.length} .vdb frame)`);
      continue;
    }

    const archive = await download(resolvedUrl, path.join(DOWNLOADS, name));
    const type = await archiveType(archive);
    assertSafeArchiveMembers(type, archive);
    tempDir = await mkdtemp(path.join(FIXTURES, `.${packName}-`));
    const extracted = extractArchive(type, archive, tempDir);
    if (extracted.status !== 0)
      throw new Error(extracted.error?.message ?? `${type} extraction exited with status ${extracted.status}`);

    await validateExtractedTree(tempDir);
    const frames = await vdbFrames(tempDir);
    if (frames.length === 0) {
      await rm(tempDir, { recursive: true, force: true });
      throw new Error(`${type} extraction produced no .vdb frames`);
    }
    await rm(destDir, { recursive: true, force: true });
    await rename(tempDir, destDir);
    tempDir = undefined;
    console.error(`  ✓ pack: ${path.relative(ROOT, destDir)}/ (${frames.length} .vdb frames)`);
  } catch (err) {
    if (tempDir) await rm(tempDir, { recursive: true, force: true });
    failures += 1;
    console.error(`  ✗ ${name}: ${err.message}`);
  }
}
process.exitCode = failures > 0 ? 1 : 0;
