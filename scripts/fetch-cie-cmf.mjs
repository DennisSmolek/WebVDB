#!/usr/bin/env node
// Fetch the CIE 1931 2° standard observer colour matching functions (360–830 nm, 1 nm) into fixtures/cie/.
//
// TEST-ONLY. The dataset is CC BY-SA 4.0 (CIE dataset metadata, DOI 10.25039/CIE.DS.xvudnb9b), so it is never
// committed and never shipped (DECISIONS D33): the package uses the analytic fit of Wyman, Sloan & Shirley (2013).
// The exact table is used only to measure that fit (packages/vdb/test/tools/cmf.test.ts) and by the K3 harness to
// render mjurczyk's 'planck' explosion scenes like-for-like (his renderer carries the table). Without it those tests
// skip and those parity scenes report INFO; WEBVDB_REQUIRE_FIXTURES does not require it.
//
// Usage:
//   node scripts/fetch-cie-cmf.mjs                 # → fixtures/cie/CIE_xyz_1931_2deg.csv (MD5 verified)
//   CIE_CMF_URL=https://... node scripts/fetch-cie-cmf.mjs

import { createHash } from "node:crypto";
import { readFile, rm } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { download, exists } from "./lib/download.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEST = path.join(ROOT, "fixtures", "cie", "CIE_xyz_1931_2deg.csv");
const URL_ = process.env.CIE_CMF_URL ?? "https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv";
const MD5 = "17cca777db64b17170f06f67ce9d3ab7";

const md5 = async (file) => createHash("md5").update(await readFile(file)).digest("hex");

console.error("CIE 1931 2° colour matching functions (CC BY-SA 4.0, CIE; DOI 10.25039/CIE.DS.xvudnb9b) — test-only fixture");

if ((await exists(DEST)) && (await md5(DEST)) === MD5) {
  console.error(`  ✓ already present, MD5 verified: ${path.relative(ROOT, DEST)}`);
  process.exit(0);
}
await rm(DEST, { force: true });

try {
  await download(URL_, DEST);
} catch (err) {
  console.error(`\n  ✗ download failed: ${err.message}`);
  console.error(`  Manual fallback: save ${URL_} as ${path.relative(ROOT, DEST)} and re-run (the MD5 is checked).`);
  process.exit(1);
}

const got = await md5(DEST);
if (got !== MD5) {
  await rm(DEST, { force: true });
  console.error(`  ✗ MD5 mismatch: got ${got}, expected ${MD5} (file removed)`);
  process.exit(1);
}
console.error(`  ✓ MD5 ${MD5}`);
