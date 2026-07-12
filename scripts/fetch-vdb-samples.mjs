#!/usr/bin/env node
// Fetch the classic openvdb.org sample `.vdb` models, hosted (MIT) by the
// mjurczyk/openvdb pure-JS reader project — our Phase 5 parser fixture
// corpus (docs/PLAN.md Phase 5, docs/handoffs/PHASE-3.md "Phase 4/5 entry
// points": "mjurczyk/openvdb (MIT) commits the classic openvdb.org sample
// .vdbs (sphere 0.8 MB ... bunny_cloud 80 MB)").
//
// Pinned commit (resolved via `git ls-remote https://github.com/mjurczyk/openvdb.git HEAD`
// on 2026-07-12): 2173458e3b6d646deee5401ec6493079b09c091c
//
// Attribution: the .vdb assets themselves originate from the openvdb.org
// sample-model collection (Disney/DreamWorks/mesh-to-VDB conversions),
// CC-BY-SA 4.0 (https://www.openvdb.org/download/). The hosting repository
// (mjurczyk/openvdb, this script's download source) is MIT-licensed; the
// asset *files* keep the upstream CC-BY-SA 4.0 terms regardless of host.
//
// Usage:
//   node scripts/fetch-vdb-samples.mjs            # sphere/cube/smoke/utahteapot
//   BUNNY=1 node scripts/fetch-vdb-samples.mjs     # + bunny_cloud.vdb (~80 MB)
//   VDB_SAMPLES_SHA=<sha> node scripts/fetch-vdb-samples.mjs   # override the pin

import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { download, exists } from "./lib/download.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = path.join(ROOT, "fixtures", "vdb-samples");

const PINNED_SHA =
  process.env.VDB_SAMPLES_SHA ?? "2173458e3b6d646deee5401ec6493079b09c091c";
const BASE_URL = `https://raw.githubusercontent.com/mjurczyk/openvdb/${PINNED_SHA}/examples/public/assets`;

const SAMPLES = ["sphere.vdb", "cube.vdb", "smoke.vdb", "utahteapot.vdb"];
if (process.env.BUNNY === "1") {
  SAMPLES.push("bunny_cloud.vdb"); // ~80 MB — opt-in only.
}

console.error("VDB sample-model fixtures (mjurczyk/openvdb, MIT-hosted; assets CC-BY-SA 4.0 openvdb.org)");
console.error(`  pinned commit: ${PINNED_SHA}`);
if (process.env.BUNNY !== "1") {
  console.error("  (skipping bunny_cloud.vdb, ~80 MB — set BUNNY=1 to include it)");
}

let failures = 0;
for (const name of SAMPLES) {
  const dest = path.join(FIXTURES, name);
  if (await exists(dest)) {
    console.error(`  ✓ already present: fixtures/vdb-samples/${name}`);
    continue;
  }
  try {
    await download(`${BASE_URL}/${name}`, dest);
  } catch (err) {
    failures++;
    console.error(`  ✗ failed to fetch ${name}: ${err.message}`);
  }
}

if (failures > 0) {
  console.error(
    [
      "",
      "  Manual fallback: download the file(s) directly from",
      `    ${BASE_URL}/<name>.vdb`,
      `  and save under ${path.relative(ROOT, FIXTURES)}/.`,
    ].join("\n"),
  );
  process.exit(1);
}

console.error(`  ✓ done — ${SAMPLES.length} file(s) in fixtures/vdb-samples/`);
