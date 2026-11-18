#!/usr/bin/env node
// Fetch the classic openvdb.org sample `.vdb` models, hosted (MIT) by the
// mjurczyk/openvdb pure-JS reader project — our Phase 5 parser fixture
// corpus (docs/PLAN.md Phase 5, docs/handoffs/PHASE-3.md "Phase 4/5 entry
// points": "mjurczyk/openvdb (MIT) commits the classic openvdb.org samples").
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

import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { download, verifyFile } from "./lib/download.mjs";
import { VDB_SAMPLES, VDB_SAMPLES_COMMIT } from "./lib/vdb-samples.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = path.join(ROOT, "fixtures", "vdb-samples");

const BASE_URL =
  `https://raw.githubusercontent.com/mjurczyk/openvdb/${VDB_SAMPLES_COMMIT}` +
  "/examples/public/assets";

console.error("VDB sample-model fixtures (mjurczyk/openvdb, MIT-hosted; assets CC-BY-SA 4.0 openvdb.org)");
console.error(`  pinned commit: ${VDB_SAMPLES_COMMIT}`);
console.error("  Stanford bunny derivatives are intentionally excluded; see fixtures/ATTRIBUTION.md");

let failures = 0;
for (const [name, expected] of Object.entries(VDB_SAMPLES)) {
  const dest = path.join(FIXTURES, name);
  try {
    await download(`${BASE_URL}/${name}`, dest, {
      expectedBytes: expected.bytes,
      expectedSha256: expected.sha256,
    });
    const verified = await verifyFile(dest, {
      expectedBytes: expected.bytes,
      expectedSha256: expected.sha256,
    });
    if (!verified.ok) throw new Error(`post-download integrity check failed for ${name}`);
    console.error(`    sha256 ${verified.actualSha256}`);
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

console.error(`  ✓ done — ${Object.keys(VDB_SAMPLES).length} verified file(s) in fixtures/vdb-samples/`);
