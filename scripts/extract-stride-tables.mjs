#!/usr/bin/env node
// Regenerates packages/nanovdb-wgsl/vendor/stride-tables.json from the
// vendored PNanoVDB.h. Run after bumping the vendored header; the vendor
// test suite fails if the JSON drifts out of sync.

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extractStrideTables } from "./lib/pnanovdb-extract.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HEADER = path.join(ROOT, "packages/nanovdb-wgsl/vendor/upstream/PNanoVDB.h");
const OUT = path.join(ROOT, "packages/nanovdb-wgsl/vendor/stride-tables.json");

// Provenance for the vendored header — update alongside vendor/VENDOR.md.
const UPSTREAM = {
  repo: "https://github.com/AcademySoftwareFoundation/openvdb",
  path: "nanovdb/nanovdb/PNanoVDB.h",
  commit: "a532de5526ef791280b6483a872336a811a68542",
};

const headerText = await readFile(HEADER, "utf8");
const sha256 = createHash("sha256").update(headerText).digest("hex");
const tables = extractStrideTables(headerText, { ...UPSTREAM, sha256 });

await writeFile(OUT, `${JSON.stringify(tables, null, 2)}\n`);
console.error(
  `✓ wrote ${path.relative(ROOT, OUT)} — ABI ${tables.$meta.abi}, ` +
    `${Object.keys(tables.gridTypes).length} grid types, ` +
    `${Object.keys(tables.gridTypeConstants).length} constant rows`,
);
