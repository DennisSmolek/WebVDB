#!/usr/bin/env node
// Regenerates packages/nanovdb-wgsl/src/wgsl/pnanovdb-constants.generated.wgsl
// from packages/nanovdb-wgsl/vendor/stride-tables.json

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { generateWgslConstants } from "./lib/gen-wgsl-constants.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const IN = path.join(ROOT, "packages/nanovdb-wgsl/vendor/stride-tables.json");
const OUT = path.join(ROOT, "packages/nanovdb-wgsl/src/wgsl/pnanovdb-constants.generated.wgsl");

const jsonText = await readFile(IN, "utf8");
const tables = JSON.parse(jsonText);
const wgslContent = generateWgslConstants(tables);

await mkdir(path.dirname(OUT), { recursive: true });
await writeFile(OUT, wgslContent);

const lineCount = wgslContent.split("\n").length - 1; // -1 for trailing newline
console.error(
  `✓ wrote ${path.relative(ROOT, OUT)} — ${lineCount} lines`,
);
