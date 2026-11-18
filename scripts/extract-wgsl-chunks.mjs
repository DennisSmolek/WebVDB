#!/usr/bin/env node
// Regenerates packages/vdb/src/core/wgsl/pnanovdb.chunks.generated.ts from the
// unmodified vendored packages/vdb/vendor/pnanovdb.wgsl (K1.4a; design note
// docs/design/K1-core.md §3.3). Run after any deliberate vendored edit.

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extractWgslChunks, generateChunksTs, pinnedCommitFromVendorDoc } from "./lib/wgsl-chunks.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const IN = path.join(ROOT, "packages/vdb/vendor/pnanovdb.wgsl");
const DOC = path.join(ROOT, "packages/vdb/vendor/VENDOR.md");
const OUT = path.join(ROOT, "packages/vdb/src/core/wgsl/pnanovdb.chunks.generated.ts");

const source = await readFile(IN, "utf8");
const pinnedCommit = pinnedCommitFromVendorDoc(await readFile(DOC, "utf8"));
await writeFile(OUT, generateChunksTs(source, { pinnedCommit }));
console.error(`wrote ${path.relative(ROOT, OUT)}: ${extractWgslChunks(source).length} declarations`);
