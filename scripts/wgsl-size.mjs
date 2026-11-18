// Measures the generated WGSL size of every variant in the K1 matrix (design
// note §3.5 method, via scripts/lib/wgsl-variants.mjs) and compares it with, or
// with --write regenerates, packages/vdb/src/core/wgsl/size-snapshot.json.
//
// The test packages/vdb/test/core/wgsl-size.test.ts fails when a variant's min
// bytes differ from this snapshot, and checks the §3.6 budgets (size-budgets.json).
//
// Run:  pnpm build && pnpm wgsl:size            (check: exit 1 on any difference)
//       pnpm build && pnpm wgsl:size --write    (regenerate the snapshot)

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { vdbModuleWgsl } from "../packages/vdb/dist/core/index.js";
import { variantMatrix, wgslSizes } from "./lib/wgsl-variants.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const SNAPSHOT_PATH = path.join(ROOT, "packages/vdb/src/core/wgsl/size-snapshot.json");
const WRITE = process.argv.includes("--write");

/** Sizes of every matrix variant, keys sorted, so the snapshot is deterministic. */
export function measureVariants() {
  const entries = variantMatrix().map(({ id, options }) => {
    const { min, gz } = wgslSizes(vdbModuleWgsl(options));
    return [id, { gz, min }];
  });
  entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return Object.fromEntries(entries);
}

export function snapshotText(variants) {
  const snapshot = {
    method:
      "design note §3.5: min = comments stripped, whitespace collapsed, no mangling (wgsl-variants minifyWgsl); gz = gzip level 9 of min",
    unit: "bytes",
    variants,
  };
  return `${JSON.stringify(snapshot, null, 2)}\n`;
}

async function main() {
  const current = measureVariants();
  const text = snapshotText(current);

  if (WRITE) {
    await writeFile(SNAPSHOT_PATH, text);
    console.log(`wrote ${Object.keys(current).length} variants to ${SNAPSHOT_PATH}`);
    return;
  }

  const previous = JSON.parse(await readFile(SNAPSHOT_PATH, "utf8")).variants;
  const ids = new Set([...Object.keys(previous), ...Object.keys(current)]);
  const diffs = [];
  for (const id of [...ids].sort()) {
    const was = previous[id];
    const now = current[id];
    if (!was || !now) diffs.push(`${id}: ${was ? "removed" : "added"}`);
    else if (was.min !== now.min || was.gz !== now.gz) {
      diffs.push(`${id}: min ${was.min} -> ${now.min}, gz ${was.gz} -> ${now.gz}`);
    }
  }
  if (diffs.length === 0) {
    console.log(`WGSL sizes match the snapshot (${ids.size} variants).`);
    return;
  }
  console.log(`WGSL sizes differ from the snapshot (${diffs.length}):`);
  for (const line of diffs) console.log(`  ${line}`);
  console.log("If intended, run `pnpm wgsl:size --write` and commit.");
  process.exitCode = 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
