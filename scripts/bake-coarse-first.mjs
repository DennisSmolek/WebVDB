#!/usr/bin/env node
/**
 * bake-coarse-first.mjs — K4.16/K4.18: bake a coarse-first `.nvdb`
 * (`vdbBakeCoarseFirst`: `[<name>.lod48, <name>]`) and measure what a stream
 * reader needs before the preview can be drawn, with codec NONE and ZIP, with
 * and without HTTP-style gzip on top.
 *
 *   pnpm build && node scripts/bake-coarse-first.mjs [in.vdb] [--out out.nvdb] [--codec zip|none] [--resolution 48]
 *
 * Default input: fixtures/vdb-samples/smoke.vdb (`pnpm fixtures:vdb-samples`).
 * A `.nvdb` input bakes its grid 0 as an image (the full grid kept as encoded,
 * e.g. Fp8): gallery demo 16 streams
 * `fixtures/wdas/wdas_cloud_quarter_fp8.nvdb --out fixtures/wdas/wdas_cloud_quarter.coarse.nvdb`.
 * "First paint" is the number of transferred bytes a streaming reader has when
 * `NvdbStreamReader` emits the preview: for the raw transfer it is the event's
 * `bytesReceived`; for a gzip transfer it is the gzip stream offset (fflate's
 * streaming Gunzip, 64-byte steps) at which that many file bytes exist. Sizes
 * are deterministic; no timings are claimed.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { Gunzip } = createRequire(path.join(root, "packages/vdb/package.json"))("fflate");
const T = await import(path.join(root, "packages/vdb/dist/tools/index.js"));

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
};
const out = flag("--out", null);
const outCodec = flag("--codec", "zip");
const resolution = Number(flag("--resolution", "48"));
const input = args[0] ?? path.join(root, "fixtures/vdb-samples/smoke.vdb");
if (!existsSync(input)) {
  console.log(`skip: ${input} missing (pnpm fixtures:vdb-samples)`);
  process.exit(0);
}

const KiB = (n) => `${(n / 1024).toFixed(1)} KiB`;

function streamOffset(compressed, prefixBytes, step = 64) {
  let produced = 0;
  let found = -1;
  const g = new Gunzip((chunk) => {
    produced += chunk.length;
  });
  for (let off = 0; off < compressed.length; off += step) {
    const end = Math.min(off + step, compressed.length);
    g.push(compressed.subarray(off, end), end === compressed.length);
    if (produced >= prefixBytes) {
      found = end;
      break;
    }
  }
  return found;
}

const buf = readFileSync(input);
const arrayBuffer = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
let grid;
if (input.endsWith(".nvdb")) {
  const file = T.NvdbFile.fromArrayBuffer(arrayBuffer);
  grid = file.gridImage(0);
  console.log(`${path.basename(input)}: grid "${file.grids[0].name}" (${file.grids[0].gridType})`);
} else {
  grid = T.parseVdb(arrayBuffer).grids.find((g) => g.valueType === "float");
  console.log(`${path.basename(input)}: grid "${grid.name}"`);
}

const results = {};
for (const codec of ["none", "zip"]) {
  const t = performance.now();
  const bytes = T.vdbBakeCoarseFirst(grid, { previewResolution: resolution, codec });
  const bakeS = (performance.now() - t) / 1000;
  const reader = new T.NvdbStreamReader();
  const events = [];
  for (let off = 0; off < bytes.length; off += 4096) events.push(...reader.push(bytes.subarray(off, off + 4096)));
  reader.end();
  const [preview, full] = events;
  const gz = gzipSync(bytes, { level: 9 });
  results[codec] = {
    bytes,
    total: bytes.length,
    gzTotal: gz.length,
    previewEnd: preview.bytesReceived,
    gzFirst: streamOffset(gz, preview.bytesReceived),
    names: events.map((e) => `${e.metadata.name} (${e.metadata.gridType}, ${e.metadata.voxelCount} active, image ${KiB(e.metadata.gridByteSize)})`),
    bakeS,
    full,
  };
}

for (const [codec, r] of Object.entries(results)) {
  console.log(`\ncodec ${codec} (bake ${r.bakeS.toFixed(1)} s): ${r.names.join(" -> ")}`);
  console.log(`  file ${KiB(r.total)}; gzip-9 of the file ${KiB(r.gzTotal)}`);
  console.log(`  first paint, raw transfer:  ${KiB(r.previewEnd)} (${r.previewEnd} B)`);
  console.log(`  first paint, gzip transfer: ${KiB(r.gzFirst)} (${r.gzFirst} B, 64-byte steps)`);
}
console.log(`\ngate (his smoke LOD0): 14.8 KiB (15155 B)`);

if (out) {
  writeFileSync(out, results[outCodec].bytes);
  console.log(`wrote ${out} (codec ${outCodec}, ${KiB(results[outCodec].total)})`);
}
