#!/usr/bin/env node
// K3.2: a `.wvdb` LOD → `.nvdb` with identical voxels (the dense-vs-sparse A/B input, R1 §6 Step 2).
//
// The density texels his renderer uploads for that LOD (`wvdbDenseVolume`: 8-bit clamp(v / scale, 0, 1))
// become a NanoVDB FLOAT grid voxel for voxel (`wvdbDenseToNvdb`: texel k → k / 255 as f32, texel
// (x, y, z) → index (x, y, z), zero texels inactive). The harness (examples/src/harness/k3-sparse.ts)
// converts in the browser; this writes the same grid to a file for native tools or other hosts.
//
//   pnpm build && node scripts/k3/wvdb-to-nvdb.mjs fixtures/k3/bunny_cloud.wvdb out.nvdb [--lod N] [--zip]
//
// Prints the dims, active voxels, the dense mean density and the grid's mean of non-zero voxels.

import { readFileSync, writeFileSync } from "node:fs";
import { readWvdb, wvdbDenseToNvdb, wvdbDenseVolume, writeNvdb } from "../../packages/vdb/dist/tools/index.js";

const args = process.argv.slice(2);
const positional = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--lod");
const [input, output] = positional;
if (!input || !output) {
  console.error("usage: node scripts/k3/wvdb-to-nvdb.mjs <in.wvdb> <out.nvdb> [--lod N] [--zip]");
  process.exit(2);
}
const lodAt = args.indexOf("--lod");
const lod = lodAt >= 0 ? Number(args[lodAt + 1]) : undefined;

const dense = wvdbDenseVolume(readWvdb(new Uint8Array(readFileSync(input))), { mipmaps: false, ...(lod !== undefined ? { lod } : {}) });
const grid = wvdbDenseToNvdb(dense);
writeFileSync(output, new Uint8Array(writeNvdb([grid.image], { codec: args.includes("--zip") ? "zip" : "none" })));
console.log(
  JSON.stringify(
    {
      input,
      output,
      lod: dense.lod,
      dimensions: dense.dimensions,
      textureIndexBox: grid.textureIndexBox,
      activeVoxels: grid.voxelCount,
      denseMeanDensity: dense.meanDensity,
      denseStatsSource: dense.densityStatsSource,
      imageBytes: grid.image.byteLength,
    },
    null,
    2,
  ),
);
