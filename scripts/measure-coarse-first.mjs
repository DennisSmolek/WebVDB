#!/usr/bin/env node
/**
 * measure-coarse-first.mjs — K4.4/K4.5 numbers: resample sizes and errors, and
 * bytes to first paint for the coarse-first `.nvdb` delivery options
 * (docs/design/K4-5-coarse-first.md).
 *
 *   pnpm build && node scripts/measure-coarse-first.mjs [smoke|wdas|all]
 *
 * Needs fixtures/vdb-samples/smoke.vdb (`pnpm fixtures:vdb-samples`) and
 * fixtures/wdas/wdas_cloud_quarter.vdb (`pnpm fixtures:wdas`). Sizes are gzip
 * level 9 (Node zlib) of the `.nvdb` file, standing in for HTTP compression as in
 * the comparison report (research/INDEX). Errors are `vdbResampleError`: mean |diff|
 * over every 7th active source voxel as % of the mean, target read back with
 * `vdbSampleTrilinear`.
 *
 * "Stream offset" = how many gzip bytes of the whole multi-grid file a streaming
 * decoder must receive before the first grid (file header + its metadata + image)
 * is fully decoded — the real first-paint cost of option (a). Measured by pushing
 * the compressed file through fflate's streaming Gunzip in 64-byte steps.
 */
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// fflate is a dependency of packages/vdb (pnpm does not hoist it to the root).
const { Gunzip } = createRequire(path.join(root, "packages/vdb/package.json"))("fflate");
const T = await import(path.join(root, "packages/vdb/dist/tools/internal.js"));

const KiB = (n) => (n / 1024).toFixed(1);
const gz = (bytes) => gzipSync(bytes, { level: 9 });
const nvdb = (images) => new Uint8Array(T.writeNvdb(images));

/** gzip bytes a streaming decoder needs before `prefixBytes` of output exist. */
function streamOffset(compressed, prefixBytes, step = 64) {
  let out = 0;
  let found = -1;
  const g = new Gunzip((chunk) => {
    out += chunk.length;
  });
  for (let off = 0; off < compressed.length; off += step) {
    const end = Math.min(off + step, compressed.length);
    g.push(compressed.subarray(off, end), end === compressed.length);
    if (out >= prefixBytes) {
      found = end;
      break;
    }
  }
  return found;
}

/** Bytes before the first leaf of a grid image (GridData + Tree + Root + uppers + lowers). */
function topologyBytes(image) {
  const nodes = T.treeNodes(image);
  const leaves = nodes.filter((n) => n.level === 0);
  const first = Math.min(...leaves.map((n) => n.address));
  return {
    bytes: first,
    upper: nodes.filter((n) => n.level === 2).length,
    lower: nodes.filter((n) => n.level === 1).length,
    leaf: leaves.length,
  };
}

/** Sum of all leaf values and active tile values times voxel volume (fog "mass"). */
function mass(image, voxelSize) {
  let s = 0;
  for (const n of T.treeNodes(image)) {
    for (const t of n.activeTiles) s += t.value * t.dim ** 3;
    if (n.level !== 0) continue;
    const { values } = T.leafVoxels(image, n.address);
    for (const v of values) s += v;
  }
  return s * voxelSize[0] * voxelSize[1] * voxelSize[2];
}

/**
 * Option (c) proxy: what the topology alone can draw. A voxel reads its lower
 * node's average if its 8^3 block has a leaf (lower child mask), else the
 * background: a blocky silhouette at leaf resolution. Same error metric.
 */
function topologyProxyError(grid, image, stride = 7) {
  const nodes = T.treeNodes(image);
  const leafBlocks = new Set();
  const lowerAve = new Map();
  for (const n of nodes) {
    if (n.level === 0) leafBlocks.add(`${n.origin[0] >> 3},${n.origin[1] >> 3},${n.origin[2] >> 3}`);
    if (n.level === 1) lowerAve.set(`${n.origin[0] >> 7},${n.origin[1] >> 7},${n.origin[2] >> 7}`, T.nodeStats(image, 1, n.address).ave);
  }
  let k = 0;
  let sumAbs = 0;
  let sumVal = 0;
  let count = 0;
  const leaves = [...grid.iterLeaves()].sort((p, q) => p.origin[0] - q.origin[0] || p.origin[1] - q.origin[1] || p.origin[2] - q.origin[2]);
  for (const leaf of leaves) {
    for (let n = 0; n < 512; n++) {
      if (((leaf.valueMask[n >>> 5] >>> (n & 31)) & 1) === 0) continue;
      if (k++ % stride !== 0) continue;
      const o = leaf.origin;
      const proxy = leafBlocks.has(`${o[0] >> 3},${o[1] >> 3},${o[2] >> 3}`)
        ? lowerAve.get(`${o[0] >> 7},${o[1] >> 7},${o[2] >> 7}`)
        : 0;
      sumAbs += Math.abs(proxy - leaf.values[n]);
      sumVal += leaf.values[n];
      count++;
    }
  }
  return (100 * sumAbs) / sumVal;
}

/** mjurczyk's `.wvdb` of the same grid: bytes to each LOD and its error (same metric, texture-style trilinear). */
function wvdbBaseline(grid, bytes, stride = 7) {
  const { deflateSync } = createRequire(import.meta.url)("node:zlib");
  const { bytes: file } = T.vdbToWvdb(bytes, { dimensions: [128, 128, 128], compress: (b) => deflateSync(b, { level: 9 }) });
  const info = T.readWvdbHeader(file, { fileLength: file.byteLength });
  const wv = T.readWvdb(file);
  const g = wv.grids[0];
  const [min, max] = g.bbox;
  const size = [0, 1, 2].map((a) => max[a] - min[a]);
  const m = grid.transform.matrix;
  const leaves = [...grid.iterLeaves()].sort((p, q) => p.origin[0] - q.origin[0] || p.origin[1] - q.origin[1] || p.origin[2] - q.origin[2]);
  const rows = [];
  for (const level of g.levels) {
    const [X, Y, Z] = level.dimensions;
    const at = (x, y, z) => level.values[x + X * (y + Y * z)];
    let k = 0;
    let sumAbs = 0;
    let sumVal = 0;
    for (const leaf of leaves) {
      for (let n = 0; n < 512; n++) {
        if (((leaf.valueMask[n >>> 5] >>> (n & 31)) & 1) === 0) continue;
        if (k++ % stride !== 0) continue;
        const ijk = [leaf.origin[0] + (n >> 6), leaf.origin[1] + ((n >> 3) & 7), leaf.origin[2] + (n & 7)];
        const u = [0, 1, 2].map((a) => {
          const w = ijk[a] * m[a * 5] + m[a * 4 + 3];
          const d = [X, Y, Z][a];
          return Math.min(Math.max(((w - min[a]) / size[a]) * d - 0.5, 0), d - 1);
        });
        const b = u.map(Math.floor);
        const f = u.map((v, a) => v - b[a]);
        const c = (a, o) => Math.min(b[a] + o, [X, Y, Z][a] - 1);
        let v = 0;
        for (let dz = 0; dz < 2; dz++)
          for (let dy = 0; dy < 2; dy++)
            for (let dx = 0; dx < 2; dx++)
              v += (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]) * at(c(0, dx), c(1, dy), c(2, dz));
        sumAbs += Math.abs(v - leaf.values[n]);
        sumVal += leaf.values[n];
      }
    }
    rows.push({ lod: level.lod, dims: level.dimensions.join("x"), prefix: T.wvdbPrefixLength(info, level.lod), err: (100 * sumAbs) / sumVal });
  }
  return { total: file.byteLength, rows };
}

function loadGrid(file) {
  const buf = readFileSync(file);
  const t = performance.now();
  const vdb = T.parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const grid = vdb.grids.find((g) => g.valueType === "float");
  console.log(`parsed ${path.basename(file)} "${grid.name}" in ${((performance.now() - t) / 1000).toFixed(1)} s`);
  return { grid, bytes: new Uint8Array(buf) };
}

function run(label, file, { resolutions, hisBox = false, filters = ["point", "box", "trilinear"] }) {
  if (!existsSync(file)) {
    console.log(`skip ${label}: ${file} missing`);
    return;
  }
  console.log(`\n## ${label}`);
  const { grid, bytes } = loadGrid(file);
  let t = performance.now();
  const full = T.buildFromVdbDetailed(grid);
  const fullFpn = T.vdbQuantizeDetailed(full, "fpn");
  const fullFp8 = T.vdbQuantizeDetailed(full, "fp8");
  const fullBytes = nvdb([fullFpn.image]);
  const fullGz = gz(fullBytes);
  const srcMass = mass(full.image, full.voxelSizeXYZ);
  console.log(
    `full: active ${full.voxelCount}, leaves ${full.nodeCounts.leaf}, tiles ${JSON.stringify(full.tileCounts)}, float gz ${KiB(gz(nvdb([full.image])).length)} KiB, ` +
      `fp8 gz ${KiB(gz(nvdb([fullFp8.image])).length)} KiB, fpn raw ${KiB(fullBytes.length)} KiB gz ${KiB(fullGz.length)} KiB ` +
      `(${((performance.now() - t) / 1000).toFixed(1)} s); fpn err ${T.vdbResampleError(grid, fullFpn.image).percent.toFixed(2)}%`,
  );

  // (c)/(d): what a range read or a gzip stream of the full grid gets before its first leaf.
  const topo = topologyBytes(fullFpn.image);
  const headerBytes = 16 + 176 + new TextEncoder().encode(fullFpn.gridName).length + 1;
  console.log(
    `topology (c/d): ${topo.upper} upper, ${topo.lower} lower, ${topo.leaf} leaves; prefix before first leaf ` +
      `${KiB(headerBytes + topo.bytes)} KiB raw, gzip stream offset ${KiB(streamOffset(fullGz, headerBytes + topo.bytes))} KiB, ` +
      `leaf-stats-only payload ${KiB(topo.leaf * 16)} KiB raw (16 B per leaf, scattered); ` +
      `topology proxy err ${topologyProxyError(grid, full.image).toFixed(1)}%`,
  );

  if (hisBox) {
    const wv = wvdbBaseline(grid, bytes);
    console.log(`his .wvdb (our byte-exact re-bake, Node zlib 9): ${KiB(wv.total)} KiB`);
    for (const r of wv.rows) console.log(`  LOD${r.lod} ${r.dims}: first ${KiB(r.prefix)} KiB, err ${r.err.toFixed(1)}%`);
    const box = T.wvdbSourceFromVdbGrid(grid).bbox;
    for (const filter of filters) {
      t = performance.now();
      for (const d of [32, 64]) {
        const c = T.vdbResampleGrid(grid, { dimensions: [d, d, d], bbox: box, filter, quantize: "fpn" });
        const ce = T.vdbResampleError(grid, c.image);
        console.log(`his ${d}^3 box, ${filter}, fpn: active ${c.voxelCount}, gz ${KiB(gz(nvdb([c.image])).length)} KiB, err ${ce.percent.toFixed(1)}%`);
      }
      const r = T.vdbResampleGrid(grid, { dimensions: [128, 128, 128], bbox: box, filter, quantize: "fpn" });
      const e = T.vdbResampleError(grid, r.image);
      console.log(
        `his 128^3 box, ${filter}, fpn: active ${r.voxelCount}, gz ${KiB(gz(nvdb([r.image])).length)} KiB, ` +
          `err ${e.percent.toFixed(2)}% (n ${e.count}), ${((performance.now() - t) / 1000).toFixed(2)} s`,
      );
    }
  }

  console.log("| resolution | filter | quant | dims | active | preview raw KiB | preview gz KiB | (a) stream offset KiB | (a) whole file gz KiB | err % | mass ratio |");
  console.log("|---|---|---|---|---|---|---|---|---|---|---|");
  for (const resolution of resolutions) {
    for (const filter of filters) {
      for (const quant of ["fp8", "fpn"]) {
        const r = T.vdbResampleGrid(grid, { resolution, filter, quantize: quant });
        const previewFile = nvdb([r.image]);
        const previewGz = gz(previewFile).length;
        const multi = nvdb([r.image, fullFpn.image]);
        const multiGz = gz(multi);
        // header + both FileMetaData + names (io::Segment writes all metadata first, K4.16) + preview image
        const prefix = previewFile.length + 176 + new TextEncoder().encode(fullFpn.gridName).length + 1;
        const offset = streamOffset(multiGz, prefix);
        const e = T.vdbResampleError(grid, r.image);
        const m = mass(r.image, r.voxelSizeXYZ) / srcMass;
        console.log(
          `| ${resolution} | ${filter} | ${quant} | ${r.dimensions.join("x")} | ${r.voxelCount} | ${KiB(previewFile.length)} | ${KiB(previewGz)} | ` +
            `${KiB(offset)} | ${KiB(multiGz.length)} | ${e.percent.toFixed(1)} | ${m.toFixed(3)} |`,
        );
      }
    }
  }
}

const which = process.argv[2] ?? "all";
if (which === "smoke" || which === "all") {
  run("smoke.vdb", path.join(root, "fixtures/vdb-samples/smoke.vdb"), {
    resolutions: [16, 24, 32, 48, 64],
    hisBox: true,
  });
}
if (which === "wdas" || which === "all") {
  run("wdas_cloud_quarter.vdb", path.join(root, "fixtures/wdas/wdas_cloud_quarter.vdb"), {
    resolutions: [16, 24, 32, 48, 64, 128],
    filters: ["point", "box"],
  });
}
