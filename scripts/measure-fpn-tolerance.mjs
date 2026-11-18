#!/usr/bin/env node
/**
 * measure-fpn-tolerance.mjs — K4.11: our FpN encoder vs native `nanovdb_convert` FpN on the fog primitives.
 *
 *   pnpm build && node scripts/measure-fpn-tolerance.mjs [--tol 0.01,0.001,...]
 *
 * For each primitive: loads the native float grid, the native FpN grid, and our `vdbQuantize(float, 'fpn', tol)`;
 * reports bytes, the per-leaf bit-width histogram, and max / mean abs error against the float source
 * (every voxel of the index bbox, active or not). Needs fixtures/primitives/*.nvdb (`pnpm fixtures:bake`).
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const T = await import(path.join(root, "packages/vdb/dist/tools/index.js"));
const B = await import(path.join(root, "packages/vdb/dist/tools/nanovdb/bytes.js"));
const args = process.argv.slice(2);
const ti = args.indexOf("--tol");
const tols = ti >= 0 ? args[ti + 1].split(",").map(Number) : [0.01, 0.005, 0.002, 0.001, 0.0005];

function load(file) {
  const buf = readFileSync(file);
  const f = T.NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  return f.gridImage(0);
}

/** Per-leaf log2 bit widths of an FpN image, walking the variable-size leaf block. */
function leafBits(image) {
  const v = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const tree = B.GRID_SIZE;
  let off = tree + Number(v.getBigUint64(tree + B.TREE_OFF_NODE_OFFSET_LEAF, true));
  const n = v.getUint32(tree + B.TREE_OFF_NODE_COUNT_LEAF, true);
  const hist = [0, 0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    const log = v.getUint8(off + B.LEAF_OFF_BBOX_DIF_AND_FLAGS + 3) >> 5;
    hist[log]++;
    off += B.FP_LEAF_HEADER_SIZE + (1 << log) * 64;
  }
  return hist;
}

function errors(src, enc, bbox) {
  let max = 0, sum = 0, n = 0, maxActive = 0, sumActive = 0, nActive = 0;
  const [[x0, y0, z0], [x1, y1, z1]] = bbox;
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) {
    const a = T.vdbReadValueCpu(src, [x, y, z]);
    const b = T.vdbReadValueCpu(enc, [x, y, z]);
    const d = Math.abs(a.value - b.value);
    sum += d; n++; if (d > max) max = d;
    if (a.active) { sumActive += d; nActive++; if (d > maxActive) maxActive = d; }
  }
  return { max, mean: sum / n, maxActive, meanActive: sumActive / nActive, nActive };
}

for (const prim of ["sphere", "torus", "box"]) {
  const dir = path.join(root, "fixtures/primitives");
  const fl = path.join(dir, `${prim}_fog_float.nvdb`), fp = path.join(dir, `${prim}_fog_fpn.nvdb`);
  if (!existsSync(fl) || !existsSync(fp)) { console.log(`skip ${prim}: fixture missing`); continue; }
  const src = load(fl), native = load(fp);
  const side = JSON.parse(readFileSync(path.join(dir, `${prim}_fog_fpn.sidecar.json`), "utf8"));
  const bbox = side.grid.indexBBox;
  const entries = [["native fpn", native], ...tols.map((t) => [`ours tol=${t}`, T.vdbQuantize(src, "fpn", t)])];
  for (const [label, img] of entries) {
    const e = errors(src, img, bbox);
    const bits = leafBits(img).join("/");
    console.log(`${prim.padEnd(6)} ${label.padEnd(18)} ${String(img.byteLength).padStart(8)} B  leaves(1/2/4/8/16)=${bits.padEnd(14)} maxAbs=${e.max.toExponential(3)} meanAbs=${e.mean.toExponential(3)} | active: max=${e.maxActive.toExponential(3)} mean=${e.meanActive.toExponential(3)}`);
  }
  console.log(`${prim.padEnd(6)} float source           ${src.byteLength} B`);
}
