// Size comparison: NanoVDB float vs NanoVDB Fp8 vs PicoVDB, raw and gzipped.
//
//   node experiments/picovdb-spike/scripts/sizes.mjs
//
// Prints a markdown table (copied into docs/spikes/PICOVDB.md) and writes
// .out/sizes.json. All PicoVDB numbers come from src/convert.ts, which is
// byte-identical to the native converter (test/convert.test.ts).

import { gzipSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { SPIKE_DIR, tsLoader } from "./lib/load-ts.mjs";

const ts = await tsLoader();
const rows = [];
try {
  const fx = await ts.load("src/fixtures.ts");
  const cv = await ts.load("src/convert.ts");
  const cmp = await ts.load("src/compare.ts");
  const { inspect, quantize, writeNvdb } = await ts.load("../../packages/vdb-web-tools/src/index.ts");
  const gz = (b) => gzipSync(b, { level: 6 }).byteLength;

  for (const f of fx.loadAll()) {
    const fp8Image = quantize(f.image, "fp8");
    const fp8 = new Uint8Array(writeNvdb([fp8Image]));
    const fpn = new Uint8Array(writeNvdb([quantize(f.image, "fpn")]));
    const picoF32 = cv.convertToPicoVDB([{ image: f.image, gridClassId: f.gridClassId }]);
    const picoFixed = cv.convertToPicoVDB([{ image: f.image, gridClassId: f.gridClassId }], { layout: "fixed" });
    if (picoFixed.bytes.byteLength !== picoF32.bytes.byteLength) throw new Error("fixed layout changed size");
    const picoU8 = f.kind === "sdf" ? cv.convertToPicoVDB([{ image: f.image, gridClassId: f.gridClassId }], { valueType: "u8" }) : null;
    const act = cmp.enumerateActive(f.image);
    const d = picoF32.diagnostics;
    const nanoFloat = inspect(f.image);
    const nanoFp8 = inspect(fp8Image);
    rows.push({
      id: f.id,
      kind: f.kind,
      activeVoxels: nanoFloat.voxelCount,
      activeTiles: act.activeTiles,
      nodes: { upper: d.uppers, lower: d.lowers, leaf: d.leaves },
      nanoFloat: { file: f.nvdbFloat.byteLength, gz: gz(f.nvdbFloat), breakdown: nanoFloat.memoryBreakdown },
      nanoFp8Ts: { file: fp8.byteLength, gz: gz(fp8), breakdown: nanoFp8.memoryBreakdown },
      nanoFpnTs: { file: fpn.byteLength, gz: gz(fpn) },
      nanoFp8Native: f.nativeFp8 ? { file: f.nativeFp8.byteLength, gz: gz(f.nativeFp8) } : null,
      picoF32: {
        file: picoF32.bytes.byteLength,
        gz: gz(picoF32.bytes),
        breakdown: {
          headerGridsRoots: 32 + 64 + Math.ceil(d.roots / 2) * 2 * 8,
          uppers: d.uppers * cv.UPPER_SIZE,
          lowers: d.lowers * cv.LOWER_SIZE,
          leaves: d.leaves * cv.LEAF_SIZE,
          data: picoF32.bytes.byteLength - (32 + 64 + Math.ceil(d.roots / 2) * 2 * 8 + d.uppers * cv.UPPER_SIZE + d.lowers * cv.LOWER_SIZE + d.leaves * cv.LEAF_SIZE),
        },
        dataElems: d.dataElems,
      },
      picoU8: picoU8 ? { file: picoU8.bytes.byteLength, gz: gz(picoU8.bytes) } : null,
    });
  }
} finally {
  await ts.close();
}

const kb = (n) => (n / 1024).toFixed(0);
const pct = (a, b) => `${((a / b) * 100).toFixed(0)}%`;
console.log("| Fixture | Class | Active voxels | Nodes U/L/Leaf | NanoVDB float | NanoVDB Fp8 (TS) | NanoVDB Fp8 (native bake) | NanoVDB FpN (TS) | PicoVDB f32 | PicoVDB u8 | PicoVDB f32 ÷ Fp8 (TS) |");
console.log("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
for (const r of rows) {
  console.log(
    `| ${r.id} | ${r.kind} | ${r.activeVoxels.toLocaleString("en-US")} | ${r.nodes.upper}/${r.nodes.lower}/${r.nodes.leaf} ` +
      `| ${kb(r.nanoFloat.file)} KiB | ${kb(r.nanoFp8Ts.file)} KiB | ${r.nanoFp8Native ? kb(r.nanoFp8Native.file) + " KiB" : "—"} | ${kb(r.nanoFpnTs.file)} KiB ` +
      `| **${kb(r.picoF32.file)} KiB** | ${r.picoU8 ? kb(r.picoU8.file) + " KiB" : "n/a (no fog u8)"} | ${pct(r.picoF32.file, r.nanoFp8Ts.file)} |`,
  );
}
console.log("\nGzipped (level 6), as served over the wire:\n");
console.log("| Fixture | NanoVDB float .gz | NanoVDB Fp8 (TS) .gz | NanoVDB FpN (TS) .gz | PicoVDB f32 .gz | PicoVDB u8 .gz | PicoVDB f32 .gz ÷ Fp8 .gz |");
console.log("|---|--:|--:|--:|--:|--:|--:|");
for (const r of rows) {
  console.log(
    `| ${r.id} | ${kb(r.nanoFloat.gz)} KiB | ${kb(r.nanoFp8Ts.gz)} KiB | ${kb(r.nanoFpnTs.gz)} KiB | **${kb(r.picoF32.gz)} KiB** | ${r.picoU8 ? kb(r.picoU8.gz) + " KiB" : "—"} | ${pct(r.picoF32.gz, r.nanoFp8Ts.gz)} |`,
  );
}
console.log("\nBreakdown (KiB): NanoVDB Fp8 upper/lower/leaf vs PicoVDB f32 upper/lower/leaf/data:\n");
console.log("| Fixture | Fp8 upper | Fp8 lower | Fp8 leaf | Pico upper | Pico lower | Pico leaf masks | Pico values |");
console.log("|---|--:|--:|--:|--:|--:|--:|--:|");
for (const r of rows) {
  const b = r.nanoFp8Ts.breakdown;
  const p = r.picoF32.breakdown;
  const pick = (o, re) => Object.entries(o).filter(([k]) => re.test(k)).reduce((s, [, v]) => s + v, 0);
  console.log(
    `| ${r.id} | ${kb(pick(b, /upper/i))} | ${kb(pick(b, /lower/i))} | ${kb(pick(b, /leaf/i))} | ${kb(p.uppers)} | ${kb(p.lowers)} | ${kb(p.leaves)} | ${kb(p.data)} |`,
  );
}
mkdirSync(path.join(SPIKE_DIR, ".out"), { recursive: true });
writeFileSync(path.join(SPIKE_DIR, ".out/sizes.json"), JSON.stringify(rows, null, 2));
console.log("\nwrote .out/sizes.json");
