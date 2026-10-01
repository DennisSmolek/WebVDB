/**
 * Oracle (b): src/convert.ts is byte-identical to upstream's native Zig
 * converter. Goldens in native-golden.json were recorded by
 * scripts/native-oracle.mjs from a native build at the pinned commit.
 *
 * Also: upstream's own TS loader (vendor/picovdb.ts PicoVDBFile) accepts our
 * output, and the converter's scope errors fire.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { PicoVDBFile } from "../vendor/picovdb.js";
import { convertToPicoVDB } from "../src/convert.js";
import { PicoVDBView } from "../src/reader.js";
import { loadPrimitive, primitivesAvailable, samplesAvailable } from "../src/fixtures.js";
import { converted, corpus, variants } from "./helpers.js";
import { NanoVDBFile } from "../../../packages/nanovdb-wgsl/src/nvdb-file.js";

const golden = JSON.parse(readFileSync(new URL("./native-golden.json", import.meta.url), "utf8")) as Record<
  string,
  { inputSha256: string } & Partial<Record<"f32" | "u8", { sha256: string; bytes: number }>>
>;
const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
const haveCorpus = primitivesAvailable() && samplesAvailable();

describe.skipIf(!haveCorpus)("TS converter vs native picovdb (byte-for-byte)", () => {
  it("the corpus is the one the goldens were recorded from", () => {
    const ids = corpus().map((f) => f.id);
    expect(ids).toEqual(Object.keys(golden).filter((k) => k !== "$meta"));
    for (const f of corpus()) expect(sha(f.nvdbFloat), f.id).toBe(golden[f.id]!.inputSha256);
  });

  const cases = () => corpus().flatMap((f) => variants(f).map((vt) => [f, vt] as const));
  it("every fixture x value type is byte-identical to the native output", () => {
    const results = cases().map(([f, vt]) => {
      const { bytes } = converted(f, { valueType: vt });
      const g = golden[f.id]![vt]!;
      return { id: `${f.id} ${vt}`, size: bytes.byteLength === g.bytes, sha: sha(bytes) === g.sha256 };
    });
    expect(results.length).toBe(10);
    for (const r of results) expect(r, r.id).toEqual({ id: r.id, size: true, sha: true });
  });

  it("native layout reports the tile mis-indexing it reproduces (primitives only)", () => {
    for (const f of corpus()) {
      const d = converted(f).diagnostics;
      expect(d.childlessRootTiles, f.id).toBe(0);
      if (f.id.startsWith("primitives")) expect(d.misindexedTileValues, f.id).toBeGreaterThan(0);
      else expect(d.misindexedTileValues, f.id).toBe(0);
    }
  });

  it('"fixed" layout changes value order only: same size, same masks/node records', () => {
    for (const f of corpus().filter((x) => x.id.startsWith("primitives"))) {
      const a = converted(f).bytes;
      const b = converted(f, { layout: "fixed" }).bytes;
      expect(b.byteLength).toBe(a.byteLength);
      const va = new PicoVDBView(a);
      const vb = new PicoVDBView(b);
      // Leaf records may differ in baseActiveIndex; masks and packed indices must not.
      for (let i = 0; i < va.leafCount; i++) {
        const oa = va.off.leaves + i * 52;
        const ob = vb.off.leaves + i * 52;
        expect(Array.from(vb.u32.subarray(ob + 4, ob + 52))).toEqual(Array.from(va.u32.subarray(oa + 4, oa + 52)));
      }
      expect(convertToPicoVDB([{ image: f.image }], { layout: "fixed" }).diagnostics.misindexedTileValues).toBe(0);
    }
  });

  it("upstream's own loader (vendor/picovdb.ts) parses every output consistently with our view", () => {
    for (const f of corpus()) {
      for (const vt of variants(f)) {
        const bytes = converted(f, { valueType: vt }).bytes;
        const file = new PicoVDBFile(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
        const view = new PicoVDBView(bytes);
        const s = view.bindingSlices();
        expect(file.getSize()).toBe(bytes.byteLength);
        expect(file.rootsBuffer.byteLength).toBe(s.roots.byteLength); // padded: what WGSL arrayLength sees
        expect(file.uppersBuffer.byteLength).toBe(s.uppers.byteLength);
        expect(file.leavesBuffer.byteLength).toBe(s.leaves.byteLength);
        expect(file.dataBuffer.byteLength).toBe(s.data.byteLength);
        const g = file.getGrid(0);
        expect(g.gridType).toBe(view.grid(0).gridType);
        expect(Array.from(g.indexBoundsMin)).toEqual(f.indexBBox.min);
        expect(Array.from(g.indexBoundsMax)).toEqual(f.indexBBox.max);
        expect(file.getVoxelCount()).toBe(view.grid(0).dataElemCount - 2);
      }
    }
  });
});

describe.skipIf(!primitivesAvailable())("converter scope (same as native, but loud)", () => {
  it("rejects quantized NanoVDB sources (native would silently read Fp8 bytes as f32)", () => {
    const fp8 = NanoVDBFile.fromArrayBuffer(loadPrimitive("sphere").nativeFp8!.buffer);
    expect(() => convertToPicoVDB([{ image: fp8.gridImage(0) }])).toThrow(/not FLOAT/);
  });

  it("rejects fog + u8 (native tags FOG_FLOAT but writes level-set u8 bytes)", () => {
    const f = loadPrimitive("box");
    expect(() => convertToPicoVDB([{ image: f.image }], { valueType: "u8" })).toThrow(/fog \+ u8/);
  });

  it("multi-grid files: second grid starts 16-byte aligned with grid-relative indices", () => {
    const f = loadPrimitive("box");
    const { bytes, diagnostics } = convertToPicoVDB([{ image: f.image }, { image: f.image }], { layout: "fixed" });
    expect(diagnostics.grids).toBe(2);
    const v = new PicoVDBView(bytes);
    const g0 = v.grid(0);
    const g1 = v.grid(1);
    expect(g1.upperStart).toBe(8);
    expect(g1.dataStart).toBeGreaterThan(g0.dataStart);
    expect(g1.dataElemCount).toBe(g0.dataElemCount);
  });
});
