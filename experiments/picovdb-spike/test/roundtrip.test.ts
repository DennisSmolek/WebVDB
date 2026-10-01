/**
 * Oracle (a): every active voxel (plus corner/centre samples of every active
 * tile) read back through the WGSL-mirroring CPU reader matches our proven
 * NanoVDB readValue on the source grid, under PicoVDB's value mapping
 * (src/compare.ts). Random probeCoords probes add inactive/background coverage.
 *
 * Documents the upstream defect: native layout mis-reads active internal
 * tiles that share a node with children (primitives). The format-compatible
 * "fixed" layout reads back exactly.
 */

import { describe, expect, it } from "vitest";

import { comparePico, enumerateActive, type Coord } from "../src/compare.js";
import { primitivesAvailable, samplesAvailable } from "../src/fixtures.js";
import { probeCoords } from "../../../packages/nanovdb-wgsl/src/cpu/probe-coords.js";
import { converted, corpus, variants } from "./helpers.js";

const haveCorpus = primitivesAvailable() && samplesAvailable();
const HALF_STEP_U8 = 3 / 255 + 1e-6;

describe.skipIf(!haveCorpus)("round-trip vs NanoVDB readValue", () => {
  it("vdb-samples: every active voxel, f32 exact / u8 within half a step (native layout)", () => {
    for (const f of corpus().filter((x) => x.id.startsWith("vdb-samples"))) {
      const act = enumerateActive(f.image);
      expect(act.activeTiles, f.id).toEqual({ root: 0, upper: 0, lower: 0 }); // our builder is leaf-only
      for (const vt of variants(f)) {
        const s = comparePico(f.image, converted(f, { valueType: vt }).bytes, act.leafVoxels, { voxelSize: f.voxelSize });
        expect(s.checked, f.id).toBe(act.leafVoxels.length);
        expect(s.valueMismatch, `${f.id} ${vt}`).toBe(0);
        expect(s.activeMismatch, `${f.id} ${vt}`).toBe(0);
        if (vt === "f32") expect(s.maxAbsDelta).toBe(0);
        else expect(s.maxAbsDelta).toBeLessThanOrEqual(HALF_STEP_U8);
      }
    }
  });

  it("primitives (fog with active lower tiles): native layout mis-reads tile values, leaf voxels stay exact", () => {
    for (const f of corpus().filter((x) => x.id.startsWith("primitives"))) {
      const act = enumerateActive(f.image);
      expect(act.activeTiles.lower, f.id).toBeGreaterThan(0);
      const bytes = converted(f).bytes;
      const leaves = comparePico(f.image, bytes, act.leafVoxels, { voxelSize: f.voxelSize });
      expect(leaves.valueMismatch, f.id).toBe(0);
      const tiles = comparePico(f.image, bytes, act.tileSamples, { voxelSize: f.voxelSize });
      expect(tiles.valueMismatch, f.id).toBeGreaterThan(0);
      expect(tiles.activeMismatch, f.id).toBe(0); // topology is right; only the value slot is wrong
    }
  });

  it('primitives: "fixed" layout reads every active voxel and tile sample back exactly', () => {
    for (const f of corpus().filter((x) => x.id.startsWith("primitives"))) {
      const act = enumerateActive(f.image);
      const all: Coord[] = [...act.leafVoxels, ...act.tileSamples];
      const s = comparePico(f.image, converted(f, { layout: "fixed" }).bytes, all, { voxelSize: f.voxelSize });
      expect(s.checked).toBe(all.length);
      expect(s.valueMismatch, f.id).toBe(0);
      expect(s.activeMismatch, f.id).toBe(0);
      expect(s.maxAbsDelta).toBe(0);
    }
  });

  it("random probes (active + inactive + outside bbox) match the mapped source value", () => {
    for (const f of corpus()) {
      const coords = probeCoords({ seed: 0x5eedn, count: 20_000, bboxMin: f.indexBBox.min, bboxMax: f.indexBBox.max });
      const layout = f.id.startsWith("primitives") ? "fixed" : "native";
      for (const vt of variants(f)) {
        const s = comparePico(f.image, converted(f, { valueType: vt, layout }).bytes, coords, { voxelSize: f.voxelSize });
        expect(s.valueMismatch, `${f.id} ${vt}`).toBe(0);
        expect(s.activeMismatch, `${f.id} ${vt}`).toBe(0);
        // PicoVDB keeps no inactive values. For the baked fog primitives the
        // root background is 3 while inactive voxels hold 0, so every inactive
        // probe is "lossy" (it reads 3). The smoke sample (background 0) is lossless.
        if (f.id === "vdb-samples/smoke") expect(s.inactiveLossy).toBe(0);
        if (f.id.startsWith("primitives")) expect(s.inactiveLossy).toBe(s.checked - s.active);
      }
    }
  });
});

describe("WGSL root search reads the even-count padding root (upstream defect)", () => {
  it("a 1-root grid away from the origin resolves [0,4096)^3 lookups through a nonexistent upper[1]", async () => {
    const { buildFromDense } = await import("../../../packages/vdb-web-tools/src/index.js");
    const { convertToPicoVDB } = await import("../src/convert.js");
    const { PicoVDBAccessor, PicoVDBView } = await import("../src/reader.js");
    const image = buildFromDense(new Float32Array(512).fill(0.5), [8, 8, 8], { origin: [-100, -100, -100] });
    const view = new PicoVDBView(convertToPicoVDB([{ image }]).bytes);
    expect(view.upperCount).toBe(1);
    expect(view.rootArrayLength).toBe(2); // ts/picovdb.ts rootsBuffer includes the zero-key padding root
    // Correct answer: root miss (level 3, background). The WGSL search matches
    // the padding root's key (0,0) and descends into upper index 1, past the
    // end of the uppers binding (robust-access territory on GPU).
    expect(new PicoVDBAccessor(view).read([5, 5, 5]).level).toBe(2);
    expect(new PicoVDBAccessor(view).read([-5, 5, 5]).level).toBe(3);
  });
});
