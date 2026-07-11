import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { probePoints } from "../src/cpu/probe-coords.js";
import { readValue } from "../src/cpu/read-value.js";
import { sampleTrilinear } from "../src/cpu/sample-trilinear.js";
import { defineNumber, gridTypeConstantsFor } from "../src/cpu/stride-tables.js";

/**
 * Ground-truth checks for the pure-TS CPU trilinear sampler (`src/cpu/`)
 * against the baked `sphere_fog_float` fixture. This anchors the sampling
 * math the Phase 2 WGSL suite will also be checked against — see the
 * convention doc comment in `src/cpu/sample-trilinear.ts`.
 *
 * `.nvdb` slicing here mirrors `test/cpu-read-value.test.ts`'s test-local
 * `extractGridImage` (a stand-in for the concurrently-developed
 * `NanoVDBFile` loader, src/nvdb-file.ts).
 */

const FILE_HEADER_SIZE = 16;
const FILE_METADATA_SIZE = 176;
const FILE_METADATA_OFF_GRID_SIZE = 0;
const FILE_METADATA_OFF_NAME_SIZE = 136;

const fixturesDir = new URL("../../../fixtures/primitives/", import.meta.url);
const fixturesPresent = existsSync(fixturesDir);

interface Sample {
  ijk: [number, number, number];
  value: number;
  active: boolean;
}

interface Sidecar {
  grid: { type: string; indexBBox: [[number, number, number], [number, number, number]] };
  samples: Sample[];
}

/** Test-local `.nvdb` parser — a stand-in for `NanoVDBFile` (Phase 1). */
function extractGridImage(buffer: ArrayBuffer): Uint32Array {
  const view = new DataView(buffer);

  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== "NanoVDB2") {
    throw new Error(`extractGridImage: unexpected FileHeader magic ${JSON.stringify(magic)}`);
  }
  const gridCount = view.getUint16(12, true);
  const codec = view.getUint16(14, true);
  if (gridCount !== 1) {
    throw new Error(`extractGridImage: expected a single-grid fixture, got gridCount=${gridCount}`);
  }
  if (codec !== 0) {
    throw new Error(`extractGridImage: expected Codec NONE, got codec=${codec}`);
  }

  const metaOffset = FILE_HEADER_SIZE;
  const gridSize = view.getBigUint64(metaOffset + FILE_METADATA_OFF_GRID_SIZE, true);
  const nameSize = view.getUint32(metaOffset + FILE_METADATA_OFF_NAME_SIZE, true);
  const gridNameOffset = metaOffset + FILE_METADATA_SIZE;
  const gridImageOffset = gridNameOffset + nameSize;
  const gridImageByteLength = Number(gridSize);

  const copy = new Uint8Array(gridImageByteLength);
  copy.set(new Uint8Array(buffer, gridImageOffset, gridImageByteLength));
  const words = new Uint32Array(copy.buffer);

  const gridMagicAscii = new TextDecoder().decode(copy.subarray(0, 8));
  if (gridMagicAscii !== "NanoVDB1") {
    throw new Error(
      `extractGridImage: grid image does not start with a NanoVDB magic (got ${JSON.stringify(gridMagicAscii)})`,
    );
  }

  return words;
}

async function loadFixture(name: string): Promise<{ words: Uint32Array; sidecar: Sidecar }> {
  const nvdbBuffer = await readFile(new URL(`${name}.nvdb`, fixturesDir));
  const sidecar = JSON.parse(
    await readFile(new URL(`${name}.sidecar.json`, fixturesDir), "utf8"),
  ) as Sidecar;
  const words = extractGridImage(
    nvdbBuffer.buffer.slice(nvdbBuffer.byteOffset, nvdbBuffer.byteOffset + nvdbBuffer.byteLength),
  );
  return { words, sidecar };
}

describe.skipIf(!fixturesPresent)("cpu trilinear sample vs. baked ground-truth (sphere_fog_float)", () => {
  it("at integer coords, sampleTrilinear equals readValue's value for a dozen sidecar coords", async () => {
    const { words, sidecar } = await loadFixture("sphere_fog_float");
    const dozen = sidecar.samples.slice(0, 12);
    expect(dozen.length).toBe(12);

    for (const sample of dozen) {
      const expected = readValue(words, sample.ijk).value;
      const got = sampleTrilinear(words, sample.ijk);
      expect(got, `ijk=${JSON.stringify(sample.ijk)}`).toBeCloseTo(expected, 5);
    }
  });

  it("the midpoint between two axis-neighbor voxels equals the average of their readValue values", async () => {
    const { words } = await loadFixture("sphere_fog_float");

    const pairs: Array<[[number, number, number], [number, number, number]]> = [
      [[0, 0, 0], [1, 0, 0]],
      [[0, 0, 0], [0, 1, 0]],
      [[0, 0, 0], [0, 0, 1]],
      [[10, 10, 10], [11, 10, 10]],
      [[10, 10, 10], [10, 11, 10]],
      [[10, 10, 10], [10, 10, 11]],
      [[-20, 5, 5], [-19, 5, 5]],
      [[5, -20, 5], [5, -19, 5]],
      [[49, 0, 0], [50, 0, 0]], // straddles the bbox edge (partly background)
    ];

    for (const [a, b] of pairs) {
      const va = readValue(words, a).value;
      const vb = readValue(words, b).value;
      const midpoint: [number, number, number] = [
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
        (a[2] + b[2]) / 2,
      ];
      const got = sampleTrilinear(words, midpoint);
      expect(got, `midpoint of ${JSON.stringify(a)} / ${JSON.stringify(b)}`).toBeCloseTo(
        (va + vb) / 2,
        5,
      );
    }
  });

  it("interpolation is bounded by the min/max of its 8 taps (property test, ~200 random points)", async () => {
    const { words, sidecar } = await loadFixture("sphere_fog_float");
    const [mn, mx] = sidecar.grid.indexBBox;

    const points = probePoints({
      seed: 0xc0ffee1234567890n,
      count: 200,
      bboxMin: mn,
      bboxMax: mx,
      dilate: 4,
    });

    for (const xyz of points) {
      const bx = Math.floor(xyz[0]);
      const by = Math.floor(xyz[1]);
      const bz = Math.floor(xyz[2]);

      let lo = Infinity;
      let hi = -Infinity;
      for (const dx of [0, 1] as const) {
        for (const dy of [0, 1] as const) {
          for (const dz of [0, 1] as const) {
            const v = readValue(words, [bx + dx, by + dy, bz + dz]).value;
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
          }
        }
      }

      const got = sampleTrilinear(words, xyz);
      const epsilon = 1e-4;
      expect(got, `xyz=${JSON.stringify(xyz)} got=${got} lo=${lo} hi=${hi}`).toBeGreaterThanOrEqual(
        lo - epsilon,
      );
      expect(got, `xyz=${JSON.stringify(xyz)} got=${got} lo=${lo} hi=${hi}`).toBeLessThanOrEqual(
        hi + epsilon,
      );
    }
  });

  it("a deep-outside point returns the grid's background value", async () => {
    const { words } = await loadFixture("sphere_fog_float");

    // Independently read root_off_background (mirrors the equivalent check in
    // cpu-read-value.test.ts) rather than assuming background is 0 — for this
    // fog fixture it is not (see below).
    const gt = gridTypeConstantsFor("FLOAT");
    const treeAddress = defineNumber("PNANOVDB_GRID_SIZE");
    const rootOffOff = defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT");
    const rootOffsetLo = words[(treeAddress + rootOffOff) / 4]!;
    const rootOffsetHi = words[(treeAddress + rootOffOff) / 4 + 1]!;
    const rootOffset = Number((BigInt(rootOffsetHi) << 32n) | BigInt(rootOffsetLo));
    const rootAddress = treeAddress + rootOffset;
    const f32 = new Float32Array(words.buffer, words.byteOffset, words.length);
    const background = f32[(rootAddress + gt.root_off_background) / 4]!;

    const deepOutside: [number, number, number] = [1_000_000.37, 1_000_000.5, 1_000_000.9];
    const got = sampleTrilinear(words, deepOutside);
    expect(got).toBe(background);

    // Sanity: readValue at the same integer taps is also background (inactive).
    const base = readValue(words, [1_000_000, 1_000_000, 1_000_000]);
    expect(base.active).toBe(false);
    expect(base.value).toBe(background);
  });
});
