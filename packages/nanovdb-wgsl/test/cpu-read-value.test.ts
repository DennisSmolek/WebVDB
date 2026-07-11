import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { readValue } from "../src/cpu/read-value.js";
import { defineNumber, gridTypeConstantsFor } from "../src/cpu/stride-tables.js";

/**
 * Ground-truth replay for the pure-TS CPU NanoVDB reference (`src/cpu/`)
 * against the baked fixture sidecars. This anchors the layout math the
 * Phase 2 WGSL traversal suite will also be checked against.
 *
 * `.nvdb` files are plain, single-grid, Codec NONE containers:
 *   FileHeader (16 B): 8-byte magic "NanoVDB2", u32 version, u16 gridCount,
 *     u16 codec
 *   FileMetaData (176 B): u64 gridSize at +0, u32 nameSize at +136
 *   gridName (nameSize bytes)
 *   raw grid image (gridSize bytes) — a self-contained NanoVDB GridData block
 *
 * `extractGridImage` below is test-local scaffolding standing in for the
 * concurrently-developed `NanoVDBFile` loader (src/nvdb-file.ts, Phase 1);
 * replace this with `NanoVDBFile` once that lands.
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
  grid: { type: string };
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

  // Copy into a fresh, word-aligned buffer rather than fussing with the
  // source ArrayBuffer's alignment.
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
  const words = extractGridImage(nvdbBuffer.buffer.slice(nvdbBuffer.byteOffset, nvdbBuffer.byteOffset + nvdbBuffer.byteLength));
  return { words, sidecar };
}

const PRIMITIVES = ["sphere_fog", "torus_fog", "box_fog"];
const VARIANTS = ["float", "fp8", "fpn"];
const FIXTURE_NAMES = PRIMITIVES.flatMap((p) => VARIANTS.map((v) => `${p}_${v}`));

const GRID_TYPE_ID: Record<string, number> = { float: 1, fp8: 14, fpn: 16 };

describe.skipIf(!fixturesPresent)("cpu readValue vs. baked ground-truth sidecars", () => {
  it("fixtures directory is present", async () => {
    const names = await readdir(fixturesDir);
    for (const f of FIXTURE_NAMES) {
      expect(names).toContain(`${f}.nvdb`);
      expect(names).toContain(`${f}.sidecar.json`);
    }
  });

  for (const name of FIXTURE_NAMES) {
    const variant = name.split("_").pop()!;

    it(`${name}: grid type + magic are structurally sound`, async () => {
      const { words, sidecar } = await loadFixture(name);
      expect(GRID_TYPE_ID[variant]).toBeDefined();

      // Word 636/4 = 159 is PNANOVDB_GRID_OFF_GRID_TYPE.
      const gridTypeId = words[159];
      expect([1, 14, 16]).toContain(gridTypeId);
      expect(gridTypeId).toBe(GRID_TYPE_ID[variant]);
      expect(sidecar.grid.type).toBe(variant);

      // Word 0/1 = PNANOVDB_GRID_OFF_MAGIC (u64, low word first).
      const magic = (BigInt(words[1]!) << 32n) | BigInt(words[0]!);
      expect(magic).toBe(0x314244566f6e614en); // PNANOVDB_MAGIC_GRID
    });

    it(`${name}: replays all 73 sidecar samples`, async () => {
      const { words, sidecar } = await loadFixture(name);
      expect(sidecar.samples.length).toBeGreaterThanOrEqual(73);

      let passCount = 0;
      for (const sample of sidecar.samples) {
        const result = readValue(words, sample.ijk);
        expect(result.active, `${name} ${JSON.stringify(sample.ijk)} active`).toBe(sample.active);
        expect(
          Math.abs(result.value - sample.value),
          `${name} ${JSON.stringify(sample.ijk)} value (got ${result.value}, want ${sample.value})`,
        ).toBeLessThanOrEqual(1e-5);
        passCount++;
      }
      expect(passCount).toBe(sidecar.samples.length);
    });
  }

  it("out-of-bbox coord returns inactive background for the float grid", async () => {
    const { words } = await loadFixture("sphere_fog_float");

    const farAway: [number, number, number] = [1_000_000, 1_000_000, 1_000_000];
    const result = readValue(words, farAway);
    expect(result.active).toBe(false);

    // Independently read root_off_background to avoid a tautological check
    // against readValue's own descent.
    const gt = gridTypeConstantsFor("FLOAT");
    const treeAddress = defineNumber("PNANOVDB_GRID_SIZE");
    const rootOffsetLo = words[(treeAddress + defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT")) / 4]!;
    const rootOffsetHi =
      words[(treeAddress + defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT")) / 4 + 1]!;
    const rootOffset = Number((BigInt(rootOffsetHi) << 32n) | BigInt(rootOffsetLo));
    const rootAddress = treeAddress + rootOffset;
    const backgroundWordIndex = (rootAddress + gt.root_off_background) / 4;
    const f32 = new Float32Array(words.buffer, words.byteOffset, words.length);
    const background = f32[backgroundWordIndex]!;

    expect(result.value).toBe(background);
  });

  it("throws a clear error for unsupported grid types", async () => {
    const { words } = await loadFixture("sphere_fog_float");
    const mutated = words.slice();
    // Word 159 = PNANOVDB_GRID_OFF_GRID_TYPE / 4. Flip FLOAT (1) to DOUBLE (2).
    mutated[159] = 2;
    expect(() => readValue(mutated, [0, 0, 0])).toThrowError(/unsupported grid type/i);
  });
});
