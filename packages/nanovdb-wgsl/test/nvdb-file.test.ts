import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { zlibSync } from "fflate";
import { describe, expect, it } from "vitest";
import { Codec, FILE_HEADER_SIZE, FILE_METADATA_SIZE, NanoVDBFile } from "../src/index.js";

/**
 * Parses all 9 baked primitive fixtures (fixtures/primitives/*.nvdb) with
 * NanoVDBFile and checks the result against their committed sidecar ground
 * truth. `.nvdb` files themselves are git-ignored (regenerate with
 * `pnpm fixtures:bake`), so this whole suite is skipped when the directory
 * is absent — it is present on this machine.
 */

const fixturesDir = new URL("../../../fixtures/primitives/", import.meta.url);
const fixturesPresent = existsSync(fixturesDir);

const PRIMITIVES = ["sphere_fog", "torus_fog", "box_fog"];
const VARIANTS = ["float", "fp8", "fpn"];
const FIXTURE_NAMES = PRIMITIVES.flatMap((p) => VARIANTS.map((v) => `${p}_${v}`));

// sidecar "type" is lowercase (float/fp8/fpn); GridMetadata.gridType uses the
// NanoVDB.h enum identifier casing (matches SUPPORTED_GRID_TYPES).
const EXPECTED_GRID_TYPE: Record<string, string> = { float: "Float", fp8: "Fp8", fpn: "FpN" };

interface Sidecar {
  grid: {
    name: string;
    type: string;
    class: string;
    gridByteSize: number;
    activeVoxelCount: number;
    indexBBox: [[number, number, number], [number, number, number]];
    worldBBox: [[number, number, number], [number, number, number]];
    voxelSize: [number, number, number];
  };
}

async function readArrayBuffer(url: URL): Promise<ArrayBuffer> {
  const buf = await readFile(url);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

async function loadFixture(name: string): Promise<{ buffer: ArrayBuffer; sidecar: Sidecar }> {
  const buffer = await readArrayBuffer(new URL(`${name}.nvdb`, fixturesDir));
  const sidecar = JSON.parse(
    await readFile(new URL(`${name}.sidecar.json`, fixturesDir), "utf8"),
  ) as Sidecar;
  return { buffer, sidecar };
}

describe.skipIf(!fixturesPresent)("NanoVDBFile — primitive fixtures", () => {
  it("fixtures directory is present with all 9 primitive x variant pairs", async () => {
    const names = await readdir(fixturesDir);
    for (const f of FIXTURE_NAMES) {
      expect(names).toContain(`${f}.nvdb`);
      expect(names).toContain(`${f}.sidecar.json`);
    }
  });

  for (const name of FIXTURE_NAMES) {
    const variant = name.split("_").pop()!;

    it(`${name}: metadata matches the sidecar ground truth`, async () => {
      const { buffer, sidecar } = await loadFixture(name);
      const file = NanoVDBFile.fromArrayBuffer(buffer);

      expect(file.grids).toHaveLength(1);
      const grid = file.grids[0]!;

      expect(grid.name).toBe(sidecar.grid.name);
      expect(grid.gridType).toBe(EXPECTED_GRID_TYPE[variant]);
      expect(grid.gridClass).toBe(sidecar.grid.class);
      expect(grid.voxelCount).toBe(sidecar.grid.activeVoxelCount);
      expect(grid.gridByteSize).toBe(sidecar.grid.gridByteSize);
      expect(grid.voxelSize).toEqual(sidecar.grid.voxelSize);

      expect([grid.indexBBox.min[0], grid.indexBBox.min[1], grid.indexBBox.min[2]]).toEqual(
        sidecar.grid.indexBBox[0],
      );
      expect([grid.indexBBox.max[0], grid.indexBBox.max[1], grid.indexBBox.max[2]]).toEqual(
        sidecar.grid.indexBBox[1],
      );

      for (let a = 0; a < 3; a++) {
        expect(grid.worldBBox.min[a]).toBeCloseTo(sidecar.grid.worldBBox[0][a]!, 4);
        expect(grid.worldBBox.max[a]).toBeCloseTo(sidecar.grid.worldBBox[1][a]!, 4);
      }
    });

    it(`${name}: gridImage byte length matches gridByteSize and starts with a valid magic`, async () => {
      const { buffer, sidecar } = await loadFixture(name);
      const file = NanoVDBFile.fromArrayBuffer(buffer);
      const image = file.gridImage(0);

      expect(image.byteLength).toBe(sidecar.grid.gridByteSize);
      expect(image).toBeInstanceOf(Uint32Array);

      const bytes = new Uint8Array(image.buffer, image.byteOffset, 8);
      const magic = String.fromCharCode(...bytes);
      expect(["NanoVDB0", "NanoVDB1", "NanoVDB2"]).toContain(magic);

      // u32 at byte offset 16 is GridData::mVersion (PNANOVDB_GRID_OFF_VERSION).
      const version = image[16 / 4]!;
      const major = (version >>> 21) & 0x7ff;
      expect(major).toBe(32);
    });
  }

  it("box_fog fixtures land on a 4-byte-aligned grid segment (zero-copy)", async () => {
    // box_fog has an 8-byte name ("box_fog\0"), so the grid segment starts
    // at byte 200 (16 + 176 + 8) — 4-byte aligned.
    for (const variant of VARIANTS) {
      const { buffer } = await loadFixture(`box_fog_${variant}`);
      const file = NanoVDBFile.fromArrayBuffer(buffer);
      expect(file.isGridImageZeroCopy(0), `box_fog_${variant}`).toBe(true);
      expect(file.gridImage(0).buffer, `box_fog_${variant}`).toBe(buffer);
    }
  });

  it("sphere_fog/torus_fog fixtures land on an unaligned grid segment (copy)", async () => {
    // sphere_fog's name is 11 bytes, torus_fog's is 10 — neither leaves the
    // grid segment 4-byte aligned, so gridImage() must fall back to a copy.
    for (const primitive of ["sphere_fog", "torus_fog"]) {
      for (const variant of VARIANTS) {
        const { buffer } = await loadFixture(`${primitive}_${variant}`);
        const file = NanoVDBFile.fromArrayBuffer(buffer);
        expect(file.isGridImageZeroCopy(0), `${primitive}_${variant}`).toBe(false);
        expect(file.gridImage(0).buffer, `${primitive}_${variant}`).not.toBe(buffer);
      }
    }
  });

  it("ZIP round-trip: recompressing a NONE fixture's grid segment yields a byte-identical image", async () => {
    const { buffer: noneBuffer } = await loadFixture("box_fog_float");
    const noneFile = NanoVDBFile.fromArrayBuffer(noneBuffer);
    const expectedImage = noneFile.gridImage(0);

    const zipBuffer = rewriteAsZip(noneBuffer);
    const zipFile = NanoVDBFile.fromArrayBuffer(zipBuffer);

    expect(zipFile.grids).toHaveLength(1);
    expect(zipFile.grids[0]).toEqual(noneFile.grids[0]);
    expect(zipFile.isGridImageZeroCopy(0)).toBe(false); // ZIP always decompresses into a copy

    const zipImage = zipFile.gridImage(0);
    expect(zipImage.byteLength).toBe(expectedImage.byteLength);
    // Byte-for-byte comparison via Buffer.equals — deep-equality matchers
    // (toEqual) are far too slow over multi-megabyte typed arrays.
    const zipBytes = Buffer.from(zipImage.buffer, zipImage.byteOffset, zipImage.byteLength);
    const expectedBytes = Buffer.from(
      expectedImage.buffer,
      expectedImage.byteOffset,
      expectedImage.byteLength,
    );
    expect(zipBytes.equals(expectedBytes)).toBe(true);
  });
});

describe("NanoVDBFile — error handling", () => {
  it("throws for garbage magic bytes", () => {
    const buffer = new ArrayBuffer(32);
    new Uint8Array(buffer).set([..."NotANvdb"].map((c) => c.charCodeAt(0)), 0);
    expect(() => NanoVDBFile.fromArrayBuffer(buffer)).toThrowError(/unrecognized magic/);
  });

  it("throws a helpful error for the BLOSC codec value", () => {
    const buffer = new ArrayBuffer(FILE_HEADER_SIZE);
    const view = new DataView(buffer);
    new Uint8Array(buffer).set([..."NanoVDB2"].map((c) => c.charCodeAt(0)), 0);
    view.setUint32(8, (32 << 21) | (9 << 10) | 1, true); // version 32.9.1
    view.setUint16(12, 1, true); // gridCount
    view.setUint16(14, Codec.BLOSC, true); // codec = BLOSC
    expect(() => NanoVDBFile.fromArrayBuffer(buffer)).toThrowError(/BLOSC/);
    expect(() => NanoVDBFile.fromArrayBuffer(buffer)).toThrowError(/codec NONE or ZIP/);
  });

  it("throws for a truncated buffer (short FileHeader)", () => {
    const buffer = new ArrayBuffer(10);
    new Uint8Array(buffer).set([..."NanoVDB2"].map((c) => c.charCodeAt(0)), 0);
    expect(() => NanoVDBFile.fromArrayBuffer(buffer)).toThrowError(/truncated buffer/);
  });

  it("throws for a buffer truncated mid grid-data", async () => {
    if (!fixturesPresent) return;
    const { buffer } = await loadFixture("box_fog_float");
    const truncated = buffer.slice(0, buffer.byteLength - 1000);
    expect(() => NanoVDBFile.fromArrayBuffer(truncated)).toThrowError(/truncated buffer/);
  });

  it("gridImage/isGridImageZeroCopy reject an out-of-range index", async () => {
    if (!fixturesPresent) return;
    const { buffer } = await loadFixture("box_fog_float");
    const file = NanoVDBFile.fromArrayBuffer(buffer);
    expect(() => file.gridImage(5)).toThrowError(/out of range/);
    expect(() => file.isGridImageZeroCopy(5)).toThrowError(/out of range/);
  });
});

/**
 * Rewrites a Codec-NONE single-grid segment buffer into an equivalent
 * Codec-ZIP buffer: the grid data segment is zlib-compressed (fflate's
 * `zlibSync`, matching C zlib's `compress()` used by `io::Internal::write` in
 * IO.h) and framed as an 8-byte LE compressed-size prefix + payload. The
 * FileHeader and FileMetaData codec fields are flipped to ZIP; gridSize is
 * left untouched (it is always the *decompressed* size).
 */
function rewriteAsZip(noneBuffer: ArrayBuffer): ArrayBuffer {
  const srcView = new DataView(noneBuffer);
  const gridCount = srcView.getUint16(12, true);
  if (gridCount !== 1) throw new Error("rewriteAsZip: only single-grid fixtures are supported");

  const nameSize = srcView.getUint32(FILE_HEADER_SIZE + 136, true);
  const gridSize = Number(srcView.getBigUint64(FILE_HEADER_SIZE + 0, true));
  const gridOffset = FILE_HEADER_SIZE + FILE_METADATA_SIZE + nameSize;

  const gridBytes = new Uint8Array(noneBuffer, gridOffset, gridSize);
  const compressed = zlibSync(gridBytes);

  const headerAndMeta = new Uint8Array(noneBuffer.slice(0, gridOffset));
  const out = new Uint8Array(headerAndMeta.byteLength + 8 + compressed.byteLength);
  out.set(headerAndMeta, 0);

  const outView = new DataView(out.buffer);
  outView.setUint16(14, Codec.ZIP, true); // FileHeader.codec
  outView.setUint16(FILE_HEADER_SIZE + 168, Codec.ZIP, true); // FileMetaData.codec
  outView.setBigUint64(headerAndMeta.byteLength, BigInt(compressed.byteLength), true);
  out.set(compressed, headerAndMeta.byteLength + 8);

  return out.buffer;
}
