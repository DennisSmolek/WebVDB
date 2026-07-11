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

  // --- Finding 1: FileMetaData.gridSize must agree with the grid image's own
  // internal GridData.mGridSize (u64 at grid-image byte offset 32) ---------

  it("throws when FileMetaData.gridSize is smaller than the grid's internal mGridSize (truncation/corruption)", async () => {
    if (!fixturesPresent) return;
    const { buffer } = await loadFixture("box_fog_float");
    const patched = buffer.slice(0);
    const view = new DataView(patched);
    const trueGridSize = view.getBigUint64(FILE_HEADER_SIZE + 0, true);
    // Lie about gridSize: 1000 bytes smaller than what's actually there.
    // Before Finding 1's fix this silently sliced a short grid image instead
    // of throwing.
    view.setBigUint64(FILE_HEADER_SIZE + 0, trueGridSize - 1000n, true);

    expect(() => NanoVDBFile.fromArrayBuffer(patched)).toThrowError(
      /FileMetaData\.gridSize \(\d+\) does not match its own internal GridData\.mGridSize \(\d+\)/,
    );
  });

  it("throws when FileMetaData.gridSize is larger than the grid's internal mGridSize", async () => {
    if (!fixturesPresent) return;
    const { buffer } = await loadFixture("box_fog_float");
    const patched = buffer.slice(0);
    const view = new DataView(patched);
    const trueGridSize = view.getBigUint64(FILE_HEADER_SIZE + 0, true);
    view.setBigUint64(FILE_HEADER_SIZE + 0, trueGridSize + 1000n, true);

    // Growing gridSize by 1000 also grows how many bytes parseSegments tries
    // to slice out of the buffer, so this manifests as "truncated buffer"
    // (there aren't 1000 more real bytes to read) rather than the internal
    // mismatch error — both are refusals to silently load a wrong image, so
    // assert on the truncation message here.
    expect(() => NanoVDBFile.fromArrayBuffer(patched)).toThrowError(/truncated buffer/);
  });

  // --- Finding 2: bounded ZIP inflation --------------------------------

  it("ZIP path throws a bounded size error for a payload that inflates past the declared gridSize (zip-bomb shape)", () => {
    const declaredGridSize = 128;
    // Highly compressible (all-zero) 2MB payload: compresses down to a tiny
    // handful of bytes, but "decompresses" to far more than declaredGridSize
    // — the zip-bomb shape the finding describes. If the fix regressed to
    // unbounded unzlibSync(compressed), this would still throw eventually
    // (byteLength check) but only after fully materializing 2MB; the point
    // of this test is that it throws the *new* bounded-overflow message.
    const bigPayload = new Uint8Array(2_000_000);
    const buffer = buildZipSegmentBuffer(declaredGridSize, bigPayload);

    expect(() => NanoVDBFile.fromArrayBuffer(buffer)).toThrowError(
      /exceeds its declared gridSize of 128 bytes/,
    );
  });

  it("ZIP path still throws a clear error when the decompressed payload is smaller than declared gridSize", () => {
    const declaredGridSize = 1000;
    const smallPayload = new Uint8Array(10);
    const buffer = buildZipSegmentBuffer(declaredGridSize, smallPayload);

    expect(() => NanoVDBFile.fromArrayBuffer(buffer)).toThrowError(
      /ZIP-decompressed grid #0 \("g"\) is 10 bytes, expected 1000/,
    );
  });

  it("throws a clear error for a gridSize u64 that doesn't fit in a safe JS integer", () => {
    const name = "g\0";
    const nameSize = name.length;
    const total = FILE_HEADER_SIZE + FILE_METADATA_SIZE + nameSize;
    const buffer = new ArrayBuffer(total);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    bytes.set([..."NanoVDB2"].map((c) => c.charCodeAt(0)), 0);
    view.setUint32(8, (32 << 21) | (9 << 10) | 1, true); // version 32.9.1
    view.setUint16(12, 1, true); // gridCount
    view.setUint16(14, Codec.NONE, true); // codec

    const meta = FILE_HEADER_SIZE;
    view.setBigUint64(meta + 0, 2n ** 60n, true); // grossly unrepresentable gridSize
    view.setUint32(meta + 136, nameSize, true); // nameSize
    bytes.set([...name].map((c) => c.charCodeAt(0)), meta + FILE_METADATA_SIZE);

    expect(() => NanoVDBFile.fromArrayBuffer(buffer)).toThrowError(/unrepresentable gridSize/);
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

/**
 * Builds a minimal single-grid Codec-ZIP `.nvdb` segment buffer (FileHeader +
 * FileMetaData + name "g\0" + ZIP framing) whose FileMetaData.gridSize is
 * `declaredGridSize`, but whose ZIP payload actually decompresses to
 * `decompressedPayload` (which may be a different length — that mismatch is
 * exactly what Finding 2's tests exercise). The grid data itself is never
 * required to look like a real GridData header here: the ZIP size checks
 * throw before any of that content would be interpreted.
 */
function buildZipSegmentBuffer(declaredGridSize: number, decompressedPayload: Uint8Array): ArrayBuffer {
  const name = "g\0";
  const nameSize = name.length;
  const compressed = zlibSync(decompressedPayload);

  const headerAndMetaSize = FILE_HEADER_SIZE + FILE_METADATA_SIZE + nameSize;
  const total = headerAndMetaSize + 8 + compressed.byteLength;
  const buffer = new ArrayBuffer(total);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  bytes.set([..."NanoVDB2"].map((c) => c.charCodeAt(0)), 0);
  view.setUint32(8, (32 << 21) | (9 << 10) | 1, true); // version 32.9.1
  view.setUint16(12, 1, true); // gridCount
  view.setUint16(14, Codec.ZIP, true); // codec

  const meta = FILE_HEADER_SIZE;
  view.setBigUint64(meta + 0, BigInt(declaredGridSize), true); // gridSize
  view.setBigUint64(meta + 24, 0n, true); // voxelCount
  view.setUint32(meta + 32, 1, true); // gridType = Float
  view.setUint32(meta + 36, 2, true); // gridClass = FogVolume
  view.setUint32(meta + 136, nameSize, true); // nameSize
  view.setUint16(meta + 168, Codec.ZIP, true); // codec (FileMetaData mirror)

  bytes.set([...name].map((c) => c.charCodeAt(0)), meta + FILE_METADATA_SIZE);

  const zipOffset = meta + FILE_METADATA_SIZE + nameSize;
  view.setBigUint64(zipOffset, BigInt(compressed.byteLength), true);
  bytes.set(compressed, zipOffset + 8);

  return buffer;
}

/** Copies a typed array's view into a fresh, standalone zero-offset ArrayBuffer. */
function standaloneImageBuffer(image: Uint32Array): ArrayBuffer {
  return image.buffer.slice(image.byteOffset, image.byteOffset + image.byteLength);
}

function bufferOf(image: Uint32Array): Buffer {
  return Buffer.from(image.buffer, image.byteOffset, image.byteLength);
}

/**
 * Finding 3: `parseRawGridBuffer` (the magic-`NanoVDB1` raw-grid-image path,
 * used when a buffer has no FileHeader) shipped with zero tests. These build
 * raw-buffer inputs directly from the real fixtures — via `gridImage()` on a
 * segment-parsed file, which is itself already the bare GridData image
 * (confirmed: every fixture's grid image begins with the "NanoVDB1" magic,
 * NanoVDB.h's `NANOVDB_MAGIC_GRID`) — so no magic rewriting is needed; the
 * extracted bytes are fed to `fromArrayBuffer` as-is.
 */
describe.skipIf(!fixturesPresent)("NanoVDBFile — raw grid buffer (magic NanoVDB1, no FileHeader)", () => {
  it("sniffs a bare grid image (no FileHeader) as the raw-buffer format via its own NanoVDB1 magic", async () => {
    const { buffer } = await loadFixture("box_fog_float");
    const segmentFile = NanoVDBFile.fromArrayBuffer(buffer);
    const image = segmentFile.gridImage(0);
    const magicBytes = new Uint8Array(image.buffer, image.byteOffset, 8);
    expect(String.fromCharCode(...magicBytes)).toBe("NanoVDB1");
  });

  for (const name of ["box_fog_float", "sphere_fog_fp8", "torus_fog_fpn"]) {
    it(`${name}: raw-buffer parse of the extracted grid image matches the segment-parsed metadata and gridImage bytes`, async () => {
      const { buffer } = await loadFixture(name);
      const segmentFile = NanoVDBFile.fromArrayBuffer(buffer);
      const rawBuffer = standaloneImageBuffer(segmentFile.gridImage(0));

      const rawFile = NanoVDBFile.fromArrayBuffer(rawBuffer);
      expect(rawFile.grids).toHaveLength(1);

      const segGrid = segmentFile.grids[0]!;
      const rawGrid = rawFile.grids[0]!;

      expect(rawGrid.name).toBe(segGrid.name);
      expect(rawGrid.gridType).toBe(segGrid.gridType);
      expect(rawGrid.gridClass).toBe(segGrid.gridClass);
      expect(rawGrid.voxelCount).toBe(segGrid.voxelCount);
      expect(rawGrid.gridByteSize).toBe(segGrid.gridByteSize);
      expect(rawGrid.indexBBox).toEqual(segGrid.indexBBox);
      expect(rawGrid.voxelSize).toEqual(segGrid.voxelSize);
      for (let a = 0; a < 3; a++) {
        expect(rawGrid.worldBBox.min[a]).toBeCloseTo(segGrid.worldBBox.min[a]!, 6);
        expect(rawGrid.worldBBox.max[a]).toBeCloseTo(segGrid.worldBBox.max[a]!, 6);
      }

      // gridImage() round-trips byte-for-byte through the raw path.
      expect(bufferOf(rawFile.gridImage(0)).equals(bufferOf(segmentFile.gridImage(0)))).toBe(true);
    });
  }

  it("parses a concatenated two-grid raw buffer by walking mGridSize, once mGridCount/mGridIndex agree across grids", async () => {
    const { buffer: bufferA } = await loadFixture("box_fog_float");
    const { buffer: bufferB } = await loadFixture("box_fog_fp8");
    const fileA = NanoVDBFile.fromArrayBuffer(bufferA);
    const fileB = NanoVDBFile.fromArrayBuffer(bufferB);

    // Each fixture is independently produced as a single-grid buffer, so its
    // image's own mGridCount is 1 — concatenating two of those as-is would
    // make parseRawGridBuffer stop after the first (it trusts grid #0's
    // mGridCount as the total). A genuine N-grid NanoVDB1 buffer has every
    // grid's mGridCount set to N and mGridIndex set to its position, so we
    // patch those two fields (GridData offsets 28 and 24) to simulate one.
    const patchedA = new Uint8Array(standaloneImageBuffer(fileA.gridImage(0)));
    const patchedB = new Uint8Array(standaloneImageBuffer(fileB.gridImage(0)));
    new DataView(patchedA.buffer).setUint32(24, 0, true); // mGridIndex = 0
    new DataView(patchedA.buffer).setUint32(28, 2, true); // mGridCount = 2
    new DataView(patchedB.buffer).setUint32(24, 1, true); // mGridIndex = 1
    new DataView(patchedB.buffer).setUint32(28, 2, true); // mGridCount = 2

    const combined = new Uint8Array(patchedA.byteLength + patchedB.byteLength);
    combined.set(patchedA, 0);
    combined.set(patchedB, patchedA.byteLength);

    const rawFile = NanoVDBFile.fromArrayBuffer(combined.buffer);
    expect(rawFile.grids).toHaveLength(2);

    expect(rawFile.grids[0]!.gridType).toBe(fileA.grids[0]!.gridType);
    expect(rawFile.grids[0]!.gridByteSize).toBe(patchedA.byteLength);
    expect(rawFile.grids[1]!.gridType).toBe(fileB.grids[0]!.gridType);
    expect(rawFile.grids[1]!.gridByteSize).toBe(patchedB.byteLength);

    expect(bufferOf(rawFile.gridImage(0)).equals(Buffer.from(patchedA.buffer))).toBe(true);
    expect(bufferOf(rawFile.gridImage(1)).equals(Buffer.from(patchedB.buffer))).toBe(true);
  });
});
