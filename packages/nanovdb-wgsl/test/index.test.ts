import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Codec,
  FILE_HEADER_SIZE,
  FILE_METADATA_SIZE,
  GRID_DATA_SIZE,
  MAGIC_ASCII,
  NanoVDBFile,
  SUPPORTED_GRID_TYPES,
} from "../src/index.js";

describe("nanovdb-wgsl constants", () => {
  it("exposes the NanoVDB layout constants", () => {
    expect(FILE_HEADER_SIZE).toBe(16);
    expect(FILE_METADATA_SIZE).toBe(176);
    expect(GRID_DATA_SIZE).toBe(672);
    expect(SUPPORTED_GRID_TYPES).toEqual(["Float", "Fp8", "FpN"]);
    expect(MAGIC_ASCII).toEqual(["NanoVDB0", "NanoVDB1", "NanoVDB2"]);
    expect(Codec).toEqual({ NONE: 0, ZIP: 1, BLOSC: 2 });
  });
});

describe("NanoVDBFile (Phase 1)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fromArrayBuffer rejects an empty buffer with an actionable error, not a stub error", () => {
    expect(() => NanoVDBFile.fromArrayBuffer(new ArrayBuffer(0))).toThrowError(
      /truncated buffer/,
    );
    expect(() => NanoVDBFile.fromArrayBuffer(new ArrayBuffer(0))).not.toThrowError(/Phase 1/);
  });

  it("fromURL rejects a non-OK response with an actionable error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(null, { status: 404, statusText: "Not Found" })),
    );
    await expect(NanoVDBFile.fromURL("https://example.test/missing.nvdb")).rejects.toThrowError(
      /failed to fetch/,
    );
  });

  it("fromURL parses the fetched buffer via fromArrayBuffer", async () => {
    // Build a minimal-but-valid single-grid segment buffer (see
    // nvdb-file.test.ts for the full byte-layout derivation) so we can
    // confirm fromURL actually delegates to the real parser instead of a
    // "not implemented" stub.
    const buffer = buildMinimalSegmentBuffer();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(buffer)),
    );
    const file = await NanoVDBFile.fromURL("https://example.test/grid.nvdb");
    expect(file.grids).toHaveLength(1);
    expect(file.grids[0]?.gridType).toBe("Float");
  });
});

/**
 * Builds the smallest possible valid segment-format `.nvdb` buffer: one
 * Float/FogVolume grid whose "grid data" is just a GRID_DATA_SIZE-byte
 * GridData header stub (magic + version + grid type/class only — no tree).
 * Good enough to exercise FileHeader/FileMetaData parsing without pulling in
 * a real fixture file.
 */
function buildMinimalSegmentBuffer(): ArrayBuffer {
  const name = "g\0";
  const nameSize = name.length;
  const gridSize = GRID_DATA_SIZE;
  const total = FILE_HEADER_SIZE + FILE_METADATA_SIZE + nameSize + gridSize;
  const buffer = new ArrayBuffer(total);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // FileHeader
  bytes.set([..."NanoVDB2"].map((c) => c.charCodeAt(0)), 0);
  view.setUint32(8, (32 << 21) | (9 << 10) | 1, true); // version 32.9.1
  view.setUint16(12, 1, true); // gridCount
  view.setUint16(14, 0, true); // codec NONE

  // FileMetaData
  const meta = FILE_HEADER_SIZE;
  view.setBigUint64(meta + 0, BigInt(gridSize), true); // gridSize
  view.setBigUint64(meta + 8, BigInt(gridSize), true); // fileSize
  view.setBigUint64(meta + 16, 0n, true); // nameKey
  view.setBigUint64(meta + 24, 42n, true); // voxelCount
  view.setUint32(meta + 32, 1, true); // gridType = Float
  view.setUint32(meta + 36, 2, true); // gridClass = FogVolume
  // worldBBox/indexBBox/voxelSize left at zero
  view.setUint32(meta + 136, nameSize, true); // nameSize
  view.setUint16(meta + 168, 0, true); // codec
  view.setUint16(meta + 170, 0, true); // blindDataCount
  view.setUint32(meta + 172, (32 << 21) | (9 << 10) | 1, true); // version

  // grid name
  const nameOffset = meta + FILE_METADATA_SIZE;
  bytes.set([...name].map((c) => c.charCodeAt(0)), nameOffset);

  // grid data (GridData header only)
  const gridOffset = nameOffset + nameSize;
  bytes.set([..."NanoVDB1"].map((c) => c.charCodeAt(0)), gridOffset);
  view.setUint32(gridOffset + 16, (32 << 21) | (9 << 10) | 1, true); // version
  view.setUint32(gridOffset + 24, 0, true); // gridIndex
  view.setUint32(gridOffset + 28, 1, true); // gridCount
  view.setBigUint64(gridOffset + 32, BigInt(gridSize), true); // gridSize
  view.setUint32(gridOffset + 632, 2, true); // gridClass = FogVolume
  view.setUint32(gridOffset + 636, 1, true); // gridType = Float

  return buffer;
}
