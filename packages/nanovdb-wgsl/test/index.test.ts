import { describe, expect, it } from "vitest";
import {
  FILE_HEADER_SIZE,
  FILE_METADATA_SIZE,
  GRID_DATA_SIZE,
  NanoVDBFile,
  SUPPORTED_GRID_TYPES,
} from "../src/index.js";

describe("nanovdb-wgsl stubs (Phase 0 gate)", () => {
  it("exposes the NanoVDB layout constants", () => {
    expect(FILE_HEADER_SIZE).toBe(16);
    expect(FILE_METADATA_SIZE).toBe(176);
    expect(GRID_DATA_SIZE).toBe(672);
    expect(SUPPORTED_GRID_TYPES).toEqual(["Float", "Fp8", "FpN"]);
  });

  it("NanoVDBFile stubs throw until Phase 1 lands", async () => {
    expect(() => NanoVDBFile.fromArrayBuffer(new ArrayBuffer(0))).toThrowError(
      /Phase 1/,
    );
    await expect(NanoVDBFile.fromURL("x.nvdb")).rejects.toThrowError(/Phase 1/);
  });
});
