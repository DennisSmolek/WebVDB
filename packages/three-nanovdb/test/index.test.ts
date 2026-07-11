import { describe, expect, it } from "vitest";
import { NanoVDBGrid, createVolumeRenderer } from "../src/index.js";

describe("three-nanovdb stubs (Phase 0 gate)", () => {
  it("NanoVDBGrid throws until its Phase 3 wiring lands", () => {
    expect(() => new NanoVDBGrid(new Uint32Array(0))).toThrowError(/Phase 1/);
  });

  it("createVolumeRenderer throws until Phase 3 lands", async () => {
    await expect(createVolumeRenderer()).rejects.toThrowError(/Phase 3/);
  });
});
