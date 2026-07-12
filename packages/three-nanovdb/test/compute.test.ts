import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { NanoVDBFile } from "nanovdb-wgsl";
import { NanoVDBGrid } from "../src/grid.js";
import { GRID_TYPE_FLOAT, GRID_TYPE_FP8 } from "../src/wgsl.js";
import {
  bboxSize,
  buildComputeShaderSource,
  buildValueTransformShaderSource,
  computeAtlasDims,
  decodeToAtlas,
  gridStats,
  resolveTransformBody,
  valueTransform,
  VALUE_TRANSFORM_PRESETS,
} from "../src/compute.js";

/**
 * Node-safe `compute.ts` unit tests (Phase 4, docs/PLAN.md / SPEC §3.3). No
 * GPU in Vitest, so this exercises: (a) the pure WGSL-assembly helpers
 * (`buildComputeShaderSource`/`buildValueTransformShaderSource`) against the
 * real vendored source, (b) the pure index-bbox/atlas-dims math, and
 * (c) option/gating validation, which every one of `gridStats`/
 * `valueTransform`/`decodeToAtlas` performs BEFORE touching the (here,
 * absent) `GPUDevice` — so a bogus/undefined device can stand in to prove the
 * validation really does run first. The end-to-end GPU behavior (atomics
 * correctness, CPU cross-checks) is demo 04's job (`e2e/demo-04.spec.ts`).
 */

const wgslUrl = new URL("../../nanovdb-wgsl/vendor/pnanovdb.wgsl", import.meta.url);
const fixturesDir = new URL("../../../fixtures/primitives/", import.meta.url);
const wgslPresent = existsSync(wgslUrl);
const fixturesPresent = existsSync(fixturesDir);

async function loadSource(): Promise<string> {
  return readFile(wgslUrl, "utf8");
}

async function loadGrid(name: string): Promise<NanoVDBGrid> {
  const buf = await readFile(new URL(`${name}.nvdb`, fixturesDir));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return NanoVDBGrid.fromFile(NanoVDBFile.fromArrayBuffer(ab), 0);
}

/** A `GPUDevice`-shaped value that should never actually be touched by these tests. */
const UNUSED_DEVICE = {} as unknown as Parameters<typeof gridStats>[0];

describe("bboxSize / computeAtlasDims", () => {
  it("computes an inclusive-max per-axis voxel count", () => {
    expect(bboxSize({ min: [-40, -20, -30], max: [40, 20, 30] })).toEqual([81, 41, 61]);
    expect(bboxSize({ min: [0, 0, 0], max: [0, 0, 0] })).toEqual([1, 1, 1]);
  });

  it("clamps each axis independently to maxDim (a literal clamp, not a proportional downscale)", () => {
    expect(computeAtlasDims({ min: [-40, -20, -30], max: [40, 20, 30] }, 256)).toEqual([81, 41, 61]);
    expect(computeAtlasDims({ min: [-500, -20, -30], max: [500, 20, 30] }, 256)).toEqual([256, 41, 61]);
    expect(computeAtlasDims({ min: [0, 0, 0], max: [1000, 1000, 1000] }, 128)).toEqual([128, 128, 128]);
  });
});

describe("resolveTransformBody / VALUE_TRANSFORM_PRESETS", () => {
  it("resolves known preset names to their WGSL bodies", () => {
    expect(resolveTransformBody("double")).toBe(VALUE_TRANSFORM_PRESETS.double);
    expect(resolveTransformBody("double")).toBe("return v * 2.0;");
    expect(resolveTransformBody("negate")).toBe("return -v;");
    expect(resolveTransformBody("clamp01")).toContain("clamp(v");
  });

  it("passes an unrecognized string through verbatim (a raw WGSL body)", () => {
    expect(resolveTransformBody("return v * 3.5 + 1.0;")).toBe("return v * 3.5 + 1.0;");
  });
});

describe.skipIf(!wgslPresent)("buildComputeShaderSource", () => {
  it("appends the four static entry points, unmodified vendored library (no rewrite)", async () => {
    const src = await loadSource();
    const out = buildComputeShaderSource(src);
    expect(out).toContain("fn nvdbx_root_minmax(");
    expect(out).toContain("fn nvdbx_stats_fast(");
    expect(out).toContain("fn nvdbx_stats_dump(");
    expect(out).toContain("fn nvdbx_atlas_decode(");
    // Raw-WebGPU path: the library's own global name is declared directly,
    // NOT rewritten (unlike wgsl.ts's TSL integration).
    expect(out).toContain("var<storage, read_write> nanovdb_buffer");
    expect(out).not.toContain(".value[(buf.byte_offset");
  });

  it("uses the documented atomics (atomicMin/atomicMax bitcast trick + histogram + 64-bit carry sum)", async () => {
    const src = await loadSource();
    const out = buildComputeShaderSource(src);
    expect(out).toContain("atomicMin(&nvdbx_stats_i32_out[0], bits)");
    expect(out).toContain("atomicMax(&nvdbx_stats_i32_out[1], bits)");
    expect(out).toContain("atomicAdd(&nvdbx_stats_u32_out[3u + bin], 1u)");
    expect(out).toContain("if (old_lo > (0xFFFFFFFFu - scaled))");
  });

  it("throws if pnanovdbSource has no nanovdb_buffer read site (shared assertHasBufferGlobal)", () => {
    expect(() => buildComputeShaderSource("")).toThrow(/nanovdb_buffer/);
    expect(() => buildComputeShaderSource("fn unrelated() -> u32 { return 0u; }")).toThrow(/nanovdb_buffer/);
  });
});

describe.skipIf(!wgslPresent)("buildValueTransformShaderSource", () => {
  it("inlines the caller's WGSL body into a dedicated nvdbx_transform function", async () => {
    const src = await loadSource();
    const out = buildValueTransformShaderSource(src, "return v * 2.0;");
    expect(out).toContain("fn nvdbx_transform(v : f32) -> f32 {");
    expect(out).toContain("return v * 2.0;");
    expect(out).toContain("fn nvdbx_value_transform(");
    // Only individually-addressable leaf voxels (level 0) are mutated.
    expect(out).toContain("if (resolved.level != 0u)");
    expect(out).toContain("nanovdb_buffer[resolved.address >> 2u] = bitcast<u32>(new_value);");
  });

  it("throws for an empty transform body", async () => {
    const src = await loadSource();
    expect(() => buildValueTransformShaderSource(src, "")).toThrow(/transformBody/);
    expect(() => buildValueTransformShaderSource(src, "   ")).toThrow(/transformBody/);
  });

  it("throws if pnanovdbSource has no nanovdb_buffer read site", () => {
    expect(() => buildValueTransformShaderSource("", "return v;")).toThrow(/nanovdb_buffer/);
  });
});

describe.skipIf(!wgslPresent || !fixturesPresent)("gridStats / decodeToAtlas option validation (no GPU touched)", () => {
  it("gridStats rejects a non-positive-integer histogramBins before touching the device", async () => {
    const src = await loadSource();
    const grid = await loadGrid("box_fog_float");
    await expect(gridStats(UNUSED_DEVICE, grid, src, { histogramBins: 0 })).rejects.toThrow(/histogramBins/);
    await expect(gridStats(UNUSED_DEVICE, grid, src, { histogramBins: 1.5 })).rejects.toThrow(/histogramBins/);
    await expect(gridStats(UNUSED_DEVICE, grid, src, { histogramBins: -4 })).rejects.toThrow(/histogramBins/);
  });

  it("gridStats rejects a bad pnanovdbSource before touching the device", async () => {
    const grid = await loadGrid("box_fog_float");
    await expect(gridStats(UNUSED_DEVICE, grid, "", {})).rejects.toThrow(/nanovdb_buffer/);
  });

  it("decodeToAtlas rejects invalid maxDim/filter/format before touching the device", async () => {
    const src = await loadSource();
    const grid = await loadGrid("sphere_fog_fp8");
    await expect(decodeToAtlas(UNUSED_DEVICE, grid, src, { maxDim: 0 })).rejects.toThrow(/maxDim/);
    await expect(decodeToAtlas(UNUSED_DEVICE, grid, src, { maxDim: -1 })).rejects.toThrow(/maxDim/);
    // @ts-expect-error deliberately invalid filter value
    await expect(decodeToAtlas(UNUSED_DEVICE, grid, src, { filter: "bicubic" })).rejects.toThrow(/filter/);
    // @ts-expect-error deliberately invalid format value
    await expect(decodeToAtlas(UNUSED_DEVICE, grid, src, { format: "rgba" })).rejects.toThrow(/format/);
  });

  it("valueTransform gates non-FLOAT grids before touching the device", async () => {
    const src = await loadSource();
    const fp8Grid = await loadGrid("sphere_fog_fp8");
    expect(fp8Grid.gridTypeId).toBe(GRID_TYPE_FP8);
    await expect(valueTransform(UNUSED_DEVICE, fp8Grid, src, "double")).rejects.toThrow(/FLOAT/);
  });

  it("valueTransform accepts FLOAT grids past the gate (fails later only for lacking a real device)", async () => {
    const src = await loadSource();
    const floatGrid = await loadGrid("box_fog_float");
    expect(floatGrid.gridTypeId).toBe(GRID_TYPE_FLOAT);
    // Past the FLOAT-only gate, it proceeds to touch the (fake, empty)
    // device and fails for THAT reason instead — proving the grid-type gate
    // itself isn't what rejected it.
    let message = "";
    try {
      await valueTransform(UNUSED_DEVICE, floatGrid, src, "double");
      throw new Error("expected valueTransform to reject given a fake device");
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).not.toMatch(/FLOAT/);
  });
});
