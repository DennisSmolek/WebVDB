import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { NodeMaterial, StorageBufferAttribute } from "three/webgpu";
import { NanoVDBFile } from "nanovdb-wgsl";
import type { GridMetadata } from "nanovdb-wgsl";
import { NanoVDBGrid } from "../src/grid.js";
import { NanoVDBVolumeMaterial } from "../src/material.js";
import {
  assembleVolumeWgsl,
  rewriteBufferGlobal,
  samplerForGridType,
  GRID_TYPE_FLOAT,
  GRID_TYPE_FP8,
  GRID_TYPE_FPN,
  DEFAULT_SAMPLE_BUDGET_CAP,
} from "../src/wgsl.js";

/**
 * Node-safe `NanoVDBVolumeMaterial` unit tests. No GPU in Vitest, so this
 * exercises the parts that don't need a device: the pure WGSL-assembly
 * (`assembleVolumeWgsl` and friends) and CPU-side material construction
 * (uniform surface, sampler selection). The end-to-end GPU render is the job
 * of `e2e/demo-02.spec.ts`. Fixtures are git-ignored, so the fixture-backed
 * blocks skip cleanly when absent (same pattern as `grid.test.ts`).
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

describe("samplerForGridType", () => {
  it("selects the per-type trilinear sampler", () => {
    expect(samplerForGridType(GRID_TYPE_FLOAT)).toBe("pnanovdb_sample_trilinear_float");
    expect(samplerForGridType(GRID_TYPE_FP8)).toBe("pnanovdb_sample_trilinear_fp8");
    expect(samplerForGridType(GRID_TYPE_FPN)).toBe("pnanovdb_sample_trilinear_fpn");
  });

  it("throws for an unsupported grid type id", () => {
    expect(() => samplerForGridType(7)).toThrow(/no trilinear sampler/i);
  });
});

describe("rewriteBufferGlobal", () => {
  const src = [
    "// NOTE: declare nanovdb_buffer somewhere",
    "fn pnanovdb_buf_read_uint32(buf: pnanovdb_buf_t, byte_offset: u32) -> u32 {",
    "    let data = nanovdb_buffer[(buf.byte_offset / 4u) + (byte_offset / 4u)];",
    "    return data;",
    "}",
  ].join("\n");

  it("rewrites the module-global to the TSL struct member access", () => {
    const out = rewriteBufferGlobal(src, "nvdbGrid");
    expect(out).toContain("nvdbGrid.value[(buf.byte_offset / 4u)");
    // No bare buffer-index access to the old identifier remains.
    expect(out).not.toMatch(/\bnanovdb_buffer\[/);
  });

  it("does not touch unrelated identifiers (whole-word only)", () => {
    const out = rewriteBufferGlobal("let nanovdb_buffer_size = 1u;", "nvdbGrid");
    expect(out).toContain("nanovdb_buffer_size");
  });

  // Rewrite-site count guard (verification finding 4): rewriteBufferGlobal
  // assumes the vendored source has exactly ONE live read site (plus the
  // header comment) for `nanovdb_buffer` — that's the whole B+C strategy in
  // wgsl.ts's module header. If a future vendor bump changes how many times
  // that identifier appears (e.g. a second buffer global, or the comment
  // rewritten away), the rewrite strategy needs re-review rather than
  // silently under- or over-rewriting. Pin the exact count against the REAL
  // vendored file so a vendor bump that changes it fails loudly here instead
  // of surfacing as a mystery WGSL bug later.
  it.skipIf(!wgslPresent)("the real vendored source has exactly 2 occurrences of nanovdb_buffer", async () => {
    const src = await loadSource();
    const occurrences = src.match(/\bnanovdb_buffer\b/g) ?? [];
    expect(occurrences.length).toBe(2);
  });
});

describe.skipIf(!wgslPresent)("assembleVolumeWgsl", () => {
  it("emits an entry function that calls the selected sampler", async () => {
    const src = await loadSource();
    const a = assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FP8 });
    expect(a.entryName).toBe("nvdb_volume_march");
    expect(a.samplerFn).toBe("pnanovdb_sample_trilinear_fp8");
    expect(a.entrySource).toContain(`fn ${a.entryName}(`);
    expect(a.entrySource).toContain("pnanovdb_sample_trilinear_fp8(buf, &acc,");
    expect(a.entrySource).toContain("pnanovdb_hdda_ray_clip(");
    expect(a.entrySource).toContain("nvdbx_hg_phase(");
    // Entry is the SOLE function (WGSLNodeFunction parses the first fn).
    expect(a.entrySource.match(/\bfn\s+\w+\s*\(/g)?.length).toBe(1);
  });

  it("rewrites the library buffer-global and appends helpers", async () => {
    const src = await loadSource();
    const a = assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, bufferName: "nvdbGrid" });
    expect(a.librarySource).toContain("nvdbGrid.value[");
    expect(a.librarySource).not.toMatch(/\bnanovdb_buffer\[/);
    expect(a.librarySource).toContain("fn nvdbx_hg_phase(");
    expect(a.librarySource).toContain("fn nvdbx_hash12(");
  });

  it("honors compile-time step caps", async () => {
    const src = await loadSource();
    const a = assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, maxStepsCap: 256, shadowStepsCap: 32 });
    expect(a.entrySource).toContain("clamp(max_steps, 1.0, 256.0)");
    expect(a.entrySource).toContain("clamp(shadow_steps, 0.0, 32.0)");
  });

  // Finding 1: unbounded maxSteps x shadowSteps product. max_steps/shadow_steps
  // are independently-clamped LIVE uniforms, so a user can drive ~65k trilinear
  // taps/fragment at the caps (TDR risk on real hardware). The generated WGSL
  // must enforce its own per-fragment total-sample budget: a compile-time
  // const plus a counter incremented on every trilinear tap (primary + shadow),
  // breaking the shadow loop first and then the main loop once exhausted.
  it("emits a per-fragment sample-budget const and counter that gates both loops", async () => {
    const src = await loadSource();
    const a = assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT });

    // The default budget is baked in as a compile-time literal.
    expect(a.entrySource).toContain(`nvdbx_sample_budget : i32 = ${DEFAULT_SAMPLE_BUDGET_CAP}`);
    // A counter variable is declared and incremented at least twice (once per
    // primary tap, once per shadow tap).
    expect(a.entrySource).toContain("var nvdbx_sample_count : i32 = 0");
    const increments = a.entrySource.match(/nvdbx_sample_count = nvdbx_sample_count \+ 1/g) ?? [];
    expect(increments.length).toBe(2);
    // Both the main march loop and the shadow loop check the budget.
    const budgetChecks = a.entrySource.match(/nvdbx_sample_count >= nvdbx_sample_budget/g) ?? [];
    expect(budgetChecks.length).toBeGreaterThanOrEqual(2);
  });

  it("honors a custom sampleBudgetCap", async () => {
    const src = await loadSource();
    const a = assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, sampleBudgetCap: 4096 });
    expect(a.entrySource).toContain("nvdbx_sample_budget : i32 = 4096");
    expect(a.entrySource).not.toContain(`nvdbx_sample_budget : i32 = ${DEFAULT_SAMPLE_BUDGET_CAP}`);
  });

  // Finding 3: input validation. An empty/wrong pnanovdbSource should fail
  // loudly at construction time, not as an opaque WGSL compile error later.
  it("throws if pnanovdbSource has no nanovdb_buffer read site to rewrite", () => {
    expect(() => assembleVolumeWgsl("", { gridTypeId: GRID_TYPE_FLOAT })).toThrow(/nanovdb_buffer/);
    expect(() =>
      assembleVolumeWgsl("fn totally_unrelated() -> u32 { return 0u; }", { gridTypeId: GRID_TYPE_FLOAT }),
    ).toThrow(/nanovdb_buffer/);
  });

  it("range-checks the numeric caps (integers >= 1)", async () => {
    const src = await loadSource();
    expect(() => assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, maxStepsCap: 0 })).toThrow(
      /maxStepsCap/,
    );
    expect(() => assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, maxStepsCap: -5 })).toThrow(
      /maxStepsCap/,
    );
    expect(() => assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, maxStepsCap: 1.5 })).toThrow(
      /maxStepsCap/,
    );
    expect(() => assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, shadowStepsCap: 0 })).toThrow(
      /shadowStepsCap/,
    );
    expect(() => assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, sampleBudgetCap: 0 })).toThrow(
      /sampleBudgetCap/,
    );
    expect(() => assembleVolumeWgsl(src, { gridTypeId: GRID_TYPE_FLOAT, sampleBudgetCap: -1 })).toThrow(
      /sampleBudgetCap/,
    );
  });
});

describe.skipIf(!wgslPresent || !fixturesPresent)("NanoVDBVolumeMaterial (fixture-backed)", () => {
  it("constructs a transparent, depth-write-off BackSide NodeMaterial", async () => {
    const src = await loadSource();
    const grid = await loadGrid("sphere_fog_fp8");
    const mat = new NanoVDBVolumeMaterial({ grid, pnanovdbSource: src });

    expect(mat).toBeInstanceOf(NodeMaterial);
    expect(mat.isNanoVDBVolumeMaterial).toBe(true);
    expect(mat.transparent).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.premultipliedAlpha).toBe(true);
    expect(mat.fragmentNode).toBeTruthy();
  });

  it("exposes the SPEC §3.2 uniform surface with defaults", async () => {
    const src = await loadSource();
    const grid = await loadGrid("sphere_fog_fp8");
    const mat = new NanoVDBVolumeMaterial({ grid, pnanovdbSource: src });

    expect(mat.densityScale.value).toBe(40);
    expect(mat.stepSize.value).toBe(0.75);
    expect(mat.maxSteps.value).toBe(512);
    expect(mat.anisotropy.value).toBe(0.3);
    expect(mat.shadowSteps.value).toBe(12);
    expect(mat.ambient.value).toBe(0.15);
    expect(mat.sunIntensity.value).toBe(3);
    expect(mat.jitter.value).toBe(1);
    // Overrides flow through.
    const mat2 = new NanoVDBVolumeMaterial({ grid, pnanovdbSource: src, densityScale: 10, jitter: false });
    expect(mat2.densityScale.value).toBe(10);
    expect(mat2.jitter.value).toBe(0);
  });

  it("selects the sampler from the grid's type id", async () => {
    const src = await loadSource();
    const fp8 = new NanoVDBVolumeMaterial({ grid: await loadGrid("sphere_fog_fp8"), pnanovdbSource: src });
    const fpn = new NanoVDBVolumeMaterial({ grid: await loadGrid("sphere_fog_fpn"), pnanovdbSource: src });
    const float = new NanoVDBVolumeMaterial({ grid: await loadGrid("sphere_fog_float"), pnanovdbSource: src });
    expect(fp8.samplerFn).toBe("pnanovdb_sample_trilinear_fp8");
    expect(fpn.samplerFn).toBe("pnanovdb_sample_trilinear_fpn");
    expect(float.samplerFn).toBe("pnanovdb_sample_trilinear_float");
  });

  it("requires pnanovdbSource", async () => {
    const grid = await loadGrid("sphere_fog_fp8");
    // @ts-expect-error deliberately omitting the required source
    expect(() => new NanoVDBVolumeMaterial({ grid })).toThrow(/pnanovdbSource/);
  });
});

/**
 * Grid-rebind API (the Phase 7 flag from docs/handoffs/PHASE-3.md). Pure CPU:
 * `rebindGrid` copies the new image into the bound storage attribute and bumps
 * its version — no device needed, so these run without fixtures (only the real
 * vendored WGSL, for material construction). Synthetic grids are hand-built so
 * we can vary size/type freely; content is irrelevant to the rebind mechanics.
 */
function fakeGrid(words: number, gridType = "Float", fill = 1): NanoVDBGrid {
  const image = new Uint32Array(words);
  for (let i = 0; i < words; i++) image[i] = fill + i;
  const metadata: GridMetadata = {
    name: "fake",
    gridType,
    gridClass: "FogVolume",
    worldBBox: { min: [0, 0, 0], max: [1, 1, 1] },
    indexBBox: { min: [0, 0, 0], max: [1, 1, 1] },
    voxelSize: [1, 1, 1],
    voxelCount: 0,
    gridByteSize: words * 4,
  };
  return new NanoVDBGrid({ image, metadata });
}

/** The material's bound backing array (private seam, reached only in tests). */
function boundArray(mat: NanoVDBVolumeMaterial): Uint32Array {
  return (mat as unknown as { _gridAttribute: StorageBufferAttribute })._gridAttribute.array as Uint32Array;
}

describe.skipIf(!wgslPresent)("NanoVDBVolumeMaterial.rebindGrid", () => {
  it("same-size rebind copies the new image and bumps the attribute version", async () => {
    const src = await loadSource();
    const a = fakeGrid(64, "Float", 100);
    const boundAttr = a.storageAttribute; // the instance the material will bind (default path)
    const mat = new NanoVDBVolumeMaterial({ grid: a, pnanovdbSource: src });
    expect(mat.grid).toBe(a);
    expect(mat.capacityBytes).toBe(64 * 4);
    const versionBefore = boundAttr.version;

    const b = fakeGrid(64, "Float", 500);
    mat.rebindGrid(b);

    expect(mat.grid).toBe(b);
    expect(Array.from(boundArray(mat))).toEqual(Array.from(b.image));
    expect(boundAttr.version).toBeGreaterThan(versionBefore);
  });

  it("rebinds a SMALLER grid in place (fits under the initial capacity)", async () => {
    const src = await loadSource();
    const a = fakeGrid(64, "Float");
    const mat = new NanoVDBVolumeMaterial({ grid: a, pnanovdbSource: src });
    const smaller = fakeGrid(32, "Float", 900);
    expect(() => mat.rebindGrid(smaller)).not.toThrow();
    expect(mat.grid).toBe(smaller);
    // The first 32 words hold the new image; the tail is stale-but-inert.
    expect(Array.from(boundArray(mat).subarray(0, 32))).toEqual(Array.from(smaller.image));
  });

  it("throws when the new grid exceeds capacity, pointing at maxGridBytes", async () => {
    const src = await loadSource();
    const a = fakeGrid(16, "Float");
    const mat = new NanoVDBVolumeMaterial({ grid: a, pnanovdbSource: src });
    const bigger = fakeGrid(20, "Float");
    expect(() => mat.rebindGrid(bigger)).toThrow(/capacity/i);
    expect(() => mat.rebindGrid(bigger)).toThrow(/maxGridBytes/);
    // The bound grid is unchanged after a failed rebind.
    expect(mat.grid).toBe(a);
  });

  it("maxGridBytes pre-sizes a padded buffer so a bigger grid rebinds in place", async () => {
    const src = await loadSource();
    const a = fakeGrid(16, "Float", 10);
    const mat = new NanoVDBVolumeMaterial({ grid: a, pnanovdbSource: src, maxGridBytes: 64 * 4 });
    expect(mat.capacityBytes).toBe(64 * 4);
    // Constructor copied frame 0 into the padded backing.
    expect(Array.from(boundArray(mat).subarray(0, 16))).toEqual(Array.from(a.image));

    const bigger = fakeGrid(48, "Float", 700);
    expect(() => mat.rebindGrid(bigger)).not.toThrow();
    expect(mat.grid).toBe(bigger);
    expect(Array.from(boundArray(mat).subarray(0, 48))).toEqual(Array.from(bigger.image));
  });

  it("rounds maxGridBytes up to a whole u32 and never shrinks below the grid", async () => {
    const src = await loadSource();
    const a = fakeGrid(16, "Float");
    // 65 bytes -> 17 words, but the grid already needs 16; capacity = max(16,17) = 17.
    const mat = new NanoVDBVolumeMaterial({ grid: a, pnanovdbSource: src, maxGridBytes: 65 });
    expect(mat.capacityBytes).toBe(17 * 4);
    // A maxGridBytes smaller than the grid is ignored (capacity stays at the grid size).
    const mat2 = new NanoVDBVolumeMaterial({ grid: fakeGrid(32, "Float"), pnanovdbSource: src, maxGridBytes: 8 });
    expect(mat2.capacityBytes).toBe(32 * 4);
  });

  it("refuses a grid-type change (the sampler is baked at construction)", async () => {
    const src = await loadSource();
    const floatGrid = fakeGrid(64, "Float");
    const mat = new NanoVDBVolumeMaterial({ grid: floatGrid, pnanovdbSource: src });
    const fp8Grid = fakeGrid(64, "Fp8");
    expect(() => mat.rebindGrid(fp8Grid)).toThrow(/grid type changed/i);
    expect(mat.grid).toBe(floatGrid);
  });
});

describe("unsupported grid types are gated upstream by NanoVDBGrid", () => {
  it("NanoVDBGrid throws before a material is ever built", () => {
    const bogus = {
      name: "bogus",
      gridType: "Double",
      gridClass: "Unknown",
      worldBBox: { min: [0, 0, 0], max: [1, 1, 1] },
      indexBBox: { min: [0, 0, 0], max: [1, 1, 1] },
      voxelSize: [1, 1, 1],
      voxelCount: 0,
      gridByteSize: 0,
    } as const;
    expect(() => new NanoVDBGrid({ image: new Uint32Array(0), metadata: bogus })).toThrow(/unsupported/i);
  });
});
