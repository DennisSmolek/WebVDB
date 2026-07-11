import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { NodeMaterial } from "three/webgpu";
import { NanoVDBFile } from "nanovdb-wgsl";
import { NanoVDBGrid } from "../src/grid.js";
import { NanoVDBVolumeMaterial } from "../src/material.js";
import {
  assembleVolumeWgsl,
  rewriteBufferGlobal,
  samplerForGridType,
  GRID_TYPE_FLOAT,
  GRID_TYPE_FP8,
  GRID_TYPE_FPN,
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
