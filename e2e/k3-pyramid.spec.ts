import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

// K3.2a: the sparse source's coarse pyramid built on the GPU (core `createBoxPyramid`) against its CPU
// twin (tools `vdbBoxPyramid` + three's `DataUtils.toHalfFloat`, the K3.2 path), texel for texel on
// the r16f textures (examples/src/harness/k3-pyramid.ts). Gate: every base and atlas texel within
// 1 f16 ulp (the f32 GPU sums vs the f64 CPU sums can only move a truncation by one step), no
// validation errors; a pool.replace (sequence frame) marks it stale and the rebuild matches the CPU
// pyramid of the new image; through three, vdbSparseSource's GPU textures match the CPU path's and
// a render after a replace rebuilds exactly once.
const PAGE_URL = "/src/harness/k3-pyramid.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const fixture = (rel: string) => existsSync(path.join(repoRoot, "fixtures", rel));

interface Compare {
  texels: number;
  exact: number;
  ulp1: number;
  over: number;
  maxUlp: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  assets: { name: string; base: Compare; atlas: Compare | null; level: number; dimensions: number[]; gpuBytes: { textures: number; scratch: number } }[];
  sequence?: { staleBefore: boolean; rebuilt: boolean; staleAfter: boolean; again: boolean; base: Compare; atlas: Compare | null };
  three?: {
    gpuVsCpu: { base: Compare; atlas: Compare | null };
    buildsAfterCreate: number;
    buildsAfterRender: number;
    buildsAfterReplace: number;
    buildsAfterSecondRender: number;
    replaced: { base: Compare; atlas: Compare | null };
    versions: [number, number];
  };
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string, timeout = 400_000): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_PYRAMID__?: { done: boolean } }).__VDB_K3_PYRAMID__?.done === true, undefined, { timeout });
  return (await page.evaluate(() => (window as { __VDB_K3_PYRAMID__?: unknown }).__VDB_K3_PYRAMID__)) as Result;
}

/**
 * Within 1 f16 ulp everywhere. (No bit-exact share is required: SwiftShader is bit-exact on > 99.9% of texels, but
 * on Apple Metal ~21% of the Fp8 sphere's texels sit 1 ulp off: its Fp8 values land on f16 steps, and a contracted
 * f32 multiply-add a hair below one truncates to the step under it. K3.2c hardware run, docs/handoffs/K3-2a.md.)
 */
function expectParity(c: Compare | null, what: string): void {
  expect(c, what).not.toBeNull();
  expect(c!.over, what).toBe(0);
  expect(c!.maxUlp, what).toBeLessThanOrEqual(1);
  expect(c!.exact + c!.ulp1, what).toBe(c!.texels);
}

test.describe.configure({ mode: "serial", timeout: 600_000 });

test("K3.2a GPU pyramid = CPU pyramid (≤ 1 f16 ulp): cloud04 (L₀ 1 and 2), sphere Float/Fp8, sequence replace, three source", async ({ page }) => {
  const primitives = fixture("primitives/sphere_fog_float.nvdb") && fixture("primitives/sphere_fog_fp8.nvdb") && fixture("primitives/torus_fog_float.nvdb");
  test.skip(!primitives, "fixtures missing: pnpm fixtures:bake");
  const r = await run(page, "?assets=sphere,cloud04,cloud04@32,sphere8&sequence=1&three=1&repeat=1");
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.warnings, detail).toEqual([]);
  expect(r.assets.map((a) => a.name), detail).toEqual(["sphere", "cloud04", "cloud04@32", "sphere8"]);
  for (const a of r.assets) {
    expectParity(a.base, `${a.name} base\n${detail}`);
    if (a.atlas) expectParity(a.atlas, `${a.name} atlas\n${detail}`);
    expect(a.gpuBytes.textures, detail).toBeGreaterThan(0);
  }
  // cloud04@32: 84x128x43 → L₀ 2, base 21x32x11 (z is fractional: 43/11 voxels per texel).
  const c32 = r.assets.find((a) => a.name === "cloud04@32")!;
  expect([c32.level, ...c32.dimensions], detail).toEqual([2, 21, 32, 11]);

  const s = r.sequence!;
  expect(s, detail).toBeDefined();
  expect([s.staleBefore, s.rebuilt, s.staleAfter, s.again], detail).toEqual([false, true, false, false]);
  expectParity(s.base, `sequence base\n${detail}`);
  if (s.atlas) expectParity(s.atlas, `sequence atlas\n${detail}`);

  const t = r.three!;
  expect(t, detail).toBeDefined();
  expectParity(t.gpuVsCpu.base, `three base\n${detail}`);
  expectParity(t.gpuVsCpu.atlas, `three atlas\n${detail}`);
  // One build at creation (the grid is resident), none for a render without a change, one after the replace.
  expect([t.buildsAfterCreate, t.buildsAfterRender, t.buildsAfterReplace, t.buildsAfterSecondRender], detail).toEqual([1, 1, 1, 2]);
  expectParity(t.replaced.base, `three replaced base\n${detail}`);
  expectParity(t.replaced.atlas, `three replaced atlas\n${detail}`);
  // The source's content version follows the grid's imageRevision ('frame' lighting redraws its shadow grid).
  expect(t.versions[1], detail).toBeGreaterThan(t.versions[0]);
  // K3.2e: the density mean (normalizeDensity) follows the replace: read back from the pool, the same stats as a
  // source made from the new image, and the scale uniform divides by the new mean.
  expect(t.stats.updates, detail).toBe(1);
  expect(t.stats.after, detail).toEqual(t.stats.expected);
  expect(t.stats.after.meanNonZero, detail).not.toBe(t.stats.before.meanNonZero);
  expect(t.stats.scale[1], detail).toBe(t.stats.expectedScale);
  expect(t.stats.scale[1], detail).not.toBe(t.stats.scale[0]);
});

test("K3.2a GPU pyramid = CPU pyramid on smoke.vdb (fractional box weights) and the sphere level set (inactive ±3 tiles: blocks with no active voxel read 0)", async ({ page }) => {
  test.skip(!fixture("vdb-samples/smoke.vdb") || !fixture("vdb-samples/sphere.vdb"), "fixtures missing: pnpm fixtures:vdb-samples");
  const r = await run(page, "?assets=smoke,spherels&repeat=1");
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.warnings, detail).toEqual([]);
  for (const a of r.assets) {
    expectParity(a.base, `${a.name} base\n${detail}`);
    expectParity(a.atlas, `${a.name} atlas\n${detail}`);
  }
});
