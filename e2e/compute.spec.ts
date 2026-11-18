import { expect, test } from "@playwright/test";

// K2.3 gate: the core compute tools (gridStats, decodeToAtlas,
// applyValueTransform) on a real device against their tools CPU twins
// (examples/src/harness/compute.ts). Five grids in one pool, both stats
// paths, host-encoder and self-submitted calls.
const PAGE_URL = "/src/harness/compute.html";
const PROBE_FIXTURE_URL = "/fixtures/primitives/box_fog_float.nvdb";

interface ComputeReport {
  done: boolean;
  error?: string;
  missing?: string[];
  adapter?: string;
  stats: {
    grid: string;
    fastPath: boolean;
    fastPathTwin: boolean;
    count: number;
    countTwin: number;
    minExact: boolean;
    maxExact: boolean;
    meanAbs: number;
    histogramL1: number;
    histogramSum: number;
  }[];
  atlas: { grid: string; label: string; dims: number[]; voxels: number; exact: number; maxAbs: number; dataDiff: number; dataMaxAbs: number }[];
  transform?: {
    leafSamples: number;
    tileSamples: number;
    leafDoubled: boolean;
    tilesUnchanged: boolean;
    atlasExact: number;
    atlasVoxels: number;
    statsCountMatch: boolean;
    statsMaxExact: boolean;
    versionUnchanged: boolean;
  };
  readbackFalseOk?: boolean;
  disposeRejects?: boolean;
  uncaptured: string[];
  lastError?: string;
}

test("core compute tools match their CPU twins on the GPU", async ({ page }) => {
  const probe = await page.request.get(PROBE_FIXTURE_URL);
  test.skip(!probe.ok(), `fixtures missing (${PROBE_FIXTURE_URL} -> ${probe.status()})`);
  test.setTimeout(180_000);

  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __VDB_COMPUTE__?: { done: boolean } }).__VDB_COMPUTE__?.done === true, undefined, {
    timeout: 170_000,
  });
  const r = (await page.evaluate(() => (window as { __VDB_COMPUTE__?: unknown }).__VDB_COMPUTE__)) as ComputeReport;
  const detail = JSON.stringify(r, null, 2);

  expect(r.error, detail).toBeUndefined();
  expect(r.missing, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.lastError, detail).toBeUndefined();

  expect(r.stats, detail).toHaveLength(5);
  expect(r.stats.filter((s) => !s.fastPath).map((s) => s.grid), detail).toEqual(["signed_float"]);
  for (const s of r.stats) {
    expect(s.fastPath, s.grid).toBe(s.fastPathTwin);
    expect(s.count, s.grid).toBe(s.countTwin);
    expect(s.count, s.grid).toBeGreaterThan(0);
    expect(s.histogramSum, s.grid).toBe(s.count);
    expect(s.minExact && s.maxExact, `${s.grid} min/max`).toBe(true);
    // f32 normalization may round a voxel into the next bin on a GPU with an inexact division.
    expect(s.histogramL1, `${s.grid} histogram`).toBeLessThanOrEqual(Math.max(2, s.count * 1e-4));
    expect(s.meanAbs, `${s.grid} mean`).toBeLessThan(1e-5);
  }

  expect(r.atlas, detail).toHaveLength(25);
  for (const a of r.atlas) {
    const what = `${a.grid} ${a.label}`;
    expect(a.maxAbs, what).toBeLessThan(1e-5);
    // Formatted outputs: at most one uint8 step (a rounding tie), or f32 noise.
    expect(a.dataMaxAbs, what).toBeLessThanOrEqual(a.label.includes("uint8") ? 1 : 1e-5);
  }

  const t = r.transform!;
  expect(t, detail).toBeDefined();
  expect(t.leafSamples, detail).toBeGreaterThan(0);
  expect(t.leafDoubled && t.tilesUnchanged, detail).toBe(true);
  expect(t.versionUnchanged, detail).toBe(true);
  expect(t.atlasExact, detail).toBe(t.atlasVoxels);
  expect(t.statsCountMatch && t.statsMaxExact, detail).toBe(true);

  expect(r.readbackFalseOk, detail).toBe(true);
  expect(r.disposeRejects, detail).toBe(true);

  // SwiftShader rounds every f32 op like the twins (no FMA, exact division): bit-exact there
  // (measured 2026-11-18: every count, extreme, mean, bin and atlas value). Hardware keeps the tolerances above.
  if (/swiftshader/i.test(r.adapter ?? "")) {
    for (const s of r.stats) expect([s.histogramL1, s.meanAbs], `${s.grid} bit-exact on SwiftShader`).toEqual([0, 0]);
    for (const a of r.atlas) expect([a.exact, a.dataDiff], `${a.grid} ${a.label} bit-exact on SwiftShader`).toEqual([a.voxels, 0]);
  }

  const exact = r.atlas.reduce((n, a) => n + a.exact, 0);
  const total = r.atlas.reduce((n, a) => n + a.voxels, 0);
  test.info().annotations.push({ type: "compute", description: `${r.adapter}; atlas raw ${exact}/${total} bit-exact vs f32 twin` });
});
