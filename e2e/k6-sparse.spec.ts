import { expect, test, type Page } from "@playwright/test";

// K6.7: the core sparse source (`vdbSparseSourceGpu`) on raw WebGPU against its CPU twins
// (examples/src/harness/k6-sparse.ts). Per LOD strategy ('none' + clamp01, 'coarse-pyramid', 'node-average'), five
// volumes over one pool (slot reuse, one grid under two boxes, Fp8, every aux channel), N = 1024 fixed-seed samples:
// density within the documented bound (1e-4 (1 + scale), + 2^-7 scale when a pyramid tap contributes), the aux
// channels within the level-0 bound, the LOD range / texels / max exact, the HDDA spans equal to the twin's, and
// the HDDA exactness property (every skipped sample reads 0). Each GPU-side mutation must fail the probe. G16a: a
// coarse-first swap through a followImage source creates no shader module and no pipeline, and the probe passes on
// the new image. Everything uses committed or synthetic grids: no fixtures, no skips.
const PAGE_URL = "/src/harness/k6-sparse.html";

interface Stat {
  count: number;
  over: number;
  maxAbs: number;
  maxRel: number;
}

interface SourceResult {
  name: string;
  lod: string;
  volumes: number;
  density: Stat;
  aux: Stat;
  exact: { mismatches: number; count: number };
  skip: { count: number; mismatches: number; skipped: number; enabled: number };
  exactness: { checked: number; violations: number };
}

interface Result {
  done: boolean;
  error?: string;
  sources: SourceResult[];
  swap?: { builds: number; relayouts: number; texelsBefore: number[]; texelsAfter: number[]; versionMoved: boolean; probe: SourceResult };
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string, timeout = 500_000): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K6_SPARSE__?: { done: boolean } }).__VDB_K6_SPARSE__?.done === true, undefined, { timeout });
  return (await page.evaluate(() => (window as { __VDB_K6_SPARSE__?: unknown }).__VDB_K6_SPARSE__)) as Result;
}

function passes(r: SourceResult): boolean {
  return r.density.over + r.aux.over + r.exact.mismatches + r.skip.mismatches + r.exactness.violations === 0;
}

test.describe.configure({ mode: "serial", timeout: 600_000 });

test("K6.7 core sparse source = CPU twins (3 LOD strategies, 4 volumes each, N = 1024) + G16a zero-build swap", async ({ page }) => {
  const r = await run(page, "?swap=1");
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.warnings, detail).toEqual([]);
  expect(r.sources.map((s) => s.lod), detail).toEqual(["none", "coarse-pyramid", "node-average"]);
  for (const s of r.sources) {
    expect(s.volumes, detail).toBe(4);
    expect(s.density.count, detail).toBe(1024);
    expect(passes(s), `${s.name}\n${detail}`).toBe(true);
    // The probe exercises the HDDA (some spans skip, some are enabled) and the exactness check ran.
    expect(s.skip.skipped, `${s.name} skipped\n${detail}`).toBeGreaterThan(0);
    expect(s.exactness.checked, `${s.name} exactness\n${detail}`).toBeGreaterThan(100);
  }
  const w = r.swap!;
  expect(w, detail).toBeDefined();
  expect(w.builds, detail).toBe(0);
  expect(w.relayouts, detail).toBe(1);
  expect(w.versionMoved, detail).toBe(true);
  expect(w.texelsAfter, detail).not.toEqual(w.texelsBefore);
  expect(passes(w.probe), `swap probe\n${detail}`).toBe(true);
});

for (const mutate of ["samplePos", "slot", "origin", "blend", "dilate"]) {
  test(`K6.7 fail-before: mutation '${mutate}' fails the probe`, async ({ page }) => {
    const r = await run(page, `?mutate=${mutate}&n=512`);
    const detail = JSON.stringify(r, null, 2);
    expect(r.error, detail).toBeUndefined();
    expect(r.uncaptured, detail).toEqual([]);
    expect(r.sources.length, detail).toBe(3);
    expect(r.sources.some((s) => !passes(s)), detail).toBe(true);
  });
}
