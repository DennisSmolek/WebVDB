import { expect, test } from "@playwright/test";

// K1.3b acceptance (docs/design/K1-core.md §8): nine fixtures in a VDBGridPool
// started at 4 KiB, read back on a real device; every image range must be
// byte-identical to its file and every record must point at it.
const PAGE_URL = "/src/harness/pool.html";
const PROBE_FIXTURE_URL = "/fixtures/primitives/box_fog_float.nvdb";

interface PoolReadback {
  done: boolean;
  error?: string;
  missing?: string[];
  fixtures: { name: string; identical: boolean; recordOk: boolean; imageOffset: number; firstDiff: number }[];
  versions: number[];
  byteLength: number;
  headerOk: boolean;
  validationError?: string;
}

test("VDBGridPool: nine fixtures from a 4 KiB pool read back byte-identical", async ({ page }) => {
  const probe = await page.request.get(PROBE_FIXTURE_URL);
  test.skip(!probe.ok(), `fixtures missing (${PROBE_FIXTURE_URL} -> ${probe.status()})`);
  test.setTimeout(90_000);

  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __VDB_POOL_READBACK__?: { done: boolean } }).__VDB_POOL_READBACK__?.done === true, undefined, {
    timeout: 80_000,
  });
  const r = (await page.evaluate(() => (window as { __VDB_POOL_READBACK__?: unknown }).__VDB_POOL_READBACK__)) as PoolReadback;
  const detail = JSON.stringify(r, null, 2);

  expect(r.error, detail).toBeUndefined();
  expect(r.missing, detail).toBeUndefined();
  expect(r.validationError, detail).toBeUndefined();
  expect(r.headerOk, detail).toBe(true);
  expect(r.fixtures, detail).toHaveLength(9);
  // Started at 4 KiB: every batch forced a growth, so the buffer was replaced at least three times.
  expect(r.versions[2], detail).toBeGreaterThanOrEqual(3);
  for (const f of r.fixtures) {
    expect(f.identical, `${f.name} differs at byte ${f.firstDiff}`).toBe(true);
    expect(f.recordOk, `${f.name} record`).toBe(true);
    expect(f.imageOffset % 32, f.name).toBe(0);
  }
  test.info().annotations.push({ type: "pool", description: `${r.byteLength} bytes, versions ${r.versions.join(",")}` });
});
