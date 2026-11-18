import { expect, test } from "@playwright/test";

// K2 review gate: the ./three adapter's guards on three r187 + a real GPU
// (one renderer per VDBGridNode, one VDB module per shader, the attribute
// self-test before any pool buffer reaches three, the own-device warning).
// The page is examples/src/harness/three-guards.ts.
const PAGE_URL = "/src/harness/three-guards.html";
const PROBE_FIXTURE_URL = "/fixtures/primitives/box_fog_float.nvdb";

interface Thrown {
  threw: boolean;
  code?: string;
  message?: string;
}
interface Result {
  done: boolean;
  error?: string;
  missing?: string[];
  selfTests?: number[];
  sameNodeTwice?: { ok: boolean; read: number; expected: number; error?: string };
  otherRenderer?: Thrown & { read?: number };
  ownNodeOnB?: { read: number; expected: number };
  twoModules?: Thrown;
  materialOtherRenderer?: Thrown;
  bypass?: Thrown & { poolIntact: boolean; validationError?: string };
  warnings?: string[];
  uncaptured: string[];
}

test("three adapter guards: other renderer, two modules, self-test, own device", async ({ page }) => {
  const probe = await page.request.get(PROBE_FIXTURE_URL);
  test.skip(!probe.ok(), `fixtures missing (${PROBE_FIXTURE_URL} -> ${probe.status()})`);
  test.setTimeout(180_000);

  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __VDB_THREE_GUARDS__?: { done: boolean } }).__VDB_THREE_GUARDS__?.done === true, undefined, {
    timeout: 170_000,
  });
  const r = (await page.evaluate(() => (window as { __VDB_THREE_GUARDS__?: unknown }).__VDB_THREE_GUARDS__)) as Result;
  const detail = JSON.stringify(r, null, 2);

  expect(r.error, detail).toBeUndefined();
  expect(r.missing, detail).toBeUndefined();
  // One self-test per backend: A, A again (cached), B.
  expect(r.selfTests, detail).toEqual([1, 1, 2]);
  expect(r.sameNodeTwice?.ok, detail).toBe(true);

  expect(r.otherRenderer, detail).toMatchObject({ threw: true, code: "VDB_BAD_OPTION" });
  expect(r.otherRenderer!.message, detail).toMatch(/one VDBGridNode per renderer/);
  expect(r.ownNodeOnB!.read, detail).toBe(r.ownNodeOnB!.expected);
  expect(r.ownNodeOnB!.expected, detail).not.toBe(0);

  expect(r.twoModules, detail).toMatchObject({ threw: true, code: "VDB_BAD_OPTION" });
  expect(r.twoModules!.message, detail).toMatch(/One VDB module per shader/);
  expect(r.materialOtherRenderer, detail).toMatchObject({ threw: true, code: "VDB_BAD_OPTION" });

  expect(r.bypass, detail).toMatchObject({ threw: true, code: "VDB_ENGINE_MISMATCH", poolIntact: true });
  expect(r.bypass!.message, detail).toMatch(/without going through backend.destroyAttribute/);

  const ownDevice = r.warnings!.filter((w) => w.includes("created its own GPUDevice"));
  expect(ownDevice, detail).toHaveLength(1);
  expect(r.warnings!.filter((w) => w.includes("outside the supported range")), detail).toEqual([]); // three r187
  expect(r.uncaptured, detail).toEqual([]);
});
