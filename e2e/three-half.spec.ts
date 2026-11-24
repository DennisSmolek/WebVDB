import { expect, test } from "@playwright/test";
import { collectErrors } from "./console-errors";

// K7.H: a NanoVDB Half grid renders through @pmndrs/vdb/three. No fixtures: the page
// (examples/src/harness/three-half.ts) builds a fog ball, quantizes it to Half, and renders the Float and
// Half grids from one pool with one VDBVolumeMaterial (VDBGridNode types ['float', 'half']), then the Half
// grid alone from a pool whose node takes the pool's types (['half']).
const PAGE_URL = "/src/harness/three-half.html";

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  nonBackground?: { float: number; half: number; halfOnly: number };
  diff?: { max: number; mean: number; maxHalfOnly: number };
  nodeTypes?: { mixed: string[]; halfOnly: string[] };
  uncaptured: string[];
}

test("three: a Half grid renders in VDBVolumeMaterial like its Float source", async ({ page }) => {
  test.setTimeout(120_000);
  const errors = collectErrors(page);
  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __VDB_THREE_HALF__?: { done: boolean } }).__VDB_THREE_HALF__?.done === true, undefined, {
    timeout: 110_000,
  });
  const r = (await page.evaluate(() => (window as { __VDB_THREE_HALF__?: unknown }).__VDB_THREE_HALF__)) as Result;
  const detail = JSON.stringify({ ...r, errors }, null, 2);

  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.nodeTypes, detail).toEqual({ mixed: ["float", "half"], halfOnly: ["half"] });
  // The volume covers a good part of the 160x160 frame, the same in all three.
  expect(r.nonBackground!.float, detail).toBeGreaterThan(2000);
  expect(Math.abs(r.nonBackground!.half - r.nonBackground!.float), detail).toBeLessThanOrEqual(r.nonBackground!.float * 0.01);
  // Binary16 keeps ~3.3 significant digits: the frames differ by at most a couple of 8-bit steps.
  expect(r.diff!.max, detail).toBeLessThanOrEqual(3);
  expect(r.diff!.mean, detail).toBeLessThan(0.1);
  // The same Half grid through a half-only module renders the same frame.
  expect(r.diff!.maxHalfOnly, detail).toBe(0);
  expect(errors, detail).toEqual([]);
  test.info().annotations.push({
    type: "three-half",
    description: `${r.adapter}: ${r.nonBackground!.float}/${r.nonBackground!.half} px, max diff ${r.diff!.max}/255, mean ${r.diff!.mean.toFixed(4)}`,
  });
});
