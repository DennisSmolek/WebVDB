import { expect, test, type Page } from "@playwright/test";

// K7.1 (his 62a6e3e): spot and point volume-shadow lookups use the light state the map was drawn with, not three's
// current shadow coordinates. examples/src/harness/k3-frame-shadows.ts `?case=stale`: both shadow lights move in one
// frame and the budget (1 piece) draws the decoy, so the main light's map waits. That frame's ground must equal the
// frame before the move (the shadow stays where its map says); the next frame draws the map and the shadow moves.
// Then the main light alone moves and its map is drawn in that render: that frame must equal the next one.
const PAGE_URL = "/src/harness/k3-frame-shadows.html";

interface Stale {
  groundPixels: number;
  shadowedBefore: number;
  staleChanged: number;
  staleShadowedChanged: number;
  afterChanged: number;
  movedChanged: number;
  order: { decoy: number; main: number };
  staleUpdate: { drawn: number; waiting: number };
  afterUpdate: { drawn: number; waiting: number };
  movedUpdate: { drawn: number; waiting: number };
  mainDrawnAtStale: boolean;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  warnings: string[];
  uncaptured: string[];
  stale?: Stale;
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_SHADOWS__?: { done: boolean } }).__VDB_K3_SHADOWS__?.done === true, undefined, { timeout: 380_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_SHADOWS__?: unknown }).__VDB_K3_SHADOWS__)) as Result;
}

for (const light of ["spot", "point"] as const) {
  test(`K7.1 ${light} light: a map that waits for the budget keeps its shadow in place`, async ({ page }) => {
    test.setTimeout(600_000);
    const r = await run(page, `?case=stale&light=${light}`);
    const detail = JSON.stringify({ ...r, rgba: undefined, rgbaOff: undefined }, null, 2);
    expect(r.error, detail).toBeUndefined();
    expect(r.uncaptured, detail).toEqual([]);
    expect(r.warnings, detail).toEqual([]);
    const s = r.stale!;
    console.log(`[k7-stale ${light}] ${r.adapter}: ${JSON.stringify(s)}`);
    // The setup: the main light's map is the newer one, so the decoy goes first and the main map waits one render
    expect(s.order.main, detail).toBeGreaterThan(s.order.decoy);
    expect(s.staleUpdate, detail).toEqual({ drawn: 1, waiting: 1 });
    expect(s.mainDrawnAtStale, detail).toBe(false);
    expect(s.afterUpdate, detail).toEqual({ drawn: 1, waiting: 0 });
    // The volume shadow is on the ground, and the move moves it once the map is drawn
    expect(s.groundPixels, detail).toBeGreaterThan(5000);
    expect(s.shadowedBefore, detail).toBeGreaterThan(200);
    expect(s.afterChanged, detail).toBeGreaterThan(50);
    // While the map waits, receivers read it with its drawn state: the ground is unchanged, pixel for pixel
    expect(s.staleChanged, detail).toBe(0);
    // A map drawn in the render the light moved is read with that state at once: the next render (nothing changed)
    // is identical
    expect(s.movedUpdate, detail).toEqual({ drawn: 1, waiting: 0 });
    expect(s.movedChanged, detail).toBe(0);
  });
}
