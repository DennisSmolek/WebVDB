import { expect, test, type Page } from "@playwright/test";

/**
 * Gallery demo 16 — coarse-first streaming into the froxel renderer. smoke.vdb is baked coarse-first in the page,
 * streamed in 1 KiB chunks (`?test=1`: unthrottled, the stream held until each grid's frame is read back) through
 * `vdbStreamCoarseFirst` into one pool id. The preview must draw from the first ~14 KiB and differ from the final
 * frame, and the final frame must equal the full grid added directly (`?mode=direct`, as demo 09's golden checks
 * its full grid). Runs with and without the gallery assets (`?gallery=0`: procedural sky).
 */
const DEMO = "/src/demos/16-streaming/index.html";
const SOURCE = "/fixtures/vdb-samples/smoke.vdb";
const BASE = "presentation=readback&test=1&still=1&asset=smoke&quality=low&temporal=0";
/** His smoke.wvdb needs 15,155 B before LOD0 can draw (demo 09's gate); ours with whole 1 KiB chunks. */
const GATE = Math.floor(14.8 * 1024);

interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  thumb: number[];
  fine: number[];
}

test.use({ viewport: { width: 480, height: 320 } });
test.setTimeout(600_000);

/** Mean absolute difference of the 96 × 54 grey thumbnails: a preview's lost detail averages out at 32 × 18. */
const fineDiff = (a: Snapshot, b: Snapshot): number => a.fine.reduce((s, v, i) => s + Math.abs(v - b.fine[i]!), 0) / a.fine.length;
const thumbSpread = (s: Snapshot): number => Math.max(...s.thumb) - Math.min(...s.thumb);

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

async function openDone(page: Page, query: string): Promise<NonNullable<Window["__DEMO16__"]>> {
  await page.goto(`${DEMO}?${query}`);
  await page.waitForFunction(() => window.__DEMO16__?.error !== undefined || window.__DEMO16__?.done === true, undefined, { timeout: 240_000 });
  return (await page.evaluate(() => window.__DEMO16__))!;
}

test.describe("demo 16 — gallery: streaming", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const gallery of [true, false]) {
    test(`preview from the first ~14 KiB, then the full grid in place, equal to the direct render (gallery assets ${gallery ? "on" : "off"})`, async ({ page }) => {
      const fixture = await page.request.head(SOURCE);
      test.skip(!fixture.ok(), `fixtures missing (${SOURCE} -> ${fixture.status()}): pnpm fixtures:vdb-samples`);
      const errors = collectErrors(page);
      const g = gallery ? "" : "&gallery=0";

      const state = await openDone(page, `${BASE}${g}`);
      const detail = () => JSON.stringify({ ...state, phases: state.phases.map((p) => ({ ...p, snapshot: p.snapshot ? "…" : null })), errors }, null, 2);
      expect(state.error, detail()).toBeUndefined();
      expect(state.presentation, detail()).toBe("readback");
      expect(state.asset, detail()).toBe("smoke");
      if (!gallery) expect(state.sky, detail()).toBe("procedural");

      // Two grids, the preview first, in one pool id with no growth.
      expect(state.phases.map((p) => [p.phase, p.grid]), detail()).toEqual([
        ["preview", "density.lod48"],
        ["full", "density"],
      ]);
      const [preview, full] = state.phases as [(typeof state.phases)[number], (typeof state.phases)[number]];
      expect(state.sameId, detail()).toBe(true);
      expect(state.inPlace, detail()).toBe(true);
      expect(preview.transferBytes, detail()).toBeGreaterThan(0);
      expect(preview.transferBytes, detail()).toBeLessThanOrEqual(GATE);
      expect(preview.voxelRatio, detail()).toBeGreaterThan(1);
      expect(full.transferBytes, detail()).toBe(state.totalBytes);
      expect(state.transferred, detail()).toBe(state.totalBytes);
      await expect(page.getByTestId("phase")).toHaveAttribute("data-phase", "full");

      // Both phases draw the volume; the preview is visibly coarser than the full grid.
      const p = preview.snapshot!;
      const f = full.snapshot!;
      expect(thumbSpread(p), detail()).toBeGreaterThan(10);
      expect(thumbSpread(f), detail()).toBeGreaterThan(10);
      const swap = fineDiff(p, f);

      // The streamed final frame equals the full grid added directly (no stream), placed and drawn the same way.
      const direct = await openDone(page, `${BASE}${g}&mode=direct`);
      expect(direct.error, JSON.stringify(direct)).toBeUndefined();
      const reference = direct.reference!;
      const final = fineDiff(f, reference);
      console.log(`demo 16 (gallery ${gallery ? "on" : "off"}): preview at ${preview.transferBytes} B of ${state.totalBytes}; preview->full diff ${swap.toFixed(2)}, full vs direct ${final.toFixed(3)}`);
      // Low-quality froxels blur both a lot (a 0.3 mean difference); the direct render is bit-identical (0.000), so the
      // preview must stand clearly above that floor and the streamed final at it.
      expect(swap, detail()).toBeGreaterThan(0.15);
      expect(final, detail()).toBeLessThan(swap * 0.1);

      expect(state.uncaptured, detail()).toEqual([]);
      expect(direct.uncaptured, detail()).toEqual([]);
      expect(errors, detail()).toEqual([]);
    });
  }
});
