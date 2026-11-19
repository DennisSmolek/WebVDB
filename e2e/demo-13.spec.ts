import { expect, test, type Page } from "@playwright/test";

/**
 * Demo 13 — the level-set sculpture gallery. One small piece (the icosahedron, 1 MB) keeps SwiftShader runtime low:
 * it must load in readback presentation, cover a real part of the frame (the coverage probe renders once without
 * the gallery's content and counts the pixels that change), raise no WebGPU validation error, and a material change
 * and the normal view must change the image. Runs with the gallery assets off (`?gallery=0`, procedural studio) and
 * with them on when `fixtures/gallery/` has the HDR skies (else that test is skipped).
 */
const DEMO = "/src/demos/13-level-sets/index.html";
const BASE = "presentation=readback&still=1&load=icosahedron&model=icosahedron";

interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  thumb: number[];
}

test.use({ viewport: { width: 480, height: 320 } });
test.setTimeout(360_000);

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

async function open(page: Page, query: string): Promise<NonNullable<Window["__DEMO13__"]>> {
  await page.goto(`${DEMO}?${BASE}&${query}`);
  await page.waitForFunction(() => window.__DEMO13__?.error !== undefined || window.__DEMO13__?.ready === true, undefined, { timeout: 300_000 });
  return (await page.evaluate(() => window.__DEMO13__))!;
}

/** Wait for `n` more frames after a change, then summarize the last one. */
async function settle(page: Page, n = 3): Promise<Snapshot> {
  const from = await page.evaluate(() => window.__DEMO13__!.frames);
  await page.waitForFunction(([f, k]) => window.__DEMO13__!.error !== undefined || window.__DEMO13__!.frames >= f + k, [from, n] as const, { timeout: 300_000 });
  expect((await page.evaluate(() => window.__DEMO13__!)).error).toBeUndefined();
  return (await page.evaluate(() => window.__DEMO13_API__!.capture()))!;
}

const thumbDiff = (a: Snapshot, b: Snapshot): number => a.thumb.reduce((s, v, i) => s + Math.abs(v - b.thumb[i]!), 0) / a.thumb.length;
const thumbSpread = (s: Snapshot): number => Math.max(...s.thumb) - Math.min(...s.thumb);

test.describe("demo 13 — level-set gallery", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("gallery assets off: loads, draws the piece, controls change the image, no validation errors", async ({ page }) => {
    const errors = collectErrors(page);
    const state = await open(page, "gallery=0");
    const detail = () => JSON.stringify({ state, errors }, null, 2);
    test.skip(state.error?.includes("No level-set fixtures") === true, "fixtures/vdb-samples/icosahedron.vdb missing (pnpm fixtures:vdb-samples)");
    expect(state.error, detail()).toBeUndefined();
    expect(state.presentation).toBe("readback");
    expect(state.galleryAssets).toBe(false);
    expect(state.info.env, detail()).toBe("studio");
    expect(state.info.loaded, detail()).toEqual(["icosahedron"]);

    const base = await settle(page, 1);
    expect(base.width * base.height).toBe(480 * 320);
    expect(thumbSpread(base), detail()).toBeGreaterThan(20);
    // The piece and its plinth: a real share of the frame, not all of it.
    const covered = await page.evaluate(() => window.__DEMO13_API__!.coverage());
    expect(covered, detail()).toBeGreaterThan(480 * 320 * 0.08);
    expect(covered, detail()).toBeLessThan(480 * 320 * 0.95);

    await page.selectOption("#ctl-material", "chrome");
    const chrome = await settle(page);
    const dChrome = thumbDiff(base, chrome);
    await page.locator("#ctl-normals").check();
    const normals = await settle(page, 4);
    const dNormals = thumbDiff(chrome, normals);
    // The normal view is a saturated colour image: its mean is far from grey.
    const [r, g, b] = normals.mean;
    console.log(`demo 13 diffs: chrome ${dChrome.toFixed(2)}, normals ${dNormals.toFixed(2)}; covered ${covered}; normals mean ${normals.mean.map((v) => v.toFixed(0)).join(",")}`);
    expect(dChrome, detail()).toBeGreaterThan(1);
    expect(dNormals, detail()).toBeGreaterThan(1);
    expect(Math.max(r, g, b) - Math.min(r, g, b), detail()).toBeGreaterThan(3);

    const after = (await page.evaluate(() => window.__DEMO13__))!;
    expect(after.uncaptured, detail()).toEqual([]);
    expect(errors, detail()).toEqual([]);
  });

  test("gallery assets on: the HDR sky loads and the piece draws", async ({ page }) => {
    const errors = collectErrors(page);
    const hdr = await page.request.head("/fixtures/gallery/shore-HDR.jpg");
    test.skip(!hdr.ok(), "fixtures/gallery/ (local only, licences pending) has no shore-HDR.jpg");
    const state = await open(page, "env=shore");
    const detail = () => JSON.stringify({ state, errors }, null, 2);
    test.skip(state.error?.includes("No level-set fixtures") === true, "fixtures/vdb-samples/icosahedron.vdb missing");
    expect(state.error, detail()).toBeUndefined();
    expect(state.galleryAssets).toBe(true);
    expect(state.info.env, detail()).toBe("shore");
    const shot = await settle(page, 1);
    expect(thumbSpread(shot), detail()).toBeGreaterThan(20);
    const covered = await page.evaluate(() => window.__DEMO13_API__!.coverage());
    expect(covered, detail()).toBeGreaterThan(480 * 320 * 0.08);
    await page.selectOption("#ctl-env", "studio");
    const studio = await settle(page, 3);
    expect(thumbDiff(shot, studio), detail()).toBeGreaterThan(2);
    expect((await page.evaluate(() => window.__DEMO13__))!.uncaptured, detail()).toEqual([]);
    expect(errors, detail()).toEqual([]);
  });
});

declare global {
  interface Window {
    __DEMO13__?: {
      ready: boolean;
      loading: boolean;
      error?: string;
      frames: number;
      presentation: string;
      galleryAssets: boolean;
      uncaptured: string[];
      info: { env?: string; loaded?: string[]; [k: string]: unknown };
    };
    __DEMO13_API__?: { capture(): Snapshot | null; coverage(): Promise<number> };
  }
}
