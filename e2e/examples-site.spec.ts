import { expect, test, type Page } from "@playwright/test";

/**
 * Smoke of the BUILT examples site (RELEASING.md section 4), run by `pnpm examples:smoke` against
 * `<site>/` = http://127.0.0.1:<port>/vdb/examples. Skipped without WEBVDB_EXAMPLES_SITE_URL, so the normal e2e run
 * (dev server) needs no build.
 *
 * Per demo: the page loads, reaches its ready state (the same condition as its own spec) with no `error`, logs no
 * console errors, and requests nothing outside /vdb/. A 404 is allowed only for the assets that are intentionally
 * not hosted (gallery HDR skies and the car, `k3/sky-*`, the Stanford bunny, the waterfall): demos fall back.
 */
const SITE = process.env["WEBVDB_EXAMPLES_SITE_URL"];

type Ready = "defined" | "ready" | "done";
interface DemoCase {
  n: string;
  dir: string;
  query: string;
  ready: Ready;
}
const DEMOS: DemoCase[] = [
  { n: "01", dir: "01-hello-nvdb", query: "", ready: "defined" },
  { n: "02", dir: "02-cloud", query: "?fixture=primitive&presentation=readback", ready: "ready" },
  { n: "03", dir: "03-gpu-read", query: "", ready: "defined" },
  { n: "04", dir: "04-atlas-fallback", query: "?test=1&presentation=readback", ready: "defined" },
  { n: "05", dir: "05-embergen-sequence", query: "?test=1&presentation=readback", ready: "defined" },
  { n: "06", dir: "06-explorer", query: "?test=1&sample=nvdb&presentation=readback", ready: "defined" },
  { n: "07", dir: "07-builder", query: "?test=1&presentation=readback", ready: "defined" },
  { n: "08", dir: "08-webgpu-smoke", query: "?test=1&presentation=readback", ready: "ready" },
  { n: "09", dir: "09-coarse-first", query: "?test=1", ready: "ready" },
  { n: "10", dir: "10-frame-renderer", query: "?presentation=readback&quality=low&play=0", ready: "ready" },
  { n: "11", dir: "11-sky-clouds", query: "?presentation=readback&still=1&quality=low&temporal=0", ready: "ready" },
  { n: "12", dir: "12-explosion", query: "?presentation=readback&still=1&play=0&quality=low&frame=30", ready: "ready" },
  { n: "13", dir: "13-level-sets", query: "?presentation=readback&still=1&load=icosahedron,teapot&model=icosahedron", ready: "ready" },
  { n: "14", dir: "14-points", query: "?presentation=readback&still=1", ready: "ready" },
  { n: "15", dir: "15-full-res-vs-dense", query: "?presentation=readback&still=1&quality=low&temporal=0&asset=smoke", ready: "ready" },
  { n: "16", dir: "16-streaming", query: "?presentation=readback&test=1&still=1&asset=smoke&quality=low&temporal=0", ready: "done" },
];

/** Assets deliberately not hosted (see examples/assets.manifest.json `excluded` and the pending gallery rows). */
const ALLOWED_404 = /\/fixtures\/(gallery\/(?!CloudPackVDB_wvdb)|k3\/(sky-|bunny_cloud)|vdb-samples\/(bunny|waterfall))/;

test.skip(!SITE, "set WEBVDB_EXAMPLES_SITE_URL (pnpm examples:smoke builds the server for you)");
test.use({ viewport: { width: 480, height: 320 } });
test.setTimeout(1_000_000);

function watch(page: Page): { errors: string[]; outside: string[]; missing: string[] } {
  const seen = { errors: [] as string[], outside: [] as string[], missing: [] as string[] };
  const origin = new URL(SITE!).origin;
  page.on("request", (req) => {
    const u = new URL(req.url());
    if (u.protocol === "data:" || u.protocol === "blob:") return;
    if (u.origin !== origin || !u.pathname.startsWith("/vdb/")) seen.outside.push(req.url());
  });
  page.on("response", (res) => {
    if (res.status() < 400) return;
    const p = new URL(res.url()).pathname;
    if (res.status() === 404 && ALLOWED_404.test(p)) seen.missing.push(p);
    else seen.errors.push(`${res.status()} ${res.url()}`);
  });
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    // The browser's own line for a response already judged above.
    if (/Failed to load resource/.test(msg.text()) && ALLOWED_404.test(msg.location().url)) return;
    seen.errors.push(`console: ${msg.text()}`);
  });
  page.on("pageerror", (err) => seen.errors.push(`pageerror: ${err.message}`));
  return seen;
}

test.describe("examples site", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  test("home lists every demo, links resolve under the base", async ({ page }) => {
    const seen = watch(page);
    await page.goto(`${SITE}/`);
    const hrefs = await page.$$eval("#demo-list a, #gallery-list a, a[href$='attribution.html']", (as) => as.map((a) => (a as HTMLAnchorElement).href));
    expect(hrefs.length).toBeGreaterThanOrEqual(17);
    for (const href of hrefs) {
      expect(new URL(href).pathname.startsWith("/vdb/examples/"), href).toBe(true);
      expect((await page.request.get(href)).status(), href).toBe(200);
    }
    expect(seen.errors.filter((e) => !/WebGPU/.test(e))).toEqual([]);
    expect(seen.outside).toEqual([]);
  });

  test("attribution page lists the shipped assets with licences", async ({ page }) => {
    const seen = watch(page);
    await page.goto(`${SITE}/attribution.html`);
    const items = page.locator("#assets > li");
    expect(await items.count()).toBeGreaterThanOrEqual(6);
    const text = await page.locator("#assets").innerText();
    for (const needle of ["WDAS", "CC BY-SA 3.0", "CC BY-SA 4.0", "CC0 1.0", "JangaFX", "Modified"]) expect(text).toContain(needle);
    expect(seen.errors).toEqual([]);
    expect(seen.outside).toEqual([]);
  });

  for (const demo of DEMOS) {
    test(`demo ${demo.n} loads and renders from the built site`, async ({ page }) => {
      const seen = watch(page);
      const key = `__DEMO${demo.n}__`;
      await page.goto(`${SITE}/src/demos/${demo.dir}/index.html${demo.query}`);
      const waited = page.waitForFunction(
        ([k, mode]) => {
          const s = (window as unknown as Record<string, { error?: unknown; ready?: boolean; done?: boolean } | undefined>)[k!];
          if (!s) return false;
          if (s.error !== undefined && s.error !== null && s.error !== "") return true;
          return mode === "defined" ? true : mode === "ready" ? s.ready === true : s.done === true;
        },
        [key, demo.ready] as const,
        { timeout: Number(process.env["WEBVDB_SMOKE_TIMEOUT_MS"] ?? 900_000) },
      );
      await waited.catch(async (err: unknown) => {
        const status = await page.locator("#sc-status, #sc-stats, #sc-error").allInnerTexts().catch(() => []);
        throw new Error(`demo ${demo.n} not ready: ${String(err).split("\n")[0]}\npage: ${JSON.stringify(status)}\nerrors: ${JSON.stringify(seen.errors)}`);
      });
      const state = await page.evaluate((k) => (window as unknown as Record<string, { error?: unknown; info?: Record<string, unknown> }>)[k]!, key);
      const detail = JSON.stringify({ error: state.error, errors: seen.errors, outside: seen.outside }, null, 2);
      expect(state.error ?? null, detail).toBeFalsy();
      expect(seen.errors, detail).toEqual([]);
      expect(seen.outside, detail).toEqual([]);
      const home = await page.$$eval("a", (as) => as.filter((a) => /all demos/.test(a.textContent ?? "")).map((a) => (a as HTMLAnchorElement).href));
      if (home.length) expect(home[0]).toBe(`${new URL(SITE!).origin}/vdb/examples/`);
      console.log(`demo ${demo.n}: ok; intentionally missing: ${[...new Set(seen.missing)].join(", ") || "none"}`);
      if (demo.n === "14") {
        expect(state.info?.["model"]).toBe("sphere");
        const caption = await page.locator("#caption p").first().innerText();
        expect(caption).not.toMatch(/waterfall/i);
        expect(caption).toMatch(/9,802/);
      }
    });
  }

  for (const demo of [DEMOS[0]!, DEMOS[10]!, DEMOS[15]!]) {
    test(`demo ${demo.n} shows the no-WebGPU card without navigator.gpu`, async ({ page }) => {
      await page.addInitScript(() => Object.defineProperty(navigator, "gpu", { value: undefined, configurable: true }));
      await page.goto(`${SITE}/src/demos/${demo.dir}/index.html`);
      const card = page.getByTestId("no-webgpu");
      await expect(card).toBeVisible();
      await expect(card).toContainText("needs WebGPU");
      expect(await card.locator("a").getAttribute("href")).toBe("/vdb/examples/");
    });
  }
});
