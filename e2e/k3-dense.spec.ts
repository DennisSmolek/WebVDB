import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages, PARITY_BARS } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.0 (R1 §6 Step 0), our side of the parity rig. The page is examples/src/harness/k3-dense.ts:
// the committed CC0 cloud_04_variant_0000.wvdb as the dense source his renderer would upload,
// rendered by a minimal emission-absorption march through the VDBVolumeSource node shape, read
// back offscreen. Gates: the render is deterministic (two page loads give byte-identical PNGs,
// and two frames in one page are identical), and the core vdbDenseSource WGSL matches its CPU twin.
// PNGs land in test-results and in artifacts/k3/ours/ for scripts/k3/compare.mjs.
const PAGE_URL = "/src/harness/k3-dense.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  volume?: { dimensions: number[]; lod: number; meanDensity: number; mipLevels: number; statsSource: string };
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  nonBackground?: number;
  probes?: { count: number; maxAbs: number; meanAbs: number; over: number; tolerance: number; lodLevels: number; texels: number[]; firstBad?: unknown };
  timings?: { provisional: true; renderMs: number | null; note: string };
  uncaptured: string[];
}

async function run(page: Page, query = ""): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_DENSE__?: { done: boolean } }).__VDB_K3_DENSE__?.done === true, undefined, {
    timeout: 110_000,
  });
  return (await page.evaluate(() => (window as { __VDB_K3_DENSE__?: unknown }).__VDB_K3_DENSE__)) as Result;
}

const image = (r: Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });

test("K3.0 dense source: deterministic render through the VDBVolumeSource nodes; core WGSL matches its CPU twin", async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  const first = await run(page);
  const detail = JSON.stringify({ ...first, rgba: first.rgba ? `${first.rgba.length} b64 chars` : undefined }, null, 2);
  expect(first.error, detail).toBeUndefined();
  expect(first.uncaptured, detail).toEqual([]);
  expect(first.volume, detail).toMatchObject({ dimensions: [84, 128, 43], lod: 2, mipLevels: 7, statsSource: "texels" });
  expect(first.inPageIdentical, detail).toBe(true);
  // The cloud covers a good part of the 320x240 frame.
  expect(first.nonBackground!, detail).toBeGreaterThan(5_000);

  // Core vdbDenseSource: the generated vdbSrcDensity on this device vs sampleWvdbDenseTexture.
  const p = first.probes!;
  expect(p.count, detail).toBe(2048);
  expect(p.texels, detail).toEqual([84, 128, 43]);
  expect(p.lodLevels, detail).toBe(7);
  expect(p.over, detail).toBe(0);

  // Determinism: a second page load renders byte-identical pixels and PNG bytes.
  const second = await run(page);
  expect(second.error).toBeUndefined();
  const a = image(first);
  const b = image(second);
  const cmp = compareImages(a, b, PARITY_BARS.strict);
  expect(cmp.psnr, JSON.stringify(cmp)).toBe(Infinity);
  expect(cmp.above).toBe(0);
  const pngA = encodePng(a);
  const pngB = encodePng(b);
  expect(Buffer.from(pngA).equals(Buffer.from(pngB))).toBe(true);

  writeFileSync(testInfo.outputPath("k3-dense-run1.png"), pngA);
  writeFileSync(testInfo.outputPath("k3-dense-run2.png"), pngB);
  const out = path.join(repoRoot, "artifacts", "k3", "ours");
  mkdirSync(out, { recursive: true });
  writeFileSync(path.join(out, "k3-dense-cloud04-run1.png"), pngA);
  writeFileSync(path.join(out, "k3-dense-cloud04-run2.png"), pngB);
  writeFileSync(
    path.join(out, "k3-dense-cloud04.json"),
    JSON.stringify({ adapter: first.adapter, volume: first.volume, probes: first.probes, timings: first.timings, nonBackground: first.nonBackground }, null, 2),
  );
  console.log(
    `[k3-dense] ${first.adapter}: ${first.nonBackground} volume px; probes max |d| ${p.maxAbs.toExponential(2)} (mean ${p.meanAbs.toExponential(2)}); two loads identical`,
  );
});

test("K3.0 dense source without mips (lod ignored) is deterministic too", async ({ page }) => {
  test.setTimeout(240_000);
  const first = await run(page, "?mips=0");
  expect(first.error).toBeUndefined();
  expect(first.volume?.mipLevels).toBe(0);
  expect(first.probes?.lodLevels).toBe(0);
  expect(first.probes?.over).toBe(0);
  const second = await run(page, "?mips=0");
  expect(compareImages(image(first), image(second)).psnr).toBe(Infinity);
});
