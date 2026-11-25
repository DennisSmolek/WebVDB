import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K6.5 (design §7.2): the K3 Step 6 scenes (examples/src/harness/k3-frame-shadows.ts) through the core renderer
// (`?backend=core`: the core's volume shadow maps and emission probes, received by three) against the TSL path on the
// same device. The shadow and emission terms are isolated as (on − off) per backend (castShadows off / the emission
// light off), so the froxel composite's own core-vs-TSL silhouette difference (V80) cancels; the whole frames are
// reported. Two core loads are byte-identical. PNGs: artifacts/k6/shadows/.
const PAGE_URL = "/src/harness/k3-frame-shadows.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(repoRoot, "artifacts", "k6", "shadows");

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  width?: number;
  height?: number;
  rgba?: string;
  rgbaOff?: string;
  inPageIdentical?: boolean;
  warnings: string[];
  uncaptured: string[];
  entry?: { size: number; cascades: number; drawn: number; waiting: number } | null;
  emitters?: number;
  shadowStats?: { groundPixels: number; covered: number; shadowedPixels: number; meanDarkening: number };
  lightOff?: { groundPixels: number; nonBlack: number };
  backend?: { requested: string; core: boolean; maps: number; probes: boolean };
  coverage?: string;
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_SHADOWS__?: { done: boolean } }).__VDB_K3_SHADOWS__?.done === true, undefined, { timeout: 380_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_SHADOWS__?: unknown }).__VDB_K3_SHADOWS__)) as Result;
}

const bytes = (b64: string) => new Uint8Array(Buffer.from(b64, "base64"));
const image = (r: Result, b64: string) => ({ width: r.width!, height: r.height!, data: bytes(b64) });
const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64` : undefined, rgbaOff: r.rgbaOff ? `${r.rgbaOff.length} b64` : undefined }, null, 2);

function save(name: string, r: Result, b64: string): Buffer {
  const png = Buffer.from(encodePng(image(r, b64)) as Uint8Array);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, `${name}.png`), png);
  return png;
}

/**
 * The term a feature adds on meshes (on − off, signed, per channel) of each backend, compared: max and mean |Δ| in
 * 1/255, over the pixels outside the volume's dilated coverage of either backend (inside it the term is seen through
 * the froxel composite, whose silhouette differs between the backends by V80).
 */
function termDiff(tsl: Result, core: Result): { pixels: number; affected: number; maxDelta: number; meanDelta: number; over2: number } {
  const a = bytes(tsl.rgba!);
  const aOff = bytes(tsl.rgbaOff!);
  const b = bytes(core.rgba!);
  const bOff = bytes(core.rgbaOff!);
  const covA = bytes(tsl.coverage!);
  const covB = bytes(core.coverage!);
  let maxDelta = 0;
  let sum = 0;
  let over2 = 0;
  let affected = 0;
  let pixels = 0;
  for (let i = 0; i < a.length / 4; i++) {
    if (covA[i] || covB[i]) continue;
    pixels++;
    let px = 0;
    let any = false;
    for (let c = 0; c < 3; c++) {
      const ta = a[i * 4 + c]! - aOff[i * 4 + c]!;
      const tb = b[i * 4 + c]! - bOff[i * 4 + c]!;
      if (ta !== 0 || tb !== 0) any = true;
      px = Math.max(px, Math.abs(ta - tb));
    }
    if (any) affected++;
    maxDelta = Math.max(maxDelta, px);
    sum += px;
    if (px > 2) over2++;
  }
  return { pixels, affected, maxDelta, meanDelta: sum / pixels, over2 };
}

function expectClean(r: Result): void {
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.warnings, detail).toEqual([]);
  expect(r.inPageIdentical, detail).toBe(true);
}

const SHADOW_CASES: { name: string; query: string }[] = [
  { name: "sun, dense", query: "?case=shadow" },
  { name: "sun, 4 cascades", query: "?case=shadow&cascades=4" },
  { name: "spot", query: "?case=shadow&light=spot" },
  { name: "sun, sparse", query: "?case=shadow&source=sparse" },
  { name: "sun, 'frame' lighting", query: "?case=shadow&lighting=frame" },
];

for (const c of SHADOW_CASES) {
  test(`K6.5 A/B core vs TSL: volume shadows on meshes (${c.name})`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const tsl = await run(page, c.query);
    expectClean(tsl);
    const core = await run(page, `${c.query}&backend=core`);
    expectClean(core);
    // the core drew: its renderer exists and holds the light's map (the TSL run made none)
    expect(core.backend, detailOf(core)).toMatchObject({ requested: "core", core: true, maps: 1 });
    expect(tsl.backend, detailOf(tsl)).toMatchObject({ requested: "tsl", core: false });
    const slug = c.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
    save(`shadows-${slug}-tsl`, tsl, tsl.rgba!);
    save(`shadows-${slug}-core`, core, core.rgba!);
    const term = termDiff(tsl, core);
    const frame = compareImages(image(tsl, tsl.rgba!), image(core, core.rgba!));
    const frameOff = compareImages(image(tsl, tsl.rgbaOff!), image(core, core.rgbaOff!));
    console.log(
      `[k6-shadows] ${c.name} on ${core.adapter}: shadow term (on − off) core vs TSL: ${term.affected} px affected, max |Δ| ${term.maxDelta}/255, mean ${term.meanDelta.toFixed(4)}/255, ${term.over2} px > 2/255; frames ${frame.psnr.toFixed(2)} dB (${frame.above} px > 8/255), shadows off ${frameOff.psnr.toFixed(2)} dB; TSL ${JSON.stringify(tsl.shadowStats)} core ${JSON.stringify(core.shadowStats)}; entries tsl ${JSON.stringify(tsl.entry)} core ${JSON.stringify(core.entry)}`,
    );
    // The core maps shadow the ground like the TSL maps
    expect(core.shadowStats!.shadowedPixels, detailOf(core)).toBeGreaterThan(100);
    expect(Math.abs(core.shadowStats!.shadowedPixels - tsl.shadowStats!.shadowedPixels), detailOf(core)).toBeLessThanOrEqual(tsl.shadowStats!.shadowedPixels * 0.02);
    expect(term.affected, JSON.stringify(term)).toBeGreaterThan(100);
    expect(term.maxDelta, JSON.stringify(term)).toBeLessThanOrEqual(2);
    expect(term.over2, JSON.stringify(term)).toBe(0);
    expect(core.entry, detailOf(core)).toMatchObject({ size: tsl.entry!.size, cascades: tsl.entry!.cascades });
    expect(errors).toEqual([]);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// K6.5 gate 2/3 (design §7.2, §7.3): his K3 Step 6 scenes (examples/src/harness/k3-frame.ts) through `?backend=core`
// vs `?backend=tsl` on the same device (bar PSNR ≥ 45 dB, ≤ 0.2 % pixels > 4/255), 1280 × 720 with his froxels. The
// `*-shadow-only-dir-isolated` scenes show only the shadows on his floor (the core's maps, received by three) and gate;
// K6.3: the full cast-shadow scenes (his 'neural') gate too, the core with `occlusion=0` (K6.5b's mesh → volume shadows
// are not in his renderer); the explosion scenes are reported (V80 silhouette). Images go to artifacts/k6/ours-{core,tsl}/ for scripts/k3/compare.mjs
// against his captures (the regression report). Opt in with K6_CORE_AB=1 (needs fixtures/k3).
// ---------------------------------------------------------------------------------------------------------------
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const AB_STEP6: { scene: string; gate: boolean }[] = [
  { scene: "step6-cast-shadow-shadow-only-dir-isolated", gate: true },
  { scene: "step6-cast-shadow-obstacle-shadow-only-dir-isolated", gate: true },
  { scene: "step6-cast-shadow-obstacle-cascades-4-shadow-only-dir-isolated", gate: true },
  // K6.3: his 'neural' runs in the core (the generic MLP): the full cast-shadow scenes gate, with the core's mesh → volume
  // shadows off (K6.5b: his renderer has none; the obstacle shadows the bunny in the core otherwise)
  { scene: "step6-cast-shadow-dir-isolated", gate: true },
  { scene: "step6-cast-shadow-obstacle-dir-isolated", gate: true },
  { scene: "step6-cast-shadow-obstacle-cascades-4-dir-isolated", gate: true },
  { scene: "step6-cast-shadow-obstacle-coarse-dir-isolated", gate: true },
  { scene: "step6-cast-shadow-two-bunnies-dir-isolated", gate: true },
  { scene: "step6-explosion-emission-env-isolated", gate: false },
  { scene: "step6-explosion-emission-cascades-4-env-isolated", gate: false },
  { scene: "step6-explosion-emission-probes-8-env-isolated", gate: false },
];

interface K3Result {
  done: boolean;
  error?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  uncaptured: string[];
  warnings: string[];
  backend?: string;
}

async function k3(page: Page, query: string): Promise<K3Result> {
  await page.goto(`/src/harness/k3-frame.html${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 1_200_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as K3Result;
}
const k3Image = (r: K3Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });

for (const { scene, gate } of AB_STEP6) {
  test(`K6.5 A/B core vs TSL: ${scene}${gate ? "" : " (report)"}`, async ({ page }) => {
    test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1 (needs fixtures/k3)");
    test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing");
    test.setTimeout(2_400_000);
    const tsl = await k3(page, `?scene=${scene}&backend=tsl`);
    const core = await k3(page, `?scene=${scene}&backend=core&occlusion=0`);
    expect(tsl.error, JSON.stringify({ ...tsl, rgba: undefined })).toBeUndefined();
    expect(core.error, JSON.stringify({ ...core, rgba: undefined })).toBeUndefined();
    expect(core.uncaptured, JSON.stringify({ ...core, rgba: undefined })).toEqual([]);
    expect(core.backend).toBe("core");
    const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
    for (const [dir, r] of [["ours-core", core], ["ours-tsl", tsl]] as const) {
      mkdirSync(path.join(REPO, "artifacts", "k6", dir), { recursive: true });
      writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.png`), Buffer.from(encodePng(k3Image(r)) as Uint8Array));
      writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
    }
    console.log(`[k6-ab-step6] ${scene}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff}); core warnings ${JSON.stringify(core.warnings)}`);
    if (gate) expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  });
}

test("K6.5 A/B core vs TSL: the emission light (explosion frame 10, dense and sparse); two core loads agree", async ({ page }) => {
  test.setTimeout(1_200_000);
  const errors = collectErrors(page);
  let firstCore: Buffer | null = null;
  for (const source of ["dense", "sparse"] as const) {
    const q = `?case=emission&source=${source}`;
    const tsl = await run(page, q);
    expectClean(tsl);
    const core = await run(page, `${q}&backend=core`);
    expectClean(core);
    expect(core.emitters, detailOf(core)).toBe(1);
    expect(core.backend, detailOf(core)).toMatchObject({ requested: "core", core: true, probes: true });
    expect(core.lightOff!.nonBlack, detailOf(core)).toBe(0);
    save(`emission-${source}-tsl`, tsl, tsl.rgba!);
    const png = save(`emission-${source}-core`, core, core.rgba!);
    if (source === "dense") firstCore = png;
    const term = termDiff(tsl, core);
    const frame = compareImages(image(tsl, tsl.rgba!), image(core, core.rgba!));
    console.log(`[k6-emission] ${source} on ${core.adapter}: light term (on − off) core vs TSL: ${term.affected} px affected, max |Δ| ${term.maxDelta}/255, mean ${term.meanDelta.toFixed(4)}/255, ${term.over2} px > 2/255; frames ${frame.psnr.toFixed(2)} dB (${frame.above} px > 8/255)`);
    expect(term.affected, JSON.stringify(term)).toBeGreaterThan(1000);
    expect(term.maxDelta, JSON.stringify(term)).toBeLessThanOrEqual(2);
    expect(term.over2, JSON.stringify(term)).toBe(0);
  }
  const again = await run(page, "?case=emission&source=dense&backend=core");
  expectClean(again);
  const png = save("emission-dense-core-run2", again, again.rgba!);
  expect(png.equals(firstCore!), "two core page loads byte-identical").toBe(true);
  expect(errors).toEqual([]);
});
