import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.5 (R1 §6 Step 5): mjurczyk's spectral media and measured presets (his `spectral`, `getFrameMediumProps`) on both
// lighting paths, and the 'volume'-lighting environment / hemisphere / light-probe terms (examples/src/harness/k3-frame.ts).
// Gates until his Step 5 captures exist: no validation errors or warnings; the spectral chunks match their CPU twins; every
// spectral pass (transmittance per slot group, environment grid, density, lighting; inject + integrate) matches its twin;
// each preset renders deterministically; a grey spectral medium renders as the grey path; the white furnace holds on both
// lighting modes (1D diffusion, grey and spectral); sparse vs dense >= 35 dB with spectral on.
// PNGs go to artifacts/k3/ours/ (k3-frame-spectral-*.png).
const PAGE_URL = "/src/harness/k3-frame.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OURS = path.join(repoRoot, "artifacts", "k3", "ours");

interface Twin {
  compared: number;
  maxAbs: number;
  maxRel?: number;
  over: number;
  marched?: number;
  lit?: number;
  groups?: number;
  firstBad?: unknown;
}

interface Furnace {
  centerRatio: number[];
  volumePixels: number;
  meanRatio: number;
  minRatio: number;
  maxRatio: number;
  meanAbsDeviation: number;
  p05: number;
  p95: number;
  environment: { mode: string; count: number } | null;
}

interface Result {
  done: boolean;
  error?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  nonBackground?: number;
  firstFrameMs?: number;
  settings?: Record<string, unknown>;
  spectral?: boolean;
  medium?: string | null;
  spectralGroups?: number;
  chunkProbe?: { count: number; over: number };
  spectralProbe?: Twin;
  twin?: Twin & { covered: number };
  transmittanceTwin?: Twin;
  lightingTwin?: Twin;
  environmentGridTwin?: Twin;
  densityTwin?: Twin;
  furnace?: Furnace;
  redraw?: { shadowDensity: { drawn: number; skipped: number }; transmittance: { drawn: number; skipped: number }; environmentGrid: { drawn: number; skipped: number } };
  timings?: Record<string, unknown>;
  warnings: string[];
  uncaptured: string[];
  [k: string]: unknown;
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 400_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as Result;
}

const image = (r: Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });
const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64 chars` : undefined }, null, 2).slice(0, 6000);
const channelMeans = (r: Result) => {
  const d = image(r).data;
  const m = [0, 0, 0];
  for (let i = 0; i < d.length; i += 4) for (let c = 0; c < 3; c++) m[c]! += d[i + c]!;
  return m.map((v) => v / (d.length / 4));
};

function save(name: string, r: Result): void {
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, `${name}.png`), encodePng(image(r)) as Uint8Array);
}

function expectClean(r: Result): void {
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.warnings, detail).toEqual([]);
  expect(r.inPageIdentical, detail).toBe(true);
}

/** See k3-frame-lighting.spec.ts: the density twin gets 1 in 2,000 (f32 sub-step boundaries); the others none. */
function expectTwin(t: Twin | undefined, minCompared: number, detail: string, maxOverFraction = 0): void {
  expect(t, detail).toBeDefined();
  expect(t!.compared, detail).toBeGreaterThan(minCompared);
  expect(t!.over, `${JSON.stringify(t)}\n${detail}`).toBeLessThanOrEqual(Math.floor(t!.compared * maxOverFraction));
}

const fmt = (t: (Twin & { pixels?: number }) | undefined) => (t ? `${t.compared ?? t.pixels} (max ${t.maxAbs.toExponential(2)}, over ${t.over})` : "-");

// The K3.3/K3.4 e2e scene: the committed CC0 cloud, his sun + hemisphere + ambient, 2HG renderer phase, 'frame' lighting.
const BASE = "?w=160&h=96&froxels=96x64x24&lighting=frame&lights=dir-hemi";

for (const [name, extra] of [
  ["1d", "ms=1&medium=dark-smoke"],
  ["two-sided", "ms=two-sided&medium=haze"],
  ["octaves", "ms=octaves&medium=dust"],
  ["neural-env", "ms=neural&medium=white-smoke&env=synthetic"],
  ["octaves-env", "ms=octaves&medium=soot&env=synthetic"],
] as const) {
  test(`K3.5 spectral 'frame' lighting (${name}): every pass matches its twin; deterministic`, async ({ page }) => {
    test.setTimeout(600_000);
    const q = `${BASE}&${extra}&twins=1&redraw=2`;
    const r = await run(page, q);
    const detail = detailOf(r);
    mkdirSync(OURS, { recursive: true });
    writeFileSync(path.join(OURS, `k3-frame-spectral-${name}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
    expectClean(r);
    expect(r.spectral, detail).toBe(true);
    expect(r.nonBackground!, detail).toBeGreaterThan(1_500);
    expectTwin(r.spectralProbe, 2_000, detail);
    expectTwin(r.transmittanceTwin, 10_000, detail);
    expect(r.transmittanceTwin!.marched!, detail).toBeGreaterThan(500);
    expectTwin(r.lightingTwin, 1_000, detail);
    expectTwin(r.densityTwin, 5_000, detail, 5e-4);
    if (extra.includes("env=")) {
      // Octaves: the dominant environment light takes a slot (2 slots: groups 0 and 1); diffusion: every light on the grid
      expectTwin(r.environmentGridTwin, 10_000, detail);
      expect(r.spectralGroups, detail).toBe(extra.includes("octaves") ? 2 : 1);
    }
    // Static scene: the grid and every group's transmittance are drawn once (with the environment: once more when its
    // lights arrive from the readback and the dominant one takes a slot)
    expect(r.redraw!.transmittance.drawn, detail).toBeLessThanOrEqual(extra.includes("env=") ? 2 : 1);
    expect(r.redraw!.transmittance.skipped, detail).toBeGreaterThanOrEqual(3);
    const again = await run(page, q);
    expect(again.error).toBeUndefined();
    expect(compareImages(image(r), image(again)).psnr).toBe(Infinity);
    save(`k3-frame-spectral-${name}`, r);
    console.log(
      `[k3-frame-spectral] '${name}': probe ${fmt(r.spectralProbe)}, transmittance ${fmt(r.transmittanceTwin)} / ${r.transmittanceTwin!.marched} voxels / ${r.spectralGroups} groups, lighting ${fmt(r.lightingTwin)}, density ${fmt(r.densityTwin)}, environment grid ${fmt(r.environmentGridTwin)}; first frame ${r.firstFrameMs?.toFixed(0)} ms`,
    );
  });
}

test("K3.5 four slot groups: six shadow slots (his sun + 5 directional lights) match their twins in every group", async ({ page }) => {
  test.setTimeout(600_000);
  const r = await run(page, `${BASE}&ms=1&medium=dark-smoke&extraLights=5&twins=1`);
  const detail = detailOf(r);
  expectClean(r);
  expect(r.spectralGroups, detail).toBe(4);
  expectTwin(r.transmittanceTwin, 30_000, detail);
  expect(r.transmittanceTwin!.groups, detail).toBe(4);
  expectTwin(r.lightingTwin, 1_000, detail);
  expectTwin(r.densityTwin, 5_000, detail, 5e-4);
  save("k3-frame-spectral-groups-4", r);
  console.log(`[k3-frame-spectral] 4 groups: transmittance ${fmt(r.transmittanceTwin)}, lighting ${fmt(r.lightingTwin)}`);
});

test("K3.5 a grey spectral medium renders as the grey path; a Rayleigh spectrum tints the cloud", async ({ page }) => {
  test.setTimeout(600_000);
  const grey = await run(page, `${BASE}&ms=1`);
  const spectralGrey = await run(page, `${BASE}&ms=1&spectral=1&angstrom=0`);
  expectClean(grey);
  expectClean(spectralGrey);
  const d = compareImages(image(grey), image(spectralGrey));
  expect(d.psnr, JSON.stringify(d)).toBeGreaterThan(45);
  expect(d.above, JSON.stringify(d)).toBe(0);
  // A blue-heavy spectrum (Ångström 4.09, Rayleigh air) reddens the transmitted light: the channels move apart
  const air = await run(page, `${BASE}&ms=1&spectral=1&angstrom=4.09`);
  expectClean(air);
  const [gr, , gb] = channelMeans(spectralGrey);
  const [ar, , ab] = channelMeans(air);
  expect(Math.abs(ar! - ab! - (gr! - gb!)), `air ${channelMeans(air)} grey ${channelMeans(spectralGrey)}`).toBeGreaterThan(0.5);
  save("k3-frame-spectral-grey", spectralGrey);
  save("k3-frame-spectral-air", air);
  console.log(`[k3-frame-spectral] grey spectral vs grey: ${d.psnr.toFixed(2)} dB (${d.above} px > 8/255); air channel means ${channelMeans(air).map((v) => v.toFixed(2)).join("/")} vs grey ${channelMeans(spectralGrey).map((v) => v.toFixed(2)).join("/")}`);
});

const MEDIA = ["cloud", "fog", "ice", "soot", "white-smoke", "dark-smoke", "haze", "dust"];

test("K3.5 every medium preset renders cleanly and deterministically on both lighting paths", async ({ page }) => {
  test.setTimeout(1_500_000);
  const lines: string[] = [];
  const grey = await run(page, `${BASE}&ms=1`);
  for (const medium of MEDIA) {
    const r = await run(page, `${BASE}&ms=1&medium=${medium}`);
    const detail = detailOf(r);
    expectClean(r);
    expect(r.medium, detail).toBe(medium);
    expect(r.spectral, detail).toBe(true);
    expect(r.nonBackground!, detail).toBeGreaterThan(1_000);
    expectTwin(r.spectralProbe, 2_000, detail);
    const again = await run(page, `${BASE}&ms=1&medium=${medium}`);
    expect(compareImages(image(r), image(again)).psnr).toBe(Infinity);
    const v = await run(page, `?w=160&h=96&froxels=160x96x24&lighting=volume&ms=1&medium=${medium}&twin=1`);
    const vd = detailOf(v);
    expectClean(v);
    expect(v.twin!.over, vd).toBe(0);
    expect(v.twin!.covered, vd).toBeGreaterThan(50);
    expect(v.chunkProbe!.over, vd).toBe(0);
    save(`k3-frame-spectral-medium-${medium}`, r);
    save(`k3-frame-spectral-volume-medium-${medium}`, v);
    const vsGrey = compareImages(image(r), image(grey));
    lines.push(`${medium}: vs grey 2HG ${vsGrey.psnr.toFixed(2)} dB, channel means ${channelMeans(r).map((x) => x.toFixed(1)).join("/")}; 'volume' integrate twin ${fmt(v.twin)}`);
  }
  // Soot (albedo 0.25) is darker than fog (white) in the same light
  const soot = await run(page, `${BASE}&ms=1&medium=soot`);
  const fog = await run(page, `${BASE}&ms=1&medium=fog`);
  expect(channelMeans(soot)[1]!).toBeLessThan(channelMeans(fog)[1]!);
  console.log(`[k3-frame-spectral] media:\n  ${lines.join("\n  ")}`);
});

test("K3.5 'volume' lighting: environment, hemisphere and light terms render (no warnings), change the image, and keep the twins", async ({ page }) => {
  test.setTimeout(600_000);
  const q = "?w=160&h=96&froxels=160x96x24&lighting=volume&twin=1";
  const sun = await run(page, `${q}&ms=1&lights=dir`);
  const hemi = await run(page, `${q}&ms=1&lights=dir-hemi`);
  const env = await run(page, `${q}&ms=1&lights=dir&env=synthetic`);
  const envSpectral = await run(page, `${q}&ms=1&lights=dir-hemi&env=synthetic&medium=dark-smoke`);
  const lines: string[] = [];
  for (const [name, r] of [["hemi", hemi], ["env", env], ["env-spectral", envSpectral]] as const) {
    const detail = detailOf(r);
    expectClean(r);
    expect(r.twin!.over, detail).toBe(0);
    expect(r.chunkProbe!.over, detail).toBe(0);
    const d = compareImages(image(r), image(sun));
    // The new terms add light: the frame gets brighter than the sun alone
    expect(d.above, `${name} ${JSON.stringify(d)}`).toBeGreaterThan(100);
    expect(channelMeans(r)[1]!).toBeGreaterThan(channelMeans(sun)[1]!);
    save(`k3-frame-spectral-volume-${name}`, r);
    lines.push(`${name}: vs sun only ${d.psnr.toFixed(2)} dB (${d.above} px > 8/255), integrate twin ${fmt(r.twin)}`);
  }
  expect(env.environment, detailOf(env)).toBeTruthy();
  console.log(`[k3-frame-spectral] 'volume' lighting terms: ${lines.join("; ")}`);
});

test("K3.5 sparse source with spectral on: identical voxels are >= 35 dB from dense (both lighting paths)", async ({ page }) => {
  test.setTimeout(900_000);
  const lines: string[] = [];
  for (const q of [
    "?w=320&h=180&froxels=128x128x24&lighting=frame&lights=dir-hemi&ms=1&medium=dark-smoke",
    "?w=320&h=180&froxels=128x128x24&lighting=frame&lights=dir-hemi&ms=octaves&medium=dust",
    "?w=320&h=180&froxels=128x128x24&lighting=volume&lights=dir-hemi&ms=1&medium=haze",
  ]) {
    const dense = await run(page, `${q}&source=dense`);
    const sparse = await run(page, `${q}&source=sparse`);
    expectClean(dense);
    expectClean(sparse);
    const ab = compareImages(image(dense), image(sparse));
    expect(ab.psnr, `${q}: ${JSON.stringify(ab)}`).toBeGreaterThanOrEqual(35);
    lines.push(`${q.split("&").slice(3).join("&")} ${ab.psnr.toFixed(2)} dB (${ab.above} px > 8/255)`);
  }
  console.log(`[k3-frame-spectral] sparse vs dense: ${lines.join("; ")}`);
});

// His white furnace (utils/furnace.js) on both lighting paths, grey and spectral (a white medium of any Ångström
// exponent): 1D diffusion must return the environment (v / L within 3% on average), the other modes are recorded.
test("K3.5 white furnace on 'volume' and 'frame' lighting, grey and spectral", async ({ page }) => {
  test.setTimeout(1_200_000);
  const rows: Record<string, Furnace> = {};
  for (const lighting of ["volume", "frame"]) {
    for (const spectral of ["", "&spectral=1&angstrom=1.8"]) {
      for (const ms of ["1", "0", "octaves"]) {
        const r = await run(page, `?w=160&h=96&froxels=96x64x24&lighting=${lighting}&furnace=1&ms=${ms}&extinction=40${spectral}`);
        const detail = detailOf(r);
        expectClean(r);
        const f = r.furnace!;
        expect(f.environment?.mode, detail).toBe("directional");
        expect(f.volumePixels, detail).toBeGreaterThan(200);
        rows[`${lighting}${spectral ? " spectral" : ""} ms=${ms}`] = f;
        if (ms === "1") {
          expect(Math.abs(f.meanRatio - 1), detail).toBeLessThan(0.03);
          for (const c of f.centerRatio) expect(Math.abs(c - 1), detail).toBeLessThan(0.03);
          save(`k3-frame-spectral-furnace-${lighting}${spectral ? "-spectral" : ""}`, r);
        }
      }
    }
  }
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, "k3-frame-spectral-furnace.json"), JSON.stringify(rows, null, 2));
  console.log(
    `[k3-frame-spectral] furnace (v / L, pixels with alpha > 0.5; centre 9x9 / L per channel):\n` +
      Object.entries(rows)
        .map(([k, f]) => `  ${k}: mean ${f.meanRatio.toFixed(3)} (p05 ${f.p05.toFixed(3)}, p95 ${f.p95.toFixed(3)}, min ${f.minRatio.toFixed(3)}, max ${f.maxRatio.toFixed(3)}), centre ${f.centerRatio.map((v) => v.toFixed(3)).join("/")}`)
        .join("\n"),
  );
});

// Provisional timings (D25: SwiftShader on a loaded laptop; never a gate). Opt in: K3_SPECTRAL_TIMING=1 (needs
// fixtures/k3/bunny_cloud.wvdb). His step3-default-dir-isolated scene at 1920x1080 (512x512x24 froxels, grid 64³), grey
// and spectral per lighting path: the froxel passes with the grid cached, the extra cost of a grid redraw (shadow
// density + one transmittance pass per slot group), the first frame (compile included) and the transmittance passes.
test("K3.5 timing report (opt-in): spectral vs grey pass count and ms", async ({ page }) => {
  test.skip(process.env.K3_SPECTRAL_TIMING !== "1", "report-only: K3_SPECTRAL_TIMING=1");
  test.setTimeout(3_600_000);
  const rows: Record<string, unknown>[] = [];
  for (const extra of ["ms=1", "ms=1&spectral=1", "ms=1&medium=dark-smoke&env=synthetic", "ms=octaves&medium=dark-smoke&env=synthetic", "ms=1&lighting=volume", "ms=1&lighting=volume&spectral=1"]) {
    const r = await run(page, `?scene=step3-default-dir-isolated&w=1920&h=1080&${extra}&timing=1&timingFrames=15&timingWarmup=3&redraw=1`);
    expect(r.error, detailOf(r)).toBeUndefined();
    const t = r.timings as { froxelsMs: number | null; integrateMs: number | null; sceneMs: number | null; passes?: { gridRedrawMs: number | null } };
    rows.push({ extra, groups: r.spectralGroups ?? null, transmittanceDraws: r.redraw?.transmittance.drawn ?? null, firstFrameMs: r.firstFrameMs, froxelsMs: t.froxelsMs, gridRedrawMs: t.passes?.gridRedrawMs ?? null, integrateMs: t.integrateMs, sceneMs: t.sceneMs });
  }
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, "k3-frame-spectral-timings.json"), JSON.stringify(rows, null, 2));
  console.log(`[k3-frame-spectral] timings (provisional, D25): ${JSON.stringify(rows, null, 2)}`);
});
