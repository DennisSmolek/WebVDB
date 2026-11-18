import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.4 (R1 §6 Step 4): mjurczyk's phase functions (dual-lobe, Lorenz-Mie, Mie RGB, phaseGExponent) and multiple
// scattering modes (true, 'two-sided', 'octaves', 'neural') on both lighting paths (examples/src/harness/k3-frame.ts).
// Gates until his captures exist: no validation errors or warnings; the K3.4 chunks match their CPU twins; the
// transmittance (with the network over the read-back grid mips), grid mip, density, lighting and environment grid
// passes match their twins in every mode; each mode renders, differs from multipleScattering off as expected and is
// deterministic; sparse vs dense >= 35 dB in the new modes; the flipped defaults ('frame', temporal, 'neural') render.
// The white furnace (his utils/furnace.js) is recorded per mode, not forced (his modes are approximations).
// PNGs go to artifacts/k3/ours/ (k3-frame-scattering-*.png).
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
  levels?: number;
  firstBad?: unknown;
}

interface Furnace {
  radiance: number;
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
  chunkProbe?: { count: number; over: number };
  twin?: Twin & { covered: number };
  scatterProbe?: Twin;
  gridProbe?: Twin;
  gridMipTwin?: Twin;
  transmittanceTwin?: Twin;
  lightingTwin?: Twin;
  environmentGridTwin?: Twin;
  densityTwin?: Twin;
  furnace?: Furnace;
  redraw?: { shadowDensity: { drawn: number; skipped: number }; transmittance: { drawn: number; skipped: number }; environmentGrid: { drawn: number; skipped: number } };
  warnings: string[];
  uncaptured: string[];
  [k: string]: unknown;
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 300_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as Result;
}

const image = (r: Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });
const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64 chars` : undefined }, null, 2).slice(0, 6000);
const mean = (r: Result) => image(r).data.reduce((s: number, v: number, i: number) => (i % 4 === 3 ? s : s + v), 0);

function save(name: string, r: Result): Uint8Array {
  const png = encodePng(image(r)) as Uint8Array;
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, `${name}.png`), png);
  return png;
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

const fmt = (t: Twin | undefined) => (t ? `${t.compared} (max ${t.maxAbs.toExponential(2)}, over ${t.over})` : "-");

// The K3.3 e2e scene: the committed CC0 cloud, his sun + hemisphere + ambient, 2HG renderer phase, 'frame' lighting.
const BASE = "?w=160&h=96&froxels=96x64x24&lighting=frame&lights=dir-hemi";

for (const ms of ["neural", "two-sided", "octaves"] as const) {
  test(`K3.4 'frame' lighting, multipleScattering '${ms}': passes match their twins; differs from single scattering; deterministic`, async ({ page }) => {
    test.setTimeout(500_000);
    const q = `${BASE}&ms=${ms}&twins=1&redraw=2`;
    const r = await run(page, q);
    const detail = detailOf(r);
    mkdirSync(OURS, { recursive: true });
    writeFileSync(path.join(OURS, `k3-frame-scattering-${ms}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
    expectClean(r);
    expect(r.settings!.multipleScattering, detail).toBe(ms);
    expect(r.nonBackground!, detail).toBeGreaterThan(1_500);
    expectTwin(r.scatterProbe, 1_500, detail);
    expectTwin(r.transmittanceTwin, 10_000, detail);
    expect(r.transmittanceTwin!.marched!, detail).toBeGreaterThan(500);
    expectTwin(r.lightingTwin, 1_000, detail);
    expectTwin(r.densityTwin, 5_000, detail, 5e-4);
    if (ms === "neural") {
      expectTwin(r.gridMipTwin, 1_000, detail);
      expect(r.gridMipTwin!.levels, detail).toBe(5);
    }
    // Static scene: the grid, its mips and the transmittance are drawn once
    expect(r.redraw!.transmittance.drawn, detail).toBe(1);
    expect(r.redraw!.transmittance.skipped, detail).toBeGreaterThanOrEqual(3);

    const again = await run(page, q);
    expect(again.error).toBeUndefined();
    expect(compareImages(image(r), image(again)).psnr).toBe(Infinity);

    const off = await run(page, `${BASE}&ms=0`);
    const on = await run(page, `${BASE}&ms=1`);
    for (const x of [off, on]) expect(x.error, detailOf(x)).toBeUndefined();
    const vsOff = compareImages(image(r), image(off));
    const vs1d = compareImages(image(r), image(on));
    // Multiple scattering adds light to the shadowed cloud: many pixels change and the frame gets brighter.
    expect(vsOff.above, JSON.stringify(vsOff)).toBeGreaterThan(100);
    expect(mean(r)).toBeGreaterThan(mean(off));
    // ...and it is not the 1D diffusion image.
    expect(vs1d.mse, JSON.stringify(vs1d)).toBeGreaterThan(0);
    save(`k3-frame-scattering-${ms}`, r);
    console.log(
      `[k3-frame-scattering] '${ms}': scatter probe ${fmt(r.scatterProbe)}, transmittance ${fmt(r.transmittanceTwin)} / ${r.transmittanceTwin!.marched} voxels, lighting ${fmt(r.lightingTwin)}, density ${fmt(r.densityTwin)}, grid mips ${fmt(r.gridMipTwin)}; vs off ${vsOff.psnr.toFixed(2)} dB (${vsOff.above} px > 8/255), vs 1D ${vs1d.psnr.toFixed(2)} dB; mean ${mean(r)} vs off ${mean(off)} vs 1D ${mean(on)}; first frame ${r.firstFrameMs?.toFixed(0)} ms (1D ${on.firstFrameMs?.toFixed(0)} ms)`,
    );
  });
}

test("K3.4 environment + octaves: the environment grid and the octave totals match their twins", async ({ page }) => {
  test.setTimeout(500_000);
  // Octaves: no diffusion, so the dominant environment light takes a slot and the rest go to the environment grid
  const r = await run(page, `${BASE}&env=synthetic&lights=dir-hemi&ms=octaves&twins=1`);
  const detail = detailOf(r);
  expectClean(r);
  expectTwin(r.environmentGridTwin, 10_000, detail);
  expectTwin(r.transmittanceTwin, 10_000, detail);
  expectTwin(r.lightingTwin, 1_000, detail);
  expectTwin(r.densityTwin, 5_000, detail, 5e-4);
  // Neural with the environment: every environment light goes to the grid (diffusion); the network lights the sun slot
  const n = await run(page, `${BASE}&env=synthetic&lights=dir-hemi&ms=neural&twins=1`);
  const nd = detailOf(n);
  expectClean(n);
  expectTwin(n.environmentGridTwin, 10_000, nd);
  expectTwin(n.transmittanceTwin, 10_000, nd);
  expectTwin(n.densityTwin, 5_000, nd, 5e-4);
  console.log(`[k3-frame-scattering] environment: octaves grid ${fmt(r.environmentGridTwin)}, neural grid ${fmt(n.environmentGridTwin)} transmittance ${fmt(n.transmittanceTwin)}`);
});

const PHASES: [string, string][] = [
  ["isotropic", "phase=isotropic"],
  ["lorenz-mie", "phase=lorenz-mie"],
  ["lorenz-mie-rgb", "phase=lorenz-mie-rgb"],
  ["g-exponent", "phaseGExponent=1"],
];

test("K3.4 phase functions ('frame' lighting): Lorenz-Mie, Mie RGB, phaseGExponent and isotropic render, differ from 2HG, and match the twins", async ({ page }) => {
  test.setTimeout(600_000);
  const dual = await run(page, `${BASE}&ms=1`);
  expect(dual.error).toBeUndefined();
  const lines: string[] = [];
  for (const [name, q] of PHASES) {
    const r = await run(page, `${BASE}&ms=1&${q}&twins=1`);
    const detail = detailOf(r);
    expectClean(r);
    expectTwin(r.transmittanceTwin, 10_000, detail);
    // The lighting twin takes the per-channel phase (the direct maps at the visible position) since K3.4.
    expectTwin(r.lightingTwin, 1_000, detail);
    expectTwin(r.densityTwin, 5_000, detail, 5e-4);
    const d = compareImages(image(r), image(dual));
    expect(d.mse, `${name} vs 2HG ${JSON.stringify(d)}`).toBeGreaterThan(0);
    if (name === "lorenz-mie-rgb" || name === "g-exponent") {
      // Per-channel phases: the grey cloud picks up a colour cast (channels differ somewhere).
      const px = image(r).data;
      // (g-exponent: small with his sun at ~70 degrees to the view; 17 px > 2/255 on the CC0 cloud)
      const ref = image(dual).data;
      let tinted = 0;
      for (let i = 0; i < px.length; i += 4) if (Math.abs(px[i]! - px[i + 2]! - (ref[i]! - ref[i + 2]!)) >= 2) tinted++;
      expect(tinted, `${name}: pixels whose red - blue moved by 2/255 or more from 2HG`).toBeGreaterThan(5);
      lines.push(`${name}: ${tinted} px tinted`);
    }
    save(`k3-frame-scattering-phase-${name}`, r);
    lines.push(`${name}: vs 2HG ${d.psnr.toFixed(2)} dB, lighting twin ${fmt(r.lightingTwin)}, density ${fmt(r.densityTwin)}`);
  }
  console.log(`[k3-frame-scattering] phases: ${lines.join("; ")}`);
});

test("K3.4 'volume' lighting: two-sided, octaves and Lorenz-Mie in the inject pass render, differ from 1D, and the integrate twin holds", async ({ page }) => {
  test.setTimeout(600_000);
  const q = "?w=160&h=96&froxels=160x96x24&lighting=volume&twin=1";
  const base = await run(page, `${q}&ms=1`);
  expect(base.error, detailOf(base)).toBeUndefined();
  const lines: string[] = [];
  for (const extra of ["ms=two-sided", "ms=octaves", "ms=1&phase=lorenz-mie", "ms=octaves&phase=lorenz-mie-rgb"]) {
    const r = await run(page, `${q}&${extra}`);
    const detail = detailOf(r);
    expectClean(r);
    expect(r.chunkProbe!.over, detail).toBe(0);
    expect(r.twin!.over, detail).toBe(0);
    expect(r.twin!.covered, detail).toBeGreaterThan(50);
    const d = compareImages(image(r), image(base));
    expect(d.mse, `${extra} vs 1D ${JSON.stringify(d)}`).toBeGreaterThan(0);
    save(`k3-frame-scattering-volume-${extra.replace(/[=&]/g, "-")}`, r);
    lines.push(`${extra}: vs 1D ${d.psnr.toFixed(2)} dB, mean ${mean(r)} (1D ${mean(base)})`);
  }
  console.log(`[k3-frame-scattering] 'volume' lighting: ${lines.join("; ")}`);
});

test("K3.4 sparse source: every new mode on identical voxels is >= 35 dB from dense", async ({ page }) => {
  test.setTimeout(900_000);
  const q = "?w=320&h=180&froxels=128x128x24&lighting=frame&lights=dir-hemi";
  const lines: string[] = [];
  for (const extra of ["ms=neural", "ms=two-sided", "ms=octaves", "ms=neural&phase=lorenz-mie"]) {
    const dense = await run(page, `${q}&${extra}&source=dense`);
    const sparse = await run(page, `${q}&${extra}&source=sparse`);
    expectClean(dense);
    expectClean(sparse);
    const ab = compareImages(image(dense), image(sparse));
    expect(ab.psnr, `${extra}: ${JSON.stringify(ab)}`).toBeGreaterThanOrEqual(35);
    lines.push(`${extra} ${ab.psnr.toFixed(2)} dB (${ab.above} px > 8/255)`);
  }
  console.log(`[k3-frame-scattering] sparse vs dense: ${lines.join("; ")}`);
});

test("K3.4 defaults (V33): 'frame' lighting + temporal + 'neural' render cleanly and deterministically", async ({ page }) => {
  test.setTimeout(400_000);
  const q = "?w=160&h=96&froxels=96x64x24&lighting=frame&lights=dir-hemi&defaults=1";
  const r = await run(page, q);
  const detail = detailOf(r);
  expectClean(r);
  expect([r.settings!.lighting, r.settings!.temporal, r.settings!.multipleScattering], detail).toEqual(["frame", true, "neural"]);
  expect(r.nonBackground!, detail).toBeGreaterThan(1_500);
  const again = await run(page, q);
  expect(compareImages(image(r), image(again)).psnr).toBe(Infinity);
  save("k3-frame-scattering-defaults", r);
});

// His white furnace (src/docs/examples/src/utils/furnace.js): a uniform environment of radiance L lights a white
// (non-absorbing) volume and is its background; an exact renderer returns L everywhere (the volume vanishes). His
// multiple scattering modes are approximations, so the ratio is recorded per mode, not forced. 'frame' lighting here;
// 'volume' lighting reads the environment since K3.5 (its furnace is in k3-frame-spectral.spec.ts). K3.4 (SwiftShader): 1D, 'two-sided' and 'neural' 0.998 mean
// v/L (every environment light is on the environment grid with isotropic diffusion; the network and the two-sided
// march light only scene light slots), 'octaves' 0.73 / 0.47, single scattering 0.26 / 0.16 (extinction 40 / 200).
const FURNACE_MODES = ["0", "1", "two-sided", "octaves", "neural"];

test("K3.4 white furnace ('frame' lighting, 'directional' environment): radiance / L per multiple scattering mode", async ({ page }) => {
  test.setTimeout(900_000);
  const rows: Record<string, Furnace & { extinction: number }> = {};
  for (const extinction of [40, 200]) {
    for (const ms of FURNACE_MODES) {
      const r = await run(page, `?w=160&h=96&froxels=96x64x24&lighting=frame&furnace=1&ms=${ms}&extinction=${extinction}`);
      const detail = detailOf(r);
      expectClean(r);
      const f = r.furnace!;
      expect(f, detail).toBeDefined();
      expect(f.environment?.mode, detail).toBe("directional");
      expect(f.volumePixels, detail).toBeGreaterThan(200);
      for (const v of [f.meanRatio, f.minRatio, f.maxRatio]) expect(Number.isFinite(v), detail).toBe(true);
      rows[`${ms}@${extinction}`] = { ...f, extinction };
      if (extinction === 40) save(`k3-frame-scattering-furnace-${ms}`, r);
    }
  }
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, "k3-frame-scattering-furnace.json"), JSON.stringify(rows, null, 2));
  // Recorded, not forced. One property his notes promise: diffusion keeps more of the light than single scattering.
  for (const e of [40, 200]) expect(rows[`1@${e}`]!.meanRatio).toBeGreaterThan(rows[`0@${e}`]!.meanRatio);
  console.log(
    `[k3-frame-scattering] furnace (volume radiance / L over pixels with alpha > 0.5; centre 9x9 displayed / L):\n` +
      Object.entries(rows)
        .map(([k, f]) => `  ${k}: mean ${f.meanRatio.toFixed(3)} (p05 ${f.p05.toFixed(3)}, p95 ${f.p95.toFixed(3)}, min ${f.minRatio.toFixed(3)}, max ${f.maxRatio.toFixed(3)}), |1 - r| ${f.meanAbsDeviation.toFixed(3)}, ${f.volumePixels} px; centre ${f.centerRatio.map((v) => v.toFixed(3)).join("/")}`)
        .join("\n"),
  );
});

// Provisional timings (D25: SwiftShader on a loaded laptop; never a gate). Opt in: K3_SCATTERING_TIMING=1 (needs
// fixtures/k3/bunny_cloud.wvdb). His step3-default-dir-isolated scene at 1920x1080, his 512x512x24 froxels / 32
// sub-steps / grid 64³, per multiple scattering mode: the froxel passes with the grid cached, the extra cost of a grid
// redraw (shadow density + grid mips + transmittance; the network runs in the transmittance pass) and the first frame's
// wall time (WGSL / pipeline compile included).
test("K3.4 timing report (opt-in): grid redraw ms and first-frame ms per multiple scattering mode", async ({ page }) => {
  test.skip(process.env.K3_SCATTERING_TIMING !== "1", "report-only: K3_SCATTERING_TIMING=1");
  test.setTimeout(3_600_000);
  const rows: Record<string, unknown>[] = [];
  for (const ms of ["1", "neural", "two-sided", "octaves", "0"]) {
    const r = await run(page, `?scene=step3-default-dir-isolated&w=1920&h=1080&ms=${ms}&timing=1&timingFrames=15&timingWarmup=3`);
    expect(r.error, detailOf(r)).toBeUndefined();
    const t = r.timings as { froxelsMs: number | null; integrateMs: number | null; sceneMs: number | null; passes?: { gridRedrawMs: number | null } };
    rows.push({ ms, firstFrameMs: r.firstFrameMs, froxelsMs: t.froxelsMs, gridRedrawMs: t.passes?.gridRedrawMs ?? null, integrateMs: t.integrateMs, sceneMs: t.sceneMs });
  }
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, "k3-frame-scattering-timings.json"), JSON.stringify(rows, null, 2));
  console.log(`[k3-frame-scattering] timings (provisional, D25): ${JSON.stringify(rows, null, 2)}`);
});
