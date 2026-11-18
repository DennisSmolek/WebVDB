import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.3 (R1 §6 Step 3): mjurczyk's 'frame' lighting (shadow density → world grid cascades → transmittance →
// environment grid → density → lighting → temporal) on the froxel pipeline, both volume sources
// (examples/src/harness/k3-frame.ts with ?lighting=frame). Gates until his captures exist (K3.0a/K3.1a): no
// validation errors or warnings; the grid chunks, the transmittance, environment grid, density and lighting passes match
// their CPU twins on read-back inputs; the signature skips redraws; deterministic; depth occlusion; cascades
// change the image toward a finer grid; sparse ('coarse-pyramid') vs dense on identical voxels >= 35 dB; temporal
// reprojection converges. PNGs go to artifacts/k3/ours/ (k3-frame-lighting-*.png).
const PAGE_URL = "/src/harness/k3-frame.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OURS = path.join(repoRoot, "artifacts", "k3", "ours");

interface Twin {
  compared: number;
  maxAbs: number;
  over: number;
  marched?: number;
  lit?: number;
  firstBad?: unknown;
}

interface Result {
  done: boolean;
  error?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  nonBackground?: number;
  chunkProbe?: { count: number; over: number };
  gridProbe?: Twin;
  transmittanceTwin?: Twin;
  lightingTwin?: Twin;
  environmentGridTwin?: Twin;
  densityTwin?: Twin;
  grid?: { count: number; resolution: number };
  environment?: { mode: string; count: number; directions: number[][]; colors: number[][] } | null;
  redraw?: { shadowDensity: { drawn: number; skipped: number }; transmittance: { drawn: number; skipped: number }; environmentGrid: { drawn: number; skipped: number } };
  temporalCheck?: { diffsOff: number[]; diffsOn: number[]; psnrSingleVsMean: number; psnrTemporalVsMean: number; volumePixels: number };
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

/**
 * `maxOverFraction`: the density pass marches the source with f32 sub-step counts (ceil) and visibility weights, so
 * a froxel on a float boundary can differ from the f64 twin (3 of 86,576 values on his sky-midday bunny at 640x360);
 * it gets 1 in 2,000. Every other pass must match exactly within tolerance.
 */
function expectTwin(t: Twin | undefined, minCompared: number, detail: string, maxOverFraction = 0): void {
  expect(t, detail).toBeDefined();
  expect(t!.compared, detail).toBeGreaterThan(minCompared);
  expect(t!.over, `${JSON.stringify(t)}\n${detail}`).toBeLessThanOrEqual(Math.floor(t!.compared * maxOverFraction));
}

// The e2e scene: the committed CC0 cloud, his sun + hemisphere + ambient light (frame.js), 2HG renderer phase.
const BASE = "?w=160&h=96&froxels=96x64x24&lighting=frame";

test("K3.3 'frame' lighting: grid chunks, transmittance, density and lighting passes match their CPU twins; redraws skipped; deterministic", async ({ page }) => {
  test.setTimeout(400_000);
  const q = `${BASE}&lights=dir-hemi&twins=1&redraw=2`;
  const first = await run(page, q);
  const detail = detailOf(first);
  expectClean(first);
  expect(first.nonBackground!, detail).toBeGreaterThan(1_500);
  expect(first.chunkProbe!.over, detail).toBe(0);
  expectTwin(first.gridProbe, 1_000, detail);
  expectTwin(first.transmittanceTwin, 10_000, detail);
  expect(first.transmittanceTwin!.marched!, detail).toBeGreaterThan(500);
  expectTwin(first.lightingTwin, 1_000, detail);
  expect(first.lightingTwin!.lit!, detail).toBeGreaterThan(300);
  // The density pass through the dense source's CPU sampler (sampleWvdbDenseTexture), shadow slots from the grid.
  expectTwin(first.densityTwin, 5_000, detail, 5e-4);
  // Static scene: the grid and the transmittance are drawn once; every later frame skips them (signatures).
  expect(first.redraw!.shadowDensity.drawn, detail).toBe(1);
  expect(first.redraw!.shadowDensity.skipped, detail).toBeGreaterThanOrEqual(3);
  expect(first.redraw!.transmittance.drawn, detail).toBe(1);
  expect(first.redraw!.transmittance.skipped, detail).toBeGreaterThanOrEqual(3);

  const second = await run(page, q);
  expect(second.error).toBeUndefined();
  const cmp = compareImages(image(first), image(second));
  expect(cmp.psnr, JSON.stringify(cmp)).toBe(Infinity);
  const a = save("k3-frame-lighting-run1", first);
  const b = save("k3-frame-lighting-run2", second);
  expect(Buffer.from(a).equals(Buffer.from(b))).toBe(true);

  // 'frame' and 'volume' lighting of the same scene and lights are close (same shadows, from a grid vs a march)
  // but not the same image: the frame passes run.
  const volume = await run(page, `?w=160&h=96&froxels=96x64x24&lighting=volume&lights=dir-hemi`);
  expect(volume.error).toBeUndefined();
  const vs = compareImages(image(first), image(volume));
  expect(vs.mse, JSON.stringify(vs)).toBeGreaterThan(0);
  console.log(
    `[k3-frame-lighting] twins: grid probe ${first.gridProbe!.compared} (max ${first.gridProbe!.maxAbs.toExponential(2)}), transmittance ${first.transmittanceTwin!.compared} values / ${first.transmittanceTwin!.marched} voxels (max ${first.transmittanceTwin!.maxAbs.toExponential(2)}), lighting ${first.lightingTwin!.compared} values / ${first.lightingTwin!.lit} froxels (max ${first.lightingTwin!.maxAbs.toExponential(2)}); redraws ${JSON.stringify(first.redraw)}; 'frame' vs 'volume' ${vs.psnr.toFixed(2)} dB`,
  );
});

test("K3.3 environment 'directional': lights from the PMREM readback, environment grid and environment slot match their twins", async ({ page }) => {
  test.setTimeout(400_000);
  // Diffusion: every environment light goes to the environment grid.
  const env = await run(page, `${BASE}&env=synthetic&lights=none&twins=1&redraw=2`);
  const detail = detailOf(env);
  expectClean(env);
  expect(env.environment?.mode, detail).toBe("directional");
  expect(env.environment!.count, detail).toBe(8);
  // The synthetic sky's sun is at (0.45, 0.75, 0.48): the strongest light points at it.
  const sun = [0.45, 0.75, 0.48];
  const n = Math.hypot(...sun);
  const d = env.environment!.directions[0]!;
  expect((d[0]! * sun[0]! + d[1]! * sun[1]! + d[2]! * sun[2]!) / n, detail).toBeGreaterThan(0.99);
  expectTwin(env.environmentGridTwin, 10_000, detail);
  expect(env.environmentGridTwin!.marched!, detail).toBeGreaterThan(500);
  expectTwin(env.transmittanceTwin, 10_000, detail);
  expectTwin(env.lightingTwin, 1_000, detail);
  // The density pass adds the environment grid at each volume's visible position.
  expectTwin(env.densityTwin, 5_000, detail, 5e-4);
  expect(env.redraw!.environmentGrid.drawn, detail).toBe(1);
  expect(env.redraw!.environmentGrid.skipped, detail).toBeGreaterThanOrEqual(2);
  save("k3-frame-lighting-environment", env);

  // Without the environment the volume is unlit (no scene light): the environment lights it.
  const dark = await run(page, `${BASE}&lights=none`);
  expect(dark.error).toBeUndefined();
  expect(mean(env)).toBeGreaterThan(mean(dark) * 1.05);

  // Single scattering: the dominant environment light takes a shadow slot after the sun (the lighting pass's
  // environment slot), the others the grid.
  const slot = await run(page, `${BASE}&env=synthetic&lights=dir-hemi&ms=0&twins=1`);
  const slotDetail = detailOf(slot);
  expectClean(slot);
  expectTwin(slot.transmittanceTwin, 10_000, slotDetail);
  expectTwin(slot.lightingTwin, 1_000, slotDetail);
  expectTwin(slot.environmentGridTwin, 10_000, slotDetail);
  expectTwin(slot.densityTwin, 5_000, slotDetail, 5e-4);
});

test("K3.3 'frame' lighting: an opaque mesh in front occludes the volume", async ({ page }) => {
  test.setTimeout(400_000);
  const plain = await run(page, `${BASE}&lights=dir-hemi`);
  const occluded = await run(page, `${BASE}&lights=dir-hemi&occluder=1`);
  const occluderOnly = await run(page, `${BASE}&lights=dir-hemi&occluder=1&novolume=1`);
  for (const r of [plain, occluded, occluderOnly]) expect(r.error, detailOf(r)).toBeUndefined();
  expect(occluded.uncaptured).toEqual([]);
  const occ = image(occluderOnly).data;
  const withVolume = image(occluded).data;
  const lit = image(plain).data;
  let occluderPixels = 0;
  let changedBehind = 0;
  let leaked = 0;
  for (let i = 0; i < occ.length; i += 4) {
    const isOccluder = occ[i]! > 200 && occ[i + 1]! < 120 && occ[i + 2]! > 60 && occ[i + 2]! < 140;
    if (!isOccluder) continue;
    occluderPixels++;
    if (withVolume[i] !== occ[i] || withVolume[i + 1] !== occ[i + 1] || withVolume[i + 2] !== occ[i + 2]) leaked++;
    if (Math.abs(lit[i]! - occ[i]!) + Math.abs(lit[i + 1]! - occ[i + 1]!) + Math.abs(lit[i + 2]! - occ[i + 2]!) > 30) changedBehind++;
  }
  expect(occluderPixels).toBeGreaterThan(50);
  expect(leaked, `${leaked} of ${occluderPixels} occluder pixels changed by the volume behind`).toBe(0);
  expect(changedBehind).toBeGreaterThan(occluderPixels * 0.5);
  save("k3-frame-lighting-occluded", occluded);
});

test("K3.3 cascades: a camera inside the casters gets finer cascades; the image moves toward a finer grid", async ({ page }) => {
  test.setTimeout(500_000);
  const near = "?w=240&h=135&froxels=128x96x24&lighting=frame&lights=dir-hemi&dist=0.3";
  const c4 = await run(page, `${near}&cascades=4&twins=1`);
  const c1 = await run(page, `${near}&cascades=1`);
  const fine = await run(page, `${near}&cascades=1&shadowRes=256`);
  const detail = detailOf(c4);
  expectClean(c4);
  for (const r of [c1, fine]) expect(r.error, detailOf(r)).toBeUndefined();
  expect(c4.grid!.count, detail).toBeGreaterThanOrEqual(2);
  expect(c1.grid!.count).toBe(1);
  // The transmittance twin holds across the cascades (the march steps into coarser ones).
  expectTwin(c4.transmittanceTwin, 10_000, detail);
  expectTwin(c4.lightingTwin, 1_000, detail);
  // The density pass blends the shadows across cascade borders (sampleShadows).
  expectTwin(c4.densityTwin, 5_000, detail, 5e-4);
  const diff = compareImages(image(c4), image(c1));
  expect(diff.mse, JSON.stringify(diff)).toBeGreaterThan(0);
  const toFine4 = compareImages(image(c4), image(fine));
  const toFine1 = compareImages(image(c1), image(fine));
  expect(toFine4.psnr, `4 cascades ${toFine4.psnr} dB vs 1 cascade ${toFine1.psnr} dB to the 256³ grid`).toBeGreaterThan(toFine1.psnr);
  save("k3-frame-lighting-cascades-4", c4);
  save("k3-frame-lighting-cascades-1", c1);
  console.log(`[k3-frame-lighting] cascades: ${c4.grid!.count} in use; vs a 256³ single grid: 4 cascades ${toFine4.psnr.toFixed(2)} dB, 1 cascade ${toFine1.psnr.toFixed(2)} dB`);
});

test("K3.3 sparse source: 'frame' lighting on identical voxels is >= 35 dB from dense, and its passes match their twins", async ({ page }) => {
  test.setTimeout(500_000);
  const q = "?w=320&h=180&froxels=128x128x24&lighting=frame&lights=dir-hemi";
  const dense = await run(page, `${q}&source=dense`);
  const sparse = await run(page, `${q}&source=sparse&twins=1&redraw=2`);
  const detail = detailOf(sparse);
  expectClean(dense);
  expectClean(sparse);
  expectTwin(sparse.transmittanceTwin, 10_000, detail);
  expectTwin(sparse.lightingTwin, 1_000, detail);
  expect(sparse.redraw!.shadowDensity.skipped, detail).toBeGreaterThanOrEqual(3);
  const ab = compareImages(image(dense), image(sparse));
  expect(ab.psnr, JSON.stringify(ab)).toBeGreaterThanOrEqual(35);
  save("k3-frame-lighting-dense", dense);
  save("k3-frame-lighting-sparse", sparse);
  console.log(`[k3-frame-lighting] sparse ('coarse-pyramid') vs dense, identical voxels: ${ab.psnr.toFixed(2)} dB, ${ab.above} px > 8/255`);
});

test("K3.3 temporal reprojection + jitter: a still camera converges toward the mean of the jittered frames", async ({ page }) => {
  test.setTimeout(500_000);
  const r = await run(page, `?w=160&h=96&froxels=96x64x24&lighting=frame&lights=dir-hemi&temporalCheck=16`);
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.warnings, detail).toEqual([]);
  const t = r.temporalCheck!;
  const tail = (v: number[]) => v.slice(-8).reduce((s, x) => s + x, 0) / 8;
  expect(t.volumePixels, detail).toBeGreaterThan(1_000);
  // Jitter changes every frame; the history blend (0.2) cuts the frame-to-frame change several times over.
  expect(tail(t.diffsOff), detail).toBeGreaterThan(0);
  expect(tail(t.diffsOn), detail).toBeLessThan(tail(t.diffsOff) * 0.5);
  expect(t.psnrTemporalVsMean, detail).toBeGreaterThan(t.psnrSingleVsMean + 3);
  console.log(`[k3-frame-lighting] temporal: frame-to-frame change ${tail(t.diffsOff).toFixed(3)} → ${tail(t.diffsOn).toFixed(3)} (8-bit units / channel); vs the jittered mean ${t.psnrSingleVsMean.toFixed(2)} → ${t.psnrTemporalVsMean.toFixed(2)} dB`);
});
