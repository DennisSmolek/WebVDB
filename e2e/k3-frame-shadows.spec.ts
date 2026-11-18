import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.6 (R1 §6 Step 6): volume shadows on meshes and the emission light (examples/src/harness/k3-frame-shadows.ts).
// Gates until his Step 6 captures exist: the GPU passes match their CPU twins (shadow map, receiver chain, emission
// samplers / direction / reduce / probe passes, lit Lambert ground); shadows darken the ground where an independent CPU
// march through the grid says so and nowhere else; the emission light's falloff; both sources; deterministic; no
// validation errors, no warnings. PNGs go to artifacts/k3/ours/k3-shadows-*.png.
const PAGE_URL = "/src/harness/k3-frame-shadows.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OURS = path.join(repoRoot, "artifacts", "k3", "ours");

type Stats = { compared: number; maxAbs: number; maxRel: number; over: number; within: number; firstBad?: unknown };
interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  width?: number;
  height?: number;
  rgba?: string;
  rgbaOff?: string;
  debugRgba?: string;
  inPageIdentical?: boolean;
  warnings: string[];
  uncaptured: string[];
  entry?: { size: number; cascades: number; drawn: number; waiting: number } | null;
  mapTwin?: Stats & { texels: number; shadowed: number; maxDepth: number };
  groundTwin?: Stats & { ground: number; shadowed: number; meanT: number; minT: number; strictOver: number; maxStrict: number };
  independent?: { compared: number; meanAbs: number; maxAbs: number; within05: number; within10: number };
  nowhereElse?: { unshadowed: number; changed: number; shadowed: number; darker: number };
  shadowStats?: { groundPixels: number; covered: number; shadowedPixels: number; meanDarkening: number };
  mapTau?: number[];
  emitters?: number;
  sourceProbe?: Stats & { lod: Stats };
  directionTwin?: Stats & { texels: number };
  directionMean?: { cpu: number[]; gpu: number[]; maxRel: number; centerDistance: number };
  reduceTwin?: Stats;
  probeTwin?: Stats & { probes: number };
  irradianceTwin?: Stats & { ground: number; lit: number };
  falloff?: { distance: number; measured: number; twin: number; pointLight: number }[];
  lightOff?: { groundPixels: number; nonBlack: number };
  [k: string]: unknown;
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_SHADOWS__?: { done: boolean } }).__VDB_K3_SHADOWS__?.done === true, undefined, { timeout: 380_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_SHADOWS__?: unknown }).__VDB_K3_SHADOWS__)) as Result;
}

const image = (r: Result, b64 = r.rgba!) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(b64, "base64")) });
const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64` : undefined, rgbaOff: r.rgbaOff ? `${r.rgbaOff.length} b64` : undefined, debugRgba: undefined }, null, 2);

function save(name: string, r: Result, b64 = r.rgba!): Buffer {
  const png = Buffer.from(encodePng(image(r, b64)) as Uint8Array);
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

function expectShadowTwins(r: Result, label: string): void {
  const detail = detailOf(r);
  // The Beer shadow map equals its CPU twin over the read-back grid
  expect(r.mapTwin!.texels, detail).toBeGreaterThan(1000);
  expect(r.mapTwin!.shadowed, detail).toBeGreaterThan(100);
  expect(r.mapTwin!.over, detail).toBe(0);
  // Every visible ground pixel's transmittance equals the receiver chain on the read-back map...
  expect(r.groundTwin!.ground, detail).toBeGreaterThan(2000);
  expect(r.groundTwin!.shadowed, detail).toBeGreaterThan(200);
  expect(r.groundTwin!.over, detail).toBe(0);
  // ... exactly (4e-3) except where SwiftShader's position rasterization moves the map read by < 0.06 texel (shadow edges)
  expect(r.groundTwin!.strictOver, detail).toBeLessThanOrEqual(r.groundTwin!.ground * 0.01);
  expect(r.groundTwin!.minT, detail).toBeLessThan(0.5);
  // ... and agrees with an independent CPU-integrated transmittance (512-step march toward the light through the grid)
  expect(r.independent!.within05, detail).toBeGreaterThan(0.99);
  expect(r.independent!.meanAbs, detail).toBeLessThan(0.01);
  // Nowhere else: unshadowed ground pixels are identical to the castShadows-off frame; shadowed ones are darker
  expect(r.nowhereElse!.unshadowed, detail).toBeGreaterThan(1000);
  expect(r.nowhereElse!.changed, detail).toBe(0);
  expect(r.nowhereElse!.shadowed, detail).toBeGreaterThan(100);
  expect(r.nowhereElse!.darker, detail).toBe(r.nowhereElse!.shadowed);
  console.log(
    `[k3-shadows ${label}] ${r.adapter}: map twin ${r.mapTwin!.texels} texels (${r.mapTwin!.shadowed} shadowed, τ max ${r.mapTwin!.maxDepth.toFixed(2)}) max |d| ${r.mapTwin!.maxAbs.toExponential(2)}; ground twin ${r.groundTwin!.ground} px (${r.groundTwin!.shadowed} shadowed, T min ${r.groundTwin!.minT.toFixed(3)}) max |d| ${r.groundTwin!.maxAbs.toExponential(2)} (strict: ${r.groundTwin!.strictOver} px over 4e-3, max ${r.groundTwin!.maxStrict.toFixed(3)}); independent mean |d| ${r.independent!.meanAbs.toExponential(2)}, max ${r.independent!.maxAbs.toFixed(3)}, within 0.05 ${(r.independent!.within05 * 100).toFixed(2)}%; nowhere else ${JSON.stringify(r.nowhereElse)}; map τ ${JSON.stringify(r.mapTau)}`,
  );
}

test("K3.6 volume shadows (sun, dense): the map and every ground pixel match their CPU twins; nowhere else; deterministic", async ({ page }) => {
  test.setTimeout(900_000);
  const q = "?case=shadow&twins=1";
  const first = await run(page, q);
  const a = save("k3-shadows-dir-dense-run1", first);
  if (first.debugRgba) save("k3-shadows-dir-dense-debug", first, first.debugRgba);
  if (first.rgbaOff) save("k3-shadows-dir-dense-off", first, first.rgbaOff);
  expectClean(first);
  expectShadowTwins(first, "dir dense");
  expect(first.entry, detailOf(first)).toMatchObject({ size: 512, cascades: 1 });
  // castShadows off changes the frame (the ground's shadow) and nothing else (the twins above locate it)
  const cmp = compareImages(image(first), image(first, first.rgbaOff!));
  expect(cmp.above, JSON.stringify(cmp)).toBeGreaterThan(100);

  const second = await run(page, q);
  expectClean(second);
  const b = save("k3-shadows-dir-dense-run2", second);
  expect(a.equals(b), "two page loads byte-identical").toBe(true);
});

test("K3.6 volume shadows: the sparse source and 'frame' lighting give the same shadows; 4 cascades; a spot light", async ({ page }) => {
  test.setTimeout(1_200_000);
  const dense = await run(page, "?case=shadow&twins=1");
  expectClean(dense);
  const sparse = await run(page, "?case=shadow&twins=1&source=sparse");
  expectClean(sparse);
  expectShadowTwins(sparse, "dir sparse");
  save("k3-shadows-dir-sparse", sparse);
  // Same voxels, sampler-only differences in the shadow grid (K3.2 bar 35 dB on the image)
  const ds = compareImages(image(dense), image(sparse));
  console.log(`[k3-shadows] dense vs sparse: ${ds.psnr.toFixed(2)} dB, ${ds.above} px > 8/255`);
  expect(ds.psnr).toBeGreaterThan(35);

  // 'frame' lighting draws every grid cascade; the shadow maps read the last one: the same map, the twins hold
  const frame = await run(page, "?case=shadow&twins=1&lighting=frame");
  expectClean(frame);
  expectShadowTwins(frame, "dir frame-lighting");
  expect(frame.mapTau![1], detailOf(frame)).toBeGreaterThan(500);
  expect(frame.mapTau, detailOf(frame)).toEqual(dense.mapTau);
  save("k3-shadows-dir-frame-lighting", frame);

  const cascades = await run(page, "?case=shadow&twins=1&cascades=4");
  expectClean(cascades);
  expectShadowTwins(cascades, "dir 4 cascades");
  console.log(`[k3-shadows] 4 cascades: ${JSON.stringify(cascades.entry)}`);

  const spot = await run(page, "?case=shadow&twins=1&light=spot");
  expectClean(spot);
  expectShadowTwins(spot, "spot");
  save("k3-shadows-spot", spot);
});

test("K3.6 emission light (a smooth blob and the ground explosion frame 10; dense + sparse): passes match their twins; the ground's falloff; light off is black", async ({ page }) => {
  test.setTimeout(1_500_000);
  let densePng: Buffer | null = null;
  for (const [emitter, source] of [
    ["blob", "dense"],
    ["blob", "sparse"],
    ["explosion", "dense"],
    ["explosion", "sparse"],
  ] as const) {
    const label = `${emitter} ${source}`;
    const r = await run(page, `?case=emission&twins=1&emitter=${emitter}&source=${source}`);
    const detail = detailOf(r);
    expectClean(r);
    expect(r.emitters, detail).toBe(1);
    // The source's samplers at lod 0 (both sources: the same voxels) equal the direction twin's CPU samplers
    expect(r.sourceProbe!.compared, detail).toBe(12288);
    expect(r.sourceProbe!.over, detail).toBe(0);
    if (source === "dense") expect(r.sourceProbe!.lod.over, detail).toBe(0);
    expect(r.directionTwin!.texels, detail).toBe(16);
    // Every direction texel for the smooth dense blob; otherwise the mean of the 16 directions (what the probes read):
    // SwiftShader's ~12-bit division moves the box segments by ~2e-4, which the explosion's sharp flames under 8² rays x
    // 24 steps turn into several % per direction; the sparse coarse pyramid differs from the dense mips above lod 0 (K3.2)
    if (emitter === "blob" && source === "dense") expect(r.directionTwin!.over, detail).toBe(0);
    expect(r.directionMean!.maxRel, detail).toBeLessThan(source === "dense" ? 0.01 : 0.05);
    expect(r.reduceTwin!.over, detail).toBe(0);
    expect(r.probeTwin!.probes, detail).toBeGreaterThan(1000);
    expect(r.probeTwin!.over, detail).toBe(0);
    expect(r.irradianceTwin!.ground, detail).toBeGreaterThan(2000);
    expect(r.irradianceTwin!.lit, detail).toBeGreaterThan(1000);
    expect(r.irradianceTwin!.over, detail).toBe(0);
    // Falloff: the lit ground darkens away from the emission centre, as the twin and (far out) as one point light
    const f = r.falloff!;
    expect(f.length, detail).toBeGreaterThanOrEqual(3);
    for (let i = 1; i < f.length; i++) expect(f[i]!.measured, detail).toBeLessThan(f[i - 1]!.measured);
    for (const s of f) expect(Math.abs(s.measured - s.twin), detail).toBeLessThanOrEqual(s.twin * 0.02 + 1e-4);
    const far = f[f.length - 1]!;
    expect(far.measured / far.pointLight, detail).toBeGreaterThan(0.5);
    expect(far.measured / far.pointLight, detail).toBeLessThan(2);
    // The light off: no other light, so the ground is black
    expect(r.lightOff!.groundPixels, detail).toBeGreaterThan(2000);
    expect(r.lightOff!.nonBlack, detail).toBe(0);
    const png = save(`k3-shadows-emission-${emitter}-${source}`, r);
    if (emitter === "explosion" && source === "dense") densePng = png;
    save(`k3-shadows-emission-${emitter}-${source}-off`, r, r.rgbaOff!);
    console.log(
      `[k3-emission ${label}] ${r.adapter}: source probe max |d| ${r.sourceProbe!.maxAbs.toExponential(2)}; direction twin ${r.directionTwin!.over}/${r.directionTwin!.compared} over, max rel ${r.directionTwin!.maxRel.toExponential(2)}; 16-direction mean rel ${r.directionMean!.maxRel.toExponential(2)}; reduce max |d| ${r.reduceTwin!.maxAbs.toExponential(2)}; probe twin ${r.probeTwin!.probes} probes max |d| ${r.probeTwin!.maxAbs.toExponential(2)}; ground ${r.irradianceTwin!.ground} px (${r.irradianceTwin!.lit} lit) max rel ${r.irradianceTwin!.maxRel.toExponential(2)}; falloff [d, measured, twin, point] ${JSON.stringify(f.map((s) => [Math.round(s.distance), +s.measured.toPrecision(4), +s.twin.toPrecision(4), +s.pointLight.toPrecision(4)]))}`,
    );
  }
  const again = await run(page, "?case=emission&twins=1");
  expectClean(again);
  const saved = save("k3-shadows-emission-explosion-dense-run2", again);
  expect(saved.equals(densePng!), "two page loads byte-identical").toBe(true);
});
